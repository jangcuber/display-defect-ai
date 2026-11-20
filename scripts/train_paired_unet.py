"""Train the full-resolution five-label U-Net on the paired OLED dataset."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from display_defect_ai.pixel_model import TinyUNet, segmentation_loss

KINDS = ("dark_dot", "color_dot", "horizontal_line", "vertical_line", "dark_region")


def save_checkpoint(path, value):
    temporary = path.with_suffix(".pending")
    torch.save(value, temporary)
    temporary.replace(path)


class PairedData(Dataset):
    def __init__(self, dataset, split):
        self.root, self.split = dataset, split
        self.rows = [row for row in map(json.loads, (dataset / "manifest.jsonl").read_text().splitlines())
                     if row["split"] == split]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = cv2.imread(str(self.root / row["image"]))
        encoded = cv2.imread(str(self.root / "masks" / self.split / f"{row['image_id']}.png"), cv2.IMREAD_GRAYSCALE)
        if image is None or encoded is None:
            raise OSError(row["image_id"])
        if self.split == "train":
            if random.random() < .5:
                image, encoded = image[:, ::-1], encoded[:, ::-1]
            if random.random() < .5:
                image, encoded = image[::-1], encoded[::-1]
        image = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1))
        target = np.stack([((encoded >> i) & 1).astype(np.float32) for i in range(5)])
        return torch.from_numpy(image), torch.from_numpy(np.ascontiguousarray(target))


def validate(model, loader, device):
    model.eval()
    tp = torch.zeros(5, device=device)
    predicted, actual = tp.clone(), tp.clone()
    total = 0.
    with torch.inference_mode():
        for image, target in loader:
            image, target = image.to(device, dtype=torch.float32) / 255, target.to(device)
            logits = model(image)
            prediction = logits >= 0
            tp += (prediction * target).sum((0, 2, 3))
            predicted += prediction.sum((0, 2, 3))
            actual += target.sum((0, 2, 3))
            total += float(segmentation_loss(logits, target).item()) * len(image)
    dice = ((2 * tp + 1) / (predicted + actual + 1)).cpu().numpy()
    return total / len(loader.dataset), dice


def train(dataset, output, epochs, batch, resume=False):
    if output.exists() and not resume:
        raise FileExistsError(output)
    if not torch.backends.mps.is_available():
        raise RuntimeError("Run with local Metal GPU access")
    masks = json.loads((dataset / "segmentation.json").read_text())
    digest = hashlib.sha256((dataset / "manifest.jsonl").read_bytes()).hexdigest()
    if masks["dataset_manifest_sha256"] != digest or masks["images"] != 7000:
        raise ValueError("Segmentation labels do not match the full dataset")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(43)
    random.seed(43)
    np.random.seed(43)
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    device = "mps"
    # Bound the allocator to prevent swap pressure during sequential GPU training.
    torch.mps.set_per_process_memory_fraction(.18)
    model = TinyUNet().to(device)
    training = DataLoader(PairedData(dataset, "train"), batch_size=batch, shuffle=True, num_workers=0)
    validation = DataLoader(PairedData(dataset, "val"), batch_size=batch, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    start_epoch = 1
    previous_record = None
    if resume:
        checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        if checkpoint["dataset_manifest_sha256"] != digest:
            raise ValueError("Resume data differs")
        model.load_state_dict(checkpoint["model_state"])
        start_epoch = checkpoint["epoch"] + 1
        previous_record = json.loads((output / "run.json").read_text())
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
    record = dict(model="TinyUNet", architecture=model.architecture, width=model.width,
                  parameters=sum(p.numel() for p in model.parameters()), initialization="random, seed 43",
                  dataset_manifest_sha256=digest, mask_manifest_sha256=masks["mask_manifest_sha256"],
                  training_images=len(training.dataset), validation_images=len(validation.dataset),
                  epochs_max=epochs, batch=batch, mps_memory_fraction=.18, image_size=1024, input="RGB only",
                  loss="balanced positive focal + top32 point/cell-mean line-region hard-negative focal + Dice",
                  checkpoint_selection="validation macro pixel Dice", device=device, torch=torch.__version__)
    if resume:
        record["resource_resume"] = dict(from_epoch=start_epoch - 1, previous_batch=previous_record["batch"],
                                         current_batch=batch, optimizer_restored="optimizer_state" in checkpoint,
                                         reason="Reduce memory pressure; native pixel resolution unchanged")
    (output / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)
    best, wait = -1., 0
    if resume and (output / "best.pt").exists():
        best_checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
        best = best_checkpoint["validation_macro_dice"]
        wait = max(0, start_epoch - 1 - best_checkpoint["epoch"])
    fields = ["epoch", "train_loss", "val_loss", "macro_dice", *[f"dice_{kind}" for kind in KINDS], "seconds"]
    with (output / "results.csv").open("a" if resume else "w") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if not resume:
            writer.writeheader()
        for epoch in range(start_epoch, epochs + 1):
            start = time.perf_counter()
            rate = .00001 + (.001 - .00001) * (1 + math.cos(math.pi * (epoch - 1) / epochs)) / 2
            for group in optimizer.param_groups:
                group["lr"] = rate
            model.train()
            cumulative = 0.
            for index, (image, target) in enumerate(training):
                image, target = image.to(device, dtype=torch.float32) / 255, target.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = segmentation_loss(model(image), target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                if (index + 1) % 20 == 0:
                    torch.mps.empty_cache()
                cumulative += loss.item() * len(image)
                if (index + 1) % 100 == 0:
                    print(f"UNet epoch {epoch}: {index + 1}/{len(training)} batches; loss {cumulative / ((index + 1) * batch):.5f}", flush=True)
            val_loss, dice = validate(model, validation, device)
            result = dict(epoch=epoch, train_loss=cumulative / len(training.dataset), val_loss=val_loss,
                          macro_dice=float(dice.mean()), seconds=time.perf_counter() - start,
                          **{f"dice_{kind}": float(value) for kind, value in zip(KINDS, dice, strict=True)})
            writer.writerow(result)
            file.flush()
            print(json.dumps(result), flush=True)
            checkpoint = dict(architecture=model.architecture, width=model.width, epoch=epoch,
                              model_state={key: value.detach().cpu() for key, value in model.state_dict().items()},
                              dataset_manifest_sha256=digest, mask_manifest_sha256=masks["mask_manifest_sha256"],
                              validation_macro_dice=float(dice.mean()),
                              optimizer_state={key: ({name: value.detach().cpu() if torch.is_tensor(value) else value
                                                     for name, value in state.items()} if isinstance(state, dict) else state)
                                               for key, state in optimizer.state_dict()["state"].items()})
            checkpoint["optimizer_state"] = dict(state=checkpoint["optimizer_state"],
                                                  param_groups=optimizer.state_dict()["param_groups"])
            if result["macro_dice"] > best + .0001:
                best, wait = result["macro_dice"], 0
                save_checkpoint(output / "best.pt", checkpoint)
            else:
                wait += 1
            save_checkpoint(output / "last.pt", checkpoint)
            torch.mps.empty_cache()
            if wait >= 5:
                print("Validation Dice early stopping", flush=True)
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/oled-white-paired-v2"))
    parser.add_argument("--output", type=Path, default=Path("outputs/paired-model-comparison-v1/unet"))
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(args.dataset, args.output, args.epochs, args.batch, args.resume)
