"""Fine-tune the white-screen weights on four gray levels at native resolution."""

import argparse
import csv
import json
import math
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from train_paired_unet import KINDS, PairedData, save_checkpoint
from evaluate_paired_models import digest
from display_defect_ai.gray_unet import ARCHITECTURE, normalize_rgb_chw
from display_defect_ai.pixel_model import load_unet, segmentation_loss

DATASET = Path("data/oled-gray-paired-v1")
OUTPUT = Path("outputs/oled-gray-unet-v1")
INITIAL = Path("outputs/paired-model-comparison-v1/unet/best.pt")


class GrayData(PairedData):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        normalized = torch.from_numpy(normalize_rgb_chw(image.numpy())) / 255
        return normalized, target, self.rows[index]["gray_level"]


@torch.inference_mode()
def validate(model, loader, levels):
    model.eval()
    totals = {level: torch.zeros((3, len(KINDS)), device="mps") for level in levels}
    for images, targets, batch_levels in loader:
        targets = targets.to("mps")
        predicted = model(images.to("mps")) >= 0
        counts = torch.stack(((predicted * targets).sum((2, 3)),
                              predicted.sum((2, 3)), targets.sum((2, 3))), dim=1)
        for i, level in enumerate(batch_levels.tolist()):
            totals[level] += counts[i]
    by_level = {}
    for level, count in totals.items():
        tp, pred, actual = count.cpu().numpy()
        by_level[str(level)] = ((2 * tp + 1) / (pred + actual + 1)).tolist()
    return by_level, float(np.mean(list(by_level.values())))


def train(dataset, output, initial, epochs, batch, resume):
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple Metal GPU access is required")
    cv2.setNumThreads(2)
    torch.set_num_threads(4)
    torch.mps.set_per_process_memory_fraction(.22)
    record = json.loads((dataset / "segmentation.json").read_text())
    manifest_hash = digest(dataset / "manifest.jsonl")
    mask_hash = digest(dataset / "mask_manifest.jsonl")
    if (manifest_hash != record["dataset_manifest_sha256"]
            or mask_hash != record["mask_manifest_sha256"]):
        raise ValueError("The dataset lists changed after generation")
    levels = record["gray_levels"]
    random.seed(43)
    torch.manual_seed(43)
    model = load_unet(initial, "mps")
    optimizer = torch.optim.AdamW(model.parameters(), lr=.00035, weight_decay=.0001)
    training = DataLoader(GrayData(dataset, "train"), batch_size=batch, shuffle=True, num_workers=0)
    validation = DataLoader(GrayData(dataset, "val"), batch_size=batch, shuffle=False, num_workers=0)
    destination = output / "unet"
    config = dict(architecture=ARCHITECTURE, backbone=model.architecture, width=model.width,
                  parameters=sum(value.numel() for value in model.parameters()),
                  initialization=str(initial), initial_weights_sha256=digest(initial),
                  dataset_manifest_sha256=manifest_hash, mask_manifest_sha256=mask_hash,
                  levels=levels, training_images=len(training.dataset), validation_images=len(validation.dataset),
                  epochs=epochs, batch=batch, device="mps", torch=str(torch.__version__),
                  normalization="Per-tile median of RGB average sampled every 16px; common gain 230/max(32, median)",
                  checkpoint_selection="Mean validation pixel Dice over gray levels and classes")
    best, start_epoch = -1., 1
    if resume:
        if json.loads((destination / "run.json").read_text()) != config:
            raise ValueError("Resume settings changed")
        last = torch.load(destination / "last.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(last["model_state"])
        optimizer.load_state_dict(last["optimizer_state"])
        best = last["best_dice"]
        start_epoch = last["epoch"] + 1
    else:
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "run.json").write_text(json.dumps(config, indent=2) + "\n")
        pixels, dice = validate(model, validation, levels)
        (destination / "initial_validation.json").write_text(json.dumps(
            dict(macro_dice=dice, dice_by_level=pixels, classes=KINDS), indent=2) + "\n")
        print(f"White weights with gray normalization before fine-tuning: validation Dice {dice:.4f}", flush=True)

    fields = ["epoch", "train_loss", "validation_macro_dice", "seconds"]
    # Rewrite only the log tail beyond the most recent complete checkpoint on resume.
    existing = list(csv.DictReader((destination / "results.csv").open())) if resume else []
    with (destination / "results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(row for row in existing if int(row["epoch"]) < start_epoch)
        for epoch in range(start_epoch, epochs + 1):
            started = time.perf_counter()
            random.seed(43 + epoch)
            torch.manual_seed(43 + epoch)
            model.train()
            rate = .000035 + .000315 * (1 + math.cos(math.pi * (epoch - 1) / epochs)) / 2
            for group in optimizer.param_groups:
                group["lr"] = rate
            loss_sum = 0.
            for index, (images, targets, _) in enumerate(training, 1):
                optimizer.zero_grad(set_to_none=True)
                loss = segmentation_loss(model(images.to("mps")), targets.to("mps"))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                loss_sum += loss.item() * len(images)
                if index % 20 == 0:
                    torch.mps.empty_cache()
                if index % 50 == 0:
                    print(f"Gray U-Net {epoch}/{epochs}: {index}/{len(training)} batches, "
                          f"loss={loss_sum / (index * batch):.4f}", flush=True)
            pixels, dice = validate(model, validation, levels)
            improved = dice > best
            best = max(best, dice)
            status = dict(epoch=epoch, train_loss=loss_sum / len(training.dataset),
                          validation_macro_dice=dice, seconds=time.perf_counter() - started)
            writer.writerow(status)
            stream.flush()
            print(json.dumps(dict(status, dice_by_level=pixels)), flush=True)
            checkpoint = dict(architecture=ARCHITECTURE, width=model.width, epoch=epoch,
                              model_state={key: value.detach().cpu() for key, value in model.state_dict().items()},
                              validation_dice_by_level=pixels, validation_macro_dice=dice, best_dice=best,
                              dataset_manifest_sha256=manifest_hash, mask_manifest_sha256=mask_hash)
            if improved:
                save_checkpoint(destination / "best.pt", checkpoint)
            checkpoint["optimizer_state"] = optimizer.state_dict()
            save_checkpoint(destination / "last.pt", checkpoint)
            torch.mps.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--initial", type=Path, default=INITIAL)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(args.dataset, args.output, args.initial, args.epochs, args.batch, args.resume)
