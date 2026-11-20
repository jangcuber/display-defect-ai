"""Render the fixed-test comparison as an exportable scientific chart."""
import argparse
import json
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', str(Path('artifacts/matplotlib').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def render(root):
    report = json.loads((root / 'test_comparison.json').read_text())
    labels = ['Points', 'Lines', 'Gradual regions']
    groups = ['points', 'lines', 'regions']
    colors = {'unet':'#296bcb', 'yolo':'#e99231'}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), dpi=160)
    for ax, metric, title in zip(axes, ('recall', 'precision'), ('Defects found (recall)', 'Correct detections (precision)')):
        x = np.arange(3)
        for i, model in enumerate(('unet','yolo')):
            values = [report['models'][model]['pooled'][group][metric]*100 for group in groups]
            bars = ax.bar(x + (i-.5)*.34, values, width=.32, color=colors[model], label='U-Net' if model=='unet' else 'YOLO26n')
            for bar, value in zip(bars, values):
                ax.text(bar.get_x()+bar.get_width()/2, value+1.5, f'{value:.1f}%', ha='center', fontsize=9)
        ax.set_xticks(x, labels); ax.set_ylim(0, 112); ax.set_yticks(range(0,101,20));ax.set_ylabel('%');ax.set_title(title)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    axes[0].legend(loc='lower left')
    notes=[]
    for model in ('unet','yolo'):
        m=report['models'][model]['metrics'];thin=m['one_pixel_lines']
        notes.append(f"{'U-Net' if model=='unet' else 'YOLO26n'}: normal alarms {m['normal_false_alarms']}/{m['normal_images']}; 1-panel-pixel lines {thin['hit']}/{thin['total']}")
    fig.suptitle('OLED white-screen inspection | 1,050 unseen native scenes',fontweight='bold')
    fig.text(.5,.065,'   |   '.join(notes),ha='center',fontsize=9)
    fig.text(.5,.02,'Same-class one-to-one box IoU ≥ 0.50 | Validation-frozen thresholds | Synthetic images; real captures unvalidated',ha='center',fontsize=8,color='#555')
    fig.tight_layout(rect=(0,.11,1,.93));fig.savefig(root/'comparison.png');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('outputs/paired-model-comparison-v1'));a=p.parse_args();render(a.output)
