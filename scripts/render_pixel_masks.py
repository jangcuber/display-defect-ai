"""Show unmarked native image details and the actual five-bit training masks."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

COLORS = [(0,160,170),(225,45,155),(245,145,0),(30,105,235),(145,75,210)]
KINDS = ['dark_dot','color_dot','horizontal_line','vertical_line','dark_region']


def render(dataset, output):
    rows = list(map(json.loads,(dataset/'manifest.jsonl').read_text().splitlines()))
    # Fixed first validation family; examples are not selected by model outcomes.
    sid = next(row['source_case_id'] for row in rows if row['split']=='val')
    samples = [next(row for row in rows if row['source_case_id']==sid and row['kind']==kind) for kind in KINDS]
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',18)
    small = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',14)
    canvas = Image.new('RGB',(1160,1450),'#f3f5f8'); draw = ImageDraw.Draw(canvas)
    draw.text((25,16),'Native pixel details and exact training masks',font=font,fill='#15233c')
    for x,label in ((250,'Normal source'),(550,'With defect'),(850,'Exact mask (>= 2 DN)')):
        draw.text((x,54),label,font=font,fill='#15233c')
    for index,row in enumerate(samples):
        normal=cv2.imread(str(dataset/row['normal_image']))[:,:,::-1]
        image=cv2.imread(str(dataset/row['image']))[:,:,::-1]
        mask=cv2.imread(str(dataset/'masks'/row['split']/f"{row['image_id']}.png"),0)
        box=row['objects'][0]['bbox_xyxy'];cx,cy=(box[0]+box[2])//2,(box[1]+box[3])//2
        half=32 if index<2 else 64 if index<4 else min(480,max(box[2]-box[0],box[3]-box[1])//2+32)
        x1,y1,x2,y2=max(0,cx-half),max(0,cy-half),min(1024,cx+half),min(1024,cy+half)
        colors=np.zeros((*mask.shape,3),np.uint8)+25
        for j,color in enumerate(COLORS):colors[(mask&(1<<j))!=0]=color
        top=90+index*264
        draw.text((25,top+40),KINDS[index].replace('_',' '),font=font,fill='#15233c')
        draw.text((25,top+72),f'Native crop:\n{x2-x1} x {y2-y1} px\nNo image markings',font=small,fill='#526174')
        for left,values in zip((240,540,840),(normal,image,colors)):
            tile=Image.fromarray(values[y1:y2,x1:x2]).resize((240,240),Image.Resampling.NEAREST)
            canvas.paste(tile,(left,top))
    draw.text((25,1420),f'{sid} | One synthetic panel pixel = 3 image pixels | Masks are labels; never supplied to the inspector',font=small,fill='#526174')
    output.parent.mkdir(parents=True,exist_ok=True);canvas.save(output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,default=Path('data/oled-white-paired-v2'));p.add_argument('--output',type=Path,default=Path('outputs/paired-model-comparison-v1/pixel_masks.png'));a=p.parse_args();render(a.dataset,a.output)
