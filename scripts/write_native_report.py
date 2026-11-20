"""Write a Korean comparison report from frozen experiment artifacts."""
import argparse
import csv
import json
from pathlib import Path

LABELS={'dark_dot':'어두운 점','color_dot':'색 이상점','horizontal_line':'가로선','vertical_line':'세로선','dark_region':'밝기·색 영역 저하'}


def percent(value): return f'{100*value:.1f}%'


def write(root, destination):
    result=json.loads((root/'test_comparison.json').read_text())
    frozen=json.loads((root/'frozen_selection.json').read_text())
    timing=json.loads((root/'inference_timing.json').read_text())
    pixels=json.loads((root/'unet/test_pixel_metrics.json').read_text())
    winner='U-Net' if frozen['selected_model']=='unet' else 'YOLO26n'
    lines=['# 원본 해상도 OLED 판독 비교', '',
        f'검증 분할에서 **{winner}**을 선택하고 설정을 동결한 뒤, 같은 미학습 전체 장면 1,050장으로 두 모델을 비교했습니다.',
        '정상 원본 계열이 학습·검증·시험 사이에 겹치지 않으며, 테스트 결과로 모델이나 임계값을 다시 고르지 않았습니다.', '',
        '## 데이터와 정답', '',
        '- 정상 1,000장, 단일 5종 각 1,000장, 복합 1,000장: 총 7,000장.',
        '- 학습 4,900 / 검증 1,050 / 시험 1,050. 원본 계열은 각각 700 / 150 / 150개이며 계열마다 7가지 변형이 있습니다. 학습 PNG는 실제 1024×1024 픽셀 타일입니다.',
        '- 전체 장면은 5888×3368. 같은 생성 조건과 좌표별 노이즈로 재현하고, 저장된 정상 타일과 픽셀 일치를 확인합니다.',
        '- 정답 마스크는 같은 정상 대비 단독 결함의 최대 RGB 차이가 2 DN 이상인 픽셀입니다. 5개 독립 비트로 겹치는 클래스도 보존합니다.',
        '- 마스크 7,000장 전체의 크기·해시·비트·클래스와 정상 1,000장의 빈 마스크를 확인했습니다.',
        '- 정상 짝과 합성 레시피는 정답 작성·평가에만 사용합니다. 추론 입력은 RGB 사진 한 장입니다.',
        '', '![정상·불량·정답 마스크](../outputs/paired-model-comparison-v1/pixel_masks.png)', '',
        '## 같은 기준으로 비교', '',
        '- 두 모델 모두 1024픽셀 입력, 같은 분할, 좌우·상하 반전만 사용합니다. 축소·모자이크·색 변환과 가중 중복 표집을 사용하지 않습니다.',
        '- 소형 U-Net(186,805개 파라미터)은 무작위 초기화, YOLO26n은 일반 COCO 사전학습 초기화입니다. 초기화와 손실·최적화 방식은 같지 않습니다.',
        '- 전체 장면은 중첩 128픽셀 타일로 추론합니다. 고정 ROI는 (67,67)–(5821,3301)이며 64픽셀 테두리와 광학 흐림 전이 3픽셀을 제외합니다.',
        '- 같은 클래스끼리 박스 IoU 0.5 이상으로 일대일 대응합니다. 점 종류를 틀리면 정답 검출로 세지 않습니다.',
        '- 클래스별 검증 임계값은 정상 오탐을 먼저 줄이고 F1을 높이는 순서로 고릅니다. 모델 선택도 같은 순서입니다.',
        '- 두 모델에 같은 형상 범위와 끊긴 선 결합 규칙을 적용합니다. 점 최대 폭 32px, 선 횡폭 24px 이하·길이 18px 이상·종횡비 4 이상, 영역 최소 폭·높이 24px입니다.',
        '- 선은 중심 차이 2px·최대 간격 128px·두께 비 3 이하일 때 결합합니다. 가까운 평행선과 교차하는 서로 다른 방향의 선은 유지합니다. 영역은 타일 경계에서 여러 박스로 나뉠 수 있어, 중복 억제/겹친 박스 합치기와 IoU 기준 0.35·0.25·0.15를 두 모델에 같은 검증 후보로 적용합니다.',
        '- U-Net 픽셀 임계값은 타일 검증에서 고르고, 전체 장면 검증에서 두 모델의 객체 점수 임계값을 고릅니다. 자세한 값은 frozen_selection.json에 있습니다.',
        '', '## 시험 결과', '', '|종류|U-Net 재현율|U-Net 정밀도|YOLO26n 재현율|YOLO26n 정밀도|', '|---|---:|---:|---:|---:|']
    for title,key in [('점','points'),('선','lines'),('영역','regions')]:
        u=result['models']['unet']['pooled'][key];y=result['models']['yolo']['pooled'][key]
        lines.append(f"|{title}|{percent(u['recall'])} ({u['tp']}/{u['total']})|{percent(u['precision'])}|{percent(y['recall'])} ({y['tp']}/{y['total']})|{percent(y['precision'])}|")
    lines+=['','|세부 클래스|U-Net 재현율 / 정밀도 / F1|YOLO26n 재현율 / 정밀도 / F1|','|---|---:|---:|']
    for kind,label in LABELS.items():
        u=result['models']['unet']['metrics']['classes'][kind];y=result['models']['yolo']['metrics']['classes'][kind]
        lines.append('|'+label+'|'+' / '.join(percent(u[k]) for k in ('recall','precision','f1'))+'|'+' / '.join(percent(y[k]) for k in ('recall','precision','f1'))+'|')
    lines+=['','|추가 항목|U-Net|YOLO26n|','|---|---:|---:|']
    def pair(fn):return '|'.join(fn(result['models'][m]['metrics']) for m in ('unet','yolo'))
    lines.append('|정상 사진 오탐|'+pair(lambda m:f"{m['normal_false_alarms']} / {m['normal_images']}")+'|')
    lines.append('|1패널픽셀 선 검출|'+pair(lambda m:f"{m['one_pixel_lines']['hit']} / {m['one_pixel_lines']['total']}")+'|')
    lines.append('|IoU 기준 중복 검출 수|'+pair(lambda m:str(sum(c['duplicates'] for c in m['classes'].values())))+'|')
    lines.append('|같은 선에 여러 박스가 붙은 객체|'+pair(lambda m:f"{m['line_fragmentation']['lines_with_multiple_boxes']} / {m['line_fragmentation']['total_lines']}")+'|')
    lines.append('|복합 종류 사진의 모든 정답 객체 검출|'+pair(lambda m:f"{m['mixed_type_images']['all_objects_found']} / {m['mixed_type_images']['total']}")+'|')
    lines.append('|군집점 포함 여러 객체 사진의 모든 정답 검출|'+pair(lambda m:f"{m['multiple_defect_images']['all_objects_found']} / {m['multiple_defect_images']['total']}")+'|')
    lines.append('|불량 사진에서 모든 검출 없음|'+pair(lambda m:f"{m['defect_image_misses']} / {m['defect_images']}")+'|')
    lines += ['', '|선 형태|U-Net 검출|YOLO26n 검출|', '|---|---:|---:|']
    for form, label in (('full','전체 선'),('partial','부분 선'),('broken','끊긴 선')):
        values=[]
        for model in ('unet','yolo'):
            item=result['models'][model]['metrics']['line_forms'][form]
            values.append(f"{percent(item['hit']/max(1,item['total']))} ({item['hit']}/{item['total']})")
        lines.append('|'+label+'|'+'|'.join(values)+'|')
    lines+=['','1패널픽셀은 생성 모델 좌표 기준입니다. 기본 촬영 대응은 패널 1픽셀당 이미지 3픽셀이며 흐림·대비에 따라 보이는 폭은 달라집니다. 촬영 영상 1픽셀 선의 성능으로 해석하지 않습니다.', '',
        '![두 모델 비교](../outputs/paired-model-comparison-v1/comparison.png)', '', '## 픽셀 마스크 품질 (U-Net)', '']
    lines += ['U-Net에 한해 기존 시험 타일 1,050장의 픽셀 마스크 품질도 별도로 측정했습니다. 위 표의 전체 사진 객체 검출률과 다른 지표이며, YOLO에는 픽셀 마스크가 없어 같은 수치를 만들지 않습니다.', '',
              '|클래스|시험 픽셀 Dice|시험 픽셀 IoU|', '|---|---:|---:|']
    for kind, label in LABELS.items():
        item=pixels['classes'][kind]
        lines.append(f"|{label}|{percent(item['dice'])}|{percent(item['iou'])}|")
    lines += ['', '## 학습 기록과 실행 시간', '']
    unet=list(csv.DictReader((root/'unet/results.csv').open()));yolo=list(csv.DictReader((root/'yolo/train/results.csv').open()))
    lines.append(f"U-Net은 {len(unet)}개 에포크, YOLO26n은 {len(yolo)}개 에포크를 완료했습니다. 최대 16개입니다. U-Net의 조기 종료 인내는 5회이며, YOLO는 초기 5회에서 7에포크 이후 16회로 늘려 예정한 코사인 학습률 주기를 끝까지 허용했습니다.")
    lines+=['동시 GPU 학습의 메모리 압력 때문에 순차 실행으로 바꿨고 U-Net 배치를 8→4→2, YOLO 배치를 16→4로 줄였습니다. U-Net 완료 후 시스템 메모리 여유가 76%로 회복된 것을 확인해 YOLO는 7에포크 저장 지점에서 16으로 다시 늘렸습니다. 해상도와 학습 사진은 유지했습니다.',
        'U-Net은 2에포크 뒤 최적화 상태 없이 재개해 AdamW 상태를 다시 만들었으며, 3·9에포크 뒤에는 저장한 최적화 상태를 복원했습니다. 3에포크 검증에서 점 출력이 선 주변으로 번지는 것을 확인해 점 클래스의 어려운 음성 픽셀 손실을 강화했습니다. 이 수정은 시험 데이터 접근 전에 했습니다.', '',
        'Apple M4 Pro·메모리 48 GiB의 Mac에서 MPS로 모델을 로드한 뒤 전체 사진을 캐시 없이 반복 검사한 시간입니다. 파일 디코딩·원본 타일 추론·후처리를 포함하며, 합성 렌더링·앱 이미지 저장·화면 그리기는 제외합니다.', '']
    for model in ('unet','yolo'):
        t=timing['models'][model];lines.append(f"- {'U-Net' if model=='unet' else 'YOLO26n'}: 중앙값 {t['median_seconds']:.3f}초, 모델 로드·첫 타일 준비 {t['load_and_warmup_seconds']:.3f}초.")
    lines+=['','## 판독 앱', '',f'검증에서 선택한 {winner} 가중치와 동결 설정을 앱에서 사용합니다. 모델은 프로세스에 유지하며 사진을 바꿀 때 다시 로드하지 않습니다.',
        '원본 사진이 기본 표시이고 검출 위치·픽셀 출력은 선택할 수 있습니다. 결함 목록에서 고르면 표시 없는 원본 일부를 1·2·4·8배로 볼 수 있습니다. 점수는 모델 점수이고, 박스 크기는 이미지 픽셀입니다.',
        '검증한 크기·구도·노출 범위 안에서는 검출 유무로 정상/불량을 판정하고, 범위 밖이면 확인 필요로 표시합니다. 입력·모델 오류를 정상으로 처리하지 않습니다.', '']
    if (root/'app_worker_verification.json').exists():
        qa=json.loads((root/'app_worker_verification.json').read_text())
        lines += [f"실제 상주 워커로 시험 타일 7종과 전체 정상·복합 사진, 총 {len(qa['cases'])}건을 확인했습니다. 모델은 한 번만 로드하고 잘못된 입력 뒤 정상 요청을 처리했습니다. 원본 확대 픽셀과 전체 사진의 평가/파일 입력 예측 일치도 확인했습니다.", '']
    if (root/'app_gui_verification.json').exists():
        lines += ['네이티브 앱에서 정상·얇은 가로선·복합 사진의 판정, 점 선택 확대와 표시 선택을 직접 확인했습니다. 기존 39개 계약·QA·데모 테스트와 새 추론/원본 좌표 검증도 통과했습니다.', '']
    if (root/'error_analysis.json').exists():
        errors=json.loads((root/'error_analysis.json').read_text())
        lines += ['## 남은 오류', '']
        for model in ('unet','yolo'):
            item=errors['models'][model]
            missed=', '.join(f'{LABELS[k]} {v}개' for k,v in item['missed_by_class'].items()) or '없음'
            extra=', '.join(f'{LABELS[k]} {v}개' for k,v in item['extra_by_class'].items()) or '없음'
            lines.append(f"- {'U-Net' if model=='unet' else 'YOLO26n'}: 놓친 정답 {missed}; 정답과 대응되지 않은 출력 {extra}.")
        lines += ['', 'image_id·정답 좌표·생성 속성·가장 가까운 같은 클래스의 IoU는 error_analysis.json에 있습니다. 원인 확정이나 시험 기반 재보정 없이 기록한 오류입니다.', '']
    lines += ['## 한계와 재현 자료', '',
        '모든 수치는 통제된 합성 데이터 기준입니다. 실제 카메라·패널의 픽셀 대응, 렌즈·노출·색 응답과 정상 허용 편차를 보정한 결과가 아니며, 실제 생산 판정 기준을 대신하지 않습니다. 흰 화면 출력 저하에 대한 외관 분류이고 전기적 고장 원인은 추정하지 않습니다.',
        '이번 소형 모델·16 epoch·타일 학습과 고정 후처리 조건의 비교입니다. 특히 긴 전체 선의 외접 박스를 완성해야 검출 성공으로 셉니다. 선 일부에 반응해도 전체 정답과 IoU가 부족하면 실패이며, 더 긴 학습이나 다른 YOLO 구조까지 일반화한 결론은 아닙니다.',
        '', '- 데이터/마스크 해시: data/oled-white-paired-v2/{generation,verification,segmentation,mask_verification}.json',
        '- 분할·평가 규칙: outputs/paired-model-comparison-v1/protocol.json',
        '- 중단·재개·손실 보완: resource_adjustment.json, unet/run.json, yolo/run.json 및 CSV 기록',
        '- 모델 선택: frozen_selection.json; 원시 예측: 각 모델의 raw_full_val/test.jsonl',
        '- 최종 수치: test_comparison.json; 실측 원시 시간: inference_timing.json',
        '', '```bash', '.venv/bin/python scripts/generate_oled_masks.py data/oled-white-paired-v2',
        'artifacts/yolo-env/bin/python scripts/train_paired_unet.py --epochs 16 --batch 2',
        'artifacts/yolo-env/bin/python scripts/train_paired_yolo.py --epochs 16 --batch 4',
        'bash scripts/run_native_evaluation.sh', 'bash scripts/build_mac_inspector.sh', '```', '',
        '완료된 생성·학습 폴더는 덮어쓰지 않습니다. 재실행은 다른 출력 경로를 사용하고, 자원 중단 뒤에는 해당 학습기의 --resume을 사용합니다. 위 명령은 실행 단계 안내이며, 중단/손실 수정 이력까지 단일 명령으로 재현하는 것을 뜻하지 않습니다.', '']
    destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text('\n'.join(lines))
    print(destination)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('outputs/paired-model-comparison-v1'));p.add_argument('--report',type=Path,default=Path('docs/native-comparison.md'));a=p.parse_args();write(a.output,a.report)
