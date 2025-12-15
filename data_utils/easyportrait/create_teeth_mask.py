# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser  # 예: 명령줄 인자 파싱을 위해 ArgumentParser를 import

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot  # 예: MMSegmentation의 세그멘테이션 함수들 import

import os  # 예: 파일 경로 조작을 위해 os 모듈 import
import glob  # 예: 파일 리스트를 가져오기 위해 glob 모듈 import
from tqdm import tqdm  # 예: 진행상황(progress bar) 표시를 위해 tqdm import
import numpy as np  # 예: 배열 연산을 위해 numpy import

def main():
    parser = ArgumentParser()  # 예: parser = ArgumentParser()
    parser.add_argument('datset', help='Image file')  # 예: python create_teeth_mask.py ./data/Macron
    parser.add_argument('--config', default="./data_utils/easyportrait/local_configs/easyportrait_experiments_v2/fpn-fp/fpn-fp.py", help='Config file')  # 예: --config 경로 지정
    parser.add_argument('--checkpoint', default="./data_utils/easyportrait/fpn-fp-512.pth", help='Checkpoint file')  # 예: --checkpoint 경로 지정
    # OpenMMLab의 MMSegmentation에서 제공하는 FPN-FP (Feature Pyramid Network - Feature Pyramid) 아키텍처를 기반으로 한 모델
    # EasyPortrait 데이터셋으로 학습된 파라미터를 불러옴
    args = parser.parse_args()  # 예: args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device='cuda:0')  # 예: MMSegmentation 모델을 config와 checkpoint로 초기화

    # test a single image
    dataset_path = os.path.join(args.datset, 'ori_imgs')  # 예: dataset_path = './data/Macron/ori_imgs'
    out_path = os.path.join(args.datset, 'teeth_mask')  # 예: out_path = './data/Macron/teeth_mask'
    os.makedirs(out_path, exist_ok=True)  # 예: teeth_mask 폴더가 없으면 생성

    for file in tqdm(glob.glob(os.path.join(dataset_path, '*.jpg'))):  # 예: ori_imgs 폴더의 모든 jpg 파일에 대해 반복
        result = inference_segmentor(model, file)  # 예: result = inference_segmentor(model, './data/Macron/ori_imgs/00001.jpg')
        result[0][result[0]!=7] = 0  # 예: EasyPortrait 데이터셋에 따라 치아 클래스(7번)만 남기고 나머지는 0으로 만듦
        np.save(file.replace('jpg', 'npy').replace('ori_imgs', 'teeth_mask'), result[0].astype(np.bool_))  # 예: teeth_mask/00001.npy로 저장 (True/False 마스크)
        # 원본 이미지와 동일한 높이와 너비를 가진 2차원 마스크입니다.
        # 해당 픽셀이 EasyPortrait 모델에 의해 치아(클래스 ID 7)로 분류되었다면 해당 위치의 값은 True가 됩니다.
        # 치아가 아닌 다른 모든 영역(클래스 ID가 7이 아닌 영역)은 False 값을 가집니다.
        # 따라서 생성된 .npy 파일은 각 이미지에서 치아 영역의 위치를 나타내는 이진(True/False) 마스크를 담고 있습니다.
        # 이 마스크는 나중에 치아 부분에 대한 특정 처리(예: 렌더링, 색상 조정 등)를 수행할 때 활용될 수 있습니다.
        
if __name__ == '__main__':
    main()  # 예: main() 함수 실행