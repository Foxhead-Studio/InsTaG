#!/usr/bin/python  # 파이썬 인터프리터를 지정하는 shebang
# -*- encoding: utf-8 -*-  # 소스코드의 인코딩을 UTF-8로 지정
import numpy as np  # numpy 라이브러리를 np라는 이름으로 import (수치 연산용)
from model import BiSeNet  # model.py에서 BiSeNet 클래스를 import (세그멘테이션 모델)

import torch  # PyTorch 딥러닝 프레임워크 import

import os  # OS 관련 함수 import
import os.path as osp  # os.path를 osp로 import (경로 관련 함수 사용 편의)

from PIL import Image  # 이미지 처리를 위한 PIL의 Image 모듈 import
import torchvision.transforms as transforms  # 이미지 변환을 위한 torchvision의 transforms 모듈 import
import cv2  # OpenCV 라이브러리 import (이미지 처리용)
from pathlib import Path  # pathlib의 Path 객체 import (경로 조작용)
import configargparse  # configargparse 모듈 import (명령행 인자 파싱)
import tqdm  # tqdm 모듈 import (진행상황 표시용)

# import ttach as tta  # (주석처리) test-time augmentation 라이브러리 import

# 얼굴 파싱 결과를 시각화
# 입력: 원본 이미지와 파싱 결과(각 픽셀이 어떤 얼굴 부위인지 나타내는 클래스 맵)
# 처리: 각 클래스별로 다른 색상으로 시각화
# 1-13번 클래스: 빨간색 (255, 0, 0)
# 11번 클래스: 회색 (100, 100, 100) - 특별 처리
# 14-15번 클래스: 초록색 (0, 255, 0)
# 16번 클래스: 파란색 (0, 0, 255)
# 17-18번 클래스: 검정색 (0, 0, 0)
# 18번 이후: 빨간색 (255, 0, 0)
# 출력: 색상으로 구분된 얼굴 부위 시각화 이미지
def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg',
                     img_size=(512, 512)):
    im = np.array(im)  # 입력 이미지를 numpy 배열로 변환
    vis_im = im.copy().astype(np.uint8)  # 이미지를 uint8 타입으로 복사
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)  # 파싱 결과를 uint8 타입으로 복사
    vis_parsing_anno = cv2.resize(
        vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)  # 파싱 결과를 stride만큼 리사이즈 (최근접 이웃 보간)
    vis_parsing_anno_color = np.zeros(
        (vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + np.array([255, 255, 255])  # 색상 맵을 흰색(255,255,255)으로 초기화

    num_of_class = np.max(vis_parsing_anno)  # 파싱 결과에서 클래스의 최대값(클래스 개수-1)을 구함
    # print(num_of_class)  # (주석) 클래스 개수 출력

    for pi in range(1, 14):  # 1~13번 클래스에 대해
        index = np.where(vis_parsing_anno == pi)  # 해당 클래스의 픽셀 인덱스 추출
        vis_parsing_anno_color[index[0], index[1], :] = np.array([255, 0, 0])  # 해당 픽셀을 빨간색(255,0,0)으로 칠함

    for pi in [11]:  # 11번 클래스에 대해
        index = np.where(vis_parsing_anno == pi)  # 해당 클래스의 픽셀 인덱스 추출
        vis_parsing_anno_color[index[0], index[1], :] = np.array([100, 100, 100])  # 해당 픽셀을 회색(100,100,100)으로 칠함

    for pi in range(14, 16):  # 14~15번 클래스에 대해
        index = np.where(vis_parsing_anno == pi)  # 해당 클래스의 픽셀 인덱스 추출
        vis_parsing_anno_color[index[0], index[1], :] = np.array([0, 255, 0])  # 해당 픽셀을 초록색(0,255,0)으로 칠함
    for pi in range(16, 17):  # 16번 클래스에 대해
        index = np.where(vis_parsing_anno == pi)  # 해당 클래스의 픽셀 인덱스 추출
        vis_parsing_anno_color[index[0], index[1], :] = np.array([0, 0, 255])  # 해당 픽셀을 파란색(0,0,255)로 칠함
    for pi in (17, 18):  # 17, 18번 클래스에 대해
        index = np.where(vis_parsing_anno == pi)  # 해당 클래스의 픽셀 인덱스 추출
        vis_parsing_anno_color[index[0], index[1], :] = np.array([0, 0, 0])  # 해당 픽셀을 검정색(0,0,0)으로 칠함
    for pi in range(18, num_of_class+1):  # 18~num_of_class까지의 클래스에 대해
        index = np.where(vis_parsing_anno == pi)  # 해당 클래스의 픽셀 인덱스 추출
        vis_parsing_anno_color[index[0], index[1], :] = np.array([255, 0, 0])  # 해당 픽셀을 빨간색(255,0,0)으로 칠함

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)  # 색상 맵을 uint8 타입으로 변환
    index = np.where(vis_parsing_anno == num_of_class-1)  # (사용되지 않음) 마지막 클래스의 인덱스 추출
    vis_im = cv2.resize(vis_parsing_anno_color, img_size,
                        interpolation=cv2.INTER_NEAREST)  # 색상 맵을 원본 이미지 크기로 리사이즈
    if save_im:  # 저장 옵션이 True일 때
        cv2.imwrite(save_path, vis_im)  # 지정된 경로에 이미지를 저장


def evaluate(respth='./res/test_res', dspth='./data', cp='model_final_diss.pth'):
    Path(respth).mkdir(parents=True, exist_ok=True)  # 결과 저장 폴더가 없으면 생성

    print(f'[INFO] loading model...')  # 모델 로딩 시작 메시지 출력
    n_classes = 19  # 클래스 개수(19) 지정
    net = BiSeNet(n_classes=n_classes)  # BiSeNet 모델 객체 생성
    net.cuda()  # 모델을 GPU로 이동
    net.load_state_dict(torch.load(cp))  # 지정된 경로에서 모델 파라미터 로드
    net.eval()  # 모델을 평가 모드로 설정

    to_tensor = transforms.Compose([
        transforms.ToTensor(),  # 이미지를 텐서로 변환
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # 정규화
    ])

    image_paths = sorted(os.listdir(dspth))  # 입력 이미지 폴더 내 파일명 정렬 리스트

    with torch.no_grad():  # 파라미터 업데이트 비활성화(추론 모드)
        for image_path in tqdm.tqdm(image_paths):  # 이미지 파일별로 진행상황 표시하며 반복
            if image_path.endswith('.jpg') or image_path.endswith('.png'):  # jpg 또는 png 파일만 처리
                img = Image.open(osp.join(dspth, image_path))  # 이미지를 열기
                ori_size = img.size  # 원본 이미지 크기 저장
                image = img.resize((512, 512), Image.BILINEAR)  # 이미지를 512x512로 리사이즈
                image = image.convert("RGB")  # 이미지를 RGB로 변환
                img = to_tensor(image)  # 이미지를 텐서로 변환 및 정규화

                # test-time augmentation.
                inputs = torch.unsqueeze(img, 0) # [1, 3, 512, 512]  # 배치 차원 추가
                outputs = net(inputs.cuda())  # 모델에 입력하여 예측값 얻기
                parsing = outputs.mean(0).cpu().numpy().argmax(0)  # 예측 결과를 평균내고 argmax로 클래스 맵 생성

                image_path = int(image_path[:-4])  # 파일명에서 확장자 제거 후 정수로 변환
                image_path = str(image_path) + '.png'  # 다시 문자열로 변환 후 .png 확장자 추가

                vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(respth, image_path), img_size=ori_size)
                # 파싱 결과를 시각화하여 결과 폴더에 저장


if __name__ == "__main__":  # 이 파일이 메인으로 실행될 때만 아래 코드 실행
    parser = configargparse.ArgumentParser()  # 명령행 인자 파서 생성
    parser.add_argument('--respath', type=str, default='./result/', help='result path for label')  # 결과 저장 경로 인자 추가
    parser.add_argument('--imgpath', type=str, default='./imgs/', help='path for input images')  # 입력 이미지 경로 인자 추가
    parser.add_argument('--modelpath', type=str, default='data_utils/face_parsing/79999_iter.pth')  # 모델 파라미터 경로 인자 추가
    args = parser.parse_args()  # 인자 파싱
    evaluate(respth=args.respath, dspth=args.imgpath, cp=args.modelpath)  # 평가 함수 실행
