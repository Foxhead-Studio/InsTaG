#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import torch
from PIL import Image
from typing import NamedTuple
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from tqdm import tqdm
import pandas as pd

from utils.sh_utils import SH2RGB
from utils.audio_utils import get_audio_features
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    background: np.array
    talking_dict: dict

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

'''
getNerfppNorm()
임의의 스케일과 위치를 가진 3D 장면을 "원점을 중심으로 하고 특정 크기를 가진 구 안에 들어오는" 표준화된 형태로 변환하는 데 필요한 정보 계산
radius: 평균 중심에서 가장 멀리 떨어진 카메라까지의 거리보다 10% 더 크게 반지름
translate: 장면의 평균 중심이 월드 좌표계의 원점 $(0, 0, 0)$으로 이동하도록하는 값
'''
def getNerfppNorm(cam_info):
    # getNerfppNorm 함수는 NeRF++ 논문에서 사용하는 정규화 파라미터(중심, 반지름)를 계산하는 함수입니다.
    # 입력 cam_info는 카메라 정보 리스트입니다.
    
    # train 카메라 정보에서 평균 중심과 이로부터의 이동 반경의 최대 반지름을 계산하는 함수
    def get_center_and_diag(cam_centers):
        # get_center_and_diag 함수는 여러 카메라의 중심 좌표(cam_centers)로부터
        # 전체 카메라의 평균 중심과, 그 중심으로부터 가장 멀리 떨어진 카메라까지의 거리를 계산합니다.
        cam_centers = np.hstack(cam_centers)
        # np.hstack(cam_centers)는 cam_centers 리스트에 있는 (3, 1) 형태의 배열들을
        # 수평 방향(열 방향)으로 쌓아서 (3, N) 형태의 배열로 만듭니다.
        # 예시: [array([[1],[2],[3]]), array([[4],[5],[6]])] -> array([[1,4],[2,5],[3,6]])
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        # np.mean(cam_centers, axis=1, keepdims=True)는 각 행(즉, x, y, z 좌표별)로 평균을 구합니다.
        # 결과는 (3, 1) 형태의 평균 중심 좌표가 됩니다.
        # 예시: cam_centers = [[1,4],[2,5],[3,6]] -> 평균: [[2.5],[3.5],[4.5]]
        center = avg_cam_center
        # 평균 중심 좌표를 center 변수에 저장합니다.
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        # cam_centers - center는 각 카메라 중심에서 평균 중심을 뺀 벡터입니다.
        # np.linalg.norm(..., axis=0, keepdims=True)는 각 열(카메라별)마다 유클리드 거리를 계산합니다.
        # 결과는 (1, N) 형태의 거리 배열이 됩니다.
        # 예시: center = [[2.5],[3.5],[4.5]], cam_centers = [[1,4],[2,5],[3,6]]
        # cam_centers - center = [[-1.5, 1.5], [-1.5, 1.5], [-1.5, 1.5]]
        # norm = [sqrt(1.5^2+1.5^2+1.5^2), sqrt(1.5^2+1.5^2+1.5^2)] = [2.598, 2.598]
        diagonal = np.max(dist)
        # np.max(dist)는 거리 배열에서 가장 큰 값을 diagonal로 저장합니다.
        # 즉, 평균 중심에서 가장 멀리 떨어진 카메라까지의 거리입니다.
        return center.flatten(), diagonal
        # center.flatten()은 (3, 1) -> (3,) 1차원 배열로 변환합니다.
        # 중심 좌표와 대각선 길이를 반환합니다.

    cam_centers = []
    # cam_centers 리스트를 초기화합니다. 각 카메라의 중심 좌표를 저장할 예정입니다.

    for cam in cam_info:
        # cam_info 리스트의 각 카메라(cam)에 대해 반복합니다.
        W2C = getWorld2View2(cam.R, cam.T)
        # getWorld2View2(cam.R, cam.T)는 월드 좌표계에서 카메라 좌표계로 변환하는 4x4 행렬을 만듭니다.
        # cam.R: 회전 행렬, cam.T: 이동 벡터
        C2W = np.linalg.inv(W2C)
        # np.linalg.inv(W2C)는 W2C 행렬을 역행렬로 만들어 카메라 좌표계에서 월드 좌표계로 변환하는 행렬(C2W)을 만듭니다.
        cam_centers.append(C2W[:3, 3:4])
        # C2W[:3, 3:4]는 변환 행렬의 마지막 열(3번째 인덱스)에서 x, y, z 위치만 추출합니다.
        # (3, 1) 형태의 카메라 중심 좌표를 cam_centers에 추가합니다.

    center, diagonal = get_center_and_diag(cam_centers)
    # get_center_and_diag(cam_centers)를 호출하여 전체 카메라의 평균 중심(center)과 대각선 길이(diagonal)를 구합니다.

    radius = diagonal * 1.1
    # radius는 diagonal에 1.1을 곱해서 약간 더 여유 있게 잡습니다.
    # 즉, 평균 중심에서 가장 멀리 떨어진 카메라까지의 거리보다 10% 더 크게 반지름을 설정합니다.

    translate = -center
    # translate는 중심 좌표의 음수입니다.
    # 나중에 씬 전체를 원점으로 이동시키는 데 사용됩니다.

    return {"translate": translate, "radius": radius}
    # translate와 radius 값을 딕셔너리로 반환합니다.

    # translate (이동 벡터):
    # 이동 벡터는 계산된 평균 중심의 음수 값입니다 (예: $(-100, -50, -20)$). 이 값은 앞으로 장면 내의 모든 3D 요소(가우시안 포인트, 카메라 위치 등)에 더해져, 장면의 평균 중심이 월드 좌표계의 원점 $(0, 0, 0)$으로 이동하도록 만듭니다. 
    # 이렇게 하면 3D 모델이 항상 원점 주변의 "정규화된" 공간에서 작업하게 되어 수치적 안정성이 향상됩니다.

    # radius (반지름):
    # 이 반지름 값 (예: $2.75$)은 정규화된 장면이 대략 이 반경을 가진 구 안에 들어온다는 것을 나타냅니다. 
    # 3D 모델은 이 radius 정보를 활용하여 렌더링할 때 공간을 효율적으로 샘플링하거나, 무한한 3D 공간을 특정 유한한 영역으로 매핑할 수 있습니다. 
    # 예를 들어, 모델이 3D 공간을 탐색할 때 이 반지름 범위 내에서만 탐색하도록 제한하여 불필요한 계산을 줄일 수 있습니다.

    # 결론적으로, getNerfppNorm 함수는 임의의 스케일과 위치를 가진 3D 장면을 "원점을 중심으로 하고 특정 크기를 가진 구 안에 들어오는" 표준화된 형태로 변환하는 데 필요한 정보를 계산하여, 
    # 복잡한 3D 렌더링 및 재구성 모델의 학습 및 추론 과정을 단순화하고 안정화하는 데 핵심적인 역할을 합니다.

# 좌표, 색상, 법선 벡터를 받아 포인트 클라우드를 생성하는 함수
def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

# 3D 포인트 클라우드 데이터를 PLY 파일로 저장하는 함수
def storePly(path, xyz, rgb):
    # path: 저장할 파일 경로 (예: 'macron/points3d.ply')
    # xyz: (N, 3) 크기의 numpy 배열, 각 행은 한 점의 (x, y, z) 좌표입니다.
    # rgb: (N, 3) 크기의 numpy 배열, 각 행은 한 점의 (r, g, b) 색상값입니다. 값의 범위는 0~255여야 합니다.

    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    # dtype은 PLY 파일의 vertex 속성에 맞게 각 필드의 데이터 타입을 지정합니다.
    # 'f4'는 32비트 float, 'u1'은 8비트 unsigned int입니다.
    # 예시: x, y, z는 float32, red, green, blue는 uint8로 저장됩니다.

    normals = np.zeros_like(xyz)
    # normals는 (N, 3) 크기의 0으로 채워진 배열입니다.
    # 각 점의 노멀 벡터(nx, ny, nz)를 모두 0으로 초기화합니다.
    # 예시: normals[0] = [0, 0, 0]

    elements = np.empty(xyz.shape[0], dtype=dtype)
    # elements는 N개의 구조화된 배열로, 각 점의 속성을 저장할 공간입니다.
    # 예시: elements[0]은 (x, y, z, nx, ny, nz, r, g, b) 정보를 담습니다.

    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    # xyz, normals, rgb를 (N, 9)로 합칩니다.
    # 예시: attributes[0] = [x, y, z, 0, 0, 0, r, g, b]

    elements[:] = list(map(tuple, attributes))
    # attributes의 각 행을 튜플로 변환하여 elements에 할당합니다.
    # 예시: elements[0] = (x, y, z, 0, 0, 0, r, g, b)

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    # vertex_element는 'vertex'라는 이름의 PlyElement 객체입니다.
    # 이 객체는 PLY 파일의 vertex section을 정의합니다.

    ply_data = PlyData([vertex_element])
    # ply_data는 PlyData 객체로, 여러 PlyElement를 포함할 수 있습니다.
    # 여기서는 vertex_element만 포함합니다.

    ply_data.write(path)
    # ply_data를 지정한 경로(path)에 PLY 파일로 저장합니다.
    # 예시: '/home/user/points3d.ply' 파일이 생성됩니다.

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".jpg", audio_file='', audio_extractor='deepspeech', preload=True):
    # path: '/home/white/github/InsTaG/data/pretrain/macron'
    # transformsfile: 'transforms_train.json'
    # white_background: False
    # extension: '.jpg'
    # audio_file: ''
    # audio_extractor: 'deepspeech'
    # preload: True

    # readCamerasFromTransforms 함수는 주어진 경로(path)와 변환 파일(transforms_train.json)을 이용해 카메라 정보를 읽어오는 함수입니다.
    # white_background, extension, audio_file, audio_extractor, preload는 추가적인 옵션입니다.
    cam_infos = [] # 개별 카메라 정보 객체들을 저장할 준비
    # cam_infos 리스트를 초기화합니다. 이 리스트에는 각 카메라의 정보가 저장될 예정입니다.
    postfix_dict = {"deepspeech": "ds", "esperanto": "eo", "hubert": "hu"} #  오디오 추출기 종류에 따른 파일명 접미사
    # 오디오 추출기(audio_extractor)별로 파일명에 붙는 접미사를 딕셔너리로 정의합니다.
    # 예시: audio_extractor가 'deepspeech'면 'ds'가 postfix로 사용됩니다.

    # 1. 카메라의 초점 거리(focal_len)와 각 이미지 프레임(frames)에 대한 상세 데이터를 저장
    with open(os.path.join(path, transformsfile)) as json_file: # '/home/white/github/InsTaG/data/pretrain/macron' + 'transforms_train.json'
        # os.path.join을 이용해 path와 transformsfile을 합쳐 JSON 파일 경로를 만들고, 해당 파일을 읽기 모드로 열기.
        contents = json.load(json_file) # json.load를 사용해 JSON 파일의 내용을 파이썬 딕셔너리로 읽어옵니다.
        focal_len = contents["focal_len"] # 1200.0
        # contents에서 "focal_len" 키의 값을 가져와 focal_len 변수에 저장합니다.

        frames = contents["frames"] # 모든 프레임 정보
        # contents에서 "frames" 키의 값을 가져와 frames 변수에 저장합니다.
        # frames는 각 이미지 프레임에 대한 딕셔너리 리스트(img_id, aud_id, transform_matrix)가 프레임 개수만큼 존재합니다.


        # 2. 오디오 피처를 가져와 (프레임 수, 채널 수, 특징 차원)과 같은 형태로 재정렬하여 auds 변수에 할당.
        if audio_extractor == "ave": # 디폴트 값이 'deepspeech'이므로 건너 뜀.
            # audio_extractor가 "ave"인 경우에만 아래 블록을 실행합니다.
            from torch.utils.data import DataLoader
            # PyTorch의 DataLoader를 임포트합니다.
            from scene.motion_net import AudioEncoder
            # scene.motion_net 모듈에서 AudioEncoder 클래스를 임포트합니다.
            from utils.audio_utils import AudDataset
            # utils.audio_utils 모듈에서 AudDataset 클래스를 임포트합니다.
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # CUDA가 사용 가능하면 'cuda', 아니면 'cpu'를 device로 설정합니다.
            model = AudioEncoder().to(device).eval()
            # AudioEncoder 모델을 생성하여 device로 옮기고, eval()로 평가 모드로 전환합니다.
            ckpt = torch.load('./data/audio_visual_encoder.pth')
            # './data/audio_visual_encoder.pth'에서 사전학습된 가중치(체크포인트)를 불러옵니다.
            model.load_state_dict({f'audio_encoder.{k}': v for k, v in ckpt.items()})
            # 체크포인트의 키에 'audio_encoder.'를 붙여서 모델에 가중치를 로드합니다.
            if audio_file == '':
                # audio_file이 빈 문자열이면(즉, 별도 오디오 파일이 지정되지 않은 경우)
                dataset = AudDataset(os.path.join(path, 'aud.wav'))
                # path 경로에 있는 'aud.wav' 파일을 AudDataset으로 불러옵니다.
            else:
                # audio_file이 지정된 경우
                dataset = AudDataset(audio_file)
                # 해당 audio_file을 AudDataset으로 불러옵니다.
            data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
            # DataLoader를 사용해 dataset을 배치 크기 64로 불러옵니다. shuffle은 하지 않습니다.
            outputs = []
            # 오디오 인코더의 출력을 저장할 리스트를 초기화합니다.
            for mel in data_loader:
                # data_loader에서 mel-spectrogram 배치를 하나씩 가져옵니다.
                mel = mel.to(device)
                # mel 데이터를 device(GPU 또는 CPU)로 옮깁니다.
                with torch.no_grad():
                    # gradient 계산을 하지 않는(no_grad) 블록입니다.
                    out = model(mel)
                    # 오디오 인코더 모델에 mel을 입력하여 출력을 얻습니다.
                outputs.append(out)
                # 출력(out)을 outputs 리스트에 추가합니다.
            outputs = torch.cat(outputs, dim=0).cpu()
            # outputs 리스트의 텐서들을 첫 번째 차원(dim=0)으로 이어붙이고, CPU로 옮깁니다.
            first_frame, last_frame = outputs[:1], outputs[-1:]
            # outputs의 첫 프레임과 마지막 프레임을 각각 first_frame, last_frame에 저장합니다.
            aud_features = torch.cat([first_frame.repeat(2, 1), outputs, last_frame.repeat(2, 1)],
                                        dim=0).unsqueeze(0).permute(1, 2, 0).numpy()
            # first_frame을 2번 반복, outputs, last_frame을 2번 반복하여 이어붙입니다.
            # 이어붙인 결과를 첫 번째 차원에 대해 unsqueeze(0)로 차원을 추가합니다.
            # permute(1, 2, 0)으로 차원 순서를 바꿉니다.
            # 마지막으로 numpy()로 넘파이 배열로 변환합니다.
            # 예시: (8732, 16) -> (8736, 16) -> (1, 8736, 16) -> (8736, 16, 1)
            # aud_features = np.load(os.path.join(self.root_path, 'aud_ave.npy'))
            # (주석) 기존에는 np.load로 바로 불러오기도 했습니다.
        
        # 디폴트 세팅의 경우 아래를 실행
        elif audio_file == '':
            # audio_extractor가 "ave"가 아니고, audio_file이 빈 문자열인 경우
            # postfix_dict = {"deepspeech": "ds", "esperanto": "eo", "hubert": "hu"} 이므로
            aud_features = np.load(os.path.join(path, 'aud_{}.npy'.format(postfix_dict[audio_extractor]))) # 'aud_ds.npy'
            # path 경로에 있는 'aud_ds.npy' 파일을 np.load로 불러옵니다.

        else:
            # audio_file이 지정된 경우
            aud_features = np.load(audio_file)
            # 해당 audio_file을 np.load로 불러옵니다.
        
        aud_features = torch.from_numpy(aud_features)
        # np.load로 불러온 aud_features를 torch 텐서로 변환합니다.
        aud_features = aud_features.float().permute(0, 2, 1)
        # aud_features를 float 타입으로 변환하고, 차원 순서를 (0, 2, 1)로 바꿉니다.
        # 예시: 원래 (8732, 16, 29)였다면 (8732, 29, 16)으로 바뀝니다.
        auds = aud_features # torch.Size([8732, 29, 16])
        # auds 변수에 aud_features를 할당합니다.


        '''
        3. 얼굴 표정 특징을 가져와 au_exp 변수에 할당.
        이 코드 블록은 `readCamerasFromTransforms` 함수 내에서 `au.csv` 파일로부터 얼굴 액션 유닛(Action Unit, AU) 데이터를 읽어와서 특정 방식으로 처리하는 과정을 담당합니다.
        이 데이터는 주로 얼굴 표정이나 움직임, 특히 입술 움직임이나 눈 깜빡임과 같은 세부적인 얼굴 동작을 나타내는 데 사용됩니다.

        1.  **`au.csv` 파일 로드**:
            먼저, 주어진 경로(예: `/home/white/github/InsTaG/data/pretrain/macron/`)에 있는 `au.csv` 파일을 `pandas` 라이브러리를 사용하여 읽어 들여 데이터프레임으로 만듭니다.
            이 CSV 파일에는 시간에 따른 다양한 얼굴 액션 유닛의 강도 값이 기록되어 있습니다.

        2.  **주요 액션 유닛 추출**:
            *   `au_blink`: 데이터프레임에서 'AU45\_r' 열의 값을 추출하여 `au_blink` 변수에 저장합니다. 'AU45\_r'은 주로 눈 깜빡임(blink)과 관련된 액션 유닛입니다.
            *   `au25`: 'AU25\_r' 열의 값을 추출하여 `au25` 변수에 저장합니다. 'AU25\_r'은 입술 벌림(lips part)과 관련된 액션 유닛입니다.

        3.  **`au25` 값 정규화 및 백분위수 계산**:
            *   `au25` 값의 범위를 정규화합니다. 이 값은 0보다 작으면 0으로, 그리고 전체 `au25` 값 중 95번째 백분위수보다 크면 해당 95번째 백분위수 값으로 제한됩니다.
                이는 데이터의 이상치(outlier)를 제거하고 값의 스케일을 조정하여 모델 학습에 안정성을 더하기 위함입니다.
            *   정규화된 `au25` 값에 대해 25번째, 50번째, 75번째 백분위수(각각 1사분위수, 중앙값, 3사분위수)와 최대값을 계산하여 `au25_25`, `au25_50`, `au25_75`, `au25_100` 변수에 저장합니다.
                이러한 백분위수 정보는 `au25` 값의 분포를 이해하고 후속 처리에서 기준점으로 활용될 수 있습니다.

        4.  **선택된 액션 유닛 집합 준비**:
            *   `AU1`, `AU4`, `AU5`, `AU6`, `AU7`, 그리고 `AU45`와 같이 미리 정의된 특정 액션 유닛들만 선택하여 `au_exp`라는 리스트에 모읍니다.
            *   각 액션 유닛 값은 데이터프레임에서 추출된 후, 차원이 `(데이터 개수,)`에서 `(데이터 개수, 1)` 형태로 확장되어 리스트에 추가됩니다.
            *   특히 'AU45\_r' 값은 추출 후 0에서 2 사이로 값이 제한됩니다. 이는 눈 깜빡임 강도의 현실적인 범위를 반영하기 위한 처리로 보입니다.

        5.  **최종 액션 유닛 배열 생성**:
            `au_exp` 리스트에 모인 모든 액션 유닛 값들을 마지막 차원(`axis=-1`)을 기준으로 이어 붙여 하나의 NumPy 배열로 만듭니다. 이 배열은 `np.float32` 타입으로 변환됩니다.
            결과적으로 이 배열은 각 프레임에 대한 선택된 액션 유닛들의 강도 값을 포함하는 `(프레임 수, 선택된 AU 개수)` 형태의 2차원 데이터가 됩니다.

        이러한 과정을 통해 얼굴의 주요 움직임(눈 깜빡임, 입술 움직임, 기타 표정)을 수치화한 액션 유닛 데이터가 준비되며, 이는 3D 가우시안 스플래팅과 같은 후속 렌더링 또는 애니메이션 생성 과정에서 얼굴의 동적인 변화를 제어하는 입력으로 사용될 수 있습니다.
        '''

        au_info = pd.read_csv(os.path.join(path, 'au.csv'))
        # au.csv 파일을 읽어서 DataFrame으로 불러옵니다. #(8732, 714)
        # 예시: '/home/white/github/InsTaG/data/pretrain/macron/au.csv'
        
        au_blink = au_info['AU45_r'].values 
        # AU45_r(눈 깜빡임, blink) 액션 유닛의 값을 numpy array로 추출합니다.
        # shape 예시: (8732,)
        
        au25 = au_info['AU25_r'].values 
        # AU25_r(입벌림, lips part) 액션 유닛의 값을 numpy array로 추출합니다.
        # shape 예시: (8732,)
        
        au25 = np.clip(au25, 0, np.percentile(au25, 95)) 
        # AU25_r 값이 0보다 작으면 0으로, 95번째 백분위수보다 크면 그 값으로 잘라줍니다.
        # 즉, 이상치(outlier)를 제거하고 값의 범위를 제한합니다.
        # 예시: au25의 95% 백분위수가 2.5라면, 이를 넘는 값들은 모두 2.5로 바뀝니다.
        
        au25_25 = np.percentile(au25, 25)
        # AU25_r 값의 25번째 백분위수(1사분위수)를 구합니다.
        au25_50 = np.percentile(au25, 50)
        # AU25_r 값의 50번째 백분위수(중앙값, 2사분위수)를 구합니다.
        au25_75 = np.percentile(au25, 75)
        # AU25_r 값의 75번째 백분위수(3사분위수)를 구합니다.
        au25_100 = au25.max()
        # AU25_r 값의 최대값을 구합니다.
        
        au_exp = []
        # 여러 액션 유닛의 값을 모아둘 리스트를 만듭니다.
        for i in [1, 4, 5, 6, 7, 45]:
            # AU1, AU4, AU5, AU6, AU7, AU45(눈 깜빡임)만 추출합니다.
            _key = 'AU' + str(i).zfill(2) + '_r'
            # 예시: i=1이면 'AU01_r', i=45면 'AU45_r'
            au_exp_t = au_info[_key].values
            # 해당 액션 유닛의 값을 numpy array로 추출합니다.
            # 예시: au_info['AU01_r'].values의 형태는 (8732,)입니다.
            if i == 45:
                au_exp_t = au_exp_t.clip(0, 2)
                # AU45(눈 깜빡임)는 값의 범위를 0~2로 제한합니다.
                # 예시: 2보다 큰 값은 2로 바뀝니다.
            au_exp.append(au_exp_t[:, None])
            # (8732,) -> (8732, 1)로 차원을 확장해서 리스트에 추가합니다.
        au_exp = np.concatenate(au_exp, axis=-1, dtype=np.float32)
        # 리스트에 모은 여러 액션 유닛을 (8732, 6) 형태로 합칩니다.
        # dtype을 float32로 맞춥니다.
        # 예시: [[AU01, AU04, AU05, AU06, AU07, AU45], ...]
        

        '''
        4. 얼굴 랜드마크 정보를 가져와 ldmks_lips, ldmks_mouth, ldmks_lhalf 변수에 할당.
        이 코드 블록은 각 이미지 프레임에 해당하는 얼굴 랜드마크 데이터(`*.lms` 파일)를 읽어와, 얼굴의 특정 영역(입술, 입 내부, 얼굴 하단)에 대한 경계 상자(bounding box)와 움직임 정보를 계산하고 저장하는 역할을 합니다.

        1.  **랜드마크 데이터 로드**:
            위에서 생성한 `frames` 리스트를 반복하면서 각 프레임의 `img_id`를 사용하여 해당 랜드마크 파일을 불러옵니다.
            이 랜드마크 파일(`*.lms`)은 얼굴의 68개 주요 지점의 2차원 좌표를 포함하는 NumPy 배열입니다.

        2.  **관심 영역 랜드마크 인덱스 정의**:
            68개의 랜드마크 중에서 입술(lips)과 입 내부(mouth)를 구성하는 특정 랜드마크들의 인덱스 범위를 `slice` 객체로 정의합니다.
            예를 들어, 48번부터 59번 랜드마크는 입술 영역, 60번부터 67번 랜드마크는 입 내부 영역을 나타냅니다.

        3.  **영역별 경계값 계산**:
            *   **입술 영역**: 정의된 입술 랜드마크들을 사용하여 해당 영역의 최소/최대 X, Y 좌표를 계산하여 입술의 경계 상자를 구합니다. 이 경계 상자 정보는 `ldmks_lips` 리스트에 저장됩니다.
            *   **입 내부 영역**: 입 내부 랜드마크들의 최소/최대 Y 좌표를 계산하여 `ldmks_mouth` 리스트에 저장합니다. 이는 입의 수직적인 움직임 범위를 나타내는 데 사용됩니다.
            *   **얼굴 하단 영역**: 얼굴의 특정 랜드마크(31번부터 35번)와 전체 얼굴 랜드마크의 경계값을 조합하여 얼굴 하단 영역의 경계 상자를 계산합니다. 이 정보는 `ldmks_lhalf` 리스트에 저장됩니다.

        4.  **NumPy 배열로 변환**:
            모든 프레임에 대한 반복이 끝나면, 각 리스트(`ldmks_lips`, `ldmks_mouth`, `ldmks_lhalf`)에 저장된 정보들을 효율적인 처리를 위해 NumPy 배열로 변환합니다.

        5.  **입 내부 움직임 범위 계산**:
            `ldmks_mouth` 배열을 사용하여 각 프레임에서 입 내부 영역의 수직적인 길이(최대 Y좌표 - 최소 Y좌표)를 계산합니다. 이 길이들 중에서 가장 작은 값(`mouth_lb`)과 가장 큰 값(`mouth_ub`)을 찾아 저장합니다.
            이 값들은 입이 얼마나 벌어질 수 있는지의 최소/최대 범위를 나타냅니다.

        결론적으로, 이 코드 섹션은 원본 이미지에서 추출된 얼굴 랜드마크를 분석하여 입술, 입 내부, 얼굴 하단과 같은 핵심 얼굴 부위의 동적인 위치와 크기 정보를 시간에 따라 정량화합니다. 이러한 정보는 주로 3D 얼굴 애니메이션, 특히 말하는 얼굴이나 표정 변화를 모델링하고 렌더링하는 데 필요한 세부적인 제어 파라미터로 활용될 수 있습니다.
        '''
        ldmks_lips = []
        # 입술 영역의 랜드마크 정보를 담을 리스트입니다.
        ldmks_mouth = []
        # 입 안(구강) 영역의 랜드마크 정보를 담을 리스트입니다.
        ldmks_lhalf = []
        # 얼굴 하단(아랫부분) 영역의 랜드마크 정보를 담을 리스트입니다.
        for idx, frame in tqdm(enumerate(frames)):
            # tqdm(enumerate(frames))를 사용해서 프레임별로 반복합니다.
            # idx는 인덱스, frame은 각 프레임의 정보를 담고 있습니다.
            lms = np.loadtxt(os.path.join(path, 'ori_imgs', str(frame['img_id']) + '.lms')) # [68, 2]
            # os.path.join을 이용해 'ori_imgs' 폴더에서 해당 프레임(img_id = 0, 1, 2, ...)의 랜드마크(.lms) 파일을 불러옵니다.
            # np.loadtxt로 68개의 랜드마크 좌표(2차원)를 읽어옵니다. 예시: (68, 2) shape
            lips = slice(48, 60)
            # 입술(lips) 영역의 랜드마크 인덱스(48~59, 파이썬 슬라이스는 끝-1까지)를 지정합니다.
            mouth = slice(60, 68)
            # 입(mouth) 내부 영역의 랜드마크 인덱스(60~67)를 지정합니다.
            xmin, xmax = int(lms[lips, 1].min()), int(lms[lips, 1].max())
            # 입술 영역의 y좌표(1번 인덱스)의 최소, 최대값을 구해서 xmin, xmax로 저장합니다.
            ymin, ymax = int(lms[lips, 0].min()), int(lms[lips, 0].max())
            # 입술 영역의 x좌표(0번 인덱스)의 최소, 최대값을 구해서 ymin, ymax로 저장합니다.
            ldmks_lips.append([int(xmin), int(xmax), int(ymin), int(ymax)])
            # 입술 영역의 경계값(xmin, xmax, ymin, ymax)을 리스트에 추가합니다.
            ldmks_mouth.append([int(lms[mouth, 1].min()), int(lms[mouth, 1].max())])
            # 입 내부 영역의 y좌표(1번 인덱스) 최소, 최대값을 구해서 리스트에 추가합니다.
            lh_xmin, lh_xmax = int(lms[31:36, 1].min()), int(lms[:, 1].max()) # actually lower half area
            # 얼굴 하단(31~35번 랜드마크)의 y좌표 최소값과 전체 랜드마크의 y좌표 최대값을 구합니다.
            xmin, xmax = int(lms[:, 1].min()), int(lms[:, 1].max())
            # 전체 랜드마크의 y좌표 최소, 최대값을 구합니다.
            ymin, ymax = int(lms[:, 0].min()), int(lms[:, 0].max())
            # 전체 랜드마크의 x좌표 최소, 최대값을 구합니다.
            # self.face_rect.append([xmin, xmax, ymin, ymax])
            # (주석) 전체 얼굴 영역의 경계값을 저장할 수 있습니다.
            ldmks_lhalf.append([lh_xmin, lh_xmax, ymin, ymax])
            # 얼굴 하단 영역의 경계값(lh_xmin, lh_xmax, ymin, ymax)을 리스트에 추가합니다.
            
        ldmks_lips = np.array(ldmks_lips)
        # 입술 영역 경계값 리스트를 numpy array로 변환합니다. shape 예시: (8732, 4)
        ldmks_mouth = np.array(ldmks_mouth)
        # 입 내부 영역 경계값 리스트를 numpy array로 변환합니다. shape 예시: (8732, 2)
        ldmks_lhalf = np.array(ldmks_lhalf)
        # 얼굴 하단 영역 경계값 리스트를 numpy array로 변환합니다. shape 예시: (8732, 4)
        mouth_lb = (ldmks_mouth[:, 1] - ldmks_mouth[:, 0]).min()
        # 각 프레임별로 입 내부의 y좌표 길이(최대-최소)를 구한 뒤, 그 중 최소값을 mouth_lb로 저장합니다.
        mouth_ub = (ldmks_mouth[:, 1] - ldmks_mouth[:, 0]).max()
        # 각 프레임별로 입 내부의 y좌표 길이(최대-최소)를 구한 뒤, 그 중 최대값을 mouth_ub로 저장합니다.

        '''
        이제 각 이미지 프레임에 대한 포괄적인 카메라 및 얼굴 관련 데이터를 수집하고 처리하여 `CameraInfo` 객체 리스트를 생성합니다.
        이 리스트는 나중에 3D 장면을 구성하고 얼굴 애니메이션을 제어하는 데 사용됩니다.

        1.  **프레임 반복 및 카메라 변환 처리**:
            *   코드에 있는 모든 프레임(`frames` 리스트)을 `tqdm` 진행 표시줄과 함께 하나씩 반복 처리합니다.
            *   각 프레임에서 transform_matrix에 해당하는 카메라-월드 변환 행렬(`c2w`)을 가져옵니다.
                이 행렬은 얼굴이 원점에 있을 때, 카메라의 상대적인 위치를 나타냅니다.
                이후 이 행렬은 Blender나 OpenGL에서 사용되는 카메라 축(Y는 위, Z는 뒤)을 COLMAP에서 사용되는 축(Y는 아래, Z는 앞)으로 변경하기 위해 조정됩니다.
            *   이 변환 행렬을 역변환하여 카메라가 원점에 있을 때 얼굴의 상대적인 위치를 나타내는 월드-카메라 변환 행렬(`w2c`)을 얻습니다.
            *   `w2c`에서 회전 행렬(`R`)과 변환 벡터(`T`)를 추출합니다. `R`은 CUDA 코드의 'glm' 라이브러리 호환성을 위해 전치(transpose)됩니다.

        2.  **이미지 및 `talking_dict` 초기화**:
            *   현재 프레임에 대한 다양한 동적 얼굴 정보를 저장할 `talking_dict`를 초기화합니다.
            *   이미지 ID, 이미지 경로, 이미지 이름을 `talking_dict`에 추가합니다.
            *   `preload` 옵션이 활성화되어 있거나 첫 번째 프레임인 경우, 해당 프레임의 실제 이미지 파일(`gt_imgs` 폴더 내)을 열어 NumPy 배열로 변환하고 이미지 너비(`w`)와 높이(`h`)를 얻습니다.

        3.  **마스크 데이터 로드 및 처리**:
            *   해당 프레임의 치아 마스크(`.npy` 파일)와 파싱 마스크(`.png` 파일) 경로를 `talking_dict`에 저장합니다.
            *   `preload` 옵션이 활성화된 경우, 이 마스크 파일들을 로드하여 실제 마스크 데이터를 가져옵니다.
                파싱 마스크는 얼굴, 머리카락, 입 영역을 나타내는 세분화된 마스크로 변환되어 `talking_dict`에 추가됩니다.
                `face_mask`는 파싱 마스크의 특정 색상 값을 기반으로 생성되며 `teeth_mask`와 XOR 연산을 통해 보정됩니다.

        4.  **오디오 및 액션 유닛(AU) 데이터 통합**:
            *   `get_audio_features` 함수를 사용하여 이전에 로드된 전체 오디오 특징(`auds`)에서 현재 프레임에 해당하는 오디오 특징을 추출하여 `talking_dict`에 저장합니다.
                오디오 특징 배열의 길이를 초과하는 프레임이 발생하면 경고 메시지를 출력하고 루프를 중단합니다.
            *   이전에 준비된 `au_blink` (눈 깜빡임), `au25` (입술 벌림), `au_exp` (선택된 액션 유닛 집합) 데이터에서 현재 프레임에 해당하는 값을 추출합니다.
                `au_blink` 값은 0에서 2 사이로 클리핑된 후 2로 나누어 정규화됩니다.
                `au25`는 해당 프레임의 값과 미리 계산된 백분위수 값들(`au25_25`, `au25_50`, `au25_75`, `au25_100`)을 포함하는 리스트로 저장됩니다.
                이 모든 액션 유닛 값들은 PyTorch 텐서로 변환되어 `talking_dict`에 추가됩니다.

        5.  **얼굴 영역 경계 상자 및 움직임 정보 추가**:
            *   이전에 계산된 입술(`ldmks_lips`) 및 얼굴 하단(`ldmks_lhalf`) 바운딩 박스 정보를 기반으로 현재 프레임에 대한 경계 상자를 가져와 `talking_dict`에 저장합니다.
                입술 경계 상자는 가로세로 비율을 맞추기 위해 패딩(padding)됩니다.
            *   입 내부 영역의 최소/최대 길이(`mouth_lb`, `mouth_ub`)와 현재 프레임의 입 내부 길이를 포함하는 `mouth_bound` 정보도 `talking_dict`에 추가됩니다.
            *   현재 프레임의 `img_id` 또한 `talking_dict`에 다시 저장됩니다.

        6.  **시야(Field of View, FoV) 계산**:
            *   JSON 파일에서 읽어온 초점 거리(`focal_len`)와 이미지 너비/높이(`w, h`)를 사용하여 수평 시야(`FovX`)와 수직 시야(`FovY`)를 계산합니다.

        7.  **`CameraInfo` 객체 생성 및 저장**:
            *   위에서 수집된 모든 정보(카메라 ID, 회전 행렬, 변환 벡터, 시야 정보, 이미지 데이터, 경로, 이름, 크기, 배경, `talking_dict` 등)를 하나의 `CameraInfo` NamedTuple 객체로 묶습니다.
            *   이 `CameraInfo` 객체는 `cam_infos` 리스트에 추가됩니다.
            *   (선택적으로) `idx`가 5000을 초과하면 루프를 조기 종료하는 조건이 있습니다.

        8.  **결과 반환**: 모든 프레임 처리가 완료되면, 함수는 생성된 `CameraInfo` 객체들의 리스트인 `cam_infos`를 반환합니다.

        전반적으로 이 코드는 3D 가우시안 스플래팅과 같은 기술을 사용하여 말하는 얼굴을 렌더링하고 애니메이션화하는 데 필요한 모든 관련 입력 데이터(카메라 위치, 이미지, 얼굴 마스크, 오디오 특징, 얼굴 표정)를 프레임별로 체계적으로 준비하는 데 목적이 있습니다.
        '''

        for idx, frame in tqdm(enumerate(frames)):
            # tqdm을 사용해 frames 리스트를 enumerate로 반복하며 진행 상황을 시각적으로 표시함
            cam_name = os.path.join(path, 'gt_imgs', str(frame["img_id"]) + extension)
            # 각 프레임의 이미지 파일 경로를 생성함 (예: path/gt_imgs/0001.jpg)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # 프레임의 transform_matrix를 numpy 배열로 변환함
            # 이 행렬은 카메라 좌표계에서 월드 좌표계로 변환하는 역할을 함

            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1
            # OpenGL/Blender의 축(Y가 위, Z가 뒤)을 COLMAP의 축(Y가 아래, Z가 앞)으로 변환하기 위해
            # 행렬의 1, 2번째 열(즉, Y, Z축)을 -1을 곱해 반전시킴

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            # c2w의 역행렬을 구해 월드-카메라 변환 행렬을 얻음 (카메라는 원점에 고정돼있고, 얼굴이 움직이는 좌표)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            # 회전 행렬(R)을 추출하고, CUDA의 glm 라이브러리 호환을 위해 전치(transpose)함
            T = w2c[:3, 3]
            # 변환 벡터(T)를 추출함

            talking_dict = {} # # 프레임별 얼굴 및 카메라 관련 정보를 담을 딕셔너리 생성
            talking_dict['img_id'] = frame['img_id'] # 현재 프레임의 이미지 id 저장 (0, 1, 2, ...)
            
            image_path = os.path.join(path, cam_name)
            # 이미지 파일의 전체 경로 생성 (예: path/gt_imgs/0001.jpg)
            image_name = Path(cam_name).stem
            # 이미지 파일명(확장자 제외) 추출 (예: 0001)
            talking_dict['image_path'] = image_path
            # 이미지 경로를 딕셔너리에 저장 (예: path/gt_imgs/0001.jpg)
            if preload or idx==0:
                # preload가 True이거나 첫 프레임이면 이미지를 미리 로드함
                image = Image.open(image_path)
                # 이미지를 열고
                w, h = image.size[0], image.size[1] # 512, 512
                # 이미지의 너비(w), 높이(h) 추출
                image = np.array(image.convert("RGB"))
                # 이미지를 RGB로 변환 후 numpy 배열로 변환

            bg = None
            # 배경 정보는 사용하지 않으므로 None으로 설정

            teeth_mask_path = os.path.join(path, 'teeth_mask', str(frame['img_id']) + '.npy')
            # 치아 마스크 파일 경로 생성 (예: path/teeth_mask/0001.npy)
            mask_path = os.path.join(path, 'parsing', str(frame['img_id']) + '.png')
            # 세그멘테이션 마스크 파일 경로 생성 (예: path/parsing/0001.png)
            talking_dict['teeth_mask_path'] = teeth_mask_path
            # 치아 마스크 경로 저장
            talking_dict['mask_path'] = mask_path
            # 파싱 마스크 경로 저장
            if preload:
                # preload가 True이면 마스크 파일들도 미리 로드함 (디폴트가 True)
                teeth_mask = np.load(teeth_mask_path)
                # 치아 마스크를 numpy 배열로 로드
                mask = np.array(Image.open(mask_path).convert("RGB")) * 1.0
                # 파싱 마스크를 RGB로 변환 후 numpy 배열로 로드 (곱하기 1.0은 float 변환)
                talking_dict['face_mask'] = (mask[:, :, 2] > 254) * (mask[:, :, 0] == 0) * (mask[:, :, 1] == 0) ^ teeth_mask
                # 얼굴 마스크: 파란색 채널이 254보다 크고, 빨강/초록 채널이 0인 픽셀
                # 치아 마스크와 XOR 연산
                # 얼굴 영역(파란색)에 포함되지만 치아 마스크에는 없는 픽셀: face_mask에 포함됩니다.
                # 치아 마스크에 포함되지만 얼굴 영역(파란색)에는 없는 픽셀: face_mask에 포함됩니다.
                # 두 영역 모두에 포함되거나, 두 영역 모두에 포함되지 않는 픽셀: face_mask에서 제외됩니다.
                # face_mask는 '치아를 제외한 얼굴 피부'와 '치아'를 합친 형태가 되어 전체 얼굴 영역을 나타내게 될 것
                talking_dict['hair_mask'] = (mask[:, :, 0] < 1) * (mask[:, :, 1] < 1) * (mask[:, :, 2] < 1)
                # 머리카락 마스크: 모든 채널이 1보다 작은(즉, 검정색) 픽셀
                talking_dict['mouth_mask'] = (mask[:, :, 0] == 100) * (mask[:, :, 1] == 100) * (mask[:, :, 2] == 100) + teeth_mask
                # 입 마스크: 모든 채널이 100인(회색) 픽셀에 치아 마스크를 더함
                # 입 안쪽 영역(회색)에 포함되거나, 치아 마스크에 포함되는 모든 픽셀은 mouth_mask에 포함
            
            if audio_file == '':
                # 오디오 파일이 지정되지 않은 경우 (디폴트)
                talking_dict['auds'] = get_audio_features(auds, 2, frame['img_id'])
                # 전체 오디오 특징(auds)에서 현재 프레임의 img_id에 해당하는 특징 추출
                if frame['img_id'] > auds.shape[0]:
                    # 만약 img_id가 오디오 특징 배열 길이보다 크면
                    print("[warnining] audio feature is too short")
                    # 경고 메시지 출력
                    break
                    # 반복문 종료
            else:
                # 오디오 파일이 지정된 경우
                talking_dict['auds'] = get_audio_features(auds, 2, idx)
                # 전체 오디오 특징(auds)에서 현재 인덱스(idx)에 해당하는 특징 추출
                if idx >= auds.shape[0]:
                    # 인덱스가 오디오 특징 배열 길이 이상이면
                    break
                    # 반복문 종료


            talking_dict['blink'] = torch.as_tensor(np.clip(au_blink[frame['img_id']], 0, 2) / 2)
            # 눈 깜빡임(au_blink) 값을 0~2로 클리핑 후 2로 나눠 정규화, torch tensor로 변환
            talking_dict['au25'] = [au25[frame['img_id']], au25_25, au25_50, au25_75, au25_100]
            # 입술 벌림(au25) 값과 미리 계산된 백분위수 값들을 리스트로 저장

            talking_dict['au_exp'] = torch.as_tensor(au_exp[frame['img_id']])
            # 선택된 액션 유닛 집합(au_exp) 값을 torch tensor로 저장


            [xmin, xmax, ymin, ymax] = ldmks_lips[idx].tolist()
            # 입술 랜드마크의 경계 상자 좌표를 추출
            # padding to H == W
            cx = (xmin + xmax) // 2
            # 중심 x좌표 계산
            cy = (ymin + ymax) // 2
            # 중심 y좌표 계산

            l = max(xmax - xmin, ymax - ymin) // 2
            # 입술 경계 상자를 정사각형으로 만들기 위해 새로운 한 변의 길이(l)를 결정
            # 이 l은 원래 경계 상자의 가로 길이(xmax - xmin)와 세로 길이(ymax - ymin) 중 더 큰 값의 절반으로 설정
            # 입술 영역을 충분히 포함하면서도 정사각형 형태를 유지할 수 있는 최소한의 크기
            xmin = cx - l
            xmax = cx + l
            ymin = cy - l
            ymax = cy + l
            # 계산된 중심 cx, cy와 새로운 한 변의 절반 길이 l을 사용하여 xmin, xmax, ymin, ymax를 다시 계산합니다. 이 과정을 통해 원래 입술 영역의 중심을 기준으로 가로와 세로 길이가 같은 정사각형 형태의 새로운 경계 상자가 생성됩니다.
            talking_dict['lips_rect'] = [xmin, xmax, ymin, ymax]
            # 최종적으로 패딩된 입술 경계 상자 [xmin, xmax, ymin, ymax]는 talking_dict['lips_rect']에 저장됩니다.
            # 이 과정은 주로 딥러닝 모델이 입력으로 정사각형 이미지를 선호하거나, 입술 주변의 일정하고 충분한 컨텍스트(context) 영역을 제공하기 위해 수행됩니다.
            # 예를 들어, 입술 움직임을 분석하는 모델에 입력할 때 일관된 크기의 영역을 제공하여 모델 학습의 안정성을 높이는 데 활용될 수 있습니다.
            
            talking_dict['lhalf_rect'] = ldmks_lhalf[idx]
            # 얼굴 하단 경계 상자 좌표 저장
            talking_dict['mouth_bound'] = [mouth_lb, mouth_ub, ldmks_mouth[idx, 1] - ldmks_mouth[idx, 0]]
            # 입 내부 영역의 최소/최대 길이와 현재 프레임의 입 내부 길이 저장
            talking_dict['img_id'] = frame['img_id']
            # 이미지 id 다시 저장 (중복이지만 일관성 유지)

            FovX = focal2fov(focal_len, w)
            # 초점 거리(focal_len)와 이미지 너비(w)를 이용해 수평 시야각(FovX) 계산
            FovY = focal2fov(focal_len, h)
            # 초점 거리와 이미지 높이(h)로 수직 시야각(FovY) 계산
            
            # if idx > 5000: break # 인덱스가 5000을 넘으면 반복문 조기 종료 (데이터가 너무 많을 때 방지) ######################################
            
            if idx > 500: break
            # (주석 처리된 코드: 디버깅 용으로 500개까지만 처리할 때 사용)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=w, height=h, background=None, talking_dict=talking_dict))
            # 지금까지 수집한 모든 정보를 CameraInfo 객체로 묶어 cam_infos 리스트에 추가
            
    '''
    ### `talking_dict`에 포함되는 정보들

    1.  `img_id`
        *   예시: `0`
        *   차원: 스칼라 (단일 정수)
        *   설명: 현재 처리 중인 이미지 프레임의 고유 식별 번호입니다. 데이터셋의 순서에 따라 0부터 시작하는 정수 값을 가집니다.

    2.  `image_path`
        *   예시: `/home/white/github/InsTaG/data/pretrain/macron/gt_imgs/0000.jpg`
        *   차원: 문자열
        *   설명: 현재 프레임에 해당하는 원본 이미지 파일의 절대 경로입니다.

    3.  `teeth_mask_path`
        *   예시: `/home/white/github/InsTaG/data/pretrain/macron/teeth_mask/0000.npy`
        *   차원: 문자열
        *   설명: 현재 프레임에 해당하는 치아 마스크 파일(`.npy` 형식)의 절대 경로입니다.

    4.  `mask_path`
        *   예시: `/home/white/github/InsTaG/data/pretrain/macron/parsing/0000.png`
        *   차원: 문자열
        *   설명: 현재 프레임에 해당하는 얼굴 segmentation 마스크 파일(`.png` 형식)의 절대 경로입니다.

    5.  `face_mask`
        *   예시: `numpy.ndarray` (값이 `True` 또는 `False`인 부울 배열)
        *   차원: H X W (이미지 높이 x 이미지 너비)
        *   설명: 얼굴 영역을 나타내는 이진 마스크입니다. 픽셀 값이 `True`이면 얼굴 영역이고, `False`이면 배경 또는 다른 영역입니다.

    6.  `hair_mask`
        *   예시: `numpy.ndarray` (값이 `True` 또는 `False`인 부울 배열)
        *   차원: H X W
        *   설명: 머리카락 영역을 나타내는 이진 마스크입니다.

    7.  `mouth_mask`
        *   예시: `numpy.ndarray` (값이 `True` 또는 `False`인 부울 배열)
        *   차원: H X W
        *   설명: 입 안쪽 공간과 치아 영역을 모두 포함하는 이진 마스크입니다.

    8.  `auds`
        *   예시: `torch.Tensor` (1, 29, 16)
        *   차원: 1 X {오디오 특징 차원} X {오디오 채널 수} (예: 1 X 29 X 16)
        *   설명: 현재 프레임에 해당하는 오디오 특징(예: mel-spectrogram 또는 임베딩 벡터)을 담고 있는 PyTorch 텐서입니다.

    9.  `blink`
        *   예시: `torch.Tensor` (스칼라 값, 0.5)
        *   차원: 스칼라 (단일 float)
        *   설명: 현재 프레임의 눈 깜빡임 강도 값을 0~1 사이로 정규화한 PyTorch 텐서입니다. 0은 눈을 뜨고 있는 상태, 1은 완전히 감은 상태를 나타낼 수 있습니다.

    10. `au25`
        *   예시: `[torch.Tensor(0.8), 0.1, 0.3, 0.6, 1.2]` (리스트)
        *   차원: 리스트 (첫 번째 요소는 스칼라 텐서, 나머지 4개는 스칼라 float)
        *   설명: 현재 프레임의 입술 벌림(AU25\_r) 강도 값과 함께, 전체 데이터셋에서 계산된 AU25\_r 값의 25, 50, 75, 100(최대) 백분위수 값을 포함하는 리스트입니다.

    11. `au_exp`
        *   예시: `torch.Tensor` (6,)
        *   차원: 선택된 AU 개수 (예: 6)
        *   설명: 현재 프레임에 해당하는 선택된 얼굴 액션 유닛(AU1, AU4, AU5, AU6, AU7, AU45)들의 강도 값을 담고 있는 PyTorch 텐서입니다.

    12. `lips_rect`
        *   예시: `[100, 150, 120, 170]` (리스트)
        *   차원: 4 (정수 리스트)
        *   설명: 정사각형으로 패딩된 입술 영역의 경계 상자 좌표 `[xmin, xmax, ymin, ymax]`입니다.

    13. `lhalf_rect`
        *   예시: `[80, 200, 110, 240]` (리스트)
        *   차원: 4 (정수 리스트)
        *   설명: 얼굴 하단(lower half) 영역의 경계 상자 좌표 `[xmin, xmax, ymin, ymax]`입니다.

    14. `mouth_bound`
        *   예시: `[0.1, 0.8, 0.4]` (리스트)
        *   차원: 3 (float 리스트)
        *   설명: 입 내부 영역의 최소 벌림 길이(`mouth_lb`), 최대 벌림 길이(`mouth_ub`), 그리고 현재 프레임의 입 내부 벌림 길이(`ldmks_mouth[idx, 1] - ldmks_mouth[idx, 0]`)를 포함하는 리스트입니다.
    '''

    return cam_infos
    # 모든 프레임 처리 후 CameraInfo 객체 리스트 반환

'''
readNerfSyntheticInfo는 데이터 전처리 과정 중 다음을 담당한다.
1. train, test용 데이터셋에서, 프레임별 카메라 정보, 액션 유닛, 오디오 정보 등을 읽어옴
2. train 카메라 궤적들의 평균 좌표와 이로부터의 이동 반경의 최대 반지름을 계산
3. 랜덤 포인트 클라우드를 생성
4. 랜덤 포인트 클라우드, train 정보, test 정보, 포인트 클라우드 경로를 하나의 SceneInfo 객체로 묶어 반환
'''
def readNerfSyntheticInfo(path, white_background, eval, extension=".jpg", args=None, preload=True):
    # readNerfSyntheticInfo 함수는 NeRF synthetic(Blender) 데이터셋을 읽어와서 씬 정보를 구성하는 함수입니다.
    # path: '/home/white/github/InsTaG/data/pretrain/macron'
    # white_background: False
    # eval: False
    # extension: '.jpg'
    # args: _dataset = {'sh_degree': 2, 'source_path': '/home/white/github/InsTaG/data/pretrain/macron', 'model_path': 'debug_01/macron', 'images': 'images', 'resolution': -1, 'white_background': False, 
    #                   'data_device': 'cpu', 'eval': False, 'audio': '', 'init_num': 2000, 'N_views': -1, 'audio_extractor': 'deepspeech', 'type': 'face', 'preload': True, 'all_for_train': False}
    # preload: True

    audio_file = args.audio    # args 객체에서 오디오 파일 경로를 가져옵니다. 디폴트: ''
    audio_extractor = args.audio_extractor    # args 객체에서 오디오 추출기 종류를 가져옵니다. 예: 'deepspeech'

    if not eval:
        # eval이 False(즉, 학습 모드)이면 학습용 카메라 정보를 읽어옵니다. 디폴트로 False.
        print("Reading Training Transforms")
        # 학습용 transforms 파일을 읽는다는 메시지를 출력합니다.
        # 데이터 전처리 과정에서 매 프레임별 카메라, 액션 유닛, 오디오 정보 등을 읽어오는 과정으로, 시간이 많이 걸림.
        train_cam_infos = readCamerasFromTransforms(
            path, "transforms_train.json", white_background, extension, audio_file, audio_extractor, preload=preload
        )
        # readCamerasFromTransforms 함수를 이용해 학습용 카메라 정보를 읽어와 train_cam_infos에 저장합니다.
    print("Reading Test Transforms")
    # 테스트용 transforms 파일을 읽는다는 메시지를 출력합니다.
    test_cam_infos = readCamerasFromTransforms(
        path, "transforms_val.json", white_background, extension, audio_file, audio_extractor, preload=preload
    )
    # readCamerasFromTransforms 함수를 이용해 테스트용 카메라 정보를 읽어와 test_cam_infos에 저장합니다.
    
    # if not eval:
    #     train_cam_infos.extend(test_cam_infos)
    #     test_cam_infos = []
    # (주석 처리된 코드: eval이 False일 때 train에 test를 합치고 test를 비우는 코드. 현재는 사용하지 않음)
    
    if eval:
        # eval이 True(즉, 평가 모드)이면 train_cam_infos를 test_cam_infos로 대체합니다.
        train_cam_infos = test_cam_infos
        # 평가 모드에서는 학습 카메라와 테스트 카메라를 동일하게 사용합니다.

    # train 카메라 궤적들의 평균 좌표와 이로부터의 이동 반경의 최대 반지름을 계산
    nerf_normalization = getNerfppNorm(train_cam_infos)
    # getNerfppNorm 함수를 이용해 카메라 정규화 파라미터(카메라 중심의 평균과 이로부터의 이동 반경의 최대 반지름)를 계산합니다.

    # 나중에 가우시안 초기화의 중심점이 될 랜덤 포인트 클라우드 생성
    ply_path = os.path.join(path, "points3d.ply")
    # 포인트 클라우드 파일(.ply)의 경로를 만듭니다. 예: '/macron/points3d.ply'
    if not os.path.exists(ply_path) or True:
        # ply_path가 존재하지 않거나(항상 True이므로 무조건 실행됨)
        # Blender 데이터셋에는 colmap 포인트 클라우드가 없으므로 무작위 포인트 클라우드를 생성해야 하기 때문.
        num_pts = args.init_num # 2000
        # 생성할 포인트 개수를 args에서 가져옵니다. 예: 2000
        print(f"Generating random point cloud ({num_pts})...")
        # 무작위 포인트 클라우드를 생성한다는 메시지를 출력합니다.
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 0.2 - 0.1
        # (num_pts, 3) 크기의 무작위 xyz 좌표를 생성합니다. 각 좌표는 -0.1 ~ 0.1 범위에 분포합니다.
        shs = np.random.random((num_pts, 3)) / 255.0
        # (num_pts, 3) 크기의 무작위 SH 색상값을 생성합니다. 0~1/255 범위의 값입니다.
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        # BasicPointCloud 객체를 생성합니다. 좌표는 xyz, 색상은 SH2RGB(shs)로 변환, 노멀은 0으로 초기화합니다.

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
        # 생성한 포인트 클라우드를 ply 파일로 저장합니다. 색상값은 0~255 범위로 변환하여 저장합니다.
    try:
        pcd = fetchPly(ply_path)
        # ply_path에서 포인트 클라우드를 읽어와 pcd에 저장합니다.
    except:
        pcd = None
        # 만약 ply 파일을 읽는 데 실패하면 pcd를 None으로 설정합니다.

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path
    )
    # SceneInfo 객체를 생성하여 씬의 모든 정보를 하나로 묶습니다.
    return scene_info
    # 생성된 SceneInfo 객체를 반환합니다.

sceneLoadTypeCallbacks = {
    "Colmap": None,
    "Blender" : readNerfSyntheticInfo
} # 우리가 사용할 데이터는 Blender 데이터셋이므로 위의 함수를 호출
