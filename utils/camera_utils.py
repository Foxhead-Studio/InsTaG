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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    # loadCam 함수는 cam_info(카메라 정보 객체)로부터 Camera 객체를 생성하는 함수입니다.
    # args: 데이터셋 및 환경 설정 객체, id: 프레임 인덱스, cam_info: CameraInfo 객체, resolution_scale: 해상도 스케일 예: 1.0

    image_rgb = PILtoTorch(cam_info.image).type("torch.ByteTensor")
    # cam_info.image(PIL 이미지)를 torch 텐서로 변환하고, 타입을 torch.ByteTensor로 지정합니다. '/home/white/github/InsTaG/data/pretrain/macron/gt_imgs/0000.jpg'로부터 생성된 image를 텐서로 변경.
    # 예시: (3, 512, 512) 크기의 RGB 이미지 텐서가 생성됩니다.

    if cam_info.background is not None:
        # cam_info에 배경 이미지가 존재하는 경우
        background = PILtoTorch(cam_info.background)[:3, ...].type("torch.ByteTensor")
        # 배경 이미지를 torch 텐서로 변환한 뒤, RGB 채널만 추출하고 ByteTensor로 변환합니다.
        # 예시: (3, 512, 512) 크기의 배경 텐서
    else: # 디폴트 None
        background = None
        # 배경 이미지가 없으면 None으로 설정합니다. 디폴트.

    gt_image = image_rgb[:3, ...]
    # image_rgb에서 RGB 채널만 추출하여 gt_image로 저장합니다.
    # 예시: (3, 512, 512) 크기의 텐서

    loaded_mask = None
    # 알파 마스크(불투명도 마스크)는 현재 None으로 설정되어 있습니다.

    return Camera(
        colmap_id=cam_info.uid,  # colmap_id: 카메라의 고유 ID(cam_info.uid)
        R=cam_info.R,            # R: W2C의 카메라의 회전 행렬(3x3)
        T=cam_info.T,            # T: W2C의 카메라의 위치 벡터(3,)
        FoVx=cam_info.FovX,      # FoVx: x축 시야각(도 단위)
        FoVy=cam_info.FovY,      # FoVy: y축 시야각(도 단위) (정사각형 이미지이므로 두 값은 동일) 0.42
        image=gt_image,          # image: torch 텐서로 변환된 RGB 이미지
        gt_alpha_mask=loaded_mask, # gt_alpha_mask: 알파 마스크(현재 None)
        background=background,   # background: torch 텐서로 변환된 배경 이미지 또는 None (디폴트 None)
        talking_dict=cam_info.talking_dict, # talking_dict: 프레임별 부가 정보(입, 눈, 오디오 등)
        image_name=cam_info.image_name,     # image_name: 이미지 파일 이름(예: '8431')
        uid=id,                  # uid: cameraList_from_camInfos에서 부여하는 인덱스(0, 1, 2, ...), colmap_id, cam_info.uid, id, uid 모두 동일하게 0, 1, 2, ...
        data_device=args.data_device # data_device: 데이터가 저장될 디바이스 정보('cpu')
    )
    # Camera 객체를 생성하여 반환합니다.

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    # cameraList_from_camInfos 함수는 cam_infos 리스트(여러 프레임의 CameraInfo 객체)로부터
    # Camera 객체 리스트를 생성하는 함수입니다.
    # 예시 호출: cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
    # cam_infos: CameraInfo 객체들의 리스트(예: train_cameras)
    # resolution_scale: 해상도 스케일(예: 1.0)
    # args: 데이터셋 및 환경 설정 객체

    camera_list = []
    # 결과로 반환할 Camera 객체 리스트를 초기화합니다.

    for id, c in enumerate(cam_infos):
        # cam_infos 리스트의 각 CameraInfo 객체 c에 대해 반복합니다.
        # id는 0부터 시작하는 인덱스입니다.
        camera_list.append(loadCam(args, id, c, resolution_scale))
        # loadCam 함수를 호출하여 Camera 객체를 생성하고, camera_list에 추가합니다.
        # 예시: id=0, c=첫 번째 CameraInfo 객체

    return camera_list
    # 완성된 Camera 객체 리스트를 반환합니다.
    # 예시: [Camera1, Camera2, Camera3, ...]

def camera_to_JSON(id, camera : Camera):
    # camera_to_JSON 함수는 Camera 객체를 JSON(딕셔너리) 형식으로 변환합니다.
    Rt = np.zeros((4, 4))
    # 4x4 크기의 0으로 채워진 행렬을 생성합니다.
    Rt[:3, :3] = camera.R.transpose()
    # Rt의 좌상단 3x3 부분에 카메라의 회전 행렬의 전치(transpose)를 할당합니다.
    Rt[:3, 3] = camera.T
    # Rt의 마지막 열(위 3개)에 카메라의 위치 벡터를 할당합니다.
    Rt[3, 3] = 1.0
    # 동차 좌표계(homogeneous coordinates)를 위해 마지막 원소를 1로 설정합니다.

    W2C = np.linalg.inv(Rt)
    # Rt의 역행렬을 계산하여 W2C(월드->카메라 변환 행렬)를 만듭니다.
    pos = W2C[:3, 3]
    # W2C의 마지막 열(위 3개)은 카메라의 월드 좌표계 위치입니다.
    rot = W2C[:3, :3]
    # W2C의 좌상단 3x3은 카메라의 회전 행렬입니다.
    serializable_array_2d = [x.tolist() for x in rot]
    # rot(3x3 numpy 배열)을 리스트로 변환하여 JSON 직렬화가 가능하도록 합니다.

    camera_entry = {
        'id' : id,  # 카메라 인덱스
        'img_name' : camera.image_name,  # 이미지 파일 이름
        'width' : camera.width,          # 이미지 너비(픽셀)
        'height' : camera.height,        # 이미지 높이(픽셀)
        'position': pos.tolist(),        # 카메라 위치(리스트)
        'rotation': serializable_array_2d, # 카메라 회전 행렬(리스트)
        'fy' : fov2focal(camera.FovY, camera.height), # y축 초점거리(픽셀 단위)
        'fx' : fov2focal(camera.FovX, camera.width)   # x축 초점거리(픽셀 단위)
    }
    # 카메라 정보를 JSON(딕셔너리) 형식으로 정리합니다.

    return camera_entry
    # 완성된 카메라 JSON 정보를 반환합니다.
