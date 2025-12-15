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

import torch
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    # 4x4 크기의 0으로 채워진 행렬을 생성합니다.
    Rt[:3, :3] = R.transpose()
    # Rt의 좌상단 3x3 부분에 R의 전치행렬을 할당합니다.
    # R은 3x3 회전행렬이며, 전치(transpose)를 취하는 이유는
    # 월드->카메라 변환에서 회전 방향이 반대이기 때문입니다.
    Rt[:3, 3] = t
    # Rt의 마지막 열(위에서 3개)에 t(이동 벡터)를 할당합니다.
    # 즉, Rt는 [R^T | t] 형태가 됩니다.
    Rt[3, 3] = 1.0
    # 동차좌표(homogeneous coordinates)를 위해 마지막 원소를 1로 설정합니다.

    C2W = np.linalg.inv(Rt)
    # Rt의 역행렬을 구해 C2W(Camera-to-World) 행렬을 만듭니다.
    # 즉, 카메라 좌표계에서 월드 좌표계로 변환하는 행렬입니다.

    cam_center = C2W[:3, 3]
    # C2W의 마지막 열(위에서 3개)은 카메라의 월드 좌표계상 중심 위치입니다.
    # 예시: cam_center = [x, y, z]

    cam_center = (cam_center + translate) * scale
    # 카메라 중심에 translate(이동 벡터)를 더하고, scale(스케일)을 곱합니다.
    # 예시: translate = [1, 2, 3], scale = 0.5라면
    # cam_center = ([x, y, z] + [1, 2, 3]) * 0.5

    C2W[:3, 3] = cam_center
    # 변환된 카메라 중심 좌표를 다시 C2W 행렬의 마지막 열에 넣어줍니다.

    Rt = np.linalg.inv(C2W)
    # 다시 C2W의 역행렬을 구해 Rt(World-to-Camera) 행렬을 만듭니다.
    # 즉, 월드 좌표계에서 카메라 좌표계로 변환하는 최종 행렬입니다.

    return np.float32(Rt)
    # Rt를 float32 타입으로 변환하여 반환합니다.

#  3D 공간의 점들을 2D 이미지 평면으로 투영하기 위한 투영 행렬(Projection Matrix)을 생성하는 함수
#  3D 공간의 가우시안 포인트들을 카메라 시점에서 2D 이미지로 렌더링할 때 사용
def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2)) # 수직 시야각의 절반 탄젠트
    tanHalfFovX = math.tan((fovX / 2)) # 수평 시야각의 절반 탄젠트

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    # focal: 초점 거리(예: 1200)
    # pixels: 이미지의 가로(또는 세로) 픽셀 수(예: 512)
    # 
    # 이 함수는 카메라의 초점 거리(focal)와 이미지의 픽셀 수(pixels)를 이용해
    # 시야각(Field of View, FOV)을 라디안 단위로 계산합니다.
    #
    # 수식: $FOV = 2 \cdot \arctan\left(\frac{pixels}{2 \cdot focal}\right)$
    # 
    # 예시: focal = 1200, pixels = 512일 때
    # $$
    # FOV = 2 \cdot \arctan\left(\frac{512}{2 \times 1200}\right)
    #     = 2 \cdot \arctan\left(\frac{512}{2400}\right)
    #     = 2 \cdot \arctan(0.2133...)
    #     \approx 2 \cdot 0.2102
    #     \approx 0.4204 \text{ (라디안)}
    # $$
    # 
    # 왜 이렇게 계산하는가?
    # - 카메라의 투영 원리를 생각하면, 이미지 평면의 한쪽 끝에서 반대쪽 끝까지의 각도를 구하는 것이 FOV입니다.
    # - 이미지 센서의 크기를 pixels로 보고, 초점 거리(focal)로 삼각함수를 적용하면
    #   $\tan(\theta/2) = \frac{pixels/2}{focal}$이 됩니다.
    # - 양변에 arctan을 취하고 2를 곱하면 전체 FOV가 나옵니다.
    return 2*math.atan(pixels/(2*focal))