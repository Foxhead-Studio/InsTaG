import os  # OS 관련 기능을 사용하기 위한 모듈 import
import sys  # 시스템 관련 기능을 사용하기 위한 모듈 import
import cv2  # OpenCV 라이브러리 import (이미지 처리용)
import argparse  # 명령행 인자 파싱을 위한 모듈 import
from pathlib import Path  # 경로 처리를 위한 Path 객체 import
import torch  # PyTorch 딥러닝 프레임워크 import
import numpy as np  # 수치 연산을 위한 numpy import
from data_loader import load_dir  # 데이터 로딩 함수 import
from facemodel import Face_3DMM  # 3DMM 얼굴 모델 클래스 import
from util import *  # 유틸리티 함수들 import
from render_3dmm import Render_3DMM  # 3DMM 렌더링 함수 import

# torch.autograd.set_detect_anomaly(True)  # (주석처리) autograd의 이상 탐지 기능 활성화

dir_path = os.path.dirname(os.path.realpath(__file__))  # 현재 파일의 디렉토리 경로를 얻음

def set_requires_grad(tensor_list):
    # 입력된 텐서 리스트의 requires_grad 속성을 True로 설정 (학습 가능하게 만듦)
    for tensor in tensor_list:
        tensor.requires_grad = True

# 명령행 인자 파서 생성
parser = argparse.ArgumentParser()
# --path 인자: 타겟 인물의 이미지 폴더 경로 지정 (기본값: obama/ori_imgs)
parser.add_argument(
    "--path", type=str, default="obama/ori_imgs", help="idname of target person"
)
# --img_h 인자: 이미지 높이 지정 (기본값: 512)
parser.add_argument("--img_h", type=int, default=512, help="image height")
# --img_w 인자: 이미지 너비 지정 (기본값: 512)
parser.add_argument("--img_w", type=int, default=512, help="image width")
# --frame_num 인자: 사용할 이미지(프레임) 개수 지정 (기본값: 11000)
parser.add_argument("--frame_num", type=int, default=11000, help="image number")
# 인자 파싱
args = parser.parse_args()

start_id = 0  # 시작 프레임 인덱스 (0부터 시작)
end_id = args.frame_num  # 종료 프레임 인덱스 (frame_num까지)

# 이미지 폴더에서 랜드마크(lms)와 이미지 경로(img_paths) 불러오기
lms, img_paths = load_dir(args.path, start_id, end_id)
num_frames = lms.shape[0]  # 전체 프레임(이미지) 개수
h, w = args.img_h, args.img_w  # 이미지 높이, 너비
# 이미지 중심 좌표 (cx, cy) 계산 후 CUDA 텐서로 변환
cxy = torch.tensor((w / 2.0, h / 2.0), dtype=torch.float).cuda()
# 3DMM 파라미터 차원 및 포인트 개수 설정
id_dim, exp_dim, tex_dim, point_num = 100, 79, 100, 34650
# Face_3DMM 모델 객체 생성 (3DMM 폴더 경로와 파라미터 전달)
model_3dmm = Face_3DMM(
    os.path.join(dir_path, "3DMM"), id_dim, exp_dim, tex_dim, point_num
)

# 초점거리(focal length) 추정을 위해 40프레임마다 하나씩 샘플링
sel_ids = np.arange(0, num_frames, 40)
sel_num = sel_ids.shape[0]  # 샘플링된 프레임 개수
arg_focal = 1600  # 기본 초점거리 값 (사용 안함, 참고용)
arg_landis = 1e5  # 랜드마크 거리 손실 기본값 (사용 안함, 참고용)

print(f'[INFO] fitting focal length...')  # 초점거리 피팅 시작 로그 출력

# 초점거리(focal length)를 여러 값(600~1400, 100 간격)으로 바꿔가며 최적의 값을 찾는 반복문
for focal in range(600, 1500, 100):  # 예: focal=600, 700, ..., 1400
    # id_para: identity(고유 얼굴 형태) 파라미터, shape=(1, id_dim), 학습 가능하게 초기화
    id_para = lms.new_zeros((1, id_dim), requires_grad=True)  # 예: [0, 0, ..., 0] (100차원)
    # exp_para: expression(표정) 파라미터, shape=(sel_num, exp_dim), 학습 가능하게 초기화
    exp_para = lms.new_zeros((sel_num, exp_dim), requires_grad=True)  # 예: [sel_num, 79]
    # euler_angle: 각 프레임의 3D 회전(오일러 각), shape=(sel_num, 3), 학습 가능하게 초기화
    euler_angle = lms.new_zeros((sel_num, 3), requires_grad=True)  # 예: [sel_num, 3]
    # trans: 각 프레임의 3D 이동, shape=(sel_num, 3), 학습 가능하게 초기화
    trans = lms.new_zeros((sel_num, 3), requires_grad=True)  # 예: [sel_num, 3]
    # z축(깊이) 이동값을 -7만큼 이동시킴(얼굴이 카메라에서 약간 떨어지게)
    trans.data[:, 2] -= 7  # 예: [x, y, z-7]
    # focal_length: 현재 focal 값으로 초기화, requires_grad=False(고정)
    focal_length = lms.new_zeros(1, requires_grad=False)  # 예: [0.]
    focal_length.data += focal  # 예: [600], [700], ...
    # requires_grad=True로 명시적으로 설정(혹시라도 누락된 텐서가 있으면)
    set_requires_grad([id_para, exp_para, euler_angle, trans])

    # id_para, exp_para(고유+표정) 파라미터용 Adam 옵티마이저, 학습률 0.1
    optimizer_idexp = torch.optim.Adam([id_para, exp_para], lr=0.1)
    # euler_angle, trans(자세+이동) 파라미터용 Adam 옵티마이저, 학습률 0.1
    optimizer_frame = torch.optim.Adam([euler_angle, trans], lr=0.1)

    # 1단계: 자세(회전, 이동)만 먼저 2000번 최적화
    for iter in range(2000):
        # id_para를 sel_num개로 복제(프레임별로 동일한 id 사용)
        id_para_batch = id_para.expand(sel_num, -1)  # 예: [sel_num, id_dim]
        # 3DMM 모델로 3D 랜드마크 좌표 생성
        geometry = model_3dmm.get_3dlandmarks(
            id_para_batch, exp_para, euler_angle, trans, focal_length, cxy
        )  # 예: [sel_num, 랜드마크개수, 3]
        # 3D 랜드마크를 2D 이미지 평면으로 투영
        proj_geo = forward_transform(geometry, euler_angle, trans, focal_length, cxy)  # 예: [sel_num, 랜드마크개수, 3]
        # 예측 랜드마크와 GT 랜드마크(lms[sel_ids])의 거리(손실) 계산
        loss_lan = cal_lan_loss(proj_geo[:, :, :2], lms[sel_ids].detach())  # 예: 평균 L2 거리
        loss = loss_lan  # 이 단계에서는 랜드마크 손실만 사용
        optimizer_frame.zero_grad()  # 자세/이동 파라미터의 기울기 초기화
        loss.backward()  # 역전파
        optimizer_frame.step()  # 파라미터 업데이트
        # if iter % 100 == 0:
        #     print(focal, 'pose', iter, loss.item())

    # 2단계: id, exp, 자세, 이동을 모두 2500번 최적화(정규화 항 포함)
    for iter in range(2500):
        id_para_batch = id_para.expand(sel_num, -1)  # [sel_num, id_dim]
        geometry = model_3dmm.get_3dlandmarks(
            id_para_batch, exp_para, euler_angle, trans, focal_length, cxy
        )
        proj_geo = forward_transform(geometry, euler_angle, trans, focal_length, cxy)
        # 랜드마크 손실
        loss_lan = cal_lan_loss(proj_geo[:, :, :2], lms[sel_ids].detach())
        # id, exp 파라미터의 L2 정규화(너무 커지지 않게)
        loss_regid = torch.mean(id_para * id_para)  # 예: id_para의 제곱 평균
        loss_regexp = torch.mean(exp_para * exp_para)  # 예: exp_para의 제곱 평균
        # 전체 손실: 랜드마크 손실 + id 정규화(0.5배) + exp 정규화(0.4배)
        loss = loss_lan + loss_regid * 0.5 + loss_regexp * 0.4
        optimizer_idexp.zero_grad()  # id/exp 파라미터 기울기 초기화
        optimizer_frame.zero_grad()  # 자세/이동 파라미터 기울기 초기화
        loss.backward()  # 역전파
        optimizer_idexp.step()  # id/exp 파라미터 업데이트
        optimizer_frame.step()  # 자세/이동 파라미터 업데이트
        # if iter % 100 == 0:
        #     print(focal, 'poseidexp', iter, loss_lan.item(), loss_regid.item(), loss_regexp.item())

        # 1500, 3000번째에 학습률을 0.2배로 감소(더 미세하게 학습)
        if iter % 1500 == 0 and iter >= 1500:
            for param_group in optimizer_idexp.param_groups:
                param_group["lr"] *= 0.2
            for param_group in optimizer_frame.param_groups:
                param_group["lr"] *= 0.2

    # 현재 focal 값, 최종 랜드마크 손실, 평균 z이동값(깊이)을 출력
    print(focal, loss_lan.item(), torch.mean(trans[:, 2]).item())

    # 만약 현재 focal에서의 손실이 기존 최소값(arg_landis)보다 작으면, focal을 갱신
    if loss_lan.item() < arg_landis:
        arg_landis = loss_lan.item()  # 최소 손실값 갱신
        arg_focal = focal  # 최소 손실을 주는 focal 값 저장

print("[INFO] find best focal:", arg_focal)

print(f'[INFO] coarse fitting...')

# for all frames, do a coarse fitting ???
id_para = lms.new_zeros((1, id_dim), requires_grad=True)
exp_para = lms.new_zeros((num_frames, exp_dim), requires_grad=True)
tex_para = lms.new_zeros(
    (1, tex_dim), requires_grad=True
)  # not optimized in this block ???
euler_angle = lms.new_zeros((num_frames, 3), requires_grad=True)
trans = lms.new_zeros((num_frames, 3), requires_grad=True)
light_para = lms.new_zeros((num_frames, 27), requires_grad=True)
trans.data[:, 2] -= 7 # ???
focal_length = lms.new_zeros(1, requires_grad=True)
focal_length.data += arg_focal

set_requires_grad([id_para, exp_para, tex_para, euler_angle, trans, light_para])

optimizer_idexp = torch.optim.Adam([id_para, exp_para], lr=0.1)
optimizer_frame = torch.optim.Adam([euler_angle, trans], lr=1)

for iter in range(1500):
    id_para_batch = id_para.expand(num_frames, -1)
    geometry = model_3dmm.get_3dlandmarks(
        id_para_batch, exp_para, euler_angle, trans, focal_length, cxy
    )
    proj_geo = forward_transform(geometry, euler_angle, trans, focal_length, cxy)
    loss_lan = cal_lan_loss(proj_geo[:, :, :2], lms.detach())
    loss = loss_lan
    optimizer_frame.zero_grad()
    loss.backward()
    optimizer_frame.step()
    if iter == 1000:
        for param_group in optimizer_frame.param_groups:
            param_group["lr"] = 0.1
    # if iter % 100 == 0:
    #     print('pose', iter, loss.item())

for param_group in optimizer_frame.param_groups:
    param_group["lr"] = 0.1

for iter in range(2000):
    id_para_batch = id_para.expand(num_frames, -1)
    geometry = model_3dmm.get_3dlandmarks(
        id_para_batch, exp_para, euler_angle, trans, focal_length, cxy
    )
    proj_geo = forward_transform(geometry, euler_angle, trans, focal_length, cxy)
    loss_lan = cal_lan_loss(proj_geo[:, :, :2], lms.detach())
    loss_regid = torch.mean(id_para * id_para)
    loss_regexp = torch.mean(exp_para * exp_para)
    loss = loss_lan + loss_regid * 0.5 + loss_regexp * 0.4
    optimizer_idexp.zero_grad()
    optimizer_frame.zero_grad()
    loss.backward()
    optimizer_idexp.step()
    optimizer_frame.step()
    # if iter % 100 == 0:
    #     print('poseidexp', iter, loss_lan.item(), loss_regid.item(), loss_regexp.item())
    if iter % 1000 == 0 and iter >= 1000:
        for param_group in optimizer_idexp.param_groups:
            param_group["lr"] *= 0.2
        for param_group in optimizer_frame.param_groups:
            param_group["lr"] *= 0.2

print(loss_lan.item(), torch.mean(trans[:, 2]).item())

print(f'[INFO] fitting light...')

batch_size = 32

device_default = torch.device("cuda:0")
device_render = torch.device("cuda:0")
renderer = Render_3DMM(arg_focal, h, w, batch_size, device_render)

sel_ids = np.arange(0, num_frames, int(num_frames / batch_size))[:batch_size]
imgs = []
for sel_id in sel_ids:
    imgs.append(cv2.imread(img_paths[sel_id])[:, :, ::-1])
imgs = np.stack(imgs)
sel_imgs = torch.as_tensor(imgs).cuda()
sel_lms = lms[sel_ids]
sel_light = light_para.new_zeros((batch_size, 27), requires_grad=True)
set_requires_grad([sel_light])

optimizer_tl = torch.optim.Adam([tex_para, sel_light], lr=0.1)
optimizer_id_frame = torch.optim.Adam([euler_angle, trans, exp_para, id_para], lr=0.01)

for iter in range(71):
    sel_exp_para, sel_euler, sel_trans = (
        exp_para[sel_ids],
        euler_angle[sel_ids],
        trans[sel_ids],
    )
    sel_id_para = id_para.expand(batch_size, -1)
    geometry = model_3dmm.get_3dlandmarks(
        sel_id_para, sel_exp_para, sel_euler, sel_trans, focal_length, cxy
    )
    proj_geo = forward_transform(geometry, sel_euler, sel_trans, focal_length, cxy)

    loss_lan = cal_lan_loss(proj_geo[:, :, :2], sel_lms.detach())
    loss_regid = torch.mean(id_para * id_para)
    loss_regexp = torch.mean(sel_exp_para * sel_exp_para)

    sel_tex_para = tex_para.expand(batch_size, -1)
    sel_texture = model_3dmm.forward_tex(sel_tex_para)
    geometry = model_3dmm.forward_geo(sel_id_para, sel_exp_para)
    rott_geo = forward_rott(geometry, sel_euler, sel_trans)
    render_imgs = renderer(
        rott_geo.to(device_render),
        sel_texture.to(device_render),
        sel_light.to(device_render),
    )
    render_imgs = render_imgs.to(device_default)

    mask = (render_imgs[:, :, :, 3]).detach() > 0.0
    render_proj = sel_imgs.clone()
    render_proj[mask] = render_imgs[mask][..., :3].byte()
    loss_col = cal_col_loss(render_imgs[:, :, :, :3], sel_imgs.float(), mask)

    if iter > 50:
        loss = loss_col + loss_lan * 0.05 + loss_regid * 1.0 + loss_regexp * 0.8
    else:
        loss = loss_col + loss_lan * 3 + loss_regid * 2.0 + loss_regexp * 1.0

    optimizer_tl.zero_grad()
    optimizer_id_frame.zero_grad()
    loss.backward()

    optimizer_tl.step()
    optimizer_id_frame.step()

    if iter % 50 == 0 and iter > 0:
        for param_group in optimizer_id_frame.param_groups:
            param_group["lr"] *= 0.2
        for param_group in optimizer_tl.param_groups:
            param_group["lr"] *= 0.2
    # print(iter, loss_col.item(), loss_lan.item(), loss_regid.item(), loss_regexp.item())


light_mean = torch.mean(sel_light, 0).unsqueeze(0).repeat(num_frames, 1)
light_para.data = light_mean

exp_para = exp_para.detach()
euler_angle = euler_angle.detach()
trans = trans.detach()
light_para = light_para.detach()

print(f'[INFO] fine frame-wise fitting...')

for i in range(int((num_frames - 1) / batch_size + 1)):

    if (i + 1) * batch_size > num_frames:
        start_n = num_frames - batch_size
        sel_ids = np.arange(num_frames - batch_size, num_frames)
    else:
        start_n = i * batch_size
        sel_ids = np.arange(i * batch_size, i * batch_size + batch_size)

    imgs = []
    for sel_id in sel_ids:
        imgs.append(cv2.imread(img_paths[sel_id])[:, :, ::-1])
    imgs = np.stack(imgs)
    sel_imgs = torch.as_tensor(imgs).cuda()
    sel_lms = lms[sel_ids]

    sel_exp_para = exp_para.new_zeros((batch_size, exp_dim), requires_grad=True)
    sel_exp_para.data = exp_para[sel_ids].clone()
    sel_euler = euler_angle.new_zeros((batch_size, 3), requires_grad=True)
    sel_euler.data = euler_angle[sel_ids].clone()
    sel_trans = trans.new_zeros((batch_size, 3), requires_grad=True)
    sel_trans.data = trans[sel_ids].clone()
    sel_light = light_para.new_zeros((batch_size, 27), requires_grad=True)
    sel_light.data = light_para[sel_ids].clone()

    set_requires_grad([sel_exp_para, sel_euler, sel_trans, sel_light])

    optimizer_cur_batch = torch.optim.Adam(
        [sel_exp_para, sel_euler, sel_trans, sel_light], lr=0.005
    )

    sel_id_para = id_para.expand(batch_size, -1).detach()
    sel_tex_para = tex_para.expand(batch_size, -1).detach()

    pre_num = 5

    if i > 0:
        pre_ids = np.arange(start_n - pre_num, start_n)

    for iter in range(50):
        
        geometry = model_3dmm.get_3dlandmarks(
            sel_id_para, sel_exp_para, sel_euler, sel_trans, focal_length, cxy
        )
        proj_geo = forward_transform(geometry, sel_euler, sel_trans, focal_length, cxy)
        loss_lan = cal_lan_loss(proj_geo[:, :, :2], sel_lms.detach())
        loss_regexp = torch.mean(sel_exp_para * sel_exp_para)

        sel_geometry = model_3dmm.forward_geo(sel_id_para, sel_exp_para)
        sel_texture = model_3dmm.forward_tex(sel_tex_para)
        geometry = model_3dmm.forward_geo(sel_id_para, sel_exp_para)
        rott_geo = forward_rott(geometry, sel_euler, sel_trans)
        render_imgs = renderer(
            rott_geo.to(device_render),
            sel_texture.to(device_render),
            sel_light.to(device_render),
        )
        render_imgs = render_imgs.to(device_default)

        mask = (render_imgs[:, :, :, 3]).detach() > 0.0

        loss_col = cal_col_loss(render_imgs[:, :, :, :3], sel_imgs.float(), mask)

        if i > 0:
            geometry_lap = model_3dmm.forward_geo_sub(
                id_para.expand(batch_size + pre_num, -1).detach(),
                torch.cat((exp_para[pre_ids].detach(), sel_exp_para)),
                model_3dmm.rigid_ids,
            )
            rott_geo_lap = forward_rott(
                geometry_lap,
                torch.cat((euler_angle[pre_ids].detach(), sel_euler)),
                torch.cat((trans[pre_ids].detach(), sel_trans)),
            )
            loss_lap = cal_lap_loss(
                [rott_geo_lap.reshape(rott_geo_lap.shape[0], -1).permute(1, 0)], [1.0]
            )
        else:
            geometry_lap = model_3dmm.forward_geo_sub(
                id_para.expand(batch_size, -1).detach(),
                sel_exp_para,
                model_3dmm.rigid_ids,
            )
            rott_geo_lap = forward_rott(geometry_lap, sel_euler, sel_trans)
            loss_lap = cal_lap_loss(
                [rott_geo_lap.reshape(rott_geo_lap.shape[0], -1).permute(1, 0)], [1.0]
            )


        if iter > 30:
            loss = loss_col * 0.5 + loss_lan * 1.5 + loss_lap * 100000 + loss_regexp * 1.0
        else:
            loss = loss_col * 0.5 + loss_lan * 8 + loss_lap * 100000 + loss_regexp * 1.0

        optimizer_cur_batch.zero_grad()
        loss.backward()
        optimizer_cur_batch.step()

        # if iter % 10 == 0:
        #     print(
        #         i,
        #         iter,
        #         loss_col.item(),
        #         loss_lan.item(),
        #         loss_lap.item(),
        #         loss_regexp.item(),
        #     )

    print(str(i) + " of " + str(int((num_frames - 1) / batch_size + 1)) + " done")

    render_proj = sel_imgs.clone()
    render_proj[mask] = render_imgs[mask][..., :3].byte()

    exp_para[sel_ids] = sel_exp_para.clone()
    euler_angle[sel_ids] = sel_euler.clone()
    trans[sel_ids] = sel_trans.clone()
    light_para[sel_ids] = sel_light.clone()

torch.save(
    {
        "id": id_para.detach().cpu(),
        "exp": exp_para.detach().cpu(),
        "euler": euler_angle.detach().cpu(),
        "trans": trans.detach().cpu(),
        "focal": focal_length.detach().cpu(),
    },
    os.path.join(os.path.dirname(args.path), "track_params.pt"),
)

print("params saved")


"""
`track_params.pt` 파일의 각 파라미터를 구체적인 예시로 설명해드리겠습니다.

## **예시: 100프레임 비디오의 경우**

### **1. `id` (얼굴 ID 파라미터)**
```python
id: torch.tensor([0.1, -0.3, 0.8, ..., 0.2])  # shape: [100]
```
- **의미**: 이 사람의 **고유한 얼굴 형태**를 나타내는 100차원 벡터
- **특징**: **모든 프레임에서 동일** (한 사람의 얼굴이므로)
- **예시**: 
  - `id[0]`: 얼굴의 전체적인 크기
  - `id[1]`: 얼굴의 길이/폭 비율
  - `id[2]`: 코의 높이
  - `id[3]`: 눈의 크기
  - ... (100개 파라미터로 얼굴의 모든 형태적 특징을 표현)

### **2. `exp` (표정 파라미터)**
```python
exp: torch.tensor([
    [0.0, 0.1, 0.0, ..., 0.0],  # 프레임 0: 무표정
    [0.2, 0.3, 0.1, ..., 0.0],  # 프레임 1: 약간 웃음
    [0.5, 0.8, 0.2, ..., 0.1],  # 프레임 2: 더 큰 웃음
    ...
])  # shape: [100, 79]
```
- **의미**: 각 프레임에서의 **얼굴 표정**을 나타내는 79차원 벡터
- **특징**: **프레임마다 다름** (시간에 따라 표정이 변하므로)
- **예시**:
  - `exp[0, 0]`: 입꼬리 올리기 (웃음)
  - `exp[0, 1]`: 눈썹 올리기 (놀람)
  - `exp[0, 2]`: 입 벌리기 (말하기)
  - ... (79개 파라미터로 모든 표정 변화를 표현)

### **3. `euler` (회전 각도)**
```python
euler: torch.tensor([
    [0.1, 0.0, 0.0],   # 프레임 0: 약간 오른쪽으로 돌아봄
    [0.2, 0.1, 0.0],   # 프레임 1: 더 오른쪽으로, 약간 위를 봄
    [0.0, -0.1, 0.0],  # 프레임 2: 왼쪽으로 돌아봄
    ...
])  # shape: [100, 3]
```
- **의미**: 각 프레임에서 얼굴의 **3D 회전 각도** (라디안)
- **순서**: `[pitch, yaw, roll]`
  - **pitch**: 위아래 고개 끄덕임
  - **yaw**: 좌우 고개 돌리기
  - **roll**: 좌우로 기울이기

### **4. `trans` (3D 변환)**
```python
trans: torch.tensor([
    [0.1, 0.0, -7.2],   # 프레임 0: x=0.1, y=0.0, z=-7.2
    [0.2, 0.1, -7.1],   # 프레임 1: x=0.2, y=0.1, z=-7.1
    [0.0, 0.0, -7.3],   # 프레임 2: x=0.0, y=0.0, z=-7.3
    ...
])  # shape: [100, 3]
```
- **의미**: 각 프레임에서 얼굴의 **3D 위치** (미터 단위)
- **좌표계**: 카메라 기준
  - **x**: 좌우 이동 (양수=오른쪽)
  - **y**: 위아래 이동 (양수=위쪽)
  - **z**: 앞뒤 이동 (음수=카메라에서 멀어짐)

### **5. `focal` (카메라 초점거리)**
```python
focal: torch.tensor([1200.0])  # shape: [1]
```
- **의미**: 카메라의 **초점거리** (픽셀 단위)
- **특징**: **모든 프레임에서 동일** (같은 카메라로 촬영)
- **예시**: 1200픽셀 = 약 50mm 렌즈 (이미지 크기에 따라 다름)

## **실제 사용 예시**:

```python
# 프레임 5에서의 얼굴 렌더링
frame_id = 5
face_geometry = model_3dmm.get_3dlandmarks(
    id_para,                    # [100] - 이 사람의 얼굴 형태
    exp_para[frame_id],         # [79] - 프레임 5의 표정
    euler_angle[frame_id],      # [3] - 프레임 5의 회전
    trans[frame_id],            # [3] - 프레임 5의 위치
    focal_length,               # [1] - 카메라 초점거리
    camera_center               # [2] - 이미지 중심
)
```

이렇게 **각 프레임의 3D 얼굴 파라미터**를 통해 InsTaG이 **시간에 따라 변하는 얼굴의 3D 모델**을 정확히 재구성할 수 있습니다.
"""