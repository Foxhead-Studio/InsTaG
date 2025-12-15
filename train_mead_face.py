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
import random
import torch
from random import randint
from utils.loss_utils import l1_loss, l2_loss, patchify, ssim, normalize
from gaussian_renderer import render, render_motion
import sys
from scene import Scene, GaussianModel, MotionNetwork # 여기서 개인에 대한 가우시안 scene을 만드는 것은, pretrain_face.py의 pretrain_scene과 다른 파일이다.
from utils.general_utils import safe_state
import lpips
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.normal_utils import depth_to_normal
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from tensorboardX import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, mode_long, pretrain_ckpt_path):
    testing_iterations = [1] + [i for i in range(0, opt.iterations + 1, 10000)] # [1, 0, 10000]
    # saving_iterations가 비어있지 않으면(인자로 전달되었으면) 그것을 사용하고, 비어있으면 10000 간격으로 생성
    if not saving_iterations:
        saving_iterations = [i for i in range(0, opt.iterations + 1, 10000)] + [opt.iterations] # [2000, 4000, 6000, ..., opt.iterations]
    # checkpoint_iterations가 비어있지 않으면(인자로 전달되었으면) 그것을 사용하고, 비어있으면 10000 간격으로 생성
    if not checkpoint_iterations:
        checkpoint_iterations = [i for i in range(0, opt.iterations + 1, 10000)] + [opt.iterations] # [0, 10000, opt.iterations]

    # vars
    warm_step = 3000
    opt.densify_until_iter = opt.iterations - 1000 # 9000
    bg_iter = opt.iterations # 10000, 해당 스텝 이후에는 모델 업데이트를 멈추는데, 이터레이션과 동일한 값이라 의미가 없음. 
    lpips_start_iter = opt.densify_until_iter - 1500 # 7500, pretrain과 달리 여기서는 1500회 남기고 lpips 손실도 계산
    motion_stop_iter = bg_iter # 10000 역시 의미 없는 값.
    mouth_select_iter = opt.iterations # 10000 의미 없는 값
    mouth_step = 1 / max(mouth_select_iter, 1) # 1 / 10000 = 1e-04
    hair_mask_interval = 7
    select_interval = 10

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset)
    # 아무런 값도 세팅되지 않은 초기 가우시안 초기화.
    # 개인에 대한 PMF(neural_motion_grid)도 초기화.
    
    scene = Scene(dataset, gaussians)
    # scene 객체에는 위에서 초기화한 가우시안과 프레임 별 카메라 정보, 이 두 가지가 담겨있음.

    motion_net = MotionNetwork(args=dataset).cuda() # Face 브랜치의 UMF 초기화.
    motion_optimizer = torch.optim.AdamW(motion_net.get_params(5e-3, 5e-4), betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01) # UMF의 파라미터를 옵티마이저에 저장해서 학습 되도록 함.
    scheduler = torch.optim.lr_scheduler.LambdaLR(motion_optimizer, lambda iter: 0.1 if iter < warm_step else 0.5 ** (iter / opt.iterations))
    if mode_long:
        scheduler = torch.optim.lr_scheduler.LambdaLR(motion_optimizer, lambda iter: 0.1 if iter < warm_step else 0.1 ** (iter / opt.iterations))
    
    # ==============모델 불러오기================== #
    # Load pre-trained
    (motion_params, _, _) = torch.load(pretrain_ckpt_path) # Face 브랜치의 UMF 로드 'output/pretrain_ave/chkpnt_ema_face_latest.pth'
    # gaussians.restore(model_params, opt)
    motion_net.load_state_dict(motion_params) # Face 브랜치의 UMF 체크포인트 로드.
    
    # ##################### 중요! #####################
    # 아래는 Face 브랜치의 PMF를 불러오는 코드인데, 주석처리되어 사용하지 않음. 즉, 오로지 UMF만을 사용한다는 뜻.
    # 왜냐하면, pretrain_face.py에서는 PMF에서 가우시안의 변위량을 계산한 뒤 negative contrast loss를 걸어 공통된 motion만 UMF에 저장하고 개별적인 모션은 PMF에 저장한다.
    # 그러나 fine tuning 단계인 여기서는 PMF의 개별적인 가우시안 변위량을 사용하는 것이 아니라, UMF의 변화량을 개인에 adaptation하는 p_xyz, p_scale만을 사용하기 때문이다.
    # 즉 pretrain 단계에서 학습한 personal motion은 아예 사용이 안 되니까 weight를 불러오는 게 의미가 없는 것!
    
    # (model_params, _, _, _) = torch.load(os.path.join("output/pretrain4/macron/chkpnt_face_latest.pth"))
    # gaussians.neural_motion_grid.load_state_dict(model_params[-1])
    # #################################################

    lpips_criterion = lpips.LPIPS(net='alex').eval().cuda()

    gaussians.training_setup(opt)
    # 위에서 UMF에 대한 학습률 스케줄러를 설정하여 학습할 준비를 완료했듯,
    # 이제 가우시안의 속성(좌표, 회전값, 색상 등) + PMF(neural_motion_grid)를 파라미터화 해서 옵티마이저에 등록, 옵티마이저를 생성하고 각 파라미터에 대한 학습률 스케줄러를 설정.

    if checkpoint: # 만약 fine-tuning하던 중 중단된 경우, 이어서 학습. 현재는 None으로 넘어감. (어차피 한 번 학습하는데 5분이면 돼서 이어서 학습할 것도 없음.)
        (model_params, motion_params, motion_optimizer_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        motion_net.load_state_dict(motion_params)
        motion_optimizer.load_state_dict(motion_optimizer_params)


    # print(gaussians.get_xyz)

    if not mode_long: # not Flase -> True
        gaussians.max_sh_degree = 1 # 짧게 학습할 것이므로 SH의 맥스 차원을 1로 세팅.
    # ==================================== #
    bg_color = [0, 1, 0]   # [1, 1, 1] # if dataset.white_background else [0, 0, 0]
    # 현재는 배경을 초록색으로 세팅.
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda") # 배경도 텐서로 바꾼 뒤 GPU에 올리기


    iter_start = torch.cuda.Event(enable_timing = True) # CUDA 이벤트 객체인 iter_start 생성. `enable_timing = True`는 이 이벤트가 시간 측정에 사용될 수 있도록 한다.
    iter_end = torch.cuda.Event(enable_timing = True) # CUDA 이벤트 객체인 iter_end 생성.

    viewpoint_stack = None # scene.train_cameras에 있는 정보를 복사할 변수
    ema_loss_for_log = 0.0 # 로그 기록을 위한 EMA(지수이동평균) 손실값을 0.0으로 초기화
    progress_bar = tqdm(range(first_iter, opt.iterations), ascii=True, dynamic_ncols=True, desc="Training progress")
    first_iter += 1 # 첫번째 이터레이션이니 0에서 1로 설정. 이후 반복문에서 1씩 증가.
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration) # 가우시안 좌표 X, Y, Z에 대한 러닝 레이트를 업데이트 하기 위해, 현재 이터레이션에 맞는 learning rate를 스케줄러 함수로부터 받아와 할당.

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
            # active_sh_degree를 max_sh_degree이전까지 1씩 증가시킨다.
            # mode_long이 아니므로 0 -> 1로 증가하고 그 이후에는 더 이상 증가하지 않음.

        # Pick a random Camera
        # 멀티뷰인 경우 front 뷰만 사용하여 프레임 선택
        if not viewpoint_stack:
            all_cameras = scene.getTrainCameras().copy()
            if scene.multiview_data is not None:
                # 멀티뷰인 경우, front 뷰의 프레임만 필터링
                viewpoint_stack = [cam for cam in all_cameras if cam.talking_dict.get('view_name') == 'front']
                if not viewpoint_stack:
                    # front 뷰가 없으면 모든 카메라 사용 (하위 호환성)
                    viewpoint_stack = all_cameras
            else:
                # 단일 뷰인 경우 모든 카메라 사용
                viewpoint_stack = all_cameras
            # 프레임별 FoV, [R|T], H, W, img, img path, talking dict (img id, img, teeth mask, mask path, 오디오 피처, 액션유닛, 입, 입술, 하관 바운딩 박스 좌표)
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1)) # 랜덤한 프레임을 선택 (front 뷰 기준)
        selected_img_id = viewpoint_cam.talking_dict['img_id']  # 선택된 프레임의 img_id 저장

        # find a big mouth
        # # viewpoint_cam.talking_dict['mouth_bound'] = [전체 프레임에서 입 내부 최소 벌림, 최대 벌림, 현재 프레임의 입이 벌어진 길이]
        mouth_global_lb = viewpoint_cam.talking_dict['mouth_bound'][0] # 0번째 인덱스인 입 내부 최소 벌림
        mouth_global_ub = viewpoint_cam.talking_dict['mouth_bound'][1] # 1번째 인덱스인 입 내부 최대 벌림
        mouth_global_lb += (mouth_global_ub - mouth_global_lb) * 0.2 # # 최소 벌림 값을 전체의 20%만큼 올림 0 + (27 - 0) * 0.2 = 5.4
        mouth_window = (mouth_global_ub - mouth_global_lb) * 0.5 # pretrain과 달리 0.2가 아니라 0.5
        # # 입 범위의 50%만큼 윈도우(탐색 범위)를 설정합니다.  (27 - 5.4) * 0.5 = 10.8
        # train_face.py에서는 mouth_window가 훨씬 넓다. 이는 학습 초반부터 더 넓은 범위의 입 모양을 탐색하고, 샘플링 필터링이 덜 엄격하게 적용됨을 의미한다. 
        # 즉, 다양한 입 벌림 크기의 프레임을 더 쉽게 학습에 포함하여 일반화된 얼굴 모션 학습에 더 적합하다.


        # mouth_global_lb : 전체 데이터셋의 최소 입벌림(여기에 전체 구간의 20%를 더함)
        # mouth_global_ub : 전체 데이터셋의 최대 입벌림
        # mouth_step : 입 구간 샘플 이동 속도
        # mouth_window : 입 구간 탐색 윈도우 (구간의 50%)
        # 예시: mouth_global_lb=0, mouth_global_ub=27, mouth_step=0.00005
        #       >> mouth_global_lb += (27-0)*0.2 = 5.4, mouth_window = (27-5.4)*0.5 = 10.8

        # ① mouth_lb 계산 (iteration에 따라 증가):
        #    mouth_lb = mouth_global_lb + mouth_step * iteration * (mouth_global_ub - mouth_global_lb)
        #    예: iteration=100일 때
        #        mouth_lb = 5.4 + 0.00005 * 100 * (27 - 5.4)
        #               = 5.4 + 0.00005 * 100 * 21.6
        #               = 5.4 + 0.108 = 5.508
        #        iteration=15000일 때
        #        mouth_lb = 5.4 + 0.00005 * 15000 * 21.6
        #               = 5.4 + 16.2 = 21.6

        mouth_lb = mouth_global_lb + mouth_step * iteration * (mouth_global_ub - mouth_global_lb)

        # ② mouth_ub 계산 (윈도우만큼 더함)
        #    mouth_ub = mouth_lb + mouth_window
        #    예: iteration=100인 경우
        #        mouth_ub = 5.508 + 10.8 = 16.308
        #        iteration=15000인 경우
        #        mouth_ub = 21.6 + 10.8 = 32.4

        mouth_ub = mouth_lb + mouth_window

        # ③ mouth_lb를 다시 mouth_window만큼 낮춰서 window 구간을 만들기
        #    mouth_lb = mouth_lb - mouth_window
        #    예: iteration=100인 경우
        #        mouth_lb = 5.508 - 10.8 = -5.292
        #        iteration=15000인 경우
        #        mouth_lb = 21.6 - 10.8 = 10.8

        mouth_lb = mouth_lb - mouth_window

        # 따라서 최종적으로 [mouth_lb, mouth_ub] 구간 예시:
        #  - iteration=0      : mouth_lb = 5.4 - 10.8 = -5.4,      mouth_ub = 5.4 + 10.8 = 16.2
        #  - iteration=100    : mouth_lb = 5.508 - 10.8 = -5.292,  mouth_ub = 5.508 + 10.8 = 16.308
        #  - iteration=15000  : mouth_lb = 21.6  - 10.8 = 10.8,    mouth_ub = 21.6 + 10.8 = 32.4
        # 즉, 학습이 진행될수록 입벌림 샘플링 구간이 우측(더 큰 입 벌림)으로 이동합니다.

        au_global_lb = 0 # 액션 유닛 하한값
        au_global_ub = 1 # 액션 유닛 상한값
        au_window = 0.4 # 액션 유닛 탐색 윈도우, pretrain_face.py의 0.3이 아니라 0.4로 설정
        # train_face.py에서는 au_window가 더 넓다. 이는 AU 값의 더 넓은 범위를 탐색하며, pretrain_face.py에 비해 AU 기반 샘플링에서 더 많은 프레임을 허용한다.

        au_lb = au_global_lb + mouth_step * iteration * (au_global_ub - au_global_lb) # # 액션 유닛 하한값을 학습 시간에 따라 변화
        au_ub = au_lb + au_window # # 액션 유닛 상한값을 윈도우에 따라 업데이트
        au_lb = au_lb - au_window * 1.5 # 0.5가 아니라 1.5 하한값을 윈도우의 1.5배만큼 더 낮춤
        # train_face.py에서는 au_lb를 훨씬 더 큰 폭으로 낮춘다. 이는 AU 값이 매우 낮은(예: 눈을 거의 감지 않거나 특정 표정이 거의 없는) 프레임까지도 학습에 적극적으로 포함시키려는 시도이다.
        
        # au_lb가 과도하게 높아지지 않도록 제한 (blink 값은 0~1 범위이므로)
        # au_lb의 최대값을 0.15로 제한하여 항상 충분한 범위의 프레임을 샘플링할 수 있도록 함
        # 0.15로 제한하면 au_lb 범위가 [-0.6, 0.15]가 되어 대부분의 blink 값을 포함할 수 있음
        au_lb_max = 0.15 # au_lb의 최대값 제한 (더 낮게 설정하여 다양한 프레임 샘플링)
        if au_lb > au_lb_max:
            au_lb = au_lb_max
            au_ub = au_lb + au_window # au_lb가 제한되면 au_ub도 재조정

        if iteration < warm_step and iteration < mouth_select_iter: # 학습 초반 (iteration < 3,000 < 10,000)
            if iteration % select_interval == 0: # iteration = 10의 배수일 때
                max_attempts = 100 # 최대 시도 횟수
                attempts = 0 # 현재 시도 횟수
                while (viewpoint_cam.talking_dict['mouth_bound'][2] < mouth_lb or viewpoint_cam.talking_dict['mouth_bound'][2] > mouth_ub) and attempts < max_attempts: # mouth_lb < 해당 프레임 입벌림 < mouth_ub일 때까지 다시 뽑기
                    if not viewpoint_stack: # pop을 너무 많이 해서 다 지워졌다면,
                        all_cameras = scene.getTrainCameras().copy()
                        if scene.multiview_data is not None:
                            # 멀티뷰인 경우 front 뷰만 필터링
                            viewpoint_stack = [cam for cam in all_cameras if cam.talking_dict.get('view_name') == 'front']
                        else:
                            viewpoint_stack = all_cameras
                    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1)) # 다시 뽑기
                    selected_img_id = viewpoint_cam.talking_dict['img_id']  # 선택된 프레임의 img_id 업데이트
                    attempts += 1
                
                # 조건을 만족하는 프레임을 찾지 못한 경우, 가장 가까운 프레임 선택
                if attempts >= max_attempts:
                    all_cameras = scene.getTrainCameras()
                    # 멀티뷰인 경우 front 뷰만 필터링
                    if scene.multiview_data is not None:
                        all_cameras = [cam for cam in all_cameras if cam.talking_dict.get('view_name') == 'front']
                    best_cam = None
                    min_distance = float('inf')
                    violation_type = None # 'lower' 또는 'upper'
                    
                    for cam in all_cameras:
                        mouth_val = cam.talking_dict['mouth_bound'][2]
                        if mouth_val < mouth_lb:
                            # 하한을 위반한 경우
                            distance = mouth_lb - mouth_val
                            if distance < min_distance:
                                min_distance = distance
                                best_cam = cam
                                violation_type = 'lower'
                        elif mouth_val > mouth_ub:
                            # 상한을 위반한 경우
                            distance = mouth_val - mouth_ub
                            if distance < min_distance:
                                min_distance = distance
                                best_cam = cam
                                violation_type = 'upper'
                        else:
                            # 조건을 만족하는 프레임이 있는 경우 (이론적으로는 여기 도달하지 않아야 함)
                            best_cam = cam
                            violation_type = None
                            break
                    
                    if best_cam is not None:
                        viewpoint_cam = best_cam
                        selected_img_id = best_cam.talking_dict['img_id']  # 선택된 프레임의 img_id 업데이트
                        if violation_type == 'lower':
                            print(f"[ITER {iteration}] 입 벌림 샘플링: 하한({mouth_lb:.2f})을 만족하는 프레임이 없어, 가장 가까운 프레임(입벌림={best_cam.talking_dict['mouth_bound'][2]:.2f}) 선택")
                        elif violation_type == 'upper':
                            print(f"[ITER {iteration}] 입 벌림 샘플링: 상한({mouth_ub:.2f})을 만족하는 프레임이 없어, 가장 가까운 프레임(입벌림={best_cam.talking_dict['mouth_bound'][2]:.2f}) 선택")

        # iteration에 따른 입 벌림 범위 예시 (mouth_global_lb=5.4, mouth_global_ub=27, mouth_window=10.8 가정):
        # iteration = 0: mouth_lb = -5.4, mouth_ub = 16.2 (범위: [-5.4, 16.2])
        # iteration = 1000: mouth_lb = -3.24, mouth_ub = 18.36 (범위: [-3.24, 18.36])
        # iteration = 2000: mouth_lb = -1.08, mouth_ub = 20.52 (범위: [-1.08, 20.52])
        # iteration = 3000: mouth_lb = 1.08, mouth_ub = 22.68 (범위: [1.08, 22.68])

        # iteration이 증가하면:
        # mouth_lb가 올라가면서 작은 입 벌림 값이 제외됨
        # mouth_ub도 함께 올라가면서 더 큰 입 벌림을 요구함
        # 결과적으로 샘플링 범위가 더 큰 입 벌림 쪽으로 이동함

        # 짧은 비디오의 경우:
        # 프레임 수가 적어 입 벌림 값 분포가 제한적
        # 후반부 iteration에서 요구 범위에 맞는 프레임이 없을 수 있음
        # 예: iteration=3000에서 [1.08, 22.68] 범위를 요구하지만, 실제 입 벌림이 0~1에만 있다면 무한 루프 발생

        if warm_step < iteration < mouth_select_iter: # 학습 후반 (3,000 < iteration < 10,000)
            if iteration % select_interval == 0: # iteration = 10의 배수일 때
                max_attempts = 100 # 최대 시도 횟수
                attempts = 0 # 현재 시도 횟수
                blink_val_current = viewpoint_cam.talking_dict['blink'].item() if torch.is_tensor(viewpoint_cam.talking_dict['blink']) else viewpoint_cam.talking_dict['blink']
                while (blink_val_current < au_lb or blink_val_current > au_ub) and attempts < max_attempts: # au_lb < 해당 프레임의 눈을 감은 정도의 액션유닛 < au_ub일 때까지 다시 뽑기
                    if not viewpoint_stack: # pop을 너무 많이 해서 다 지워졌다면,
                        all_cameras = scene.getTrainCameras().copy()
                        if scene.multiview_data is not None:
                            # 멀티뷰인 경우 front 뷰만 필터링
                            viewpoint_stack = [cam for cam in all_cameras if cam.talking_dict.get('view_name') == 'front']
                        else:
                            viewpoint_stack = all_cameras
                    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1)) # 다시 뽑기
                    selected_img_id = viewpoint_cam.talking_dict['img_id']  # 선택된 프레임의 img_id 업데이트
                    blink_val_current = viewpoint_cam.talking_dict['blink'].item() if torch.is_tensor(viewpoint_cam.talking_dict['blink']) else viewpoint_cam.talking_dict['blink']
                    attempts += 1
                
                # 조건을 만족하는 프레임을 찾지 못한 경우 처리
                if attempts >= max_attempts:
                    all_cameras = scene.getTrainCameras()
                    # 멀티뷰인 경우 front 뷰만 필터링
                    if scene.multiview_data is not None:
                        all_cameras = [cam for cam in all_cameras if cam.talking_dict.get('view_name') == 'front']
                    
                    if len(all_cameras) == 0:
                        print(f"[ITER {iteration}] blink 샘플링: 사용 가능한 카메라가 없습니다.")
                        continue
                    
                    # 가장 가까운 프레임 찾기
                    best_cam = None
                    min_distance = float('inf')
                    violation_type = None # 'lower' 또는 'upper'
                    
                    for cam in all_cameras:
                        blink_val = cam.talking_dict['blink'].item() if torch.is_tensor(cam.talking_dict['blink']) else cam.talking_dict['blink']
                        if blink_val < au_lb:
                            # 하한을 위반한 경우
                            distance = au_lb - blink_val
                            if distance < min_distance:
                                min_distance = distance
                                best_cam = cam
                                violation_type = 'lower'
                        elif blink_val > au_ub:
                            # 상한을 위반한 경우
                            distance = blink_val - au_ub
                            if distance < min_distance:
                                min_distance = distance
                                best_cam = cam
                                violation_type = 'upper'
                        else:
                            # 조건을 만족하는 프레임이 있는 경우 (이론적으로는 여기 도달하지 않아야 함)
                            best_cam = cam
                            violation_type = None
                            break
                    
                    # 조건을 만족하지 못한 경우 랜덤 샘플링 수행
                    # 같은 프레임만 반복되는 것을 방지하기 위해 항상 랜덤 샘플링
                    viewpoint_cam = all_cameras[randint(0, len(all_cameras)-1)]
                    selected_img_id = viewpoint_cam.talking_dict['img_id']
                    blink_val_random = viewpoint_cam.talking_dict['blink'].item() if torch.is_tensor(viewpoint_cam.talking_dict['blink']) else viewpoint_cam.talking_dict['blink']
                    print(f"[ITER {iteration}] blink 샘플링: 조건을 만족하는 프레임을 찾지 못해 랜덤 샘플링 수행 (blink={blink_val_random:.2f}, 요구 범위=[{au_lb:.2f}, {au_ub:.2f}])")

        # iteration에 따른 범위 예시:
        # iteration = 3000: au_lb = -0.3, au_ub = 0.7 (범위: [-0.3, 0.7])
        # iteration = 5000: au_lb = -0.1, au_ub = 0.9 (범위: [-0.1, 0.9])
        # iteration = 7000: au_lb = 0.1, au_ub = 1.1 (범위: [0.1, 1.1])
        # iteration = 9000: au_lb = 0.3, au_ub = 1.3 (범위: [0.3, 1.3])

        # iteration이 증가하면:
        # au_lb가 올라가면서 낮은 blink 값이 제외됨
        # au_ub가 1을 넘어가지만 blink는 최대 1이므로 상한은 실질적으로 1
        # 결과적으로 허용 범위가 좁아짐

        # 짧은 비디오의 경우:
        # 프레임 수가 적어 blink 값 분포가 제한적
        # 후반부 iteration에서 요구 범위에 맞는 프레임이 없을 수 있음
        # 예: iteration=9000에서 [0.3, 1.0] 범위를 요구하지만, 모든 프레임에서 눈을 감는 프레임이 없어, 실제 blink가 0.0~0.2에만 있다면 무한 루프 발생

        # Render
        if (iteration - 1) == debug_from: # debug_from값이 0이면, 첫 번째 이터레이션부터 디버그 세팅을 활성화. 현재는 -1이므로 디버그 세팅을 활성화하지 않음.
            pipe.debug = True

        # 멀티뷰인 경우 선택된 프레임의 모든 뷰를 가져옴
        if scene.multiview_data is not None:
            multiview_cameras = scene.getMultiViewFrame(selected_img_id)
            # front 뷰를 먼저 처리하기 위해 정렬
            view_order = sorted(multiview_cameras.keys())
            if 'front' in view_order:
                view_order.remove('front')
                view_order.insert(0, 'front')
        else:
            # 단일 뷰인 경우 현재 카메라만 사용
            multiview_cameras = {'single_view': viewpoint_cam}
            view_order = ['single_view']
        
        # 모든 뷰에 대해 손실을 누적할 변수 초기화
        total_loss = None
        total_Ll1 = None
        
        # 각 뷰에 대해 순차적으로 가우시안 피팅 수행
        for view_name in view_order:
            viewpoint_cam = multiview_cameras[view_name]
            
            # 현재 뷰포인트 캠의 3가지 마스크를 가져와 GPU에 할당.
            face_mask = torch.as_tensor(viewpoint_cam.talking_dict["face_mask"]).cuda() # [512, 512] True/False 마스크
            hair_mask = torch.as_tensor(viewpoint_cam.talking_dict["hair_mask"]).cuda() # [512, 512] True/False 마스크
            mouth_mask = torch.as_tensor(viewpoint_cam.talking_dict["mouth_mask"]).cuda() # [512, 512] True/False 마스크
            head_mask =  face_mask + hair_mask # 얼굴, 머리카락 마스크를 합친 마스크도 생성.

            # (1) iteration > lpips_start_iter(=7500)일 때, mouth_mask에 두 번의 max pooling 연산을 적용한다.
            #     max pooling을 두 번 거치면 mask가 조금 더 넓고, 경계가 자연스럽게 부드러워진다.
            #     이렇게 하면 입 주변의 sharp한(날카로운) 경계에서생기는 artifact(블록 노이즈, 딱딱한 패턴)를 줄이고, 학습 시 gradient가 보다 자연스럽게 퍼진다.
            #     즉, 마스크 경계에서 생길 수 있는 인위적(unnatural) 패턴 또는 렌더링 노이즈를 완화하는 효과가 있다.
            if iteration > lpips_start_iter: # 7500 이상일 경우 아래 수행
                max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
                mouth_mask = (-max_pool(-max_pool(mouth_mask[None].float())))[0].bool()

            # (2) hair_mask_iter는 머리카락 부분(머리카락 마스크)의 학습 손실 적용 여부를 결정하는 플래그다.
            #     3000 < iteration < 6500이고, iteration이 7의 배수가 아닐 때 True가 되어 머리카락 부분 픽셀만 따로 background로 지운다.
            #     이렇게 일정 간격으로 머리카락에 대한 복원 loss를 off했다가 on/off를 반복함으로써, 머리카락 복원은 7번에 한 번만 하고, 나머지 경우는 얼굴 주요 부분 학습에 더 집중하게 유도한다.
            hair_mask_iter = (warm_step < iteration < lpips_start_iter - 1000) and iteration % hair_mask_interval != 0
            
            # ##################### 중요! #####################
            if iteration < warm_step: # 3000 이전
                # render_pkg = render(viewpoint_cam, gaussians, pipe, background)
                enable_align = iteration > 1000 # 1000번 이전까진 align 안 함.
                render_pkg = render_motion(viewpoint_cam, gaussians, motion_net, pipe, background, return_attn=True, personalized=False, align=enable_align)
                # for param in motion_net.parameters():
                #     param.requires_grad = False
            else:
                render_pkg = render_motion(viewpoint_cam, gaussians, motion_net, pipe, background, return_attn=True, personalized=False, align=True)
                # for param in motion_net.parameters():
                #     param.requires_grad = True

            # 렌더링 방식은 iteration에 따라 달라진다:
            # (1) iteration < 1000: align=False, personalized=False
            #     - PMF(neural_motion_grid)를 사용하지 않음
            #     - UMF만 사용하여 움직임 예측
            #     - 예: iteration=500일 때, align=False이므로 PMF 없이 UMF만으로 렌더링
            # (2) 1000 <= iteration < 3000: align=True, personalized=False
            #     - PMF를 사용하여 스케일 보정만 수행 (p_xyz, p_scale 사용)
            #     - UMF의 변위를 개인화된 스케일로 조정
            #     - 예: iteration=2000일 때, align=True이므로 PMF로 스케일 보정 후 UMF 적용
            # (3) iteration >= 3000: align=True, personalized=False
            #     - PMF를 사용하여 스케일 보정 수행
            #     - UMF의 변위를 개인화된 스케일로 조정
            #     - 예: iteration=5000일 때, align=True이므로 PMF로 스케일 보정 후 UMF 적용
            
            # align=True일 때의 동작:
            # 1. PMF(neural_motion_grid)로부터 p_xyz, p_scale 예측
            # 2. xyz = xyz + p_xyz (캐노니컬 가우시안 위치에 개인화된 변위를 추가하여 얼굴 크기를 UMF의 스케일에 맞게 조정)
            # 3. UMF로부터 d_xyz, d_scale, d_rot 예측
            # 4. d_xyz *= p_scale (UMF의 변위를 개인화된 스케일로 조정하여 얼굴 크기에 맞는 움직임 반경 조정)
            # 이렇게 하면 UMF와 캐노니컬 가우시안 간의 스케일 차이를 보정할 수 있다.
            # ################################################

            image_white, alpha, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["alpha"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            # render_pkg에서 렌더링 결과를 추출한다:
            # image_white: 렌더링된 최종 RGB 이미지. [3, H, W] 형태의 텐서로, 각 픽셀의 RGB 값이 0~1 범위로 정규화되어 있다.
            #              예: [3, 512, 512] 크기의 텐서에서 [0, :, :]는 R 채널, [1, :, :]는 G 채널, [2, :, :]는 B 채널을 나타낸다.
            # alpha: 알파 채널(투명도) 정보. [1, H, W] 형태의 텐서로, 각 픽셀의 불투명도를 나타낸다 (0=완전 투명, 1=완전 불투명).
            #        예: [1, 512, 512] 크기의 텐서에서 alpha[0, i, j]는 (i, j) 픽셀의 투명도를 나타낸다.
            #        배경 부분은 0에 가깝고, 얼굴/머리 부분은 1에 가깝다.
            # viewspace_point_tensor: 뷰스페이스(카메라 시점) 상의 포인트 좌표. [N, 3] 형태의 텐서로, N개의 가우시안 포인트의 2D 스크린 좌표와 깊이 정보를 담고 있다.
            #                          예: [2000, 3] 크기의 텐서에서 각 행은 하나의 가우시안 포인트의 (x, y, depth) 좌표를 나타낸다.
            #                          이 값은 gradient 추적을 위해 사용되며, densification 과정에서 가우시안 분할/제거 판단에 활용된다.
            # visibility_filter: 반지름이 0보다 큰 가우시안만 포함하는 불리언 마스크. [N] 형태의 텐서로, 각 가우시안이 화면에 보이는지 여부를 나타낸다.
            #                    예: [2000] 크기의 텐서에서 True는 해당 가우시안이 화면에 보이고, False는 frustum culling되거나 반지름이 0인 가우시안이다.
            #                    이 필터는 densification 과정에서 보이지 않는 가우시안을 제외하고 업데이트할 때 사용된다.
            # radii: 각 가우시안의 2D 스크린 공간에서의 반지름(픽셀 단위). [N] 형태의 텐서로, 각 가우시안이 화면에 투영되었을 때의 크기를 나타낸다.
            #        예: [2000] 크기의 텐서에서 radii[i]는 i번째 가우시안의 2D 반지름을 나타낸다.
            #        이 값은 가우시안이 화면에 얼마나 큰 영역을 차지하는지 결정하며, rasterization 과정에서 사용된다.

            gt_image  = viewpoint_cam.original_image.cuda() / 255.0
            # `viewpoint_cam` 객체에서 원본 이미지(`original_image`)를 가져와 GPU에 할당하고, 픽셀 값을 0-1 범위로 정규화.
            gt_image_white = gt_image * head_mask + background[:, None, None] * ~head_mask
            # `gt_image_white`는 원본 이미지(`gt_image`)에 `head_mask`(얼굴과 머리카락을 포함한 마스크)를 적용하고, 머리 영역이 아닌 부분(`~head_mask`)에는 `background` 색상을 채워 넣어 생성한다.


            # ##################### 중요! #####################
            # 이 부분이 나중에 무표정한 캐노니컬 가우시안을 만들 때 내가 쓸 수도 있어서 자세하게 주석을 단다.
            
            # 학습 후반부 파라미터 동결 (현재는 motion_stop_iter=bg_iter=10000이므로 실행되지 않지만, 더 긴 학습 시나리오를 위한 코드):
            # (1) iteration > motion_stop_iter일 때: motion network(UMF)의 모든 파라미터를 동결
            #     - UMF가 충분히 학습되었다고 가정하고, 더 이상 업데이트하지 않음
            #     - 예: iteration=15000일 때, motion_net의 파라미터들이 requires_grad=False로 설정되어 gradient 계산이 비활성화됨
            #     - 이렇게 하면 motion network의 학습이 멈추고, 가우시안 파라미터만 계속 학습됨
            if iteration > motion_stop_iter: # 10000 이상일 때라 의미 없음
                for param in motion_net.parameters():
                    param.requires_grad = False
            
            # (2) iteration > bg_iter일 때: 가우시안의 구조적 파라미터들을 동결
            #     - 가우시안의 위치(_xyz), 불투명도(_opacity), 스케일(_scaling), 회전(_rotation)을 고정
            #     - 예: iteration=15000일 때, 가우시안의 기본 구조가 이미 최적화되었다고 가정하고 동결
            #     - 이렇게 하면 가우시안의 색상/특징(_features_dc, _features_rest)만 계속 학습됨
            #     - 목적: 기본적인 구조는 고정하고, 색상/텍스처 등 시각적 디테일만 미세 조정
            #     - 주의: _features_dc, _features_rest는 주석 처리되어 있어 색상은 계속 학습됨
            if iteration > bg_iter: # 10000 이상일 때라 의미 없음
                gaussians._xyz.requires_grad = False # 가우시안의 3D 위치를 동결
                gaussians._opacity.requires_grad = False # 가우시안의 불투명도를 동결
                # gaussians._features_dc.requires_grad = False # 가우시안의 DC 색상 계수를 동결 (주석 처리됨)
                # gaussians._features_rest.requires_grad = False # 가우시안의 나머지 SH 계수를 동결 (주석 처리됨)
                gaussians._scaling.requires_grad = False # 가우시안의 스케일(크기)을 동결
                gaussians._rotation.requires_grad = False # 가우시안의 회전을 동결
            

            # Loss
            if iteration < bg_iter: # 항상 적용
                # 머리카락 마스크 적용: hair_mask_iter가 True일 때만 실행 (3000 < iteration < 6500이고 iteration이 7의 배수가 아닌 모든 경우)
                # 이렇게 하면 일정 간격으로 머리카락 부분을 손실 계산에서 제외하여, 얼굴 주요 부분 학습에 집중할 수 있다.
                if hair_mask_iter:
                    # 렌더링된 이미지에서 머리카락 영역을 배경색으로 채운다.
                    # 예: image_white가 [3, 512, 512]이고 hair_mask가 [512, 512] 불리언 마스크일 때,
                    #     image_white[:, hair_mask]는 머리카락 영역의 RGB 픽셀들을 선택하고, background[:, None]으로 채운다.
                    #     background는 [3] 형태이므로 [:, None]으로 [3, 1]로 확장하여 브로드캐스팅한다.
                    image_white[:, hair_mask] = background[:, None]
                    # 정답 이미지에서도 머리카락 영역을 배경색으로 채운다.
                    # 이렇게 해야 렌더링된 이미지와 정답 이미지를 비교할 수 있다.
                    gt_image_white[:, hair_mask] = background[:, None]
                
                
                # ###################### 중요! ######################
                # 입 마스크 적용: 정답 이미지에서 입 영역을 배경색으로 채운다.
                # 입 내부 영역은 별도의 입 모델(train_mouth.py)에서 학습하므로, 얼굴 학습에서는 입 내부 영역을 제외한다.
                # 주석 처리된 image_white[:, mouth_mask] = 1은 디버깅용 코드로 보인다.
                # image_white[:, mouth_mask] = 1
                gt_image_white[:, mouth_mask] = background[:, None]

                # 기본 손실 함수 계산:
                # Ll1: L1 손실 (Mean Absolute Error). 렌더링된 이미지와 정답 이미지의 픽셀 차이의 절댓값 평균을 계산한다.
                #      예: image_white와 gt_image_white가 [3, 512, 512]일 때, 각 픽셀의 RGB 값 차이의 절댓값을 모두 더한 후 평균을 낸다.
                view_Ll1 = l1_loss(image_white, gt_image_white)
                # 전체 손실: L1 손실 + DSSIM 손실 (구조적 유사도 손실)
                # opt.lambda_dssim는 DSSIM 손실의 가중치로, 일반적으로 0.2 정도의 값이다.
                # ssim(image_white, gt_image_white)는 구조적 유사도 지수(0~1 범위)를 반환하므로,
                # (1.0 - ssim(...))을 계산하여 손실로 변환한다 (값이 클수록 손실이 크다).
                # 예: ssim=0.9이면 (1.0-0.9)=0.1이 되어 손실이 작고, ssim=0.5이면 (1.0-0.5)=0.5가 되어 손실이 크다.
                view_loss = view_Ll1 + opt.lambda_dssim * (1.0 - ssim(image_white, gt_image_white))

                # 추가 손실 함수: mode_long=False이고 iteration > 5000일 때만 적용
                # 학습 후반부에 3D 구조 정보(normal, depth)를 활용하여 더 정확한 얼굴 형상을 학습한다.
                if not mode_long and iteration > warm_step + 2000:
                    # Normal 손실: 예측된 표면 노멀과 GT 노멀의 코사인 유사도 기반 손실
                    # viewpoint_cam.talking_dict["normal"]: GT 표면 노멀 맵 [H, W, 3] (각 픽셀의 3D 노멀 벡터)
                    # render_pkg["normal"]: 렌더링된 표면 노멀 맵 [H, W, 3]
                    # 두 노멀 벡터의 내적을 계산하면 코사인 유사도를 얻을 수 있다 (내적 = |a||b|cos(θ), 정규화된 벡터면 cos(θ))
                    # (1 - normal_gt * normal_pred).sum(0)는 각 픽셀에서 1 - cos(θ)를 계산하여 [H, W] 형태가 된다.
                    # head_mask^mouth_mask는 XOR 연산으로, head_mask에서 mouth_mask를 제외한 영역을 의미한다.
                    # .mean()으로 평균을 내어 스칼라 손실 값을 얻고, 0.01 가중치를 곱하여 전체 손실에 추가한다.
                    # 예: iteration=6000일 때, 얼굴과 머리카락 영역(입 제외)에서 노멀 벡터가 일치하도록 유도한다.
                    view_loss += 0.01 * (1 - viewpoint_cam.talking_dict["normal"].cuda() * render_pkg["normal"]).sum(0)[head_mask^mouth_mask].mean()
                    # 위 loss는 head_mask에서 mouth_mask를 제외한 영역에 대해 적용됨

                    # Depth 손실: 예측된 깊이 맵과 GT 깊이 맵의 차이에 대한 L1 손실
                    # opacity_reset_interval이 3000일 때, iteration % 3000 > 100 조건은:
                    # - iteration=3000~3100: False (opacity 리셋 직후 100 step 동안 depth loss 비활성화)
                    # - iteration=3101~6000: True (나머지 구간에서는 depth loss 활성화)
                    # - iteration=6000~6100: False (다시 opacity 리셋 직후 100 step 동안 비활성화)
                    # 이렇게 하는 이유는 opacity 리셋 직후에는 가우시안 구조가 불안정하므로 depth loss를 잠시 비활성화하기 위함이다.
                    if iteration % opt.opacity_reset_interval > 100:
                        # render_pkg["depth"]는 [1, H, W] 형태이므로 [0]으로 첫 번째 채널을 선택한다.
                        # 예: depth가 [1, 512, 512]일 때, depth[0]은 [512, 512] 형태가 된다.
                        depth = render_pkg["depth"][0]
                        # GT 깊이 맵을 GPU로 이동시킨다.
                        depth_mono = viewpoint_cam.talking_dict['depth'].cuda()
                        # normalize 함수는 깊이 맵을 평균 0, 표준편차 1로 정규화한다.
                        # normalize(depth): 예측 깊이 맵을 정규화 [512, 512]
                        # normalize(depth_mono): GT 깊이 맵을 정규화 [512, 512]
                        # face_mask^mouth_mask: 얼굴 영역에서 입 영역을 제외한 마스크
                        # .abs().mean(): 절댓값 차이의 평균을 계산하여 L1 손실을 얻는다.
                        # 1e-2 가중치를 곱하여 전체 손실에 추가한다.
                        # 예: iteration=5000일 때, 얼굴 영역(입 제외)에서 깊이 값이 일치하도록 유도한다.
                        view_loss += 1e-2 * (normalize(depth)[face_mask^mouth_mask] - normalize(depth_mono)[face_mask^mouth_mask]).abs().mean()
                        # 위 loss는 face_mask에서 mouth_mask를 제외한 영역에 대해 적용됨
                    
                # mouth_alpha_loss = 1e-2 * (alpha[:,mouth_mask]).mean()
                # if not torch.isnan(mouth_alpha_loss):
                    # loss += mouth_alpha_loss
                # print(alpha[:,mouth_mask], mouth_mask.sum())

                # 학습 후반부(iteration > 3000)에 추가되는 정규화 손실 및 어텐션 손실:
                if iteration > warm_step:
                    # Motion 정규화 손실: UMF와 PMF가 예측한 변위량이 너무 크지 않도록 제한하는 L1 정규화 손실
                    # 이 손실들은 모션 네트워크가 과도하게 큰 변위를 예측하는 것을 방지하여 학습 안정성을 높인다.
                    
                    # d_xyz: UMF가 예측한 3D 위치 변위 [N, 3]. 각 가우시안의 위치 변화량을 나타낸다.
                    #        예: d_xyz가 [2000, 3] 형태일 때, d_xyz.abs()는 각 가우시안의 위치 변위 절댓값을 계산하고,
                    #        .mean()으로 모든 가우시안의 평균 변위를 계산한다.
                    #        1e-5 가중치로 작은 정규화 손실을 추가하여 변위가 과도하게 커지지 않도록 유도한다.
                    view_loss += 1e-5 * (render_pkg['motion']['d_xyz'].abs()).mean()
                    
                    # d_rot: UMF가 예측한 회전 변위 [N, 4] (quaternion). 각 가우시안의 회전 변화량을 나타낸다.
                    #        예: d_rot이 [2000, 4] 형태일 때, d_rot.abs().mean()은 모든 가우시안의 회전 변위 절댓값의 평균을 계산한다.
                    #        quaternion은 4차원 벡터로 회전을 표현하므로, 각 성분의 절댓값을 계산하여 회전 변화량을 측정한다.
                    view_loss += 1e-5 * (render_pkg['motion']['d_rot'].abs()).mean()
                    
                    # d_opa: UMF가 예측한 불투명도 변위 [N, 1]. 각 가우시안의 투명도 변화량을 나타낸다.
                    #        예: d_opa가 [2000, 1] 형태일 때, d_opa.abs().mean()은 모든 가우시안의 불투명도 변위 절댓값의 평균을 계산한다.
                    #        불투명도가 급격히 변하면 렌더링 결과가 불안정해질 수 있으므로, 작은 정규화 손실로 제한한다.
                    view_loss += 1e-5 * (render_pkg['motion']['d_opa'].abs()).mean()
                    
                    # d_scale: UMF가 예측한 스케일 변위 [N, 3]. 각 가우시안의 크기 변화량을 나타낸다.
                    #          예: d_scale이 [2000, 3] 형태일 때, d_scale.abs().mean()은 모든 가우시안의 스케일 변위 절댓값의 평균을 계산한다.
                    #          가우시안의 크기가 급격히 변하면 렌더링 품질이 저하될 수 있으므로, 작은 정규화 손실로 제한한다.
                    view_loss += 1e-5 * (render_pkg['motion']['d_scale'].abs()).mean()
                    
                    # p_xyz: PMF가 예측한 개인화된 위치 변위 [N, 3]. 캐노니컬 가우시안과 UMF 간의 스케일 보정을 위한 변위를 나타낸다.
                    #        예: p_xyz가 [2000, 3] 형태일 때, p_xyz.abs().mean()은 모든 가우시안의 개인화된 위치 변위 절댓값의 평균을 계산한다.
                    #        이 변위는 얼굴 크기 차이를 보정하기 위한 것이므로, 과도하게 크면 안 되므로 작은 정규화 손실로 제한한다.
                    view_loss += 1e-5 * (render_pkg['p_motion']['p_xyz'].abs()).mean()

                    # Alpha 마스크 손실: 렌더링된 alpha(불투명도)가 head_mask와 일치하도록 유도하는 손실
                    # alpha는 [1, H, W] 형태의 불투명도 맵이고, head_mask는 [H, W] 형태의 불리언 마스크이다.
                    # (1-alpha) * head_mask: head_mask 영역에서는 alpha가 1에 가까워야 함 (불투명해야 함)
                    #                        예: alpha[0, i, j]=0.9이고 head_mask[i, j]=True이면 (1-0.9)*1 = 0.1로 손실이 작고,
                    #                            alpha[0, i, j]=0.5이고 head_mask[i, j]=True이면 (1-0.5)*1 = 0.5로 손실이 크다.
                    # alpha * ~head_mask: head_mask 외부 영역에서는 alpha가 0에 가까워야 함 (투명해야 함)
                    #                     예: alpha[0, i, j]=0.1이고 head_mask[i, j]=False이면 0.1*1 = 0.1로 손실이 작고,
                    #                         alpha[0, i, j]=0.5이고 head_mask[i, j]=False이면 0.5*1 = 0.5로 손실이 크다.
                    # .mean()으로 두 항의 평균을 계산하고, 1e-3 가중치를 곱하여 전체 손실에 추가한다.
                    # 이 손실은 가우시안이 얼굴/머리 영역에만 존재하고 배경에는 존재하지 않도록 유도한다.
                    view_loss += 1e-3 * (((1-alpha) * head_mask).mean() + (alpha * ~head_mask).mean())

                    # 어텐션 손실: 모션 네트워크의 어텐션 맵이 올바른 영역에 집중하도록 유도하는 손실
                    # render_pkg["attn"]은 [3, H, W] 형태의 어텐션 맵으로:
                    # - attn[0]: audio attention (오디오 피처가 각 가우시안에 미치는 영향) [H, W]
                    # - attn[1]: eye attention (표정 피처가 각 가우시안에 미치는 영향) [H, W]
                    # - attn[2]: 사용되지 않음 (zeros)
                    # 어텐션 값이 높을수록 해당 픽셀의 가우시안이 오디오/표정 변화에 더 많이 반응한다는 의미이다.
                    
                    # 입술 영역 좌표를 가져온다.
                    [xmin, xmax, ymin, ymax] = viewpoint_cam.talking_dict['lips_rect']
                    # 입술 영역 어텐션 손실: 입술 영역에서 eye attention(표정 어텐션)은 낮아야 함.
                    # 그러나 입술의 모양은 얼굴 표정이 아닌 오디오에 따른 움직임에 더 큰 영향을 받아야 하니, 입술 영역에 부여된 얼굴 표정 기반 어텐션은 작아져야 한다.
                    # 예: attn[1, xmin:xmax, ymin:ymax]는 입술 영역의 eye attention 값을 선택하고 [ymax-ymin, xmax-xmin] 형태가 되며,
                    #     .mean()으로 평균을 계산하여 입술 영역에서 어텐션이 작아지도록 유도한다.
                    #     1e-4 가중치로 작은 손실을 추가하여 입술 영역의 어텐션을 높인다.
                    # 즉, 얼굴 표정이 입술 모양이 미치는 영향 (어텐션)을 줄이기 위해, 이 값에 가중치를 부여한 뒤 loss에 추가하는 것이다.
                    view_loss += 1e-4 * (render_pkg["attn"][1, xmin:xmax, ymin:ymax]).mean()
                    
                    # 머리카락 영역 어텐션 손실: hair_mask_iter가 False일 때만 적용 (머리카락 학습 중일 때)
                    # 머리카락은 오디오나 표정 변화에 크게 반응하지 않아야 하므로, 어텐션이 낮아야 한다.
                    if not hair_mask_iter:
                        # 머리카락 영역은 audio attention, eye attention 모두에서 그 값이 낮아야 한다.
                        # 예: attn[1][hair_mask]는 머리카락 영역의 eye attention 값을 선택하고,
                        #     hair_mask가 [512, 512] 불리언 마스크일 때, True인 위치의 어텐션 값들만 선택된다.
                        #     .mean()으로 평균을 계산하여 머리카락 영역에서 어텐션이 낮도록 유도한다.
                        #     어텐션이 낮으면 손실이 작으므로, 어텐션을 낮추는 방향으로 학습이 진행된다.
                        view_loss += 1e-4 * (render_pkg["attn"][1][hair_mask]).mean()
                        # 머리카락 영역에서 audio attention(오디오 어텐션)이 낮아야 함
                        # 예: attn[0][hair_mask]는 머리카락 영역의 audio attention 값을 선택하고,
                        #     .mean()으로 평균을 계산하여 머리카락 영역에서 어텐션이 낮도록 유도한다.
                        #     머리카락은 오디오 변화에 반응하지 않아야 하므로, 오디오 어텐션도 낮아야 한다.
                        view_loss += 1e-4 * (render_pkg["attn"][0][hair_mask]).mean()

                    # 주석 처리된 코드: 입술 영역에 대한 L2 손실 (디버깅용으로 보임)
                    # loss += l2_loss(image_white[:, xmin:xmax, ymin:ymax], image_white[:, xmin:xmax, ymin:ymax])
                    

                # LPIPS 손실 계산을 위해 이미지를 복사한다.
                # image_t와 gt_image_t는 이후 LPIPS 손실 계산에서 사용되며, 입술 영역을 마스킹한 후 패치 단위로 손실을 계산한다.
                image_t = image_white.clone()
                gt_image_t = gt_image_white.clone()


            else: # 해당 없음 (iteration >= bg_iter)
                # with real bg
                image = image_white - background[:, None, None] * (1.0 - alpha) + viewpoint_cam.background.cuda() / 255.0 * (1.0 - alpha)
                gt_image = viewpoint_cam.original_image.cuda() / 255.0

                view_Ll1 = l1_loss(image, gt_image)
                view_loss = view_Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

                image_t = image.clone()
                gt_image_t = gt_image.clone()
                # mode_long이 True일 때만 아래 loss 계산.
                # lpips_start_iter는 7500으로 설정되어 있음.
                # 즉, iteration > 7500일 때만 아래의 LPIPS 기반 patch loss가 적용된다.
            if iteration > lpips_start_iter:   
                # 현재 카메라의 입술 영역 좌표를 가져온다.
                [xmin, xmax, ymin, ymax] = viewpoint_cam.talking_dict['lips_rect']
                # mode_long이 True일 때, 입술 영역에 대해 LPIPS loss를 0.01 가중치로 추가한다.
                if mode_long:
                    view_loss += 0.01 * lpips_criterion(
                        image_t.clone()[:, xmin:xmax, ymin:ymax] * 2 - 1, 
                        gt_image_t.clone()[:, xmin:xmax, ymin:ymax] * 2 - 1
                    ).mean()
                # 입술 영역을 배경색으로 덮어서 이후 patch loss 계산에서 입술 영향이 없도록 한다.
                image_t[:, xmin:xmax, ymin:ymax] = background[:, None, None]
                gt_image_t[:, xmin:xmax, ymin:ymax] = background[:, None, None]
                # patch 크기를 64~96 사이에서 랜덤하게 정한다.
                patch_size = random.randint(32, 48) * 2
                # mode_long이 True일 때, patchify된 이미지에 대해 LPIPS loss를 0.2 가중치로 추가한다.
                if mode_long:
                    view_loss += 0.2 * lpips_criterion(
                        patchify(image_t[None, ...] * 2 - 1, patch_size), 
                        patchify(gt_image_t[None, ...] * 2 - 1, patch_size)
                    ).mean()
                # patchify된 이미지에 대해 LPIPS loss를 0.01 가중치로 추가한다.
                view_loss += 0.01 * lpips_criterion(
                    patchify(image_t[None, ...] * 2 - 1, patch_size), 
                    patchify(gt_image_t[None, ...] * 2 - 1, patch_size)
                ).mean()
                # (주석) 전체 이미지에 대해 LPIPS loss를 0.5 가중치로 추가할 수도 있으나, 현재는 사용하지 않음.
                # view_loss += 0.5 * lpips_criterion(image_t[None, ...] * 2 - 1, gt_image_t[None, ...] * 2 - 1).mean()
        # 현재 뷰의 손실을 누적 (if/else 블록 밖에서 공통으로 처리)
        
            if total_loss is None:
                total_loss = view_loss
                total_Ll1 = view_Ll1
            else:
                total_loss = total_loss + view_loss
                total_Ll1 = total_Ll1 + view_Ll1
            
            # densification을 위해 마지막 뷰의 정보 사용 (viewspace_point_tensor, visibility_filter, radii)
            # 여러 뷰가 있으면 마지막 뷰의 정보를 사용
        
        # 멀티뷰 루프 종료 후, 누적된 손실을 평균으로 나누기
        num_views = len(view_order)
        if num_views > 1:
            loss = total_loss / num_views
            Ll1 = total_Ll1 / num_views
        else:
            loss = total_loss
            Ll1 = total_Ll1

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            # 로그 기록을 위한 EMA(Exponential Moving Average) 손실 값을 업데이트
            # 현재 손실(`loss.item()`)에 $0.4$의 가중치를, 이전 EMA 손실(`ema_loss_for_log`)에 $0.6$의 가중치를 부여하여 새로운 EMA 손실을 계산
            # `loss.item()`은 손실 텐서에서 스칼라 값을 추출하는 함수
            if iteration % 10 == 0: # 현재 반복 횟수(`iteration`)가 10의 배수일 경우
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{5}f}", "Mouth": f"{mouth_lb:.{1}f}-{mouth_ub:.{1}f}"}) # , "AU25": f"{au_lb:.{1}f}-{au_ub:.{1}f}"
                # 진행 바(`progress_bar`)의 후행 메시지를 업데이트
                # 현재 EMA 손실 값과 `mouth_lb`, `mouth_ub` 값을 소수점 이하 자릿수를 지정하여 표시
                progress_bar.update(10) # 진행 바를 10만큼 업데이트 (매 10회 반복마다 10씩 업데이트되므로 실제 반복 횟수와 일치)
            if iteration == opt.iterations: # 현재 반복 횟수(`iteration`)가 총 반복 횟수(`opt.iterations`)와 동일한지 확인
                progress_bar.close() # 조건이 참이면 진행 바를 닫는다. (학습 완료)

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, motion_net, render if iteration < warm_step else render_motion, (pipe, background))
            # `training_report` 함수를 호출하여 학습 관련 지표들을 로깅
            # 인자로는 `tb_writer`(TensorBoard 라이터), 현재 반복 횟수(`iteration`), L1 손실(`Ll1`), 총 손실(`loss`), L1 손실 함수 객체, 각 반복에 소요된 시간(`iter_start.elapsed_time(iter_end)`), 테스트 반복 횟수(`testing_iterations`),
            # 장면 객체(`scene`), 모션 네트워크(`motion_net`), 현재 학습 단계에 맞는 렌더링 함수(`render` 또는 `render_motion`), 그리고 렌더링 파이프라인 설정(`pipe`, `background`)

            if (iteration in saving_iterations): # 현재 반복 횟수(`iteration`)가 `saving_iterations` 리스트의 원소에 해당한다면
                print("\n[ITER {}] Saving Gaussians".format(iteration)) # 가우시안 모델을 저장한다는 메시지를 출력
                scene.save(str(iteration)+'_face')
                
                # deformed 가우시안도 함께 저장
                # scene.save가 저장하는 경로에 deformed 가우시안도 저장
                point_cloud_path = os.path.join(scene.model_path, "point_cloud/iteration_{}_face".format(iteration))
                os.makedirs(point_cloud_path, exist_ok=True)
                
                if scene.multiview_data is not None:
                    # 멀티뷰인 경우: 선택된 프레임의 모든 뷰에 대해 deformed 가우시안 저장
                    multiview_cameras = scene.getMultiViewFrame(selected_img_id)
                    # front 뷰를 먼저 처리하기 위해 정렬
                    view_order = sorted(multiview_cameras.keys())
                    if 'front' in view_order:
                        view_order.remove('front')
                        view_order.insert(0, 'front')
                    
                    with torch.no_grad():
                        for view_name, view_cam in multiview_cameras.items():
                            # 각 뷰에 대해 render_motion을 호출하여 motion 정보 획득
                            render_pkg = render_motion(view_cam, gaussians, motion_net, pipe, background, return_attn=False, personalized=False, align=True)
                            # deformed 가우시안 저장
                            # render_pkg['motion']에서 예측된 변위(d_xyz, d_scale, d_rot)를 원본 가우시안 속성에 더해서 deformed 상태를 만든다
                            # 각 뷰별로 다른 파일명 사용 (예: deformed_front.ply, deformed_left_30.ply)
                            deformed_ply_path = os.path.join(point_cloud_path, "deformed_{}.ply".format(view_name))
                            gaussians.save_deformed_ply(
                                gaussians.get_xyz + render_pkg['motion']['d_xyz'],
                                gaussians._scaling + render_pkg['motion']['d_scale'],
                                gaussians._rotation + render_pkg['motion']['d_rot'],
                                deformed_ply_path
                            )
                            print("[ITER {}] Deformed Gaussians saved to {} (view: {})".format(iteration, deformed_ply_path, view_name))
                else:
                    # 단일 뷰인 경우: 현재 카메라에 대해 deformed 가우시안 저장
                    with torch.no_grad():
                        # render_motion을 호출하여 motion 정보 획득
                        render_pkg = render_motion(viewpoint_cam, gaussians, motion_net, pipe, background, return_attn=False, personalized=False, align=True)
                        # deformed 가우시안 저장
                        # render_pkg['motion']에서 예측된 변위(d_xyz, d_scale, d_rot)를 원본 가우시안 속성에 더해서 deformed 상태를 만든다
                        deformed_ply_path = os.path.join(point_cloud_path, "deformed.ply")
                        gaussians.save_deformed_ply(
                            gaussians.get_xyz + render_pkg['motion']['d_xyz'],
                            gaussians._scaling + render_pkg['motion']['d_scale'],
                            gaussians._rotation + render_pkg['motion']['d_rot'],
                            deformed_ply_path
                        )
                        print("[ITER {}] Deformed Gaussians saved to {}".format(iteration, deformed_ply_path))

            if (iteration in checkpoint_iterations): # 현재 반복 횟수(`iteration`)가 `checkpoint_iterations` 리스트의 원소에 해당한다면
                print("\n[ITER {}] Saving Checkpoint".format(iteration)) # 체크포인트를 저장한다는 메시지를 출력
                ckpt = (gaussians.capture(), motion_net.state_dict(), motion_optimizer.state_dict(), iteration)
                # UMF인 `motion_net`의 상태 딕셔너리, `motion_optimizer`의 상태 딕셔너리, 현재 반복 횟수(`iteration`)를 묶어 `ckpt` 변수에 저장
                # `state_dict()`는 모델이나 옵티마이저의 학습 가능한 파라미터들을 딕셔너리 형태로 반환
                torch.save(ckpt, scene.model_path + "/chkpnt_face_" + str(iteration) + ".pth")
                # `ckpt`를 `.pth` 확장자를 가진 파일로 저장
                # `dataset.model_path`는 모델이 저장될 경로이며, `chkpnt_face_latest`는 최신 체크포인트 파일 이름을 나타냄
                torch.save(ckpt, scene.model_path + "/chkpnt_face_latest" + ".pth")


            # Densification: 가우시안의 밀도 조절 과정 (가우시안 추가/제거)
            # iteration < opt.densify_until_iter (9000)일 때만 densification을 수행한다.
            # 학습 후반부에는 가우시안 구조를 고정하고 파라미터만 최적화한다.
            if iteration < opt.densify_until_iter: # 9000 보다 작은 경우
                # max_radii2D 업데이트: 각 가우시안의 2D 스크린 공간에서의 최대 반지름을 추적한다.
                # visibility_filter는 현재 화면에 보이는 가우시안만을 나타내는 불리언 마스크이다.
                # radii는 현재 렌더링에서 각 가우시안의 2D 반지름이다.
                # torch.max를 사용하여 기존 max_radii2D와 현재 radii 중 더 큰 값을 저장한다.
                # 예: max_radii2D[visibility_filter]가 [5.0, 3.0, 7.0]이고 radii[visibility_filter]가 [6.0, 2.0, 7.0]이면,
                #     결과는 [6.0, 3.0, 7.0]이 되어 각 가우시안의 최대 반지름을 업데이트한다.
                # 이 값은 나중에 너무 큰 가우시안을 제거할 때 사용된다.
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                
                # densification 통계 추가: 가우시안의 위치에 대한 gradient 정보를 누적한다.
                # viewspace_point_tensor는 뷰스페이스 상의 포인트 좌표로, gradient 추적이 활성화되어 있다.
                # add_densification_stats는 viewspace_point_tensor의 gradient를 사용하여 각 가우시안의 위치 변화량을 추적한다.
                # 이 정보는 나중에 densify_and_prune에서 어떤 가우시안을 분할하거나 복제할지 결정하는 데 사용된다.
                # visibility_filter에 해당하는 가우시안만 업데이트하여, 보이지 않는 가우시안은 제외한다.
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # densification 실행: 일정 간격마다 가우시안을 분할/복제하고 불필요한 가우시안을 제거한다.
                # iteration > opt.densify_from_iter (500)이고 iteration이 opt.densification_interval (100)의 배수일 때 실행된다.
                # 예: iteration=600, 700, 800, ... 8900일 때 실행된다.
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # size_threshold 설정: 너무 큰 가우시안을 제거하기 위한 임계값이다.
                    # iteration > opt.opacity_reset_interval (3000)일 때만 size_threshold=20을 사용하고,
                    # 그 이전에는 None으로 설정하여 크기 제한을 적용하지 않는다.
                    # 예: iteration=5000일 때 size_threshold=20이 되어, 반지름이 20픽셀보다 큰 가우시안을 제거한다.
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    
                    # densify_and_prune 실행: 가우시안을 밀도화하고 제거한다.
                    # opt.densify_grad_threshold: gradient가 이 값보다 큰 가우시안을 분할/복제한다.
                    # 0.05 + 0.25 * iteration / opt.densify_until_iter: 최소 opacity 임계값으로, 학습이 진행될수록 증가한다.
                    #    예: iteration=1000일 때 0.05 + 0.25 * 1000/9000 = 0.078,
                    #        iteration=8000일 때 0.05 + 0.25 * 8000/9000 = 0.272
                    #    opacity가 이 값보다 작은 가우시안은 제거된다.
                    # scene.cameras_extent: 카메라 범위로, 가우시안의 크기를 결정하는 데 사용된다.
                    # size_threshold: 너무 큰 가우시안을 제거하기 위한 임계값 (위에서 설정됨)
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.05 + 0.25 * iteration / opt.densify_until_iter, scene.cameras_extent, size_threshold)
                
                # Opacity 리셋: 가우시안의 불투명도를 주기적으로 초기화한다.
                # mode_long이 False일 때만 실행된다 (현재는 항상 실행됨).
                if not mode_long:  # mode_long=False이므로 항상 True
                    # opacity 리셋 조건:
                    # 1. iteration이 opt.opacity_reset_interval (3000)의 배수일 때
                    # 2. 흰색 배경을 사용하고 iteration이 opt.densify_from_iter (500)일 때
                    # 예: iteration=3000, 6000, 9000일 때 opacity가 리셋된다.
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        # 모든 가우시안의 opacity를 초기값으로 리셋한다.
                        # 이렇게 하면 학습 중 opacity가 잘못 수렴한 경우를 방지하고, 가우시안의 가시성을 재조정할 수 있다.
                        gaussians.reset_opacity()
            
            # 배경 색상 기반 가우시안 제거 (bg prune)
            # iteration > opt.densify_from_iter (500)이고 iteration이 opt.densification_interval (100)의 배수일 때 실행된다.
            # 배경색(녹색)과 유사한 색상을 가진 가우시안을 제거하여 배경 노이즈를 줄인다.
            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                # Spherical Harmonics (SH) 유틸리티 함수를 임포트한다.
                # SH는 방향에 따른 색상 변화를 표현하는 데 사용된다.
                from utils.sh_utils import eval_sh

                # SH 계수를 뷰 방향에 맞게 변환한다.
                # gaussians.get_features는 [N, 3, (max_sh_degree+1)^2] 형태의 SH 계수이다.
                # transpose(1, 2)로 [N, (max_sh_degree+1)^2, 3]로 변환하고,
                # view(-1, 3, (gaussians.max_sh_degree+1)**2)로 [N, 3, (max_sh_degree+1)^2] 형태로 재구성한다.
                # 예: max_sh_degree=1일 때 (1+1)^2=4이므로, [N, 3, 4] 형태가 된다.
                shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree+1)**2)
                
                # 각 가우시안에서 카메라 중심으로의 방향 벡터를 계산한다.
                # gaussians.get_xyz는 [N, 3] 형태의 가우시안 3D 위치이다.
                # viewpoint_cam.camera_center는 [3] 형태의 카메라 중심 좌표이다.
                # repeat(gaussians.get_features.shape[0], 1)로 [N, 3] 형태로 복제한다.
                # dir_pp는 [N, 3] 형태로, 각 가우시안에서 카메라로의 방향 벡터이다.
                dir_pp = (gaussians.get_xyz - viewpoint_cam.camera_center.repeat(gaussians.get_features.shape[0], 1))
                
                # 방향 벡터를 정규화하여 단위 벡터로 만든다.
                # dir_pp.norm(dim=1, keepdim=True)는 각 가우시안의 방향 벡터의 길이를 계산한다 [N, 1].
                # dir_pp를 이 길이로 나누어 정규화한다.
                # 예: dir_pp=[3, 4, 0]이면 norm=5이고, 정규화 후 [0.6, 0.8, 0]이 된다.
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                
                # SH 계수를 사용하여 뷰 방향에 따른 RGB 색상을 계산한다.
                # gaussians.active_sh_degree: 현재 활성화된 SH 차수 (학습 중 증가함)
                # shs_view: SH 계수 [N, 3, (max_sh_degree+1)^2]
                # dir_pp_normalized: 정규화된 방향 벡터 [N, 3]
                # eval_sh는 SH 계수와 방향 벡터를 사용하여 RGB 색상을 계산한다.
                # sh2rgb는 [N, 3] 형태로, 각 가우시안의 RGB 색상이다.
                sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
                
                # 색상을 0~1 범위로 클리핑한다.
                # sh2rgb + 0.5: SH 색상을 중앙값 기준으로 조정 (SH는 -0.5~0.5 범위를 가정)
                # torch.clamp_min(..., 0.0): 음수 값을 0으로 클리핑
                # colors_precomp는 [N, 3] 형태로, 각 가우시안의 RGB 색상 (0~1 범위)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

                # 배경색(녹색) 마스크 생성: 배경색과 유사한 색상을 가진 가우시안을 식별한다.
                # 배경색은 RGB(0, 1, 0) = (0, 255, 0) = 녹색이다.
                # 조건: R < 30/255, G > 225/255, B < 30/255
                # 즉, 녹색 성분이 매우 높고 빨강/파랑 성분이 낮은 가우시안을 배경으로 간주한다.
                # bg_color_mask는 [N] 형태의 불리언 마스크로, True인 가우시안이 배경색과 유사하다는 의미이다.
                # 예: colors_precomp[i] = [0.05, 0.95, 0.03]이면 True가 되어 배경 가우시안으로 식별된다.
                bg_color_mask = (colors_precomp[..., 0] < 30/255) * (colors_precomp[..., 1] > 225/255) * (colors_precomp[..., 2] < 30/255)
                
                # 배경색 가우시안 제거: bg_color_mask에 해당하는 가우시안을 제거한다.
                # squeeze()는 불필요한 차원을 제거하여 [N] 형태로 만든다.
                # prune_points는 마스크가 True인 가우시안을 가우시안 모델에서 완전히 제거한다.
                gaussians.prune_points(bg_color_mask.squeeze())
                
                # 깊이 기반 가우시안 제거: mode_long이 False일 때만 실행된다.
                # z축(깊이 방향) 좌표가 -0.07보다 작은(너무 뒤에 있는) 가우시안을 제거한다.
                if not mode_long:
                    # gaussians.get_xyz[:, -1]은 모든 가우시안의 z좌표만 추출한다 [N].
                    # (gaussians.get_xyz[:, -1] < -0.07)는 z좌표가 -0.07보다 작은 가우시안에 대해 True인 불리언 마스크를 만든다.
                    # squeeze()는 불필요한 차원을 제거한다.
                    # 예: z좌표가 -0.1인 가우시안은 True가 되어 제거된다.
                    gaussians.prune_points((gaussians.get_xyz[:, -1] < -0.07).squeeze())

            # Optimizer step
            if iteration < opt.iterations:
                motion_optimizer.step()
                gaussians.optimizer.step()

                motion_optimizer.zero_grad()
                gaussians.optimizer.zero_grad(set_to_none = True)

                scheduler.step()



def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, motion_net, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(5, 100, 5)]}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):

                    if renderFunc is render:
                        render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    else:
                        render_pkg = renderFunc(viewpoint, scene.gaussians, motion_net, return_attn=True, frame_idx=0, align=True, *renderArgs)

                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    alpha = render_pkg["alpha"]
                    normal = render_pkg["normal"] * 0.5 + 0.5
                    depth = render_pkg["depth"] * alpha + (render_pkg["depth"] * alpha).mean() * (1 - alpha)
                    depth = (depth - depth.min()) / (depth.max() - depth.min())
                    
                    depth_normal = depth_to_normal(viewpoint, render_pkg["depth"]).permute(2,0,1)
                    depth_normal = depth_normal * (alpha).detach()
                    depth_normal = depth_normal * 0.5 + 0.5
                    
                    image = image - renderArgs[1][:, None, None] * (1.0 - alpha) + viewpoint.background.cuda() / 255.0 * (1.0 - alpha)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda") / 255.0, 0.0, 1.0)
                    
                    mouth_mask = torch.as_tensor(viewpoint.talking_dict["mouth_mask"]).cuda()
                    max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
                    mouth_mask_post = (-max_pool(-max_pool(mouth_mask[None].float())))[0].bool()
                    
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/mouth_mask_post".format(viewpoint.image_name), (~mouth_mask_post * gt_image)[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/mouth_mask".format(viewpoint.image_name), (~mouth_mask[None] * gt_image)[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/normal".format(viewpoint.image_name), normal[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/normal_from_depth".format(viewpoint.image_name), depth_normal[None], global_step=iteration)
                        # tb_writer.add_images(config['name'] + "_view_{}/normal_mono".format(viewpoint.image_name), (viewpoint.talking_dict["normal"]*0.5+0.5)[None], global_step=iteration)
                        # if config['name']=="train":
                        #     depth_mono = 1.0 - viewpoint.talking_dict['depth'].cuda()
                        #     tb_writer.add_images(config['name'] + "_view_{}/depth_mono".format(viewpoint.image_name), depth_mono[None, None], global_step=iteration)

                        if renderFunc is not render:
                            tb_writer.add_images(config['name'] + "_view_{}/attn_a".format(viewpoint.image_name), (render_pkg["attn"][0] / render_pkg["attn"][0].max())[None, None], global_step=iteration)  
                            tb_writer.add_images(config['name'] + "_view_{}/attn_e".format(viewpoint.image_name), (render_pkg["attn"][1] / render_pkg["attn"][1].max())[None, None], global_step=iteration)  

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1") # --ip 인자를 추가, 기본값은 127.0.0.1 (로컬호스트)
    parser.add_argument('--port', type=int, default=6009) # --port 인자를 추가, 기본값은 6009
    parser.add_argument('--debug_from', type=int, default=-1) # --debug_from 인자를 추가, 기본값을 -1로 두어 훈련 내내 디버깅을 하지 않음. 모델 훈련 코드에서, iteration - 1부터 디버깅을 시작하도록 설정해둠.
    parser.add_argument('--detect_anomaly', action='store_true', default=False) # 이상현상을 탐지할지 여부. action='store_true'는 이 인자가 명령줄에 있으면 True, 없으면 False로 세팅한다는 뜻.
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[]) # 몇 번째 이터레이션마다 테스트를 할지를 정함. nargs="+"는 인자를 하나 이상 받을 수 있다는 뜻. 따라서 자료형도 리스트 []로 되어있음.
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[2000, 4000, 6000, 8000, 10000]) # 몇 번째 이터레이션마다 가우시안 모델을 포인트 클라우드(.ply)로 저장할지 정함. 마찬가지로 리스트 형태로 되어있음.
    parser.add_argument("--quiet", action="store_true") # 출력을 최소화할지 여부. action='store_true'이므로 인자를 명령줄에 적으면 활성화 되는 속성.
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[]) # 가우시안 모델 상태 (gaussians.capture()), Motion network 상태 (motion_net.state_dict()), Optimizer 상태 (motion_optimizer.state_dict()), 현재 이터레이션 번호를 묶어서 .pth 파일로 저장.
    # 아래의 start_checkpoint 인자에 해당 pth를 적어두면, 해당 이터레이션부터 학습을 이어서 진행할 수 있음.
    parser.add_argument("--start_checkpoint", type=str, default = None) # fine-tuning 과정 (train_face.py 자체 코드)에서 이어서 fine-tuning을 진행할 경우 사용.
    parser.add_argument("--long", action='store_true', default=False) # 더 오래 학습할지 여부. 디폴트는 False.
    # False일 때:
    # max_sh_degree = 1
    # normal/depth loss 추가
    # opacity reset 추가
    # z축 기반 포인트 제거 수행
    
    # True일 때:
    # 입술 영역과 패치 LPIPS loss 추가
    parser.add_argument("--pretrain_path", type=str, default = None) # pre-train 된 모델의 체크포인트 경로 'output/pretrain_ave/chkpnt_ema_face_latest.pth'
    args = parser.parse_args(sys.argv[1:]) # 명령줄 인자를 파싱하여 args에 저장
    args.save_iterations.append(args.iterations) # 아까 빈 리스트로 선언한 save_iterations에 디폴트 이터레이션 값인 10,000을 추가
    
    print("Optimizing " + args.model_path) # 저장될 모델 경로 출력.

    # Initialize system state (RNG)
    safe_state(args.quiet) # quiet를 False로 줘서 모든 메시지 출력.

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly) # autograd anomaly detection을 옵션은 default = False로 비활성화.
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.long, args.pretrain_path)

    # All done
    print("\nTraining complete.")
