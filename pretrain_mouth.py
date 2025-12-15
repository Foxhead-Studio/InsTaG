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
from torch_ema import ExponentialMovingAverage
from random import randint
from utils.loss_utils import l1_loss, l2_loss, patchify, ssim
from gaussian_renderer import render, render_motion, render_motion_mouth_con
import sys, copy
from scene_pretrain import Scene, GaussianModel, MouthMotionNetwork, MotionNetwork
from utils.general_utils import safe_state
import lpips
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from tensorboardX import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    # data_list = ["macron", "Shaheen", "may", "Jae-in", "Obama1"]
    data_list = ["macron"]

    testing_iterations = [i * len(data_list) for i in range(0, opt.iterations + 1, 2000)]
    checkpoint_iterations =  saving_iterations = [i * len(data_list) for i in range(0, opt.iterations + 1, 5000)] + [opt.iterations * len(data_list)]

    # vars
    warm_step = 3000 * len(data_list) # 3000 (얼굴 훈련은 1000이었는데, 좀 더 길다.)
    opt.densify_until_iter = (opt.iterations - 1000) * len(data_list) # 30000 - 1000 = 29000
    lpips_start_iter = 999999999999 # 999999999999
    motion_stop_iter = opt.iterations * len(data_list) # 30000 * 1 = 30000
    mouth_select_iter = (opt.iterations - 10000) * len(data_list) # (30000 - 10000) * 1 = 29000 (좀 더 길다.)
    p_motion_start_iter = 0
    mouth_step = 1 / max(mouth_select_iter, 1) # 5e-05
    select_interval = 7
    # hair_mask_interval이 존재하지 않음.

    opt.iterations *= len(data_list)

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    
    # shared_audio_net이 존재하지 않음.

    scene_list = []
    for data_name in data_list:  
        gaussians = GaussianModel(dataset)
        _dataset = copy.deepcopy(dataset)
        _dataset.source_path = os.path.join(dataset.source_path, data_name)
        _dataset.model_path = os.path.join(dataset.model_path, data_name)
        
        os.makedirs(_dataset.model_path, exist_ok = True)
        with open(os.path.join(_dataset.model_path, "cfg_args"), 'w') as cfg_log_f:
            cfg_log_f.write(str(Namespace(**vars(_dataset))))
        
        scene = Scene(_dataset, gaussians)
        scene_list.append(scene)
        gaussians.training_setup(opt)
        
        # 랜덤 초기화 된 가우시안을 입술 중심으로 모으기
        with torch.no_grad(): # 이러한 가우시안의 움직임은 학습에 사용되면 안 되니 torch.no_grad()를 붙임.
            gaussians._xyz /= 2 # [5000, 3]의 랜덤 초기화된 가우시안의 좌표를 2로 나누어 0.5만큼 줄임.
            gaussians._xyz[:,1] -= 0.05 # [5000, 3]의 가우시안의 y좌표를 0.05만큼 낮춤. 이렇게 해서 산재된 가우시안이 입술 쪽에 모이도록 데이터 전처리.
        
        # Face 훈련 때 저장한 PMF 불러오기.
        _dataset_face = copy.deepcopy(dataset)
        _dataset_face.type = "face" # dataset을 복제한 뒤, face로 타입 변경 (원래 args 때문에 mouth 모델로 초기화되어 있었음)
        gaussians_face = GaussianModel(_dataset_face) # 해당 데이터셋을 이용해 얼굴 가우시안 초기화
        (model_params, _, _, _) = torch.load(os.path.join(_dataset.model_path, "chkpnt_face_latest.pth"))
        gaussians_face.restore(model_params, None) # 마크롱 얼굴에 대해 가우시안 값, GridRenderer, PersonalizedMotionNetwork 등을 체크 포인트로부터 받아옴.
        scene.gaussians_2 = gaussians_face # scene의 인자 중 gaussians_2에 마크롱 얼굴 가우시안을 저장.

    # face mouth hook이 구현된 입술 모션 네트워크 초기화.
    motion_net = MouthMotionNetwork(args=dataset).cuda()
    motion_optimizer = torch.optim.AdamW(motion_net.get_params(5e-3, 5e-4), betas=(0.9, 0.99), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(motion_optimizer, lambda iter: (0.5 ** (iter / mouth_select_iter)) if iter < mouth_select_iter else 0.1 ** (iter / opt.iterations))
    ema_motion_net = ExponentialMovingAverage(motion_net.parameters(), decay=0.995)
    
    # Face 훈련 때 저장한 UMF 불러오기.
    with torch.no_grad():
        motion_net_face = MotionNetwork(args=dataset).cuda()
        (motion_params, _, _) = torch.load(os.path.join(dataset.model_path, "chkpnt_ema_face_latest.pth"))
        # gaussians.restore(model_params, opt)
        motion_net_face.load_state_dict(motion_params)

    lpips_criterion = lpips.LPIPS(net='alex').eval().cuda()

    bg_color = [0, 1, 0] # if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), ascii=True, dynamic_ncols=True, desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()
        
        cur_scene_idx = randint(0, len(scene_list)-1) # 매 이터레이션마다 마크롱, 오바마, 문제인 중 하나를 선택하여 학습. (인물 순서대로가 아님!)
        scene = scene_list[cur_scene_idx]
        gaussians = scene.gaussians
        
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        # if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # find a big mouth
        # face 사전학습에서는 입벌림과 눈 깜빡임 AU를 사용했다면, 여기서는 입벌림 AU를 사용.
        au_global_lb = viewpoint_cam.talking_dict['au25'][1] # [0.96, 0.07, 0.46, 0.93, 1.74] 중 두번째 값: 전체 프레임에서 입이 벌어진 최소값
        au_global_ub = viewpoint_cam.talking_dict['au25'][4] # [0.96, 0.07, 0.46, 0.93, 1.74] 중 다섯번째 값: 전체 프레임에서 입이 벌어진 최대값
        au_window = (au_global_ub - au_global_lb) * 0.2 # (1.74 - 0.07) * 0.2 = 0.334

        au_ub = au_global_ub # 1.74 학습이 진행됨에 따라 입 벌림 상한선은 전역 최대 벌림 값으로 고정한다.
        au_lb = au_ub - mouth_step * iteration * (au_global_ub - au_global_lb) # 1.74 - 0.00005 * iteration * (1.74 - 0.07) = 1.7399165
        # 학습 iteration이 증가함에 따라 au_lb (입 벌림 하한선)가 점진적으로 감소한다. 이는 학습이 진행될수록 au25 값이 낮은 (즉, 입이 더 작게 벌어진) 프레임까지 학습에 포함하여, 입의 미세한 닫힘/벌림 움직임을 더 정교하게 학습하도록 유도한다. 
        
        if iteration < warm_step: # 3000 이하
            while viewpoint_cam.talking_dict['au25'][0] < au_global_ub: # 현재 프레임의 입 벌림 정도가 au_global_ub보다 큰 것을 찾을 때까지 반복
                if not viewpoint_stack: # 이 과정은 face 훈련 네트워크와 달리 7 인터벌마다 실행하는 것이 아니라 매번 실행한다.
                    viewpoint_stack = scene.getTrainCameras().copy()
                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1)) # 충분히 큰 입을 샘플링할 때까지 반복

        if warm_step < iteration < mouth_select_iter: # 3000 이상 20000 이하
            if iteration % select_interval == 0: # 여기서부턴 7 인터벌마다
                while viewpoint_cam.talking_dict['au25'][0] < au_lb or viewpoint_cam.talking_dict['au25'][0] > au_ub: # au_lb < 현재 프레임의 입 벌림 정도 < au_ub인 프레임을 찾을 때까지 반복
                    if not viewpoint_stack:
                        viewpoint_stack = scene.getTrainCameras().copy()
                    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1)) # au_lb가 점차 감소하므로, 학습이 진행될수록 더 다양한 (작게 벌린) 입 모양들을 이 단계에서 학습하게 된다.

            while torch.as_tensor(viewpoint_cam.talking_dict["mouth_mask"]).cuda().sum() < 20: # 그러나 위 au25 기반 샘플링 이후, 현재 선택된 viewpoint_cam의 입 마스크(mouth_mask) 픽셀 합이 20보다 큰 프레임을 찾을 때까지 반복
                if not viewpoint_stack:
                    viewpoint_stack = scene.getTrainCameras().copy()
                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
                # 이 조건은 입이 너무 작게 벌어졌거나 거의 닫혀서 의미 있는 입 모션 정보가 적은 프레임을 제외하기 위함이다. 즉, 입 모션 학습에 충분히 유효한 입 모양을 가진 프레임만을 학습에 사용하도록 필터링한다.

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        # if iteration > bg_iter:
        #     # turn to black
        #     bg_color = [0, 0, 0] # if dataset.white_background else [0, 0, 0]
        #     background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        face_mask = torch.as_tensor(viewpoint_cam.talking_dict["face_mask"]).cuda()
        hair_mask = torch.as_tensor(viewpoint_cam.talking_dict["hair_mask"]).cuda()
        mouth_mask = torch.as_tensor(viewpoint_cam.talking_dict["mouth_mask"]).cuda()
        head_mask =  face_mask + hair_mask
        # face 훈련과 달리 head_mask는 사용하지 않음.

        
        [xmin, xmax, ymin, ymax] = viewpoint_cam.talking_dict['lips_rect'] 
        # viewpoint_cam.talking_dict['lips_rect']는 입술이 위치한 사각형 영역의 좌표 [xmin, xmax, ymin, ymax]를 담고 있다.
        # 예를 들어 [100, 150, 200, 250]이면, x축 100~150, y축 200~250 범위가 입술 영역이다.

        lips_mask = torch.zeros_like(mouth_mask)
        # mouth_mask와 동일한 크기의 0(=False)로 채워진 텐서를 만든다.
        # 이 텐서는 입술 영역만 True로 바꿔서 입술만 마스킹하는 용도로 쓴다.

        lips_mask[xmin:xmax, ymin:ymax] = True
        # lips_mask에서 [xmin:xmax, ymin:ymax] 범위에 해당하는 부분만 True(=1)로 바꾼다.
        # 즉, 입술 사각형 영역만 True가 되고 나머지는 False로 남는다.
        # 이 lips_mask는 이후에 입술 부분만 따로 loss를 주거나, 시각화할 때 쓸 수 있다.

        # lips_mask: 입술을 포함하는 직사각형 마스크
        # mouth_mask: 입이 벌어졌을 때, 입 내부만 나타내는 마스크 (직사각형이 아님)

        if iteration < warm_step:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        elif iteration < p_motion_start_iter:
            render_pkg = render_motion_mouth_con(viewpoint_cam, gaussians, motion_net, scene.gaussians_2, motion_net_face, pipe, background)
        else:
            render_pkg = render_motion_mouth_con(viewpoint_cam, gaussians, motion_net, scene.gaussians_2, motion_net_face, pipe, background, personalized=True)
            
        image_green, alpha, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["alpha"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        gt_image  = viewpoint_cam.original_image.cuda() / 255.0
        # `viewpoint_cam` 객체에서 원본 이미지(`original_image`)를 가져와 GPU에 할당하고, 픽셀 값을 0-1 범위로 정규화.
        gt_image_green = gt_image * mouth_mask + background[:, None, None] * ~mouth_mask
        # `gt_image_green`는 원본 이미지(`gt_image`)에 `mouth_mask`를 적용하고, 입술 영역이 아닌 부분(`~mouth_mask`)에는 `background` 색상을 채워 넣어 생성한다.

        if iteration > motion_stop_iter:
            for param in motion_net.parameters():
                param.requires_grad = False
        # if iteration > bg_iter:
        #     gaussians._xyz.requires_grad = False
        #     gaussians._opacity.requires_grad = False
        #     # gaussians._features_dc.requires_grad = False
        #     # gaussians._features_rest.requires_grad = False
        #     gaussians._scaling.requires_grad = False
        #     gaussians._rotation.requires_grad = False
        
        # Loss            
        image_green[:, (lips_mask ^ mouth_mask)] = background[:, None]
        # 렌더링된 이미지(`image_green`)에서 `lips_mask`와 `mouth_mask`의 XOR 연산으로 얻은 영역의 픽셀 값을 `background` 색상으로 채웁니다.
        # 이 연산 `(lips_mask ^ mouth_mask)`은 `lips_mask`와 `mouth_mask` 중 한쪽에만 해당하는 영역을 식별합니다.
        # 즉, 입 내부 픽셀들을 제외하고 배경색으로 채움으로써, 해당 영역의 픽셀은 손실 계산에서 제외되거나, 모델이 특별히 신경 쓰지 않도록 유도합니다.

        Ll1 = l1_loss(image_green, gt_image_green)
        loss = Ll1 + opt.lambda_dssim * (1.0 - ssim(image_green, gt_image_green))
        # 렌더링된 이미지(`image_green`)와 실제 이미지(`gt_image_green`) 사이의 L1 손실을 계산합니다.
        # L1 손실은 두 텐서의 각 요소 차이의 절댓값 합을 나타내며, 이미지 픽셀 값의 직접적인 차이를 측정하는 데 사용됩니다.
        # 이는 픽셀 단위의 정확성을 평가하는 데 유용하다.

        if iteration > warm_step: # `warm_step` 3000 이터레이션 이후부터 추가적인 손실 항들을 계산합니다.
            loss += 1e-5 * (render_pkg['motion']['d_xyz'].abs()).mean()
            # `render_pkg`에서 일반 모션(`motion`) 네트워크가 예측한 가우시안의 XYZ 위치 변화량(`d_xyz`)의 절대값 평균에
            # 작은 계수 ($1 \times 10^{-5}$)를 곱하여 전체 손실에 더합니다.
            # 이는 가우시안의 위치가 과도하게 변동하는 것을 제어하고, 보다 부드럽고 자연스러운 움직임을 유도하는 정규화 항입니다.
            loss += 1e-5 * (render_pkg['motion']['d_rot'].abs()).mean()
            # 일반 모션(`motion`) 네트워크가 예측한 가우시안의 회전 변화량(`d_rot`)의 절대값 평균에
            # 작은 계수 ($1 \times 10^{-5}$)를 곱하여 전체 손실에 더합니다.
            # 이 또한 회전의 급격한 변화를 제어하고 자연스러운 회전 움직임을 유도하는 정규화 항이다.
            loss += 1e-3 * (((1-alpha) * lips_mask).mean() + (alpha * ~lips_mask).mean())
            # 이 손실 항은 입술 마스크(`lips_mask`)를 기반으로 가우시안의 투명도(`alpha`)를 조절합니다.
            # 첫 번째 항 `((1-alpha) * lips_mask).mean()`은 `lips_mask` 영역 내부에서 투명한 부분(`1-alpha`)의 평균을 계산하여, 입술 내부가 불투명해지도록 유도합니다.
            # 두 번째 항 `(alpha * ~lips_mask).mean()`은 `lips_mask` 영역 밖(`~lips_mask`)에서 불투명한 부분(`alpha`)의 평균을 계산하여, 입술 외부가 투명해지도록 유도합니다.
            # 이 손실 항은 입술의 경계를 명확히 하고, 입술이 주변 배경과 잘 분리되도록 학습하는 데 기여한다.

            if iteration > p_motion_start_iter: # `p_motion_start_iter` 0이므로, 위의 조건인 3000 이터레이션 이후부터 추가적인 손실 항들을 계산합니다.
                loss += 1e-5 * (render_pkg['p_motion']['d_xyz'].abs()).mean()
                loss += 1e-5 * (render_pkg['p_motion']['d_rot'].abs()).mean()
                # loss += 1e-5 * (render_pkg['p_motion']['p_xyz'].abs()).mean()
                # loss += 1e-5 * (render_pkg['p_motion']['p_scale'].abs()).mean()
        
                # Contrast
                # 대조 손실(Contrastive Loss) 계산 부분입니다. 이는 현재 인물의 모션이 다른 인물의 모션과 구분되도록 유도한다.
                audio_feat = viewpoint_cam.talking_dict["auds"].cuda()
                # 현재 카메라 시점의 `talking_dict`에서 'auds'(오디오 특징) 값을 가져와 GPU에 할당하고 `audio_feat` 변수에 저장합니다.
                p_motion_preds = gaussians.neural_motion_grid(gaussians.get_xyz, audio_feat) # face 훈련과 달리 표정 피처가 input으로 들어가지 않고, output에도 가우시안의 불투명도와 크기 변화는 없다.
                # `gaussians` 객체의 `neural_motion_grid` (PersonalizedMotionNetwork) 모델을 사용하여 개인화된 모션 예측(`p_motion_preds`)을 수행합니다.

                with torch.no_grad():
                    # 이 블록 내의 연산에서는 그래디언트가 계산되지 않습니다.
                    # 이는 다른 인물의 모션 예측이 현재 학습 중인 모델에 영향을 주지 않도록 하기 위함이다.
                    tmp_scene_idx = randint(0, len(scene_list)-1) #  `scene_list`에서 무작위로 인덱스를 하나 선택하여 다른 인물 장면을 선택합니다.
                    while tmp_scene_idx == cur_scene_idx: tmp_scene_idx = randint(0, len(scene_list)-1) # 현재 학습 중인 인물과 동일한 인물이 선택되지 않도록, 다른 인물이 선택될 때까지 반복합니다.
                    tmp_scene = scene_list[tmp_scene_idx] # 선택된 다른 인물에 해당하는 `Scene` 객체를 가져옵니다.
                    tmp_gaussians = tmp_scene.gaussians # 해당 `Scene`의 가우시안 모델을 가져옵니다.
                    tmp_p_motion_preds = tmp_gaussians.neural_motion_grid(gaussians.get_xyz, audio_feat)
                    # 동일한 오디오 특징(`audio_feat`)과 (현재 인물의) 가우시안 위치(`gaussians.get_xyz`)를 사용하여,
                    # 선택된 다른 인물(`tmp_gaussians`)의 개인화된 모션 예측(`tmp_p_motion_preds`)을 수행합니다.
                    # 이는 동일한 입력에 대해 다른 인물이 어떻게 반응할지 시뮬레이션하는 것이다.
                contrast_loss = (tmp_p_motion_preds['d_xyz'] * p_motion_preds['d_xyz']).sum(-1)
                # 현재 인물의 `d_xyz` 예측(`p_motion_preds['d_xyz']`)과 다른 인물의 `d_xyz` 예측(`tmp_p_motion_preds['d_xyz']`) 간의 내적을 계산합니다.
                # `.sum(-1)`은 마지막 차원(예: 3D 벡터)에 대해 합을 구하여 두 모션 벡터의 유사도를 측정합니다.
                contrast_loss[contrast_loss < 0] = 0
                # `contrast_loss` 값이 0보다 작을 경우 0으로 설정합니다.
                # 이는 두 모션 벡터가 유사하지 않거나 반대 방향일 경우 손실을 부여하지 않고, 유사할 경우에만 손실을 통해 조절합니다.
                # 즉, 유사도가 높은 경우에만 페널티를 부여하여 두 모션이 서로 멀어지도록 유도한다.
                loss += contrast_loss.mean()
                # 계산된 `contrast_loss`의 평균을 전체 손실(`loss`)에 추가합니다.
                # 이 손실은 현재 장면의 모션이 다른 장면의 모션과 너무 유사해지지 않도록 하여, 개인화된 모션의 고유성을 강화하는 역할을 합니다.
                # 즉, 현재 배우의 모션이 특정 오디오에 대해 다른 배우의 모션과는 구별되도록 유도한다.
                
        image_t = image_green.clone()
        gt_image_t = gt_image_green.clone()

        if iteration > lpips_start_iter: # 해당 사항 없음.
            patch_size = random.randint(16, 21) * 2
            loss += 0.5 * lpips_criterion(patchify(image_t[None, ...] * 2 - 1, patch_size), patchify(gt_image_t[None, ...] * 2 - 1, patch_size)).mean()



        loss.backward()

        iter_end.record()

       
        with torch.no_grad():
            # Progress bar (face 훈련과 동일)
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{5}f}", "AU25": f"{au_lb:.{1}f}-{au_ub:.{1}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # if (iteration in saving_iterations):
            #     print("\n[ITER {}] Saving Gaussians".format(iteration))
            #     scene.save(str(iteration)+'_mouth')

            # Log and save
            # face 훈련과 동일
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, motion_net, motion_net_face, render if iteration < warm_step else render_motion_mouth_con, (pipe, background))
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                ckpt = (motion_net.state_dict(), motion_optimizer.state_dict(), iteration)
                torch.save(ckpt, dataset.model_path + "/chkpnt_mouth_latest" + ".pth")
                with ema_motion_net.average_parameters():
                    ckpt_ema = (motion_net.state_dict(), motion_optimizer.state_dict(), iteration)
                    torch.save(ckpt, dataset.model_path + "/chkpnt_ema_mouth_latest" + ".pth")
                for _scene in scene_list:
                    _gaussians = _scene.gaussians
                    ckpt = (_gaussians.capture(), motion_net.state_dict(), motion_optimizer.state_dict(), iteration)
                    torch.save(ckpt, _scene.model_path + "/chkpnt_mouth_" + str(iteration) + ".pth")
                    torch.save(ckpt, _scene.model_path + "/chkpnt_mouth_latest" + ".pth")

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.05 + 0.25 * iteration / opt.densify_until_iter, scene.cameras_extent, size_threshold)

                    shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree+1)**2)
                    dir_pp = (gaussians.get_xyz - viewpoint_cam.camera_center.repeat(gaussians.get_features.shape[0], 1))
                    dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                    from utils.sh_utils import eval_sh
                    sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
                    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

                    bg_color_mask = (colors_precomp[..., 0] < 20/255) * (colors_precomp[..., 1] > 235/255) * (colors_precomp[..., 2] < 20/255) # 임계값을 좀 더 엄격하게 잡는다.
                    gaussians.xyz_gradient_accum[bg_color_mask] /= 2
                    # 배경 가우시안으로 식별된 영역의 XYZ 그래디언트 누적 값을 절반으로 줄인다. 이는 해당 가우시안의 위치 업데이트를 둔화시켜 움직임을 억제한다.
                    gaussians._opacity[bg_color_mask] = gaussians.inverse_opacity_activation(torch.ones_like(gaussians._opacity[bg_color_mask]) * 0.1)
                    # 배경 가우시안의 불투명도를 크게 낮춘다 (거의 투명하게 만든다). 이는 배경 가우시안이 렌더링에 미치는 영향력을 최소화한다.
                    gaussians._scaling[bg_color_mask] /= 10
                    # 배경 가우시안의 스케일(크기)을 10분의 1로 줄인다. 이는 배경 가우시안이 시야에서 작아지도록 하여 영향력을 감소시킨다.

                # if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                #     gaussians.reset_opacity()

            # Optimizer step (face 훈련과 동일)
            if iteration < opt.iterations:
                motion_optimizer.step()
                gaussians.optimizer.step()

                motion_optimizer.zero_grad()
                gaussians.optimizer.zero_grad(set_to_none = True)

                scheduler.step()
                ema_motion_net.update()


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

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, motion_net, motion_net_face, renderFunc, renderArgs):
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : [scene.getTestCameras()[idx % len(scene.getTestCameras())] for idx in range(5, 100, 10)]}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg_p = None
                    if renderFunc is render:
                        render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    else:
                        render_pkg = renderFunc(viewpoint, scene.gaussians, motion_net, scene.gaussians_2, motion_net_face, *renderArgs)
                        render_pkg_p = renderFunc(viewpoint, scene.gaussians, motion_net, scene.gaussians_2, motion_net_face, personalized=True, *renderArgs)

                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    alpha = render_pkg["alpha"]
                    # image = image - renderArgs[1][:, None, None] * (1.0 - alpha) + viewpoint.background.cuda() / 255.0 * (1.0 - alpha)
                    image = image
                    # gt_image = torch.clamp(viewpoint.original_image.to("cuda") / 255.0, 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda") / 255.0, 0.0, 1.0) * alpha + renderArgs[1][:, None, None] * (1.0 - alpha)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}_mouth/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if render_pkg_p is not None:
                            tb_writer.add_images(config['name'] + "_view_{}_mouth/render_p".format(viewpoint.image_name), render_pkg_p["render"][None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}_mouth/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}_mouth/depth".format(viewpoint.image_name), (render_pkg["depth"] / render_pkg["depth"].max())[None], global_step=iteration)


                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # 얼굴 훈련 코드와 달리 --share_audio_net 인자가 없다.
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
