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
from gaussian_renderer import render, render_motion, render_motion_emotion
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

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, mode_long, pretrain_ckpt_path, use_va):
    testing_iterations = [1] + [i for i in range(0, opt.iterations + 1, 10000)] # [1, 0, 10000]
    checkpoint_iterations =  saving_iterations = [i for i in range(0, opt.iterations + 1, 10000)] + [opt.iterations] # [0, 10000, 10000]

    # vars
    # 전체 iter: 10,000회
    warm_step = 3000
    opt.densify_until_iter = opt.iterations - 1000 # 9000
    bg_iter = opt.iterations # 해당 스텝 이후에는 모델 업데이트를 멈추는데, 이터레이션과 동일한 값이라 의미가 없음. 10000 
    lpips_start_iter = opt.densify_until_iter - 1500 # 7500 # pretrain과 달리 여기서는 1500회 남기고 lpips 손실도 계산
    motion_stop_iter = bg_iter # 10000 역시 의미 없는 값.
    mouth_select_iter = opt.iterations # 10000 의미 없는 값
    mouth_step = 1 / max(mouth_select_iter, 1) # 1 / 10000 = 1e-04
    hair_mask_interval = 7
    select_interval = 10

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset) # 파인튜닝 할 Lieu 가우시안 초기화
    scene = Scene(dataset, gaussians) # Lieu에 대한 가우시안과 카메라 정보를 저장

    motion_net = MotionNetwork(args=dataset).cuda() # pretrain 된 Face 브랜치의 UMF를 불러오기 전에 초기화
    motion_optimizer = torch.optim.AdamW(motion_net.get_params(5e-3, 5e-4), betas=(0.9, 0.99), eps=1e-8, weight_decay=0.01) # UMF의 파라미터를 업데이트 할 옵티마이저 설정
    scheduler = torch.optim.lr_scheduler.LambdaLR(motion_optimizer, lambda iter: 0.1 if iter < warm_step else 0.5 ** (iter / opt.iterations))
    if mode_long:
        scheduler = torch.optim.lr_scheduler.LambdaLR(motion_optimizer, lambda iter: 0.1 if iter < warm_step else 0.1 ** (iter / opt.iterations))
    
    # =============== Face UMF 모델 불러오기 ================== #
    # Load pre-trained
    (motion_params, _, _) = torch.load(pretrain_ckpt_path) # pretrain 된 Face 브랜치의 UMF weight를 로드 ('output/pretrain_ave/chkpnt_ema_face_latest.pth')
    # gaussians.restore(model_params, opt)
    motion_net.load_state_dict(motion_params)
    
    # (model_params, _, _, _) = torch.load(os.path.join("output/pretrain4/macron/chkpnt_face_latest.pth"))
    # gaussians.neural_motion_grid.load_state_dict(model_params[-1])

    lpips_criterion = lpips.LPIPS(net='alex').eval().cuda() # 오차 계산용 LPIPS 손실을 계산하기 위한 함수 초기화.

    gaussians.training_setup(opt) # 가우시안의 속성들 및 보조 네트워크들의 파라미터를 옵티마이저에 등록하고, 스케줄러 생성.

    # 디폴트가 None이므로, 아래 분기는 생략.
    checkpoint = 'output/emotion/angry/chkpnt_face_latest.pth'

    if checkpoint:
        (model_params, motion_params, motion_optimizer_params, first_iter) = torch.load(checkpoint)
        first_iter = 0
        gaussians.restore(model_params, opt)
        motion_net.load_state_dict(motion_params)
        motion_optimizer.load_state_dict(motion_optimizer_params)


    # print(gaussians.get_xyz)

    if not mode_long: # not False -> True
        gaussians.max_sh_degree = 1 # 짧게 학습 시킬 것이므로 sh_degree를 3까지 올리지 않고, 1로 세팅.
    # ==================================== #
    bg_color = [0, 1, 0]   # [1, 1, 1] # if dataset.white_background else [0, 0, 0] # 배경을 초록색 (G)만 1로 세팅.
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), ascii=True, dynamic_ncols=True, desc="Training progress")
    
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)
        # 가우시안의 파라미터 러닝 레이트는 iteration에 따라 달라지므로, for문 안에서 업데이트 해줘야 한다.

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy() # train 데이터에 있는 모든 카메라 정보 복사
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1)) # 랜덤 선택

        # find a big mouth
        mouth_global_lb = viewpoint_cam.talking_dict['mouth_bound'][0]
        mouth_global_ub = viewpoint_cam.talking_dict['mouth_bound'][1]
        mouth_global_lb += (mouth_global_ub - mouth_global_lb) * 0.2
        mouth_window = (mouth_global_ub - mouth_global_lb) * 0.5 # pretrain과 달리 0.2가 아니라 0.5
        # train_face.py에서는 mouth_window가 훨씬 넓다. 이는 학습 초반부터 더 넓은 범위의 입 모양을 탐색하고, 샘플링 필터링이 덜 엄격하게 적용됨을 의미한다.
        # 즉, 다양한 입 벌림 크기의 프레임을 더 쉽게 학습에 포함하여 일반화된 얼굴 모션 학습에 더 적합하다.

        mouth_lb = mouth_global_lb + mouth_step * iteration * (mouth_global_ub - mouth_global_lb)
        mouth_ub = mouth_lb + mouth_window
        mouth_lb = mouth_lb - mouth_window


        au_global_lb = 0
        au_global_ub = 1
        au_window = 0.4 # 0.3이 아니라 0.4로 설정
        # train_face.py에서는 au_window가 더 넓다. 이는 AU 값의 더 넓은 범위를 탐색하며, pretrain_face.py에 비해 AU 기반 샘플링에서 더 많은 프레임을 허용한다.

        au_lb = au_global_lb + mouth_step * iteration * (au_global_ub - au_global_lb)
        au_ub = au_lb + au_window
        au_lb = au_lb - au_window * 1.5 # 0.5가 아니라 1.5
        # train_face.py에서는 au_lb를 훨씬 더 큰 폭으로 낮춘다. 이는 AU 값이 매우 낮은(예: 눈을 거의 감지 않거나 특정 표정이 거의 없는) 프레임까지도 학습에 적극적으로 포함시키려는 시도이다.

        if iteration < warm_step and iteration < mouth_select_iter: # iter < 3000 < 10000
            if iteration % select_interval == 0: # 10 인터벌마다
                while viewpoint_cam.talking_dict['mouth_bound'][2] < mouth_lb or viewpoint_cam.talking_dict['mouth_bound'][2] > mouth_ub:
                    if not viewpoint_stack:
                        viewpoint_stack = scene.getTrainCameras().copy()
                    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1)) # mouth_lb < 해당 프레임 입벌림 < mouth_ub일 때까지 다시 뽑기


        if warm_step < iteration < mouth_select_iter: # 3000 < iter < 10000
            if iteration % select_interval == 0: # 10 인터벌마다
                while viewpoint_cam.talking_dict['blink'] < au_lb or viewpoint_cam.talking_dict['blink'] > au_ub:
                    if not viewpoint_stack:
                        viewpoint_stack = scene.getTrainCameras().copy()
                    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1)) # au_lb < 눈 크기 < au_ub 일 때까지 다시 뽑기



        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        face_mask = torch.as_tensor(viewpoint_cam.talking_dict["face_mask"]).cuda()
        hair_mask = torch.as_tensor(viewpoint_cam.talking_dict["hair_mask"]).cuda()
        mouth_mask = torch.as_tensor(viewpoint_cam.talking_dict["mouth_mask"]).cuda()
        head_mask =  face_mask + hair_mask


        if iteration > lpips_start_iter: # 7500 이상일 경우 아래 수행
        # lpips_start_iter(=7500) 이후부터 적용하는 이유는,
        # 학습 초기는 강한 마스킹으로 모델이 정밀하게 입 모양 근처의 정보를 배우도록 유도하고,
        # 후반에는 부드러운 마스크를 씌워 일반화 및 다양한 입 주변 표현을 잘 학습하도록 전환하려는 전략이다.
            max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            # 입 주변 경계를 더 부드럽게 확장(mouth_mask에 두 번 max pooling)하는 이유는,
            # 학습 후기에 네트워크가 입 주변까지 더 넓은 영역을 고려해 robust하게 학습하게 만들기 위함이다.
            mouth_mask = (-max_pool(-max_pool(mouth_mask[None].float())))[0].bool()

        
        hair_mask_iter = (warm_step < iteration < lpips_start_iter - 1000) and iteration % hair_mask_interval != 0
        # 3000 < iter < 6500 일 때 7 인터벌로 머리카락 마스킹

        # 약간의 진동(perturbation/noise)을 더해 감정 값이 고정되지 않도록 한다.
        valence_center = -0.8
        arousal_center = 0.8
        # 예시: 표정이 약간씩 흔들리는 효과를 주기 위해 가우시안 노이즈(진동) 추가
        valence = valence_center + 0.03 * torch.randn(1).item()
        arousal = arousal_center + 0.03 * torch.randn(1).item()

        # happy: [0.8, 0.6]
        # angry: [-0.8, 0.8]
        # disgust: [-0.9, 0.6]

        if iteration < warm_step: # 3000 이전
            # render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            enable_align = iteration > 1000 # 1000번 이전까진 align 안 함.
            # Valence와 Arousal 값을 텐서로 변환하여 전달
            va_tensor = torch.tensor([[valence, arousal]], dtype=torch.float32).cuda() if use_va else None
            render_pkg = render_motion_emotion(viewpoint_cam, gaussians, motion_net, pipe, background, return_attn=True, personalized=False, align=enable_align, va=va_tensor) # personalized=False로 바꿔둘 것.
            # for param in motion_net.parameters():
            #     param.requires_grad = False
        else:
            # Valence와 Arousal 값을 텐서로 변환하여 전달
            va_tensor = torch.tensor([[valence, arousal]], dtype=torch.float32).cuda() if use_va else None
            render_pkg = render_motion_emotion(viewpoint_cam, gaussians, motion_net, pipe, background, return_attn=True, personalized=False, align=True, va=va_tensor)
            # for param in motion_net.parameters():
            #     param.requires_grad = True
                
        image_white, alpha, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["alpha"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        gt_image  = viewpoint_cam.original_image.cuda() / 255.0
        gt_image_white = gt_image * head_mask + background[:, None, None] * ~head_mask

        # 생략
        if iteration > motion_stop_iter: # 10000 이상일 때라 의미 없음
            for param in motion_net.parameters():
                param.requires_grad = False
        if iteration > bg_iter: # 10000 이상일 때라 의미 없음
            gaussians._xyz.requires_grad = False
            gaussians._opacity.requires_grad = False
            # gaussians._features_dc.requires_grad = False
            # gaussians._features_rest.requires_grad = False
            gaussians._scaling.requires_grad = False
            gaussians._rotation.requires_grad = False
        
        # Loss
        if iteration < bg_iter: # 항상 적용
            if hair_mask_iter:
                image_white[:, hair_mask] = background[:, None]
                gt_image_white[:, hair_mask] = background[:, None]
            
            # image_white[:, mouth_mask] = 1
            
            gt_image_white[:, mouth_mask] = background[:, None]
            # mouth_mask 영역(입 부분)에 해당하는 gt_image_white 픽셀을 백그라운드 색상으로 덮어씌워서 입술은 학습하지 않음.

            Ll1 = l1_loss(image_white, gt_image_white)
            loss = Ll1 + opt.lambda_dssim * (1.0 - ssim(image_white, gt_image_white))

            if not mode_long and iteration > warm_step + 2000:
                # 예측 surface normal과 GT normal의 코사인 유사도 기반 loss를 추가하여, 예측된 normal이 GT normal과 유사하도록 유도함
                loss += 0.01 * (1 - viewpoint_cam.talking_dict["normal"].cuda() * render_pkg["normal"]).sum(0)[head_mask^mouth_mask].mean()
                # 위 loss는 head_mask에서 mouth_mask를 제외한 영역에 대해 적용됨

                # opacity_reset_interval이 3000일 때, iteration % 3000 > 100 이므로
                # 0~100 구간(즉, 3000n ~ 3000n+100)에서는 False, 그 외에는 True가 됨.
                # 즉, opacity_reset_interval마다 100 step 동안만 이 loss를 끄고, 나머지 구간에서는 loss를 적용함.
                if iteration % opt.opacity_reset_interval > 100:
                    # depth map 기반의 정규화된 깊이값과 GT 깊이값의 차이에 대한 L1 loss를 추가하여, 예측된 depth가 GT depth와 유사하도록 유도함
                    depth = render_pkg["depth"][0]
                    depth_mono = viewpoint_cam.talking_dict['depth'].cuda()
                    loss += 1e-2 * (normalize(depth)[face_mask^mouth_mask] - normalize(depth_mono)[face_mask^mouth_mask]).abs().mean()
                    # 위 loss는 face_mask에서 mouth_mask를 제외한 영역에 대해 적용됨
                
            # mouth_alpha_loss = 1e-2 * (alpha[:,mouth_mask]).mean()
            # if not torch.isnan(mouth_alpha_loss):
                # loss += mouth_alpha_loss
            # print(alpha[:,mouth_mask], mouth_mask.sum())

            if iteration > warm_step: # 3000 이상일 때
                loss += 1e-5 * (render_pkg['motion']['d_xyz'].abs()).mean()
                loss += 1e-5 * (render_pkg['motion']['d_rot'].abs()).mean()
                loss += 1e-5 * (render_pkg['motion']['d_opa'].abs()).mean()
                loss += 1e-5 * (render_pkg['motion']['d_scale'].abs()).mean()
                loss += 1e-5 * (render_pkg['p_motion']['p_xyz'].abs()).mean()

                loss += 1e-3 * (((1-alpha) * head_mask).mean() + (alpha * ~head_mask).mean())


                [xmin, xmax, ymin, ymax] = viewpoint_cam.talking_dict['lips_rect']
                loss += 1e-4 * (render_pkg["attn"][1, xmin:xmax, ymin:ymax]).mean()
                if not hair_mask_iter:
                    loss += 1e-4 * (render_pkg["attn"][1][hair_mask]).mean()
                    loss += 1e-4 * (render_pkg["attn"][0][hair_mask]).mean()

                # loss += l2_loss(image_white[:, xmin:xmax, ymin:ymax], image_white[:, xmin:xmax, ymin:ymax])
                

            image_t = image_white.clone()
            gt_image_t = gt_image_white.clone()

        else: # 해당 없음
            # with real bg
            image = image_white - background[:, None, None] * (1.0 - alpha) + viewpoint_cam.background.cuda() / 255.0 * (1.0 - alpha)

            Ll1 = l1_loss(image, gt_image)
            loss = Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

            image_t = image.clone()
            gt_image_t = gt_image.clone()

        # lpips_start_iter는 7500으로 설정되어 있음.
        # 즉, iteration > 7500일 때만 아래의 LPIPS 기반 patch loss가 적용된다.
        if iteration > lpips_start_iter:   
            # 현재 카메라의 입술 영역 좌표를 가져온다.
            [xmin, xmax, ymin, ymax] = viewpoint_cam.talking_dict['lips_rect']
            # mode_long이 True일 때, 입술 영역에 대해 LPIPS loss를 0.01 가중치로 추가한다.
            if mode_long:
                loss += 0.01 * lpips_criterion(
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
                loss += 0.2 * lpips_criterion(
                    patchify(image_t[None, ...] * 2 - 1, patch_size), 
                    patchify(gt_image_t[None, ...] * 2 - 1, patch_size)
                ).mean()
            # patchify된 이미지에 대해 LPIPS loss를 0.01 가중치로 추가한다.
            loss += 0.01 * lpips_criterion(
                patchify(image_t[None, ...] * 2 - 1, patch_size), 
                patchify(gt_image_t[None, ...] * 2 - 1, patch_size)
            ).mean()
            # (주석) 전체 이미지에 대해 LPIPS loss를 0.5 가중치로 추가할 수도 있으나, 현재는 사용하지 않음.
            # loss += 0.5 * lpips_criterion(image_t[None, ...] * 2 - 1, gt_image_t[None, ...] * 2 - 1).mean()


        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{5}f}", "Mouth": f"{mouth_lb:.{1}f}-{mouth_ub:.{1}f}"}) # , "AU25": f"{au_lb:.{1}f}-{au_ub:.{1}f}"
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, motion_net, render if iteration < warm_step else render_motion, (pipe, background), valence=valence, arousal=arousal)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(str(iteration)+'_face')

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                ckpt = (gaussians.capture(), motion_net.state_dict(), motion_optimizer.state_dict(), iteration)
                torch.save(ckpt, scene.model_path + "/chkpnt_face_" + str(iteration) + ".pth")
                torch.save(ckpt, scene.model_path + "/chkpnt_face_latest" + ".pth")


            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.05 + 0.25 * iteration / opt.densify_until_iter, scene.cameras_extent, size_threshold)
                
                # [설명] 이 부분은 가우시안의 opacity(불투명도) 값을 주기적으로 리셋하는 코드다.
                # mode_long이 False일 때만 실행된다(현재 mode_long=False이므로 항상 실행됨).
                # opacity를 리셋하는 시점은 두 가지 경우다:
                #   1. iteration이 opacity_reset_interval의 배수일 때 (즉, 일정 주기마다)
                #   2. 흰색 배경을 쓸 때, densify_from_iter 500(초기 densification 시작 시점)에서 한 번
                # 이때 gaussians.reset_opacity()를 호출해서 모든 가우시안의 opacity를 다시 초기화한다.
                if not mode_long:  # mode_long=False이므로 항상 True
                    if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                        gaussians.reset_opacity()
            
            # bg prune
            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                from utils.sh_utils import eval_sh

                shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree+1)**2)
                dir_pp = (gaussians.get_xyz - viewpoint_cam.camera_center.repeat(gaussians.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

                bg_color_mask = (colors_precomp[..., 0] < 30/255) * (colors_precomp[..., 1] > 225/255) * (colors_precomp[..., 2] < 30/255)
                gaussians.prune_points(bg_color_mask.squeeze())
                
                # [설명] 이 부분은 z축(깊이 방향) 좌표가 -0.07보다 작은(즉, 너무 뒤에 있는) 가우시안 포인트들을 제거(prune)하는 코드다.
                # mode_long이 False일 때만 실행된다(현재 mode_long=False이므로 항상 실행됨).
                # gaussians.get_xyz[:, -1]은 모든 가우시안의 z좌표만 추출한다.
                # (gaussians.get_xyz[:, -1] < -0.07)는 z좌표가 -0.07보다 작은 포인트에 대해 True인 불리언 마스크를 만든다.
                # squeeze()는 불필요한 차원을 제거한다.
                # gaussians.prune_points(...)를 통해 해당 마스크에 해당하는 포인트들을 실제로 삭제한다.
                if not mode_long:
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

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, motion_net, renderFunc, renderArgs, valence=None, arousal=None):
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
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--long", action='store_true', default=False) # 더 많은 프레임 추출? False
    parser.add_argument("--pretrain_path", type=str, default = None) # 이전 모델 체크포인트 경로 'output/pretrain_ave/chkpnt_ema_face_latest.pth'
    parser.add_argument("--use_va", action='store_true', default=False) # Valence/Arousal 피처 사용 여부
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.long, args.pretrain_path, args.use_va)

    # All done
    print("\nTraining complete.")
