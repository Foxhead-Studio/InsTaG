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
from utils.loss_utils import l1_loss, l2_loss, patchify, ssim
from gaussian_renderer import render, render_motion, render_motion_mouth_con
import sys
import copy
from scene import Scene, GaussianModel, MotionNetwork, MouthMotionNetwork
from utils.general_utils import safe_state
import lpips
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    opt.densify_until_iter = 0

    testing_iterations = [i for i in range(0, opt.iterations + 1, 2000)] # [0, 2000]
    checkpoint_iterations = [opt.iterations] # [2000]

    # vars
    bg_iter = opt.densify_until_iter # 0
    lpips_start_iter = opt.iterations // 2 # 1000

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    
    dataset.type = "face"
    gaussians = GaussianModel(copy.deepcopy(dataset)) # 학습된 Lieu 얼굴 가우시안
    dataset.type = "mouth"
    gaussians_mouth = GaussianModel(copy.deepcopy(dataset)) # 학습된 Lieu 입안 가우시안
    
    scene = Scene(dataset, gaussians) # 어차피 여기선 랜덤 초기화 된 포인트 클라우드 기반으로 가우시안을 초기화하니, gaussians가 들어가나 gaussians_mouth가 들어가나 상관없는 듯 하다.
    with torch.no_grad():
        motion_net_mouth = MouthMotionNetwork(args=dataset).cuda() # 입안 내부용 UMF
        motion_net = MotionNetwork(args=dataset).cuda() # 얼굴 생성용 UMF

    # gaussians.training_setup(opt)
    # gaussians_mouth.training_setup(opt)

    (model_params, motion_params, _, _) = torch.load(os.path.join(scene.model_path, "chkpnt_face_latest.pth"))
    gaussians.restore(model_params, opt) # 학습된 Lieu 얼굴 가우시안 체크포인트 로드
    motion_net.load_state_dict(motion_params) # 얼굴 생성용 UMF 체크포인트 로드

    (model_params, motion_params, _, _) = torch.load(os.path.join(scene.model_path, "chkpnt_mouth_latest.pth"))
    gaussians_mouth.restore(model_params, opt) # 학습된 Lieu 입안 가우시안 체크포인트 로드
    motion_net_mouth.load_state_dict(motion_params) # 입안 내부용 UMF 체크포인트 로드

    lpips_criterion = lpips.LPIPS(net='alex').eval().cuda()

    bg_color = [0, 1, 0]   # [1, 1, 1] # if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), ascii=True, dynamic_ncols=True, desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1): # 2000회 반복

        iter_start.record()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        gaussians.update_learning_rate(iteration)

        face_mask = torch.as_tensor(viewpoint_cam.talking_dict["face_mask"]).cuda()
        hair_mask = torch.as_tensor(viewpoint_cam.talking_dict["hair_mask"]).cuda()
        mouth_mask = torch.as_tensor(viewpoint_cam.talking_dict["mouth_mask"]).cuda()
        head_mask = face_mask + hair_mask + mouth_mask

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        render_pkg = render_motion(viewpoint_cam, gaussians, motion_net, pipe, background, align=True) # 얼굴 전체를 렌더링한다.
        # render_pkg_mouth = render_motion_mouth(viewpoint_cam, gaussians_mouth, motion_net_mouth, pipe, background, align=True)
        render_pkg_mouth = render_motion_mouth_con(viewpoint_cam, gaussians_mouth, motion_net_mouth, gaussians, motion_net, pipe, background, align=True) # 입안(구강 내부)을 렌더링한다.
        viewspace_point_tensor, visibility_filter = render_pkg["viewspace_points"], render_pkg["visibility_filter"] # 얼굴 전체 렌더링 결과에서 3D 포인트와 가시성 정보를 가져온다.
        viewspace_point_tensor_mouth, visibility_filter_mouth = render_pkg_mouth["viewspace_points"], render_pkg_mouth["visibility_filter"] # 입안 렌더링 결과에서 3D 포인트와 가시성 정보를 가져온다.

        alpha_mouth = render_pkg_mouth["alpha"] # 입안 렌더링 결과의 알파(투명도) 맵을 가져온다.
        alpha = render_pkg["alpha"] # 얼굴 전체 렌더링 결과의 알파(투명도) 맵을 가져온다.
        
        # 입안(구강 내부) 이미지를 생성한다.
        # - render_pkg_mouth["render"]: 입안 가우시안으로부터 렌더링된 RGB 이미지 (0~1 범위, shape: [3, H, W])
        # - alpha_mouth: 입안 가우시안의 알파(투명도) 맵 (shape: [1, H, W])
        # - background: 배경 색상 텐서 ([3], 예: [0, 1, 0]은 초록색)
        # - viewpoint_cam.background: 원본 이미지의 배경 색상 ([3], 0~255 범위)
        # 
        # 입안 이미지는 다음과 같이 합성된다:
        #   1. 입안 가우시안이 투명한 영역(1 - alpha_mouth)에는 배경색을 넣는다.
        #   2. 배경색은 두 가지로 나뉜다:
        #      - background[:, None, None]: 기본 배경색(예: 초록색)
        #      - viewpoint_cam.background.cuda() / 255.0: 원본 이미지의 배경색(정규화)
        #   3. 두 배경색 중 viewpoint_cam.background는 실제 카메라별 배경색을 반영한다.
        #   4. 최종적으로, 입안 가우시안이 불투명한 영역(alpha_mouth)에는 render_pkg_mouth["render"]가, 
        #      투명한 영역(1 - alpha_mouth)에는 viewpoint_cam.background가 들어간다.
        mouth_image = (
            render_pkg_mouth["render"]  # 입안 가우시안이 불투명한 영역(입 내부)
            - background[:, None, None] * (1.0 - alpha_mouth)  # 기본 배경색(초록색) 제거
            + viewpoint_cam.background.cuda() / 255.0 * (1.0 - alpha_mouth)  # 카메라별 배경색 추가
        )

        # 얼굴 전체 이미지를 생성한다.
        # - render_pkg["render"]: 얼굴 전체(외부) 가우시안으로부터 렌더링된 RGB 이미지
        # - alpha: 얼굴 전체 가우시안의 알파(투명도) 맵
        # 
        # 얼굴 전체 이미지는 다음과 같이 합성된다:
        #   1. 얼굴 전체 가우시안이 투명한 영역(1 - alpha)에는 mouth_image(입안 이미지)를 넣는다.
        #      즉, 입이 벌어진 부분(얼굴 가우시안이 투명한 곳)에만 입안 이미지를 삽입한다.
        #   2. 얼굴 전체 가우시안이 불투명한 영역(alpha)에는 render_pkg["render"]가 그대로 들어간다.
        #   3. 이로써, 얼굴 전체 이미지에서 입이 벌어진 부분만 입안 이미지로 치환된 최종 합성 이미지가 만들어진다.
        image = (
            render_pkg["render"]  # 얼굴 전체 가우시안이 불투명한 영역(얼굴 외부)
            - background[:, None, None] * (1.0 - alpha)  # 기본 배경색(초록색) 제거
            + mouth_image * (1.0 - alpha)  # 얼굴 가우시안이 투명한 영역(입 부분)에 입안 이미지 삽입
        )
                
        gt_image  = viewpoint_cam.original_image.cuda() / 255.0 # 정답(ground truth) 이미지를 0~1로 정규화하여 가져온다.
        gt_image_white = gt_image * head_mask + background[:, None, None] * ~head_mask # head_mask(얼굴+머리+입안) 영역만 정답 이미지를 사용하고, 나머지는 배경색으로 채운다.

        if iteration > bg_iter:
            # 전부 다 동결시키고 PMF만 학습 시킨다.
            for param in motion_net.parameters():
                param.requires_grad = False
            for param in motion_net_mouth.parameters():
                param.requires_grad = False
                
            gaussians._xyz.requires_grad = False
            # gaussians._opacity.requires_grad = False
            gaussians._scaling.requires_grad = False
            gaussians._rotation.requires_grad = False

            gaussians_mouth._xyz.requires_grad = False
            gaussians_mouth._opacity.requires_grad = False
            gaussians_mouth._scaling.requires_grad = False
            gaussians_mouth._rotation.requires_grad = False
        

        # Loss
        if iteration < bg_iter: # 해당 안 됨.
            image[:, ~head_mask] = background[:, None]
            # gt_image_white[:, ~head_mask] = background[:, None]

            Ll1 = l1_loss(image, gt_image_white)
            loss = Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image_white))
            loss += 1e-3 * (((1-alpha) * head_mask).mean() + (alpha * ~head_mask).mean())

            image_t = image.clone()
            gt_image_t = gt_image_white.clone()

        else:
            Ll1 = l1_loss(image, gt_image)
            loss = Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

            image_t = image.clone()
            gt_image_t = gt_image.clone()

        if iteration > lpips_start_iter: # 해당 안 됨.    
            # mask mouth
            # [xmin, xmax, ymin, ymax] = viewpoint_cam.talking_dict['lips_rect']
            # image_t[:, xmin:xmax, ymin:ymax] = 1
            # gt_image_t[:, xmin:xmax, ymin:ymax] = 1
            
            patch_size = random.randint(16, 21) * 2
            loss += 0.05 * lpips_criterion(patchify(image_t[None, ...] * 2 - 1, patch_size), patchify(gt_image_t[None, ...] * 2 - 1, patch_size)).mean()



        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{5}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, testing_iterations, image, gt_image)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                ckpt = (gaussians.capture(), motion_net.state_dict(), gaussians_mouth.capture(), motion_net_mouth.state_dict())
                torch.save(ckpt, scene.model_path + "/chkpnt_fuse_" + str(iteration) + ".pth")
                torch.save(ckpt, scene.model_path + "/chkpnt_fuse_latest" + ".pth")


            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                gaussians_mouth.add_densification_stats(viewspace_point_tensor_mouth, visibility_filter_mouth)

                if iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.3, scene.cameras_extent, size_threshold)
                    gaussians_mouth.densify_and_prune(opt.densify_grad_threshold, 0.3, scene.cameras_extent, size_threshold)


            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians_mouth.optimizer.step()

                gaussians.optimizer.zero_grad(set_to_none = True)
                gaussians_mouth.optimizer.zero_grad(set_to_none = True)


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


def training_report(tb_writer, iteration, testing_iterations, image, gt_image):
    # Report test and samples of training set
    if iteration in testing_iterations:
        tb_writer.add_images("fuse/render", image[None], global_step=iteration)
        tb_writer.add_images("fuse/ground_truth", gt_image[None], global_step=iteration)



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
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