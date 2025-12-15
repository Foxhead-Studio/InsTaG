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
from gaussian_renderer import render, render_motion
import sys, copy
from scene_pretrain import Scene, GaussianModel, MotionNetwork, PersonalizedMotionNetwork
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

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, share_audio_net):    
    # datset = {'sh_degree': 2, 'source_path': '/home/white/github/InsTaG/data/pretrain', 'model_path': 'debug_01', 'images': 'images', 'resolution': -1, 'white_background': False, 'data_device': 'cpu', 'eval': False, 
    #           'audio': '', 'init_num': 2000, 'N_views': -1, 'audio_extractor': 'deepspeech', 'type': 'face', 'preload': True, 'all_for_train': False}
    # opt = {'iterations': 30000, 'position_lr_init': 0.00016, 'position_lr_final': 1.6e-06, 'position_lr_delay_mult': 0.01, 'position_lr_max_steps': 45000, 'feature_lr': 0.0025, 'opacity_lr': 0.05, 'scaling_lr': 0.003, 
    #       'rotation_lr': 0.001, 'percent_dense': 0.005, 'lambda_dssim': 0.2, 'densification_interval': 100, 'opacity_reset_interval': 3000, 'densify_from_iter': 500, 'densify_until_iter': 29000, 
    #       'densify_grad_threshold': 0.0005, 'random_background': False}
    # pipe = {'convert_SHs_python': False, 'compute_cov3D_python': False, 'debug': False}
    # testing_iterations = []
    # saving_iterations = [30000]
    # checkpoint_iterations = []
    # checkpoint = None
    # debug_from = -1
    # share_audio_net = False
    
    # data_list = ["macron", "shaheen", "may", "jaein", "obama1"]
    data_list = ["macron"]

    testing_iterations = [i * len(data_list) for i in range(0, opt.iterations + 1, 2000)] # 모델의 테스트를 수행할 이터레이션(반복) 시점을 저장하는 리스트
    # [] -> [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000, 24000, 26000, 28000, 30000] * n
    checkpoint_iterations =  saving_iterations = [i * len(data_list) for i in range(0, opt.iterations + 1, 10000)] + [opt.iterations * len(data_list)] # 모델의 체크포인트를 생성하고 저장할 이터레이션 시점을 저장하는 리스트
    # [0, 10000, 20000, 30000, 30000]
    
    # n = 1일때 가정(현재는 macron만 실험 중)
    warm_step = 1000 * len(data_list) # 1000 * 1 = 1000
    # 웜업 (학습 초기 단계)는 어느 이터레이션까지 진행할 것인지 결정하는 변수
    opt.densify_until_iter = (opt.iterations - 1000) * len(data_list) # (30000 - 1000) * 1 = 29000
    # 가우시안 모델의 밀집화(densification)를 수행할 최대 이터레이션입니다. 밀집화는 모델의 표현력을 높이기 위해 새로운 가우시안을 추가하는 과정입니다. 후반의 1000 이터레이션은 밀집화를 진행하지 않음
    lpips_start_iter = 99999999 * len(data_list) # 99999999 * 1 = 99999999
    # LPIPS 손실 계산을 시작할 이터레이션입니다. 이 값이 매우 크게 설정되어 있어, 사실상 LPIPS 손실은 이 학습 과정에서 사용되지 않을 것임을 나타냅니다.
    motion_stop_iter = opt.iterations * len(data_list) # 30000 * 1 = 30000
    # 모션 네트워크의 학습을 중단할 이터레이션입니다. 전체 학습 스텝 동안 동안 멈추지 않고 계속 업데이트 됩니다.
    mouth_select_iter = (opt.iterations - 10000) * len(data_list) # (30000 - 10000) * 1 = 20000
    # 특정 입 영역과 AU가 있는 이미지를 샘플링하는 과정을 언제까지 진행할지 결정하는 변수. (warmup step부터 20000까지 진행)
    p_motion_start_iter = 0 # 개인화된 모션 네트워크의 학습을 시작할 이터레이션입니다. 0으로 설정되어 있으나, warmup step 이전까지는 캐노니컬 가우시안을 생성하므로, 그 이후부터 전체 이터레이션 동안 진행.
    mouth_step = 1 / max(mouth_select_iter, 1) # 1 / max(20000, 1) = 1 / 20000 = 5e-05
    # 입 크기를 점진적으로 키워나가기 위한 일종의 가중치.
    hair_mask_interval = 7 # 머리카락 마스크를 적용하는 간격을 나타내는 변수.
    select_interval = 15 # 특정 범위 내의 입벌림, AU를 샘플링하는 간격.
    
    opt.iterations *= len(data_list) # 30000 * n
    # 전체 학습 이터레이션 수를 처음의 값 30000에 사람의 수 n을 곱한 값으로 업데이트 합니다.
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    
    # false이므로 건너 뜀
    if share_audio_net:
        motion_net = MotionNetwork(args=dataset).cuda()
        motion_optimizer = torch.optim.AdamW(motion_net.get_params(5e-3, 5e-4), betas=(0.9, 0.99), eps=1e-8)
        scheduler = torch.optim.lr_scheduler.LambdaLR(motion_optimizer, lambda iter: (0.5 ** (iter / mouth_select_iter)) if iter < mouth_select_iter else 0.1 ** (iter / opt.iterations))
        ema_motion_net = ExponentialMovingAverage(motion_net.parameters(), decay=0.995)
    
    scene_list = []
    # scene_list는 각 데이터셋(예: 여러 인물)에 대한 Scene 객체를 저장하는 리스트입니다.
    for data_name in data_list: # ['macron']
        # data_list에 있는 각 데이터셋 이름(예: 'macron')에 대해 반복합니다.
        gaussians = GaussianModel(dataset)
        # GaussianModel 객체를 초기화.
        # gaussians = {'args': <arguments.GroupParams object at 0x7f9880fd59a0>, 'active_sh_degree': 0, 'max_sh_degree': 2, '_xyz': tensor([]), '_features_dc': tensor([]), '_features_rest': tensor([]), '_identity': tensor([]), 
                    # '_scaling': tensor([]), '_rotation': tensor([]), '_opacity': tensor([]), 'max_radii2D': tensor([]), 'xyz_gradient_accum': tensor([]), 'denom': tensor([]), 'optimizer': None, 'percent_dense': 0, 'spatial_lr_scale': 0, 
                    # 'neural_renderer': None, 'neural_motion_grid': None}
        # neural_renderer는 가우시안 포인트와 바라보는 방향을 input으로 받아 밀도와 컬러 값을 출력한다.
        # neural_motion_grid는 PMF 네트워크이다.
        
        _dataset = copy.deepcopy(dataset)
        # dataset 객체를 깊은 복사하여 _dataset에 저장합니다.
        # 이렇게 하면 원본 dataset을 변경하지 않고 각 데이터셋별로 독립적으로 경로 등을 수정할 수 있습니다.
        _dataset.source_path = os.path.join(dataset.source_path, data_name)
        # _dataset의 source_path를 data_name(예: 'macron')까지 붙여서 업데이트.
        # 예시: '/home/white/github/InsTaG/data/pretrain/macron'
        _dataset.model_path = os.path.join(dataset.model_path, data_name)
        # _dataset의 model_path도 마찬가지로 data_name까지 붙여서 업데이트.
        # 예시: 'debug_01/macron'
        
        # _dataset = {'sh_degree': 2, 'source_path': '/home/white/github/InsTaG/data/pretrain/macron', 'model_path': 'debug_01/macron', 'images': 'images', 'resolution': -1, 'white_background': False, 'data_device': 'cpu', 
        #               'eval': False, 'audio': '', 'init_num': 2000, 'N_views': -1, 'audio_extractor': 'deepspeech', 'type': 'face', 'preload': True, 'all_for_train': False}
        
        # 아규먼트 값을 output 폴더에 저장.
        os.makedirs(_dataset.model_path, exist_ok = True) # 'debug_01/macron' 생성
        # model_path 디렉토리가 없으면 새로 생성합니다. 이미 있으면 아무 일도 일어나지 않습니다.
        with open(os.path.join(_dataset.model_path, "cfg_args"), 'w') as cfg_log_f: # 'debug_01/macron/cfg_args' 생성
            # model_path 하위에 'cfg_args'라는 파일을 생성(또는 덮어쓰기)합니다.
            cfg_log_f.write(str(Namespace(**vars(_dataset))))
            # _dataset의 모든 속성을 Namespace로 감싸서 문자열로 변환한 뒤 파일에 기록합니다.
            # 이렇게 하면 실험에 사용된 설정값을 나중에 추적할 수 있습니다.
            # 즉, cfg_args = _dataset
        
        scene = Scene(_dataset, gaussians)
        # Scene 객체를 생성. 크게 나눠 가우시안과 프레임 별 카메라 정보 두 가지를 담고 있음.
        
        # 해당 안 됨
        if share_audio_net:
            gaussians.neural_motion_grid.audio_net = motion_net.audio_net
            gaussians.neural_motion_grid.audio_att_net = motion_net.audio_att_net
        
        scene_list.append(scene)
        gaussians.training_setup(opt)
        # 가우시안의 속성(좌표, 회전값, 색상 등) + 보조 네트워크 (`neural_renderer`, `neural_motion_grid`)를 파라미터화 해서 옵티마이저에 등록, 옵티마이저를 생성하고 각 파라미터에 대한 학습률 스케줄러를 설정.
        # 이 때, 가우시안 좌표는 이터레이션에 따라 학습률을 매뉴얼하게 설정하여 부여.

    if not share_audio_net:
        motion_net = MotionNetwork(args=dataset).cuda() # 모션 네트워크 초기화. 여기서 초기화되는 네트워크는 UMF이고, gaussians 객체 내부에 PMF가 존재.
        motion_optimizer = torch.optim.AdamW(motion_net.get_params(5e-3, 5e-4), betas=(0.9, 0.99), eps=1e-8) # 모션 네트워크의 파라미터를 옵티마이저에 저장해서 학습 되도록 함.
        scheduler = torch.optim.lr_scheduler.LambdaLR(motion_optimizer, lambda iter: (0.5 ** (iter / mouth_select_iter)) if iter < mouth_select_iter else 0.1 ** (iter / opt.iterations)) # mouth_select_iter 전후로 다른 감쇠율을 적용 -> 초기에는 빠른 학습률로 수렴 유도, 이후에는 미세하게 조정할 수 있도록 최적화
        ema_motion_net = ExponentialMovingAverage(motion_net.parameters(), decay=0.995) # motion_net의 파라미터를 전달받아 이들의 이동 평균을 기록하는 모듈
    
    lpips_criterion = lpips.LPIPS(net='alex').eval().cuda() # LPIPS 손실을 계산하기 위한 함수 초기화. 모델이 학습을 하면 안 되니 evel()로 설정.

    bg_color = [0, 1, 0]   # [1, 1, 1] # if dataset.white_background else [0, 0, 0], 현재는 배경을 초록색 (G)만 1로 세팅.
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda") # 배경도 텐서로 바꾼 뒤 GPU에 올리기


    iter_start = torch.cuda.Event(enable_timing = True) # CUDA 이벤트 객체인 iter_start 생성. `enable_timing = True`는 이 이벤트가 시간 측정에 사용될 수 있도록 한다.
    iter_end = torch.cuda.Event(enable_timing = True) # CUDA 이벤트 객체인 iter_end 생성.

    viewpoint_stack = None # scene.train_cameras에 있는 정보를 복사할 변수
    ema_loss_for_log = 0.0 # 로그 기록을 위한 EMA(지수이동평균) 손실값을 0.0으로 초기화
    progress_bar = tqdm(range(first_iter, opt.iterations), ascii=True, dynamic_ncols=True, desc="Training progress")
    first_iter += 1 # 첫번째 이터레이션이니 0에서 1로 설정. 이후 반복문에서 1씩 증가.
    for iteration in range(first_iter, opt.iterations + 1): # 1부터 30001까지 반복.

        ## 초반 세팅 (scene에서 가우시안과 카메라 정보 가져오기 등) ##
        iter_start.record()
        cur_scene_idx = randint(0, len(scene_list)-1) # 매 이터레이션마다 마크롱, 오바마, 문재인 중 누구부터 진행할지 랜덤으로 선택 (인물별 순차적 학습이 아님!)
        scene = scene_list[cur_scene_idx] # 선택된 인덱스에 해당하는 Scene 객체를 가져옴
        gaussians = scene.gaussians # 해당 Scene의 가우시안 모델을 for문 안의 변수 gaussian에 할당.
        gaussians.update_learning_rate(iteration) # 가우시안 좌표 X, Y, Z에 대한 러닝 레이트를 업데이트 하기 위해, 현재 이터레이션에 맞는 learning rate를 스케줄러 함수로부터 받아와 할당.
        
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree() # active_sh_degree를 max_sh_degree이전까지 1씩 증가시킨다. 0 -> 1 -> 2
        # Pick a random Camera
        # if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras().copy() # FoV, [R|T], H, W, img, img path, talking dict (img id, img, teeth mask, mask path, 오디오 피처, 액션유닛, 입, 입술, 하관 바운딩 박스 좌표)
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1)) # randint로 랜덤 인덱스를 뽑아, 그에 해당하는 요소를 pop 함수를 통해 가져와 할당


        ## 점진적으로 커지는 입, AU에 대한 추가 샘플링 기능 구현 ##
        # 학습 초반에는 입을 작게 벌린 이미지들을 주로 샘플링하도록 해당 범위를 설정하고, 학습이 진행될수록 입을 더 크게 벌린 이미지들을 샘플링하도록 그 기준 범위를 점차 이동시킨다.
        # 이는 모델이 다양한 입 모양 변화를 효과적으로 학습할 수 있도록 유도하는 전략이다.
        mouth_global_lb = viewpoint_cam.talking_dict['mouth_bound'][0] # [전체 프레임에서 입 내부 최소 벌림, 최대 벌림, 현재 프레임의 입이 벌어진 길이] = [0, 27, 7] 중 최소 벌림 0
        mouth_global_ub = viewpoint_cam.talking_dict['mouth_bound'][1] # 최대 벌림 27
        mouth_global_lb += (mouth_global_ub - mouth_global_lb) * 0.2 # 최소 벌림 값을 전체의 20%만큼 올림 0 + (27 - 0) * 0.2 = 5.4
        mouth_window = (mouth_global_ub - mouth_global_lb) * 0.2 # 입 범위의 20%만큼 윈도우(탐색 범위)를 설정합니다.  (27 - 5.4) * 0.2 = 4.32

        mouth_lb = mouth_global_lb + mouth_step * iteration * (mouth_global_ub - mouth_global_lb)
        # mouth_global_lb(5.4)에 (mouth_step * iteration * (mouth_global_ub - mouth_global_lb))을 더한다.
        # 예를 들어, iteration이 100일 경우, 5.4 + (0.00005 * 100 * 21.6) = 5.4 + 0.108 = 5.508이 mouth_lb가 된다.
        # iteration이 15000일 경우, 5.4 + (0.00005 * 15000 * 21.6) = 5.4 + 16.2 = 21.6이 mouth_lb가 된다.
        # 이 값은 학습 진행 상황에 따라 입 벌림 길이의 하한선을 동적으로 조절한다.

        mouth_ub = mouth_lb + mouth_window
        # 현재 계산된 mouth_lb에 mouth_window(4.32)를 더한다.
        # 예를 들어, iteration이 100일 경우, 5.508 + 4.32 = 9.828이 mouth_ub가 된다.
        # iteration이 15000일 경우, 21.6 + 4.32 = 25.92가 mouth_ub가 된다.
        # 이는 입 벌림 길이의 상한선을 설정한다.

        mouth_lb = mouth_lb - mouth_window
        # 현재 계산된 mouth_lb에서 mouth_window(4.32)를 뺀다.
        # 예를 들어, iteration이 100일 경우, 5.508 - 4.32 = 1.188이 mouth_lb가 된다.
        # iteration이 15000일 경우, 21.6 - 4.32 = 17.28이 mouth_lb가 된다.
        # 이 과정으로 mouth_lb와 mouth_ub는 특정 범위([mouth_lb, mouth_ub])를 형성하며, 이 범위는 mouth_window 크기를 가진다.
        #
        # 최종적으로 mouth_lb와 mouth_ub는 iteration에 따라 아래와 같이 변한다:
        #   - iteration=0      : mouth_lb = 5.4 - 4.32 = 1.08,      mouth_ub = 5.4 + 4.32 = 9.72
        #   - iteration=100    : mouth_lb = 5.508 - 4.32 = 1.188,   mouth_ub = 5.508 + 4.32 = 9.828
        #   - iteration=15000  : mouth_lb = 21.6 - 4.32 = 17.28,    mouth_ub = 21.6 + 4.32 = 25.92
        #   - iteration=max    : mouth_lb와 mouth_ub 모두 오른쪽(더 큰 입 벌림)으로 이동
        # 즉, 학습이 진행될수록 샘플링되는 입 벌림 길이 구간([mouth_lb, mouth_ub])이 점차 더 큰 입 모양을 포함하도록 우측으로 이동한다.

        au_global_lb = 0 # 액션 유닛 하한값
        au_global_ub = 1 # 액션 유닛 상한값
        au_window = 0.3 # 액션 유닛 탐색 윈도우

        au_lb = au_global_lb + mouth_step * iteration * (au_global_ub - au_global_lb) # 액션 유닛 하한값을 학습 시간에 따라 변화
        au_ub = au_lb + au_window # 액션 유닛 상한값을 윈도우에 따라 업데이트
        au_lb = au_lb - au_window * 0.5 # 하한값을 윈도우의 절반만큼 더 낮춤

        # 예시: iteration에 따라 au_lb, au_ub가 어떻게 변하는지
        #   - iteration=0      : au_lb = 0 - 0.15 = -0.15,                                  au_ub = 0 + 0.3 = 0.3
        #   - iteration=100    : au_lb = (0.00005 * 100) - 0.15 = 0.005 - 0.15 = -0.145,    au_ub = 0.005 + 0.3 = 0.305
        #   - iteration=15000  : au_lb = (0.00005 * 15000) - 0.15 = 0.75 - 0.15 = 0.6,      au_ub = 0.75 + 0.3 = 1.05
        #   - iteration=max    : au_lb, au_ub 모두 오른쪽(더 큰 AU)으로 이동
        # 즉, 학습이 진행될수록 샘플링되는 AU 구간([au_lb, au_ub])이 점차 더 큰 AU 값을 포함하도록 우측으로 이동한다.

        if iteration < warm_step and iteration < mouth_select_iter: # 학습 초반 (iteration  1000 <  20000)
        # 학습 초반에는 점진적으로 커지는 입모양을 학습할 수 있도록 매 15번마다 아래와 같은 과정을 추가 학습시킨다.
            if iteration % select_interval == 0: # select_interval 15마다 실행
                while viewpoint_cam.talking_dict['mouth_bound'][2] < mouth_lb or viewpoint_cam.talking_dict['mouth_bound'][2] > mouth_ub: # mouth_lb < 해당 프레임 입벌림 < mouth_ub일 때까지 다시 뽑기
                    if not viewpoint_stack:
                        viewpoint_stack = scene.getTrainCameras().copy() # pop을 너무 많이 해서 다 지워졌을 수도 있으니
                    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1)) # mouth_lb < 해당 프레임 입모양 < mouth_ub일 때까지 다시 뽑기

        if warm_step < iteration < mouth_select_iter: # 학습 중반 (1000 < iteration < 20000)
        # 학습 중반에는 이제 입 샘플링은 따로 하지 않고, AU 샘플링을 비슷한 방법으로 진행한다.
            if iteration % select_interval == 0:
                while viewpoint_cam.talking_dict['blink'] < au_lb or viewpoint_cam.talking_dict['blink'] > au_ub:
                    if not viewpoint_stack:
                        viewpoint_stack = scene.getTrainCameras().copy()
                    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1)) # au_lb < 눈 크기 < au_ub 일 때까지 다시 뽑기


        # Render
        if (iteration - 1) == debug_from: # debug_from값이 0이면, 첫 번째 이터레이션부터 디버그 세팅을 활성화
            pipe.debug = True
        
        # 현재 뷰포인트 캠의 3가지 마스크를 가져와 GPU에 할당.
        face_mask = torch.as_tensor(viewpoint_cam.talking_dict["face_mask"]).cuda() # [512, 512] True/False 마스크
        hair_mask = torch.as_tensor(viewpoint_cam.talking_dict["hair_mask"]).cuda() # [512, 512] True/False 마스크
        mouth_mask = torch.as_tensor(viewpoint_cam.talking_dict["mouth_mask"]).cuda()
        head_mask =  face_mask + hair_mask # 얼굴, 머리카락 마스크를 합친 마스크도 생성.

        if iteration > lpips_start_iter: # lpips_start_iter의 값이 9999999999로 매우 크므로 해당 과정은 스킵.
            max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            mouth_mask = (-max_pool(-max_pool(mouth_mask[None].float())))[0].bool()

        
        hair_mask_iter = (warm_step < iteration < lpips_start_iter - 1000) and iteration % hair_mask_interval != 0
        # 1. `iteration`이 `warm_step`보다 크다.
        # 2. `iteration`이 `lpips_start_iter - 1000`보다 작다. (사실상 고려 X)
        # 3. `iteration`이 `hair_mask_interval` = 7의 배수가 아니다.
        # 특정 학습 구간에서 머리카락 부분에 마스크를 적용하여 나머지 부분만 loss를 계산하도록 할지 여부를 결정하는 플래그로, 셋을 모두 만족할 때 True가 된다.
        # 즉, 7번에 한 번씩은 머리카락 부분을 마스킹하여 지워버리고, 얼굴만 집중해서 복원하도록 한다.

        if iteration < warm_step: # 학습 초기 (1000 iter 이전)
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        elif iteration < p_motion_start_iter: # p_motion_start_iter = 0이므로 해당 분기는 실행 안 됨.
            render_pkg = render_motion(viewpoint_cam, gaussians, motion_net, pipe, background, return_attn=True)
        else:
            render_pkg = render_motion(viewpoint_cam, gaussians, motion_net, pipe, background, return_attn=True, personalized=True)
            # 움직임을 렌더링하기 위해 motion_net가 추가 인자로 입력. `personalized=True` 인자를 추가하여 개인화된 모션 렌더링을 수행한다.
            
        image_white, alpha, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["alpha"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        # `image_white`: 렌더링된 최종 이미지. [3, 512, 512]
        # `alpha`: 알파 채널(투명도) 정보. [1, 512, 512]
        # `viewspace_points`: 뷰스페이스(카메라 시점) 상의 포인트 좌표. [2000, 3]
        # `visibility_filter`: 반지름이 0보다 큰 Gaussian만 포함하는 가시성 필터. [2000]
        # `radii`: 각 가우시안의 2D 반지름. [2000]

        gt_image  = viewpoint_cam.original_image.cuda() / 255.0
        # `viewpoint_cam` 객체에서 원본 이미지(`original_image`)를 가져와 GPU에 할당하고, 픽셀 값을 0-1 범위로 정규화.
        gt_image_white = gt_image * head_mask + background[:, None, None] * ~head_mask
        # `gt_image_white`는 원본 이미지(`gt_image`)에 `head_mask`를 적용하고, 머리 영역이 아닌 부분(`~head_mask`)에는 `background` 색상을 채워 넣어 생성한다.
        
        if iteration > motion_stop_iter: # motion_net의 학습을 멈추는 시점 30000에선 업데이트를 동결 (그러나 해당 스텝이 전체 학습 스텝과 같으므로 스킵)
            for param in motion_net.parameters():
                param.requires_grad = False
        
        # Loss
        if hair_mask_iter: # hair_mask_iter이 True인 iteration에만 실행
            image_white[:, hair_mask] = background[:, None] # 렌더링된 이미지(`image_white`)에서 `hair_mask`에 해당하는 영역의 픽셀 값을 `background` 색상으로 채운다.
            # 이렇게 하여 머리카락 부분은 손실 계산에서 제외하도록 한다.
            # `[:, None]`은 텐서의 차원을 확장하여 브로드캐스팅(broadcasting)한 것.
            gt_image_white[:, hair_mask] = background[:, None] # 실제 이미지 기반의 `gt_image_white`에서도 `hair_mask`에 해당하는 영역의 픽셀 값을 `background` 색상으로 채운다.
            # 이렇게 해야 image_white와 비교가 가능.
        
        # image_white[:, mouth_mask] = 1
        gt_image_white[:, mouth_mask] = background[:, None] # `gt_image_white`에서 `mouth_mask`에 해당하는 영역의 픽셀 값을 `background` 색상으로 채운다.
        # 이는 입 영역을 손실 계산에서 제외하기 위함이다.

        # 초기 손실 계산
        Ll1 = l1_loss(image_white, gt_image_white) # 렌더링된 이미지(`image_white`)와 실제 이미지(`gt_image_white`) 사이의 L1 손실 (두 텐서의 각 요소 차이의 절댓값 합을 나타내며, 이미지 픽셀 값의 직접적인 차이를 측정하는 데 사용)을 계산
        loss = Ll1 + opt.lambda_dssim * (1.0 - ssim(image_white, gt_image_white))
        # L1 손실(`Ll1`)에 SSIM(Structural Similarity Index Measure) 기반의 손실 항을 더한다.
        # `ssim(image_white, gt_image_white)`는 두 이미지의 구조적 유사성을 측정하며, 이 값이 높을수록 이미지가 유사하다는 의미이다.
        # 따라서 `(1.0 - ssim(...))`는 유사성이 낮을수록 높은 손실을 반환하고, `opt.lambda_dssim`은 이 손실 항에 대한 가중치이다.
        # 이 결합 손실 함수는 픽셀 단위의 정확성(L1)과 지각적 유사성(SSIM)을 모두 고려하여 이미지 품질을 최적화한다.


        # 워밍업 이후 render_motion이 학습되기 시작하면, 이에 따른 추가 손실항도 추가된다.
        if iteration > warm_step: # 학습 중반 (1000 iter 이후)
            loss += 1e-5 * (render_pkg['motion']['d_xyz'].abs()).mean()
            loss += 1e-5 * (render_pkg['motion']['d_rot'].abs()).mean()
            loss += 1e-5 * (render_pkg['motion']['d_opa'].abs()).mean()
            loss += 1e-5 * (render_pkg['motion']['d_scale'].abs()).mean()
            # 가우시안 변화량의 절대값을 평균낸 후 작은 계수를 곱하여 손실에 더하여 정규화 항으로 사용
            loss += 1e-3 * (((1-alpha) * head_mask).mean() + (alpha * ~head_mask).mean())
            # 첫 번째 항 `((1-alpha) * head_mask).mean()`은 `head_mask` 영역에서 투명한 부분(`1-alpha`)의 평균을 계산한다.
            # 두 번째 항 `(alpha * ~head_mask).mean()`은 `head_mask` 영역 밖(`~head_mask`)에서 불투명한 부분(`alpha`)의 평균을 계산한다.
            # 이 손실 항은 머리 영역 내부가 불투명하고, 머리 영역 외부가 투명하도록 유도하는 역할을 한다.
            
            # 개인화 모션을 학습할 때의 추가 손실 항            
            if iteration > p_motion_start_iter:
                loss += 1e-5 * (render_pkg['p_motion']['d_xyz'].abs()).mean()
                loss += 1e-5 * (render_pkg['p_motion']['d_rot'].abs()).mean()
                loss += 1e-5 * (render_pkg['p_motion']['d_opa'].abs()).mean()
                loss += 1e-5 * (render_pkg['p_motion']['d_scale'].abs()).mean()
                # loss += 1e-5 * (render_pkg['p_motion']['p_xyz'].abs()).mean()
                # loss += 1e-5 * (render_pkg['p_motion']['p_scale'].abs()).mean()
                # 여전히 가우시안 변화량에 대한 정규화 항은 그대로 유지한다. 다만 여기서 걸리는 손실은 개인 모션에 대한 것이다.
            
                # Contrast
                # 가장 중요한 Negative Contrastive Loss를 계산
                audio_feat = viewpoint_cam.talking_dict["auds"].cuda() # 현재 카메라 시점의 `talking_dict`에서 'auds'(오디오 특징) 값을 가져와 GPU에 할당하고 `audio_feat` 변수에 저장한다.
                exp_feat = viewpoint_cam.talking_dict["au_exp"].cuda() # 현재 카메라 시점의 `talking_dict`에서 'au_exp'(액션 유닛 표현) 값을 가져와 GPU에 할당하고 `exp_feat` 변수에 저장한다.
                p_motion_preds = gaussians.neural_motion_grid(gaussians.get_xyz, audio_feat, exp_feat) # `gaussians` 객체의 `neural_motion_grid` 모델을 사용하여 개인화된 모션 예측(`p_motion_preds`)을 수행한다.
                # # `gaussians.get_xyz`(가우시안의 XYZ 좌표), `audio_feat`, `exp_feat`를 입력으로 사용한다.
                contrast_loss = 0 # 비교 손실을 0으로 초기화

                for tmp_scene_idx in range(len(scene_list)): # 마크롱, 오바마, 문제인에 대한 scene을 순차적으로 순회.
                    if tmp_scene_idx == cur_scene_idx: continue # 마크롱을 마크롱과 비교할 수는 없으니 건너 뛰기.
                    with torch.no_grad(): # 예시를 들어 오바마와 비교한다고 하자.
                        tmp_scene = scene_list[tmp_scene_idx] # 오바마에 해당하는 scene을 가져옴
                        tmp_gaussians = tmp_scene.gaussians # 오바마의 가우시안 모델을 가져옴
                        tmp_p_motion_preds = tmp_gaussians.neural_motion_grid(gaussians.get_xyz, audio_feat, exp_feat) # 동일한 오디오와 표정 특징을 사용하여 오바마의 개인화된 모션 예측을 수행
                    contrast_loss_i = (tmp_p_motion_preds['d_xyz'] * p_motion_preds['d_xyz']).sum(-1)
                    # 오바마와 마크롱의 동일 오디오, 동일 표정에 대해 예측된 모션 변화량을 내적 값을 계산
                     # `.sum(-1)`은 마지막 차원(예: 3D 벡터)에 대해 합을 구하여 벡터 내적과 유사한 효과를 낸다. 이는 두 모션 벡터의 유사도를 측정하는 방식이다.
                    contrast_loss_i[contrast_loss_i < 0] = 0 # `contrast_loss_i` 값이 0보다 작을 경우 0으로 설정한다. 즉, 두 모션 벡터가 유사하지 않거나 반대 방향일 경우 손실을 부여하지 않고, 유사할 경우에만 손실을 통해 조절한다.
                    contrast_loss += contrast_loss_i.mean() # `contrast_loss_i`의 평균을 `contrast_loss`에 더한다.
                loss += contrast_loss # 계산된 `contrast_loss`를 전체 손실(`loss`)에 추가한다.
                # 이 손실은 현재 장면의 모션이 다른 장면의 모션과 너무 유사해지지 않도록 하여, 개인화된 모션의 고유성을 강화하는 역할을 한다.
                # 즉, 현재 배우의 모션이 특정 오디오에 대해 다른 배우의 모션과는 구별되도록 유도한다.

            [xmin, xmax, ymin, ymax] = viewpoint_cam.talking_dict['lips_rect'] # 현재 카메라 시점의 `talking_dict`에서 'lips_rect'(입술 영역의 바운딩 박스 좌표)를 가져와 `xmin`, `xmax`, `ymin`, `ymax` 변수에 할당한다.
            loss += 5e-3 * (render_pkg["attn"][1, xmin:xmax, ymin:ymax]).mean()
            # `render_pkg` 딕셔너리에서 어텐션 맵(`attn`)의 특정 채널(인덱스 1)에 해당하는 입술 영역(`xmin:xmax, ymin:ymax`)의 평균 값에 $5 \times 10^{-3}$를 곱하여 손실에 추가한다.
            # 이는 입술 영역의 어텐션을 특정 방향으로 유도하거나 정규화하는 역할을 한다. 예를 들어, 입술 움직임에 대한 모델의 집중도를 높일 수 있다.

            # 이터레이션 0 이상부터
            if iteration > p_motion_start_iter:
                loss += 5e-3 * (render_pkg["p_attn"][1, xmin:xmax, ymin:ymax]).mean()
                # `render_pkg` 딕셔너리에서 개인화된 어텐션 맵(`p_attn`)의 특정 채널(인덱스 1)에 해당하는 입술 영역의 평균 값에 $5 \times 10^{-3}$를 곱하여 손실에 추가한다.
                # gaussian_renderer.py에서 render_motion 함수를 보면 개인화된 어텐션 맵(`p_attn`)의 1번 인덱스는 얼굴 표정 어텐션에 따른 가우시안 움직임이다.
                # 그러나 입술의 모양은 얼굴 표정이 아닌 오디오에 따른 움직임에 더 큰 영향을 받아야 하니, 입술 영역에 부여된 얼굴 표정 기반 어텐션은 작아져야 한다.
                # 즉, 얼굴 표정이 입술 모양이 미치는 영향 (어텐션)을 줄이기 위해, 이 값에 가중치를 부여한 뒤 loss에 추가하는 것이다.

            if not hair_mask_iter: # 머리카락을 마스킹하지 않을 때는
                loss += 1e-4 * (render_pkg["attn"][1][hair_mask]).mean() # 어텐션 맵(`attn`)의 얼굴 표정(인덱스 1)에서 `hair_mask`에 해당하는 영역의 평균 값에 $1 \times 10^{-4}$를 곱하여 손실에 추가
                loss += 1e-4 * (render_pkg["attn"][0][hair_mask]).mean() # 어텐션 맵(`attn`)의 목소리(인덱스 0)에서 `hair_mask`에 해당하는 영역의 평균 값에 $1 \times 10^{-4}$를 곱하여 손실에 추가
                # UMF가 머리카락 영역의 가우시안 움직임을 계산할 때, 얼굴 표정이나 목소리에 영향을 받으면 안 된다. (머리카락이 목소리에 따라 바뀌는 것은 안 되므로)
                # 따라서 머리카락 영역에 대한 얼굴 표정과 목소리 어텐션 값을 계산한 뒤, 이 값에 가중치를 부여한 뒤 loss에 추가하는 것이다.

            # loss += l2_loss(image_white[:, xmin:xmax, ymin:ymax], image_white[:, xmin:xmax, ymin:ymax])
            # 이 줄은 렌더링된 이미지의 입술 영역에 대한 L2 손실을 추가하는 주석 처리된 코드이다. 현재는 비활성화되어 있다.

        image_t = image_white.clone()
        # 렌더링된 이미지 `image_white`를 복사하여 `image_t`에 저장한다.
        # `clone()` 메서드는 원본 텐서와 독립적인 복사본을 생성한다.
        # 이는 이후 `image_white`가 변경될 수 있더라도 `image_t`는 원래 값을 유지하도록 하기 위함이다.
        gt_image_t = gt_image_white.clone()
        # 실제 이미지 기반의 `gt_image_white`를 복사하여 `gt_image_t`에 저장한다.


        loss.backward()
        # 계산된 최종 손실(`loss`)에 대해 역전파(backpropagation)를 수행한다.
        # 이 과정에서 requires_grad=True로 설정된 네트워크들의 모든 파라미터에 대한 그래디언트가 계산되어 저장된다.
        # 여기에는 가우시안의 속성(위치, 색상, 투명도 등), PMF 파라미터, UMF 파라미터 모두 포함된다.
        # 해당 코드의 훈련 과정에서, 따로 contrast_loss.backward()를 통해 PMF의 파라미터만을 업데이트하진 않는다.

        # 그럼에도 불구하고 UMF가 보편적인 움직임을 PMF와 분리하여 학습할 수 있는 이유는 EMA가 적용돼있기 때문이다.
        # EMA는 학습 과정에서 UMF 파라미터들의 변화를 부드럽게 평균화한다.
        # 이는 학습 데이터에 존재하는 개별 인물의 미세하고 노이즈가 많은 변동성을 완화하여, UMF가 더 안정적이고 일반화 성능이 좋은 보편적인 움직임 특징을 추출하도록 돕는다.
        # 평가나 추론 시 EMA가 적용된 UMF를 사용함으로써, 특정 인물에게 과적합되지 않은, 모든 인물에게 잘 작동하는 베이스 모션을 제공할 수 있게 된다.
        # 반면 EMA가 적용되지 않은 PMF의 파라미터는 각 학습 스텝에서 현재 배치의 손실에 따라 더 민감하고 직접적으로 업데이트된다.
        # 이는 PMF가 개인별로 다를 수 있는 섬세하고 때로는 불규칙한 움직임 특징까지 유연하게 포착하고 학습할 수 있도록 한다. 

        iter_end.record()
        # `iter_end` CUDA 이벤트 객체에 현재 시간을 기록한다.
        # 이는 현재 학습 반복(iteration)이 종료되는 시점을 기록하여, 이전에 기록된 `iter_start`와 함께 각 반복에 소요된 시간을 측정하는 데 사용된다.

        with torch.no_grad():
        # 메모리 사용량을 줄이고 계산 속도를 높이며, 학습과는 무관한 부가 작업에서 모델 파라미터가 실수로 업데이트되는 것을 방지하기 위해 아래부터 그래디언트 추적 비활성화.
            
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

            # if (iteration in saving_iterations):
            #     print("\n[ITER {}] Saving Gaussians".format(iteration))
            #     scene.save(str(iteration)+'_face')

            if (iteration in checkpoint_iterations): # 현재 반복 횟수(`iteration`)가 `checkpoint_iterations` 리스트의 원소에 해당한다면
                print("\n[ITER {}] Saving Checkpoint".format(iteration)) # 체크포인트를 저장한다는 메시지를 출력
                ckpt = (motion_net.state_dict(), motion_optimizer.state_dict(), iteration)
                # `motion_net`의 상태 딕셔너리, `motion_optimizer`의 상태 딕셔너리, 현재 반복 횟수(`iteration`)를 묶어 `ckpt` 변수에 저장
                # `state_dict()`는 모델이나 옵티마이저의 학습 가능한 파라미터들을 딕셔너리 형태로 반환
                torch.save(ckpt, dataset.model_path + "/chkpnt_face_latest" + ".pth")
                # `ckpt`를 `.pth` 확장자를 가진 파일로 저장
                # `dataset.model_path`는 모델이 저장될 경로이며, `chkpnt_face_latest`는 최신 체크포인트 파일 이름을 나타냄
                with ema_motion_net.average_parameters():
                # `ema_motion_net` 객체의 `average_parameters()` 컨텍스트를 사용하여 내부 블록을 실행
                # 이 컨텍스트 내에서는 `motion_net`의 파라미터들이 EMA(Exponential Moving Average) 값으로 일시적으로 대체
                    ckpt_ema = (motion_net.state_dict(), motion_optimizer.state_dict(), iteration)
                    # EMA 파라미터로 대체된 `motion_net`의 상태 딕셔너리, `motion_optimizer`의 상태 딕셔너리, 현재 반복 횟수를 묶어 `ckpt_ema` 변수에 저장
                    torch.save(ckpt, dataset.model_path + "/chkpnt_ema_face_latest" + ".pth")
                    # `ckpt_ema` (EMA 기반의 체크포인트)를 `.pth` 확장자를 가진 파일로 저장
                for _scene in scene_list:
                # `scene_list`의 각 장면(`_scene`)에 대해 반복문을 실행
                    _gaussians = _scene.gaussians # 현재 장면(`_scene`)의 가우시안 모델을 `_gaussians` 변수에 할당
                    ckpt = (_gaussians.capture(), motion_net.state_dict(), motion_optimizer.state_dict(), iteration)
                    # 현재 가우시안 모델의 상태(`_gaussians.capture()`), `motion_net`의 상태 딕셔너리, `motion_optimizer`의 상태 딕셔너리, 현재 반복 횟수를 묶어 `ckpt` 변수에 저장
                    torch.save(ckpt, _scene.model_path + "/chkpnt_face_" + str(iteration) + ".pth")
                    # 현재 장면의 특정 반복 횟수에 해당하는 체크포인트를 저장
                    torch.save(ckpt, _scene.model_path + "/chkpnt_face_latest" + ".pth")
                    # 현재 장면의 최신 체크포인트를 저장

            # Densification
            if iteration < opt.densify_until_iter: # 현재 반복 횟수(`iteration`)가 `opt.densify_until_iter` 29000 변수보다 작은지
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                # `gaussians` 객체의 `max_radii2D` 속성을 업데이트
                # `visibility_filter`에 해당하는 가우시안들에 대해, 기존의 `max_radii2D` 값과 현재 렌더링된 `radii` 값 중 더 큰 값을 선택하여 저장
                # 이는 이미지 공간에서 각 가우시안이 차지했던 최대 크기를 추적하여, 이후 가지치기 시 불필요하게 작은 가우시안을 제거하는 데 사용
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                # `gaussians` 객체의 `add_densification_stats` 메서드를 호출하여 밀집화 관련 통계를 업데이트
                # `viewspace_point_tensor`(뷰스페이스 포인트)와 `visibility_filter`(가시성 필터)를 사용하여, 어떤 가우시안들이 뷰에 기여했는지, 그리고 어떤 영역에 새로운 가우시안을 추가할지 등을 결정하기 위한 통계 정보를 수집

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                # `iteration`이 `opt.densify_from_iter` 500 보다 크고 `opt.densification_interval` 100의 배수일 경우에만 다음 줄을 실행하는 조건문
                # 이는 특정 반복 횟수 이후에, 일정 간격으로 밀집화 및 가지치기 작업을 수행하도록 한다.
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    # `size_threshold` 변수를 정의
                    # `iteration`이 `opt.opacity_reset_interval` 3000보다 크면 `20`으로 설정하고, 그렇지 않으면 `None`으로 설정
                    # 이 값은 가지치기 시 가우시안의 크기를 기준으로 제거할지 말지를 결정하는 임계값으로 사용
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.05 + 0.25 * iteration / opt.densify_until_iter, scene.cameras_extent, size_threshold)
                    # `gaussians` 객체의 `densify_and_prune` 메서드를 호출하여 가우시안의 밀집화 및 가지치기를 수행
                    # `opt.densify_grad_threshold`: 그래디언트 임계값을 기준으로 가우시안을 복제하거나 분할하는 데 사용
                    # `0.05 + 0.25 * iteration / opt.densify_until_iter`: 불투명도(opacity) 임계값을 동적으로 조절
                    # 학습이 진행됨에 따라 임계값이 증가하여 더 많은 가우시안이 제거될 수 있다.
                    # `scene.cameras_extent`: 장면의 카메라 범위로, 새로운 가우시안을 추가할 때의 공간적 제한을 설정
                    # `size_threshold`: 앞서 계산된 `size_threshold` 값으로, 이미지 공간에서의 크기를 기준으로 가우시안을 제거하는 데 사용
            
            # bg prune
            # 배경 가우시안을 가지치기하는 부분
            if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            # 밀집화 과정과 동일하게, iteration이 opt.densify_from_iter (예시에서 500)보다 크고 opt.densification_interval (예시에서 100)의 배수일 때만 배경 가지치기가 실행된다.
            # 이는 학습 초반에는 가우시안을 안정화하고, 이후 일정 주기마다 배경을 정리하는 전략이다.
                from utils.sh_utils import eval_sh
                # `eval_sh` 함수를 `utils.sh_utils` 모듈로부터 임포트한다.
                # eval_sh 함수는 가우시안의 구면 고조파 계수와 특정 방향 벡터를 입력받아 해당 방향에서 관측될 색상(RGB)을 계산하는 데 사용된다.

                shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree+1)**2)
                # `gaussians`의 특징(`get_features`)을 재구성하여 뷰 의존적인 구면 고조파 계수(`shs_view`)를 준비한다.
                dir_pp = (gaussians.get_xyz - viewpoint_cam.camera_center.repeat(gaussians.get_features.shape[0], 1))
                # 각 가우시안의 중심(`gaussians.get_xyz`)에서 카메라 중심(`viewpoint_cam.camera_center`)으로 향하는 방향 벡터(`dir_pp`)를 계산한다.
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                # 방향 벡터를 정규화하여 단위 벡터(`dir_pp_normalized`)를 생성한다.
                sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
                # 정규화된 방향 벡터를 사용하여 구면 고조파 계수(`shs_view`)를 RGB 색상(`sh2rgb`)으로 평가한다.
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
                # 평가된 RGB 색상에 $0.5$를 더하고, 최소값을 $0.0$으로 클램핑하여 최종 사전 계산된 색상(`colors_precomp`)을 얻는다.
                # 이는 색상 값을 $0$에서 $1$ 사이의 유효한 범위로 조정한다.

                bg_color_mask = (colors_precomp[..., 0] < 30/255) * (colors_precomp[..., 1] > 225/255) * (colors_precomp[..., 2] < 30/255)
                # 사전 계산된 색상(`colors_precomp`)을 기반으로 배경색 마스크(`bg_color_mask`)를 생성한다.
                # 이 마스크는 RGB 채널이 각각 (빨강 < 30/255), (초록 > 225/255), (파랑 < 30/255) 조건을 만족하는 가우시안들을 식별한다.
                # 즉, 강한 초록색을 띠는 가우시안(여기서는 `bg_color = [0, 1, 0]`와 유사한 색상)을 배경으로 간주하여 제거하기 위한 마스크이다.
                
                gaussians.prune_points(bg_color_mask.squeeze())
                # `gaussians` 객체의 `prune_points` 메서드를 호출하여 `bg_color_mask`에 의해 식별된 가우시안들을 제거한다.
                # `.squeeze()`는 마스크 텐서의 차원을 줄여 `prune_points` 함수에 적합한 형태로 만든다.

            # Optimizer step
            # 옵티마이저 스텝 부분
            if iteration < opt.iterations:
            # 현재 반복 횟수(`iteration`)가 총 반복 횟수(`opt.iterations`)보다 작은지 확인하는 조건문
            # 이는 최종 반복 횟수에 도달하기 전까지만 옵티마이저 스텝을 수행하도록 한다.
                motion_optimizer.step()
                # `motion_optimizer`를 사용하여 UMF의 파라미터들을 업데이트한다.
                # 이전에 계산된 그래디언트를 기반으로 파라미터들을 조정하여 손실을 최소화한다.
                gaussians.optimizer.step()
                # `gaussians.optimizer`를 사용하여 가우시안 모델의 파라미터들을 업데이트한다.
                motion_optimizer.zero_grad()
                # `motion_optimizer`의 모든 파라미터에 저장된 그래디언트를 $0$으로 초기화한다.
                # 이는 다음 학습 반복에서 새로운 그래디언트를 깨끗하게 계산하기 위함이다.
                gaussians.optimizer.zero_grad(set_to_none = True)
                # `gaussians.optimizer`의 모든 파라미터에 저장된 그래디언트를 `None`으로 초기화한다.
                # `set_to_none=True`는 그래디언트를 $0$으로 채우는 대신 `None`으로 설정하여 메모리 사용량을 약간 더 최적화할 수 있다.

                scheduler.step()
                # 학습률 스케줄러(`scheduler`)를 한 단계 업데이트한다.
                # 이는 미리 정의된 정책에 따라 학습률을 조정하여 학습이 진행됨에 따라 학습률을 변경한다.
                ema_motion_net.update()
                # `ema_motion_net` 객체를 업데이트한다.
                # `motion_net`의 현재 파라미터들을 사용하여 EMA(Exponential Moving Average) 값을 갱신한다.



'''
`prepare_output_and_logger(args)` 함수
이 함수는 학습을 시작하기 전에 필요한 출력 디렉토리와 로깅 환경을 설정하는 역할을 한다.

*   주요 기능:
    *   모델 학습의 결과물(체크포인트, 로그 등)이 저장될 고유한 폴더 경로를 생성하고, 이 경로를 출력한다.
    *   또한, TensorBoard를 사용하여 학습 진행 상황을 시각적으로 기록하기 위한 `SummaryWriter` 객체를 초기화한다.
    *   `args` 객체에 포함된 학습 설정 인자들을 파일로 저장하여 학습 재현성을 높이는 역할도 수행한다.
    *   TensorBoard가 설치되어 있지 않으면 콘솔 메시지를 통해 로깅이 불가능함을 알린다.
'''
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

'''
`training_report(tb_writer, iteration, ..., renderArgs)` 함수
이 함수는 학습 과정 중 주기적으로 모델의 성능을 평가하고, 그 결과를 로깅하며, 시각적인 피드백을 제공하는 역할을 한다.

*   주요 기능:
    *   학습 지표 로깅: `tb_writer` (TensorBoard SummaryWriter)를 사용하여 현재 `iteration`에서의 L1 손실, 전체 손실, 반복 시간 등 학습 관련 스칼라 지표들을 TensorBoard에 기록한다.
    *   평가 및 샘플 렌더링: 미리 정의된 `testing_iterations`에 현재 `iteration`이 포함될 경우, 모델의 성능을 평가한다. 이를 위해 테스트 카메라와 학습 카메라의 일부를 선택하여 실제로 렌더링을 수행한다.
    *   성능 지표 계산 및 로깅: 렌더링된 이미지와 실제 이미지(`ground_truth`)를 비교하여 L1 손실과 PSNR(Peak Signal-to-Noise Ratio) 같은 평가 지표를 계산하고, 이 값들을 TensorBoard에 기록한다.
    *   시각화 자료 저장: 렌더링된 샘플 이미지, 개인화된 렌더링 결과, 실제 이미지, 깊이 맵, 마스크(입 마스크), 법선 맵, 그리고 모션 네트워크의 어텐션 맵 등 다양한 시각화 자료들을 TensorBoard에 이미지 형태로 저장하여 모델의 작동 방식과 학습 진도를 개발자가 쉽게 확인할 수 있도록 돕는다.
    *   가우시안 통계 로깅: `scene.gaussians` 객체의 불투명도 히스토그램이나 총 가우시안 포인트 수와 같은 모델의 내부 상태 통계도 TensorBoard에 기록한다.
    *   GPU 캐시 비우기: 평가 작업 후에는 `torch.cuda.empty_cache()`를 호출하여 GPU 메모리 캐시를 비워 다음 학습 반복을 위한 메모리 공간을 확보한다.
'''
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
                    render_pkg_p = None
                    if renderFunc is render:
                        render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    else:
                        render_pkg = renderFunc(viewpoint, scene.gaussians, motion_net, return_attn=True, frame_idx=0, *renderArgs)
                        render_pkg_p = renderFunc(viewpoint, scene.gaussians, motion_net, return_attn=True, frame_idx=0, personalized=True, *renderArgs)

                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    alpha = render_pkg["alpha"]
                    normal = render_pkg["normal"] * 0.5 + 0.5
                    
                    # image = image - renderArgs[1][:, None, None] * (1.0 - alpha) + background[:, None, None].cuda() / 255.0 * (1.0 - alpha)
                    image = image
                    # gt_image = torch.clamp(viewpoint.original_image.to("cuda") / 255.0, 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda") / 255.0, 0.0, 1.0) * alpha + renderArgs[1][:, None, None] * (1.0 - alpha)
                    
                    mouth_mask = torch.as_tensor(viewpoint.talking_dict["mouth_mask"]).cuda()
                    max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
                    mouth_mask_post = (-max_pool(-max_pool(mouth_mask[None].float())))[0].bool()
                    
                    if tb_writer and (idx < 10):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if render_pkg_p is not None:
                            tb_writer.add_images(config['name'] + "_view_{}/render_p".format(viewpoint.image_name), render_pkg_p['render'][None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                        # tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), (render_pkg["depth"] / render_pkg["depth"].max())[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/mouth_mask_post".format(viewpoint.image_name), (~mouth_mask_post * gt_image)[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/mouth_mask".format(viewpoint.image_name), (~mouth_mask[None] * gt_image)[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/normal".format(viewpoint.image_name), normal[None], global_step=iteration)
                        # tb_writer.add_images(config['name'] + "_view_{}/normal_mono".format(viewpoint.image_name), (viewpoint.talking_dict["normal"]*0.5+0.5)[None], global_step=iteration)

                        if renderFunc is not render:
                            tb_writer.add_images(config['name'] + "_view_{}/attn_a".format(viewpoint.image_name), (render_pkg["attn"][0] / render_pkg["attn"][0].max())[None, None], global_step=iteration)  
                            tb_writer.add_images(config['name'] + "_view_{}/attn_e".format(viewpoint.image_name), (render_pkg["attn"][1] / render_pkg["attn"][1].max())[None, None], global_step=iteration)  
                            if render_pkg_p is not None:
                                tb_writer.add_images(config['name'] + "_view_{}/p_attn_a".format(viewpoint.image_name), (render_pkg_p["p_attn"][0] / render_pkg_p["p_attn"][0].max())[None, None], global_step=iteration)  
                                tb_writer.add_images(config['name'] + "_view_{}/p_attn_e".format(viewpoint.image_name), (render_pkg_p["p_attn"][1] / render_pkg_p["p_attn"][1].max())[None, None], global_step=iteration)  
                                
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
    # 명령줄 인자 파서를 설정함
    parser = ArgumentParser(description="Training script parameters")  # ArgumentParser 객체를 생성하고, 설명을 추가함
    lp = ModelParams(parser)  # 모델 관련 인자들을 파서에 추가하는 객체 생성
    op = OptimizationParams(parser)  # 최적화 관련 인자들을 파서에 추가하는 객체 생성
    pp = PipelineParams(parser)  # 파이프라인 관련 인자들을 파서에 추가하는 객체 생성
    parser.add_argument('--ip', type=str, default="127.0.0.1")  # --ip 인자를 추가, 기본값은 127.0.0.1 (로컬호스트)
    parser.add_argument('--port', type=int, default=6009)  # --port 인자를 추가, 기본값은 6009
    parser.add_argument('--debug_from', type=int, default=-1)  # --debug_from 인자를 추가, 기본값을 -1로 두어 훈련 내내 디버깅을 하지 않음. 모델 훈련 코드에서, iteration - 1부터 디버깅을 시작하도록 설정해둠
    parser.add_argument('--detect_anomaly', action='store_true', default=False)  # action='store_true'는 이 인자가 명령줄에 있으면 True, 없으면 False로 설정됨
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])  # nargs="+"는 인자를 하나 이상 받을 수 있다는 뜻. 따라서 자료형도 리스트 []로 되어있으며, 해당 변수는 몇 번째 이터레이션마다 테스트를 할지를 정하는 변수임
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])  # --save_iterations 인자. 마찬가지로 리스트 [] 형태로 여러 개의 int를 받을 수 있음
    parser.add_argument("--quiet", action="store_true")  # --quiet 플래그 추가, True면 출력 최소화. 디폴트는 False임
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])  # --checkpoint_iterations 인자, 여러 개의 int를 받을 수 있음, 기본값은 빈 리스트
    parser.add_argument("--start_checkpoint", type=str, default = None)  # --start_checkpoint 인자, 체크포인트 파일 경로, 이어서 학습할 경우 사용
    parser.add_argument('--share_audio_net', action='store_true', default=False)  # --share_audio_net 플래그 추가, True면 오디오 네트워크 공유
    args = parser.parse_args(sys.argv[1:])  # 명령줄 인자를 파싱하여 args에 저장
    args.save_iterations.append(args.iterations)  # 빈 리스트에 이폴트 이터레이션 값인 30000 추가
    
    print("Optimizing " + args.model_path)  # 최적화할 모델 경로를 출력함

    # 시스템 상태(난수 시드 등) 초기화
    safe_state(args.quiet)  # quiet 모드에 따라 출력 조정 및 시드 고정. 디폴트는 False

    # GUI 서버 시작, 트레이닝 설정 및 실행
    torch.autograd.set_detect_anomaly(args.detect_anomaly)  # autograd anomaly detection을 옵션에 따라 활성화 default = False
    training(
        lp.extract(args),  # 모델 파라미터 추출
        op.extract(args),  # 최적화 파라미터 추출
        pp.extract(args),  # 파이프라인 파라미터 추출
        args.test_iterations,  # 테스트 iteration 리스트
        args.save_iterations,  # 저장 iteration 리스트
        args.checkpoint_iterations,  # 체크포인트 iteration 리스트
        args.start_checkpoint,  # 시작 체크포인트 경로
        args.debug_from,  # 디버깅 시작 iteration
        args.share_audio_net  # 오디오 네트워크 공유 여부
    )

    # 트레이닝 완료 메시지 출력
    print("\nTraining complete.")  # 트레이닝이 끝났음을 알림