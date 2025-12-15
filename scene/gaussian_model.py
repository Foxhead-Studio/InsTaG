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
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.neural_renderer import GridRenderer
from scene.motion_net import PersonalizedMotionNetwork

class GaussianModel:

    def setup_functions(self):
        # setup_functions 함수는 GaussianModel에서 사용할 다양한 활성화 함수와 변환 함수를 멤버 변수(클래스 내부에서 self.로 접근하는 변수)로 등록하는 역할을 한다.
        # 예를 들어, scaling, opacity, rotation, covariance 등에 대한 forward/역변환 함수를 미리 지정해둔다.
        # 이렇게 하면 이후 코드에서 self.scaling_activation(x)처럼 일관된 방식으로 함수 호출이 가능하다.

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            # build_covariance_from_scaling_rotation 함수는 scaling, scaling_modifier, rotation을 받아서 가우시안의 3D 공분산 행렬을 계산하는 함수이다.
            # scaling_modifier * scaling을 통해 스케일 값을 조정하고, build_scaling_rotation으로 변환 행렬 L을 만든다.
            # L @ L.transpose(1, 2)는 L과 그 전치행렬을 곱해 실제 공분산 행렬을 만든다.
            # strip_symmetric 함수로 대칭 행렬의 불필요한 중복을 제거한다.
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.nn.functional.softplus # scaling 값을 양수로 만드는 활성화 함수이다. (softplus는 exp와 비슷하지만 더 완만하다.)
        self.scaling_inverse_activation = lambda x: x + torch.log(-torch.expm1(-x)) # softplus의 역함수이다. (입력값을 원래 스케일 파라미터로 되돌린다.)

        self.covariance_activation = build_covariance_from_scaling_rotation # scaling, rotation으로부터 공분산을 만드는 함수이다.

        self.opacity_activation = torch.sigmoid # opacity(불투명도)를 0~1로 정규화하는 활성화 함수이다.
        self.inverse_opacity_activation = inverse_sigmoid # sigmoid의 역함수로, 0~1 범위의 값을 원래 파라미터로 되돌린다.

        self.rotation_activation = torch.nn.functional.normalize # rotation(회전 쿼터니언 등)을 정규화(단위 벡터화)하는 함수이다.

    
    def __init__(self, args):
        # args = {'sh_degree': 2, 'source_path': '/home/white/github/InsTaG/data/pretrain',
                # 'model_path': 'output/debug_01', 'images': 'images', 'resolution': -1, 'white_background': False,
                # 'data_device': 'cpu', 'eval': False, 'audio': '', 'init_num': 2000, 'N_views': -1, 'audio_extractor': 'deepspeech',
                # 'type': 'face', 'preload': True, 'all_for_train': False}

        self.args = args  # GaussianModel 클래스의 생성자에서 전달받은 설정 객체(args)를 인스턴스 변수로 저장합니다.

        self.active_sh_degree = 0  # 현재 활성화된 구면 조화 함수(Spherical Harmonics)의 차수를 0으로 초기화합니다.
                                   # 구면 조화 함수는 3D 공간에서 색상 표현 등에 사용됩니다.

        self.max_sh_degree = args.sh_degree  # 최대 구면 조화 함수 차수를 args에서 받아와 저장합니다.
                                             # 예시: args.sh_degree가 2라면 self.max_sh_degree도 2가 됩니다.

        self._xyz = torch.empty(0)  # 3D 가우시안의 중심 좌표를 저장할 텐서를 빈 상태로 초기화합니다.
                                    # 예시: shape이 [N, 3]이 될 수 있습니다. (여기서는 아직 N=0)

        self._features_dc = torch.empty(0)  # 가우시안의 색상 특징 중 DC(평균 색상) 성분을 저장할 텐서를 빈 상태로 초기화합니다.
                                            # 예시: shape이 [N, 3] (RGB)일 수 있습니다.

        self._features_rest = torch.empty(0)  # 가우시안의 색상 특징 중 나머지(SH 계수) 성분을 저장할 텐서를 빈 상태로 초기화합니다.
                                              # 예시: shape이 [N, (SH 차수에 따른 채널 수)]가 될 수 있습니다.

        self._identity = torch.empty(0)  # 각 가우시안의 ID 또는 식별 정보를 저장할 텐서를 빈 상태로 초기화합니다.
                                         # 예시: shape이 [N, K] (K는 identity 차원)일 수 있습니다.

        self._scaling = torch.empty(0)  # 각 가우시안의 스케일(크기) 정보를 저장할 텐서를 빈 상태로 초기화합니다.
                                        # 예시: shape이 [N, 3] (x, y, z축 scaling)일 수 있습니다.

        self._rotation = torch.empty(0)  # 각 가우시안의 회전(방향) 정보를 저장할 텐서를 빈 상태로 초기화합니다.
                                         # 예시: shape이 [N, 4] (쿼터니언)일 수 있습니다.

        self._opacity = torch.empty(0)  # 각 가우시안의 불투명도(opacity) 정보를 저장할 텐서를 빈 상태로 초기화합니다.
                                        # 예시: shape이 [N, 1]일 수 있습니다.

        self.max_radii2D = torch.empty(0)  # 2D 투영 시 가우시안의 최대 반지름 정보를 저장할 텐서를 빈 상태로 초기화합니다.
                                           # 예시: shape이 [N]일 수 있습니다.

        self.xyz_gradient_accum = torch.empty(0)  # xyz 좌표에 대한 그래디언트 누적값을 저장할 텐서를 빈 상태로 초기화합니다.

        self.denom = torch.empty(0)  # 그래디언트 누적을 위한 분모 값을 저장할 텐서를 빈 상태로 초기화합니다.

        self.optimizer = None  # 모델 파라미터를 최적화할 옵티마이저를 아직 생성하지 않고 None으로 초기화합니다.
                               # 실제 옵티마이저는 training_setup 등에서 할당됩니다.

        self.percent_dense = 0  # 덴시피케이션과 관련된 밀집도(density) 임계값을 0으로 초기화합니다.
                                # 예시: 일정 임계값 이상일 때만 덴시피케이션이 일어날 수 있습니다.

        self.spatial_lr_scale = 0  # 공간 학습률 스케일(spatial learning rate scale)을 0으로 초기화합니다.
                                   # 예시: 위치 파라미터에 대한 학습률 조정에 사용됩니다.

        self.neural_renderer = None  # 신경 렌더러(GridRenderer) 객체를 아직 생성하지 않고 None으로 초기화합니다.
                                     # 예시: create_from_pcd, restore 등에서 실제로 할당됩니다.

        self.neural_motion_grid = None  # 신경 모션 그리드(PersonalizedMotionNetwork) 객체를 아직 생성하지 않고 None으로 초기화합니다.
                                        # 예시: create_from_pcd, restore 등에서 self.args를 인자로 하여 할당됩니다.

        self.setup_functions()  # 모델에서 사용할 활성화 함수 및 covariance 계산 함수를 설정하는 메서드를 호출합니다.
                                # 예시: scaling, covariance, opacity, rotation에 대한 활성화 함수가 정의됩니다.

    # 현재 가우시안의 상태를 저장하는 함수
    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._identity,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.neural_renderer.state_dict(),
            self.neural_motion_grid.state_dict(),
        )
    
    # 저장된 가우시안을 불러올 때 사용하는 함수
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._identity,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale,
        neural_renderer_state,
        neural_motion_grid_state) = model_args
        # 가우시안의 모든 속성을 복원
        
        if neural_renderer_state is not None:
            self.neural_renderer = GridRenderer() # 뉴럴 렌더러 복원
            self.neural_renderer.recover_from_ckpt(neural_renderer_state)
            self.neural_renderer.cuda()
        if neural_motion_grid_state is not None:
            self.neural_motion_grid = PersonalizedMotionNetwork(args=self.args).cuda() # PMF 복원
            self.neural_motion_grid.load_state_dict(neural_motion_grid_state)
            self.neural_motion_grid.cuda()
        if training_args is not None: # 훈련 정보도 있으면 복원
            self.training_setup(training_args)
            self.optimizer.load_state_dict(opt_dict) # 옵티마이저 복원.
        self.xyz_gradient_accum = xyz_gradient_accum # xyz_gradient_accum (XYZ 좌표의 그래디언트 누적 값) 복원
        self.denom = denom # denom (밀집화 과정에서 사용되는 분모 값) 복원
        # 이 두 변수들은 가우시안 밀집화(densification) 및 가지치기(pruning) 과정에서 사용된다.

    @property
    # @property는 이 메서드를 "함수처럼 호출"하지 않고 "속성처럼 접근"할 수 있게 해주는 파이썬 데코레이터
    # 즉, obj.get_scaling()이 아니라 obj.get_scaling처럼 쓸 수 있게 해준다.
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_identity(self):
        return self._identity
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    # 차수를 최대 이전까지 올리는 함수
    # 모델 훈련 과정에서 1000 iteration마다 하나씩 올린다.
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    # 랜덤 초기화 된 포인트 클라우드에서 가우시안을 생성하는 함수.
    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        # 모델 훈련 코드에서 scene 인스턴스를 만들 때, __init__.py에서 self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)로 호출하면,
        # pcd는 scene_info.point_cloud(즉, 초기 포인트 클라우드 객체)이고,
        # spatial_lr_scale은 self.cameras_extent(즉, 씬의 공간 범위, float) 값이 들어온다.

        self.spatial_lr_scale = spatial_lr_scale
        # spatial_lr_scale(카메라 공간 범위, 예: 1.0~2.0 등)을 멤버 변수로 저장한다.
        # 이 값은 이후 학습률 스케줄링 등에 사용된다. 씬의 크기가 크면 가우시안 위치를 조정하는 학습률을 더 작게 만든다.

        # <인스턴스 변수를 만들기 위해 사용하는 텐서들>
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        # pcd.points는 (N, 3) 형태의 numpy array로, 각 포인트의 3D 좌표(x, y, z)이다.
        # 이를 torch 텐서로 변환하고 float32 타입으로 바꾼 뒤, GPU로 올린다.
        # 예시: [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], ...] → torch.Size([N, 3])

        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        # pcd.colors는 (N, 3) 형태의 numpy array로, 각 포인트의 RGB 색상이다.
        # torch 텐서로 변환 후, RGB2SH 함수로 Spherical Harmonics(SH) 계수로 변환한다.
        # SH 변환은 색상을 SH 표현(0차 계수)로 바꿔준다.
        # 예시: [[0.8, 0.7, 0.6], ...] → SH 계수로 변환됨

        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        # self.max_sh_degree = 2일 때, 마지막 차원인 (self.max_sh_degree + 1) ** 2 = (2 + 1) ** 2 = 9가 된다.
        # 따라서 features의 shape은 (N, 3, 9)가 된다.
        # 여기서 N은 포인트 개수, 3은 RGB 채널, 9는 SH 계수 개수(0차:1, 1차:3, 2차:5 → 총 9개).
        # 즉, features[n, c, k]에서 n은 포인트 인덱스, c는 RGB 채널, k는 SH 계수 인덱스(0~8)를 의미한다.
        # 예시: features[0, 0, :]은 0번 포인트의 R 채널 SH 계수 9개, features[0, 1, :]은 G 채널, features[0, 2, :]은 B 채널 SH 계수 9개를 담는다.

        features[:, :3, 0 ] = fused_color
        # features의 0번째 SH 계수(DC 성분)에 RGB 색상을 할당한다.
        # 즉, 각 포인트의 SH 0차 계수에 RGB 값을 넣는다.

        features[:, 3:, 1:] = 0.0
        # features의 1차 이후 SH 계수(1~)는 0으로 초기화한다.
        # (실제로 features는 (N, 3, SH차수^2)라서 3:는 의미 없지만, 혹시 4채널 이상일 때 대비)

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        # 초기화된 포인트 개수를 출력한다.
        # 예시: Number of points at initialisation :  2000

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        # dist2의 shape은 (N,) 1차원 텐서
        # 여기서 N은 포인트 클라우드의 포인트 개수(예: 2000)
        # 즉, dist2[i]는 i번째 포인트에서 가장 가까운 다른 포인트까지의 거리의 제곱 값
        # 예시: 만약 pcd.points가 (2000, 3)이라면, dist2는 (2000,)
        # 각 원소는 float 값이고, clamp_min으로 0.0000001보다 작지 않게 보정
        # distCUDA2 함수는 입력 포인트들에 대해 각 포인트별 최근접 이웃까지의 거리^2를 반환하는 함수

        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        # log를 사용하는 이유:
        # 가우시안의 scale(크기) 파라미터는 보통 log-공간에서 최적화하는 것이 수치적으로 더 안정적이기 때문
        # 예를 들어, scale이 0에 가까워지면 직접적으로 최적화할 때 gradient가 불안정해질 수 있는데,
        # log-공간에서는 scale이 항상 양수로 유지되고, 작은 값도 잘 다룰 수 있다.
        # 즉, 실제로는 exp(log_scale)로 복원해서 사용하게 되므로, log로 저장하면 학습이 더 잘 됨.

        # (N, 3) 의미:
        # 3은 x, y, z 축 각각에 대한 가우시안의 scale 값을 의미
        # 즉, 각 포인트마다 x, y, z 방향으로 독립적인 크기를 가질 수 있도록 (N, 3) 형태로 만든다.
        # 여기서는 sqrt(dist2)로부터 나온 값을 x, y, z에 동일하게 복사해서 초기화하지만,
        # 이후 학습 과정에서 각 축별로 값이 달라질 수 있다.
        # 예시: scales[0] = [a, a, a] (초기값), 학습 후 scales[0] = [a1, a2, a3] (각 축별로 다를 수 있음)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        # rots는 (N, 4) 크기의 0으로 채워진 텐서이다.
        # 각 포인트의 쿼터니언 회전(4차원) 파라미터를 저장한다.

        rots[:, 0] = 1
        # 쿼터니언의 첫 번째 성분을 1로 설정하여 단위 쿼터니언(회전 없음)으로 초기화한다.
        # 즉, 모든 포인트의 초기 회전은 없음.

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        # opacities는 (N, 1) 크기의 텐서로, 각 포인트의 불투명도(opacity) 파라미터이다.
        # 0.1로 초기화한 뒤, inverse_sigmoid로 변환하여 네트워크 파라미터 공간에 맞춘다.
        # 예시: 0.1 → inverse_sigmoid(0.1) ≈ -2.197

        identity = torch.zeros((fused_point_cloud.shape[0], 1), device="cuda")
        # identity는 (N, 1) 크기의 텐서로, 각 포인트의 identity(개인화 등) 파라미터를 0으로 초기화한다.
        # torch.optim.Adam 옵티마이저에 학습 가능한 파라미터로 등록되며, 1e-2의 학습률로 업데이트된다.
        # 이를 통해 모델 학습 과정에서 각 가우시안의 개인화 특징이 최적화된다.


        # <위에 선언한 변수들을 이용하여 __init__에서 생성한 인스턴스 변수를 업데이트 하고, 이를 학습 가능하게 만든다.>
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        # 포인트의 3D 위치(xyz)를 nn.Parameter로 등록한다.
        # requires_grad_(True)로 학습 가능하게 한다.

        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        # features의 DC 성분(0차 SH 계수, (N, 3, 1))을 (N, 1, 3)으로 transpose 후, nn.Parameter로 등록한다.
        # RGB DC 계수만 따로 관리한다.

        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        # features의 나머지 SH 계수(1차~)를 (N, 8, 3)으로 transpose 후, nn.Parameter로 등록한다.
        # SH 고차 계수만 따로 관리한다.

        self._identity = nn.Parameter(identity.requires_grad_(True))
        # identity 파라미터를 nn.Parameter로 등록한다.

        self._scaling = nn.Parameter(scales.requires_grad_(True))
        # scale 파라미터를 nn.Parameter로 등록한다.

        self._rotation = nn.Parameter(rots.requires_grad_(True))
        # rotation(쿼터니언) 파라미터를 nn.Parameter로 등록한다.

        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        # opacity 파라미터를 nn.Parameter로 등록한다.

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        # max_radii2D는 각 포인트가 2D 이미지로 투영될 때의 최대 반지름(픽셀 단위 등)을 저장하는 용도
        # 따라서 첫 번째 차원(포인트 개수)만 존재하는 1차원 텐서를 생성한다.

        # 해당 네트워크는 모델 학습에서 쓰이지 않는다.
        self.neural_renderer = GridRenderer(
                            bound=((self._xyz.max(0).values-self._xyz.min(0).values).max())/2 * 1.2,
                            coord_center=self._xyz.mean(0)
                        ).cuda()
        # 해당 네트워크는 모델 학습에서 쓰이지 않는다.
        # 가우시안의 좌표와 바라보는 방향을 입력 받아 각 가우시안의 밀도와 RGB 색상을 계산하는 GridRenderer를 초기화한다.
        # bound를 xyz의 최대-최소 범위의 최댓값/2 * 1.2로 씬의 크기로 설정하는 이유

        # 예를 들어, self._xyz가 (N, 3) 크기의 포인트 클라우드 좌표라면,
        # self._xyz.max(0).values는 각 축(x, y, z)별 최대값, self._xyz.min(0).values는 최소값
        # (self._xyz.max(0).values - self._xyz.min(0).values)는 각 축별 전체 범위(길이)를 의미
        # 예시: x축이 1~5, y축이 2~8, z축이 0~3이라면, 각각 4, 6, 3이다.
        # 그 중 가장 큰 축(여기선 y축 6)을 선택해서, 씬의 "가장 긴 변"을 기준으로 크기를 잡는다.
        # 그리고 씬의 중심을 기준으로 좌우로 반씩 나누기 위해서 /2를 한다.
        # 즉, 씬의 중심에서 양쪽으로 최대 길이의 절반만큼 확장된 구간을 커버하기 위함이다.
        # 마지막으로 *1.2를 곱하는 이유는, 씬의 경계에 딱 맞추면 경계에 있는 포인트가 잘릴 수 있으니
        # 여유를 두기 위해 20% 정도 더 크게 잡는 것이다.

        self.neural_motion_grid = PersonalizedMotionNetwork(args=self.args).cuda()
        # PersonalizedMotionNetwork(신경 모션 네트워크)를 초기화하고 GPU에 올린다.

    '''
    `training_setup(self, training_args)` 함수는 초기화 된 Gaussian 모델을 받아 이것의 학습을 위한 초기 설정을 담당한다. 이 함수는 다음과 같은 역할을 수행한다.

    *   밀집화 관련 변수 초기화: 포인트 밀집화(`densification`)에 필요한 `percent_dense`와 각 포인트의 그래디언트 누적을 위한 `xyz_gradient_accum`, `denom` 텐서들을 초기화한다.
    *   옵티마이저에 파라미터 추가: 3D Gaussian 포인트 클라우드의 위치(`_xyz`), 색상(`_features_dc`, `_features_rest`), 불투명도(`_opacity`), 스케일(`_scaling`), 회전(`_rotation`), 개인화 ID(`_identity`)와 같은 핵심 파라미터들을 옵티마이저에 등록한다.
    *   학습률 설정: 각 파라미터 그룹에 초기 학습률(`lr`)을 할당하며, 특히 위치 파라미터(`_xyz`)에 대해서는 `spatial_lr_scale`을 적용하여 공간적 학습률을 조절한다.
    *   보조 네트워크 파라미터 추가: `neural_renderer`와 `neural_motion_grid`와 같은 보조 신경망의 파라미터들도 함께 옵티마이저에 등록한다.
    *   옵티마이저 생성: `torch.optim.Adam` 옵티마이저를 생성하여 등록된 모든 파라미터 그룹을 관리한다.
    *   학습률 스케줄러 설정: `xyz` 파라미터의 학습률을 지수적으로 감소시키는 스케줄링 함수(`xyz_scheduler_args`)를 설정하여 학습 진행에 따라 학습률이 자동으로 조정되도록 준비한다.

    '''
    def training_setup(self, training_args):
        
        # 아래 세 변수는 가우시안의 포인트 밀집화 과정에서 사용되는 변수들이다.
        # 이것은 가우시안의 훈련과 관련 있기 때문에, create_from_pcd 함수가 아닌 training_setup 함수에서 초기화하는 것이다.
        self.percent_dense = training_args.percent_dense
        # training_args.percent_dense 값을 self.percent_dense에 저장한다.
        # percent_dense는 전체 포인트 중에서 dense하게 사용할 비율을 의미한다.

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # self.get_xyz.shape[0]는 포인트 개수 N이다.
        # (N, 1) 크기의 0으로 채워진 텐서를 생성해서 xyz_gradient_accum에 저장한다.
        # 이 텐서는 각 포인트의 xyz gradient를 누적하는 용도로 사용된다.
        # 그래디언트가 큰 포인트는 해당 영역에서 모델이 충분히 표현하지 못하고 있음을 의미할 수 있으며, 이러한 포인트는 밀집화(더 많은 가우시안 생성)의 대상이 된다.

        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # (N, 1) 크기의 0으로 채워진 텐서를 denom에 저장한다.
        # 각 가우시안 포인트의 그래디언트가 얼마나 여러 번 계산되었는지(또는 업데이트에 기여했는지) 횟수를 누적한다.
        # denom은 gradient normalization 등에서 분모로 쓰일 누적값을 저장한다.

        # 옵티마이저에 등록될 파라미터와 이에 대한 러닝 레이트를 딕셔너리 형태로 선언
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            # self._xyz(포인트 위치 파라미터)에 대해 learning rate를 position_lr_init * spatial_lr_scale로 설정한다.
            # "name"은 파라미터 그룹을 구분하기 위한 태그이다.

            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            # self._features_dc(색상 DC 성분)에 대해 feature_lr을 learning rate로 쓴다.

            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            # self._features_rest(색상 고차항)에 대해 feature_lr의 1/20 값을 learning rate로 쓴다.

            {'params': [self._identity], 'lr': 1e-2, "name": "identity"},
            # self._identity(개인화 identity 파라미터)에 대해 learning rate를 0.01로 쓴다.

            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            # self._opacity(불투명도 파라미터)에 대해 opacity_lr을 learning rate로 쓴다.

            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            # self._scaling(스케일 파라미터)에 대해 scaling_lr을 learning rate로 쓴다.

            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
            # self._rotation(회전 파라미터)에 대해 rotation_lr을 learning rate로 쓴다.
        ]

        # 보조 네트워크 파라미터 추가
        l += self.neural_renderer.get_params(lr=5e-3, lr_net=5e-4)
        # neural_renderer(신경 렌더러)의 파라미터와 learning rate를 받아서 l에 추가한다.
        # lr=0.005, lr_net=0.0005로 설정한다.

        l += self.neural_motion_grid.get_params(lr=1e-3, lr_net=1e-4)
        # neural_motion_grid(신경 모션 네트워크)의 파라미터와 learning rate를 받아서 l에 추가한다.
        # lr=0.001, lr_net=0.0001로 설정한다.

        # 옵티마이저 생성
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        # Adam 옵티마이저를 생성한다.
        # 파라미터 그룹 l을 등록하고, 전체 learning rate는 0.0(개별 그룹별 lr 사용), eps=1e-15로 설정한다.
        # 이 옵티마이저는 가우시안으로부터 렌더링 된 2D 이미지와 실제 이미지 간의 오차를 계산한 후, 해당 오차를 줄이는 방향으로 가우시안의 위치, 색상, 밀도 등을 최적화 한다.

        # 학습률 스케줄러 생성
        # 여기서 생성된 스케줄러는 위의 모든 인자들에 대한 러닝 레이트가 아니라, self._xyz에 대한 레닝 레이트인 training_args.position_lr_init * self.spatial_lr_scale만을 스케줄링 한다.
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init*self.spatial_lr_scale,
            lr_final=training_args.position_lr_final*self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps)

    '''
    `update_learning_rate(self, iteration)` 함수는 학습 과정 중 매 이터레이션(iteration)마다 `xyz` 파라미터의 학습률을 동적으로 갱신하는 역할을 한다.

    *   학습률 갱신: 이 함수는 `optimizer`에 등록된 여러 파라미터 그룹 중에서 `name`이 "xyz"인 그룹을 찾아, 위에서 생성한 `xyz_scheduler_args` 함수를 사용하여 현재 `iteration`에 맞는 새로운 학습률을 계산하고 해당 그룹에 적용한다.
    '''
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups: # optimizer에 등록된 파라미터 그룹들을 하나씩 순회한다.
            if param_group["name"] == "xyz": # 만약 파라미터 그룹의 이름이 "xyz"라면 (즉, 3D 위치 파라미터에 해당)
                lr = self.xyz_scheduler_args(iteration) # 현재 이터레이션에 맞는 learning rate를 스케줄러 함수로부터 받아온다.
                param_group['lr'] = lr # 해당 파라미터 그룹의 learning rate를 갱신한다.
                return lr # 새로 설정한 learning rate를 반환한다.

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_deformed_ply(self, xyz, scale, rotation, path):
        mkdir_p(os.path.dirname(path))

        xyz = xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = torch.log(self.scaling_activation(scale)).detach().cpu().numpy()
        rotation = rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'neural' in group["name"]: continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._identity = optimizable_tensors["identity"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if 'neural' in group["name"]: continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_identity, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "identity": new_identity,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._identity = optimizable_tensors["identity"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_identity = self._identity[selected_pts_mask].repeat(N,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_identity, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_identity = self._identity[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_identity, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1