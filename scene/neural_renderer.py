import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=False))
            # l == 0일 때는 입력 차원(self.dim_in)에서 첫 번째 은닉층(self.dim_hidden)으로 연결되는 Linear 레이어를 만든다.
            # l == num_layers - 1(마지막 레이어)일 때는 마지막 은닉층(self.dim_hidden)에서 출력 차원(self.dim_out)으로 연결되는 Linear 레이어를 만든다.
            # 그 외에는 은닉층끼리(self.dim_hidden → self.dim_hidden) 연결한다.
            # bias=False로 설정되어 있으므로, 각 Linear 레이어는 bias 항이 없다.

        self.net = nn.ModuleList(net)
        # net 리스트를 nn.ModuleList로 감싸서, PyTorch가 내부적으로 레이어를 추적하고 파라미터를 업데이트할 수 있도록 한다.

    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            # x에 대해 각 Linear 레이어를 순차적으로 적용한다.
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
                # 마지막 레이어가 아니면 ReLU 활성화 함수를 적용한다.
                # inplace=True로 설정해서 메모리 사용을 줄인다.
                # (예시) x가 [1.0, -2.0, 3.0]이면, ReLU 적용 후 [1.0, 0.0, 3.0]이 된다.
                
                # x = F.dropout(x, p=0.1, training=self.training)
                # (주석 처리됨) 드롭아웃을 적용하려면 이 줄을 활성화하면 된다.

        return x
        # 마지막 레이어까지 통과한 결과를 반환한다.
        # 예시: sigma_net = MLP(32, 64, 64, 3)로 생성하면,
        # 1번째 레이어: 입력 32 → 은닉 64
        # 2번째 레이어: 은닉 64 → 은닉 64
        # 3번째(마지막) 레이어: 은닉 64 → 출력 64
        # 총 3개의 Linear 레이어가 쌓인다.
        
# 가우시안의 좌표와 바라보는 방향을 입력 받아 각 가우시안의 밀도와 RGB 색상을 계산하는 네트워크.
class GridRenderer(nn.Module):
    def __init__(self,
                 bound = 1.,
                 coord_center=[0., 0., 0.],
                 keep_sigma=False
                 ):
        super().__init__()
        # bound: create_from_pcd에선 xyz의 최대-최소 범위의 최댓값/2 * 1.2로 씬의 크기를 설정한다.
        # coord_center: 씬의 중심 좌표를 설정한다.
        # keep_sigma: sigma 값을 고정할지 여부를 결정한다.
        # True면 sigma를 한 번만 계산해서 계속 사용한다.
        # sigma_results_static: keep_sigma가 True일 때 sigma 결과를 저장하는 변수다.
        # None이면 sigma를 매번 새로 계산한다.
        
        # bound 값을 float32 텐서로 변환하여 buffer로 등록한다.
        self.register_buffer('bound', torch.as_tensor(bound, dtype=torch.float32).detach())
        # coord_center(중심 좌표)를 float32 텐서로 변환하여 buffer로 등록한다.
        self.register_buffer('coord_center', torch.as_tensor(coord_center, dtype=torch.float32).detach())
        # register_buffer로 저장하면 좋은 점:
        # 1. 모델의 state_dict에 포함되어 저장/로드가 자동으로 된다.
        #    예시: model.state_dict()에 'bound','coord_center'가 포함되어 체크포인트 저장 시 같이 저장된다.
        # 2. requires_grad=False라서 학습 파라미터로 취급되지 않는다.
        #    즉, optimizer가 업데이트하지 않는다. (학습 대상 아님)
        # 3. .cuda(), .to(device) 등으로 모델을 이동할 때 buffer도 같이 이동된다.
        #    예시: model.cuda() 하면 coord_center도 GPU로 자동 이동된다.
        # 4. forward에서 상수처럼 쓰고 싶을 때, 일반 변수로 두면 device mismatch가 날 수 있는데,
        #    buffer로 두면 항상 모델과 같은 device에 있게 되어 안전하다.

        self.keep_sigma = keep_sigma
        # sigma_results_static은 keep_sigma가 True일 때 sigma 결과를 저장하는 변수다.
        
        self.sigma_results_static = None
        # 여기서 sigma_results_static은 keep_sigma=True일 때 한 번 계산한 sigma 값을 저장해두는 변수다.
        # 즉, 매번 새로 계산하지 않고, 이전에 계산한 sigma를 재사용할 때 쓴다.
        # 예시: self.sigma_results_static = torch.tensor([...])처럼 sigma 결과를 캐싱해둔다.

        self.num_levels = 16
        # 해시 그리드 인코더 레벨 수: 16
        self.level_dim = 2
        # 각 레벨의 차원수: 2
        self.base_resolution = 16
        # 해시 그리드의 base resolution (최초 해상도): 16
        self.table_size = 19
        # 해시 테이블 크기: 2^19개
        self.desired_resolution = 512
        # 원하느 최종 해상도: 512
        self.encoder_x, self.in_dim_x = self.create_encoder()
        # 해시 그리드 인코더와 인코더 출력 차원을 생성한다.
        # 3 -> 32

        ## sigma network
        self.num_layers = 3
        # sigma 네트워크 레이어 수: 3
        self.hidden_dim = 64
        # sigma 네트워크의 hidden 차원: 64
        self.geo_feat_dim = 64
        # geometry feature의 차원을 64로 설정한다.
        self.sigma_net = MLP(self.in_dim_x, 1 + self.geo_feat_dim, self.hidden_dim, self.num_layers)
        # sigma_net은 입력 차원 self.in_dim_x, 출력 차원 1+geo_feat_dim, hidden_dim, num_layers로 구성된 MLP다.
        # 차원: 32 → 64 → 64 + 1

        ## color network
        self.num_layers_color = 2
        # color 네트워크의 레이어 수를 2로 설정한다.
        self.hidden_dim_color = 64
        # color 네트워크의 hidden 차원을 64로 설정한다.
        self.encoder_dir, self.in_dim_dir = get_encoder('sphere_harmonics')
        # 방향 인코더와 그 출력 차원을 sphere_harmonics로부터 생성한다.
        self.color_net = MLP(self.in_dim_dir + self.geo_feat_dim, 3, self.hidden_dim_color, self.num_layers_color)
        # 차원: (16 + 64) -> 3
    
    def create_encoder(self):
        # 해시 그리드 인코더를 생성한다.
        # self.bound.cpu()는 bound 값을 CPU 텐서로 변환한다.
        # desired_resolution * bound로 실제 공간 해상도를 맞춘다.
        self.encoder_x, self.in_dim_x = get_encoder(
            'hashgrid', input_dim=3, num_levels=self.num_levels, level_dim=self.level_dim, 
            base_resolution=self.base_resolution, log2_hashmap_size=self.table_size, desired_resolution=self.desired_resolution * self.bound.cpu())
        # encoder_x와 in_dim_x를 반환한다.
        # in_dim_x는 해시 그리드 인코더의 출력 차원이다.
        # 예를 들어, num_levels=16, level_dim=2라면 in_dim_x=32가 된다.
        # 즉, in_dim_x = num_levels * level_dim 수식이 성립한다.
        # 이 값은 이후 MLP 등 네트워크의 입력 feature 차원으로 사용된다.
        return self.encoder_x, self.in_dim_x

    def recover_from_ckpt(self, state_dict):
        # state_dict에서 bound 값을 불러와 self.bound에 할당한다.
        self.bound = state_dict['bound']
        # 인코더를 다시 생성한다(파라미터 shape 맞추기 위함).
        self.encoder_x, self.in_dim_x = self.create_encoder()
        # 전체 state_dict를 불러온다(모델 파라미터 복원).
        self.load_state_dict(state_dict)

    def encode_x(self, x):
        # x: [N, 3], 입력 좌표(월드 좌표계)
        # self.coord_center를 빼서 중심 정렬 후, bound를 넘긴다.
        # 예시: x=[1,2,3], coord_center=[0.5,0.5,0.5]면 x-coord_center=[0.5,1.5,2.5]
        return self.encoder_x(x - self.coord_center, bound=self.bound)


    def forward(self, x, d):
        # x: [N, 3], 입력 좌표(월드 좌표계, [-bound, bound] 범위)
        # d: [N, 3], 방향 벡터(정규화, [-1, 1] 범위)
        enc_x = self.encode_x(x)
        # density(=sigma)와 geometry feature를 계산한다.
        sigma_result = self.density(x, enc_x)
        # sigma(밀도)만 추출한다.
        sigma = sigma_result['sigma']
        # color(색상)를 계산한다.
        color = self.color(sigma_result, d)
        # sigma와 color를 반환한다.
        return sigma, color
    

    def color(self, sigma_result, d):
        # sigma_result에서 geometry feature를 추출한다.
        geo_feat = sigma_result['geo_feat']
        # d(방향 벡터)를 방향 인코더로 인코딩한다.
        enc_d = self.encoder_dir(d)
        # 방향 인코딩과 geometry feature를 concat한다.
        h = torch.cat([enc_d, geo_feat], dim=-1)

        # color_net에 입력하여 색상 예측값을 얻는다.
        h_color = self.color_net(h)
        # sigmoid로 0~1 범위로 변환, 약간의 범위 확장(1+2*0.001) 후 -0.001로 shift
        # 예시: color = torch.sigmoid(h_color)*(1.002) - 0.001
        color = torch.sigmoid(h_color)*(1 + 2*0.001) - 0.001
        return color


    def density(self, x, enc_x=None):
        # x: [N, 3], 입력 좌표(월드 좌표계)
        # keep_sigma가 True이고, sigma_results_static이 이미 있으면 그 값을 반환한다.
        if self.keep_sigma and self.sigma_results_static is not None:
            return self.sigma_results_static
        
        # enc_x가 None이면 encode_x로 인코딩한다.
        if enc_x is None:
            enc_x = self.encode_x(x)

        # sigma_net에 enc_x를 넣어 h를 얻는다.
        h = self.sigma_net(enc_x)
        # h의 첫 번째 채널이 sigma(밀도)다.
        sigma = h[..., 0]
        # sigma = torch.exp(h[..., 0])  # (주석) exp로 변환할 수도 있다.
        # sigma = torch.sigmoid(h[..., 0])  # (주석) sigmoid로 변환할 수도 있다.
        # 나머지 채널은 geometry feature다.
        geo_feat = h[..., 1:]

        # keep_sigma가 True면 결과를 static 변수에 저장한다.
        if self.keep_sigma:
            self.sigma_results_static = {
                    'sigma': sigma,
                    'geo_feat': geo_feat,
                }

        # sigma와 geo_feat을 dict로 반환한다.
        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }


    # optimizer utils
    def get_params(self, lr, lr_net, wd=0):

        # 인코더, sigma_net, color_net의 파라미터를 optimizer에 넘길 수 있도록 dict로 묶는다.
        params = [
            {'params': self.encoder_x.parameters(), 'name': 'neural_encoder', 'lr': lr},
            {'params': self.sigma_net.parameters(), 'name': 'neural_sigma', 'lr': lr_net, 'weight_decay': wd},
            {'params': self.color_net.parameters(), 'name': 'neural_color', 'lr': lr_net, 'weight_decay': wd}, 
        ]
        
        # 파라미터 리스트를 반환한다.
        return params