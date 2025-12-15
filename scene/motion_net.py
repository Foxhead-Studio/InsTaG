import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder


class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, leakyReLU=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout)
        )
        if leakyReLU:
            self.act = nn.LeakyReLU(0.02)
        else:
            self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)
    
    
# Audio feature extractor
class AudioAttNet(nn.Module):
    def __init__(self, dim_aud=64, seq_len=8):
        super(AudioAttNet, self).__init__()
        self.seq_len = seq_len
        self.dim_aud = dim_aud
        self.attentionConvNet = nn.Sequential(  # b x subspace_dim x seq_len
            nn.Conv1d(self.dim_aud, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(4, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(2, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True)
        )
        self.attentionNet = nn.Sequential(
            nn.Linear(in_features=self.seq_len, out_features=self.seq_len, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x: [1, seq_len, dim_aud]
        y = x.permute(0, 2, 1)  # [1, dim_aud, seq_len]
        #  x의 차원 순서를 [배치, 시퀀스 길이, 오디오 차원]에서 [배치, 오디오 차원, 시퀀스 길이]로 변경
        # attentionConvNet이 1D 컨볼루션을 시간 축(seq_len)을 따라 적용할 수 있도록, 오디오 특징 채널을 입력 채널로, 시퀀스 길이를 시퀀스 길이 차원으로 만듭니다.

        y = self.attentionConvNet(y)
        # 입력: [1, 32, 8]
        # 출력: [1, 1, 8]
        # 오디오 시퀀스 내의 시간적 패턴과 각 오디오 특징 채널 간의 관계를 학습하여, 어떤 시점의 오디오가 중요한지에 대한 저수준 특징 맵을 생성합니다. 최종 출력은 각 시퀀스 위치에 대한 단일 스코어(채널 1)를 가집니다.
        y = self.attentionNet(y.view(1, self.seq_len)).view(1, self.seq_len, 1)
        # 출력: [1, 8, 1]
        return torch.sum(y * x, dim=1) # [1, dim_aud]


# Audio feature extractor
class AudioNet(nn.Module):
    def __init__(self, dim_in=29, dim_aud=64, win_size=16):
        super(AudioNet, self).__init__()
        self.win_size = win_size
        self.dim_aud = dim_aud
        # 여러 개의 1D 컨볼루션 (nn.Conv1d) 레이어와 LeakyReLU 활성화 함수로 구성
        # 입력: [N, 29, 16] 형태의 텐서입니다. 여기서 N은 배치 크기(예: 1 또는 8), 29는 오디오 특징의 채널 수(예: deepspeech 특징), 16은 시간 윈도우 크기(win_size)입니다.
        # 출력: [N, 64, 1]로, 16의 윈도우를 1로 줄이되, 오디오 특징은 64로 늘려 오디오의 특징을 잘 표현합니다.
        # 목적: 긴 오디오 특징 시퀀스를 점진적으로 압축하여 핵심적인 시간적 패턴을 추출합니다.
        self.encoder_conv = nn.Sequential(  # n x 29 x 16
            nn.Conv1d(dim_in, 32 if dim_in < 128 else 128, kernel_size=3, stride=2, padding=1, bias=True),  # n x 32 x 8
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32 if dim_in < 128 else 128, 32 if dim_in < 128 else 128, kernel_size=3, stride=2, padding=1, bias=True),  # n x 32 x 4
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32 if dim_in < 128 else 128, 64, kernel_size=3, stride=2, padding=1, bias=True),  # n x 64 x 2
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),  # n x 64 x 1
            nn.LeakyReLU(0.02, True),
        )
        # 입력: self.encoder_conv의 최종 출력은 [N, 64, 1] 형태이므로, 이를 [N, 64]로 압축하여 선형 레이어의 입력으로 사용합니다.
        # 출력: [N, 32]로 정보를 압축합니다.
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.02, True),
            nn.Linear(64, dim_aud),
        )

    def forward(self, x):
        half_w = int(self.win_size/2)
        x = x[:, :, 8-half_w:8+half_w]
        x = self.encoder_conv(x).squeeze(-1)
        x = self.encoder_fc1(x)
        return x


class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0), )

    def forward(self, x):
        out = self.audio_encoder(x)
        out = out.squeeze(2).squeeze(2)

        return out

# Audio feature extractor
class AudioNet_ave(nn.Module):
    def __init__(self, dim_in=29, dim_aud=64, win_size=16):
        super(AudioNet_ave, self).__init__()
        self.win_size = win_size
        self.dim_aud = dim_aud
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.02, True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.02, True),
            nn.Linear(128, dim_aud),
        )
    def forward(self, x):
        # half_w = int(self.win_size/2)
        # x = x[:, :, 8-half_w:8+half_w]
        # x = self.encoder_conv(x).squeeze(-1)
        x = self.encoder_fc1(x).permute(1,0,2).squeeze(0)
        return x


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

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
                # x = F.dropout(x, p=0.1, training=self.training)
                
        return x

# Face 브렌치의 UMF
class MotionNetwork(nn.Module):
    def __init__(self,
                 audio_dim = 32,
                 ind_dim = 0,
                 args = None,
                 ):
        super(MotionNetwork, self).__init__()

        if 'esperanto' in args.audio_extractor:
            self.audio_in_dim = 44
        elif 'deepspeech' in args.audio_extractor: # 디폴트
            self.audio_in_dim = 29
        elif 'hubert' in args.audio_extractor:
            self.audio_in_dim = 1024
        elif 'ave' in args.audio_extractor:
            self.audio_in_dim = 32
        else:
            raise NotImplementedError
    
        self.bound = 0.15
        self.exp_eye = True # 개인 모션 필드처럼 여기도 True -> 표정 피처를 활용

        
        self.individual_dim = ind_dim # 특정 개인의 고유한 외모나 움직장 특성을 인코딩하는 학습 가능한 잠재 코드의 차원
        if self.individual_dim > 0:
            self.individual_codes = nn.Parameter(torch.randn(10000, self.individual_dim) * 0.1) 

        # audio network
        self.audio_dim = audio_dim
        if args.audio_extractor == 'ave':
            self.audio_net = AudioNet_ave(self.audio_in_dim, self.audio_dim)
        else:
            self.audio_net = AudioNet(self.audio_in_dim, self.audio_dim)
        self.audio_att_net = AudioAttNet(self.audio_dim)

        # DYNAMIC PART
        self.num_levels = 12
        self.level_dim = 1
        self.encoder_xy, self.in_dim_xy = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)
        self.encoder_yz, self.in_dim_yz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)
        self.encoder_xz, self.in_dim_xz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)
        
        self.in_dim = self.in_dim_xy + self.in_dim_yz + self.in_dim_xz


        self.num_layers = 3       
        self.hidden_dim = 64 # Personalized Motion Network의 은닉층 차원은 32였는데, 여기선 64로 두 배

        # 표정 피처 생성용 네트워크 두 개
        self.exp_in_dim = 6 - 1
        self.eye_dim = 6 if self.exp_eye else 0
        self.exp_encode_net = MLP(self.exp_in_dim, self.eye_dim - 1, 16, 2)

        self.eye_att_net = MLP(self.in_dim, self.eye_dim, 16, 2)

        # 프레임 별 가우시안 변화량 피처
        # rot: 4   xyz: 3   opac: 1  scale: 3
        self.out_dim = 11 # 여기는 face, mouth decomposition을 사용하지 않기 때문에 분기가 존재하지 않고 모두 11차원 추출
        self.sigma_net = MLP(self.in_dim + self.audio_dim + self.eye_dim + self.individual_dim, self.out_dim, self.hidden_dim, self.num_layers)
        
        # 오디오 가중치 생성 네트워크
        self.aud_ch_att_net = MLP(self.in_dim, self.audio_dim, 32, 2)
        
        self.cache = None


    @staticmethod
    @torch.jit.script
    def split_xyz(x):
        xy, yz, xz = x[:, :-1], x[:, 1:], torch.cat([x[:,:1], x[:,-1:]], dim=-1)
        return xy, yz, xz


    # 3D 좌표를 XY, YZ, XZ 평면으로 나누어 각 평면별로 해시 그리드 인코더를 통과하여 피처를 추출한 뒤 concat하여 return하는 함수
    def encode_x(self, xyz, bound):
        # x: [N, 3], in [-bound, bound]
        N, M = xyz.shape
        xy, yz, xz = self.split_xyz(xyz)
        feat_xy = self.encoder_xy(xy, bound=bound)
        feat_yz = self.encoder_yz(yz, bound=bound)
        feat_xz = self.encoder_xz(xz, bound=bound)
        
        return torch.cat([feat_xy, feat_yz, feat_xz], dim=-1)
    

    # 오디오 피처를 audio_net, audio_att_net에 순차적으로 통과하여 피처를 추출하는 함수
    def encode_audio(self, a):
        # a: [1, 29, 16] or [8, 29, 16], audio features from deepspeech
        # if emb, a should be: [1, 16] or [8, 16]

        # fix audio traininig
        if a is None: return None

        enc_a = self.audio_net(a) # [1/8, 64]
        enc_a = self.audio_att_net(enc_a.unsqueeze(0)) # [1, 64]
            
        return enc_a

    def forward(self, x, a, e=None, c=None):
        # 1. 입력 3D 좌표(x)를 XY, YZ, XZ 평면별 해시 인코더에 통과시켜 피처 추출
        enc_x = self.encode_x(x, bound=self.bound)

        # 2. 오디오 피처(a)를 오디오 네트워크에 통과시켜 임베딩 추출
        enc_a = self.encode_audio(a)
        # 3. 포인트 개수만큼 오디오 임베딩을 복제
        enc_a = enc_a.repeat(enc_x.shape[0], 1)
        # 4. 오디오 채널 어텐션 피처 추출
        aud_ch_att = self.aud_ch_att_net(enc_x)
        # 5. 오디오 임베딩과 어텐션을 곱해 최종 오디오 피처 생성
        enc_w = enc_a * aud_ch_att
        
        # 6. 눈 어텐션 피처 추출 (ReLU 활성화)
        eye_att = torch.relu(self.eye_att_net(enc_x))
        # 7. 표정 피처(e)에서 마지막 차원 제외하고 인코딩
        enc_e = self.exp_encode_net(e[:-1])
        # 8. 마지막 차원(e[-1:])을 붙여줌
        enc_e = torch.cat([enc_e, e[-1:]], dim=-1)
        # 9. 표정 피처에 눈 어텐션 곱함
        enc_e = enc_e * eye_att

        # 10. 개인 정보 피처(c)가 있으면 포인트 개수만큼 복제 후 concat
        if c is not None:
            c = c.repeat(enc_x.shape[0], 1)
            h = torch.cat([enc_x, enc_w, enc_e, c], dim=-1)
        else:
            h = torch.cat([enc_x, enc_w, enc_e], dim=-1)

        # 11. 최종 피처(h)를 sigma_net에 통과시켜 가우시안 변화량 예측
        h = self.sigma_net(h)

        # 12. 예측 결과를 각 파라미터별로 분리
        d_xyz = h[..., :3] * 1e-2      # 위치 변화량
        d_rot = h[..., 3:7]            # 회전 변화량
        d_opa = h[..., 7:8]            # 불투명도 변화량
        d_scale = h[..., 8:11]         # 스케일 변화량
        
        # 13. 결과 딕셔너리로 반환 (어텐션 norm 포함)
        results = {
            'd_xyz': d_xyz,
            'd_rot': d_rot,
            'd_opa': d_opa,
            'd_scale': d_scale,
            'ambient_aud' : aud_ch_att.norm(dim=-1, keepdim=True),
            'ambient_eye' : eye_att.norm(dim=-1, keepdim=True),
        }
        self.cache = results
        return results


    # optimizer utils
    def get_params(self, lr, lr_net, wd=0):

        params = [
            {'params': self.audio_net.parameters(), 'lr': lr_net, 'weight_decay': wd}, 
            {'params': self.encoder_xy.parameters(), 'lr': lr},
            {'params': self.encoder_yz.parameters(), 'lr': lr},
            {'params': self.encoder_xz.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
        ]
        params.append({'params': self.audio_att_net.parameters(), 'lr': lr_net * 5, 'weight_decay': 0.0001})
        if self.individual_dim > 0:
            params.append({'params': self.individual_codes, 'lr': lr_net, 'weight_decay': wd})
        
        params.append({'params': self.aud_ch_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd})
        params.append({'params': self.eye_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd})
        params.append({'params': self.exp_encode_net.parameters(), 'lr': lr_net, 'weight_decay': wd})

        return params



# Mouth 브렌치의 UMF
class MouthMotionNetwork(nn.Module):
    def __init__(self,
                 audio_dim = 32,
                 ind_dim = 0,
                 args = None,
                 ):
        super(MouthMotionNetwork, self).__init__()

        if 'esperanto' in args.audio_extractor:
            self.audio_in_dim = 44
        elif 'deepspeech' in args.audio_extractor:
            self.audio_in_dim = 29
        elif 'hubert' in args.audio_extractor:
            self.audio_in_dim = 1024
        elif 'ave' in args.audio_extractor:
            self.audio_in_dim = 32
        else:
            raise NotImplementedError
        
        
        self.bound = 0.15
        # 입술 모션에는 표정 피처가 없기 때문에 self.exp_eye 변수는 존재하지 않는다.

        
        self.individual_dim = ind_dim
        if self.individual_dim > 0:
            self.individual_codes = nn.Parameter(torch.randn(10000, self.individual_dim) * 0.1) 

        # audio network
        self.audio_dim = audio_dim
        if args.audio_extractor == 'ave':
            self.audio_net = AudioNet_ave(self.audio_in_dim, self.audio_dim)
        else:
            self.audio_net = AudioNet(self.audio_in_dim, self.audio_dim)
        self.audio_att_net = AudioAttNet(self.audio_dim)

        # DYNAMIC PART
        self.num_levels = 12
        self.level_dim = 1
        self.encoder_xy, self.in_dim_xy = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=64, log2_hashmap_size=17, desired_resolution=384 * self.bound)
        self.encoder_yz, self.in_dim_yz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=64, log2_hashmap_size=17, desired_resolution=384 * self.bound)
        self.encoder_xz, self.in_dim_xz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=64, log2_hashmap_size=17, desired_resolution=384 * self.bound)

        self.in_dim = self.in_dim_xy + self.in_dim_yz + self.in_dim_xz

        ## sigma network
        self.num_layers = 3
        self.hidden_dim = 32 # Personalized Motion Network의 은닉층 차원과 동일한 32

        # 표정 어텐션 모듈 모두 부재. exp_in_dim, eye_dim, exp_encode_net, eye_att_net

        self.out_dim = 7 # 입술 모션은 가우시안의 좌표와 회전값만 추출하므로 7차원 (MotionNetwork는 11)
        self.move_dim = 3 # rot: 4   xyz: 3
        self.sigma_net = MLP(self.in_dim + self.audio_dim + self.individual_dim + self.move_dim, self.out_dim, self.hidden_dim, self.num_layers)
        self.scaler_net = MLP(self.in_dim + self.move_dim, 1, 16, 3)

        self.aud_ch_att_net = MLP(self.in_dim, self.audio_dim, 32, 2)
    

    def encode_audio(self, a):
        # a: [1, 29, 16] or [8, 29, 16], audio features from deepspeech
        # if emb, a should be: [1, 16] or [8, 16]

        # fix audio traininig
        if a is None: return None

        enc_a = self.audio_net(a) # [1/8, 64]
        enc_a = self.audio_att_net(enc_a.unsqueeze(0)) # [1, 64]
            
        return enc_a
    

    @staticmethod
    @torch.jit.script
    def split_xyz(x):
        xy, yz, xz = x[:, :-1], x[:, 1:], torch.cat([x[:,:1], x[:,-1:]], dim=-1)
        return xy, yz, xz


    def encode_x(self, xyz, bound):
        # x: [N, 3], in [-bound, bound]
        N, M = xyz.shape
        xy, yz, xz = self.split_xyz(xyz)
        feat_xy = self.encoder_xy(xy, bound=bound)
        feat_yz = self.encoder_yz(yz, bound=bound)
        feat_xz = self.encoder_xz(xz, bound=bound)
        
        return torch.cat([feat_xy, feat_yz, feat_xz], dim=-1)

    # 논문에 소개된 face_moth_hook의 수식과 실제 구현이 어떻게 다른지 파악해본다.
    def forward(self, x, a, move):
        # x: [N, 3], in [-bound, bound]
        # move: [1, D]
        enc_x = self.encode_x(x, bound=self.bound) # 입력 3D 좌표(x)를 XY, YZ, XZ 평면별 해시 인코더에 통과시켜 피처 추출

        enc_a = self.encode_audio(a) # 오디오 피처(a)를 오디오 네트워크에 통과시켜 임베딩 추출
        enc_w = enc_a.repeat(enc_x.shape[0], 1) # 포인트 개수만큼 오디오 임베딩을 복제 (MotionNetwork와 달리 따로 aud_ch_att를 곱해서 enc_w로 만들지 않음)

        # move = torch.as_tensor([[move_max, move_min]]).repeat(enc_x.shape[0], 1).cuda()
        move = move.repeat(enc_x.shape[0], 1)
        # 원래는 φ를 만들 때 위의 수식처럼 얼굴 움직임에서 최대 값, 최소 값이 입 모양일 것이라 가정하지만, 여기서는 단순히 가우시안 움직임을 복제해서 사용.
        # 왜냐하면 render_motion_mouth_con에서 이미 해당 과정을 다 계산한 뒤에 move에 제공하기 때문이다.

        h = torch.cat([enc_x, enc_w, move], dim=-1)
        h = self.sigma_net(h)
        
        h_s = torch.cat([enc_x, move], dim=-1)
        h_s = self.scaler_net(h_s)
        # scaler_net은 논문에 명시된 실수배 τ를 계산하는 네트워크로, 입술 영역 가우시안의 XYZ 변화량($d_{xyz}$)에 스케일 팩터를 적용하는 역할을 한다.
        # enc_x (공간 피처)와 move (움직임 피처)를 입력으로 받아 h_s라는 스칼라 값을 출력한다. 이 h_s 값은 torch.sigmoid(h_s) * 2를 통해 $0$에서 $2$ 사이의 스케일 팩터로 변환되어 최종 $d_{xyz}$에 곱해진다.
        # 목적: 입술 영역의 움직임은 매우 미세하고 정교해야 하므로, scaler_net은 각 가우시안의 공간적 움직임($d_{xyz}$)의 크기를 동적으로 조절하는 역할을 한다.
        # personalized motion network의 align_net은 캐노니컬 가우시안과 UMF 간의 스케일 불일치를 해결하고 개인화된 움직임 생성을 위한 오프셋($p_{xyz}$) 및 스케일 팩터($p_{scale}$) 계산.
        # 이는 6 (3D 오프셋 + 3D 스케일 팩터)을 출력하는데 반해, scaler_net은 1차원 스케일 팩터만 출력.
        
        d_xyz = h[..., :3] * 1e-2 # 위치 변화량
        d_xyz[..., 0] = d_xyz[..., 0] / 5 # x축 위치 변화량을 5로 나누어 줄임.
        d_xyz[..., 2] = d_xyz[..., 2] / 5 # z축 위치 변화량을 5로 나누어 줄임.
        # 입술 움직임은 위아래로만 일어나므로 위와 같이 실수배로 나누는 과정을 거친다.
        d_rot = h[..., 3:] # 회전 변화량
        
        return {
            'd_xyz': d_xyz * torch.sigmoid(h_s) * 2, # 입술의 위치 변화량에 스케일 팩터 τ를 곱해준다.
            'd_rot': d_rot,
            # 'ambient_aud' : aud_ch_att.norm(dim=-1, keepdim=True),
        }


    # optimizer utils
    def get_params(self, lr, lr_net, wd=0):

        params = [
            {'params': self.audio_net.parameters(), 'lr': lr_net, 'weight_decay': wd}, 
            {'params': self.encoder_xy.parameters(), 'lr': lr},
            {'params': self.encoder_yz.parameters(), 'lr': lr},
            {'params': self.encoder_xz.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.scaler_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
        ]
        params.append({'params': self.audio_att_net.parameters(), 'lr': lr_net * 5, 'weight_decay': 0.0001})
        if self.individual_dim > 0:
            params.append({'params': self.individual_codes, 'lr': lr_net, 'weight_decay': wd})
        
        params.append({'params': self.aud_ch_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd})

        return params



# class PersonalizedMotionNetwork(nn.Module):
#     def __init__(self):
#         super(PersonalizedMotionNetwork, self).__init__()
#         self.bound = 0.15
#         self.exp_eye = True

#         # DYNAMIC PART
#         self.num_levels = 12
#         self.level_dim = 1
#         self.encoder_xy, self.in_dim_xy = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)
#         self.encoder_yz, self.in_dim_yz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)
#         self.encoder_xz, self.in_dim_xz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)

#         self.in_dim = self.in_dim_xy + self.in_dim_yz + self.in_dim_xz
        
#         self.num_layers = 3
#         self.hidden_dim = 32

#         self.out_dim = 4
#         self.sigma_net = MLP(self.in_dim, self.out_dim, self.hidden_dim, self.num_layers)

#     @staticmethod
#     @torch.jit.script
#     def split_xyz(x):
#         xy, yz, xz = x[:, :-1], x[:, 1:], torch.cat([x[:,:1], x[:,-1:]], dim=-1)
#         return xy, yz, xz

#     def encode_x(self, xyz, bound):
#         # x: [N, 3], in [-bound, bound]
#         N, M = xyz.shape
#         xy, yz, xz = self.split_xyz(xyz)
#         feat_xy = self.encoder_xy(xy, bound=bound)
#         feat_yz = self.encoder_yz(yz, bound=bound)
#         feat_xz = self.encoder_xz(xz, bound=bound)
        
#         return torch.cat([feat_xy, feat_yz, feat_xz], dim=-1)


#     def forward(self, x):
#         # x: [N, 3], in [-bound, bound]
#         enc_x = self.encode_x(x, bound=self.bound)
#         h = self.sigma_net(enc_x)

#         d_xyz = h[..., :3] * 1e-2
#         d_scale = h[..., 3:]
#         return {
#             'd_xyz': d_xyz,
#             'd_scale': d_scale,
#         }
        

#     # optimizer utils
#     def get_params(self, lr, lr_net, wd=0):

#         params = [
#             {'params': self.encoder_xy.parameters(), 'name': 'neural_encoder_xy', 'lr': lr},
#             {'params': self.encoder_yz.parameters(), 'name': 'neural_encoder_yz', 'lr': lr},
#             {'params': self.encoder_xz.parameters(), 'name': 'neural_encoder_xz', 'lr': lr},
#             {'params': self.sigma_net.parameters(), 'name': 'neural_encoder_net', 'lr': lr_net, 'weight_decay': wd},
#         ]

#         return params


# Face 브랜치의 PMF
# Mouth 브랜치의 PMF
class PersonalizedMotionNetwork(nn.Module):
    def __init__(self,
                 audio_dim = 32,
                 ind_dim = 0,
                 args = None,
                 ):
        super(PersonalizedMotionNetwork, self).__init__()

        if 'esperanto' in args.audio_extractor:
            self.audio_in_dim = 44
        elif 'deepspeech' in args.audio_extractor: # 디폴트 오디오 추출기
            self.audio_in_dim = 29
        elif 'hubert' in args.audio_extractor:
            self.audio_in_dim = 1024
        elif 'ave' in args.audio_extractor:
            self.audio_in_dim = 32
        else:
            raise NotImplementedError

        self.args = args
        self.bound = 0.15
        self.exp_eye = args.type == 'face' # 'face'일 경우 True로, AU와 랜드마크 정보등을 추가로 활용

        self.individual_dim = ind_dim # 특정 개인의 고유한 외모나 움직임 특성을 인코딩하는 학습 가능한 잠재 코드의 차원
        if self.individual_dim > 0:
            self.individual_codes = nn.Parameter(torch.randn(10000, self.individual_dim) * 0.1) 

        # audio network
        self.audio_dim = audio_dim
        if args.audio_extractor == 'ave':
            self.audio_net = AudioNet_ave(self.audio_in_dim, self.audio_dim)
        else: # 디폴트 오디오 추출기
            self.audio_net = AudioNet(self.audio_in_dim, self.audio_dim) # 전체 윈도우의 오디오를 현재 프레임에 압축하여 반영
        self.audio_att_net = AudioAttNet(self.audio_dim) # 전체 윈도우의 시퀀스별 중요도를 가중치화 하여 현재 프레임에 반영

        # DYNAMIC PART
        self.num_levels = 12 # 해시 그리드 인코더의 레벨 수
        self.level_dim = 1 # 해시 그리드 인코더의 레벨 차원
        self.encoder_xy, self.in_dim_xy = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)
        self.encoder_yz, self.in_dim_yz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)
        self.encoder_xz, self.in_dim_xz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=16, log2_hashmap_size=17, desired_resolution=256 * self.bound)
        # 3개의 해시 그리드 인코더 생성
        self.in_dim = self.in_dim_xy + self.in_dim_yz + self.in_dim_xz # 각 triplane 해시 인코더를 통과한 피처들을 연결할 차원


        self.num_layers = 3
        self.hidden_dim = 32 if args.type == 'face' else 16
    
        self.exp_in_dim = 6 - 1 # exp encode net에 입력되는 차원 (e에서 마지막 차원은 제외)
        self.eye_dim = 6 if self.exp_eye else 0 # exp encode net을 통과한 피처와, e의 마지막 차원을 합친 차원 6
        if self.exp_eye: # 'face'일 경우 True로, exp encode net과 eye att net 모듈을 생성
            self.exp_encode_net = MLP(self.exp_in_dim, self.eye_dim - 1, 16, 2) # AU와 랜드마크 정보등을 모델이 이해할 수 있는 고차원 피처로 임베딩
            self.eye_att_net = MLP(self.in_dim, self.eye_dim, 16, 2) # N개의 가우시안 중 표정 피처가 미치는 영향에 대한 가중치 계산 (눈썹 등의 움직임은 영향을 강하게)

        # rot: 4   xyz: 3   opac: 1  scale: 3
        self.out_dim = 11 if args.type == 'face' else 7 # 'face'일 경우 11개의 차원(rot: 4, xyz: 3, opac: 1, scale: 3), 그 외에는 7개의 차원(rot: 4, xyz: 3)
        
        # # Valence/Arousal 피처 추가
        # # self.use_va = getattr(args, 'use_va', False) # args.use_va가 존재하면 그 값을 사용하고, 없으면 False
        # self.use_va = False
        # self.va_dim = 0 # Valence/Arousal 피처의 차원 초기화
        # if self.use_va:
        #     self.va_dim = 16 # Valence/Arousal 피처를 임베딩할 차원
        #     self.va_encode_net = MLP(2, self.va_dim, 16, 2) # 2 (Valence, Arousal) -> va_dim으로 임베딩하는 네트워크
        
        # self.sigma_net = MLP(self.in_dim + self.audio_dim + self.eye_dim + self.individual_dim + self.va_dim, self.out_dim, self.hidden_dim, self.num_layers) # C, enc_e, enc_x, enc_w, enc_va를 받아 프레임별 가우시안 움직임 계산
        
        self.sigma_net = MLP(self.in_dim + self.audio_dim + self.eye_dim + self.individual_dim, self.out_dim, self.hidden_dim, self.num_layers) # C, enc_e, enc_x, enc_w를 받아 프레임별 가우시안 움직임 계산
        
        self.align_net = MLP(self.in_dim, 6, self.hidden_dim, 2) # 인물이 바뀔 때마다 생성되는 캐노니컬 가우시안과, 보편적인 움직임을 계산하는 UMF간의 스케일 언매칭을 해결하기 위해 오프셋과 스케일 팩터를 계산

        self.aud_ch_att_net = MLP(self.in_dim, self.audio_dim, 32, 2) #  N개의 가우시안 중 오디오 피처가 미치는 영향에 대한 가중치 계산 (입술의 움직임은 영향을 강하게)


    @staticmethod
    @torch.jit.script
    def split_xyz(x):
        xy, yz, xz = x[:, :-1], x[:, 1:], torch.cat([x[:,:1], x[:,-1:]], dim=-1)
        return xy, yz, xz


    def encode_x(self, xyz, bound):
        # x: [N, 3], in [-bound, bound]
        # x, y, z 좌표를 XY, YZ, XZ 평면으로 나누어 각 평면별로 해시 그리드 인코더를 통과하여 피처를 추출한 뒤 concat하여 return하는 함수
        N, M = xyz.shape
        xy, yz, xz = self.split_xyz(xyz)
        feat_xy = self.encoder_xy(xy, bound=bound)
        feat_yz = self.encoder_yz(yz, bound=bound)
        feat_xz = self.encoder_xz(xz, bound=bound)
        
        return torch.cat([feat_xy, feat_yz, feat_xz], dim=-1)
    

    def encode_audio(self, a):
        # a: [1, 29, 16] or [8, 29, 16], audio features from deepspeech
        # if emb, a should be: [1, 16] or [8, 16]

        # fix audio traininig
        if a is None: return None

        enc_a = self.audio_net(a) # [1/8, 64]
        enc_a = self.audio_att_net(enc_a.unsqueeze(0)) # [1, 64]
            
        return enc_a


    def forward(self, x, a, e=None, c=None, va=None): # e는 face 훈련시에는 표정 정보로 들어가지만, mouth 훈련시에는 그냥 None으로 주어진다.
        # x: [N, 3], in [-bound, bound]
        enc_x = self.encode_x(x, bound=self.bound) # x, y, z 좌표를 XY, YZ, XZ 평면으로 나누어 각 평면별로 해시 그리드 인코더를 통과하여 피처를 추출

        # enc_x와 enc_w를 생성하여 h에 concat하는 과정
        enc_a = self.encode_audio(a) # 오디오 피처를 audio_net, audio_att_net에 순차적으로 통과하여 피처를 추출
        enc_a = enc_a.repeat(enc_x.shape[0], 1) # 프레임 수만큼 반복하여 각 프레임에 대한 오디오 피처를 추출
        aud_ch_att = self.aud_ch_att_net(enc_x) # 각 프레임에 대한 오디오 피처가 미치는 영향에 대한 가중치 계산
        enc_w = enc_a * aud_ch_att # 오디오 피처가 미치는 영향에 대한 가중치를 곱하여 오디오 피처를 추출
        h = torch.cat([enc_x, enc_w], dim=-1) # 각 프레임에 대한 오디오 피처와 오디오 피처가 미치는 영향에 대한 가중치를 연결

        # enc_e를 생성해서 h에 추가로 concat하는 과정 (face 훈련시에만 사용되고, mouth 훈련시에는 exp_eye가 False이므로 사용되지 않음)
        if self.exp_eye: # face 네트워크일 경우 True로, 표정 정보를 추가로 활용
            eye_att = torch.relu(self.eye_att_net(enc_x)) # 
            enc_e = self.exp_encode_net(e[:-1])
            enc_e = torch.cat([enc_e, e[-1:]], dim=-1)
            enc_e = enc_e * eye_att
            h = torch.cat([h, enc_e], dim=-1)
        
        # c를 생성해서 h에 추가로 concat하는 과정
        if c is not None:
            c = c.repeat(enc_x.shape[0], 1)
            h = torch.cat([h, c], dim=-1)

        # # va를 생성해서 h에 추가로 concat하는 과정 (use_va가 True일 때만 사용)
        # if self.use_va: # use_va가 True일 경우
        #     if va is not None: # va 정보가 제공될 경우
        #         enc_va = self.va_encode_net(va) # Valence와 Arousal을 va_encode_net에 통과시켜 피처 추출
        #         enc_va = enc_va.repeat(enc_x.shape[0], 1) # 프레임 수만큼 반복하여 각 프레임에 대한 va 피처를 추출
        #     else: # va 정보가 제공되지 않을 경우
        #         enc_va = torch.zeros(enc_x.shape[0], self.va_dim).to(enc_x.device) # 0으로 채운 텐서를 생성하여 차원 맞춤
        #     h = torch.cat([h, enc_va], dim=-1) # 추출된 va 피처를 h에 연결

        # h를 sigma_net에 통과시켜 프레임별 가우시안 움직임 계산
        h = self.sigma_net(h) 

        d_xyz = h[..., :3] * 1e-2
        d_rot = h[..., 3:7]

        # Face, Mouth Decomposition을 해도, 각 브랜치별로 모델이 따로 있는 것은 아니고, 모두 동일한 sigma_net을 사용.
        # 그러나 face를 훈련할 때는 투명도와 스케일 변화량도 return하고, mouth를 훈련할 때는 투명도와 스케일 변화량을 제외한 값을 return.
        if self.args.type == "face":
            d_opa = h[..., 7:8]
            d_scale = h[..., 8:11]
        else:
            d_opa = d_scale = None
            
        # enc_x를 align_net에 통과시켜 인물별 캐노니컬 가우시안과 UMF 움직임간의 스케일 보정
        p = self.align_net(enc_x)
        p_xyz = p[..., :3] * 1e-2
        p_scale = torch.tanh(p[..., 3:] / 5) * 0.25 + 1
        
        return {
            'd_xyz': d_xyz,
            'd_rot': d_rot,
            'd_opa': d_opa,
            'd_scale': d_scale,
            'ambient_aud' : aud_ch_att.norm(dim=-1, keepdim=True), # 오디오 피처가 가우시안들에 미치는 가중치
            'ambient_eye' : eye_att.norm(dim=-1, keepdim=True) if self.exp_eye else None, # 표정 피처가 가우시안들에 미치는 가중치
            'p_xyz': p_xyz,
            'p_scale': p_scale,
        }


    # get_params 함수는 모델의 각 서브 네트워크별로 학습률(lr)과 weight decay(wd) 등
    # 옵티마이저에 전달할 파라미터 그룹을 만들어서 리스트로 반환하는 함수이다.
    # 예를 들어, audio_net, encoder_xy, sigma_net 등 각 모듈별로
    # 서로 다른 학습률을 적용할 수 있도록 파라미터 그룹을 분리해서 관리한다.
    # 이 리스트를 옵티마이저(예: AdamW)에 넘기면, 각 그룹별로 지정한 학습률과 weight decay가 적용된다.
    # 즉, 네트워크의 각 부분을 세밀하게 튜닝할 수 있게 해주는 유틸 함수이다.
    def get_params(self, lr, lr_net, wd=0):

        params = [
            {'params': self.audio_net.parameters(), 'name': 'neural_audio_net', 'lr': lr_net, 'weight_decay': wd}, 
            # audio_net의 파라미터 그룹. lr_net 학습률과 wd를 적용한다.
            {'params': self.encoder_xy.parameters(), 'name': 'neural_encoder_xy', 'lr': lr},
            # encoder_xy의 파라미터 그룹. lr 학습률을 적용한다.
            {'params': self.encoder_yz.parameters(), 'name': 'neural_encoder_xy', 'lr': lr},
            # encoder_yz의 파라미터 그룹. lr 학습률을 적용한다.
            {'params': self.encoder_xz.parameters(), 'name': 'neural_encoder_xy', 'lr': lr},
            # encoder_xz의 파라미터 그룹. lr 학습률을 적용한다.
            {'params': self.sigma_net.parameters(), 'name': 'neural_sigma_net', 'lr': lr_net, 'weight_decay': wd},
            # sigma_net의 파라미터 그룹. lr_net 학습률과 wd를 적용한다.
            {'params': self.align_net.parameters(), 'name': 'neural_align_net', 'lr': lr_net / 2, 'weight_decay': wd},
            # align_net의 파라미터 그룹. lr_net의 절반 학습률과 wd를 적용한다.
        ]
        params.append({'params': self.audio_att_net.parameters(), 'name': 'neural_audio_att_net', 'lr': lr_net * 5, 'weight_decay': 0.0001})
        # audio_att_net의 파라미터 그룹. lr_net의 5배 학습률과 0.0001의 weight decay를 적용한다.
        if self.individual_dim > 0:
            params.append({'params': self.individual_codes, 'name': 'neural_individual_codes', 'lr': lr_net, 'weight_decay': wd})
            # individual_dim이 0보다 크면, 개인별 latent code 파라미터도 학습에 포함한다.

        params.append({'params': self.aud_ch_att_net.parameters(), 'name': 'neural_aud_ch_att_net', 'lr': lr_net, 'weight_decay': wd})
        # aud_ch_att_net의 파라미터 그룹. lr_net 학습률과 wd를 적용한다.

        if self.exp_eye:
            params.append({'params': self.eye_att_net.parameters(), 'name': 'neural_eye_att_net', 'lr': lr_net, 'weight_decay': wd})
            # exp_eye가 True면, eye_att_net 파라미터도 학습에 포함한다.
            params.append({'params': self.exp_encode_net.parameters(), 'name': 'neural_exp_encode_net', 'lr': lr_net, 'weight_decay': wd})
            # exp_encode_net 파라미터도 학습에 포함한다.

        # if self.use_va:
        #     params.append({'params': self.va_encode_net.parameters(), 'name': 'neural_va_encode_net', 'lr': lr_net, 'weight_decay': wd})

        return params
        # params 리스트를 반환한다. 이 리스트는 옵티마이저에 넘겨서 각 파라미터 그룹별로
        # 서로 다른 학습률과 weight decay를 적용할 수 있게 한다.