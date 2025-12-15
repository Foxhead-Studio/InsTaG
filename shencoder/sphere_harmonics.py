import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd 

try:
    import _shencoder as _backend
except ImportError:
    from .backend import _backend

class _sh_encoder(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32) # force float32 for better precision
    def forward(ctx, inputs, degree, calc_grad_inputs=False):
        # inputs: [B, input_dim], float in [-1, 1]
        # RETURN: [B, F], float

        inputs = inputs.contiguous()
        B, input_dim = inputs.shape # batch size, coord dim
        output_dim = degree ** 2
        
        outputs = torch.empty(B, output_dim, dtype=inputs.dtype, device=inputs.device)

        if calc_grad_inputs:
            dy_dx = torch.empty(B, input_dim * output_dim, dtype=inputs.dtype, device=inputs.device)
        else:
            dy_dx = None

        _backend.sh_encode_forward(inputs, outputs, B, input_dim, degree, dy_dx)

        ctx.save_for_backward(inputs, dy_dx)
        ctx.dims = [B, input_dim, degree]

        return outputs
    
    @staticmethod
    #@once_differentiable
    @custom_bwd
    def backward(ctx, grad):
        # grad: [B, C * C]

        inputs, dy_dx = ctx.saved_tensors

        if dy_dx is not None:
            grad = grad.contiguous()
            B, input_dim, degree = ctx.dims
            grad_inputs = torch.zeros_like(inputs)
            _backend.sh_encode_backward(grad, inputs, B, input_dim, degree, dy_dx, grad_inputs)
            return grad_inputs, None, None
        else:
            return None, None, None



sh_encode = _sh_encoder.apply


class SHEncoder(nn.Module):
    def __init__(self, input_dim=3, degree=4):
        super().__init__()
        # input_dim은 입력 좌표의 차원 수를 의미한다. SH 인코더는 3차원 좌표만 지원한다.
        self.input_dim = input_dim # coord dims, must be 3
        # degree는 SH(Spherical Harmonics)의 차수를 의미한다. 0~4(혹은 최대 8)까지 지원한다.
        self.degree = degree # 0 ~ 4
        # output_dim은 SH 계수의 개수로, degree^2로 계산된다.
        # 예시: degree=3이면 output_dim=9, degree=4면 output_dim=16
        self.output_dim = degree ** 2

        # input_dim이 3이 아니면 에러를 발생시킨다.
        assert self.input_dim == 3, "SH encoder only support input dim == 3"
        # degree가 1~8 범위를 벗어나면 에러를 발생시킨다.
        assert self.degree > 0 and self.degree <= 8, "SH encoder only supports degree in [1, 8]"
        
    def __repr__(self):
        # 객체를 print할 때 input_dim과 degree 정보를 출력한다.
        return f"SHEncoder: input_dim={self.input_dim} degree={self.degree}"
    
    def forward(self, inputs, size=1):
        # inputs: [..., input_dim] 형태의 텐서다. normalized real world positions in [-size, size]
        # 예시: (N, 3) 또는 (B, N, 3) 등 다양한 shape가 들어올 수 있다.
        # size는 입력 좌표의 정규화 범위로, 기본값은 1이다.
        # return: [..., degree^2] 형태의 SH 인코딩 결과를 반환한다.

        # 입력 좌표를 [-1, 1] 범위로 정규화한다.
        # 예시: size=2이면, inputs가 [-2, 2] 범위라면 [-1, 1]로 맞춰준다.
        inputs = inputs / size # [-1, 1]

        # 입력의 prefix shape(마지막 차원 제외 shape)을 저장한다.
        # 예시: inputs.shape = (B, N, 3)라면 prefix_shape = [B, N]
        prefix_shape = list(inputs.shape[:-1])
        # 입력을 (전체 개수, 3) 형태로 reshape한다.
        # 예시: (B, N, 3) → (B*N, 3)
        inputs = inputs.reshape(-1, self.input_dim)

        # sh_encode 함수로 SH 인코딩을 수행한다.
        # inputs.requires_grad가 True면, 입력에 대해 gradient를 계산한다.
        # outputs shape: (B*N, degree^2)
        outputs = sh_encode(inputs, self.degree, inputs.requires_grad)
        # 다시 원래 prefix shape에 맞게 reshape한다.
        # 예시: (B*N, degree^2) → (B, N, degree^2)
        outputs = outputs.reshape(prefix_shape + [self.output_dim])

        # SH 인코딩 결과를 반환한다.
        return outputs