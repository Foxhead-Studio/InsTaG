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
import math
# from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene.motion_net import MotionNetwork, MouthMotionNetwork
from utils.sh_utils import eval_sh

'''
`render` 함수는 3D Gaussian Splatting 기술을 사용하여 장면을 렌더링하는 역할을 한다. 이 함수는 입력된 카메라 시점, Gaussian 모델, 렌더링 파이프라인 설정, 그리고 배경 색상을 기반으로 2D 이미지를 생성한다.

함수의 주요 기능은 다음과 같다.

*   **카메라 시점 및 Gaussian 모델 초기화**: 렌더링에 필요한 카메라 정보와 3D Gaussian 모델을 설정한다.
*   **스크린 공간 점 계산**: 3D Gaussian의 중심점을 2D 스크린 공간으로 투영하여 스크린 공간 점을 계산한다. 이는 2D 기울기(gradient) 계산에 사용될 수 있다.
*   **래스터화 설정**: 이미지 높이, 너비, 시야각(FoV), 배경색, 스케일 수정자, 뷰 및 투영 행렬 등 래스터화에 필요한 다양한 설정을 초기화한다. (래스터화는 3D 공간의 가우시안을 2D 이미지로 투영하여 픽셀로 표현하는 것이다.)
*   **Gaussian 래스터라이저 생성**: 위에서 설정한 래스터화 설정을 바탕으로 `GaussianRasterizer` 객체를 생성한다. 이 객체는 3D Gaussian을 2D 이미지로 변환하는 핵심적인 역할을 한다.
*   **Gaussian 속성 준비**: 3D Gaussian 모델로부터 평균(means3D), 불투명도(opacity), 스케일(scales), 회전(rotations) 또는 미리 계산된 3D 공분산(cov3D_precomp), 그리고 색상(colors_precomp) 또는 구면 고조파(shs)와 같은 속성들을 가져온다.
*   **색상 계산**: 만약 미리 계산된 색상이 제공되지 않았다면, 구면 고조파(Spherical Harmonics, SH)를 사용하여 카메라 시점에서 3D Gaussian의 색상을 계산한다. 이 과정에서 `eval_sh` 함수가 사용된다.
*   **렌더링 실행**: 준비된 모든 Gaussian 속성들을 `GaussianRasterizer`에 전달하여 최종 렌더링을 수행한다. 이 결과로 렌더링된 이미지, 깊이 맵, 법선 맵, 알파 맵, 그리고 Gaussian의 반지름 등의 정보가 반환된다.
*   **결과 반환**: 렌더링된 이미지와 추가적인 정보(뷰 공간 점, 가시성 필터, 깊이, 알파, 법선, 반지름)를 딕셔너리 형태로 반환한다.

아래는 각 코드 줄에 대한 자세한 설명이다.
'''

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    # render_pkg = render(viewpoint_cam, gaussians, pipe, background)
    """
    Render the scene.
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    # 3D 가우시안이 2D 스크린 공간으로 투영된 점들에 대해, 각 중심에 대한 기울기를 반환하기 위해 0으로 이루어진 텐서를 생성
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    # pc.get_xyz를 통해 얻은 xyz 좌표와 동일한 개수, 데이터 타입, 디바이스를 가지는 0으로 이루어진 텐서를 생성하고, 기울기 계산을 활성화
    try: # 예외 처리 진행
        screenspace_points.retain_grad() # `screenspace_points` 텐서에 대한 기울기를 유지하도록 설정한다.
    except: # 에러가 발생하면
        pass # 아무것도 하지 않고 그냥 넘어간다.

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5) # 카메라의 수평 시야각(FoVx)의 절반에 대한 탄젠트 값을 계산한다.
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5) # 카메라의 수직 시야각(FoVy)의 절반에 대한 탄젠트 값을 계산한다.

    raster_settings = GaussianRasterizationSettings( # Gaussian 래스터화 설정을 위한 객체를 생성한다.
        image_height=int(viewpoint_camera.image_height), # 렌더링될 이미지의 높이를 설정한다.
        image_width=int(viewpoint_camera.image_width), # 렌더링될 이미지의 너비를 설정한다.
        tanfovx=tanfovx, # 수평 시야각의 탄젠트 값을 설정한다.
        tanfovy=tanfovy, # 수직 시야각의 탄젠트 값을 설정한다.
        bg=bg_color, # 배경 색상을 설정한다.
        scale_modifier=scaling_modifier, # Gaussian의 스케일을 조절하는 수정자를 설정한다. 디폴트 값 1.0
        viewmatrix=viewpoint_camera.world_view_transform, # 월드 좌표계에서 카메라 시점으로의 변환 행렬을 설정한다.
        projmatrix=viewpoint_camera.full_proj_transform, # 전체 투영 변환 행렬을 설정한다.
        sh_degree=pc.active_sh_degree, # 구면 고조파(SH)의 차수를 설정한다.
        campos=viewpoint_camera.camera_center, # 카메라 궤적들의 중심 위치를 설정한다.
        prefiltered=False, # Gaussian이 미리 필터링되었는지 여부를 설정한다. 여기서는 필터링되지 않았다고 설정한다.
        debug=pipe.debug # 디버그 모드 활성화 여부를 설정한다.
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings) # 위에서 정의한 설정으로 GaussianRasterizer 객체를 생성한다.

    means3D = pc.get_xyz # 3D Gaussian의 3D 평균 위치(중심점)를 가져온다.
    means2D = screenspace_points # 위에서 생성한 스크린 공간의 2D 평균 위치 텐서를 사용한다.
    opacity = pc.get_opacity # 3D Gaussian의 불투명도 값을 가져온다.

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    # 미리 계산된 3D 공분산이 제공되면 사용하고, 그렇지 않으면 래스터라이저가 스케일/회전에서 계산한다.
    scales = None
    rotations = None
    cov3D_precomp = None
    # 스케일, 회전값, 미리 계산된 3D 공분산 값을 초기화 한다.
    if pipe.compute_cov3D_python: # 파이프라인에서 3D 공분산을 Python으로 계산하도록 설정되어 있다면,
        cov3D_precomp = pc.get_covariance(scaling_modifier) # Gaussian 모델에서 3D 공분산을 가져온다.
    else: # 그렇지 않다면, (디폴트)
        scales = pc.get_scaling # Gaussian 모델에서 스케일 값을 가져온다.
        rotations = pc.get_rotation # Gaussian 모델에서 회전 값을 가져온다.

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # 미리 계산된 색상이 제공되면 사용하고, 그렇지 않으면 Python에서 SH로부터 색상을 미리 계산한다.
    shs = None
    colors_precomp = None
    # SH와 미리 계산된 색상 값을 초기화 한다.
    if override_color is None: # 색상을 재정의하는 오버라이드 색상이 없다면,
        if pipe.convert_SHs_python: # 파이프라인에서 SH를 파이썬으로 변환하도록 설정되어 있다면,
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2) # Gaussian 모델의 특징(SH 계수)을 가져와 뷰에 맞게 형태를 변경한다.
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1)) # 각 Gaussian 중심에서 카메라 중심까지의 방향 벡터를 계산한다.
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True) # 방향 벡터를 정규화한다.
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized) # 정규화된 방향 벡터와 SH 계수를 사용하여 RGB 색상으로 변환한다.
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0) # 계산된 색상에 0.5를 더하고, 0 미만의 값은 0으로 클램핑하여 미리 계산된 색상으로 설정한다.
        else: # 파이프라인에서 SH를 파이썬으로 변환하지 않는다면, (디폴트)
            shs = pc.get_features # 가우시안 모델의 특징(SH 계수)을 직접 SH 값으로 사용한다.
    else: # 오버라이드 색상이 있다면,
        colors_precomp = override_color # 오버라이드 색상을 미리 계산된 색상으로 사용한다.
    
    # GaussianRasterizer를 사용하여 장면을 렌더링한다.
    rendered_image, rendered_depth, rendered_norm, rendered_alpha, radii, extra = rasterizer(
        means3D = means3D, # 3D 평균 위치를 입력한다.
        means2D = means2D, # 2D 평균 위치를 입력한다.
        shs = shs, # 구면 고조파(SH) 값을 입력한다.
        colors_precomp = colors_precomp, # 미리 계산된 색상 값을 입력한다.
        opacities = opacity, # 불투명도 값을 입력한다.
        scales = scales, # 스케일 값을 입력한다.
        rotations = rotations, # 회전 값을 입력한다.
        cov3Ds_precomp = cov3D_precomp, # 미리 계산된 3D 공분산 값을 입력한다.
        extra_attrs = torch.ones_like(opacity) # 추가 속성으로 불투명도와 동일한 크기의 1 텐서를 입력한다.
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # 시야 절두체 컬링되거나 반지름이 0인 Gaussian은 보이지 않는다.
    # 이러한 Gaussian들은 분할 기준에 사용되는 값 업데이트에서 제외된다.
    return {"render": rendered_image, # 렌더링된 이미지이다.
            "viewspace_points": screenspace_points, # 뷰 공간의 점들이다.
            "visibility_filter" : radii > 0, # 반지름이 0보다 큰 Gaussian만 포함하는 가시성 필터이다.
            "depth": rendered_depth, # 렌더링된 깊이 맵이다.
            "alpha": rendered_alpha, # 렌더링된 알파 맵이다.
            "normal": rendered_norm, # 렌더링된 법선 맵이다.
            "radii": radii} # 각 Gaussian의 반지름이다.

'''
`render_motion` 함수는 `render` 함수와 유사하게 3D Gaussian Splatting을 사용하여 장면을 렌더링하지만, 주요 차이점은 **움직임(motion)**을 모델링하고 렌더링하는 데 특화되어 있다는 점이다. 이 함수는 얼굴 움직임과 같은 동적인 요소를 장면에 통합한다.

함수의 주요 기능은 다음과 같다.

*   **카메라 및 Gaussian 모델 초기화**: `render` 함수와 마찬가지로 카메라 시점과 3D Gaussian 모델을 설정한다.
*   **모션 네트워크 통합**: `MotionNetwork` 객체를 입력으로 받아, Gaussian의 3D 위치, 스케일, 회전에 대한 변위(displacement)를 예측한다. 이 변위는 오디오 특징(audio_feat) 및 표현 특징(exp_feat)과 같은 입력에 기반하여 계산된다.
*   **개인화된 움직임 처리**: `personalized` 플래그가 True일 경우, `pc.neural_motion_grid`를 통해 개인화된 움직임 예측을 수행하고, 이를 기본 움직임 예측에 추가하여 특정 인물의 고유한 움직임 특성을 반영한다.
*   **Gaussian 속성 업데이트**: 모션 네트워크에서 예측된 변위($d\_xyz$, $d\_scale$, $d\_rot$)를 Gaussian 모델의 원래 3D 위치, 스케일, 회전에 적용하여 동적으로 변화된 Gaussian 속성($means3D$, $scales$, $rotations$)을 생성한다.
*   **래스터화 및 렌더링**: 업데이트된 Gaussian 속성을 `GaussianRasterizer`에 전달하여 장면을 렌더링하고, 렌더링된 이미지, 깊이 맵, 법선 맵, 알파 맵 등을 반환한다.
*   **어텐션 맵 렌더링**: `return_attn` 플래그가 True일 경우, 모션 네트워크에서 예측된 어텐션(ambient_aud, ambient_eye)을 기반으로 별도의 어텐션 맵($rendered\_attn$, $p\_rendered\_attn$)을 렌더링한다. 이는 특정 부위(예: 입, 눈)의 움직임에 대한 기여도를 시각화하는 데 사용될 수 있다.
*   **결과 반환**: 렌더링된 이미지 및 깊이, 알파, 법선 등의 정보와 함께 모션 예측 결과, 개인화된 모션 예측 결과, 그리고 어텐션 맵을 포함하는 딕셔너리를 반환한다.

아래는 각 코드 줄에 대한 자세한 설명이다.
'''

def render_motion(viewpoint_camera, pc : GaussianModel, motion_net : MotionNetwork, pipe, bg_color : torch.Tensor, \
                    scaling_modifier=1.0, frame_idx=None, return_attn=False, personalized=False, align=False, detach_motion=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    # # 래스터화 설정을 진행한다.
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings) # 위에서 정의한 설정으로 GaussianRasterizer 객체를 생성한다. (여기까지 render 함수와 동일)
    
    audio_feat = viewpoint_camera.talking_dict["auds"].cuda() # 카메라의 `talking_dict`에서 오디오 특징을 가져와 GPU로 이동시킨다. 이는 `motion_net`의 입력으로 사용된다.
    exp_feat = viewpoint_camera.talking_dict["au_exp"].cuda() # 카메라의 `talking_dict`에서 표현 특징(AU_expression)을 가져와 GPU로 이동시킨다. 이는 `motion_net`의 입력으로 사용된다.
    # AU1, 4, 5, 6, 7, 45의 강도 특징을 담고 있는 변수이다.
    
    xyz = pc.get_xyz # Gaussian 모델의 현재 3D 위치를 가져온다.

    # PMF로부터 개인적인 움직임 변화량 예측
    if personalized or align: # `personalized` 또는 `align` 플래그가 True인 경우, (디폴트)
        p_motion_preds = pc.neural_motion_grid(pc.get_xyz, audio_feat, exp_feat) # Gaussian 모델의 `neural_motion_grid`를 사용하여 개인화된 움직임 예측을 수행한다.
        # pc.neural_motion_grid는 PersonalizedMotionNetwork(신경 모션 네트워크)로, 캐노니컬 가우시안으로부터의 가우시안 변위량을 계산한다.
    
    if align: # `align` 플래그가 True인 경우, (디폴트)
        xyz = xyz + p_motion_preds['p_xyz'] # UMF와 캐노니컬 가우시안 간의 스케일 보정을 위해 추가적인 변위를 적용한다.
        # pass
    
    # UMF로부터 보편적인 움직임 변화량 예측
    motion_preds = motion_net(xyz, audio_feat, exp_feat)
    
    
    d_xyz = motion_preds['d_xyz']
    d_scale = motion_preds['d_scale']
    d_rot = motion_preds['d_rot']
    
    # UMF의 변위에 PMF의 변위 합치기
    if personalized: # True 디폴트
        # d_xyz *= (1 + p_motion_preds['p_scale'])
        d_xyz += p_motion_preds['d_xyz']
        d_scale += p_motion_preds['d_scale']
        d_rot += p_motion_preds['d_rot']
    
    if align: # True 디폴트
        d_xyz *= p_motion_preds['p_scale'] # 3D 위치 변위를 개인화된 스케일 예측으로 스케일링한다.
    
    # 특정한 이유로 모션 네트워크의 파라미터를 업데이트 하고 싶지 않다면
    if detach_motion: # `detach_motion` 플래그가 True인 경우,
        d_xyz = d_xyz.detach() # 3D 위치 변위의 기울기 계산을 비활성화한다.
        d_scale = d_scale.detach() # 스케일 변위의 기울기 계산을 비활성화한다.
        d_rot = d_rot.detach() # 회전 변위의 기울기 계산을 비활성화한다.

    means3D = pc.get_xyz + d_xyz # Gaussian 모델의 원래 3D 위치에 예측된 3D 위치 변위를 더하여 최종 3D 평균 위치를 계산한다.
    means2D = screenspace_points # 스크린 공간의 2D 평균 위치 텐서를 사용한다.
    # opacity = pc.opacity_activation(pc._opacity + motion_preds['d_opa']) # 주석 처리된 코드이다. 불투명도에 대한 변위를 고려하여 최종 불투명도를 계산하는 코드이다.
    opacity = pc.get_opacity # Gaussian 모델의 불투명도 값을 가져온다.

    cov3D_precomp = None # 미리 계산된 3D 공분산 값을 초기화한다. 여기서는 사용되지 않는다.
    # scales = pc.get_scaling # 주석 처리된 코드이다. Gaussian 모델의 원래 스케일 값을 사용하는 코드이다.
    scales = pc.scaling_activation(pc._scaling + d_scale) # Gaussian 모델의 원래 스케일에 예측된 스케일 변위를 더하고 softplus 활성화 함수를 적용하여 최종 스케일 값을 양수로 만든다.
    rotations = pc.rotation_activation(pc._rotation + d_rot) # Gaussian 모델의 원래 회전에 예측된 회전 변위를 더하고 정규화 함수를 적용하여 단위 벡터로 만든다.

    colors_precomp = None # 미리 계산된 색상 값을 초기화한다.
    shs = pc.get_features # Gaussian 모델의 특징(SH 계수)을 가져와 SH 값으로 사용한다.

    # GaussianRasterizer를 사용하여 장면을 렌더링한다.
    rendered_image, rendered_depth, rendered_norm, rendered_alpha, radii, extra = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3Ds_precomp = cov3D_precomp,
        extra_attrs = torch.ones_like(opacity)
    )
    
    # Attn # 어텐션 관련 섹션이다.
    rendered_attn = p_rendered_attn = None # 렌더링된 어텐션 맵과 개인화된 렌더링 어텐션 맵을 초기화한다.
    if return_attn: # `return_attn` 플래그가 True인 경우,
        attn_precomp = torch.cat([motion_preds['ambient_aud'], motion_preds['ambient_eye'], torch.zeros_like(motion_preds['ambient_eye'])], dim=-1)
        # 모션 예측에서 얻은 오디오 및 눈 주변의 어텐션 특징을 결합하여 미리 계산된 어텐션 값을 생성한다. 마지막은 0으로 채워진다.
        rendered_attn, _, _, _, _, _ = rasterizer( # 래스터라이저를 사용하여 어텐션 맵을 렌더링한다.
            means3D = means3D.detach(), # 3D 평균 위치는 기울기 계산을 비활성화한 상태로 입력한다.
            means2D = means2D, # 2D 평균 위치를 입력한다.
            shs = None, # SH 값은 사용하지 않는다.
            colors_precomp = attn_precomp, # 미리 계산된 어텐션 값을 색상으로 사용한다.
            opacities = opacity.detach(), # 불투명도는 기울기 계산을 비활성화한 상태로 입력한다.
            scales = scales.detach(), # 스케일은 기울기 계산을 비활성화한 상태로 입력한다.
            rotations = rotations.detach(), # 회전은 기울기 계산을 비활성화한 상태로 입력한다.
            cov3Ds_precomp = cov3D_precomp, # 미리 계산된 3D 공분산 값을 입력한다.
            extra_attrs = torch.ones_like(opacity) # 추가 속성으로 불투명도와 동일한 크기의 1 텐서를 입력한다.
        ) # 어텐션 맵 렌더링 완료이다.
        
        if personalized: # `personalized` 플래그가 True인 경우,
            p_attn_precomp = torch.cat([p_motion_preds['ambient_aud'], p_motion_preds['ambient_eye'], torch.zeros_like(p_motion_preds['ambient_eye'])], dim=-1)
            # 개인화된 모션 예측에서 얻은 오디오 및 눈 주변의 어텐션 특징을 결합하여 미리 계산된 개인화된 어텐션 값을 생성한다.
            p_rendered_attn, _, _, _, _, _ = rasterizer( # 래스터라이저를 사용하여 개인화된 어텐션 맵을 렌더링한다.
                means3D = means3D.detach(), # 3D 평균 위치는 기울기 계산을 비활성화한 상태로 입력한다.
                means2D = means2D, # 2D 평균 위치를 입력한다.
                shs = None, # SH 값은 사용하지 않는다.
                colors_precomp = p_attn_precomp, # 미리 계산된 개인화된 어텐션 값을 색상으로 사용한다.
                opacities = opacity.detach(), # 불투명도는 기울기 계산을 비활성화한 상태로 입력한다.
                scales = scales.detach(), # 스케일은 기울기 계산을 비활성화한 상태로 입력한다.
                rotations = rotations.detach(), # 회전은 기울기 계산을 비활성화한 상태로 입력한다.
                cov3Ds_precomp = cov3D_precomp, # 미리 계산된 3D 공분산 값을 입력한다.
                extra_attrs = torch.ones_like(opacity) # 추가 속성으로 불투명도와 동일한 크기의 1 텐서를 입력한다.
            ) # 개인화된 어텐션 맵 렌더링 완료이다.


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "depth": rendered_depth, 
            "alpha": rendered_alpha,
            "normal": rendered_norm,
            "radii": radii,
            "motion": motion_preds, # 모션 예측 결과이다.
            "p_motion": p_motion_preds if personalized or align else None, # 개인화된 모션 예측 결과이다. `personalized` 또는 `align`이 True일 때만 반환된다.
            'attn': rendered_attn, # 렌더링된 어텐션 맵이다.
            "p_attn": p_rendered_attn} # 렌더링된 개인화된 어텐션 맵이다.

# 입술의 움직임을 렌더링하는 함수
# 이 때, 얼굴의 가우시안인 pc_face와 motion_net_face가 추가 입력으로 들어간다.
def render_motion_mouth_con(viewpoint_camera, pc : GaussianModel, motion_net : MouthMotionNetwork, pc_face : GaussianModel, motion_net_face : MotionNetwork, pipe, bg_color : torch.Tensor, \
                        scaling_modifier=1.0, frame_idx=None, return_attn=False, personalized=False, align=False, k=10, inference=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # 레스터라이저 세팅
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    audio_feat = viewpoint_camera.talking_dict["auds"].cuda()
    # 입술만 생성할 것이므로 render_motion 함수와 달리 au_exp는 사용하지 않는다.
   
    xyz = pc.get_xyz # Gaussian 모델의 현재 3D 위치를 가져온다.
    
    if personalized or align:
        p_motion_preds = pc.neural_motion_grid(pc.get_xyz, audio_feat) # Mouth 브랜치의 PMF인 PersonalizedMotionNetwork로부터 생성됨.
    # 개인화된 움직임을 예측할 것이라면, neural_motion_grid를 통해 캐노니컬 입술로부터 생성된 가우시안의 변위를 예측한다.

    if align:
        xyz = xyz + p_motion_preds['p_xyz'] # 캐노니컬 가우시안에 변위량을 더해준다.
        # pass
    
    if not inference:
        exp_feat = viewpoint_camera.talking_dict["au_exp"].cuda()
        # viewpoint_camera의 talking_dict에서 "au_exp" 키에 해당하는 값을 가져와서 GPU로 옮긴다.
        # "au_exp"는 표정(action unit expression) 피처로, 얼굴의 표정 정보를 담고 있다.
        exp_feat = torch.zeros_like(exp_feat)
        # exp_feat와 동일한 shape의 0으로 채워진 텐서를 만든다.
        # 즉, 표정 피처를 모두 0으로 초기화한다. (입술만 생성할 것이므로 표정 정보를 사용하지 않겠다는 의미)
        motion_preds_face = motion_net_face(pc_face.get_xyz, audio_feat, exp_feat) # Face 브랜치의 UMF인 MotionNetwork로부터 생성됨.
        # motion_net_face에 얼굴 가우시안의 3D 위치(pc_face.get_xyz), 오디오 피처(audio_feat), 0으로 채운 표정 피처(exp_feat)를 입력하여 얼굴의 움직임 예측값(motion_preds_face)을 얻는다.
        # 여기서 얻은 변화량은 단순히 어디가 가장 많이 움직였는지 파악하는데만 쓰이므로, 감정 정보는 필요가 없다.
    else:
        motion_preds_face = motion_net_face.cache
        # inference 모드일 때는 motion_net_face의 캐시된 결과를 사용한다.
        
    with torch.no_grad():
        motion_max, _ = motion_preds_face["d_xyz"][..., 1].topk(k, 0, True, True)
        # motion_preds_face["d_xyz"] 텐서에서 마지막 차원(1번 인덱스, 즉 y축 방향)의 값을 기준으로 내림차순 상위 k개를 뽑는다.
        # motion_max는 가장 많이 움직인(y축 기준) k개의 값, _는 해당 인덱스이다.
        motion_min, _ =  motion_preds_face["d_xyz"][..., 1].topk(k, 0, False, True)
        # 위와 동일하게, 오름차순(가장 적게 움직인) k개의 값을 뽑는다.
        move_feat = torch.as_tensor([[motion_max[-1], motion_min[-1], motion_max[-1] - motion_min[-1]]]).cuda() * 1e2
        # motion_max의 마지막 값(=k번째로 큰 값), motion_min의 마지막 값(=k번째로 작은 값), 그리고 두 값의 차이를 하나의 벡터로 만든다.
        # 이 벡터를 100배(1e2) 스케일링해서 move_feat으로 만든 뒤, GPU로 옮긴다.
        # move_feat은 입술 움직임의 범위 정보를 motion_net에 입력하기 위한 피처이다.
        
    motion_preds = motion_net(xyz, audio_feat, move_feat.detach()) # Mouth 브랜치의 UMF인 MouthMotionNetwork로부터 생성됨.
    # motion_net에 xyz(입술 가우시안 위치), 오디오 피처, move_feat(입술 움직임 범위 피처)를 입력하여 입술의 움직임 예측값(motion_preds)을 얻는다.
    d_xyz = motion_preds['d_xyz']
    # motion_preds에서 'd_xyz' 키에 해당하는 값을 d_xyz에 할당한다. (입술 가우시안의 3D 위치 변화량)
    # d_rot = motion_preds['d_rot']
    # (회전 변화량은 주석 처리되어 사용하지 않는다.)
    
    if personalized:
        # d_xyz *= (1 + p_motion_preds['p_scale'])
        # (개인화 스케일 적용은 주석 처리되어 있다.)
        d_xyz += p_motion_preds['d_xyz']
        # personalized가 True일 때, p_motion_preds의 'd_xyz'를 d_xyz에 더해준다.
        # 즉, 개인화된 움직임 변위량을 추가로 반영한다.
        # d_rot += p_motion_preds['d_rot']
        # (회전 변위량도 주석 처리되어 있다.)
                
    means3D = pc.get_xyz + d_xyz
    # 최종적으로 입술 가우시안의 3D 위치(means3D)는 원래 위치(pc.get_xyz)에 d_xyz(예측된 변위량)를 더해서 계산한다.
    means2D = screenspace_points
    # means2D는 스크린(2D) 좌표로, screenspace_points(gradient 추적용 0 텐서)를 그대로 사용한다.
    opacity = pc.get_opacity
    # opacity는 pc(입술 가우시안)의 불투명도 값을 가져온다.

    cov3D_precomp = None
    # 3D 공분산 미리 계산값은 사용하지 않으므로 None으로 설정한다.
    scales = pc.get_scaling
    # 가우시안의 스케일(크기) 값을 가져온다.
    rotations = pc.rotation_activation(pc._rotation) # + d_rot)
    # 가우시안의 회전값을 rotation_activation 함수로 활성화하여 가져온다.
    # d_rot(회전 변화량)은 주석 처리되어 더하지 않는다.

    colors_precomp = None
    # 색상 미리 계산값은 사용하지 않으므로 None으로 설정한다.
    shs = pc.get_features
    # 가우시안의 SH(Spherical Harmonics) 피처를 가져온다.

    rendered_image, rendered_depth, rendered_norm, rendered_alpha, radii, extra = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3Ds_precomp = cov3D_precomp,
        extra_attrs = torch.ones_like(opacity)
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "depth": rendered_depth, 
            "alpha": rendered_alpha,
            "radii": radii,
            "motion": motion_preds,
            "p_motion": p_motion_preds if personalized or align else None
            }



# def render_motion_emotion(viewpoint_camera, pc : GaussianModel, motion_net : MotionNetwork, pipe, bg_color : torch.Tensor, \
#                     scaling_modifier=1.0, frame_idx=None, return_attn=False, personalized=False, align=False, detach_motion=False, va = None):
#     """
#     Render the scene. 
    
#     Background tensor (bg_color) must be on GPU!
#     """
 
#     # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
#     screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
#     try:
#         screenspace_points.retain_grad()
#     except:
#         pass

#     # Set up rasterization configuration
#     # # 래스터화 설정을 진행한다.
#     tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
#     tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

#     raster_settings = GaussianRasterizationSettings(
#         image_height=int(viewpoint_camera.image_height),
#         image_width=int(viewpoint_camera.image_width),
#         tanfovx=tanfovx,
#         tanfovy=tanfovy,
#         bg=bg_color,
#         scale_modifier=scaling_modifier,
#         viewmatrix=viewpoint_camera.world_view_transform,
#         projmatrix=viewpoint_camera.full_proj_transform,
#         sh_degree=pc.active_sh_degree,
#         campos=viewpoint_camera.camera_center,
#         prefiltered=False,
#         debug=pipe.debug
#     )

#     rasterizer = GaussianRasterizer(raster_settings=raster_settings) # 위에서 정의한 설정으로 GaussianRasterizer 객체를 생성한다. (여기까지 render 함수와 동일)
    
#     audio_feat = viewpoint_camera.talking_dict["auds"].cuda() # 카메라의 `talking_dict`에서 오디오 특징을 가져와 GPU로 이동시킨다. 이는 `motion_net`의 입력으로 사용된다.
#     exp_feat = viewpoint_camera.talking_dict["au_exp"].cuda() # 카메라의 `talking_dict`에서 표현 특징(AU_expression)을 가져와 GPU로 이동시킨다. 이는 `motion_net`의 입력으로 사용된다.
    
#     # valence, arousal 피처를 결합하여 입력으로 사용한다.
#     # va = torch.cat([valence, arousal], dim=-1).cuda()
    
#     xyz = pc.get_xyz # Gaussian 모델의 현재 3D 위치를 가져온다.

#     # PMF로부터 개인적인 움직임 변화량 예측
#     if personalized or align: # `personalized` 또는 `align` 플래그가 True인 경우, (디폴트)
#         p_motion_preds = pc.neural_motion_grid(pc.get_xyz, audio_feat, exp_feat, va = va) # Gaussian 모델의 `neural_motion_grid`를 사용하여 개인화된 움직임 예측을 수행한다.
#         # pc.neural_motion_grid는 PersonalizedMotionNetwork(신경 모션 네트워크)로, 캐노니컬 가우시안으로부터의 가우시안 변위량을 계산한다.
    
#     if align: # `align` 플래그가 True인 경우, (디폴트)
#         xyz = xyz + p_motion_preds['p_xyz'] # UMF와 캐노니컬 가우시안 간의 스케일 보정을 위해 추가적인 변위를 적용한다.
#         # pass
    
#     # UMF로부터 보편적인 움직임 변화량 예측
#     motion_preds = motion_net(xyz, audio_feat, exp_feat)
    
    
#     d_xyz = motion_preds['d_xyz']
#     d_scale = motion_preds['d_scale']
#     d_rot = motion_preds['d_rot']
    
#     # UMF의 변위에 PMF의 변위 합치기
#     if personalized: # True 디폴트
#         # d_xyz *= (1 + p_motion_preds['p_scale'])
#         d_xyz += p_motion_preds['d_xyz']
#         d_scale += p_motion_preds['d_scale']
#         d_rot += p_motion_preds['d_rot']
    
#     if align: # True 디폴트
#         d_xyz *= p_motion_preds['p_scale'] # 3D 위치 변위를 개인화된 스케일 예측으로 스케일링한다.
    
#     # 특정한 이유로 모션 네트워크의 파라미터를 업데이트 하고 싶지 않다면
#     if detach_motion: # `detach_motion` 플래그가 True인 경우,
#         d_xyz = d_xyz.detach() # 3D 위치 변위의 기울기 계산을 비활성화한다.
#         d_scale = d_scale.detach() # 스케일 변위의 기울기 계산을 비활성화한다.
#         d_rot = d_rot.detach() # 회전 변위의 기울기 계산을 비활성화한다.

#     means3D = pc.get_xyz + d_xyz # Gaussian 모델의 원래 3D 위치에 예측된 3D 위치 변위를 더하여 최종 3D 평균 위치를 계산한다.
#     means2D = screenspace_points # 스크린 공간의 2D 평균 위치 텐서를 사용한다.
#     # opacity = pc.opacity_activation(pc._opacity + motion_preds['d_opa']) # 주석 처리된 코드이다. 불투명도에 대한 변위를 고려하여 최종 불투명도를 계산하는 코드이다.
#     opacity = pc.get_opacity # Gaussian 모델의 불투명도 값을 가져온다.

#     cov3D_precomp = None # 미리 계산된 3D 공분산 값을 초기화한다. 여기서는 사용되지 않는다.
#     # scales = pc.get_scaling # 주석 처리된 코드이다. Gaussian 모델의 원래 스케일 값을 사용하는 코드이다.
#     scales = pc.scaling_activation(pc._scaling + d_scale) # Gaussian 모델의 원래 스케일에 예측된 스케일 변위를 더하고 softplus 활성화 함수를 적용하여 최종 스케일 값을 양수로 만든다.
#     rotations = pc.rotation_activation(pc._rotation + d_rot) # Gaussian 모델의 원래 회전에 예측된 회전 변위를 더하고 정규화 함수를 적용하여 단위 벡터로 만든다.

#     colors_precomp = None # 미리 계산된 색상 값을 초기화한다.
#     shs = pc.get_features # Gaussian 모델의 특징(SH 계수)을 가져와 SH 값으로 사용한다.

#     # GaussianRasterizer를 사용하여 장면을 렌더링한다.
#     rendered_image, rendered_depth, rendered_norm, rendered_alpha, radii, extra = rasterizer(
#         means3D = means3D,
#         means2D = means2D,
#         shs = shs,
#         colors_precomp = colors_precomp,
#         opacities = opacity,
#         scales = scales,
#         rotations = rotations,
#         cov3Ds_precomp = cov3D_precomp,
#         extra_attrs = torch.ones_like(opacity)
#     )
    
#     # Attn # 어텐션 관련 섹션이다.
#     rendered_attn = p_rendered_attn = None # 렌더링된 어텐션 맵과 개인화된 렌더링 어텐션 맵을 초기화한다.
#     if return_attn: # `return_attn` 플래그가 True인 경우,
#         attn_precomp = torch.cat([motion_preds['ambient_aud'], motion_preds['ambient_eye'], torch.zeros_like(motion_preds['ambient_eye'])], dim=-1)
#         # 모션 예측에서 얻은 오디오 및 눈 주변의 어텐션 특징을 결합하여 미리 계산된 어텐션 값을 생성한다. 마지막은 0으로 채워진다.
#         rendered_attn, _, _, _, _, _ = rasterizer( # 래스터라이저를 사용하여 어텐션 맵을 렌더링한다.
#             means3D = means3D.detach(), # 3D 평균 위치는 기울기 계산을 비활성화한 상태로 입력한다.
#             means2D = means2D, # 2D 평균 위치를 입력한다.
#             shs = None, # SH 값은 사용하지 않는다.
#             colors_precomp = attn_precomp, # 미리 계산된 어텐션 값을 색상으로 사용한다.
#             opacities = opacity.detach(), # 불투명도는 기울기 계산을 비활성화한 상태로 입력한다.
#             scales = scales.detach(), # 스케일은 기울기 계산을 비활성화한 상태로 입력한다.
#             rotations = rotations.detach(), # 회전은 기울기 계산을 비활성화한 상태로 입력한다.
#             cov3Ds_precomp = cov3D_precomp, # 미리 계산된 3D 공분산 값을 입력한다.
#             extra_attrs = torch.ones_like(opacity) # 추가 속성으로 불투명도와 동일한 크기의 1 텐서를 입력한다.
#         ) # 어텐션 맵 렌더링 완료이다.
        
#         if personalized: # `personalized` 플래그가 True인 경우,
#             p_attn_precomp = torch.cat([p_motion_preds['ambient_aud'], p_motion_preds['ambient_eye'], torch.zeros_like(p_motion_preds['ambient_eye'])], dim=-1)
#             # 개인화된 모션 예측에서 얻은 오디오 및 눈 주변의 어텐션 특징을 결합하여 미리 계산된 개인화된 어텐션 값을 생성한다.
#             p_rendered_attn, _, _, _, _, _ = rasterizer( # 래스터라이저를 사용하여 개인화된 어텐션 맵을 렌더링한다.
#                 means3D = means3D.detach(), # 3D 평균 위치는 기울기 계산을 비활성화한 상태로 입력한다.
#                 means2D = means2D, # 2D 평균 위치를 입력한다.
#                 shs = None, # SH 값은 사용하지 않는다.
#                 colors_precomp = p_attn_precomp, # 미리 계산된 개인화된 어텐션 값을 색상으로 사용한다.
#                 opacities = opacity.detach(), # 불투명도는 기울기 계산을 비활성화한 상태로 입력한다.
#                 scales = scales.detach(), # 스케일은 기울기 계산을 비활성화한 상태로 입력한다.
#                 rotations = rotations.detach(), # 회전은 기울기 계산을 비활성화한 상태로 입력한다.
#                 cov3Ds_precomp = cov3D_precomp, # 미리 계산된 3D 공분산 값을 입력한다.
#                 extra_attrs = torch.ones_like(opacity) # 추가 속성으로 불투명도와 동일한 크기의 1 텐서를 입력한다.
#             ) # 개인화된 어텐션 맵 렌더링 완료이다.


#     # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
#     # They will be excluded from value updates used in the splitting criteria.
#     return {"render": rendered_image,
#             "viewspace_points": screenspace_points,
#             "visibility_filter" : radii > 0,
#             "depth": rendered_depth, 
#             "alpha": rendered_alpha,
#             "normal": rendered_norm,
#             "radii": radii,
#             "motion": motion_preds, # 모션 예측 결과이다.
#             "p_motion": p_motion_preds if personalized or align else None, # 개인화된 모션 예측 결과이다. `personalized` 또는 `align`이 True일 때만 반환된다.
#             'attn': rendered_attn, # 렌더링된 어텐션 맵이다.
#             "p_attn": p_rendered_attn} # 렌더링된 개인화된 어텐션 맵이다.
