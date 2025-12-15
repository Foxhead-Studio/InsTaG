import numpy as np  # numpy 라이브러리 임포트 (행렬, 배열 연산에 사용)
from scipy.io import loadmat  # .mat 파일을 읽기 위한 scipy 함수 임포트

# 이 스크립트는 BFM(Basel Face Model) 데이터를 InsTaG 프로젝트에서 사용할 수 있는 형태로 변환하는 역할을 합니다.

original_BFM = loadmat("3DMM/01_MorphableModel.mat")  # BFM의 원본 .mat 파일을 불러옴 (예: 3D 얼굴 모델 데이터)
sub_inds = np.load("3DMM/topology_info.npy", allow_pickle=True).item()["sub_inds"]  # 사용할 얼굴 부분의 인덱스 정보 불러오기 (예: 하위 메쉬 인덱스)

shapePC = original_BFM["shapePC"]  # 얼굴 형태 주성분(Principal Components) 불러오기 (예: 얼굴 모양을 결정하는 벡터들)
shapeEV = original_BFM["shapeEV"]  # 얼굴 형태 고유값(Eigenvalues) 불러오기 (예: 각 주성분의 분산)
shapeMU = original_BFM["shapeMU"]  # 평균 얼굴 형태 벡터 불러오기 (예: 평균 얼굴 좌표)
texPC = original_BFM["texPC"]      # 얼굴 텍스처 주성분 불러오기 (예: 얼굴 색상/질감의 주성분)
texEV = original_BFM["texEV"]      # 얼굴 텍스처 고유값 불러오기 (예: 텍스처 주성분의 분산)
texMU = original_BFM["texMU"]      # 평균 얼굴 텍스처 벡터 불러오기 (예: 평균 얼굴 색상/질감)

b_shape = shapePC.reshape(-1, 199).transpose(1, 0).reshape(199, -1, 3)  # (예: [N*3, 199] → [199, N, 3]) 형태로 변환하여 각 주성분별 3D 좌표로 만듦
mu_shape = shapeMU.reshape(-1, 3)  # 평균 얼굴 형태를 (N, 3) 형태로 변환 (예: 각 점의 3D 좌표)

b_tex = texPC.reshape(-1, 199).transpose(1, 0).reshape(199, -1, 3)  # 텍스처 주성분도 동일하게 변환 (예: [N*3, 199] → [199, N, 3])
mu_tex = texMU.reshape(-1, 3)  # 평균 텍스처를 (N, 3) 형태로 변환 (예: 각 점의 RGB 값)

b_shape = b_shape[:, sub_inds, :].reshape(199, -1)  # 필요한 부분(sub_inds)만 추출 후 (199, M*3) 형태로 변환 (예: 하위 메쉬만 사용)
mu_shape = mu_shape[sub_inds, :].reshape(-1)        # 평균 얼굴도 sub_inds만 추출 후 1차원 벡터로 변환
b_tex = b_tex[:, sub_inds, :].reshape(199, -1)      # 텍스처 주성분도 sub_inds만 추출 후 (199, M*3) 형태로 변환
mu_tex = mu_tex[sub_inds, :].reshape(-1)            # 평균 텍스처도 sub_inds만 추출 후 1차원 벡터로 변환

exp_info = np.load("3DMM/exp_info.npy", allow_pickle=True).item()  # 표정 관련 정보 불러오기 (예: 평균 표정, 표정 주성분 등)
np.save(
    "3DMM/3DMM_info.npy",  # 최종적으로 변환된 3DMM 정보를 저장할 파일명
    {
        "mu_shape": mu_shape,  # 평균 얼굴 형태 (예: [M*3])
        "b_shape": b_shape,    # 얼굴 형태 주성분 (예: [199, M*3])
        "sig_shape": shapeEV.reshape(-1),  # 얼굴 형태 고유값 (예: [199])
        "mu_exp": exp_info["mu_exp"],      # 평균 표정 벡터 (예: [표정 차원])
        "b_exp": exp_info["base_exp"],     # 표정 주성분 (예: [표정 차원, ...])
        "sig_exp": exp_info["sig_exp"],    # 표정 고유값 (예: [표정 차원])
        "mu_tex": mu_tex,                  # 평균 텍스처 (예: [M*3])
        "b_tex": b_tex,                    # 텍스처 주성분 (예: [199, M*3])
        "sig_tex": texEV.reshape(-1),      # 텍스처 고유값 (예: [199])
    },
)  # 위에서 가공한 모든 정보를 하나의 npy 파일로 저장 (예: InsTaG에서 바로 사용할 수 있도록)