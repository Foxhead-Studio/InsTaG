# 얼굴 파싱 모델의 학습된 가중치 파일을 다운로드합니다. 얼굴 영역을 분석하고 분할하는 데 사용됩니다.
wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_parsing/79999_iter.pth?raw=true -O data_utils/face_parsing/79999_iter.pth

# 3DMM(3D Morphable Model) 관련 데이터를 저장할 디렉토리를 생성합니다.
mkdir -p data_utils/face_tracking/3DMM

# 3DMM의 표정 정보(exp_info.npy) 파일을 다운로드합니다.
wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_tracking/3DMM/exp_info.npy?raw=true -O data_utils/face_tracking/3DMM/exp_info.npy
# 3DMM의 키 정보(keys_info.npy) 파일을 다운로드합니다.
wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_tracking/3DMM/keys_info.npy?raw=true -O data_utils/face_tracking/3DMM/keys_info.npy
# 3DMM의 서브 메쉬(sub_mesh.obj) 파일을 다운로드합니다.
wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_tracking/3DMM/sub_mesh.obj?raw=true -O data_utils/face_tracking/3DMM/sub_mesh.obj
# 3DMM의 토폴로지 정보(topology_info.npy) 파일을 다운로드합니다.
wget https://github.com/YudongGuo/AD-NeRF/blob/master/data_util/face_tracking/3DMM/topology_info.npy?raw=true -O data_utils/face_tracking/3DMM/topology_info.npy

# 오디오-비주얼 인코더의 학습된 가중치 파일을 다운로드합니다.
wget https://github.com/ZiqiaoPeng/SyncTalk/blob/main/nerf_triplane/checkpoints/audio_visual_encoder.pth?raw=true -O data_utils/audio_visual_encoder.pth