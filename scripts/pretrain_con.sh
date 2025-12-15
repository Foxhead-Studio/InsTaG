dataset='data/pretrain' # 변수에 값을 할당할 때 등호(=) 양쪽에 공백이 있으면 안 됩니다. 
# 예시: dataset= 'data/pretrain' (X), dataset='data/pretrain' (O)
workspace='debug_01' # workspace도 마찬가지로 공백 없이 할당해야 합니다.
gpu_id='0' # gpu_id도 동일하게 공백 없이 할당합니다.
audio_extractor='deepspeech' # deepspeech, esperanto, hubert
# 주석은 # 뒤에 한 칸 띄우는 것이 가독성에 좋습니다.

export CUDA_VISIBLE_DEVICES=$gpu_id # 환경 변수 설정은 정상적으로 작성되어 있습니다.

python pretrain_face.py -s $dataset -m $workspace --type face --init_num 2000 --densify_grad_threshold 0.0005 --audio_extractor $audio_extractor --iterations 30000
python pretrain_mouth.py -s $dataset -m $workspace --type mouth --init_num 5000 --audio_extractor $audio_extractor  --iterations 30000