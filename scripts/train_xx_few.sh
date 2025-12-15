dataset=$1 # data/Lieu
workspace=$2 # output/Lieu_debug
gpu_id=$3 # 0
audio_extractor='deepspeech' # deepspeech, esperanto, hubert, ave

pretrain_project_path="output/pretrain_ave/"

pretrain_face_path=${pretrain_project_path}/chkpnt_ema_face_latest.pth
pretrain_mouth_path=${pretrain_project_path}/chkpnt_ema_mouth_latest.pth

# n_views=500 # 20s
n_views=250 # 10s
# n_views=125 # 5s


export CUDA_VISIBLE_DEVICES=$gpu_id

python train_face.py --type face -s $dataset -m $workspace --init_num 2000 --densify_grad_threshold 0.0005 --audio_extractor $audio_extractor --pretrain_path $pretrain_face_path --iterations 10000 --sh_degree 1 --N_views $n_views
python train_mouth.py --type mouth -s $dataset -m $workspace --audio_extractor $audio_extractor --pretrain_path $pretrain_mouth_path --init_num 5000 --iterations 10000 --sh_degree 1 --N_views $n_views
python train_fuse_con.py -s $dataset -m $workspace --opacity_lr 0.001 --audio_extractor $audio_extractor --iterations 2000 --sh_degree 1 --N_views $n_views

python synthesize_fuse.py -s $dataset -m $workspace --eval --audio_extractor $audio_extractor --dilate
python metrics.py $workspace/test/ours_None/renders/out.mp4 $workspace/test/ours_None/gt/out.mp4