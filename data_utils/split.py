import os
import glob
import tqdm
import json
import argparse
import cv2
import numpy as np


def save_transforms(base_dir, ori_imgs_dir):
    print(f'[INFO] ===== save transforms =====')

    import torch

    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))
    
    # read one image to get H/W
    tmp_image = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED) # [H, W, 3]
    h, w = tmp_image.shape[:2]

    params_dict = torch.load(os.path.join(base_dir, 'track_params.pt'))
    focal_len = params_dict['focal']
    euler_angle = params_dict['euler']
    trans = params_dict['trans'] / 10.0
    valid_num = euler_angle.shape[0]

    def euler2rot(euler_angle):
        batch_size = euler_angle.shape[0]
        theta = euler_angle[:, 0].reshape(-1, 1, 1)
        phi = euler_angle[:, 1].reshape(-1, 1, 1)
        psi = euler_angle[:, 2].reshape(-1, 1, 1)
        one = torch.ones((batch_size, 1, 1), dtype=torch.float32, device=euler_angle.device)
        zero = torch.zeros((batch_size, 1, 1), dtype=torch.float32, device=euler_angle.device)
        rot_x = torch.cat((
            torch.cat((one, zero, zero), 1),
            torch.cat((zero, theta.cos(), theta.sin()), 1),
            torch.cat((zero, -theta.sin(), theta.cos()), 1),
        ), 2)
        rot_y = torch.cat((
            torch.cat((phi.cos(), zero, -phi.sin()), 1),
            torch.cat((zero, one, zero), 1),
            torch.cat((phi.sin(), zero, phi.cos()), 1),
        ), 2)
        rot_z = torch.cat((
            torch.cat((psi.cos(), -psi.sin(), zero), 1),
            torch.cat((psi.sin(), psi.cos(), zero), 1),
            torch.cat((zero, zero, one), 1)
        ), 2)
        return torch.bmm(rot_x, torch.bmm(rot_y, rot_z))


    # train_val_split = int(valid_num*0.5)
    train_val_split = valid_num - 25 * 12 - 1 # take the last 12s as valid set.
    # train_val_split = int(valid_num * 10 / 11)

    train_ids = torch.arange(0, train_val_split)
    val_ids = torch.arange(train_val_split, valid_num)

    rot = euler2rot(euler_angle)
    rot_inv = rot.permute(0, 2, 1)
    trans_inv = -torch.bmm(rot_inv, trans.unsqueeze(2))

    pose = torch.eye(4, dtype=torch.float32)
    save_ids = ['train', 'val']
    train_val_ids = [train_ids, val_ids]
    mean_z = -float(torch.mean(trans[:, 2]).item())

    for split in range(2):
        transform_dict = dict()
        transform_dict['focal_len'] = float(focal_len[0])
        transform_dict['cx'] = float(w/2.0)
        transform_dict['cy'] = float(h/2.0)
        transform_dict['frames'] = []
        ids = train_val_ids[split]
        save_id = save_ids[split]

        for i in ids:
            i = i.item()
            frame_dict = dict()
            frame_dict['img_id'] = i
            frame_dict['aud_id'] = i

            pose[:3, :3] = rot_inv[i]
            pose[:3, 3] = trans_inv[i, :, 0]

            frame_dict['transform_matrix'] = pose.numpy().tolist()

            transform_dict['frames'].append(frame_dict)

        with open(os.path.join(base_dir, 'transforms_' + save_id + '.json'), 'w') as fp:
            json.dump(transform_dict, fp, indent=2, separators=(',', ': '))

    print(f'[INFO] ===== finished saving transforms =====')


'''
학습/검증 데이터셋 분할: 전체 프레임(valid_num)을 학습(train_ids)과 검증(val_ids) 세트로 나눕니다.
split.py에서는 train_val_split = valid_num - 25 * 12 - 1로 설정되어, 마지막 12초 분량의 프레임을 검증 세트로 사용하는 것으로 보입니다 (프레임 레이트가 25fps라고 가정).
process.py에서는 train_val_split = int(valid_num * 10 / 11)로 설정되어, 전체 프레임의 약 1/11을 검증 세트로 사용합니다.
transforms_*.json 파일 생성 및 저장: 각 프레임에 대한 transform_matrix (월드 좌표계 변환 행렬)와 카메라 내외부 파라미터(초점 거리, 이미지 중심)를 포함하는 JSON 파일을 transforms_train.json과 transforms_val.json으로 저장합니다.
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="path to video file")
    parser.add_argument('--task', type=int, default=-1, help="-1 means all")
    parser.add_argument('--asr', type=str, default='deepspeech', help="wav2vec or deepspeech")

    opt = parser.parse_args()

    base_dir = os.path.dirname(opt.path)
    
    wav_path = os.path.join(base_dir, 'aud.wav')
    ori_imgs_dir = os.path.join(base_dir, 'ori_imgs')
    parsing_dir = os.path.join(base_dir, 'parsing')
    gt_imgs_dir = os.path.join(base_dir, 'gt_imgs')
    torso_imgs_dir = os.path.join(base_dir, 'torso_imgs')

    os.makedirs(ori_imgs_dir, exist_ok=True)
    os.makedirs(parsing_dir, exist_ok=True)
    os.makedirs(gt_imgs_dir, exist_ok=True)
    os.makedirs(torso_imgs_dir, exist_ok=True)

    # save transforms.json
    if opt.task == -1 or opt.task == 9:
        save_transforms(base_dir, ori_imgs_dir)