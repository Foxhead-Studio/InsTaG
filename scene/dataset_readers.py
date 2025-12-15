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

import os
import sys
import torch
from PIL import Image
from typing import NamedTuple
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from tqdm import tqdm
import pandas as pd
import glob

from utils.sh_utils import SH2RGB
from utils.audio_utils import get_audio_features
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    background: np.array
    talking_dict: dict

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
# pretrain과 동일
def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".jpg", audio_file='', audio_extractor='deepspeech', N_views=-1, preload=True):
    cam_infos = []
    postfix_dict = {"deepspeech": "_ds", "esperanto": "_eo", "hubert": "_hu"}
    
    N_views = N_views if "train" in transformsfile and audio_file=='' else -1 # 이게 다른 점. 여기선 250개의 이미지만 샘플링 한다. 처음에 args 줄 때, "--N_views", "10"로 하면 10개만 샘플링.
    # transformsfile = 'transforms_val.json'로 주어지면 N_views = -1이 되어 모든 프레임을 샘플링 한다.
    with open(os.path.join(path, transformsfile)) as json_file: # train일 경우 transformsfile = 'transforms_train.json'
        contents = json.load(json_file)
        focal_len = contents["focal_len"] # 카메라 초점 길이는 모든 프레임에서 동일하므로 미리 가져옴. 1300.0

        frames = contents["frames"][:N_views] # [:250] 프레임만 샘플링 한다. transformsfile = 'transforms_val.json'로 주어지면 N_views = -1이 되어 모든 프레임을 샘플링 한다.
        
        if audio_extractor == "ave":
            from torch.utils.data import DataLoader
            from scene.motion_net import AudioEncoder
            from utils.audio_utils import AudDataset
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = AudioEncoder().to(device).eval()
            ckpt = torch.load('./data_utils/audio_visual_encoder.pth')
            model.load_state_dict({f'audio_encoder.{k}': v for k, v in ckpt.items()})
            read_cache = False
            if audio_file == '':
                if os.path.exists(os.path.join(path, 'aud_ave.npy')):
                    aud_features = np.load(os.path.join(path, 'aud_ave.npy'))
                    print(aud_features.shape)
                    read_cache = True
                else:
                    dataset = AudDataset(os.path.join(path, 'aud.wav')) 
            else:
                dataset = AudDataset(audio_file)
            if not read_cache:
                data_loader = DataLoader(dataset, batch_size=128, shuffle=False)
                outputs = []
                for mel in tqdm(data_loader):
                    mel = mel.to(device)
                    with torch.no_grad():
                        out = model(mel)
                    outputs.append(out)
                outputs = torch.cat(outputs, dim=0).cpu()
                first_frame, last_frame = outputs[:1], outputs[-1:]
                aud_features = torch.cat([first_frame.repeat(2, 1), outputs, last_frame.repeat(2, 1)],
                                            dim=0).unsqueeze(0).permute(1, 2, 0).numpy()
                if audio_file == '':
                    np.save(os.path.join(path, 'aud_ave.npy'), aud_features)
            # aud_features = np.load(os.path.join(self.root_path, 'aud_ave.npy'))
        elif audio_file == '': # 디폴트로 해당 오디오 피처 넘파이를 로드.
            aud_features = np.load(os.path.join(path, 'aud{}.npy'.format(postfix_dict[audio_extractor]))) # '/home/white/github/InsTaG/data/Lieu/aud.npy'
        else:
            aud_features = np.load(audio_file)
        aud_features = torch.from_numpy(aud_features)
        aud_features = aud_features.float().permute(0, 2, 1)
        auds = aud_features
        
        # loop
        if audio_file != '': # 건너 뜀
            loop_time = auds.shape[0] // len(frames) + 1
            frames *= loop_time
        

        au_info=pd.read_csv(os.path.join(path, 'au.csv'))
        au_blink = au_info['AU45_r'].values
        au25 = au_info['AU25_r'].values
        au25 = np.clip(au25[:N_views], 0, np.percentile(au25[:N_views], 95)) # [:N_views] 프레임만 샘플링 한다.

        au25_25, au25_50, au25_75, au25_100 = np.percentile(au25, 25), np.percentile(au25, 50), np.percentile(au25, 75), au25.max()

        au_exp = [] # 얼굴 액션 유닛(AU1(눈썹 안쪽을 들어올리는 동작), AU4(눈썹을 아래로 내리는 동작), AU5(위 눈꺼풀을 들어올리는 동작), AU6(볼을 들어올리는 동작), AU7(눈꺼풀을 단단히 조이는 동작), AU45(눈을 깜빡이는 동작))들의 강도 값을 담고 있는 PyTorch 텐서.
        for i in [1,4,5,6,7,45]:
            _key = 'AU' + str(i).zfill(2) + '_r'
            au_exp_t = au_info[_key].values
            if i == 45:
                au_exp_t = au_exp_t.clip(0, 2)
            au_exp.append(au_exp_t[:, None])
        au_exp = np.concatenate(au_exp, axis=-1, dtype=np.float32)

        ldmks_lips = []
        ldmks_mouth = []
        ldmks_lhalf = []
        
        for idx, frame in tqdm(enumerate(frames)):
            lms = np.loadtxt(os.path.join(path, 'ori_imgs', str(frame['img_id']) + '.lms')) # [68, 2]
            lips = slice(48, 60) # 입술 바깥 경계
            mouth = slice(60, 68) # 입술 안쪽 경계
            xmin, xmax = int(lms[lips, 1].min()), int(lms[lips, 1].max())
            ymin, ymax = int(lms[lips, 0].min()), int(lms[lips, 0].max())

            ldmks_lips.append([int(xmin), int(xmax), int(ymin), int(ymax)])
            ldmks_mouth.append([int(lms[mouth, 1].min()), int(lms[mouth, 1].max())])

            lh_xmin, lh_xmax = int(lms[31:36, 1].min()), int(lms[:, 1].max()) # actually lower half area
            xmin, xmax = int(lms[:, 1].min()), int(lms[:, 1].max())
            ymin, ymax = int(lms[:, 0].min()), int(lms[:, 0].max())
            # self.face_rect.append([xmin, xmax, ymin, ymax])
            ldmks_lhalf.append([lh_xmin, lh_xmax, ymin, ymax])
            
        ldmks_lips = np.array(ldmks_lips)
        ldmks_mouth = np.array(ldmks_mouth)
        ldmks_lhalf = np.array(ldmks_lhalf)
        mouth_lb = (ldmks_mouth[:, 1] - ldmks_mouth[:, 0]).min()
        mouth_ub = (ldmks_mouth[:, 1] - ldmks_mouth[:, 0]).max()



        for idx, frame in tqdm(enumerate(frames)):
            cam_name = os.path.join(path, 'gt_imgs', str(frame["img_id"]) + extension) # '/home/white/github/InsTaG/data/Lieu/gt_imgs/0.jpg'

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            talking_dict = {}
            talking_dict['img_id'] = frame['img_id']
            
            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            talking_dict['image_path'] = image_path
            if preload or idx==0:
                image = Image.open(image_path)
                w, h = image.size[0], image.size[1]
                image = np.array(image.convert("RGB"))

            # torso, bg 이미지를 추가하는 것이 다른 점
            torso_img_path = os.path.join(path, 'torso_imgs', str(frame['img_id']) + '.png') # 머리가 없고, 목부터 상체만 있으며 뒷 배경이 투명색인 이미지 '/home/white/github/InsTaG/data/Lieu/torso_imgs/0.png'
            bg_img_path = os.path.join(path, 'bc.jpg') # 백그라운드 이미지. '/home/white/github/InsTaG/data/Lieu/bc.jpg'
            talking_dict['torso_img_path'] = torso_img_path
            talking_dict['bg_img_path'] = bg_img_path # 각각의 데이터 디렉토리를 추가
            if preload: # True
                torso_img = np.array(Image.open(torso_img_path).convert("RGBA")) * 1.0 # 토르소 이미지를 RGBA로 변환한 후 넘파이 배열로 로드. 1.0을 곱해 float로 변환.
                bg_img = np.array(Image.open(bg_img_path).convert("RGB")) # 백그라운드 이미지를 RGB로 변환한 후 넘파이 배열로 로드.
                bg = torso_img[..., :3] * torso_img[..., 3:] / 255.0 + bg_img * (1 - torso_img[..., 3:] / 255.0) # torso_img의 RGB 채널에 alpha를 곱하고, bg_img에는 (1 - alpha)를 곱해서 더한다.
                bg = bg.astype(np.uint8) # float 타입의 bg를 uint8(0~255)로 변환한다.
                
            if not preload: # 건너 뜀
                image = bg = None

            teeth_mask_path = os.path.join(path, 'teeth_mask', str(frame['img_id']) + '.npy')
            mask_path = os.path.join(path, 'parsing', str(frame['img_id']) + '.png')
            talking_dict['teeth_mask_path'] = teeth_mask_path
            talking_dict['mask_path'] = mask_path
            if preload:
                teeth_mask = np.load(teeth_mask_path)
                mask = np.array(Image.open(mask_path).convert("RGB")) * 1.0
                talking_dict['face_mask'] = (mask[:, :, 2] > 254) * (mask[:, :, 0] == 0) * (mask[:, :, 1] == 0) ^ teeth_mask
                talking_dict['hair_mask'] = (mask[:, :, 0] < 1) * (mask[:, :, 1] < 1) * (mask[:, :, 2] < 1)
                talking_dict['mouth_mask'] = (mask[:, :, 0] == 100) * (mask[:, :, 1] == 100) * (mask[:, :, 2] == 100) + teeth_mask

            
            if audio_file == '':
                talking_dict['auds'] = get_audio_features(auds, 2, frame['img_id'])
                if frame['img_id'] > auds.shape[0]:
                    print("[warnining] audio feature is too short")
                    break
            else:
                talking_dict['auds'] = get_audio_features(auds, 2, idx)
                if idx >= auds.shape[0]:
                    break


            talking_dict['blink'] = torch.as_tensor(np.clip(au_blink[frame['img_id']], 0, 2) / 2)
            talking_dict['au25'] = [au25[frame['img_id']], au25_25, au25_50, au25_75, au25_100]

            talking_dict['au_exp'] = torch.as_tensor(au_exp[frame['img_id']])


            [xmin, xmax, ymin, ymax] = ldmks_lips[idx].tolist()
            # padding to H == W
            cx = (xmin + xmax) // 2
            cy = (ymin + ymax) // 2

            l = max(xmax - xmin, ymax - ymin) // 2
            xmin = cx - l
            xmax = cx + l
            ymin = cy - l
            ymax = cy + l

            talking_dict['lips_rect'] = [xmin, xmax, ymin, ymax]
            talking_dict['lhalf_rect'] = ldmks_lhalf[idx]
            talking_dict['mouth_bound'] = [mouth_lb, mouth_ub, ldmks_mouth[idx, 1] - ldmks_mouth[idx, 0]]
            talking_dict['img_id'] = frame['img_id']
            
            # 사피엔스 prior를 사용하는 것이 다른 점
            if "train" in transformsfile and N_views > 0:
                # transformsfile에 "train"이 포함되어 있고 N_views가 0보다 크면 실행한다.
                normal_path_candidates = glob.glob(os.path.join(path, 'sapiens/normal/sapiens_*'))
                # path/sapiens/normal/ 디렉토리에서 'sapiens_'로 시작하는 모든 폴더(또는 파일) 경로를 리스트로 가져온다.
                normal_path_candidates.sort(reverse = True)
                # normal_path_candidates 리스트를 내림차순으로 정렬한다.
                normal_path = os.path.join(normal_path_candidates[0], str(frame['img_id']) + '.npy')
                # 가장 앞에 있는(내림차순 정렬된) 폴더에 현재 프레임의 img_id에 해당하는 .npy 파일 경로를 만든다.
                talking_dict['normal_path'] = normal_path
                # normal_path를 talking_dict에 저장한다.
                
                depth_path_candidates = glob.glob(os.path.join(path, 'sapiens/depth/sapiens_*'))
                # path/sapiens/depth/ 디렉토리에서 'sapiens_'로 시작하는 모든 폴더(또는 파일) 경로를 리스트로 가져온다.
                depth_path_candidates.sort(reverse = True)
                # depth_path_candidates 리스트를 내림차순으로 정렬한다.
                depth_path = os.path.join(depth_path_candidates[0], str(frame['img_id']) + '.npy')
                # 가장 앞에 있는(내림차순 정렬된) 폴더에 현재 프레임의 img_id에 해당하는 .npy 파일 경로를 만든다.
                talking_dict['depth_path'] = depth_path
                # depth_path를 talking_dict에 저장한다.
                if preload:
                    # preload가 True일 때만 아래를 실행한다.
                    normal = np.load(normal_path)
                    # normal_path에서 normal 맵(.npy 파일)을 numpy 배열로 불러온다.
                    talking_dict['normal'] = torch.as_tensor(normal).permute(2, 0, 1)
                    # normal 배열을 torch tensor로 변환하고, (H, W, C) -> (C, H, W)로 차원 순서를 바꾼 뒤 talking_dict에 저장한다.
                    depth = np.load(depth_path)
                    # depth_path에서 depth 맵(.npy 파일)을 numpy 배열로 불러온다.
                    talking_dict['depth'] = torch.as_tensor(depth)
                    # depth 배열을 torch tensor로 변환해서 talking_dict에 저장한다.
            
            FovX = focal2fov(focal_len, w)
            FovY = focal2fov(focal_len, h)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=w, height=h, background=bg, talking_dict=talking_dict))
            
            # if idx > 6500: break
            
    return cam_infos

def readMultiViewCamerasFromTransforms(parent_path, transformsfile, white_background, extension=".jpg", audio_file='', audio_extractor='deepspeech', N_views=-1, preload=True):
    # 멀티뷰 데이터를 읽는 함수
    # parent_path: 상위 폴더 경로 (예: './data/MEAD/027')
    # 각 뷰 폴더(front, left_30 등)에서 transforms_train.json을 읽어서 img_id별로 그룹화
    
    # 상위 폴더에서 모든 뷰 폴더 탐색
    view_folders = []
    if os.path.isdir(parent_path):
        for item in os.listdir(parent_path):
            item_path = os.path.join(parent_path, item) # '/home/white/github/InsTaG/data/MEAD/027/left_30'
            if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, transformsfile)):
                view_folders.append(item)
    
    if not view_folders:
        raise ValueError(f"No view folders found in {parent_path} with {transformsfile}")
    
    # front 뷰를 기준 뷰로 사용
    if 'front' not in view_folders:
        raise ValueError(f"'front' view folder not found in {parent_path}")
    
    view_folders.sort()  # 일관된 순서를 위해 정렬
    # front를 맨 앞으로 이동
    if 'front' in view_folders:
        view_folders.remove('front')
        view_folders.insert(0, 'front')
    
    print(f"Found {len(view_folders)} view folders: {view_folders}")
    
    # 각 뷰의 카메라 정보를 읽어서 img_id별로 그룹화
    multiview_cameras = {}  # {img_id: {view_name: CameraInfo, ...}, ...}
    
    # 먼저 front 뷰를 읽어서 img_id 목록을 얻음
    front_path = os.path.join(parent_path, 'front')
    front_cam_infos = readCamerasFromTransforms(front_path, transformsfile, white_background, extension, audio_file, audio_extractor, N_views, preload)
    
    # front 뷰의 img_id를 기준으로 다른 뷰들도 읽음
    front_img_ids = {cam.talking_dict['img_id'] for cam in front_cam_infos}
    
    for view_name in view_folders:
        view_path = os.path.join(parent_path, view_name)
        print(f"Reading {view_name} view from {view_path}")
        
        try:
            view_cam_infos = readCamerasFromTransforms(view_path, transformsfile, white_background, extension, audio_file, audio_extractor, N_views, preload)
            
            # img_id별로 그룹화
            for cam_info in view_cam_infos:
                img_id = cam_info.talking_dict['img_id']
                
                # front 뷰에 있는 img_id만 사용 (다른 뷰에만 있는 프레임은 제외)
                if img_id in front_img_ids:
                    if img_id not in multiview_cameras:
                        multiview_cameras[img_id] = {}
                    multiview_cameras[img_id][view_name] = cam_info
        except Exception as e:
            print(f"Warning: Failed to read {view_name} view: {e}")
            continue
    
    # front 뷰에 있는 모든 img_id가 모든 뷰에 있는지 확인하고, 없는 경우 경고
    for img_id in front_img_ids:
        if img_id not in multiview_cameras:
            print(f"Warning: img_id {img_id} not found in any view")
        else:
            missing_views = set(view_folders) - set(multiview_cameras[img_id].keys())
            if missing_views:
                print(f"Warning: img_id {img_id} missing views: {missing_views}")
    
    # 리스트 형태로 변환 (하위 호환성을 위해)
    # 각 CameraInfo에 view_name 정보를 talking_dict에 추가
    cam_infos_list = []
    for img_id in sorted(multiview_cameras.keys()):
        for view_name, cam_info in multiview_cameras[img_id].items():
            # talking_dict에 view_name 추가 (원본 수정 방지를 위해 복사)
            talking_dict = cam_info.talking_dict.copy()
            talking_dict['view_name'] = view_name
            talking_dict['is_multiview'] = True
            
            # CameraInfo는 NamedTuple이므로 새로 생성해야 함
            new_cam_info = CameraInfo(
                uid=cam_info.uid,
                R=cam_info.R,
                T=cam_info.T,
                FovY=cam_info.FovY,
                FovX=cam_info.FovX,
                image=cam_info.image,
                image_path=cam_info.image_path,
                image_name=cam_info.image_name,
                width=cam_info.width,
                height=cam_info.height,
                background=cam_info.background,
                talking_dict=talking_dict
            )
            cam_infos_list.append(new_cam_info)
    
    # 멀티뷰 구조도 함께 반환하기 위해 별도 속성으로 저장
    # 하지만 SceneInfo는 리스트만 받으므로, 나중에 Scene 클래스에서 처리
    return cam_infos_list, multiview_cameras

def readNerfSyntheticInfo(path, white_background, eval, extension=".jpg", args=None, preload=True, all_for_train=False):
    audio_file = args.audio # ''
    audio_extractor = args.audio_extractor # 'deepspeech'
    
    # MEAD 데이터셋인지 확인 (path에 'MEAD'가 포함되어 있고, 상위 폴더에 여러 뷰 폴더가 있는 경우)
    is_multiview = 'MEAD' in path and os.path.isdir(path)
    multiview_data = None
    
    if is_multiview:
        # 상위 폴더에서 뷰 폴더 확인
        view_folders = [item for item in os.listdir(path) if os.path.isdir(os.path.join(path, item)) and os.path.exists(os.path.join(path, item, "transforms_train.json"))]
        is_multiview = len(view_folders) > 1 and 'front' in view_folders
    
    if not eval: # not False -> True
        print("Reading Training Transforms") # 학습 카메라 정보를 읽는다는 메시지를 출력한다.
        if is_multiview:
            print("Detected multiview MEAD dataset, reading all views...")
            train_cam_infos, multiview_data = readMultiViewCamerasFromTransforms(path, "transforms_train.json", white_background, extension, audio_file, audio_extractor, args.N_views, preload=preload)
        else:
            train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, audio_file, audio_extractor, args.N_views, preload=preload)
    print("Reading Test Transforms")
    if is_multiview:
        test_cam_infos, _ = readMultiViewCamerasFromTransforms(path, "transforms_val.json", white_background, extension, audio_file, audio_extractor, args.N_views, preload=preload)
    else:
        test_cam_infos = readCamerasFromTransforms(path, "transforms_val.json", white_background, extension, audio_file, audio_extractor, args.N_views, preload=preload)
    
    if all_for_train: # False
        train_cam_infos.extend(test_cam_infos)
        # test_cam_infos = []
    
    
    if eval: train_cam_infos = test_cam_infos # for getNerfppNorm

    nerf_normalization = getNerfppNorm(train_cam_infos) 


    ply_path = os.path.join(path, "points3d.ply")
    mesh_path = os.path.join(path, "track_params.pt")      
    if not os.path.exists(ply_path) or True:
        # Since this data set has no colmap data, we start with random points
        num_pts = args.init_num
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 0.2 - 0.1
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        
        # facial_mesh = torch.load(mesh_path)["vertices"]
        # average_facial_mesh = torch.mean(facial_mesh, dim=0)
        # xyz = average_facial_mesh.cpu().numpy() * 0.1
        # shs = np.random.random((xyz.shape[0], 3)) / 255.0
        # pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((xyz.shape[0], 3)))        

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    
    # 멀티뷰 데이터가 있으면 함께 반환
    if is_multiview and multiview_data is not None:
        return scene_info, multiview_data
    else:
        return scene_info, None

sceneLoadTypeCallbacks = {
    "Colmap": None,
    "Blender" : readNerfSyntheticInfo
}
