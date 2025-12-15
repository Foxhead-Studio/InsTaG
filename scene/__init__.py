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

import gc
import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.motion_net import MotionNetwork, MouthMotionNetwork
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path # 'output/Lieu_few_frames' (모델 결과가 저장될 경로), 'output/MEAD/027_multiview'
        self.loaded_iter = None
        self.gaussians = gaussians # 인자로 받아온, Lieu에 대한 단순 초기화 된 가우시안

        if load_iteration: # None
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.multiview_data = None  # 멀티뷰 데이터 저장: {img_id: {view_name: CameraInfo, ...}, ...}
        self.camera_index_map = {}  # 멀티뷰용 인덱스 맵: {(img_id, view_name): camera_index, ...}

        # 멀티뷰 데이터셋인지 확인 (상위 폴더에 transforms_train.json이 없고, 하위 폴더에 여러 뷰가 있는 경우)
        is_multiview = False
        if not os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            # 상위 폴더에 transforms_train.json이 없으면 멀티뷰 가능성 확인
            if os.path.isdir(args.source_path):
                view_folders = [item for item in os.listdir(args.source_path) 
                              if os.path.isdir(os.path.join(args.source_path, item)) 
                              and os.path.exists(os.path.join(args.source_path, item, "transforms_train.json"))]
                is_multiview = len(view_folders) > 1 and 'front' in view_folders
        
        if os.path.exists(os.path.join(args.source_path, "transforms_train.json")) or is_multiview:
            if is_multiview:
                print("Detected multiview dataset structure, assuming Blender data set!")
            else:
                print("Found transforms_train.json file, assuming Blender data set!")
            result = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, args=args, preload=args.preload, all_for_train=args.all_for_train)
            # source path = '/home/white/github/InsTaG/data/Lieu' white_background = False, eval = False, all_for_train = False
            # 멀티뷰 데이터가 있으면 튜플로 반환됨
            if isinstance(result, tuple):
                scene_info, self.multiview_data = result
            else:
                scene_info = result
                self.multiview_data = None
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            
            # 멀티뷰인 경우 인덱스 맵 생성 (shuffle 후에도 빠른 검색을 위해)
            if self.multiview_data is not None:
                self.camera_index_map[resolution_scale] = {}
                for idx, cam in enumerate(self.train_cameras[resolution_scale]):
                    img_id = cam.talking_dict.get('img_id')
                    view_name = cam.talking_dict.get('view_name')
                    if img_id is not None and view_name is not None:
                        self.camera_index_map[resolution_scale][(img_id, view_name)] = idx

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

        gc.collect()
        

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def getMultiViewFrame(self, img_id, scale=1.0):
        # 특정 img_id에 해당하는 모든 뷰의 Camera 객체를 반환
        # 반환: {view_name: Camera, ...} 형태의 딕셔너리
        multiview_cameras = {}
        
        if self.multiview_data is None:
            # 멀티뷰가 아닌 경우, 해당 img_id의 첫 번째 카메라만 반환
            for cam in self.train_cameras[scale]:
                if cam.talking_dict.get('img_id') == img_id:
                    multiview_cameras['single_view'] = cam
                    break
        else:
            # 멀티뷰인 경우, 해당 img_id의 모든 뷰를 찾아서 반환
            if img_id in self.multiview_data:
                # 인덱스 맵이 있으면 빠르게 찾기, 없으면 순차 검색
                if scale in self.camera_index_map:
                    for view_name in self.multiview_data[img_id].keys():
                        key = (img_id, view_name)
                        if key in self.camera_index_map[scale]:
                            idx = self.camera_index_map[scale][key]
                            multiview_cameras[view_name] = self.train_cameras[scale][idx]
                else:
                    # 인덱스 맵이 없는 경우 순차 검색 (하위 호환성)
                    for view_name in self.multiview_data[img_id].keys():
                        for cam in self.train_cameras[scale]:
                            if (cam.talking_dict.get('img_id') == img_id and 
                                cam.talking_dict.get('view_name') == view_name):
                                multiview_cameras[view_name] = cam
                                break
        
        return multiview_cameras
    