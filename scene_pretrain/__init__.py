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
from scene_pretrain.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.motion_net import MotionNetwork, MouthMotionNetwork, PersonalizedMotionNetwork
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel
    gaussians_2 : GaussianModel # pretrain_mouth.py 코드에서 입술 가우시안을 학습시킬 때, 이전에 학습시켰던 face 가우시안을 불러올 변수.
    p_motion_grid = None

    # _dataset: {'sh_degree': 2, 'source_path': '/home/white/github/InsTaG/data/pretrain/macron', 
    # 'model_path': 'output/debug_01/macron', 'images': 'images', 'resolution': -1, 'white_background': False, 
    # 'data_device': 'cpu', 'eval': False, 'audio': '', 'init_num': 2000, 'N_views': -1, 'audio_extractor': 'deepspeech', 
    # 'type': 'face', 'preload': True, 'all_for_train': False}

    # gaussians: {'args': <arguments.GroupParams object at 0x7faf9ec67c10>, 'active_sh_degree': 0, 'max_sh_degree': 2, '_xyz': tensor([]), 
    # '_features_dc': tensor([]), '_features_rest': tensor([]), '_identity': tensor([]), '_scaling': tensor([]), '_rotation': tensor([]), 
    # '_opacity': tensor([]), 'max_radii2D': tensor([]), 'xyz_gradient_accum': tensor([]), 'denom': tensor([]), 'optimizer': None, 'percent_dense': 0, 
    # 'spatial_lr_scale': 0, 'neural_renderer': None, 'neural_motion_grid': None, 'scaling_activation': <built-in function softplus>, 
    # 'scaling_inverse_activation': <function GaussianModel.setup_functions.<locals>.<lambda> at 0x7fb0ea3d03a0>, 
    # 'covariance_activation': <function GaussianModel.setup_functions.<locals>.build_covariance_from_scaling_rotation at 0x7fb0ea3d0820>, 
    # 'opacity_activation': <built-in method sigmoid of type object at 0x7fb0a4127140>, 'inverse_opacity_activation': <function inverse_sigmoid at 0x7fb0e82e95e0>, 
    # 'rotation_activation': <function normalize at 0x7fb0462f55e0>}

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
     # Scene 클래스의 생성자입니다.
     # 'args': 모델 파라미터(예: dataset 객체)를 담고 있는 ModelParams 객체입니다. train_face.py 코드에서 _dataset이 여기에 해당됩니다.
     # 'gaussians': 이전에 생성된 GaussianModel 인스턴스입니다. 사용자님의 gaussians 객체가 여기에 해당됩니다.
     # 'load_iteration': 학습된 모델을 불러올 특정 반복(iteration) 횟수입니다. 기본값은 None입니다.
     # 'shuffle': 카메라 순서를 섞을지 여부를 결정합니다. 기본값은 True입니다.
     # 'resolution_scales': 렌더링 해상도 스케일 목록입니다. 기본값은 [1.0]입니다.
     
        """b
        :param path: Path to colmap scene main folder.
        """
        # :param path: Colmap 씬의 메인 폴더 경로를 의미합니다.
        self.model_path = args.model_path # 'debug_01/macron'
        self.loaded_iter = None # 불러온 모델의 반복 횟수(`loaded_iter`)를 초기에는 None으로 설정. 이는 기존 모델을 로드할 때 업데이트
        self.gaussians = gaussians
        # 전달받은 `gaussians` 객체를 현재 Scene 인스턴스의 `gaussians` 속성에 저장합니다.
        # 이제 Scene 객체는 이 `GaussianModel` 인스턴스를 사용하여 3D 씬을 표현하고 조작합니다.

        # 해당 안 됨
        if load_iteration:
            # `load_iteration`가 None이 아니라면 이 블록을 실행합니다.
            if load_iteration == -1:
                # `load_iteration`이 -1인 경우,
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
                # `searchForMaxIteration` 함수를 사용하여 `model_path/point_cloud` 디렉토리에서 가장 최근에 저장된 모델의 반복 횟수를 찾습니다.
                # 이는 일반적으로 가장 최신 학습 상태를 로드할 때 사용됩니다.
            else:
                # `load_iteration`이 -1이 아닌 다른 값인 경우,
                self.loaded_iter = load_iteration
                # 지정된 `load_iteration` 값을 `loaded_iter`에 저장합니다.
            print("Loading trained model at iteration {}".format(self.loaded_iter))
            # 어떤 반복 횟수의 학습된 모델을 불러오는지 출력합니다.

        self.train_cameras = {}
        # 학습용 카메라 정보를 저장할 딕셔너리 `train_cameras`를 초기화합니다.
        # 해상도 스케일을 키로 사용합니다.
        self.test_cameras = {}
        # 테스트용 카메라 정보를 저장할 딕셔너리 `test_cameras`를 초기화합니다.
        # 해상도 스케일을 키로 사용합니다.

        if os.path.exists(os.path.join(args.source_path, "transforms_train.json")): # '/home/white/github/InsTaG/data/pretrain/macron/transforms_train.json'가 존재한다면
            # `args.source_path` 디렉토리에 "transforms_train.json" 파일이 존재하는지 확인합니다.
            # 이 파일은 Blender 데이터셋에서 사용되는 일반적인 구성 파일입니다.
            print("Found transforms_train.json file, assuming Blender data set!")
            # 파일이 발견되면 Blender 데이터셋으로 가정하고 메시지를 출력합니다.
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, args=args, preload=args.preload)
            # `sceneLoadTypeCallbacks` 딕셔너리에서 "Blender" 타입에 해당하는 콜백 함수를 호출하여 씬 정보를 로드합니다.
            # 이 함수는 point_cloud, train_cameras, test_cameras, nerf_normalization, ply_path 정보를 반환합니다.
            
        else:
            # "transforms_train.json" 파일이 없는 경우,
            assert False, "Could not recognize scene type!"
            # 씬 타입을 인식할 수 없다는 오류를 발생시킵니다.

        # print("scene_info:", vars(scene_info))

        # 해당 됨
        if not self.loaded_iter:
            # 만약 이전에 학습된 모델을 불러오지 않는다면 (즉, `loaded_iter`가 None이라면) 이 블록을 실행합니다.
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                # `scene_info`에서 가져온 .ply 파일 경로('/home/white/github/InsTaG/data/pretrain/macron/points3d.ply')에서 원본 포인트 클라우드 파일을 읽어와,
                # `model_path` 내의 "input.ply"로 복사합니다. 이는 원본 데이터를 보존하고 관리하기 위함입니다.
                dest_file.write(src_file.read())
                # 파일 내용을 읽어서 다른 파일에 씁니다.
            json_cams = []
            # JSON 형식으로 저장할 카메라 목록을 초기화합니다.
            camlist = []
            # `scene_info`에서 가져올 카메라 목록을 초기화합니다.
            if scene_info.test_cameras:
                # 테스트 카메라 정보가 있다면,
                camlist.extend(scene_info.test_cameras)
                # `camlist`에 추가합니다.
            if scene_info.train_cameras:
                # 학습 카메라 정보가 있다면,
                camlist.extend(scene_info.train_cameras)
                # `camlist`에 추가합니다.
            for id, cam in enumerate(camlist): # camlist[0]으로 확인해볼 것
                # `camlist`에 있는 각 카메라에 대해 반복합니다.
                # cam = CameraInfo 객체
                json_cams.append(camera_to_JSON(id, cam))
                # `camera_to_JSON` 함수를 사용하여 카메라 정보를 JSON 형식으로 변환하여 `json_cams`에 추가합니다.
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                # `model_path` 내에 "cameras.json" 파일을 생성하고,
                json.dump(json_cams, file)
                # 변환된 카메라 정보를 JSON 형식으로 파일에 저장합니다.
                # {'id': 0, 'img_name': '8431', 'width': 512, 'height': 512, 'position': [...], 'rotation': [...], 'fy': 1200.0, 'fx': 1200.0} 값이 프레임별로 추가됨

        if shuffle:
            # `shuffle` 매개변수가 True인 경우,
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            # 학습용 카메라 목록의 순서를 무작위로 섞습니다. 이는 학습 시 데이터 다양성을 확보하는 데 도움이 됩니다.
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
            # 테스트용 카메라 목록의 순서를 무작위로 섞습니다.

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        # `scene_info`에서 `nerf_normalization`의 "radius" 값을 가져와 `cameras_extent`에 저장합니다.
        # 이는 씬의 공간적 범위를 나타내는 값으로, 가우시안 모델 초기화에 사용됩니다.

        for resolution_scale in resolution_scales:
            # `resolution_scales` 리스트에 포함된 각 해상도 스케일에 대해 반복합니다.
            # 예시: resolution_scales = [1.0]이므로 한 번만 반복합니다.
            print("Loading Training Cameras")
            # 학습용 카메라를 불러온다는 메시지를 출력합니다.
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            # `scene_info.train_cameras`와 현재 `resolution_scale`, 그리고 `args`를 인자로 하여
            # cameraList_from_camInfos 함수를 호출합니다.
            # 반환된 학습용 카메라 리스트를 self.train_cameras 딕셔너리에 저장합니다.

            print("Loading Test Cameras")
            # 테스트용 카메라를 불러온다는 메시지를 출력합니다.
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            # `scene_info.test_cameras`와 현재 `resolution_scale`, 그리고 `args`를 인자로 하여
            # cameraList_from_camInfos 함수를 호출합니다.
            # 반환된 테스트용 카메라 리스트를 self.test_cameras 딕셔너리에 저장합니다.

        if self.loaded_iter: # 디폴트 None
            # 만약 self.loaded_iter가 None이 아니면(즉, 학습된 모델을 불러오는 경우),
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            # self.model_path/point_cloud/iteration_{self.loaded_iter}/point_cloud.ply 경로의 ply 파일을 불러옵니다.
            # 이 파일에는 가우시안 모델의 파라미터(위치, 색상, 스케일 등)가 저장되어 있습니다.
        else:
            # self.loaded_iter가 None이면(즉, 새로 학습을 시작하는 경우),
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            # scene_info.point_cloud(초기 포인트 클라우드)와 self.cameras_extent(카메라 공간 범위)를 사용하여
            # 가우시안 모델을 초기화합니다.
            # 이 과정에서 3D 포인트 클라우드로부터 가우시안 파라미터들이 생성됩니다.

        gc.collect()
        # 가비지 컬렉션을 수행하여 불필요한 메모리를 해제합니다.
        

    def save(self, iteration):
        # 현재 가우시안 모델의 파라미터를 저장하는 함수입니다.
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        # 저장할 디렉토리 경로를 생성합니다.
        # 예시: self.model_path/point_cloud/iteration_10000
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        # 해당 경로에 point_cloud.ply 파일로 가우시안 모델을 저장합니다.

    def getTrainCameras(self, scale=1.0):
        # 주어진 scale(기본값 1.0)에 해당하는 학습용 카메라 리스트를 반환합니다.
        return self.train_cameras[scale]
        # 예시: self.train_cameras[1.0]을 반환합니다.

    def getTestCameras(self, scale=1.0):
        # 주어진 scale(기본값 1.0)에 해당하는 테스트용 카메라 리스트를 반환합니다.
        return self.test_cameras[scale]
        # 예시: self.test_cameras[1.0]을 반환합니다.