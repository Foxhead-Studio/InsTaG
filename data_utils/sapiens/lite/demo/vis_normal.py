# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing as mp
import os
import tempfile
import time
from argparse import ArgumentParser
from functools import partial
from multiprocessing import cpu_count, Pool, Process

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from adhoc_image_dataset import AdhocImageDataset
from tqdm import tqdm

from worker_pool import WorkerPool

torchvision.disable_beta_transforms_warning()

timings = {}
BATCH_SIZE = 32


def warmup_model(model, batch_size):
    # Warm up the model with a dummy input.
    imgs = torch.randn(batch_size, 3, 1024, 768).to(dtype=torch.bfloat16).cuda()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        for i in range(3):
            model(imgs)
    torch.cuda.current_stream().wait_stream(s)
    imgs = imgs.detach().cpu().float().numpy()
    del imgs, s


def inference_model(model, imgs, dtype=torch.bfloat16):
    with torch.no_grad():
        results = model(imgs.to(dtype).cuda())
        imgs.cpu()

    results = [r.cpu() for r in results]

    return results


def fake_pad_images_to_batchsize(imgs):
    return F.pad(imgs, (0, 0, 0, 0, 0, 0, 0, BATCH_SIZE - imgs.shape[0]), value=0)


def img_save_and_viz(image, result, output_path, seg_dir):
    seg_dir = None
    
    output_file = (
        output_path.replace(".jpg", ".png")
        .replace(".jpeg", ".png")
        .replace(".png", ".npy")
    )

    seg_logits = F.interpolate(
        result.unsqueeze(0), size=image.shape[:2], mode="bilinear"
    ).squeeze(0)
    normal_map = seg_logits.float().data.numpy().transpose(1, 2, 0)  ## H x W. seg ids.
    if seg_dir is not None:
        mask_path = os.path.join(
            seg_dir,
            os.path.basename(output_path)
            .replace(".png", ".npy")
            .replace(".jpg", ".npy")
            .replace(".jpeg", ".npy"),
        )
        mask = np.load(mask_path)
    else:
        mask = np.ones_like(normal_map)
        mask = mask==1
    normal_map_norm = np.linalg.norm(normal_map, axis=-1, keepdims=True)
    normal_map_normalized = normal_map / (normal_map_norm + 1e-5)  # Add a small e
    np.save(output_file, normal_map_normalized)

    normal_map_normalized[mask == 0] = -1  ## visualize background (nan) as black
    normal_map = ((normal_map_normalized + 1) / 2 * 255).astype(np.uint8)
    normal_map = normal_map[:, :, ::-1]

    vis_image = np.concatenate([image, normal_map], axis=1)
    cv2.imwrite(output_path, vis_image)

def load_model(checkpoint, use_torchscript=False):
    if use_torchscript:
        return torch.jit.load(checkpoint)
    else:
        return torch.export.load(checkpoint).module()

def main():
    parser = ArgumentParser()
    parser.add_argument("checkpoint", help="Checkpoint file")
    parser.add_argument("--input", help="Input image dir")
    parser.add_argument(
        "--output_root", "--output-root", default=None, help="Path to output dir"
    )
    parser.add_argument("--seg_dir", default=None, help="Path to seg dir")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument(
        "--batch_size",
        "--batch-size",
        type=int,
        default=32,
        help="Set batch size to do batch inference. ",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        default=[1024, 768],
        help="input image size (height, width)",
    )
    parser.add_argument(
        "--fp16", action="store_true", default=False, help="Model inference dtype"
    )
    args = parser.parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3,) + tuple(args.shape)
    else:
        raise ValueError("invalid input shape")

    mp.log_to_stderr()
    torch._inductor.config.force_fuse_int_mm_with_mul = True
    torch._inductor.config.use_mixed_mm = True

    start = time.time()

    USE_TORCHSCRIPT = '_torchscript' in args.checkpoint

    # build the model from a checkpoint file
    exp_model = load_model(args.checkpoint, USE_TORCHSCRIPT)

    ## no precision conversion needed for torchscript. run at fp32
    if not USE_TORCHSCRIPT:
        dtype = torch.half if args.fp16 else torch.bfloat16
        exp_model.to(dtype)
        exp_model = torch.compile(exp_model, mode="max-autotune", fullgraph=True)
    else:
        dtype = torch.float32  # TorchScript models use float32
        exp_model = exp_model.to(args.device)

    input = args.input
    image_names = []

    # Check if the input is a directory or a text file
    if os.path.isdir(input):
        input_dir = input  # Set input_dir to the directory specified in input
        image_names = [
            image_name
            for image_name in sorted(os.listdir(input_dir))
            if image_name.endswith(".jpg")
            or image_name.endswith(".png")
            or image_name.endswith(".jpeg")
        ]
    elif os.path.isfile(input) and input.endswith(".txt"):
        # If the input is a text file, read the paths from it and set input_dir to the directory of the first image
        with open(input, "r") as file:
            image_paths = [line.strip() for line in file if line.strip()]
        image_names = [
            os.path.basename(path) for path in image_paths
        ]  # Extract base names for image processing
        input_dir = (
            os.path.dirname(image_paths[0]) if image_paths else ""
        )  # Use the directory of the first image path

    if not os.path.exists(args.output_root):
        os.makedirs(args.output_root)

    global BATCH_SIZE
    BATCH_SIZE = args.batch_size

    n_batches = (len(image_names) + args.batch_size - 1) // args.batch_size

    inference_dataset = AdhocImageDataset(
        [os.path.join(input_dir, img_name) for img_name in image_names],
        (input_shape[1], input_shape[2]),
        mean=[123.5, 116.5, 103.5],
        std=[58.5, 57.0, 57.5],
    )
    inference_dataloader = torch.utils.data.DataLoader(
        inference_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(min(args.batch_size, cpu_count()), 1),
    )
    total_results = []
    image_paths = []
    img_save_pool = WorkerPool(
        img_save_and_viz, processes=max(min(args.batch_size, cpu_count()), 1)
    )
    for batch_idx, (batch_image_name, batch_orig_imgs, batch_imgs) in tqdm(
        enumerate(inference_dataloader), total=len(inference_dataloader)
    ):
        valid_images_len = len(batch_imgs)
        batch_imgs = fake_pad_images_to_batchsize(batch_imgs)
        result = inference_model(exp_model, batch_imgs, dtype=dtype)

        args_list = [
            (
                i,
                r,
                os.path.join(args.output_root, os.path.basename(img_name)),
                args.seg_dir,
            )
            for i, r, img_name in zip(
                batch_orig_imgs[:valid_images_len],
                result[:valid_images_len],
                batch_image_name,
            )
        ]
        img_save_pool.run_async(args_list)

    img_save_pool.finish()

    total_time = time.time() - start
    fps = 1 / ((time.time() - start) / len(image_names))
    print(
        f"\033[92mTotal inference time: {total_time:.2f} seconds. FPS: {fps:.2f}\033[0m"
    )


if __name__ == "__main__":
    main()

"""
### `vis_normal.py` 코드 설명

`vis_normal.py` 스크립트는 이미지에서 **표면 노멀(surface normal)**을 직접 추정하고 시각화하는 파이썬 스크립트입니다.
이는 `vis_depth.py`와 유사하지만, 깊이 맵을 추정하는 대신 바로 표면 노멀을 예측하는 모델을 사용하며,
주된 목적은 각 픽셀의 표면 방향을 파악하고 이를 시각적으로 표현하는 것입니다.

1.  **모델 로드 및 최적화**:
    *   사전 학습된 딥러닝 모델을 불러옵니다.
    *   이 모델은 이미지로부터 각 픽셀의 3D 표면 노멀(surface normal)을 직접 예측하도록 훈련된 모델입니다.
    *   모델의 성능을 최대화하기 위해 워밍업(warm-up) 과정을 거치고, 혼합 정밀도(mixed precision) 설정을 통해 추론 속도와 효율성을 높입니다.

2.  **이미지 전처리**:
    *   입력 이미지를 불러와서 모델이 요구하는 특정 크기(기본값: 1024x768)로 조정하고, 채널 순서를 변경하며, 미리 정의된 평균 및 표준편차 값으로 정규화하여 모델 입력에 적합한 형태로 변환합니다.

3.  **표면 노멀 추정 (Inference)**:
    *   전처리된 이미지 배치에 대해 로드된 모델을 실행하여 표면 노멀 맵을 추정합니다.
    *   이 맵은 각 픽셀의 3D 노멀 벡터 정보를 담고 있습니다.
    *   이 과정은 GPU 가속을 활용하며, 기울기 계산 없이(no_grad) 빠른 추론을 위해 최적화됩니다.

4.  **표면 노멀 저장 및 시각화 (`img_save_and_viz` 함수 부분)**
    *   모델이 추정한 표면 노멀 맵은 원시 데이터 형태로 시각적으로 이해하기 어렵습니다.
        따라서 이 단계에서는 노멀 맵을 처리하고 시각적으로 표현하여 최종 결과물로 저장합니다.

    *   **노멀 맵 추출 및 크기 조정**:
        모델이 예측한 노멀 맵은 원본 이미지와 다른 크기일 수 있습니다.
        따라서 이 노멀 맵을 원본 이미지와 동일한 크기로 정확하게 맞춰주는 보간(interpolation) 과정을 거칩니다.
        이 과정을 통해 각 픽셀이 3D 노멀 벡터를 나타내는 2차원 데이터가 됩니다.

    *   **노멀 벡터 정규화**:
        추출된 노멀 맵의 각 픽셀은 3D 벡터(X, Y, Z 성분)를 가집니다.
        이 벡터들은 방향이 중요하므로, 각 벡터의 길이를 1로 만들어주는 정규화(normalization) 과정을 거칩니다.
        이 정규화된 노멀 맵 데이터는 `.npy` 확장자를 가진 파일로 저장되어, 나중에 추가 분석이나 다른 3D 처리 과정에서 활용될 수 있는 원시 노멀 정보를 제공합니다.

    *   **마스크 처리 (선택 사항)**:
        이 스크립트에서는 기본적으로 모든 픽셀을 처리하지만, 경우에 따라 특정 마스크(예: 사람 영역만 포함하는 마스크)가 있다면 해당 마스크를 불러와 적용할 수 있습니다.
        마스크 바깥 영역(배경)은 노멀 정보를 시각화할 때 특별한 값(-1)으로 처리하여 검은색으로 표시될 수 있도록 합니다.

    *   **노멀 맵 시각화 (컬러 매핑)**:
        정규화된 노멀 벡터의 각 성분(X, Y, Z)은 -1에서 1 사이의 값을 가집니다.
        이를 사람이 시각적으로 이해하기 쉬운 이미지로 변환하기 위해, 이 값들을 0에서 255 사이의 RGB 색상 값으로 매핑합니다.
        일반적으로 노멀 벡터의 X, Y, Z 성분이 각각 이미지의 R, G, B 채널에 대응되도록 변환하여 표면의 방향성을 색상으로 표현합니다.
        예를 들어, 표면이 카메라를 정면으로 바라보면 특정 색상이 나타나고, 왼쪽을 향하면 다른 색상이 나타나는 식입니다.

    *   **최종 이미지 합성 및 저장**:
        마지막으로, 원본 이미지와 시각화된 표면 노멀 맵 이미지를 가로로 합쳐 하나의 큰 이미지로 만듭니다.
        이 합성된 이미지는 지정된 출력 경로에 `.png` 또는 `.jpg`와 같은 표준 이미지 파일 형식으로 저장됩니다.
        이 최종 이미지를 통해 원본 이미지와 그에 해당하는 표면의 방향 정보를 한눈에 비교하고 분석할 수 있습니다.

---

이러한 과정을 통해 `vis_normal.py`는 이미지에서 표면의 3D 방향 정보를 효과적으로 추출하고, 이를 시각적으로 풍부한 형태로 제공하여 다양한 3D 컴퓨터 비전 응용 분야에 활용될 수 있도록 합니다.
"""