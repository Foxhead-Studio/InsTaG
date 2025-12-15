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
BATCH_SIZE = 18


def warmup_model(model, batch_size):
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


def load_and_preprocess(img, shape):
    orig_img = cv2.imread(img)
    img = cv2.resize(orig_img, (768, 1024), interpolation=cv2.INTER_LINEAR).transpose(
        2, 0, 1
    )
    img = torch.from_numpy(img)
    img = img[[2, 1, 0], ...].float()
    mean = torch.tensor([123.5, 116.5, 103.5]).view(-1, 1, 1)
    std = torch.tensor([58.5, 57.0, 57.5]).view(-1, 1, 1)
    img = (img - mean) / std
    return orig_img, img


def img_save_and_viz(image, result, output_path, seg_dir):
    seg_logits = F.interpolate(
        result.unsqueeze(0), size=image.shape[:2], mode="bilinear"
    ).squeeze(0)

    depth_map = seg_logits.data.float().numpy()[0]  ## H x W
    image_name = os.path.basename(output_path)

    mask_path = os.path.join(
        seg_dir,
        image_name.replace(".png", ".npy")
        .replace(".jpg", ".npy")
        .replace(".jpeg", ".npy"),
    )

    ##-----------save depth_map to disk---------------------
    save_path = (
        output_path.replace(".png", ".npy")
        .replace(".jpg", ".npy")
        .replace(".jpeg", ".npy")
    )
    np.save(save_path, depth_map)

    # mask = np.load(mask_path)
    mask = np.ones_like(depth_map)
    mask = mask == 1
    depth_map[~mask] = np.nan
    depth_foreground = depth_map[mask]  ## value in range [0, 1]
    processed_depth = np.full((mask.shape[0], mask.shape[1], 3), 100, dtype=np.uint8)

    if len(depth_foreground) > 0:
        min_val, max_val = np.min(depth_foreground), np.max(depth_foreground)
        depth_normalized_foreground = 1 - (
            (depth_foreground - min_val) / (1 - min_val)
        )  ## for visualization, foreground is 1 (white), background is 0 (black)
        depth_normalized_foreground[depth_normalized_foreground > 1] = 1
        depth_normalized_foreground = (depth_normalized_foreground * 255.0).astype(
            np.uint8
        )

        depth_colored_foreground = cv2.applyColorMap(
            depth_normalized_foreground, cv2.COLORMAP_INFERNO
        )
        depth_colored_foreground = depth_colored_foreground.reshape(-1, 3)
        processed_depth[mask] = depth_colored_foreground

    ##---------get surface normal from depth map---------------
    depth_normalized = np.full((mask.shape[0], mask.shape[1]), np.inf)
    depth_normalized[mask > 0] = 1 - (
        (depth_foreground - min_val) / (max_val - min_val)
    )

    kernel_size = 7
    grad_x = cv2.Sobel(
        depth_normalized.astype(np.float32),
        cv2.CV_32F,
        1,
        0,
        ksize=kernel_size,
    )
    grad_y = cv2.Sobel(
        depth_normalized.astype(np.float32),
        cv2.CV_32F,
        0,
        1,
        ksize=kernel_size,
    )
    z = np.full(grad_x.shape, -1)
    normals = np.dstack((-grad_x, -grad_y, z))

    # Normalize the normals
    normals_mag = np.linalg.norm(normals, axis=2, keepdims=True)

    ## background pixels are nan.
    with np.errstate(divide="ignore", invalid="ignore"):
        normals_normalized = normals / (
            normals_mag + 1e-5
        )  # Add a small epsilon to avoid division by zero

    # Convert normals to a 0-255 scale for visualization
    normals_normalized = np.nan_to_num(
        normals_normalized, nan=-1, posinf=-1, neginf=-1
    )  ## visualize background (nan) as black
    normal_from_depth = ((normals_normalized + 1) / 2 * 255).astype(np.uint8)

    ## RGB to BGR for cv2
    normal_from_depth = normal_from_depth[:, :, ::-1]

    vis_image = np.concatenate([image, processed_depth, normal_from_depth], axis=1)
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
        default=18,
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
`vis_depth.py` 코드는 이미지에서 **깊이 맵(depth map)**을 추정하고, 추정된 깊이 맵으로부터 **표면 노멀(surface normal)**을 계산하여 시각화하는 파이썬 스크립트입니다. 이는 주로 `sapiens` 프로젝트의 `lite/demo` 부분에 속하며, 이미지 분석을 통해 3D 정보를 얻는 데 사용됩니다.

주요 기능은 다음과 같습니다:

1.  **모델 로드 및 워밍업**:
    *   사전 학습된 딥러닝 모델(`checkpoint`로 지정된 파일)을 불러옵니다. 이 모델은 이미지로부터 깊이 정보를 추정하는 역할을 합니다.
    *   모델 성능 최적화를 위해 워밍업(`warmup_model`)을 수행하고, `fp16` 또는 `bfloat16` 같은 혼합 정밀도(`mixed precision`)를 활용하여 빠르게 추론할 수 있도록 설정합니다.

2.  **이미지 전처리**:
    *   입력 이미지를 불러와서 특정 크기(기본값: 1024x768)로 리사이즈하고, 채널 순서를 변경하며, 모델 입력에 맞게 정규화(`load_and_preprocess`)합니다.

3.  **깊이 추정 (Inference)**:
    *   전처리된 이미지 배치에 대해 로드된 모델을 사용하여 깊이 맵을 추정합니다 (`inference_model`).
    *   이 과정에서 `torch.no_grad()`와 `torch.autocast`를 사용하여 효율적인 추론을 수행합니다.

4.  **깊이 맵 저장 및 시각화 (`img_save_and_viz`)**:
    #### 1. 깊이 맵 데이터 추출 및 파일 저장
    먼저, 딥러닝 모델이 추정한 깊이 맵은 원본 이미지와 다른 크기일 수 있습니다.
    따라서 이 깊이 맵을 원본 이미지와 동일한 크기로 정확하게 맞춰주는 보간(interpolation) 과정을 거칩니다.
    이 과정을 통해 각 픽셀이 해당 위치의 깊이 값을 나타내는 2차원 데이터가 만들어집니다.
    이렇게 정제된 깊이 맵 데이터는 나중에 다른 분석이나 학습에 재활용될 수 있도록 `.npy` 형식의 파일로 저장됩니다.
    이 `.npy` 파일은 각 픽셀의 실제 깊이 값을 부동소수점 형태로 담고 있는 원시 데이터입니다.

    #### 2. 깊이 맵의 전경 추출 및 시각화
    저장된 깊이 맵은 시각적으로 바로 이해하기 어렵기 때문에 사람이 보기 쉬운 형태로 변환하는 과정이 필요합니다.
    *   **전경 마스크 적용**: 일반적으로 이미지에는 관심 객체(전경)와 그 외의 배경이 있습니다.
        이 스크립트에서는 모든 픽셀을 전경으로 간주하는 단순한 마스크를 적용하거나, 경우에 따라 특정 마스크를 사용하여 전경 영역만을 분리합니다.
        마스크 바깥 영역(배경)의 깊이 값은 시각화 시 제외하기 위해 특별한 값(`NaN`)으로 처리됩니다.
    *   **깊이 값 정규화 및 컬러맵 적용**: 전경에 해당하는 깊이 값들만 추출하여, 그 중 가장 작은 깊이 값(가장 가까운 거리)과 가장 큰 깊이 값(가장 먼 거리)을 기준으로 0부터 1 사이의 값으로 정규화합니다.
        이때 가까운 물체는 밝게, 먼 물체는 어둡게 보이도록 값을 뒤집어줍니다.
        이렇게 정규화된 깊이 값은 0-255 스케일로 변환된 후, `Inferno`와 같은 컬러맵(색상 지도)이 적용됩니다.
        컬러맵은 깊이의 변화를 다양한 색상으로 표현하여 깊이 차이를 시각적으로 더욱 명확하게 보여줍니다.
    *   **시각화된 깊이 맵 생성**: 컬러맵이 적용된 전경 깊이 맵을 초기 배경 이미지에 합성하여 최종 시각화용 깊이 맵 이미지를 완성합니다.

    #### 3. 깊이 맵에서 표면 노멀(Surface Normal) 계산
    깊이 맵 정보는 3D 객체의 형상을 파악하는 데 매우 유용하며, 여기서 한 단계 더 나아가 각 표면이 향하는 방향인 **표면 노멀(Surface Normal)**을 계산할 수 있습니다.
    *   **기울기 계산**: 깊이 맵의 각 픽셀에서 X 방향과 Y 방향으로 깊이가 얼마나 변하는지 계산합니다. 이를 위해 **Sobel 필터**와 같은 이미지 처리 기법을 사용하여 깊이 맵의 기울기(변화율)를 감지합니다.
        이 기울기 정보는 표면의 경사를 나타냅니다.
    *   **3D 노멀 벡터 형성**: 계산된 X, Y 방향 기울기 정보와 깊이 방향(Z축) 정보를 결합하여 각 픽셀에 대한 3D 노멀 벡터를 형성합니다.
        이 벡터는 해당 지점의 표면이 3차원 공간에서 어느 방향을 가리키는지 나타냅니다.
    *   **노멀 벡터 정규화**: 노멀 벡터는 방향만 중요하므로, 그 길이를 1로 만드는 정규화 과정을 거칩니다.
    *   **노멀 벡터 시각화**: 정규화된 노멀 벡터의 각 성분(-1에서 1 사이의 값)을 0에서 255 사이의 RGB 색상 값으로 변환합니다.
        이렇게 하면 노멀 벡터의 방향에 따라 고유한 색상이 부여되어 표면의 세부적인 경사와 질감을 시각적으로 표현할 수 있습니다.
        예를 들어, 특정 방향을 바라보는 표면은 파란색으로, 다른 방향은 빨간색으로 보이는 식입니다.

    #### 4. 최종 이미지 합성 및 저장
    마지막으로, 모든 처리 결과물을 하나로 합쳐서 최종 결과 이미지를 만듭니다.
    *   **이미지 합성**: 원본 이미지, 시각화된 깊이 맵 이미지, 시각화된 표면 노멀 이미지 이 세 가지를 가로로 나란히 배치하여 하나의 큰 이미지를 생성합니다.
    *   **최종 저장**: 이렇게 합성된 이미지를 지정된 경로에 `.png` 또는 `.jpg`와 같은 표준 이미지 파일 형식으로 저장합니다.
    이러러한 복합적인 과정을 통해, 모델이 추정한 깊이 정보는 저장될 뿐만 아니라, 사람이 직관적으로 이해할 수 있는 깊이 맵과 표면 노멀 시각화 이미지로 변환되어 제공됩니다.
    
5.  **병렬 처리**:
    *   여러 이미지 파일을 효율적으로 처리하기 위해 `WorkerPool`을 사용하여 이미지 저장 및 시각화 작업을 병렬로 실행합니다.

**요약하자면, 이 스크립트는 입력 이미지들을 딥러닝 모델로 처리하여 각 픽셀의 깊이 정보를 알아내고, 그 깊이 정보를 이용하여 물체의 표면 방향(노멀)까지 계산한 뒤, 이 모든 정보를 보기 쉬운 형태로 저장하고 시각화하는 기능을 담당합니다.**
"""