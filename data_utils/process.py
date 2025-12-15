import os
import glob
import tqdm
import json
import argparse
import cv2
import numpy as np

def extract_audio(path, out_path, sample_rate=16000):
    # 오디오 추출 시작 로그 출력 (예: [INFO] ===== extract audio from Macron.mp4 to aud.wav =====)
    print(f'[INFO] ===== extract audio from {path} to {out_path} =====')
    # ffmpeg는 다양한 포맷의 비디오/오디오를 변환, 추출, 편집할 수 있는 강력한 오픈소스 명령줄 툴입니다.
    # 여기서는 ffmpeg를 사용해 비디오 파일에서 오디오(wav)만 추출합니다.
    # ffmpeg 명령어 문자열 생성 (예: ffmpeg -i Macron.mp4 -f wav -ar 16000 aud.wav)
    cmd = f'ffmpeg -i {path} -f wav -ar {sample_rate} {out_path}'
    # 시스템 쉘에서 ffmpeg 명령 실행 (예: os.system(cmd))
    os.system(cmd)
    # 오디오 추출 완료 로그 출력 (예: [INFO] ===== extracted audio =====)
    print(f'[INFO] ===== extracted audio =====')


def extract_audio_features(path, mode='wav2vec'):
    # 오디오 특징 추출 시작 로그 출력 (예: [INFO] ===== extract audio labels for aud.wav =====)
    print(f'[INFO] ===== extract audio labels for {path} =====')
    if mode == 'wav2vec':
        # wav2vec 모드일 때 특징 추출 명령어 생성 (예: python nerf/asr.py --wav aud.wav --save_feats)
        cmd = f'python nerf/asr.py --wav {path} --save_feats'
    else: # deepspeech
        # deepspeech 모드일 때 특징 추출 명령어 생성 (예: python data_utils/deepspeech_features/extract_ds_features.py --input aud.wav)
        cmd = f'python data_utils/deepspeech_features/extract_ds_features.py --input {path}'
    # 시스템 쉘에서 특징 추출 명령 실행 (예: os.system(cmd))
    os.system(cmd)
    # 오디오 특징 추출 완료 로그 출력 (예: [INFO] ===== extracted audio labels =====)
    print(f'[INFO] ===== extracted audio labels =====')



def extract_images(path, out_path, fps=25):

    print(f'[INFO] ===== extract images from {path} to {out_path} =====')
    cmd = f'ffmpeg -i {path} -vf fps={fps} -qmin 1 -q:v 1 -start_number 0 {os.path.join(out_path, "%d.jpg")}'
    os.system(cmd)
    print(f'[INFO] ===== extracted images =====')


def extract_semantics(ori_imgs_dir, parsing_dir):

    print(f'[INFO] ===== extract semantics from {ori_imgs_dir} to {parsing_dir} =====')
    cmd = f'python data_utils/face_parsing/test.py --respath={parsing_dir} --imgpath={ori_imgs_dir}'
    os.system(cmd)
    print(f'[INFO] ===== extracted semantics =====')


def extract_landmarks(ori_imgs_dir):
    # ori_imgs_dir 디렉토리 내의 이미지들에서 얼굴 랜드마크를 추출하는 함수입니다.

    print(f'[INFO] ===== extract face landmarks from {ori_imgs_dir} =====')
    # 처리 시작 로그 출력

    import face_alignment  # face_alignment 라이브러리 import (얼굴 랜드마크 검출용)
    try:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
        # face_alignment 객체 생성 (LandmarksType._2D: 2D 랜드마크 타입)
    except:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
        # 일부 버전 호환을 위해 예외 발생 시 다른 타입으로 객체 생성
    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))
    # ori_imgs_dir 내의 모든 jpg 이미지 경로 리스트업
    for image_path in tqdm.tqdm(image_paths):
        # 이미지별로 진행상황을 tqdm으로 표시하며 반복
        input = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) # [H, W, 3]
        # 이미지를 읽어옴
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        # BGR에서 RGB로 변환 (face_alignment는 RGB 입력 필요)
        preds = fa.get_landmarks(input)
        # 얼굴 랜드마크 예측
        if len(preds) > 0:
            # 얼굴이 검출된 경우
            lands = preds[0].reshape(-1, 2)[:,:2]
            # 첫 번째 얼굴의 랜드마크 좌표 추출 및 2D로 reshape
            np.savetxt(image_path.replace('jpg', 'lms'), lands, '%f')
            # 랜드마크 좌표를 .lms 파일로 저장
    del fa
    # face_alignment 객체 메모리 해제
    print(f'[INFO] ===== extracted face landmarks =====')
    # 처리 완료 로그 출력


def extract_background(base_dir, ori_imgs_dir):
    # ori_imgs_dir 내의 이미지들을 이용해 배경 이미지를 추출하는 함수입니다.
    
    print(f'[INFO] ===== extract background image from {ori_imgs_dir} =====')
    # 처리 시작 로그 출력

    from sklearn.neighbors import NearestNeighbors
    # 최근접 이웃 계산을 위한 라이브러리 import

    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))
    # ori_imgs_dir 내의 모든 jpg 이미지 경로 리스트업
    # only use 1/20 image_paths 
    image_paths = image_paths[::20]
    # 전체 이미지 중 20장에 1장씩만 샘플링하여 사용 (속도 향상 목적)
    # read one image to get H/W
    tmp_image = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED) # [H, W, 3]
    # 첫 번째 이미지를 읽어와서
    h, w = tmp_image.shape[:2]
    # 이미지의 높이와 너비를 얻음

    # nearest neighbors
    all_xys = np.mgrid[0:h, 0:w].reshape(2, -1).transpose()
    # 전체 픽셀 좌표를 (N, 2) 형태로 생성
    distss = []
    for image_path in tqdm.tqdm(image_paths):
        # 샘플링된 이미지별로 반복
        parse_img = cv2.imread(image_path.replace('ori_imgs', 'parsing').replace('.jpg', '.png'))
        # 해당 이미지의 파싱(세멘틱 분할) 결과를 읽어옴
        bg = (parse_img[..., 0] == 255) & (parse_img[..., 1] == 255) & (parse_img[..., 2] == 255)
        # 배경 픽셀 마스크(흰색) 생성
        fg_xys = np.stack(np.nonzero(~bg)).transpose(1, 0)
        # 전경(배경이 아닌) 픽셀 좌표 추출
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(fg_xys)
        # 전경 픽셀 좌표로 최근접 이웃 모델 학습
        dists, _ = nbrs.kneighbors(all_xys)
        # 전체 픽셀에 대해 전경까지의 거리 계산
        distss.append(dists)
        # 거리 결과 저장

    distss = np.stack(distss)
    # (이미지 수, 전체 픽셀 수, 1) 형태로 거리 배열 생성
    max_dist = np.max(distss, 0)
    # 각 픽셀별로 최대 거리(가장 멀리 떨어진 전경까지의 거리) 계산
    max_id = np.argmax(distss, 0)
    # 각 픽셀별로 최대 거리를 주는 이미지 인덱스

    bc_pixs = max_dist > 5
    # 최대 거리가 5보다 큰(즉, 충분히 배경인) 픽셀 마스크
    bc_pixs_id = np.nonzero(bc_pixs)
    # 해당 픽셀들의 인덱스
    bc_ids = max_id[bc_pixs]
    # 해당 픽셀별로 최대 거리를 주는 이미지 인덱스

    imgs = []
    num_pixs = distss.shape[1]
    for image_path in image_paths:
        img = cv2.imread(image_path)
        imgs.append(img)
    # 샘플링된 모든 이미지를 읽어서 리스트에 저장
    imgs = np.stack(imgs).reshape(-1, num_pixs, 3)
    # (이미지 수, 전체 픽셀 수, 3) 형태로 변환

    bc_img = np.zeros((h*w, 3), dtype=np.uint8)
    # 배경 이미지를 1차원으로 초기화
    bc_img[bc_pixs_id, :] = imgs[bc_ids, bc_pixs_id, :]
    # 배경 픽셀 위치에 해당하는 이미지를 선택하여 값 할당
    bc_img = bc_img.reshape(h, w, 3)
    # 다시 (H, W, 3) 형태로 reshape

    max_dist = max_dist.reshape(h, w)
    # 최대 거리 맵을 (H, W)로 변환
    bc_pixs = max_dist > 5
    # 배경 픽셀 마스크 재생성
    bg_xys = np.stack(np.nonzero(~bc_pixs)).transpose()
    # 배경이 아닌(즉, 전경) 픽셀 좌표
    fg_xys = np.stack(np.nonzero(bc_pixs)).transpose()
    # 배경 픽셀 좌표
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(fg_xys)
    # 배경 픽셀 좌표로 최근접 이웃 모델 학습
    distances, indices = nbrs.kneighbors(bg_xys)
    # 전경 픽셀에 대해 최근접 배경 픽셀 찾기
    bg_fg_xys = fg_xys[indices[:, 0]]
    # 각 전경 픽셀에 대해 가장 가까운 배경 픽셀 좌표
    bc_img[bg_xys[:, 0], bg_xys[:, 1], :] = bc_img[bg_fg_xys[:, 0], bg_fg_xys[:, 1], :]
    # 전경 픽셀을 가장 가까운 배경 픽셀 값으로 채움 (hole-filling)

    cv2.imwrite(os.path.join(base_dir, 'bc.jpg'), bc_img)
    # 최종 배경 이미지를 bc.jpg로 저장

    print(f'[INFO] ===== extracted background image =====')
    # 처리 완료 로그 출력


# extract_torso_and_gt
# InsTaG가 얼굴과 얼굴을 제외한 상체를 따로 렌더링한 후 붙이기 때문에 아래와 같이 GT 이미지와 Torso 이미지를 따로 생성한다

# 1. GT (Ground Truth) 이미지 생성
# 입력: 원본 이미지 + 배경 이미지 + 얼굴 파싱 결과
# 처리: 원본 이미지에서 배경 부분만 추출된 배경 이미지로 교체
# 출력: gt_imgs/ 폴더에 저장되는 정답 이미지들
# 가우시안의 정답 이미지

# 2. 상체 (Torso) 이미지 생성
# 입력: 위에서 생성한 GT 이미지 + 얼굴 파싱 결과 + 배경 이미지
# 처리:
# 머리 부분을 배경으로 교체
# 몸통/목 부분을 자연스럽게 위로 인페인팅 (수직 방향으로 색상 확장)
# 알파 채널 추가 (투명도 정보)
# 출력: torso_imgs/ 폴더에 RGBA 형태로 저장

def extract_torso_and_gt(base_dir, ori_imgs_dir):

    print(f'[INFO] ===== extract torso and gt images for {base_dir} =====')

    from scipy.ndimage import binary_erosion, binary_dilation

    # 배경 이미지 불러오기 (예: base_dir='sample', 'sample/bc.jpg'에서 배경 이미지 로드)
    bg_image = cv2.imread(os.path.join(base_dir, 'bc.jpg'), cv2.IMREAD_UNCHANGED)
    
    # ori_imgs_dir 내의 모든 jpg 이미지 경로 리스트업 (예: ori_imgs_dir='sample/ori_imgs', 'sample/ori_imgs/0.jpg', 'sample/ori_imgs/1.jpg' 등)
    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))

    for image_path in tqdm.tqdm(image_paths):
        # 원본 이미지 읽기 (예: image_path='sample/ori_imgs/0.jpg')
        ori_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) # [H, W, 3]

        # 세맨틱(파싱) 이미지 읽기 (예: 'sample/parsing/0.png')
        seg = cv2.imread(image_path.replace('ori_imgs', 'parsing').replace('.jpg', '.png'))
        # head_part: 빨간색(255,0,0)인 부분 마스크 (예: 얼굴 부분)
        head_part = (seg[..., 0] == 255) & (seg[..., 1] == 0) & (seg[..., 2] == 0)
        # neck_part: 초록색(0,255,0)인 부분 마스크 (예: 목 부분)
        neck_part = (seg[..., 0] == 0) & (seg[..., 1] == 255) & (seg[..., 2] == 0)
        # torso_part: 파란색(0,0,255)인 부분 마스크 (예: 상체 부분)
        torso_part = (seg[..., 0] == 0) & (seg[..., 1] == 0) & (seg[..., 2] == 255)
        # bg_part: 흰색(255,255,255)인 부분 마스크 (예: 배경 부분)
        bg_part = (seg[..., 0] == 255) & (seg[..., 1] == 255) & (seg[..., 2] == 255)

        # gt 이미지 생성: 배경 부분만 배경 이미지로 교체
        # 예시: ori_image가 사람 얼굴, 목, 상체, 배경이 모두 포함된 원본 이미지이고,
        # bg_part는 흰색(255,255,255)인 배경 부분의 마스크임.
        # 예를 들어 ori_image[100, 200]이 배경이면, gt_image[100, 200] = bg_image[100, 200]로 바뀜.
        # 즉, 얼굴/목/상체는 원본 이미지 그대로 두고, 배경 부분만 bc.jpg(배경 이미지)로 덮어씌움.
        gt_image = ori_image.copy()
        gt_image[bg_part] = bg_image[bg_part]
        # gt 이미지 저장: 예를 들어 'sample/ori_imgs/0.jpg'가 입력이면, 'sample/gt_imgs/0.jpg'로 저장됨.
        cv2.imwrite(image_path.replace('ori_imgs', 'gt_imgs'), gt_image)

        # torso 이미지 생성: head 부분(빨간색, 얼굴)을 배경으로 교체
        # 예시: head_part가 True인 픽셀(예: 얼굴 부분)만 배경 이미지로 덮어씌움.
        # 즉, 얼굴 부분만 배경으로 바뀌고, 나머지(목, 상체, 배경)는 gt_image 그대로 유지됨.
        torso_image = gt_image.copy() # rgb
        torso_image[head_part] = bg_image[head_part]
        # torso 알파 채널 생성: 모든 픽셀의 알파값을 255로 설정 (shape: (H, W, 1))
        # 예시: 512x512 이미지라면 shape=(512,512,1), 모든 값이 255
        torso_alpha = 255 * np.ones((gt_image.shape[0], gt_image.shape[1], 1), dtype=np.uint8) # alpha
        
        # torso 부분 "수직" 인페인팅: 상체 경계 위로 색을 점점 어둡게 복사
        # 예시: 상체(torso_part)와 얼굴(head_part)이 맞닿은 경계에서, 상체의 윗부분 색을 위로 9픽셀(L=9)씩 복사하면서
        # 위로 갈수록 0.98, 0.98^2, ...씩 점점 어둡게 만듦. (자연스러운 경계 효과)
        L = 8 + 1
        # torso_part에서 True인 좌표 추출: 예를 들어 [[y1,x1],[y2,x2],...] 형태로 상체 픽셀 위치를 모두 모음
        torso_coords = np.stack(np.nonzero(torso_part), axis=-1) # [M, 2]
        # y(행) 기준, x(열) 기준으로 정렬: 같은 x(열)에서 y(행)가 작은 순서(즉, 위쪽부터)로 정렬
        inds = np.lexsort((torso_coords[:, 0], torso_coords[:, 1]))
        torso_coords = torso_coords[inds]
        # 각 열(x)별로 가장 위에 있는 torso 픽셀만 선택: 예를 들어 x=100에서 y가 가장 작은(윗부분) torso 픽셀만 남김
        u, uid, ucnt = np.unique(torso_coords[:, 1], return_index=True, return_counts=True)
        top_torso_coords = torso_coords[uid] # [m, 2]
        # 위쪽 한 칸(y-1)이 head_part인 픽셀만 남김: 즉, 상체 바로 위가 얼굴인 경계 픽셀만 선택
        # 예시: top_torso_coords가 [[50,100],[51,101]]이고, [49,100]이 head_part면 [50,100]만 남음
        top_torso_coords_up = top_torso_coords.copy() - np.array([1, 0])
        mask = head_part[tuple(top_torso_coords_up.T)] 
        if mask.any():
            # 조건에 맞는 top_torso_coords만 남김: 실제로 얼굴과 맞닿은 상체 경계만 남김
            top_torso_coords = top_torso_coords[mask]
            # 해당 위치의 색상 추출: 예를 들어 top_torso_coords=[50,100]이면 gt_image[50,100,:]의 색상 추출
            top_torso_colors = gt_image[tuple(top_torso_coords.T)] # [m, 3]
            # 인페인팅 좌표 생성: 위로 L픽셀씩 복사
            # 예시: [50,100]이면 [50,100], [49,100], ..., [42,100]까지 9개 좌표 생성
            inpaint_torso_coords = top_torso_coords[None].repeat(L, 0) # [L, m, 2]
            inpaint_offsets = np.stack([-np.arange(L), np.zeros(L, dtype=np.int32)], axis=-1)[:, None] # [L, 1, 2]
            inpaint_torso_coords += inpaint_offsets
            inpaint_torso_coords = inpaint_torso_coords.reshape(-1, 2) # [Lm, 2]
            # 색상도 L번 반복: 예를 들어 [r,g,b] 색상을 위로 9번 복사
            inpaint_torso_colors = top_torso_colors[None].repeat(L, 0) # [L, m, 3]
            # 위로 갈수록 점점 어둡게: 0.98, 0.98^2, ...씩 곱해서 자연스러운 그라데이션 효과
            darken_scaler = 0.98 ** np.arange(L).reshape(L, 1, 1) # [L, 1, 1]
            inpaint_torso_colors = (inpaint_torso_colors * darken_scaler).reshape(-1, 3) # [Lm, 3]
            # 인페인팅 좌표에 색상 할당: 예를 들어 [49,100]에 어두운 색을 할당
            torso_image[tuple(inpaint_torso_coords.T)] = inpaint_torso_colors

            # 인페인팅된 영역 마스크 생성: True/False로 표시
            # 예시: inpaint_torso_coords에 해당하는 위치만 True, 나머지는 False
            inpaint_torso_mask = np.zeros_like(torso_image[..., 0]).astype(bool)
            inpaint_torso_mask[tuple(inpaint_torso_coords.T)] = True
        else:
            inpaint_torso_mask = None
            

        # 목(neck) 부분 "수직" 인페인팅(위로 색상 복사) - 한 줄씩 예시와 함께 상세 주석

        push_down = 4  # 목 경계에서 아래로 4픽셀만큼 더 내려서 인페인팅 시작 (자연스러운 연결)
        L = 48 + push_down + 1  # 인페인팅 길이: 48+4+1=53픽셀. 예: 53픽셀 위로 색상 복사

        # 목 마스크를 수직 방향으로 3번 팽창시켜 경계가 더 두꺼워지게 함
        # 예시: [[0,1,0],[0,1,0],[0,1,0]] 구조로 위아래로만 확장
        neck_part = binary_dilation(
            neck_part,
            structure=np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool),
            iterations=3
        )

        # 목 부분에서 True인 좌표(y, x)만 추출. 예: [[y1,x1],[y2,x2],...]
        neck_coords = np.stack(np.nonzero(neck_part), axis=-1)  # [M, 2], M=neck 픽셀 개수

        # x(열) 기준으로 정렬, 같은 x에서 y(행)가 작은 순(즉, 위쪽부터)으로 정렬
        # 예: x=100에서 y=50,51,52... 순서로 정렬됨
        inds = np.lexsort((neck_coords[:, 0], neck_coords[:, 1]))
        neck_coords = neck_coords[inds]

        # 각 열(x)별로 가장 위에 있는(즉, y가 가장 작은) 목 픽셀만 선택
        # 예: x=100에서 y=50이 top, x=101에서 y=48이 top
        u, uid, ucnt = np.unique(neck_coords[:, 1], return_index=True, return_counts=True)
        top_neck_coords = neck_coords[uid]  # [m, 2], m=열 개수

        # top_neck_coords의 바로 위(y-1, 같은 x)가 head_part(얼굴)이면 True
        # 예: top_neck_coords=[50,100]이면 [49,100]이 head_part인지 확인
        top_neck_coords_up = top_neck_coords.copy() - np.array([1, 0])
        mask = head_part[tuple(top_neck_coords_up.T)]  # 얼굴과 맞닿은 목 경계만 True

        # 실제로 얼굴과 맞닿은 목 경계만 남김
        top_neck_coords = top_neck_coords[mask]

        # 각 열별로 목 경계에서 아래로 push_down(4)픽셀만큼 더 내려서 인페인팅 시작
        # ucnt: 각 열별 목 픽셀 개수, ucnt[mask]-1: 경계에서 아래로 몇 픽셀 더 갈 수 있는지
        # np.minimum(ucnt[mask]-1, push_down): 최대 push_down까지만 내려감
        offset_down = np.minimum(ucnt[mask] - 1, push_down)
        # 예: top_neck_coords=[50,100], offset_down=3이면 [53,100]로 이동
        top_neck_coords += np.stack([offset_down, np.zeros_like(offset_down)], axis=-1)

        # 인페인팅 시작점의 색상 추출. 예: top_neck_coords=[53,100]이면 gt_image[53,100,:]
        top_neck_colors = gt_image[tuple(top_neck_coords.T)]  # [m, 3]

        # 인페인팅 좌표 생성: 각 시작점에서 위로 L(53)픽셀씩 복사
        # 예: [53,100]이면 [53,100],[52,100],...,[1,100]까지 53개 좌표
        inpaint_neck_coords = top_neck_coords[None].repeat(L, 0)  # [L, m, 2]
        inpaint_offsets = np.stack(
            [-np.arange(L), np.zeros(L, dtype=np.int32)], axis=-1
        )[:, None]  # [L, 1, 2], 위로 -1씩 이동
        inpaint_neck_coords += inpaint_offsets  # [L, m, 2]
        inpaint_neck_coords = inpaint_neck_coords.reshape(-1, 2)  # [L*m, 2]

        # 색상도 L번 반복: 예를 들어 [r,g,b] 색상을 위로 53번 복사
        inpaint_neck_colors = top_neck_colors[None].repeat(L, 0)  # [L, m, 3]

        # 위로 갈수록 점점 어둡게: 0.98, 0.98^2, ...씩 곱해서 자연스러운 그라데이션 효과
        # 예: 0번째(아래)는 1.0, 1번째는 0.98, 2번째는 0.98^2, ...
        darken_scaler = 0.98 ** np.arange(L).reshape(L, 1, 1)  # [L, 1, 1]
        inpaint_neck_colors = (inpaint_neck_colors * darken_scaler).reshape(-1, 3)  # [L*m, 3]

        # 인페인팅 좌표에 색상 할당: 예를 들어 [52,100]에 어두운 색을 할당
        torso_image[tuple(inpaint_neck_coords.T)] = inpaint_neck_colors

        # 인페인팅된 영역 마스크 생성: True/False로 표시
        # 예: inpaint_neck_coords에 해당하는 위치만 True, 나머지는 False
        inpaint_mask = np.zeros_like(torso_image[..., 0]).astype(bool)
        inpaint_mask[tuple(inpaint_neck_coords.T)] = True

        # 인페인팅 영역에 블러 적용(수직 줄무늬 방지)
        # 예: 5x5 가우시안 블러로 경계가 자연스럽게 섞임
        blur_img = torso_image.copy()
        blur_img = cv2.GaussianBlur(blur_img, (5, 5), cv2.BORDER_DEFAULT)
        torso_image[inpaint_mask] = blur_img[inpaint_mask]

        # 최종 마스크 생성: 목, 상체, 인페인팅 영역을 모두 포함
        mask = (neck_part | torso_part | inpaint_mask)
        if inpaint_torso_mask is not None:
            mask = mask | inpaint_torso_mask  # torso 인페인팅 영역도 포함
        torso_image[~mask] = 0  # 마스크 바깥은 0(검정)으로
        torso_alpha[~mask] = 0  # 알파도 0(투명)으로

        # 결과 이미지를 저장: ori_imgs → torso_imgs, jpg → png, 알파 채널 포함
        cv2.imwrite(
            image_path.replace('ori_imgs', 'torso_imgs').replace('.jpg', '.png'),
            np.concatenate([torso_image, torso_alpha], axis=-1)
        )

    print(f'[INFO] ===== extracted torso and gt images =====')


def face_tracking(ori_imgs_dir):

    print(f'[INFO] ===== perform face tracking =====')

    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))
    
    # read one image to get H/W
    tmp_image = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED) # [H, W, 3]
    h, w = tmp_image.shape[:2]

    cmd = f'python data_utils/face_tracking/face_tracker.py --path={ori_imgs_dir} --img_h={h} --img_w={w} --frame_num={len(image_paths)}'
    # 3DMM(3D Morphable Model)을 사용하여 비디오의 각 프레임에서 얼굴의 3D 파라미터를 추정하는 코드
    os.system(cmd)

    print(f'[INFO] ===== finished face tracking =====') # track_params.pt 저장


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
    # train_val_split = valid_num - 25 * 20 # take the last 20s as valid set.
    train_val_split = int(valid_num * 10 / 11)

    train_ids = torch.arange(0, train_val_split)
    val_ids = torch.arange(train_val_split, valid_num)

    rot = euler2rot(euler_angle)
    rot_inv = rot.permute(0, 2, 1)
    trans_inv = -torch.bmm(rot_inv, trans.unsqueeze(2))

    pose = torch.eye(4, dtype=torch.float32)
    save_ids = ['train', 'val']
    train_val_ids = [train_ids, val_ids]
    mean_z = -float(torch.mean(trans[:, 2]).item())

    '''
    학습/검증 데이터셋 분할: 전체 프레임(valid_num)을 학습(train_ids)과 검증(val_ids) 세트로 나눕니다.
    process.py에서는 train_val_split = int(valid_num * 10 / 11)로 설정되어, 전체 프레임의 약 1/11을 검증 세트로 사용합니다.
    한편 split.py에서는 train_val_split = valid_num - 25 * 12 - 1로 설정되어, 마지막 12초 분량의 프레임을 검증 세트로 사용하는 것으로 보입니다 (프레임 레이트가 25fps라고 가정).
    process.py로 데이터를 분리할 경우 12초도 나오지 않을 경우가 생겨, 이를 대비하여 split.py 코드 또한 존재하는 것입니다.
    transforms_*.json 파일 생성 및 저장: 각 프레임에 대한 transform_matrix (월드 좌표계 변환 행렬)와 카메라 내외부 파라미터(초점 거리, 이미지 중심)를 포함하는 JSON 파일을 transforms_train.json과 transforms_val.json으로 저장합니다.
    '''

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 명령줄 인자를 파싱하기 위한 ArgumentParser 객체 생성
    parser.add_argument('path', type=str, help="path to video file")  # 비디오 파일 경로를 입력받는 인자 추가
    parser.add_argument('--task', type=int, default=-1, help="-1 means all")  # 수행할 작업 번호(-1이면 전체 수행)를 입력받는 인자 추가
    parser.add_argument('--asr', type=str, default='deepspeech', help="wav2vec or deepspeech")  # 오디오 특징 추출 방식 선택 인자 추가

    opt = parser.parse_args()  # 명령줄 인자 파싱

    base_dir = os.path.dirname(opt.path)  # 입력 비디오 파일의 디렉토리 경로 추출 ex) data/Macron
    
    wav_path = os.path.join(base_dir, 'aud.wav')  # 오디오 파일 저장 경로 설정 ex) data/Macron/aud.wav
    ori_imgs_dir = os.path.join(base_dir, 'ori_imgs')  # 원본 이미지 저장 폴더 경로 설정
    parsing_dir = os.path.join(base_dir, 'parsing')  # 얼굴 시맨틱 파싱 결과 저장 폴더 경로 설정 ex) data/Macron/parsing
    gt_imgs_dir = os.path.join(base_dir, 'gt_imgs')  # GT 이미지 저장 폴더 경로 설정 ex) data/Macron/gt_imgs
    torso_imgs_dir = os.path.join(base_dir, 'torso_imgs')  # 얼굴을 제외한 상체 이미지 저장 폴더 경로 설정 ex) data/Macron/torso_imgs

    os.makedirs(ori_imgs_dir, exist_ok=True)  # 원본 이미지 폴더 생성(이미 있으면 무시)
    os.makedirs(parsing_dir, exist_ok=True)  # 파싱 결과 폴더 생성(이미 있으면 무시)
    os.makedirs(gt_imgs_dir, exist_ok=True)  # GT 이미지 폴더 생성(이미 있으면 무시)
    os.makedirs(torso_imgs_dir, exist_ok=True)  # 얼굴을 제외한 상체 이미지 폴더 생성(이미 있으면 무시)

    # 오디오 추출
    if opt.task == -1 or opt.task == 1:
        extract_audio(opt.path, wav_path)  # 비디오에서 오디오(wav) 추출

    # 오디오 특징 추출
    if opt.task == -1 or opt.task == 2:
        extract_audio_features(wav_path, mode=opt.asr)  # wav 파일에서 오디오 특징 추출

    # 이미지 프레임 추출
    if opt.task == -1 or opt.task == 3:
        extract_images(opt.path, ori_imgs_dir)  # 비디오에서 프레임 이미지를 추출하여 ori_imgs_dir에 저장

    # 얼굴 파싱(세멘틱 분할)
    if opt.task == -1 or opt.task == 4:
        extract_semantics(ori_imgs_dir, parsing_dir)  # ori_imgs_dir의 이미지에 대해 얼굴 파싱 결과를 parsing_dir에 저장

    # 배경 추출
    if opt.task == -1 or opt.task == 5:
        extract_background(base_dir, ori_imgs_dir)  # 배경 추출 함수 실행

    # 얼굴 없는 상체 및 배경 이미지가 불변하는 GT 이미지 추출
    if opt.task == -1 or opt.task == 6:
        extract_torso_and_gt(base_dir, ori_imgs_dir)  # 얼굴 없는 상체 이미지와 이미지가 불변하는 GT 이미지 추출

    # 얼굴 랜드마크 추출
    if opt.task == -1 or opt.task == 7:
        extract_landmarks(ori_imgs_dir)  # ori_imgs_dir의 이미지에서 얼굴 랜드마크 추출
        # 랜드마크를 추출하는 이유는, 바로 아래 3DMM의 랜드마크와 일치시킨 결과를 기반으로 카메라 좌표계를 추정하기 위함이다

    # 얼굴 트래킹(3DMM 파라미터 추정)
    if opt.task == -1 or opt.task == 8:
        face_tracking(ori_imgs_dir)  # ori_imgs_dir의 이미지에 대해 얼굴 트래킹 수행 -> 카메라 좌표계 추정

    # transforms.json 저장(카메라 파라미터 등)
    if opt.task == -1 or opt.task == 9:
        save_transforms(base_dir, ori_imgs_dir)  # transforms_*.json 파일 저장 -> 월드 좌표계 추정