# 카메라-월드 좌표계 설명 (마크롱 얼굴 고정, 카메라 이동)

## 개요

이 파이프라인은 **`track_params.pt`**(3DMM 추적 결과)로부터 **`transforms_val.json`**을 생성해, **얼굴(마크롱)**은 **월드 원점에 고정**하고 **카메라가 이동**하는 형태의 **월드 좌표계**를 구성합니다.
목표는 “발화 중인 얼굴을 정지된 오브젝트로 두고, 시점(카메라)만 움직이는” 3D 시각화/렌더링을 일관된 컨벤션으로 수행하는 것입니다.

---

## 입력/출력 파일

* **입력**: `track_params.pt`

  * `id` : 인물 고유 형태(프레임 전체 동일)
  * `exp[t]` : 프레임 t의 표정 파라미터(79D)
  * `euler[t]` : 프레임 t의 오일러 각 **[pitch, yaw, roll]** (rad)
  * `trans[t]` : 프레임 t의 평행이동(저장 스케일 = **미터×10**)
    → 로딩 시 `/10.0` 해서 **미터**로 사용
  * `focal` : 초점거리(픽셀)

* **출력**: `transforms_val.json`

  * `focal_len` : 초점거리(픽셀)
  * `cx`, `cy` : 주점(principal point), 보통 이미지 중심
  * `frames[i].transform_matrix` : **카메라→월드(c2w)** 4×4 변환

    * 상위 3×3 : 회전 (R_{cw})
    * 상위 3×1 : 월드에서의 카메라 중심 (\mathbf{t}_w = C_w)
    * 마지막 행 : `[0, 0, 0, 1]`

> 주의: 본 문서/코드에서는 `transform_matrix`를 **카메라→월드(c2w)** 로 **통일**합니다. (혼동 방지)

---

## 좌표계 컨벤션

* **카메라 좌표계**:
  (\quad +X) 오른쪽, (\quad +Y) 아래, (\quad +Z) 앞(렌즈가 바라보는 방향)
* **월드 좌표계**:
  데이터 전처리에서 **얼굴 중심이 월드 원점(0,0,0)**에 오도록 정렬.
  프레임마다 카메라의 **외부 파라미터**(자세/위치)만 변합니다.

---

## 수학적 정의

### 오일러 각 → 회전행렬

코드(예: `euler2rot`) 정의:
[
R_x(\theta)=
\begin{bmatrix}
1&0&0\
0&\cos\theta&\sin\theta\
0&-\sin\theta&\cos\theta
\end{bmatrix},\quad
R_y(\phi)=
\begin{bmatrix}
\cos\phi&0&-\sin\phi\
0&1&0\
\sin\phi&0&\cos\phi
\end{bmatrix},\quad
R_z(\psi)=
\begin{bmatrix}
\cos\psi&-\sin\psi&0\
\sin\psi&\cos\psi&0\
0&0&1
\end{bmatrix}.
]
[
R = R_x(\text{pitch}),R_y(\text{yaw}),R_z(\text{roll})
]

### 월드↔카메라 변환

* **월드→카메라** (참고식):
  (\mathbf{x}_c = R,\mathbf{x}_w + \mathbf{t})
* **카메라→월드** (**본 프로젝트에서 사용하는 형식**):
  [
  \mathbf{x}*w = R*{cw},\mathbf{x}*c + \mathbf{t}*w,\quad
  R*{cw}=R^\top,\quad \mathbf{t}*w = C_w = -R^\top \mathbf{t}
  ]
  따라서 `transforms_val.json`의 `transform_matrix`는
  [
  T*{c2w}=
  \begin{bmatrix}
  R*{cw} & \mathbf{t}_w\
  \mathbf{0}^\top & 1
  \end{bmatrix}.
  ]

---

## `track_params.pt` → `transforms_val.json` 생성 로직(개요)

1. **스케일 복원**: `trans = stored_trans / 10.0` → **미터** 단위.
2. **포즈 행렬**: `R = euler2rot(euler[t])`.
   얼굴-고정 월드계로 전환 시, 카메라→월드 회전은 (R_{cw}=R^\top).
3. **카메라 중심**: (\mathbf{t}_w = C_w = -R^\top,\mathbf{t}).
4. **내부 파라미터**: `focal_len=focal`, `cx, cy`는 이미지 중심 등으로 설정.
5. **프레임 엔트리**: 각 t에 대해

   ```json
   {
     "img_id": t,
     "aud_id": t (필요시),
     "transform_matrix": [[... 4x4 c2w ...]]
   }
   ```
6. **결과**: 얼굴은 원점에, 카메라는 프레임마다 `R_cw, t_w`로 이동/회전.

---

## 시각화 스크립트(요지)

* 카메라 중심(월드) : (;C_w = \mathbf{t}_w)
* 카메라 정면(+Z(*c)) 방향(월드): (;R*{cw},[0,0,1]^\top)
* **뷰 프러스텀**:
  카메라 좌표계에서 near/far z(기본 +Z)와 픽셀 코너 ((u,v))를
  (;x=(u-cx)/f\cdot z,;y=(v-cy)/f\cdot z)로 만들고
  (;p_w = C_w + R_{cw},p_c) 로 월드에 투영.
* **옵션**:

  * `--flip180`: 프러스텀/정면/주광선을 **180° 반전**(z 부호 뒤집기)
    → (0,0,0)을 “향하게” 만들 때 사용.
  * `--elev --azim`: 보기 각도(카메라 각도 아님)
  * `--near --far`: 프러스텀 크기
  * `--frustum_color`: 프레임별 프러스텀 **단일 색** 통일
  * `--ray_scale`: 주광선 길이(씬 스케일 비율)
  * `--origin_axis_scale`: 원점 큰 XYZ 축 길이(씬 스케일 비율)
  * `--view_margin`(선택 구현): 전체를 멀리서 보기 위한 뷰 박스 마진

---

## 사용 예 (시각화)

```bash
python camera_trajectory_c2w_with_ray_flip_axes_originline.py \
  --json transforms_val.json \
  --near 0.03 --far 0.08 \
  --elev 10 --azim 0 \
  --ray_scale 0.25 \
  --origin_axis_scale 0.6 \
  --max_frames 150 \
  --flip180 \
  --frustum_color "#1f77b4"
```

* **원점–프러스텀 시작점 연결선**을 그려 원점이 화면 밖에 있어도 관계를 직관적으로 확인.
* 프레임 간 색 변화 없이 **단일 색상 프러스텀**으로 깔끔한 애니메이션.

---

## 자주 헷갈리는 포인트

* **c2w vs w2c**: 본 프로젝트는 **c2w**(카메라→월드)로 저장/가정.
  다른 툴을 섞을 때는 반드시 방향을 확인하세요.
* **축 컨벤션**: 카메라 ((+X) 오른쪽, (+Y) 아래, (+Z) 앞()).
  look-up(하늘 보기) 등에서 pitch 부호가 바뀔 수 있으니 `euler2rot` 정의를 기준으로 판단.
* **스케일**: `trans`는 저장 시 ×10, 로딩 시 `/10` → **미터** 보장.
* **짐벌락**: pitch=±90° 근처는 오일러 표현이 민감. 필요시 쿼터니언/축각 사용 고려.
* **원점을 향하기**: `--flip180`로 강제 반전, 혹은 각 프레임마다 (\text{sign}=\operatorname{sign}\big((0-C_w)\cdot z_{dir}\big))로 자동 부호 결정 로직을 추가할 수 있음.

---

## 간단 의사코드 (생성 파이프라인)

```python
# load track_params.pt
focal = pt['focal']
euler = pt['euler']            # [T,3], [pitch,yaw,roll] in rad
trans = pt['trans'] / 10.0     # [T,3], meters

# for each frame t:
R = euler2rot(euler[t])        # world->cam 형태 정의
R_cw = R.T
C_w = - R.T @ trans[t]
T_c2w = compose_4x4(R_cw, C_w)

# write transforms_val.json:
{
  "focal_len": focal,
  "cx": W/2, "cy": H/2,
  "frames": [
    { "img_id": t, "transform_matrix": T_c2w.tolist() } for t in range(T)
  ]
}
```

---

## 결론

* `track_params.pt`에서 추정된 **얼굴 포즈/위치**를 **카메라 외부 파라미터(c2w)** 로 재구성하여
  **얼굴=원점 고정**, **카메라 이동**의 월드 좌표계를 확립.
* `transforms_val.json`은 이 c2w 포즈를 프레임 단위로 저장하며,
  시각화 스크립트로 **프러스텀/주광선/원점 축/원점-프러스텀 연결선** 등을 통일된 컨벤션으로 확인 가능.
