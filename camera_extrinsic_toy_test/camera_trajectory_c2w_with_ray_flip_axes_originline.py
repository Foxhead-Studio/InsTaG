# camera_trajectory_c2w_with_ray_flip_axes_originline.py
# - transform_matrix: 카메라 -> 월드 (c2w) 해석
# - 카메라 궤적, 작은 뷰 프러스텀, 주광선(ray)
# - 원점(0,0,0)과 프러스텀 시작점(near-center)을 잇는 선분 추가
# - 프러스텀 색상 통일(--frustum_color)
# - 뷰 각도 조절: --elev --azim
# - 프러스텀/정면/주광선 180도 반전: --flip180
# - 결과: GIF(기본), ffmpeg 있으면 MP4 병행 저장

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import argparse

# -------------------- 기본 파라미터 --------------------
JSON_PATH_DEFAULT = "transforms_val.json"
OUT_GIF_DEFAULT   = "cam_traj_c2w_smallfrustum_ray.gif"
OUT_MP4_DEFAULT   = "cam_traj_c2w_smallfrustum_ray.mp4"

MAX_FRAMES = 600      # 최대 사용 프레임(다운샘플링)
NEAR_Z     = 0.03     # 프러스텀 near (작게)
FAR_Z      = 0.08     # 프러스텀 far  (작게)
FPS        = 15       # 애니메이션 FPS
RAY_SCALE  = 0.25     # 광선 길이(씬 스케일 비율)
ORIGIN_AXIS_SCALE = 0.5  # 원점 큰 XYZ 축 길이의 씬 스케일 비율(보조용)

# ------------------------------------------------------


def set_equal_3d(ax, X, Y, Z):
    """3D 영역을 동일 스케일로 맞춤."""
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    if max_range == 0:
        max_range = 1.0
    margin = 3   # ★ 여기 수치를 키우면 더 멀리서 전체가 보입니다. (기본 1.0 → 1.6 등)
    max_range *= margin

    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)


def frustum_points_world_c2w(Rcw, tw, focal, cx, cy, near=0.05, far=0.12, z_sign=1.0):
    """
    c2w(Rcw, tw), intrinsics(focal, cx, cy) 사용.
    카메라 좌표계(+X right, +Y down, +Z forward)에서 near/far 평면 코너와 중심을 만들고
    p_w = t_w + R_cw @ p_c 로 월드로 변환.

    z_sign=+1: +Z(forward) 방향 (기본)
    z_sign=-1: -Z(backward)  방향 (flip180)

    반환: (C_w, near_pts_w[4], far_pts_w[4], near_center_w, far_center_w)
    """
    W = 2.0 * cx
    H = 2.0 * cy
    uvs = [(0.0, 0.0), (W, 0.0), (W, H), (0.0, H)]  # TL, TR, BR, BL

    def corners_at(z_abs):
        z = z_sign * z_abs
        pts = []
        for (u, v) in uvs:
            x = (u - cx) / focal * z
            y = (v - cy) / focal * z  # +Y down
            pts.append(np.array([x, y, z], dtype=float))
        return pts

    near_c = corners_at(near)
    far_c  = corners_at(far)

    # 주광선 중심
    near_center_c = np.array([0.0, 0.0, z_sign * near], dtype=float)
    far_center_c  = np.array([0.0, 0.0, z_sign * far ], dtype=float)

    Cw = tw
    Rt = Rcw  # c2w

    near_w = [Cw + Rt @ p for p in near_c]
    far_w  = [Cw + Rt @ p for p in far_c]
    near_center_w = Cw + Rt @ near_center_c
    far_center_w  = Cw + Rt @ far_center_c
    return Cw, near_w, far_w, near_center_w, far_center_w


def main():
    parser = argparse.ArgumentParser(description="Camera trajectory with frustums, world axes, and a ray (c2w).")
    parser.add_argument("--json", default=JSON_PATH_DEFAULT, help="Path to transforms_*.json")
    parser.add_argument("--out_gif", default=OUT_GIF_DEFAULT)
    parser.add_argument("--out_mp4", default=OUT_MP4_DEFAULT)
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--near", type=float, default=NEAR_Z)
    parser.add_argument("--far", type=float, default=FAR_Z)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--ray_scale", type=float, default=RAY_SCALE,
                        help="Ray length as scene-scale ratio")
    parser.add_argument("--origin_axis_scale", type=float, default=ORIGIN_AXIS_SCALE,
                        help="원점 XYZ 축 길이를 씬 스케일(scene_extent)에 대한 비율로 지정")
    # 뷰 각도
    parser.add_argument("--elev", type=float, default=10.0, help="View elevation (degrees)")
    parser.add_argument("--azim", type=float, default=0.0,  help="View azimuth (degrees)")
    # 프러스텀/정면/주광선 180도 반전
    parser.add_argument("--flip180", action="store_true",
                        help="Flip frustum and forward direction by 180 degrees (e.g., to face the origin)")
    # 프러스텀 단일 색상
    parser.add_argument("--frustum_color", type=str, default="#1f77b4",
                        help="Frustum line color (default: matplotlib 'tab:blue')")
    # 주광선/원점-프러스텀 연결선 색상(선택)
    parser.add_argument("--ray_color", type=str, default="#d62728",
                        help="Main ray color (default: tab:red)")
    parser.add_argument("--origin_link_color", type=str, default="#2ca02c",
                        help="Line color from frustum start to origin (default: tab:green)")
    args = parser.parse_args()

    # ----- JSON 로드 -----
    with open(args.json, "r") as f:
        data = json.load(f)

    focal = float(data.get("focal_len", 1400.0))
    cx = float(data.get("cx", 256.0))
    cy = float(data.get("cy", 256.0))
    frames = data["frames"]

    # ----- 프레임 다운샘플링 -----
    N = len(frames)
    K = min(args.max_frames, N)
    idxs = np.linspace(0, N - 1, K, dtype=int)

    centers_w = []
    R_list = []
    t_list = []
    z_dirs_w = []

    z_sign = -1.0 if args.flip180 else 1.0

    for i in idxs:
        T = np.array(frames[i]["transform_matrix"], dtype=float)
        if T.shape == (3, 4):
            T = np.vstack([T, np.array([0, 0, 0, 1.0])])
        Rcw = T[:3, :3]   # camera->world rotation
        tw  = T[:3, 3]    # camera center in world
        centers_w.append(tw)
        R_list.append(Rcw)
        t_list.append(tw)
        # 카메라 정면(+/-Z_c)을 월드로
        zc_w = Rcw @ np.array([0.0, 0.0, z_sign])
        z_dirs_w.append(zc_w / np.linalg.norm(zc_w))

    centers_w = np.stack(centers_w, axis=0)

    # 씬 스케일
    scene_extent = np.linalg.norm(np.max(centers_w, axis=0) - np.min(centers_w, axis=0))
    if scene_extent == 0:
        scene_extent = 1.0
    # 길이들
    cam_arrow_len = 0.06 * scene_extent
    ray_len       = args.ray_scale * scene_extent
    origin_len    = args.origin_axis_scale * scene_extent

    # ----- 프러스텀/주광선 시작점 미리 계산 -----
    frusta = []
    near_centers_w = []
    for Rcw, tw in zip(R_list, t_list):
        Cw, near_w, far_w, near_center_w, _ = frustum_points_world_c2w(
            Rcw, tw, focal, cx, cy, near=args.near, far=args.far, z_sign=z_sign
        )
        frusta.append((Cw, near_w, far_w))
        near_centers_w.append(near_center_w)
    near_centers_w = np.stack(near_centers_w, axis=0)

    # ----- 그림/애니메이션 설정 -----
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("X (world)")
    ax.set_ylabel("Y (world)")
    ax.set_zlabel("Z (world)")
    ax.set_title("Camera Trajectory (c2w) with Frustums, Origin-Link, and a Ray")

    # 뷰각 설정
    ax.view_init(elev=args.elev, azim=args.azim)

    # 정적 궤적
    ax.plot(centers_w[:, 0], centers_w[:, 1], centers_w[:, 2])

    # 축 범위 동일화
    set_equal_3d(ax, centers_w[:, 0], centers_w[:, 1], centers_w[:, 2])

    # ===== 원점 큰 XYZ 축(보조: 멀리 있으면 안 보일 수 있음) =====
    ax.quiver(0, 0, 0, 1, 0, 0, length=origin_len, normalize=True, linewidth=2)  # +X_w
    ax.quiver(0, 0, 0, 0, 1, 0, length=origin_len, normalize=True, linewidth=2)  # +Y_w
    ax.quiver(0, 0, 0, 0, 0, 1, length=origin_len, normalize=True, linewidth=2)  # +Z_w
    ax.text(origin_len * 1.05, 0, 0, "X")
    ax.text(0, origin_len * 1.05, 0, "Y")
    ax.text(0, 0, origin_len * 1.05, "Z")
    # =============================================================

    # 애니메이션 아티스트
    traj_line, = ax.plot([], [], [], linewidth=2)
    point_scatter = ax.plot([], [], [], marker='o')[0]
    heading_quiver = None
    frustum_lines = []
    ray_line = None
    origin_link_line = None

    def draw_frustum(Cw, near_pts_w, far_pts_w, color="#1f77b4"):
        """프러스텀을 단일 색으로 그려 반환."""
        lines = []
        loop = [0, 1, 2, 3, 0]
        # near
        xs = [near_pts_w[i][0] for i in loop]
        ys = [near_pts_w[i][1] for i in loop]
        zs = [near_pts_w[i][2] for i in loop]
        lines += ax.plot(xs, ys, zs, color=color)
        # far
        xs = [far_pts_w[i][0] for i in loop]
        ys = [far_pts_w[i][1] for i in loop]
        zs = [far_pts_w[i][2] for i in loop]
        lines += ax.plot(xs, ys, zs, color=color)
        # sides
        for i in range(4):
            xs = [near_pts_w[i][0], far_pts_w[i][0]]
            ys = [near_pts_w[i][1], far_pts_w[i][1]]
            zs = [near_pts_w[i][2], far_pts_w[i][2]]
            lines += ax.plot(xs, ys, zs, color=color)
        # center to near corners
        for i in range(4):
            xs = [Cw[0], near_pts_w[i][0]]
            ys = [Cw[1], near_pts_w[i][1]]
            zs = [Cw[2], near_pts_w[i][2]]
            lines += ax.plot(xs, ys, zs, color=color)
        return lines

    def init():
        traj_line.set_data([], [])
        traj_line.set_3d_properties([])
        point_scatter.set_data([], [])
        point_scatter.set_3d_properties([])
        return traj_line, point_scatter

    def update(i):
        nonlocal heading_quiver, frustum_lines, ray_line, origin_link_line
        xs = centers_w[:i + 1, 0]
        ys = centers_w[:i + 1, 1]
        zs = centers_w[:i + 1, 2]
        traj_line.set_data(xs, ys)
        traj_line.set_3d_properties(zs)
        point_scatter.set_data([xs[-1]], [ys[-1]])
        point_scatter.set_3d_properties([zs[-1]])

        # 이전 아티스트 제거
        if heading_quiver is not None:
            heading_quiver.remove()
            heading_quiver = None
        for ln in frustum_lines:
            ln.remove()
        frustum_lines = []
        if ray_line is not None:
            ray_line.remove()
            ray_line = None
        if origin_link_line is not None:
            origin_link_line.remove()
            origin_link_line = None

        # 현재 카메라 정면(+/-Z_c) 방향 화살표
        C = centers_w[i]
        zdir = z_dirs_w[i]
        heading_quiver = ax.quiver(C[0], C[1], C[2],
                                   zdir[0], zdir[1], zdir[2],
                                   length=cam_arrow_len, normalize=True)

        # 현재 프레임 프러스텀(단일 색상)
        Cw, npts, fpts = frusta[i]
        frustum_lines = draw_frustum(Cw, npts, fpts, color=args.frustum_color)

        # 주광선(ray): near-plane 중심에서 zdir로 ray_len
        ray_start = near_centers_w[i]
        ray_end   = ray_start + ray_len * zdir
        ray_line, = ax.plot([ray_start[0], ray_end[0]],
                            [ray_start[1], ray_end[1]],
                            [ray_start[2], ray_end[2]],
                            linewidth=2, color=args.ray_color)

        # 원점(0,0,0) <-> 프러스텀 시작점(near-center) 연결선
        origin = np.zeros(3, dtype=float)
        origin_link_line, = ax.plot([origin[0], ray_start[0]],
                                    [origin[1], ray_start[1]],
                                    [origin[2], ray_start[2]],
                                    linewidth=2, linestyle="--",
                                    color=args.origin_link_color)

        return (traj_line, point_scatter, heading_quiver,
                ray_line, origin_link_line, *frustum_lines)

    anim = animation.FuncAnimation(
        fig, update, init_func=init,
        frames=len(centers_w), interval=60, blit=False
    )

    # 저장: GIF 우선
    try:
        anim.save(args.out_gif, writer=animation.PillowWriter(fps=args.fps))
        print(f"Saved GIF: {args.out_gif}")
    except Exception as e:
        print("GIF 저장 실패:", e)

    # ffmpeg 가능 시 MP4 병행
    try:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=args.fps, metadata=dict(artist='trajectory'), bitrate=2200)
        anim.save(args.out_mp4, writer=writer)
        print(f"Saved MP4: {args.out_mp4}")
    except Exception as e:
        print("MP4 저장 실패(아마 ffmpeg 미설치):", e)

    plt.close(fig)


if __name__ == "__main__":
    main()

'''
python camera_trajectory_c2w_with_ray_flip_axes_originline.py \
  --json transforms_val.json \
  --near 0.2 --far 0.3 \
  --elev 15 --azim 0 \
  --ray_scale 0.0 \
  --origin_axis_scale 0.5 \
  --max_frames 150 \
  --flip180 \
  --frustum_color "#1f77b4" \
  --ray_color "#d62728" \
  --origin_link_color "#2ca02c"
'''