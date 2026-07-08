# Habitat 小闭环 BEV 数据生成说明

这个脚本用于用 Habitat-Sim 生成扫地机器人视角的闭环数据。它主要用于验证数据结构、坐标链、BEV 语义标签、连续轨迹和多场景渲染流程。

脚本入口：

```bash
python generate_mydata/robot_bev_closed_loop.py
```

## 输出结构

单场景输出示例：

```text
data/robot_closed_loop/
  images/000000.png
  depths/000000.png
  points/000000.bin
  bev_masks/000000.npy
  visualizations/
    depth/000000.png
    pointclouds/000000.ply
    bev_labels/000000.png
  calib/
    camera_intrinsic.txt
    camera2base.txt
    lidar2base.txt
  poses/poses.txt
  metadata.json
  summary.json
  robot_infos_train.pkl
  robot_infos_val.pkl
```

其中：

```text
images/                 RGB 图像
depths/                 原始深度图，uint16 PNG，单位毫米
points/                 伪 LiDAR 点云，float32，每点 [x, y, z, intensity, time]
bev_masks/              训练用 BEV 语义 mask，shape [6, 150, 150]
visualizations/depth/   可视化深度图，仅 --save-visualization 时保存
visualizations/pointclouds/  PLY 点云，仅 --save-visualization 时保存
visualizations/bev_labels/   彩色 BEV 标签图，仅 --save-visualization 时保存
```

## 坐标系

点云、LiDAR、BEV 使用同一个机器人自车坐标系：

```text
x: forward
y: left
z: up
```

相机坐标系保持 Habitat 默认定义：

```text
x: right
y: up
z: back
forward: -z
```

无俯仰角时，`camera2base` 的含义是：

```text
x_base = -z_camera
y_base = -x_camera
z_base =  y_camera + camera_height
```

启用 `camera_pitch_deg` 后，以 `calib/camera2base.txt` 中保存的 4x4 矩阵为准。脚本会把同一个相机外参用于 RGB/depth 渲染、深度反投影、`.bin` 点云和 `.ply` 点云。

当前 LiDAR 是由前视深度反投影得到的伪 LiDAR，坐标系与 base 重合，所以 `lidar2base.txt` 是单位矩阵。

## 默认扫地机器人参数

```text
camera_height: 0.18 m
camera_pitch_deg: -10.0 deg
agent_height: 0.25 m
agent_radius: 0.18 m
agent_max_climb: 0.03 m
agent_max_slope: 10 deg
stair_check_radius: 0.50 m
max_floor_height_delta: 0.03 m
```

扫地机器人不能爬楼梯，因此建议运行时保留 `--recompute-navmesh`。脚本会用较小的 `agent_max_climb` 和 `agent_max_slope` 重算 navmesh，并在运动过程中检测机器人附近地面高度差。如果检测到类似台阶或楼梯的局部高度突变，会回退到上一帧位姿并转向。

## BEV 范围和分辨率

默认 BEV 范围：

```text
xbound: [0.0, 3.0, 0.02]     # forward, 对应 mask H
ybound: [-1.5, 1.5, 0.02]    # left-right, 对应 mask W
```

所以默认 BEV mask shape 为：

```text
[6, 150, 150]
```

自车原点不在 BEV 图像中心，而在 `x=0, y=0` 对应的近端中线位置。默认 BEV 覆盖机器人前方 0 到 3 米，以及左右各 1.5 米。

## BEV 语义类别

当前输出 6 个语义通道，顺序固定为：

```text
0 floor
1 carpet
2 obstacle
3 wall
4 threshold
5 unknown
```

BEV mask 生成逻辑：

```text
floor:
  对 BEV 网格中心点变换到 Habitat 世界坐标，用 navmesh 判断是否可通行。

obstacle:
  从前视 depth 反投影得到伪 LiDAR 点云，再按高度阈值投影到 BEV。
  其他已知 semantic object，包括 cable-like objects，也会归入 obstacle。

carpet / wall / threshold:
  如果场景有 semantic annotations，则用 semantic id 对应对象类别名，再按关键词映射到这些类别。

unknown:
  未被任何已知类别覆盖的格子。
```

语义映射不是固定的 `semantic id -> 类别`，因为 semantic id 通常是当前场景对象索引。脚本使用：

```text
semantic id -> sim.semantic_scene.objects[id].category.name() -> BEV 类别
```

关键词规则：

```text
carpet: carpet, rug, mat
threshold: threshold, sill, door frame, doorframe, transition
wall: wall
floor: floor, ground
other known semantic objects: obstacle
```

如果当前场景没有 semantic annotations，脚本会退回到：

```text
navmesh floor + depth obstacle + unknown
```

## BEV 可视化颜色

`visualizations/bev_labels/*.png` 使用固定颜色显示 6 个类别：

```text
floor:     RGB (160, 160, 160)  gray
carpet:    RGB (70, 130, 180)   blue
obstacle:  RGB (220, 50, 47)    red
wall:      RGB (90, 90, 90)     dark gray
threshold: RGB (255, 190, 60)   yellow
unknown:   RGB (20, 20, 20)     near black
```

如果同一个 BEV cell 有多个语义通道为 1，可视化覆盖优先级为：

```text
unknown < floor < carpet < wall < threshold < obstacle
```

因此 `obstacle` 优先级最高，最终显示为红色。

## 单场景运行

默认使用 baked lighting ReplicaCAD：

```bash
python generate_mydata/robot_bev_closed_loop.py \
  --dataset data/replica_cad_baked_lighting/replicaCAD_baked.scene_dataset_config.json \
  --scene Baked_sc1_staging_00 \
  --output-dir data/robot_closed_loop \
  --num-frames 10 \
  --save-visualization \
  --recompute-navmesh
```

`--num-frames` 表示当前场景生成的帧数。帧是连续轨迹，不是每帧独立随机采样。默认第 0 帧在随机安全位置渲染，后续每帧执行一次 `move_forward / turn_left / turn_right`。

默认动作和时间戳：

```text
move_forward probability: 0.75
turn_left probability: 0.125
turn_right probability: 0.125
step_size: 0.10 m
turn_angle: 15 deg
timestamp_step: 100000 us
logical FPS: 10 FPS
```

如果要更接近扫地机器人速度，可使用：

```bash
--step-size 0.03 --timestamp-step 100000
```

对应约 `0.3 m/s`。

## 多场景运行

多个场景可以用 `--scenes`：

```bash
python generate_mydata/robot_bev_closed_loop.py \
  --dataset data/replica_cad_baked_lighting/replicaCAD_baked.scene_dataset_config.json \
  --scenes Baked_sc1_staging_00 Baked_sc2_staging_00 \
  --output-dir data/robot_closed_loop_multi \
  --num-frames 1000 \
  --save-visualization \
  --recompute-navmesh
```

注意：当前 `--num-frames 1000` 表示每个场景生成 1000 帧。上面两个场景会总共生成 2000 帧。

输出结构：

```text
data/robot_closed_loop_multi/
  Baked_sc1_staging_00/
    images/
    depths/
    points/
    bev_masks/
    robot_infos_train.pkl
    robot_infos_val.pkl
  Baked_sc2_staging_00/
    ...
  robot_infos_train.pkl
  robot_infos_val.pkl
  multi_scene_summary.json
```

每个场景会单独保存一套数据，根目录还会生成合并版 `robot_infos_train.pkl` 和 `robot_infos_val.pkl`。

也可以用文件指定场景：

```bash
python generate_mydata/robot_bev_closed_loop.py \
  --dataset data/replica_cad_baked_lighting/replicaCAD_baked.scene_dataset_config.json \
  --scenes-file scenes.txt \
  --output-dir data/robot_closed_loop_multi \
  --num-frames 1000 \
  --save-visualization \
  --recompute-navmesh
```

`scenes.txt` 示例：

```text
# one scene per line
Baked_sc1_staging_00
Baked_sc2_staging_00
```

## 切回非 baked lighting 数据集

如果需要使用非 baked lighting 数据集，可运行：

```bash
python generate_mydata/robot_bev_closed_loop.py \
  --dataset data/replica_cad/replicaCAD.scene_dataset_config.json \
  --scene apt_1 \
  --output-dir data/robot_closed_loop_interactive \
  --num-frames 10 \
  --save-visualization \
  --recompute-navmesh
```

## 快速验证

生成后可以检查：

```bash
python - <<'PY'
import pickle
import numpy as np
from PIL import Image

root = "data/robot_closed_loop"
mask = np.load(f"{root}/bev_masks/000000.npy")
points = np.fromfile(f"{root}/points/000000.bin", dtype=np.float32).reshape(-1, 5)
depth = np.array(Image.open(f"{root}/depths/000000.png"))
bev_vis = np.array(Image.open(f"{root}/visualizations/bev_labels/000000.png"))

with open(f"{root}/robot_infos_train.pkl", "rb") as f:
    infos = pickle.load(f)["infos"]

print("mask", mask.shape, mask.dtype, mask.sum(axis=(1, 2)))
print("points", points.shape, points.dtype)
print("depth", depth.shape, depth.dtype, depth.max())
print("bev_vis", bev_vis.shape, bev_vis.dtype)
print("infos", len(infos), infos[0].keys())
print("sweeps frame1", len(infos[1]["sweeps"]) if len(infos) > 1 else 0)
PY
```

期望输出类似：

```text
mask (6, 150, 150) uint8
points (N, 5) float32
depth (480, 640) uint16
bev_vis (150, 150, 3) uint8
infos > 0
```

也可以直接查看：

```bash
cat data/robot_closed_loop/summary.json
```

重点关注：

```text
navmesh_area > 0
point_count_min > 0
mask_channel_sums.floor > 0
mask_channel_sums.obstacle > 0
semantic_id_mapping_count
stair_filter_recoveries
```

如果 `semantic_id_mapping_count = 0`，说明当前 scene 没有加载到 semantic object annotations。此时 `carpet/wall/threshold` 通常为 0，BEV 主要由 `floor/obstacle/unknown` 组成。

## 常见日志

`navmesh_instances ... not found`：
ReplicaCAD 配置引用了预计算 navmesh，但本地没有这些 `.navmesh` 文件。建议加 `--recompute-navmesh`，脚本会按扫地机器人参数重新计算。

`active scene does not contain semantic annotations`：
当前 scene 没有可用 semantic annotations。脚本仍可生成 `floor/obstacle/unknown`，但 `carpet/wall/threshold` 可能为空。

`Not implemented in base PhysicsManager. Install with --bullet`：
当前 Habitat-Sim 不是 Bullet 版本，ReplicaCAD articulated objects 可能无法完整加载。建议使用带 Bullet 的 Habitat-Sim 环境。

`duplicate static plugin ... ignoring`：
Magnum 插件重复注册提示，通常不影响生成。

`data/hab_fetch_1.0/robots/fetch_no_base.urdf` 缺失：
ReplicaCAD 配置中引用了 Fetch 机器人模板。当前脚本不使用 Fetch，一般可以忽略。
