# 原始 Replica RGB-D / BEV 数据生成

入口脚本：

```text
generate_mydata/robot_bev_closed_loop.py
```

该脚本只支持原始 Replica v1 PTex 数据，不支持 ReplicaCAD。正式渲染要求 Habitat-Sim 0.2.2；0.3.x 已移除原始 Replica 的 PTex 渲染路径，会退回不正确的 PLY 顶点色外观。

## 1. 远程服务器环境

推荐使用单独的 0.2.2 headless/EGL 环境：

```bash
conda activate habitat022
python -c 'import habitat_sim; print(habitat_sim.__version__)'
```

输出必须是：

```text
0.2.2
```

原始 Replica 的每个场景必须包含：

```text
mesh.ply
textures/parameters.json
textures/*-color-ptex.hdr
habitat/sorted_faces.bin
habitat/mesh_semantic.ply
habitat/info_semantic.json
habitat/mesh_semantic.navmesh
habitat/replica_stage.stage_config.json
```

stage config 中必须使用：

```json
"semantic_descriptor_filename": "info_semantic.json"
```

## 2. 仅检查数据和版本

该命令不会创建渲染器或占用 PTex 显存：

```bash
python generate_mydata/robot_bev_closed_loop.py \
  --dataset /mnt/u/ubuntu/workspace/dataset/HIKVISION/Habitat/replica.scene_dataset_config.json \
  --scene office_1 \
  --output-dir /data/replica_robot \
  --preflight-only
```

## 3. 单场景冒烟测试

先使用体积较小的 `office_1` 生成 10 帧：

```bash
mkdir -p data/robot_closed_loop

PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
python generate_mydata/robot_bev_closed_loop.py \
  --dataset /mnt/u/ubuntu/workspace/dataset/HIKVISION/Habitat/replica.scene_dataset_config.json \
  --scene office_1 \
  --output-dir data/robot_closed_loop/replica_robot_smoke \
  --num-frames 10 \
  --gpu-id 0 \
  --disable-physics \
  --save-visualization \
  --recompute-navmesh \
  2>&1 | tee data/robot_closed_loop/replica_robot_smoke.log
```

日志中必须出现 Habitat-Sim 0.2.2 的：

```text
Loading PTEX asset
```

然后检查：

```text
data/robot_closed_loop/replica_robot_smoke/office_1/summary.json
data/robot_closed_loop/replica_robot_smoke/office_1/visualizations/
```

不要在远程 NVIDIA 服务器设置 `LIBGL_ALWAYS_SOFTWARE`、`GALLIUM_DRIVER=llvmpipe` 或 `MESA_D3D12_DEFAULT_ADAPTER_NAME`。

脚本重建 navmesh 时默认使用 Habitat-Sim 0.2.2 在 Replica 上稳定工作的 Recast 参数：水平体素 `0.05 m`、垂直体素 `0.20 m`、最大爬升 `0.20 m`、最大三角面坡度 `45°`。这些参数用于容忍扫描碰撞网格的离散和表面噪声，不表示扫地机器人真的能够爬 20 cm 台阶。额外的 9 点邻域楼梯过滤默认关闭；如果某些含楼梯场景需要它，可显式添加 `--enable-stair-filter`，并通过 `--max-floor-height-delta` 调节阈值。

## 4. 正式多场景渲染

场景列表每行一个名称，例如 `replica_scenes.txt`：

```text
apartment_0
apartment_1
office_0
office_1
room_0
```

数据划分使用 JSON，并且一个场景只能属于一个 split：

```json
{
  "train": ["apartment_0", "apartment_1", "office_0"],
  "val": ["office_1"],
  "test": ["room_0"]
}
```

运行：

```bash
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
python generate_mydata/robot_bev_closed_loop.py \
  --dataset /mnt/u/ubuntu/workspace/dataset/HIKVISION/Habitat/replica.scene_dataset_config.json \
  --scenes-file replica_scenes.txt \
  --split-file replica_splits.json \
  --output-dir /data/replica_robot_v1 \
  --num-frames 20000 \
  --num-sweeps 5 \
  --gpu-id 0 \
  --disable-physics \
  --recompute-navmesh \
  2>&1 | tee /data/replica_robot_v1.log
```

未提供 `--split-file` 时，所有请求场景都归入 train；脚本不会再按同一轨迹的前后帧切分 train/val。

## 5. 断点续跑

每帧全部文件写完后才会追加 `manifest.jsonl`。中断后使用相同参数并添加：

```text
--resume
```

允许在续跑时增大 `--num-frames`。相机、BEV、点云或语义参数发生变化时脚本会拒绝续跑，防止一个目录混入不同配置的数据。

## 6. 每个场景的输出

```text
<output>/<scene>/
  images/                 # RGB PNG
  depths/                 # uint16 PNG，毫米，0 为无效
  semantics/              # uint16 实例 ID PNG
  points/                 # float32 [x,y,z,intensity,time]
  bev_masks/              # uint8 [6,H,W]
  bev_valid_masks/        # uint8 [H,W]
  calib/
    camera_intrinsic.txt
    camera2base.txt       # OpenCV optical -> base
    camera2base_habitat.txt
    lidar2base.txt
  poses/poses.txt         # T_map_base
  manifest.jsonl
  metadata.json
  summary.json
  scene_infos.pkl
```

根目录输出：

```text
robot_infos_train.pkl
robot_infos_val.pkl
robot_infos_test.pkl
multi_scene_summary.json
```

## 7. 坐标系和深度定义

点云、LiDAR 和 BEV 使用机器人 base 坐标：

```text
x: forward
y: left
z: up
```

`camera2base.txt` 使用标准 OpenCV optical 相机坐标：

```text
x: right
y: down
z: forward
```

脚本从 Habitat 返回的实际 `sensor_states` 计算外参，因此 RGB、深度、俯仰角和 base 之间使用同一条变换链。`camera2base_habitat.txt` 仅供核对 Habitat/OpenGL 的 `x-right, y-up, z-back` 坐标。

Habitat 深度是针孔相机光轴方向的 Z-depth，单位米。当前没有模拟真实 ToF 的径向 range、多路径、飞点、空洞和深度噪声；这些应在真实传感器标定后作为单独的 domain randomization 参数加入。

## 8. BEV 六类标签

通道顺序固定为：

```text
0 floor
1 carpet
2 obstacle
3 wall
4 other
5 unknown
```

标签为 multi-hot。比如桌子同时属于 `other` 和 `obstacle`。`other` 表示已识别但不属于 floor/carpet/wall 的语义实例。

模型输入点云只由前视深度反投影得到。BEV 真值默认额外使用前、左、后、右四方向的 depth + instance semantic，并结合 navmesh 生成；`bev_valid_masks` 表示可靠真值覆盖区域。`--disable-gt-multiview` 仅用于快速调试，不建议用于正式训练数据。
