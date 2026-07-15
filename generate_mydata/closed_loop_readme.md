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
  --output-dir data/robot_closed_loop/replica_robot_observed_v3_smoke \
  --num-frames 10 \
  --gpu-id 0 \
  --disable-physics \
  --save-visualization \
  --recompute-navmesh \
  2>&1 | tee data/robot_closed_loop/replica_robot_observed_v3_smoke.log
```

日志中必须出现 Habitat-Sim 0.2.2 的：

```text
Loading PTEX asset
```

然后检查：

```text
data/robot_closed_loop/replica_robot_observed_v3_smoke/office_1/summary.json
data/robot_closed_loop/replica_robot_observed_v3_smoke/office_1/visualizations/
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
  --output-dir /data/replica_robot_observed_v3 \
  --num-frames 20000 \
  --num-sweeps 5 \
  --gpu-id 0 \
  --disable-physics \
  --recompute-navmesh \
  2>&1 | tee /data/replica_robot_observed_v3.log
```

未提供 `--split-file` 时，所有请求场景都归入 train；脚本不会再按同一轨迹的前后帧切分 train/val。

## 5. 断点续跑

每帧全部文件写完后才会追加 `manifest.jsonl`。中断后使用相同参数并添加：

```text
--resume
```

允许在续跑时增大 `--num-frames`。相机、BEV、点云或语义参数发生变化时脚本会拒绝续跑，防止一个目录混入不同配置的数据。

当前标签协议为 `generator_schema_version=3`。旧 schema v2 使用 `unknown` 第六通道，并用 NavMesh 和四方向 GT 扩大 valid mask，不能与 v3 混用；新脚本会拒绝在旧输出目录上 `--resume`。请为 v3 使用新的输出目录。

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
4 furniture
5 other
```

`visualizations/bev_labels/*.png` 的颜色固定为：

| 通道 | 类别 | RGB | Hex | 可视化含义 |
|---:|---|---|---|---|
| 0 | floor | `(160, 160, 160)` | `#A0A0A0` | 可通行地面，浅灰色 |
| 1 | carpet | `(70, 130, 180)` | `#4682B4` | 地毯，蓝色 |
| 2 | obstacle | `(220, 50, 47)` | `#DC322F` | 障碍物，红色 |
| 3 | wall | `(90, 90, 90)` | `#5A5A5A` | 墙体，深灰色 |
| 4 | furniture | `(147, 112, 219)` | `#9370DB` | 桌、椅、床、柜、沙发等家具，紫色 |
| 5 | other | `(255, 190, 60)` | `#FFBE3C` | 其他已识别语义物体，黄色/橙色 |

颜色只用于 PNG 可视化，训练标签仍以 `bev_masks/*.npy` 中的六通道 0/1 数组为准。由于标签是 multi-hot，同一网格可能属于多个类别；生成单张彩色图时按 `obstacle > furniture > other > wall > carpet > floor` 的顺序显示，高优先级类别会覆盖低优先级颜色。`bev_valid_masks==0` 的无观测区域固定显示为近黑色，但黑色不是语义类别。

标签为 multi-hot。比如桌子同时属于 `furniture` 和 `obstacle`。`other` 表示已识别但不属于 floor/carpet/wall/furniture 的语义实例。

模型输入点云、语义端点和 `bev_valid_masks` 使用同一批前视深度样本。valid mask 从前视有效深度射线生成，不由 NavMesh 或辅助视角扩大；NavMesh只在 valid 区域内生成 `floor` 标签。无观测区域六个语义通道全部为0，系统的 unknown 由 `1 - bev_valid_mask` 派生，不由网络预测。

训练六通道 sigmoid/focal loss 时，将单通道 mask 广播到六个类别：

```python
raw_loss = focal_loss(logits, target, reduction="none")
valid = bev_valid_mask[:, None].float()
loss = (raw_loss * valid).sum() / (valid.sum() * 6).clamp_min(1.0)
```
