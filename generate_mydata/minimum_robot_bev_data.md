# 自有数据训练 BEV 多语义网格地图最小需求

目标：使用扫地机器人自有数据训练多语义 BEV 网格地图，输入为前视 RGB + LiDAR，输出为多类别 BEV mask。

## 1. 每帧必须提供的数据

| 数据 | 推荐格式 | 说明 |
|------|----------|------|
| RGB 图像 | `.png` / `.jpg` | 前视摄像头图像 |
| 深度图 | `.png` uint16 | 可选但推荐保存，单位建议为毫米，0 表示无效深度 |
| LiDAR 点云 | `.bin` float32 | 推荐每点 `[x, y, z, intensity, time]`，没有强度/时间可填 0 |
| 可视化点云 | `.ply` | 可选调试格式，训练仍以 `.bin` 为准 |
| 可视化 BEV 标签 | `.png` RGB | 可选调试格式，训练仍以 `.npy` mask 为准 |
| BEV 真值 mask | `.npy` | 多语义监督标签，推荐 shape 为 `[C, H, W]` |
| 时间戳 | int / float | 用于图像、LiDAR、位姿同步 |
| 相机内参 | 3x3 矩阵 | `fx, fy, cx, cy` |
| 相机/LiDAR 外参 | 4x4 或 `R,t` | 用于 camera、LiDAR、BEV 对齐 |
| 每帧机器人位姿 | 4x4 或 `R,t` | 多帧累积需要，通常是 `T_map_base` |

## 2. BEV mask 格式

当前项目最兼容的标签格式：

```text
shape: [C, H, W]
dtype: uint8 / bool
value: 0 或 1
```

示例类别：

```yaml
map_classes:
  - floor
  - carpet
  - obstacle
  - wall
  - threshold
  - unknown
```

对应：

```text
mask[0] = floor
mask[1] = carpet
mask[2] = obstacle
...
```

当前生成脚本采用常用机器人坐标系：

```text
x: forward
y: left
z: up
```

`H, W` 由 BEV 范围和分辨率决定，例如：

```yaml
xbound: [0.0, 3.0, 0.02]     # forward
ybound: [-1.5, 1.5, 0.02]    # left-right
```

则：

```text
H = (3.0 - 0.0) / 0.02 = 150
W = (1.5 - (-1.5)) / 0.02 = 150
```

因此 mask shape 为：

```text
[C, 150, 150]
```

## 3. 多帧累积额外要求

多帧累积的关键是能把历史 LiDAR 点云变换到当前帧坐标系。最小需要：

```text
每帧 LiDAR 点云
每帧 timestamp
每帧机器人位姿 T_map_base
固定外参 T_base_lidar
```

变换链：

```text
历史 LiDAR -> 历史 base -> map -> 当前 base -> 当前 LiDAR
```

生成 `info.pkl` 时，每个样本的 `sweeps` 需要写入历史帧：

```python
"sweeps": [
    {
        "data_path": "data/robot/points/000000.bin",
        "timestamp": 1000000,
        "sensor2lidar_rotation": R_hist_lidar_to_cur_lidar,
        "sensor2lidar_translation": t_hist_lidar_to_cur_lidar,
    }
]
```

训练时 `LoadPointsFromMultiSweeps` 会用这些矩阵把历史点云对齐到当前 LiDAR 坐标系。

## 4. 推荐目录结构

```text
data/robot/
  images/
    000001.png
  depths/
    000001.png
  points/
    000001.bin
  bev_masks/
    000001.npy
  visualizations/
    depth/
      000001.png
    pointclouds/
      000001.ply
    bev_labels/
      000001.png
  calib/
    camera_intrinsic.txt
    camera2base.txt
    lidar2base.txt
  poses/
    poses.txt
  robot_infos_train.pkl
  robot_infos_val.pkl
```

## 5. 可以由脚本生成的内容

以下内容不需要人工逐帧准备，可以由转换脚本生成：

```text
info.pkl
sweeps 列表
空 3D bbox 字段
prev_token
相对位姿变换 sensor2lidar
点云 .bin 转换
```

如果不做 3D 检测，bbox 字段可以填空：

```python
"gt_boxes": np.zeros((0, 7), dtype=np.float32)
"gt_names": np.array([], dtype=object)
"gt_velocity": np.zeros((0, 2), dtype=np.float32)
"num_lidar_pts": np.zeros((0,), dtype=np.int64)
"num_radar_pts": np.zeros((0,), dtype=np.int64)
"valid_flag": np.zeros((0,), dtype=bool)
```

## 6. 最小结论

多帧、多语义 BEV 训练最少需要：

```text
RGB 图像
LiDAR 点云
多语义 BEV mask
相机内参
相机/LiDAR 外参
每帧时间戳
每帧 robot pose / odom / SLAM
```
