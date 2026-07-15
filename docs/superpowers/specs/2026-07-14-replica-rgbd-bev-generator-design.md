# Replica RGB-D 观测域 BEV 数据生成器设计

## 目标

在远程 NVIDIA 无头服务器上，使用 Habitat-Sim 0.2.2 的 PTex 渲染路径读取原始 Replica v1.0，生成真实扫地机器人 BEV 模型的合成预训练数据。模型输入为同一时刻、同一前视针孔相机的 RGB 和深度反投影点云；监督输出为六个 multi-hot BEV 语义/几何通道，以及一个不由网络预测的观测有效 mask。

本设计不支持 ReplicaCAD，不修改 Habitat-Sim 渲染器，也不把场景全局信息或辅助视角扩展为模型的观测范围。

## 运行环境

- Python 3.8。
- Habitat-Sim 0.2.2，headless/EGL 构建。
- 原始 Replica v1.0，包含 PTex、语义网格、`info_semantic.json` 和 navmesh。
- NVIDIA GPU 显存能够容纳目标场景全部 PTex atlas。
- 生成器必须拒绝 Habitat-Sim 0.3.x 的 PTex 回退路径。

## 数据契约

每帧训练数据包含：

- 前视 RGB：`uint8 [H,W,3]`；
- 前视 Z-depth：毫米制 `uint16 [H,W]`，0 表示无效；
- 前视实例语义：`uint16 [H,W]`；
- 伪 LiDAR：float32 `[N,5]`，列为 `[x,y,z,intensity,time]`，后两列第一版填 0；
- BEV 标签：`uint8 [6,H_bev,W_bev]`；
- BEV 有效 mask：`uint8 [H_bev,W_bev]`。

RGB、深度和语义共享相机位置、朝向、分辨率、HFOV 和帧时刻。点云、BEV 和标定统一使用机器人 base 坐标：`x` 向前、`y` 向左、`z` 向上。

## 六个可学习通道

通道顺序固定为：

1. `floor`：观测有效区域内，按照机器人 navmesh 判定可通行的地面；
2. `carpet`：Replica 中的 `carpet`、`rug`、`mat`；
3. `obstacle`：有效深度点中，高度位于障碍阈值区间的几何端点；
4. `wall`：Replica 的 `wall` 实例；
5. `furniture`：床、桌、椅、柜、沙发、架子等明确家具实例；
6. `other`：其他有可靠实例类别、但不属于上述语义组的物体。

标签是 multi-hot，不使用 softmax。例如桌子格同时允许 `furniture=1` 和 `obstacle=1`，地毯格同时允许 `floor=1` 和 `carpet=1`。

`unknown` 不再占据可学习通道。它由 `1 - bev_valid_mask` 确定；在无效区域，六个标签通道必须全部为 0。

## 家具映射

第一版家具集合固定为：

```text
base-cabinet, beanbag, bed, bench, cabinet, chair, desk,
nightstand, plant-stand, rack, shelf, sofa, stool, table,
tv-stand, wall-cabinet
```

类别名称统一把 `_` 和 `-` 归一为空格后做精确匹配。`wall-plug` 不能因包含 `wall` 而误映射为墙；`undefined`、`unknown`、`void`、`background` 和 `none` 不生成语义标签；其余已知类别进入 `other`。

## 观测有效 mask

`bev_valid_masks` 只表达当前模型输入传感器真实覆盖的区域：

```text
valid = 前视 RGB-D 有效深度射线覆盖
        ∪ 未来独立 LiDAR 有效射线覆盖
```

当前版本没有独立 LiDAR，`.bin` 点云由前视深度产生，因此有效 mask 与前视伪 LiDAR共享同一批深度点。RGB 本身没有可用于 BEV 遮挡终点的距离；“RGB视角覆盖”必须通过对齐的有效深度射线落实。

mask 初始化为全 0。每条射线只标记传感器原点到有效返回点之间的格子；没有深度、超量程和返回点之后的区域保持 0。实现可按水平角度合并同方向射线并保留最远有效返回，以高效得到所有射线在 BEV 平面的覆盖并集。

NavMesh 只能生成 `floor` 标签，不得把相机视角之外的全局可通行区域加入 valid mask。左、后、右辅助 GT 相机不得参与正式标签或 valid mask。

## 标签生成流程

每帧只进行一次前视深度反投影，同时返回点云和对应实例 ID：

1. 从前视深度、内参和 Habitat 实际 `sensor_state` 得到 base 坐标点；
2. 保存同一批点的 `[x,y,z,0,0]` 为模型伪 LiDAR 输入；
3. 从这些点的射线生成 `valid_mask`；
4. 将 BEV 网格中心变换到世界坐标，查询 navmesh，并与 `valid_mask` 相交生成 `floor`；
5. 按点高生成 `obstacle`；
6. 按实例 ID 生成 `carpet/wall/furniture/other`；
7. 用 `valid_mask[None]` 最终裁剪全部六个通道。

这种流程确保训练输入、几何端点、语义端点和 loss 有效区域来自完全相同的前视观测。

## 可视化与训练

`visualizations/bev_labels/*.png` 使用下列 RGB 颜色（不是 OpenCV BGR）：

| 通道 | 类别 | RGB | 十六进制 | 含义 |
|---:|---|---:|---:|---|
| 0 | `floor` | `(160, 160, 160)` | `#A0A0A0` | 可通行地面，浅灰色 |
| 1 | `carpet` | `(70, 130, 180)` | `#4682B4` | 地毯、地垫，钢蓝色 |
| 2 | `obstacle` | `(220, 50, 47)` | `#DC322F` | 几何障碍物，红色 |
| 3 | `wall` | `(90, 90, 90)` | `#5A5A5A` | 墙体，深灰色 |
| 4 | `furniture` | `(147, 112, 219)` | `#9370DB` | 家具，紫色 |
| 5 | `other` | `(255, 190, 60)` | `#FFBE3C` | 其他已知语义物体，黄橙色 |
| — | 无效/unknown | `(20, 20, 20)` | `#141414` | `valid_mask=0` 的未观测区域，近黑色；不是第七个类别 |

因为 BEV 标签是 multi-hot，同一格可以同时属于多类，而 PNG 只能显示一种颜色。可视化按 `floor < carpet < wall < other < furniture < obstacle` 的覆盖优先级绘制：越靠右优先级越高，例如同时为 `furniture` 和 `obstacle` 的格最终显示红色。最后强制将 `valid_mask=0` 覆盖为近黑色。PNG 仅供人工检查，训练必须读取 `bev_masks/*.npy` 和 `bev_valid_masks/*.npy`。

训练时网络输出 `[B,6,H,W]` logits，单通道 mask 广播到六个通道：

```python
raw = focal_loss(logits, target, reduction="none")
valid = bev_valid_mask[:, None].float()
loss = (raw * valid).sum() / (valid.sum() * 6).clamp_min(1.0)
```

推理系统可对外封装六个 sigmoid 概率和一个外部 mask；unknown 始终由 `1-valid` 派生。

## 轨迹、划分与输出

轨迹从安全 navmesh 点开始，使用机器人小步长和平滑转向。发生碰撞或可选台阶检查失败时回退并转向。保存真值 `T_map_base` 和历史 sweeps。

train/val/test 必须按场景划分，同一轨迹帧不得泄漏到不同 split。每个场景输出：

```text
<output>/<scene>/
  images/
  depths/
  semantics/
  points/
  bev_masks/
  bev_valid_masks/
  calib/
  poses/
  visualizations/       # 可选
  metadata.json
  summary.json
  manifest.jsonl
```

根目录输出合并后的 `robot_infos_{train,val,test}.pkl` 和 `multi_scene_summary.json`。

## Schema 与断点续跑

新协议使用 `generator_schema_version=3`。生成 fingerprint 必须包含 schema 版本、六类顺序和语义映射配置，而不只包含 CLI 参数。

Schema v2 的 `unknown` 标签和“NavMesh + 四方向 GT” valid mask 与 v3 不兼容。新脚本必须拒绝在 v2 输出目录上 `--resume`，避免一个 manifest 中混合两种标签。生产数据应使用新输出目录；旧 RGB/深度/语义可由独立迁移工具离线重标，但迁移不属于本次脚本改动。

## 质量与测试

单元测试必须覆盖：

- 相机内参、外参、深度反投影和 BEV 边界；
- 家具、墙、地毯、其他和忽略类别映射；
- NavMesh 不扩大 valid mask；
- 深度射线产生 valid mask；
- valid 外六通道全部为 0；
- `furniture + obstacle` multi-hot；
- 可视化无效区域为黑色；
- schema/fingerprint 阻止旧数据续跑；
- sweep 位姿、split、manifest 和原子输出回归。

Habitat 集成测试先用 `office_1` 生成 1 至 10 帧，验证两个 `.npy` 的形状和 dtype、valid 外标签为 0、家具通道非空以及 RGB/Depth/Semantic 对齐；远程正式运行再覆盖全部 Replica 场景。
