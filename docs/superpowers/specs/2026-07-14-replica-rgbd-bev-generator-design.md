# Replica RGB-D BEV 数据生成器设计

## 目标

在远程 NVIDIA 无头服务器上，使用 Habitat-Sim 0.2.2 的 PTex 渲染路径读取原始 Replica v1.0，生成可用于真实扫地机器人 BEV 模型预训练的数据。第一版模型输入为前视 RGB 和由同帧深度反投影得到的点云；未来可增加预测语义图输入分支。

本设计不支持 ReplicaCAD，也不修改 Habitat-Sim 渲染器本身。

## 运行环境

- Python 3.8。
- Habitat-Sim 0.2.2，headless/EGL 构建。
- 原始 Replica v1.0 数据，包含 `mesh.ply`、`textures/*-color-ptex.hdr`、`habitat/sorted_faces.bin`、语义网格和 navmesh。
- NVIDIA GPU，显存能够容纳目标场景的全部 PTex atlas。
- 生成脚本保存在当前仓库，但必须兼容 Habitat-Sim 0.2.2 Python API。

## 范围

生成器负责：

- 加载单个或多个 Replica 场景；
- 渲染前视 RGB、深度和实例语义；
- 将输入深度反投影为 `[x, y, z, intensity, time]` float32 点云；
- 独立生成局部 BEV multi-hot 真值；
- 生成连续闭环轨迹、时间戳、标定、位姿和历史 sweeps；
- 保存训练所需文件、可选可视化、元数据和质量汇总；
- 支持按场景划分 train/val/test；
- 支持断点续跑，并拒绝静默覆盖参数不一致的数据。

生成器不负责：

- ReplicaCAD 或其他场景数据集；
- 刚体、关节物体或 Bullet 交互；
- 真实 ToF 噪声的最终标定；
- 模型训练代码；
- PTex 到 GLB 的转换。

## 数据集发现与校验

CLI 接受 `--dataset-root`，默认在根目录查找 `replica.scene_dataset_config.json`。场景可以通过 `--scene`、`--scenes` 或 `--scenes-file` 指定。

每个场景启动前必须检查：

- `mesh.ply`；
- `textures/parameters.json`；
- 至少一个 `textures/*-color-ptex.hdr`；
- `habitat/sorted_faces.bin`；
- `habitat/mesh_semantic.ply`；
- `habitat/info_semantic.json`；
- 可加载的 navmesh，或允许重新计算 navmesh。

缺少 PTex 文件时必须失败，不能退回 PLY 顶点色，以免把水面状 RGB 混入正式训练集。

## 坐标系和传感器

训练输出统一使用机器人 base 坐标系：

- `x` 向前；
- `y` 向左；
- `z` 向上。

前视 RGB、深度和实例语义传感器共享分辨率、HFOV、位置和俯仰角。相机参数全部由 CLI 指定，并写入元数据和标定文件。

输入点云严格从当前帧前视深度反投影，保持真实部署时的视场、遮挡和盲区。点格式固定为 float32 `[x, y, z, intensity, time]`，第一版后两维填零。

## BEV 真值

输入点云与 BEV 真值必须解耦。输入只来自前视 RGB-D；真值由 Habitat 的完整场景几何、实例语义和 navmesh 生成，不把前视可见点直接当作完整标签。

第一版输出六个 multi-hot 通道：

1. `floor`：navmesh 可通行区域和语义地面；
2. `carpet`：`carpet`、`rug`、`mat` 等地面软材质；
3. `obstacle`：高于地面阈值的占据几何；
4. `wall`：语义墙面及几何边界；
5. `other`：已识别但不属于 floor/carpet/wall 的语义实例；
6. `unknown`：不在有效真值覆盖内或无法确定的格子。

通道是 multi-hot，而不是互斥单标签。`other` 可以与 `obstacle` 重叠。例如桌子既是已知的 `other` 实例，也占据 `obstacle` 通道。`unknown` 只在没有有效已知标签时置一。

同时输出一个有效性/可观测性 mask，供训练忽略没有可靠真值的格子。该 mask 不占用六个语义通道。

## 轨迹和数据划分

轨迹由安全 navmesh 点开始，使用接近扫地机器人的小步长和平滑转向。发生碰撞、台阶风险或不可导航状态时回退并转向。保存真值 `T_map_base`，为以后增加带噪 odometry 留出字段。

数据划分以场景或完整轨迹为单位。任何一条轨迹的帧不得同时出现在 train 和 val/test 中。CLI 接受显式场景列表文件，生产运行不再使用“同一轨迹前段训练、尾段验证”的旧逻辑。

## 输出协议

每个场景输出：

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

根目录根据显式 split 文件生成合并后的：

```text
robot_infos_train.pkl
robot_infos_val.pkl
robot_infos_test.pkl
multi_scene_summary.json
```

`manifest.jsonl` 每成功完成一帧追加一条原子记录，用于断点续跑。`info.pkl` 在场景完成后由 manifest 构建，避免中途崩溃留下看似完整的索引。

## 断点续跑和失败处理

- 首次运行把完整 CLI 参数、版本、场景配置哈希写入 `metadata.json`。
- `--resume` 只允许在元数据一致时继续。
- 每帧先写临时文件，全部成功后原子重命名并追加 manifest。
- PTex 加载失败、GPU device lost、语义缺失、空深度或 navmesh 无效均记录明确错误并使该场景失败。
- 多场景模式允许记录失败场景后继续，但最终退出码必须非零，并在汇总中列出失败原因。
- 默认不静默回退到无语义或顶点色模式。

## 质量检查

每个场景汇总至少包含：

- 帧数、轨迹长度和碰撞/回退次数；
- RGB 非空及动态范围；
- 深度有效像素比例、最小值、最大值；
- 点云点数统计；
- 语义有效像素比例和实例映射数量；
- 六个 BEV 通道及 valid mask 的覆盖率；
- navmesh 面积；
- PTex atlas 数量和磁盘大小；
- Habitat-Sim、Python、GPU 和数据配置版本。

若任一正式类别在整个训练 split 中覆盖率为零，生成器必须在最终汇总中报错，而不是仅打印警告。

## 测试策略

纯 Python 单元测试覆盖：

- 相机内参与坐标变换；
- 深度反投影；
- 语义类别映射；
- BEV 网格索引、multi-hot 和 valid mask；
- sweep 相对位姿；
- 场景级 split 无泄漏；
- manifest 断点续跑和元数据不一致拒绝逻辑。

Habitat 集成测试分两级：

1. 本地小场景 smoke test：`office_1`，2 至 10 帧；
2. 远程生产测试：`office_1` 后运行 `apartment_0`，检查 PTex、RGB/Depth/Semantic 对齐、显存和输出完整性。

集成测试必须从日志确认 `Loading PTEX asset`，并验证输出 RGB 不是 PLY 顶点色回退结果。
