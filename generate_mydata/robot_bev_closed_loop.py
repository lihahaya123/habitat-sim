#!/usr/bin/env python3
"""Generate a minimal RGB + pseudo-LiDAR + BEV dataset with Habitat-Sim.

This is a small closed-loop data generator. It intentionally keeps semantics
simple so the full file layout, calibration, poses, BEV mask shape, and sweeps
can be validated before adding richer category mappings.

Output point/base convention:
  x: forward, y: left, z: up.

Habitat convention:
  x: right, y: up, z: back, agent/camera forward is -z.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import magnum as mn
import numpy as np
from PIL import Image

import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis, quat_to_coeffs


MAP_CLASSES = [
    "floor",
    "carpet",
    "obstacle",
    "wall",
    "threshold",
    "unknown",
]

BEV_CLASS_COLORS = {
    "floor": (160, 160, 160),
    "carpet": (70, 130, 180),
    "obstacle": (220, 50, 47),
    "wall": (90, 90, 90),
    "threshold": (255, 190, 60),
    "unknown": (20, 20, 20),
}

BEV_VIS_PRIORITY = [
    "unknown",
    "floor",
    "carpet",
    "wall",
    "threshold",
    "obstacle",
]

SEMANTIC_KEYWORD_RULES = {
    "carpet": ["carpet", "rug", "mat"],
    "threshold": ["threshold", "sill", "door frame", "doorframe", "transition"],
    "wall": ["wall"],
    "floor": ["floor", "ground"],
}

RGB_UUID = "front_rgb"
DEPTH_UUID = "front_depth"
SEMANTIC_UUID = "front_semantic"


def parse_bound(values: Sequence[float], name: str) -> Tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{name} must have three values: min max step")
    lo, hi, step = values
    if hi <= lo or step <= 0:
        raise ValueError(f"{name} must satisfy max > min and step > 0")
    return float(lo), float(hi), float(step)


def make_camera_intrinsic(width: int, height: int, hfov_deg: float) -> np.ndarray:
    hfov = math.radians(hfov_deg)
    fx = width / (2.0 * math.tan(hfov / 2.0))
    fy = fx
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def rotation_x(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float32,
    )


def camera_to_base_matrix(camera_height: float, camera_pitch_deg: float) -> np.ndarray:
    # p_base[x_forward, y_left, z_up] = R * p_camera[x_right, y_up, z_back] + t
    habitat_to_base = np.array(
        [
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    sensor_pitch = math.radians(camera_pitch_deg)
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = habitat_to_base @ rotation_x(sensor_pitch)
    out[:3, 3] = np.array([0.0, 0.0, camera_height], dtype=np.float32)
    return out


def quat_to_rotation_matrix(rotation) -> np.ndarray:
    coeffs = quat_to_coeffs(rotation)
    x, y, z, w = [float(v) for v in coeffs]
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def map_from_base_matrix(state: habitat_sim.AgentState) -> np.ndarray:
    # Base frame is [forward, left, up]. Habitat local frame is [right, up, back].
    robot_to_habitat = np.array(
        [
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = quat_to_rotation_matrix(state.rotation) @ robot_to_habitat
    out[:3, 3] = np.asarray(state.position, dtype=np.float32)
    return out


def base_grid_to_habitat_local(x_forward: np.ndarray, y_left: np.ndarray) -> np.ndarray:
    pts = np.zeros((x_forward.size, 3), dtype=np.float32)
    pts[:, 0] = -y_left.reshape(-1)
    pts[:, 2] = -x_forward.reshape(-1)
    return pts


def transform_habitat_local_to_world(state: habitat_sim.AgentState, local: np.ndarray) -> np.ndarray:
    rot = quat_to_rotation_matrix(state.rotation)
    return local @ rot.T + np.asarray(state.position, dtype=np.float32)


def make_cfg(args: argparse.Namespace) -> habitat_sim.Configuration:
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = args.dataset
    sim_cfg.scene_id = args.scene
    sim_cfg.enable_physics = args.use_physics
    sim_cfg.physics_config_file = args.physics_config
    sim_cfg.gpu_device_id = args.gpu_id
    sim_cfg.frustum_culling = True

    sensor_specs = []
    sensor_orientation = mn.Vector3(math.radians(args.camera_pitch_deg), 0.0, 0.0)

    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = RGB_UUID
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    rgb_spec.resolution = mn.Vector2i([args.height, args.width])
    rgb_spec.position = mn.Vector3(0.0, args.camera_height, 0.0)
    rgb_spec.orientation = sensor_orientation
    rgb_spec.hfov = mn.Deg(args.hfov)
    rgb_spec.far = args.zfar
    sensor_specs.append(rgb_spec)

    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid = DEPTH_UUID
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    depth_spec.channels = 1
    depth_spec.resolution = mn.Vector2i([args.height, args.width])
    depth_spec.position = mn.Vector3(0.0, args.camera_height, 0.0)
    depth_spec.orientation = sensor_orientation
    depth_spec.hfov = mn.Deg(args.hfov)
    depth_spec.far = args.zfar
    sensor_specs.append(depth_spec)

    if args.semantic_sensor:
        semantic_spec = habitat_sim.CameraSensorSpec()
        semantic_spec.uuid = SEMANTIC_UUID
        semantic_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
        semantic_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        semantic_spec.channels = 1
        semantic_spec.resolution = mn.Vector2i([args.height, args.width])
        semantic_spec.position = mn.Vector3(0.0, args.camera_height, 0.0)
        semantic_spec.orientation = sensor_orientation
        semantic_spec.hfov = mn.Deg(args.hfov)
        semantic_spec.far = args.zfar
        sensor_specs.append(semantic_spec)

    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.height = args.agent_height
    agent_cfg.radius = args.agent_radius
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=args.step_size)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=args.turn_angle)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=args.turn_angle)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def initialize_navmesh(sim: habitat_sim.Simulator, args: argparse.Namespace) -> None:
    nav_area = sim.pathfinder.navigable_area if sim.pathfinder.is_loaded else 0.0
    if sim.pathfinder.is_loaded and nav_area > 0.0 and not args.recompute_navmesh:
        print(f"Using dataset navmesh: area={nav_area:.3f}")
        print(
            "WARNING: --agent-max-climb and --agent-max-slope only affect recomputed "
            "navmeshes. Runtime stair filtering is still active."
        )
        return

    print("Recomputing navmesh.")
    nav_settings = habitat_sim.NavMeshSettings()
    nav_settings.set_defaults()
    nav_settings.agent_height = args.agent_height
    nav_settings.agent_radius = args.agent_radius
    nav_settings.agent_max_climb = args.agent_max_climb
    nav_settings.agent_max_slope = args.agent_max_slope
    nav_settings.filter_ledge_spans = True
    nav_settings.filter_walkable_low_height_spans = True
    nav_settings.include_static_objects = args.navmesh_include_static_objects
    if not sim.recompute_navmesh(sim.pathfinder, nav_settings):
        raise RuntimeError("Failed to load or recompute navmesh.")
    print(
        f"Navmesh area={sim.pathfinder.navigable_area:.3f} "
        f"max_climb={args.agent_max_climb:.3f} max_slope={args.agent_max_slope:.1f}"
    )


def is_floor_level_safe(
    sim: habitat_sim.Simulator,
    position: Sequence[float],
    radius: float,
    max_height_delta: float,
) -> bool:
    center = np.asarray(position, dtype=np.float32)
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [radius, 0.0, 0.0],
            [-radius, 0.0, 0.0],
            [0.0, 0.0, radius],
            [0.0, 0.0, -radius],
            [radius, 0.0, radius],
            [radius, 0.0, -radius],
            [-radius, 0.0, radius],
            [-radius, 0.0, -radius],
        ],
        dtype=np.float32,
    )

    snapped_heights = []
    for offset in offsets:
        sample = center + offset
        if not sim.pathfinder.is_navigable(sample, max_y_delta=max_height_delta):
            return False
        snapped = np.asarray(sim.pathfinder.snap_point(sample), dtype=np.float32)
        if not np.all(np.isfinite(snapped)):
            return False
        snapped_heights.append(float(snapped[1]))
    return max(snapped_heights) - min(snapped_heights) <= max_height_delta


def sample_safe_navigable_point(sim: habitat_sim.Simulator, args: argparse.Namespace) -> np.ndarray:
    for _ in range(args.safe_point_max_tries):
        point = np.asarray(sim.pathfinder.get_random_navigable_point(), dtype=np.float32)
        if is_floor_level_safe(
            sim,
            point,
            args.stair_check_radius,
            args.max_floor_height_delta,
        ):
            return point
    raise RuntimeError(
        "Failed to sample a stair-safe navigable point. "
        "Try relaxing --max-floor-height-delta or using --recompute-navmesh."
    )


def initialize_agent(sim: habitat_sim.Simulator, args: argparse.Namespace) -> None:
    state = habitat_sim.AgentState()
    state.position = sample_safe_navigable_point(sim, args)
    yaw = random.uniform(-math.pi, math.pi)
    state.rotation = quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0]))
    sim.initialize_agent(0, state)


def depth_to_points(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    camera_height: float,
    camera_pitch_deg: float,
    max_depth: float,
    stride: int,
    max_points: int,
    semantic: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    depth = np.asarray(depth, dtype=np.float32)
    rows = np.arange(0, depth.shape[0], stride)
    cols = np.arange(0, depth.shape[1], stride)
    uu, vv = np.meshgrid(cols, rows)
    dd = depth[vv, uu]
    valid = (dd > 0.0) & np.isfinite(dd) & (dd < max_depth)
    if not np.any(valid):
        return np.zeros((0, 5), dtype=np.float32), None

    u = uu[valid].astype(np.float32)
    v = vv[valid].astype(np.float32)
    d = dd[valid].astype(np.float32)
    semantic_ids = None
    if semantic is not None:
        semantic_arr = np.asarray(semantic)
        semantic_ids = semantic_arr[vv[valid], uu[valid]].astype(np.int64)

    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    x_right = (u - cx) * d / fx
    y_up = -(v - cy) * d / fy
    z_back = -d

    pts_camera = np.stack([x_right, y_up, z_back], axis=1).astype(np.float32)
    t_base_camera = camera_to_base_matrix(camera_height, camera_pitch_deg)
    pts = pts_camera @ t_base_camera[:3, :3].T + t_base_camera[:3, 3]
    if pts.shape[0] > max_points:
        keep = np.linspace(0, pts.shape[0] - 1, max_points).astype(np.int64)
        pts = pts[keep]
        if semantic_ids is not None:
            semantic_ids = semantic_ids[keep]

    intensity_time = np.zeros((pts.shape[0], 2), dtype=np.float32)
    points = np.concatenate([pts.astype(np.float32), intensity_time], axis=1)
    return points, semantic_ids


def save_depth_png(depth: np.ndarray, path: Path) -> None:
    depth_m = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    depth_mm = np.zeros(depth_m.shape, dtype=np.uint16)
    depth_mm[valid] = np.clip(depth_m[valid] * 1000.0, 0.0, 65535.0).astype(np.uint16)
    Image.fromarray(depth_mm).save(path)


def save_depth_vis(depth: np.ndarray, path: Path, max_depth: float) -> None:
    depth_m = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    vis = np.zeros((*depth_m.shape, 3), dtype=np.uint8)
    if np.any(valid):
        scaled = np.clip(depth_m[valid] / max_depth, 0.0, 1.0)
        gray = ((1.0 - scaled) * 255.0).astype(np.uint8)
        vis[valid] = np.stack([gray, gray, gray], axis=1)
    Image.fromarray(vis).save(path)


def save_points_ply(points: np.ndarray, path: Path) -> None:
    xyz = np.asarray(points[:, :3], dtype=np.float32)
    with open(path, "w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for x, y, z in xyz:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def save_bev_label_vis(mask: np.ndarray, path: Path) -> None:
    height, width = mask.shape[1], mask.shape[2]
    vis = np.zeros((height, width, 3), dtype=np.uint8)
    for class_name in BEV_VIS_PRIORITY:
        class_idx = MAP_CLASSES.index(class_name)
        vis[mask[class_idx] > 0] = BEV_CLASS_COLORS[class_name]
    Image.fromarray(vis).save(path)


def semantic_category_to_map_class(category_name: str) -> Optional[str]:
    normalized = category_name.lower().replace("_", " ").replace("-", " ")
    if not normalized or normalized in {"unknown", "void", "background", "none"}:
        return None
    for map_class, keywords in SEMANTIC_KEYWORD_RULES.items():
        if any(keyword in normalized for keyword in keywords):
            return map_class
    return "obstacle"


def build_semantic_id_to_class(sim: habitat_sim.Simulator) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    scene = getattr(sim, "semantic_scene", None)
    objects = getattr(scene, "objects", None)
    if not objects:
        return mapping

    for semantic_id, obj in enumerate(objects):
        if obj is None:
            continue
        try:
            category_name = obj.category.name()
        except Exception:
            continue
        map_class = semantic_category_to_map_class(category_name)
        if map_class is not None:
            mapping[int(semantic_id)] = map_class
    return mapping


def point_indices(
    points: np.ndarray,
    xbound: Tuple[float, float, float],
    ybound: Tuple[float, float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    x_min, x_max, x_step = xbound
    y_min, y_max, y_step = ybound
    x = points[:, 0]
    y = points[:, 1]
    valid = (x >= x_min) & (x < x_max) & (y >= y_min) & (y < y_max)
    rows = ((x[valid] - x_min) / x_step).astype(np.int64)
    cols = ((y[valid] - y_min) / y_step).astype(np.int64)
    return rows, cols


def make_bev_mask(
    sim: habitat_sim.Simulator,
    state: habitat_sim.AgentState,
    points: np.ndarray,
    semantic_ids: Optional[np.ndarray],
    semantic_id_to_class: Dict[int, str],
    xbound: Tuple[float, float, float],
    ybound: Tuple[float, float, float],
    min_obstacle_height: float,
    max_obstacle_height: float,
) -> np.ndarray:
    x_min, x_max, x_step = xbound
    y_min, y_max, y_step = ybound
    height = int(round((x_max - x_min) / x_step))
    width = int(round((y_max - y_min) / y_step))
    mask = np.zeros((len(MAP_CLASSES), height, width), dtype=np.uint8)

    x_centers = x_min + (np.arange(height, dtype=np.float32) + 0.5) * x_step
    y_centers = y_min + (np.arange(width, dtype=np.float32) + 0.5) * y_step
    y_grid, x_grid = np.meshgrid(y_centers, x_centers)
    local = base_grid_to_habitat_local(x_grid, y_grid)
    world = transform_habitat_local_to_world(state, local)

    floor = np.zeros((height * width,), dtype=np.uint8)
    for idx, point in enumerate(world):
        floor[idx] = 1 if sim.pathfinder.is_navigable(point, max_y_delta=0.5) else 0
    mask[MAP_CLASSES.index("floor")] = floor.reshape(height, width)

    if points.shape[0] > 0:
        obstacle_points = points[
            (points[:, 2] >= min_obstacle_height) & (points[:, 2] <= max_obstacle_height)
        ]
        if obstacle_points.shape[0] > 0:
            rows, cols = point_indices(obstacle_points, xbound, ybound)
            mask[MAP_CLASSES.index("obstacle"), rows, cols] = 1

    if semantic_ids is not None and semantic_id_to_class and points.shape[0] == semantic_ids.shape[0]:
        for semantic_id in np.unique(semantic_ids):
            map_class = semantic_id_to_class.get(int(semantic_id))
            if map_class not in MAP_CLASSES or map_class == "unknown":
                continue
            semantic_points = points[semantic_ids == semantic_id]
            if semantic_points.shape[0] == 0:
                continue
            rows, cols = point_indices(semantic_points, xbound, ybound)
            mask[MAP_CLASSES.index(map_class), rows, cols] = 1

    known = np.any(mask[: MAP_CLASSES.index("unknown")], axis=0)
    mask[MAP_CLASSES.index("unknown")] = np.logical_not(known).astype(np.uint8)
    return mask


def write_calibration(
    out_dir: Path,
    intrinsic: np.ndarray,
    camera_height: float,
    camera_pitch_deg: float,
) -> None:
    calib_dir = out_dir / "calib"
    calib_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(calib_dir / "camera_intrinsic.txt", intrinsic, fmt="%.8f")
    np.savetxt(
        calib_dir / "camera2base.txt",
        camera_to_base_matrix(camera_height, camera_pitch_deg),
        fmt="%.8f",
    )
    np.savetxt(calib_dir / "lidar2base.txt", np.eye(4, dtype=np.float32), fmt="%.8f")


def make_empty_gt() -> Dict[str, np.ndarray]:
    return {
        "gt_boxes": np.zeros((0, 7), dtype=np.float32),
        "gt_names": np.array([], dtype=object),
        "gt_velocity": np.zeros((0, 2), dtype=np.float32),
        "num_lidar_pts": np.zeros((0,), dtype=np.int64),
        "num_radar_pts": np.zeros((0,), dtype=np.int64),
        "valid_flag": np.zeros((0,), dtype=bool),
    }


def rel_path(path: Path) -> str:
    return Path(path).as_posix()


def scene_slug(scene: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", scene.strip())
    return slug.strip("_") or "scene"


def build_infos(
    frame_records: List[Dict[str, object]],
    output_dir: Path,
    num_sweeps: int,
    intrinsic: np.ndarray,
    camera_height: float,
    camera_pitch_deg: float,
) -> List[Dict[str, object]]:
    infos: List[Dict[str, object]] = []
    t_base_lidar = np.eye(4, dtype=np.float32)
    t_base_camera = camera_to_base_matrix(camera_height, camera_pitch_deg)

    for idx, record in enumerate(frame_records):
        t_map_lidar_cur = np.asarray(record["T_map_base"], dtype=np.float32) @ t_base_lidar
        t_cur_lidar_map = np.linalg.inv(t_map_lidar_cur)
        sweeps = []

        for hist_idx in range(max(0, idx - num_sweeps), idx):
            hist = frame_records[hist_idx]
            t_map_lidar_hist = np.asarray(hist["T_map_base"], dtype=np.float32) @ t_base_lidar
            t_hist_to_cur = t_cur_lidar_map @ t_map_lidar_hist
            sweeps.append(
                {
                    "data_path": rel_path(Path(hist["points_path"])),
                    "timestamp": int(hist["timestamp"]),
                    "sensor2lidar_rotation": t_hist_to_cur[:3, :3].astype(np.float32),
                    "sensor2lidar_translation": t_hist_to_cur[:3, 3].astype(np.float32),
                }
            )

        info: Dict[str, object] = {
            "token": f"{idx:06d}",
            "prev_token": f"{idx - 1:06d}" if idx > 0 else "",
            "timestamp": int(record["timestamp"]),
            "lidar_path": rel_path(Path(record["points_path"])),
            "image_path": rel_path(Path(record["image_path"])),
            "depth_path": rel_path(Path(record["depth_path"])),
            "bev_mask_path": rel_path(Path(record["bev_mask_path"])),
            "cam_intrinsic": intrinsic.copy(),
            "camera2base": t_base_camera.copy(),
            "lidar2base": t_base_lidar.copy(),
            "T_map_base": np.asarray(record["T_map_base"], dtype=np.float32),
            "sweeps": sweeps,
        }
        if record.get("ply_path") is not None:
            info["ply_path"] = rel_path(Path(record["ply_path"]))
        if record.get("depth_vis_path") is not None:
            info["depth_vis_path"] = rel_path(Path(record["depth_vis_path"]))
        if record.get("bev_vis_path") is not None:
            info["bev_vis_path"] = rel_path(Path(record["bev_vis_path"]))
        info.update(make_empty_gt())
        infos.append(info)

    metadata = {
        "dataset": "robot_closed_loop",
        "output_dir": output_dir.as_posix(),
        "map_classes": MAP_CLASSES,
        "bev_class_colors": BEV_CLASS_COLORS,
        "point_frame": "x_forward_y_left_z_up",
        "camera_height": float(camera_height),
        "camera_pitch_deg": float(camera_pitch_deg),
        "sweeps_transform": "history_lidar_to_current_lidar",
    }
    return [{"infos": infos, "metadata": metadata}]


def save_info_pickles(
    frame_records: List[Dict[str, object]],
    output_dir: Path,
    num_sweeps: int,
    intrinsic: np.ndarray,
    camera_height: float,
    camera_pitch_deg: float,
    val_ratio: float,
) -> None:
    wrapped = build_infos(
        frame_records,
        output_dir,
        num_sweeps,
        intrinsic,
        camera_height,
        camera_pitch_deg,
    )[0]
    infos = wrapped["infos"]
    metadata = wrapped["metadata"]

    val_count = int(round(len(infos) * val_ratio))
    if len(infos) > 1 and val_ratio > 0:
        val_count = max(1, min(val_count, len(infos) - 1))
    train_infos = infos[: len(infos) - val_count] if val_count else infos
    val_infos = infos[len(infos) - val_count :] if val_count else []

    with open(output_dir / "robot_infos_train.pkl", "wb") as f:
        pickle.dump({"infos": train_infos, "metadata": metadata}, f)
    with open(output_dir / "robot_infos_val.pkl", "wb") as f:
        pickle.dump({"infos": val_infos, "metadata": metadata}, f)


def save_metadata(output_dir: Path, args: argparse.Namespace, xbound, ybound) -> None:
    metadata = {
        "scene_dataset_config_file": args.dataset,
        "scene": args.scene,
        "use_physics": args.use_physics,
        "map_classes": MAP_CLASSES,
        "bev_class_colors": BEV_CLASS_COLORS,
        "semantic_keyword_rules": SEMANTIC_KEYWORD_RULES,
        "bev_mask_generation": {
            "floor": "navmesh navigability sampled at BEV grid centers",
            "obstacle": "depth-derived points filtered by obstacle height and projected to BEV",
            "semantic": "front semantic ids are mapped by category keywords and projected with depth into BEV",
            "unknown": "cells not covered by any known channel",
        },
        "xbound": list(xbound),
        "ybound": list(ybound),
        "camera_height": float(args.camera_height),
        "camera_pitch_deg": float(args.camera_pitch_deg),
        "agent_height": float(args.agent_height),
        "agent_radius": float(args.agent_radius),
        "agent_max_climb": float(args.agent_max_climb),
        "agent_max_slope": float(args.agent_max_slope),
        "stair_check_radius": float(args.stair_check_radius),
        "max_floor_height_delta": float(args.max_floor_height_delta),
        "safe_point_max_tries": int(args.safe_point_max_tries),
        "point_frame": "x_forward_y_left_z_up",
        "habitat_frame": "x_right_y_up_z_back",
        "depth_format": "uint16_png_millimeters",
        "save_visualization": bool(args.save_visualization),
        "semantic_sensor": bool(args.semantic_sensor),
        "note": "Closed-loop BEV uses navmesh floor, depth-derived obstacles, and unknown elsewhere.",
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def create_dirs(output_dir: Path) -> None:
    for name in ["images", "depths", "points", "bev_masks", "poses"]:
        (output_dir / name).mkdir(parents=True, exist_ok=True)


def next_action(last_collided: bool) -> str:
    if last_collided:
        return random.choice(["turn_left", "turn_right"])
    return random.choices(
        ["move_forward", "turn_left", "turn_right"],
        weights=[0.75, 0.125, 0.125],
        k=1,
    )[0]


def turn_agent_away(sim: habitat_sim.Simulator, previous_state: habitat_sim.AgentState) -> None:
    state = habitat_sim.AgentState()
    state.position = np.asarray(previous_state.position, dtype=np.float32)
    turn = random.choice([-math.pi / 2.0, math.pi / 2.0, math.pi])
    state.rotation = previous_state.rotation * quat_from_angle_axis(
        turn,
        np.array([0.0, 1.0, 0.0]),
    )
    sim.get_agent(0).set_state(state)


def generate(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.use_physics = args.enable_physics or (
        habitat_sim.built_with_bullet and not args.disable_physics
    )

    xbound = parse_bound(args.xbound, "xbound")
    ybound = parse_bound(args.ybound, "ybound")
    output_dir = Path(args.output_dir)
    create_dirs(output_dir)
    if args.save_visualization:
        for name in ["depth", "pointclouds", "bev_labels"]:
            (output_dir / "visualizations" / name).mkdir(parents=True, exist_ok=True)

    intrinsic = make_camera_intrinsic(args.width, args.height, args.hfov)
    write_calibration(output_dir, intrinsic, args.camera_height, args.camera_pitch_deg)
    save_metadata(output_dir, args, xbound, ybound)

    print(f"Dataset: {args.dataset}")
    print(f"Scene: {args.scene}")
    print(f"Output: {output_dir}")
    print(f"Habitat built_with_bullet={habitat_sim.built_with_bullet}")
    print(f"Physics enabled for this run={args.use_physics}")
    if "replica_cad" in args.dataset.lower() and not args.use_physics:
        print(
            "WARNING: ReplicaCAD articulated objects need Bullet physics. "
            "This run can validate the file format, but some objects may be skipped."
        )

    cfg = make_cfg(args)
    with habitat_sim.Simulator(cfg) as sim:
        initialize_navmesh(sim, args)
        initialize_agent(sim, args)
        semantic_id_to_class = build_semantic_id_to_class(sim) if args.semantic_sensor else {}
        if args.semantic_sensor:
            print(f"Semantic id mappings={len(semantic_id_to_class)}")

        frame_records: List[Dict[str, object]] = []
        poses_lines: List[str] = []
        last_collided = False
        stair_filter_recoveries = 0
        point_counts: List[int] = []
        mask_channel_sums = np.zeros((len(MAP_CLASSES),), dtype=np.int64)

        for frame_idx in range(args.num_frames):
            if frame_idx == 0:
                obs = sim.get_sensor_observations()
            else:
                previous_state = sim.get_agent(0).get_state()
                obs = sim.step(next_action(last_collided))
                last_collided = bool(obs.get("collided", False))
                current_state = sim.get_agent(0).get_state()
                if not is_floor_level_safe(
                    sim,
                    current_state.position,
                    args.stair_check_radius,
                    args.max_floor_height_delta,
                ):
                    turn_agent_away(sim, previous_state)
                    obs = sim.get_sensor_observations()
                    last_collided = True
                    stair_filter_recoveries += 1

            state = sim.get_agent(0).get_state()
            timestamp = args.timestamp_start + frame_idx * args.timestamp_step

            image_path = output_dir / "images" / f"{frame_idx:06d}.png"
            depth_path = output_dir / "depths" / f"{frame_idx:06d}.png"
            depth_vis_path = output_dir / "visualizations" / "depth" / f"{frame_idx:06d}.png"
            points_path = output_dir / "points" / f"{frame_idx:06d}.bin"
            ply_path = output_dir / "visualizations" / "pointclouds" / f"{frame_idx:06d}.ply"
            bev_mask_path = output_dir / "bev_masks" / f"{frame_idx:06d}.npy"
            bev_vis_path = output_dir / "visualizations" / "bev_labels" / f"{frame_idx:06d}.png"

            rgb = np.asarray(obs[RGB_UUID])
            Image.fromarray(rgb[:, :, :3]).save(image_path)

            depth = np.asarray(obs[DEPTH_UUID], dtype=np.float32)
            save_depth_png(depth, depth_path)
            if args.save_visualization:
                save_depth_vis(depth, depth_vis_path, args.max_depth)

            semantic_obs = obs.get(SEMANTIC_UUID) if args.semantic_sensor else None
            points, semantic_ids = depth_to_points(
                depth,
                intrinsic,
                args.camera_height,
                args.camera_pitch_deg,
                args.max_depth,
                args.depth_stride,
                args.max_points,
                semantic_obs,
            )
            points.astype(np.float32).tofile(points_path)
            if args.save_visualization:
                save_points_ply(points, ply_path)

            mask = make_bev_mask(
                sim,
                state,
                points,
                semantic_ids,
                semantic_id_to_class,
                xbound,
                ybound,
                args.min_obstacle_height,
                args.max_obstacle_height,
            )
            np.save(bev_mask_path, mask)
            if args.save_visualization:
                save_bev_label_vis(mask, bev_vis_path)
            point_counts.append(int(points.shape[0]))
            mask_channel_sums += mask.sum(axis=(1, 2)).astype(np.int64)

            t_map_base = map_from_base_matrix(state)
            poses_lines.append(
                " ".join(
                    [f"{frame_idx:06d}", str(timestamp)]
                    + [f"{v:.8f}" for v in t_map_base.reshape(-1)]
                )
            )

            frame_records.append(
                {
                    "timestamp": timestamp,
                    "image_path": image_path,
                    "depth_path": depth_path,
                    "depth_vis_path": depth_vis_path if args.save_visualization else None,
                    "points_path": points_path,
                    "ply_path": ply_path if args.save_visualization else None,
                    "bev_mask_path": bev_mask_path,
                    "bev_vis_path": bev_vis_path if args.save_visualization else None,
                    "T_map_base": t_map_base,
                }
            )

            print(
                f"[{frame_idx + 1:03d}/{args.num_frames:03d}] "
                f"points={points.shape[0]} mask={tuple(mask.shape)}"
            )

        (output_dir / "poses" / "poses.txt").write_text(
            "\n".join(poses_lines) + "\n", encoding="utf-8"
        )
        save_info_pickles(
            frame_records,
            output_dir,
            args.num_sweeps,
            intrinsic,
            args.camera_height,
            args.camera_pitch_deg,
            args.val_ratio,
        )
        summary = {
            "num_frames": args.num_frames,
            "point_count_min": int(min(point_counts)) if point_counts else 0,
            "point_count_max": int(max(point_counts)) if point_counts else 0,
            "point_count_mean": float(np.mean(point_counts)) if point_counts else 0.0,
            "mask_channel_sums": {
                name: int(value) for name, value in zip(MAP_CLASSES, mask_channel_sums)
            },
            "navmesh_area": float(sim.pathfinder.navigable_area)
            if sim.pathfinder.is_loaded
            else 0.0,
            "use_physics": bool(args.use_physics),
            "built_with_bullet": bool(habitat_sim.built_with_bullet),
            "depth_format": "uint16_png_millimeters",
            "save_visualization": bool(args.save_visualization),
            "semantic_sensor": bool(args.semantic_sensor),
            "semantic_id_mapping_count": int(len(semantic_id_to_class)),
            "camera_height": float(args.camera_height),
            "camera_pitch_deg": float(args.camera_pitch_deg),
            "agent_max_climb": float(args.agent_max_climb),
            "agent_max_slope": float(args.agent_max_slope),
            "stair_check_radius": float(args.stair_check_radius),
            "max_floor_height_delta": float(args.max_floor_height_delta),
            "stair_filter_recoveries": int(stair_filter_recoveries),
        }
        with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))


def load_scene_list(args: argparse.Namespace) -> List[str]:
    scenes: List[str] = []
    if args.scenes:
        scenes.extend(args.scenes)
    if args.scenes_file:
        scene_file = Path(args.scenes_file)
        for line in scene_file.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                scenes.append(stripped)
    if not scenes:
        scenes = [args.scene]
    deduped = []
    seen = set()
    for scene in scenes:
        if scene not in seen:
            deduped.append(scene)
            seen.add(scene)
    return deduped


def generate_all(args: argparse.Namespace) -> None:
    scenes = load_scene_list(args)
    if len(scenes) == 1:
        args.scene = scenes[0]
        generate(args)
        return

    root_output_dir = Path(args.output_dir)
    summaries = []
    all_train_infos = []
    all_val_infos = []
    for scene_idx, scene in enumerate(scenes):
        scene_args = argparse.Namespace(**vars(args))
        scene_args.scene = scene
        scene_args.output_dir = (root_output_dir / scene_slug(scene)).as_posix()
        scene_args.seed = args.seed + scene_idx
        scene_args.timestamp_start = args.timestamp_start + scene_idx * args.scene_timestamp_stride
        print(f"=== Scene {scene_idx + 1}/{len(scenes)}: {scene} ===")
        generate(scene_args)

        summary_path = Path(scene_args.output_dir) / "summary.json"
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                scene_summary = json.load(f)
            scene_summary["scene"] = scene
            scene_summary["output_dir"] = scene_args.output_dir
            summaries.append(scene_summary)

        train_info_path = Path(scene_args.output_dir) / "robot_infos_train.pkl"
        val_info_path = Path(scene_args.output_dir) / "robot_infos_val.pkl"
        if train_info_path.exists():
            with open(train_info_path, "rb") as f:
                all_train_infos.extend(pickle.load(f)["infos"])
        if val_info_path.exists():
            with open(val_info_path, "rb") as f:
                all_val_infos.extend(pickle.load(f)["infos"])

    root_output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset": "robot_closed_loop_multi_scene",
        "output_dir": root_output_dir.as_posix(),
        "scene_count": len(scenes),
        "scenes": scenes,
        "map_classes": MAP_CLASSES,
        "bev_class_colors": BEV_CLASS_COLORS,
        "point_frame": "x_forward_y_left_z_up",
        "sweeps_transform": "history_lidar_to_current_lidar",
    }
    with open(root_output_dir / "robot_infos_train.pkl", "wb") as f:
        pickle.dump({"infos": all_train_infos, "metadata": metadata}, f)
    with open(root_output_dir / "robot_infos_val.pkl", "wb") as f:
        pickle.dump({"infos": all_val_infos, "metadata": metadata}, f)
    with open(root_output_dir / "multi_scene_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_scenes": len(scenes),
                "scenes": scenes,
                "train_info_count": len(all_train_infos),
                "val_info_count": len(all_val_infos),
                "scene_summaries": summaries,
            },
            f,
            indent=2,
        )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a small Habitat-Sim RGB/pseudo-LiDAR/BEV closed loop."
    )
    parser.add_argument(
        "--dataset",
        default="data/replica_cad_baked_lighting/replicaCAD_baked.scene_dataset_config.json",
    )
    parser.add_argument("--scene", default="Baked_sc1_staging_00")
    parser.add_argument("--scenes", nargs="+")
    parser.add_argument("--scenes-file")
    parser.add_argument("--output-dir", default="data/robot_closed_loop")
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--num-sweeps", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.2)

    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--hfov", type=float, default=90.0)
    parser.add_argument("--zfar", type=float, default=8.0)
    parser.add_argument("--camera-height", type=float, default=0.18)
    parser.add_argument("--camera-pitch-deg", type=float, default=-10.0)
    parser.add_argument("--agent-height", type=float, default=0.25)
    parser.add_argument("--agent-radius", type=float, default=0.18)
    parser.add_argument("--agent-max-climb", type=float, default=0.03)
    parser.add_argument("--agent-max-slope", type=float, default=10.0)
    parser.add_argument("--stair-check-radius", type=float, default=0.50)
    parser.add_argument("--max-floor-height-delta", type=float, default=0.03)
    parser.add_argument("--safe-point-max-tries", type=int, default=1000)

    parser.add_argument("--xbound", type=float, nargs=3, default=[0.0, 3.0, 0.02])
    parser.add_argument("--ybound", type=float, nargs=3, default=[-1.5, 1.5, 0.02])
    parser.add_argument("--min-obstacle-height", type=float, default=0.02)
    parser.add_argument("--max-obstacle-height", type=float, default=0.8)

    parser.add_argument("--max-depth", type=float, default=4.0)
    parser.add_argument("--depth-stride", type=int, default=4)
    parser.add_argument("--max-points", type=int, default=20000)
    parser.add_argument("--save-visualization", action="store_true")
    parser.add_argument("--save-ply", action="store_true", help=argparse.SUPPRESS)

    parser.add_argument("--step-size", type=float, default=0.10)
    parser.add_argument("--turn-angle", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timestamp-start", type=int, default=1_000_000)
    parser.add_argument("--timestamp-step", type=int, default=100_000)
    parser.add_argument("--scene-timestamp-stride", type=int, default=10_000_000)

    parser.add_argument("--semantic-sensor", dest="semantic_sensor", action="store_true", default=True)
    parser.add_argument("--disable-semantic-sensor", dest="semantic_sensor", action="store_false")
    parser.add_argument("--enable-physics", action="store_true")
    parser.add_argument("--disable-physics", action="store_true")
    parser.add_argument("--physics-config", default="data/default.physics_config.json")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--recompute-navmesh", action="store_true")
    parser.add_argument("--navmesh-include-static-objects", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.num_frames <= 0:
        raise ValueError("--num-frames must be positive")
    if args.depth_stride <= 0:
        raise ValueError("--depth-stride must be positive")
    if args.agent_max_climb < 0:
        raise ValueError("--agent-max-climb must be non-negative")
    if args.agent_max_slope < 0:
        raise ValueError("--agent-max-slope must be non-negative")
    if args.stair_check_radius <= 0:
        raise ValueError("--stair-check-radius must be positive")
    if args.max_floor_height_delta < 0:
        raise ValueError("--max-floor-height-delta must be non-negative")
    if args.safe_point_max_tries <= 0:
        raise ValueError("--safe-point-max-tries must be positive")
    if args.save_ply:
        args.save_visualization = True
    generate_all(args)


if __name__ == "__main__":
    main()
