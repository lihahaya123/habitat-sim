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
import hashlib
import json
import math
import os
import pickle
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import magnum as mn
import numpy as np
from PIL import Image

import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis, quat_from_coeffs, quat_to_coeffs


MAP_CLASSES = [
    "floor",
    "carpet",
    "obstacle",
    "wall",
    "other",
    "unknown",
]

BEV_CLASS_COLORS = {
    "floor": (160, 160, 160),
    "carpet": (70, 130, 180),
    "obstacle": (220, 50, 47),
    "wall": (90, 90, 90),
    "other": (255, 190, 60),
    "unknown": (20, 20, 20),
}

BEV_VIS_PRIORITY = [
    "unknown",
    "floor",
    "carpet",
    "wall",
    "other",
    "obstacle",
]

SEMANTIC_KEYWORD_RULES = {
    "carpet": ["carpet", "rug", "mat"],
    "wall": ["wall"],
    "floor": ["floor", "ground"],
}

RGB_UUID = "front_rgb"
DEPTH_UUID = "front_depth"
SEMANTIC_UUID = "front_semantic"
FRL_PTEX_ASSET_TYPE = 3  # esp::assets::AssetType::FRL_PTEX_MESH in Habitat-Sim 0.2.2
GT_SENSOR_YAWS_DEG = {
    "left": 90.0,
    "back": 180.0,
    "right": -90.0,
}


def gt_depth_uuid(direction: str) -> str:
    return f"gt_{direction}_depth"


def gt_semantic_uuid(direction: str) -> str:
    return f"gt_{direction}_semantic"


@dataclass(frozen=True)
class ReplicaSceneFiles:
    dataset_config: Path
    scene_dir: Path
    stage_config: Path
    render_mesh: Path
    semantic_mesh: Path
    semantic_descriptor: Path
    navmesh: Path
    ptex_parameters: Path
    ptex_atlases: Tuple[Path, ...]

    @property
    def ptex_atlas_count(self) -> int:
        return len(self.ptex_atlases)


@dataclass(frozen=True)
class NavmeshTopdown:
    grid: np.ndarray
    min_x: float
    min_z: float
    meters_per_pixel: float


def parse_bound(values: Sequence[float], name: str) -> Tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{name} must have three values: min max step")
    lo, hi, step = values
    if hi <= lo or step <= 0:
        raise ValueError(f"{name} must satisfy max > min and step > 0")
    return float(lo), float(hi), float(step)


def validate_replica_scene(dataset_config: Path, scene: str) -> ReplicaSceneFiles:
    """Validate the official Replica v1 scene layout required by PTex."""
    dataset_config = Path(dataset_config).expanduser().resolve()
    if not dataset_config.is_file():
        raise FileNotFoundError(f"Replica dataset config does not exist: {dataset_config}")
    if dataset_config.name != "replica.scene_dataset_config.json":
        raise RuntimeError(
            "This generator only accepts the original Replica dataset config named "
            "replica.scene_dataset_config.json (ReplicaCAD is intentionally unsupported)."
        )

    scene_dir = dataset_config.parent / scene
    habitat_dir = scene_dir / "habitat"
    stage_config = habitat_dir / "replica_stage.stage_config.json"
    expected = {
        "render mesh": scene_dir / "mesh.ply",
        "PTex parameters": scene_dir / "textures" / "parameters.json",
        "PTex sorted faces": habitat_dir / "sorted_faces.bin",
        "semantic mesh": habitat_dir / "mesh_semantic.ply",
        "semantic descriptor": habitat_dir / "info_semantic.json",
        "navmesh": habitat_dir / "mesh_semantic.navmesh",
        "stage config": stage_config,
    }
    missing = [f"{label}: {path}" for label, path in expected.items() if not path.is_file()]
    atlases = tuple(sorted((scene_dir / "textures").glob("*-color-ptex.hdr")))
    if not atlases:
        missing.append(f"PTex atlases: {scene_dir / 'textures' / '*-color-ptex.hdr'}")
    if missing:
        raise FileNotFoundError(
            f"Scene {scene!r} is not a complete original Replica PTex scene:\n  "
            + "\n  ".join(missing)
        )

    try:
        stage_data = json.loads(stage_config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot parse Replica stage config: {stage_config}") from exc
    descriptor = stage_data.get("semantic_descriptor_filename")
    if descriptor != "info_semantic.json":
        raise RuntimeError(
            f"{stage_config} must contain semantic_descriptor_filename="
            "\"info_semantic.json\" for original Replica semantics; "
            f"found {descriptor!r}."
        )
    if stage_data.get("render_asset") != "../mesh.ply":
        raise RuntimeError(
            f"Unexpected render_asset in {stage_config}; expected '../mesh.ply' for PTex."
        )
    if stage_data.get("semantic_asset") != "mesh_semantic.ply":
        raise RuntimeError(
            f"Unexpected semantic_asset in {stage_config}; expected 'mesh_semantic.ply'."
        )
    if stage_data.get("nav_asset") != "mesh_semantic.navmesh":
        raise RuntimeError(
            f"Unexpected nav_asset in {stage_config}; expected 'mesh_semantic.navmesh'."
        )

    return ReplicaSceneFiles(
        dataset_config=dataset_config,
        scene_dir=scene_dir,
        stage_config=stage_config,
        render_mesh=expected["render mesh"],
        semantic_mesh=expected["semantic mesh"],
        semantic_descriptor=expected["semantic descriptor"],
        navmesh=expected["navmesh"],
        ptex_parameters=expected["PTex parameters"],
        ptex_atlases=atlases,
    )


def load_scene_splits(
    split_file: Optional[Path], scenes: Sequence[str]
) -> Dict[str, str]:
    """Return scene -> split, rejecting adjacent-frame and cross-split leakage."""
    if split_file is None:
        return {scene: "train" for scene in scenes}
    split_path = Path(split_file).expanduser().resolve()
    try:
        data = json.loads(split_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read split JSON: {split_path}") from exc
    allowed = {"train", "val", "test"}
    unknown_keys = set(data) - allowed
    if unknown_keys:
        raise ValueError(f"Unknown split names in {split_path}: {sorted(unknown_keys)}")

    assignments: Dict[str, str] = {}
    for split in ("train", "val", "test"):
        values = data.get(split, [])
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise ValueError(f"Split {split!r} must be a JSON list of scene names")
        for scene in values:
            if scene in assignments:
                raise ValueError(
                    f"Scene {scene!r} appears in more than one split: "
                    f"{assignments[scene]} and {split}"
                )
            assignments[scene] = split
    missing = sorted(set(scenes) - set(assignments))
    if missing:
        raise ValueError(f"Requested scenes missing from split file: {missing}")
    return {scene: assignments[scene] for scene in scenes}


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


def camera_optical_to_base_matrix(t_base_camera_habitat: np.ndarray) -> np.ndarray:
    """Convert a Habitat/OpenGL camera extrinsic to OpenCV optical axes.

    Habitat camera axes are x-right, y-up, z-back. Optical axes are x-right,
    y-down, z-forward. Returned matrices therefore work with the conventional
    K^-1 [u, v, 1] depth unprojection used by RGB-D training code.
    """
    optical_to_habitat = np.eye(4, dtype=np.float32)
    optical_to_habitat[1, 1] = -1.0
    optical_to_habitat[2, 2] = -1.0
    return np.asarray(t_base_camera_habitat, dtype=np.float32) @ optical_to_habitat


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


def map_from_habitat_pose(position: Sequence[float], rotation) -> np.ndarray:
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = quat_to_rotation_matrix(rotation)
    out[:3, 3] = np.asarray(position, dtype=np.float32)
    return out


def sensor_to_base_matrix(state: habitat_sim.AgentState, sensor_uuid: str) -> np.ndarray:
    """Derive the exact sensor extrinsic returned by Habitat, including all DOF."""
    if sensor_uuid not in state.sensor_states:
        raise KeyError(f"Agent state has no sensor named {sensor_uuid!r}")
    sensor_state = state.sensor_states[sensor_uuid]
    t_map_base = map_from_base_matrix(state)
    t_map_sensor = map_from_habitat_pose(sensor_state.position, sensor_state.rotation)
    return np.linalg.inv(t_map_base) @ t_map_sensor


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

    if args.gt_multiview:
        for direction, yaw_deg in GT_SENSOR_YAWS_DEG.items():
            # Keep auxiliary GT cameras level so their yaw is unambiguous across
            # Habitat/Magnum Euler composition versions. Their exact returned
            # sensor poses are still used for all depth backprojection.
            orientation = mn.Vector3(0.0, math.radians(yaw_deg), 0.0)
            gt_depth_spec = habitat_sim.CameraSensorSpec()
            gt_depth_spec.uuid = gt_depth_uuid(direction)
            gt_depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
            gt_depth_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            gt_depth_spec.channels = 1
            gt_depth_spec.resolution = mn.Vector2i([args.gt_height, args.gt_width])
            gt_depth_spec.position = mn.Vector3(0.0, args.camera_height, 0.0)
            gt_depth_spec.orientation = orientation
            gt_depth_spec.hfov = mn.Deg(args.gt_hfov)
            gt_depth_spec.far = args.zfar
            sensor_specs.append(gt_depth_spec)

            gt_semantic_spec = habitat_sim.CameraSensorSpec()
            gt_semantic_spec.uuid = gt_semantic_uuid(direction)
            gt_semantic_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
            gt_semantic_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            gt_semantic_spec.channels = 1
            gt_semantic_spec.resolution = mn.Vector2i([args.gt_height, args.gt_width])
            gt_semantic_spec.position = mn.Vector3(0.0, args.camera_height, 0.0)
            gt_semantic_spec.orientation = orientation
            gt_semantic_spec.hfov = mn.Deg(args.gt_hfov)
            gt_semantic_spec.far = args.zfar
            sensor_specs.append(gt_semantic_spec)

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


def configure_navmesh_settings(nav_settings, args: argparse.Namespace) -> None:
    """Apply settings supported by both Habitat-Sim 0.2.2 and newer releases."""
    nav_settings.set_defaults()
    nav_settings.cell_size = args.navmesh_cell_size
    nav_settings.cell_height = args.navmesh_cell_height
    nav_settings.agent_height = args.agent_height
    nav_settings.agent_radius = args.agent_radius
    nav_settings.agent_max_climb = args.agent_max_climb
    nav_settings.agent_max_slope = args.agent_max_slope
    nav_settings.filter_ledge_spans = True
    nav_settings.filter_walkable_low_height_spans = True
    if hasattr(nav_settings, "include_static_objects"):
        nav_settings.include_static_objects = args.navmesh_include_static_objects
    elif args.navmesh_include_static_objects:
        raise RuntimeError(
            "--navmesh-include-static-objects is unavailable in Habitat-Sim 0.2.2. "
            "Original Replica's static stage mesh is included automatically; omit this flag."
        )


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
    configure_navmesh_settings(nav_settings, args)
    if not sim.recompute_navmesh(sim.pathfinder, nav_settings):
        raise RuntimeError("Failed to load or recompute navmesh.")
    print(
        f"Navmesh area={sim.pathfinder.navigable_area:.3f} "
        f"cell={args.navmesh_cell_size:.3f}x{args.navmesh_cell_height:.3f} "
        f"max_climb={args.agent_max_climb:.3f} max_slope={args.agent_max_slope:.1f}"
    )


def is_floor_level_safe(
    sim: habitat_sim.Simulator,
    position: Sequence[float],
    radius: float,
    max_height_delta: float,
) -> bool:
    center = np.asarray(position, dtype=np.float32)
    if not sim.pathfinder.is_navigable(center, max_y_delta=max(0.5, max_height_delta)):
        return False
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
    max_horizontal_snap = max(0.10, radius * 0.5)
    for offset in offsets:
        sample = center + offset
        snapped = np.asarray(sim.pathfinder.snap_point(sample), dtype=np.float32)
        if not np.all(np.isfinite(snapped)):
            continue
        horizontal_snap = float(np.linalg.norm(snapped[[0, 2]] - sample[[0, 2]]))
        if horizontal_snap > max_horizontal_snap:
            # A wall or navmesh boundary can make a surrounding probe
            # non-navigable without implying a stair or floor discontinuity.
            continue
        snapped_heights.append(float(snapped[1]))
    if not snapped_heights:
        return False
    return max(snapped_heights) - min(snapped_heights) <= max_height_delta


def sample_safe_navigable_point(sim: habitat_sim.Simulator, args: argparse.Namespace) -> np.ndarray:
    for _ in range(args.safe_point_max_tries):
        point = np.asarray(sim.pathfinder.get_random_navigable_point(), dtype=np.float32)
        if not np.all(np.isfinite(point)):
            continue
        if not args.enable_stair_filter or is_floor_level_safe(
            sim,
            point,
            args.stair_check_radius,
            args.max_floor_height_delta,
        ):
            return point
    if args.enable_stair_filter:
        raise RuntimeError(
            "Failed to sample a stair-safe navigable point. Disable "
            "--enable-stair-filter or relax --max-floor-height-delta."
        )
    raise RuntimeError("Failed to sample a finite navigable point from the navmesh.")


def initialize_agent(sim: habitat_sim.Simulator, args: argparse.Namespace) -> None:
    state = habitat_sim.AgentState()
    state.position = sample_safe_navigable_point(sim, args)
    yaw = random.uniform(-math.pi, math.pi)
    state.rotation = quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0]))
    sim.initialize_agent(0, state)


def depth_to_points(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    t_base_camera_habitat: np.ndarray,
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
    t_base_camera = np.asarray(t_base_camera_habitat, dtype=np.float32)
    if t_base_camera.shape != (4, 4):
        raise ValueError("t_base_camera_habitat must be a 4x4 matrix")
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
    Image.fromarray(depth_mm).save(path, format="PNG")


def save_depth_vis(depth: np.ndarray, path: Path, max_depth: float) -> None:
    depth_m = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    vis = np.zeros((*depth_m.shape, 3), dtype=np.uint8)
    if np.any(valid):
        scaled = np.clip(depth_m[valid] / max_depth, 0.0, 1.0)
        gray = ((1.0 - scaled) * 255.0).astype(np.uint8)
        vis[valid] = np.stack([gray, gray, gray], axis=1)
    Image.fromarray(vis).save(path, format="PNG")


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


def atomic_save_png(array: np.ndarray, path: Path) -> None:
    temp = path.with_name(path.name + ".tmp")
    Image.fromarray(array).save(temp, format="PNG")
    temp.replace(path)


def atomic_save_npy(array: np.ndarray, path: Path) -> None:
    temp = path.with_name(path.name + ".tmp")
    with open(temp, "wb") as file:
        np.save(file, array)
        file.flush()
        os.fsync(file.fileno())
    temp.replace(path)


def atomic_save_points(points: np.ndarray, path: Path) -> None:
    temp = path.with_name(path.name + ".tmp")
    with open(temp, "wb") as file:
        np.asarray(points, dtype=np.float32).tofile(file)
        file.flush()
        os.fsync(file.fileno())
    temp.replace(path)


def append_manifest(path: Path, record: Dict[str, object]) -> None:
    encoded = json.dumps(record, separators=(",", ":"), sort_keys=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(encoded + "\n")
        file.flush()
        os.fsync(file.fileno())


def load_manifest(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    records: List[Dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid manifest JSON at {path}:{line_number}") from exc
        expected_index = len(records)
        if record.get("frame_index") != expected_index:
            raise RuntimeError(
                f"Non-contiguous manifest at {path}:{line_number}; "
                f"expected frame_index={expected_index}, found {record.get('frame_index')!r}"
            )
        records.append(record)
    return records


def generation_fingerprint(args: argparse.Namespace, scene: str) -> str:
    excluded = {
        "resume",
        "output_dir",
        "scenes",
        "scenes_file",
        "split_file",
        "num_frames",
        "preflight_only",
        "allow_version_mismatch",
    }
    values = {
        key: value
        for key, value in vars(args).items()
        if key not in excluded and isinstance(value, (str, int, float, bool, list, tuple, type(None)))
    }
    values["scene"] = scene
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def save_bev_label_vis(mask: np.ndarray, path: Path) -> None:
    height, width = mask.shape[1], mask.shape[2]
    vis = np.zeros((height, width, 3), dtype=np.uint8)
    for class_name in BEV_VIS_PRIORITY:
        class_idx = MAP_CLASSES.index(class_name)
        vis[mask[class_idx] > 0] = BEV_CLASS_COLORS[class_name]
    Image.fromarray(vis).save(path, format="PNG")


def semantic_category_to_map_class(category_name: str) -> Optional[str]:
    normalized = category_name.lower().replace("_", " ").replace("-", " ")
    if not normalized or normalized in {"unknown", "void", "background", "none"}:
        return None
    for map_class, keywords in SEMANTIC_KEYWORD_RULES.items():
        if any(keyword in normalized for keyword in keywords):
            return map_class
    return "other"


def build_semantic_id_to_class(sim: habitat_sim.Simulator) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    scene = getattr(sim, "semantic_scene", None)
    objects = getattr(scene, "objects", None)
    if not objects:
        return mapping

    for fallback_id, obj in enumerate(objects):
        if obj is None:
            continue
        try:
            category_name = obj.category.name()
        except Exception:
            continue
        semantic_id = fallback_id
        object_id = str(getattr(obj, "id", ""))
        match = re.search(r"(\d+)$", object_id)
        if match:
            semantic_id = int(match.group(1))
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
    height = int(round((x_max - x_min) / x_step))
    width = int(round((y_max - y_min) / y_step))
    rows = np.floor((x[valid] - x_min) / x_step).astype(np.int64)
    cols = np.floor((y[valid] - y_min) / y_step).astype(np.int64)
    # A float32 value just below the upper bound can round to exactly
    # height/width after division (for example 1.4999999 -> column 150).
    # The continuous bounds check above establishes that the point belongs to
    # the final cell, so clamp only this numerical boundary artifact.
    rows = np.clip(rows, 0, height - 1)
    cols = np.clip(cols, 0, width - 1)
    return rows, cols


def mark_observed_rays(
    valid_mask: np.ndarray,
    points: np.ndarray,
    camera_origin: np.ndarray,
    xbound: Tuple[float, float, float],
    ybound: Tuple[float, float, float],
    angular_resolution_deg: float = 0.5,
) -> None:
    """Mark BEV cells traversed by observed depth rays.

    Rays are reduced to the farthest return per angular bin. This preserves the
    visibility envelope while keeping production generation tractable.
    """
    if points.size == 0:
        return
    origin = np.asarray(camera_origin, dtype=np.float32)[:2]
    delta = np.asarray(points[:, :2], dtype=np.float32) - origin[None, :]
    ranges = np.linalg.norm(delta, axis=1)
    finite = np.isfinite(ranges) & (ranges > 1e-4)
    if not np.any(finite):
        return
    delta = delta[finite]
    ranges = ranges[finite]
    angles = np.arctan2(delta[:, 1], delta[:, 0])
    bin_width = math.radians(angular_resolution_deg)
    bins = np.round(angles / bin_width).astype(np.int32)

    x_min, x_max, x_step = xbound
    y_min, y_max, y_step = ybound
    sample_step = min(x_step, y_step) * 0.5
    for angle_bin in np.unique(bins):
        in_bin = bins == angle_bin
        farthest = int(np.argmax(np.where(in_bin, ranges, -1.0)))
        end = origin + delta[farthest]
        distance = float(ranges[farthest])
        count = max(2, int(math.ceil(distance / sample_step)) + 1)
        alpha = np.linspace(0.0, 1.0, count, dtype=np.float32)
        samples = origin[None, :] + alpha[:, None] * (end - origin)[None, :]
        inside = (
            (samples[:, 0] >= x_min)
            & (samples[:, 0] < x_max)
            & (samples[:, 1] >= y_min)
            & (samples[:, 1] < y_max)
        )
        if not np.any(inside):
            continue
        rows = np.floor((samples[inside, 0] - x_min) / x_step).astype(np.int64)
        cols = np.floor((samples[inside, 1] - y_min) / y_step).astype(np.int64)
        rows = np.clip(rows, 0, valid_mask.shape[0] - 1)
        cols = np.clip(cols, 0, valid_mask.shape[1] - 1)
        valid_mask[rows, cols] = 1


def build_navmesh_topdown(
    sim: habitat_sim.Simulator, meters_per_pixel: float, height: float
) -> NavmeshTopdown:
    bounds = sim.pathfinder.get_bounds()
    lower = np.asarray(bounds[0], dtype=np.float32)
    upper = np.asarray(bounds[1], dtype=np.float32)
    grid = np.asarray(
        sim.pathfinder.get_topdown_view(meters_per_pixel, float(height)), dtype=np.uint8
    )
    return NavmeshTopdown(
        grid=grid,
        min_x=float(min(lower[0], upper[0])),
        min_z=float(min(lower[2], upper[2])),
        meters_per_pixel=float(meters_per_pixel),
    )


def sample_navmesh_topdown(cache: NavmeshTopdown, world: np.ndarray) -> np.ndarray:
    # Promote before subtracting the lower bound. In float32, a coordinate just
    # below the upper bound can lose that distinction during subtraction.
    world64 = np.asarray(world, dtype=np.float64)
    x_offset = world64[:, 0] - cache.min_x
    z_offset = world64[:, 2] - cache.min_z
    inside = (
        (z_offset >= 0.0)
        & (z_offset < cache.grid.shape[0] * cache.meters_per_pixel)
        & (x_offset >= 0.0)
        & (x_offset < cache.grid.shape[1] * cache.meters_per_pixel)
    )
    values = np.zeros(world.shape[0], dtype=np.uint8)
    if np.any(inside):
        rows = np.floor(z_offset[inside] / cache.meters_per_pixel).astype(np.int64)
        cols = np.floor(x_offset[inside] / cache.meters_per_pixel).astype(np.int64)
        rows = np.clip(rows, 0, cache.grid.shape[0] - 1)
        cols = np.clip(cols, 0, cache.grid.shape[1] - 1)
        values[inside] = cache.grid[rows, cols]
    return values


def make_bev_mask(
    sim: habitat_sim.Simulator,
    state: habitat_sim.AgentState,
    gt_views: Sequence[Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]],
    semantic_id_to_class: Dict[int, str],
    xbound: Tuple[float, float, float],
    ybound: Tuple[float, float, float],
    min_obstacle_height: float,
    max_obstacle_height: float,
    navmesh_topdown: Optional[NavmeshTopdown] = None,
) -> Tuple[np.ndarray, np.ndarray]:
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

    if navmesh_topdown is not None:
        floor = sample_navmesh_topdown(navmesh_topdown, world)
    else:
        floor = np.zeros((height * width,), dtype=np.uint8)
        for idx, point in enumerate(world):
            floor[idx] = 1 if sim.pathfinder.is_navigable(point, max_y_delta=0.5) else 0
    mask[MAP_CLASSES.index("floor")] = floor.reshape(height, width)
    valid_mask = floor.reshape(height, width).copy()

    for points, semantic_ids, camera_origin in gt_views:
        mark_observed_rays(valid_mask, points, camera_origin, xbound, ybound)
        if points.shape[0] == 0:
            continue
        obstacle_points = points[
            (points[:, 2] >= min_obstacle_height) & (points[:, 2] <= max_obstacle_height)
        ]
        if obstacle_points.shape[0] > 0:
            rows, cols = point_indices(obstacle_points, xbound, ybound)
            mask[MAP_CLASSES.index("obstacle"), rows, cols] = 1

        if (
            semantic_ids is None
            or not semantic_id_to_class
            or points.shape[0] != semantic_ids.shape[0]
        ):
            continue
        for semantic_id in np.unique(semantic_ids):
            map_class = semantic_id_to_class.get(int(semantic_id))
            if map_class not in MAP_CLASSES or map_class in {"unknown", "obstacle"}:
                continue
            semantic_points = points[semantic_ids == semantic_id]
            if semantic_points.shape[0] == 0:
                continue
            rows, cols = point_indices(semantic_points, xbound, ybound)
            mask[MAP_CLASSES.index(map_class), rows, cols] = 1

    mask[MAP_CLASSES.index("unknown")] = np.logical_not(valid_mask).astype(np.uint8)
    return mask, valid_mask


def write_calibration(
    out_dir: Path,
    intrinsic: np.ndarray,
    t_base_camera_habitat: np.ndarray,
) -> None:
    calib_dir = out_dir / "calib"
    calib_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(calib_dir / "camera_intrinsic.txt", intrinsic, fmt="%.8f")
    np.savetxt(
        calib_dir / "camera2base.txt",
        camera_optical_to_base_matrix(t_base_camera_habitat),
        fmt="%.8f",
    )
    np.savetxt(
        calib_dir / "camera2base_habitat.txt",
        t_base_camera_habitat,
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
    t_base_camera_habitat: np.ndarray,
) -> List[Dict[str, object]]:
    infos: List[Dict[str, object]] = []
    t_base_lidar = np.eye(4, dtype=np.float32)
    t_base_camera = camera_optical_to_base_matrix(t_base_camera_habitat)
    token_prefix = scene_slug(output_dir.name)

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
            "token": f"{token_prefix}_{idx:06d}",
            "prev_token": f"{token_prefix}_{idx - 1:06d}" if idx > 0 else "",
            "timestamp": int(record["timestamp"]),
            "lidar_path": rel_path(Path(record["points_path"])),
            "image_path": rel_path(Path(record["image_path"])),
            "depth_path": rel_path(Path(record["depth_path"])),
            "semantic_path": rel_path(Path(record["semantic_path"])),
            "bev_mask_path": rel_path(Path(record["bev_mask_path"])),
            "bev_valid_mask_path": rel_path(Path(record["bev_valid_mask_path"])),
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
        "camera_frame": "opencv_optical_x_right_y_down_z_forward",
        "camera2base_convention": "T_base_camera_optical",
        "map_frame": "habitat_world_x_right_y_up_z_back",
        "sweeps_transform": "history_lidar_to_current_lidar",
    }
    return [{"infos": infos, "metadata": metadata}]


def save_info_pickles(
    frame_records: List[Dict[str, object]],
    output_dir: Path,
    num_sweeps: int,
    intrinsic: np.ndarray,
    t_base_camera_habitat: np.ndarray,
    scene_split: str,
) -> None:
    wrapped = build_infos(
        frame_records,
        output_dir,
        num_sweeps,
        intrinsic,
        t_base_camera_habitat,
    )[0]
    infos = wrapped["infos"]
    metadata = wrapped["metadata"]
    metadata["scene_split"] = scene_split
    payload = {"infos": infos, "metadata": metadata}
    with open(output_dir / "scene_infos.pkl", "wb") as f:
        pickle.dump(payload, f)
    with open(output_dir / f"robot_infos_{scene_split}.pkl", "wb") as f:
        pickle.dump(payload, f)


def save_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    xbound,
    ybound,
    scene_files: ReplicaSceneFiles,
    scene_split: str,
    fingerprint: str,
    t_base_camera_habitat: np.ndarray,
) -> None:
    metadata = {
        "generator": "original_replica_rgbd_bev",
        "generator_schema_version": 2,
        "fingerprint": fingerprint,
        "habitat_sim_version": getattr(habitat_sim, "__version__", "unknown"),
        "python_version": sys.version.split()[0],
        "scene_dataset_config_file": scene_files.dataset_config.as_posix(),
        "scene": args.scene,
        "scene_split": scene_split,
        "replica_scene_dir": scene_files.scene_dir.as_posix(),
        "ptex_atlas_count": scene_files.ptex_atlas_count,
        "ptex_atlas_bytes": sum(path.stat().st_size for path in scene_files.ptex_atlases),
        "use_physics": args.use_physics,
        "map_classes": MAP_CLASSES,
        "bev_class_colors": BEV_CLASS_COLORS,
        "semantic_keyword_rules": SEMANTIC_KEYWORD_RULES,
        "bev_mask_generation": {
            "floor": "navmesh navigability sampled at BEV grid centers",
            "obstacle": "360-degree GT depth points filtered by height and projected to BEV",
            "semantic": "360-degree instance semantics mapped to floor/carpet/wall/other",
            "unknown": "cells outside navmesh or multi-view depth ray coverage",
        },
        "xbound": list(xbound),
        "ybound": list(ybound),
        "camera_height": float(args.camera_height),
        "camera_pitch_deg": float(args.camera_pitch_deg),
        "camera_frame": "opencv_optical_x_right_y_down_z_forward",
        "habitat_camera_frame": "x_right_y_up_z_back",
        "camera2base": camera_optical_to_base_matrix(t_base_camera_habitat).tolist(),
        "camera2base_habitat": np.asarray(t_base_camera_habitat).tolist(),
        "agent_height": float(args.agent_height),
        "agent_radius": float(args.agent_radius),
        "agent_max_climb": float(args.agent_max_climb),
        "agent_max_slope": float(args.agent_max_slope),
        "stair_filter_enabled": bool(args.enable_stair_filter),
        "stair_check_radius": float(args.stair_check_radius),
        "max_floor_height_delta": float(args.max_floor_height_delta),
        "safe_point_max_tries": int(args.safe_point_max_tries),
        "point_frame": "x_forward_y_left_z_up",
        "habitat_frame": "x_right_y_up_z_back",
        "depth_format": "uint16_png_millimeters",
        "save_visualization": bool(args.save_visualization),
        "semantic_sensor": bool(args.semantic_sensor),
        "gt_multiview": bool(args.gt_multiview),
        "gt_sensor_yaws_deg": GT_SENSOR_YAWS_DEG if args.gt_multiview else {},
        "depth_model": "Habitat pinhole optical-axis Z depth in meters",
        "tof_note": "No ToF radial-range/noise model is applied; calibrate domain noise separately.",
    }
    temp = output_dir / "metadata.json.tmp"
    temp.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temp.replace(output_dir / "metadata.json")


def create_dirs(output_dir: Path) -> None:
    for name in [
        "images",
        "depths",
        "semantics",
        "points",
        "bev_masks",
        "bev_valid_masks",
        "calib",
        "poses",
    ]:
        (output_dir / name).mkdir(parents=True, exist_ok=True)


def next_action(last_collided: bool, rng: random.Random) -> str:
    if last_collided:
        return rng.choice(["turn_left", "turn_right"])
    return rng.choices(
        ["move_forward", "turn_left", "turn_right"],
        weights=[0.75, 0.125, 0.125],
        k=1,
    )[0]


def turn_agent_away(
    sim: habitat_sim.Simulator,
    previous_state: habitat_sim.AgentState,
    rng: random.Random,
) -> None:
    state = habitat_sim.AgentState()
    state.position = np.asarray(previous_state.position, dtype=np.float32)
    turn = rng.choice([-math.pi / 2.0, math.pi / 2.0, math.pi])
    state.rotation = previous_state.rotation * quat_from_angle_axis(
        turn,
        np.array([0.0, 1.0, 0.0]),
    )
    sim.get_agent(0).set_state(state)


def state_from_manifest(record: Dict[str, object]) -> habitat_sim.AgentState:
    state = habitat_sim.AgentState()
    state.position = np.asarray(record["agent_position"], dtype=np.float32)
    state.rotation = quat_from_coeffs(np.asarray(record["agent_rotation_xyzw"], dtype=np.float64))
    return state


def frame_record_from_manifest(
    record: Dict[str, object], output_dir: Path
) -> Dict[str, object]:
    def output_path(key: str) -> Path:
        return output_dir / str(record[key])

    result: Dict[str, object] = {
        "timestamp": int(record["timestamp"]),
        "image_path": output_path("image_path"),
        "depth_path": output_path("depth_path"),
        "semantic_path": output_path("semantic_path"),
        "points_path": output_path("points_path"),
        "bev_mask_path": output_path("bev_mask_path"),
        "bev_valid_mask_path": output_path("bev_valid_mask_path"),
        "T_map_base": np.asarray(record["T_map_base"], dtype=np.float32),
    }
    for key in ("depth_vis_path", "ply_path", "bev_vis_path"):
        if record.get(key):
            result[key] = output_path(key)
    return result


def ensure_manifest_files(record: Dict[str, object], output_dir: Path) -> None:
    required = (
        "image_path",
        "depth_path",
        "semantic_path",
        "points_path",
        "bev_mask_path",
        "bev_valid_mask_path",
    )
    missing = [str(output_dir / str(record[key])) for key in required if not (output_dir / str(record[key])).is_file()]
    if missing:
        raise RuntimeError("Manifest references missing frame files:\n  " + "\n  ".join(missing))


def save_visualization_atomically(save_function, path: Path, *args) -> None:
    temp = path.with_name(path.name + ".tmp")
    save_function(*args, temp)
    temp.replace(path)


def generate(
    args: argparse.Namespace,
    scene_files: ReplicaSceneFiles,
    scene_split: str,
) -> Dict[str, object]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.use_physics = bool(args.enable_physics and not args.disable_physics)

    xbound = parse_bound(args.xbound, "xbound")
    ybound = parse_bound(args.ybound, "ybound")
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest_path = output_dir / "manifest.jsonl"
    metadata_path = output_dir / "metadata.json"
    fingerprint = generation_fingerprint(args, args.scene)

    existing_manifest = load_manifest(manifest_path)
    if metadata_path.exists() and not args.resume:
        raise RuntimeError(
            f"Output already contains metadata: {output_dir}. Use --resume with identical "
            "parameters or choose a new output directory."
        )
    if existing_manifest and not args.resume:
        raise RuntimeError(
            f"Output already contains {len(existing_manifest)} completed frames: {output_dir}. "
            "Use --resume with identical parameters or choose a new output directory."
        )
    if args.resume and metadata_path.exists():
        previous_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous_metadata.get("fingerprint") != fingerprint:
            raise RuntimeError(
                f"--resume parameter mismatch for {output_dir}; existing fingerprint "
                f"{previous_metadata.get('fingerprint')!r}, current {fingerprint!r}."
            )
        if previous_metadata.get("scene_split") != scene_split:
            raise RuntimeError(
                f"--resume split mismatch for {args.scene}: existing "
                f"{previous_metadata.get('scene_split')!r}, current {scene_split!r}."
            )
    elif existing_manifest:
        raise RuntimeError(f"Manifest exists without compatible metadata: {output_dir}")

    create_dirs(output_dir)
    if args.save_visualization:
        for name in ["depth", "pointclouds", "bev_labels"]:
            (output_dir / "visualizations" / name).mkdir(parents=True, exist_ok=True)

    intrinsic = make_camera_intrinsic(args.width, args.height, args.hfov)
    gt_intrinsic = make_camera_intrinsic(args.gt_width, args.gt_height, args.gt_hfov)
    print(f"Dataset: {scene_files.dataset_config}")
    print(f"Scene: {args.scene} ({scene_split})")
    print(f"Output: {output_dir}")
    print(
        f"Replica PTex atlases={scene_files.ptex_atlas_count} "
        f"bytes={sum(path.stat().st_size for path in scene_files.ptex_atlases)}"
    )
    print(f"Habitat-Sim={getattr(habitat_sim, '__version__', 'unknown')} GPU={args.gpu_id}")

    cfg = make_cfg(args)
    with habitat_sim.Simulator(cfg) as sim:
        stage_template = sim.get_stage_initialization_template()
        if stage_template is None:
            raise RuntimeError(f"No initialized stage template for Replica scene {args.scene!r}")
        render_asset_type = int(stage_template.render_asset_type)
        if render_asset_type != FRL_PTEX_ASSET_TYPE:
            raise RuntimeError(
                "Habitat did not classify the Replica render mesh as FRL_PTEX_MESH "
                f"(expected asset type {FRL_PTEX_ASSET_TYPE}, found {render_asset_type}). "
                "Refusing a vertex-color fallback."
            )
        sim.seed(args.seed)
        initialize_navmesh(sim, args)
        if existing_manifest:
            for record in existing_manifest:
                ensure_manifest_files(record, output_dir)
            sim.initialize_agent(0, state_from_manifest(existing_manifest[-1]))
            print(f"Resuming after frame {len(existing_manifest) - 1:06d}")
        else:
            initialize_agent(sim, args)

        current_state = sim.get_agent(0).get_state()
        navmesh_topdown = build_navmesh_topdown(
            sim, min(xbound[2], ybound[2]), float(current_state.position[1])
        )
        t_base_camera_habitat = sensor_to_base_matrix(current_state, DEPTH_UUID)
        write_calibration(output_dir, intrinsic, t_base_camera_habitat)
        if not metadata_path.exists():
            save_metadata(
                output_dir,
                args,
                xbound,
                ybound,
                scene_files,
                scene_split,
                fingerprint,
                t_base_camera_habitat,
            )

        semantic_id_to_class = build_semantic_id_to_class(sim)
        if not semantic_id_to_class:
            raise RuntimeError(
                "Replica semantic scene loaded zero instance mappings. Check that the stage "
                "config uses info_semantic.json and mesh_semantic.ply."
            )
        print(f"Semantic id mappings={len(semantic_id_to_class)}")

        frame_records = [
            frame_record_from_manifest(record, output_dir) for record in existing_manifest
        ]
        last_collided = bool(existing_manifest[-1].get("collided", False)) if existing_manifest else False
        stair_filter_recoveries = sum(
            int(bool(record.get("stair_recovery", False))) for record in existing_manifest
        )

        for frame_idx in range(len(existing_manifest), args.num_frames):
            frame_rng = random.Random((args.seed + 1) * 1_000_003 + frame_idx)
            did_stair_recovery = False
            if frame_idx == 0:
                obs = sim.get_sensor_observations()
            else:
                previous_state = sim.get_agent(0).get_state()
                obs = sim.step(next_action(last_collided, frame_rng))
                last_collided = bool(obs.get("collided", False))
                current_state = sim.get_agent(0).get_state()
                if args.enable_stair_filter and not is_floor_level_safe(
                    sim,
                    current_state.position,
                    args.stair_check_radius,
                    args.max_floor_height_delta,
                ):
                    turn_agent_away(sim, previous_state, frame_rng)
                    obs = sim.get_sensor_observations()
                    last_collided = True
                    stair_filter_recoveries += 1
                    did_stair_recovery = True

            state = sim.get_agent(0).get_state()
            timestamp = args.timestamp_start + frame_idx * args.timestamp_step
            stem = f"{frame_idx:06d}"
            image_path = output_dir / "images" / f"{stem}.png"
            depth_path = output_dir / "depths" / f"{stem}.png"
            semantic_path = output_dir / "semantics" / f"{stem}.png"
            points_path = output_dir / "points" / f"{stem}.bin"
            bev_mask_path = output_dir / "bev_masks" / f"{stem}.npy"
            bev_valid_mask_path = output_dir / "bev_valid_masks" / f"{stem}.npy"
            depth_vis_path = output_dir / "visualizations" / "depth" / f"{stem}.png"
            ply_path = output_dir / "visualizations" / "pointclouds" / f"{stem}.ply"
            bev_vis_path = output_dir / "visualizations" / "bev_labels" / f"{stem}.png"

            rgb = np.asarray(obs[RGB_UUID])[:, :, :3].astype(np.uint8)
            depth = np.asarray(obs[DEPTH_UUID], dtype=np.float32)
            semantic_obs = np.asarray(obs[SEMANTIC_UUID])
            if semantic_obs.size == 0 or int(np.max(semantic_obs)) > np.iinfo(np.uint16).max:
                raise RuntimeError("Front semantic observation is empty or exceeds uint16 range")
            depth_mm = np.zeros(depth.shape, dtype=np.uint16)
            depth_valid = np.isfinite(depth) & (depth > 0.0)
            depth_mm[depth_valid] = np.clip(
                depth[depth_valid] * 1000.0, 0.0, 65535.0
            ).astype(np.uint16)
            atomic_save_png(rgb, image_path)
            atomic_save_png(depth_mm, depth_path)
            atomic_save_png(semantic_obs.astype(np.uint16), semantic_path)

            front_extrinsic = sensor_to_base_matrix(state, DEPTH_UUID)
            points, _ = depth_to_points(
                depth,
                intrinsic,
                front_extrinsic,
                args.max_depth,
                args.depth_stride,
                args.max_points,
            )
            atomic_save_points(points, points_path)

            front_gt_points, front_semantic_ids = depth_to_points(
                depth,
                intrinsic,
                front_extrinsic,
                args.max_depth,
                args.gt_depth_stride,
                args.gt_max_points_per_view,
                semantic_obs,
            )
            gt_views: List[Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]] = [
                (front_gt_points, front_semantic_ids, front_extrinsic[:3, 3])
            ]
            if args.gt_multiview:
                for direction in GT_SENSOR_YAWS_DEG:
                    depth_uuid = gt_depth_uuid(direction)
                    semantic_uuid = gt_semantic_uuid(direction)
                    view_extrinsic = sensor_to_base_matrix(state, depth_uuid)
                    view_points, view_semantic_ids = depth_to_points(
                        np.asarray(obs[depth_uuid], dtype=np.float32),
                        gt_intrinsic,
                        view_extrinsic,
                        args.max_depth,
                        args.gt_depth_stride,
                        args.gt_max_points_per_view,
                        np.asarray(obs[semantic_uuid]),
                    )
                    gt_views.append(
                        (view_points, view_semantic_ids, view_extrinsic[:3, 3])
                    )

            mask, valid_mask = make_bev_mask(
                sim,
                state,
                gt_views,
                semantic_id_to_class,
                xbound,
                ybound,
                args.min_obstacle_height,
                args.max_obstacle_height,
                navmesh_topdown,
            )
            atomic_save_npy(mask, bev_mask_path)
            atomic_save_npy(valid_mask, bev_valid_mask_path)

            if args.save_visualization:
                temp = depth_vis_path.with_name(depth_vis_path.name + ".tmp")
                save_depth_vis(depth, temp, args.max_depth)
                temp.replace(depth_vis_path)
                save_visualization_atomically(save_points_ply, ply_path, points)
                save_visualization_atomically(save_bev_label_vis, bev_vis_path, mask)

            t_map_base = map_from_base_matrix(state)
            relative = lambda path: path.relative_to(output_dir).as_posix()
            manifest_record: Dict[str, object] = {
                "frame_index": frame_idx,
                "timestamp": timestamp,
                "image_path": relative(image_path),
                "depth_path": relative(depth_path),
                "semantic_path": relative(semantic_path),
                "points_path": relative(points_path),
                "bev_mask_path": relative(bev_mask_path),
                "bev_valid_mask_path": relative(bev_valid_mask_path),
                "T_map_base": t_map_base.tolist(),
                "agent_position": np.asarray(state.position, dtype=np.float32).tolist(),
                "agent_rotation_xyzw": quat_to_coeffs(state.rotation).tolist(),
                "collided": last_collided,
                "stair_recovery": did_stair_recovery,
                "point_count": int(points.shape[0]),
                "depth_valid_ratio": float(np.mean(depth_valid)),
            }
            frame_record: Dict[str, object] = {
                "timestamp": timestamp,
                "image_path": image_path,
                "depth_path": depth_path,
                "semantic_path": semantic_path,
                "points_path": points_path,
                "bev_mask_path": bev_mask_path,
                "bev_valid_mask_path": bev_valid_mask_path,
                "T_map_base": t_map_base,
            }
            if args.save_visualization:
                manifest_record.update(
                    {
                        "depth_vis_path": relative(depth_vis_path),
                        "ply_path": relative(ply_path),
                        "bev_vis_path": relative(bev_vis_path),
                    }
                )
                frame_record.update(
                    {
                        "depth_vis_path": depth_vis_path,
                        "ply_path": ply_path,
                        "bev_vis_path": bev_vis_path,
                    }
                )
            append_manifest(manifest_path, manifest_record)
            frame_records.append(frame_record)
            print(
                f"[{frame_idx + 1:06d}/{args.num_frames:06d}] "
                f"points={points.shape[0]} valid={float(np.mean(valid_mask)):.3f} "
                f"mask={tuple(mask.shape)}"
            )

        pose_lines = []
        for frame_idx, record in enumerate(frame_records):
            matrix = np.asarray(record["T_map_base"], dtype=np.float32)
            pose_lines.append(
                " ".join(
                    [f"{frame_idx:06d}", str(record["timestamp"])]
                    + [f"{value:.8f}" for value in matrix.reshape(-1)]
                )
            )
        pose_temp = output_dir / "poses" / "poses.txt.tmp"
        pose_temp.write_text("\n".join(pose_lines) + "\n", encoding="utf-8")
        pose_temp.replace(output_dir / "poses" / "poses.txt")
        save_info_pickles(
            frame_records,
            output_dir,
            args.num_sweeps,
            intrinsic,
            t_base_camera_habitat,
            scene_split,
        )

        final_manifest = load_manifest(manifest_path)
        point_counts = [int(record.get("point_count", 0)) for record in final_manifest]
        mask_channel_sums = np.zeros((len(MAP_CLASSES),), dtype=np.int64)
        valid_sum = 0
        for record in final_manifest:
            mask_channel_sums += np.load(output_dir / str(record["bev_mask_path"])).sum(
                axis=(1, 2)
            ).astype(np.int64)
            valid_sum += int(np.load(output_dir / str(record["bev_valid_mask_path"])).sum())
        summary = {
            "scene": args.scene,
            "split": scene_split,
            "status": "complete",
            "num_frames": len(final_manifest),
            "point_count_min": int(min(point_counts)) if point_counts else 0,
            "point_count_max": int(max(point_counts)) if point_counts else 0,
            "point_count_mean": float(np.mean(point_counts)) if point_counts else 0.0,
            "mask_channel_sums": {
                name: int(value) for name, value in zip(MAP_CLASSES, mask_channel_sums)
            },
            "valid_cell_sum": valid_sum,
            "navmesh_area": float(sim.pathfinder.navigable_area),
            "semantic_id_mapping_count": len(semantic_id_to_class),
            "ptex_atlas_count": scene_files.ptex_atlas_count,
            "stair_filter_recoveries": stair_filter_recoveries,
        }
        summary_temp = output_dir / "summary.json.tmp"
        summary_temp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary_temp.replace(output_dir / "summary.json")
        print(json.dumps(summary, indent=2))
        return summary


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
    version = str(getattr(habitat_sim, "__version__", "unknown"))
    if version != "0.2.2" and not args.allow_version_mismatch:
        raise RuntimeError(
            f"Original Replica PTex rendering requires Habitat-Sim 0.2.2; found {version}. "
            "Activate the habitat022 environment. Use --allow-version-mismatch only for "
            "non-production diagnostics; Habitat-Sim 0.3.x removed PTex rendering."
        )
    dataset_config = Path(args.dataset).expanduser().resolve()
    validated = {
        scene: validate_replica_scene(dataset_config, scene) for scene in scenes
    }
    splits = load_scene_splits(
        Path(args.split_file) if args.split_file else None,
        scenes,
    )
    for scene in scenes:
        files = validated[scene]
        print(
            f"Preflight OK: {scene} split={splits[scene]} "
            f"PTex_atlases={files.ptex_atlas_count}"
        )
    if args.preflight_only:
        return

    root_output_dir = Path(args.output_dir).expanduser().resolve()
    root_output_dir.mkdir(parents=True, exist_ok=True)
    summaries: List[Dict[str, object]] = []
    failures: List[Dict[str, str]] = []
    all_infos: Dict[str, List[Dict[str, object]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    for scene_idx, scene in enumerate(scenes):
        scene_args = argparse.Namespace(**vars(args))
        scene_args.scene = scene
        scene_args.output_dir = (root_output_dir / scene_slug(scene)).as_posix()
        scene_args.seed = args.seed + scene_idx
        scene_args.timestamp_start = args.timestamp_start + scene_idx * args.scene_timestamp_stride
        print(f"=== Scene {scene_idx + 1}/{len(scenes)}: {scene} ===")
        try:
            summary = generate(scene_args, validated[scene], splits[scene])
            summary["output_dir"] = scene_args.output_dir
            summaries.append(summary)
            scene_info_path = Path(scene_args.output_dir) / "scene_infos.pkl"
            with open(scene_info_path, "rb") as file:
                all_infos[splits[scene]].extend(pickle.load(file)["infos"])
        except Exception as exc:
            failure = {"scene": scene, "error": f"{type(exc).__name__}: {exc}"}
            failures.append(failure)
            print(f"FAILED scene {scene}: {failure['error']}", file=sys.stderr)

    metadata = {
        "dataset": "original_replica_rgbd_bev",
        "habitat_sim_version": version,
        "scene_dataset_config_file": dataset_config.as_posix(),
        "output_dir": root_output_dir.as_posix(),
        "scene_count": len(scenes),
        "scenes": scenes,
        "scene_splits": splits,
        "map_classes": MAP_CLASSES,
        "bev_class_colors": BEV_CLASS_COLORS,
        "point_frame": "x_forward_y_left_z_up",
        "camera_frame": "opencv_optical_x_right_y_down_z_forward",
        "map_frame": "habitat_world_x_right_y_up_z_back",
        "sweeps_transform": "history_lidar_to_current_lidar",
    }
    for split in ("train", "val", "test"):
        with open(root_output_dir / f"robot_infos_{split}.pkl", "wb") as file:
            pickle.dump({"infos": all_infos[split], "metadata": metadata}, file)
    multi_summary = {
        "num_scenes_requested": len(scenes),
        "num_scenes_complete": len(summaries),
        "scenes": scenes,
        "info_counts": {split: len(values) for split, values in all_infos.items()},
        "scene_summaries": summaries,
        "failures": failures,
    }
    summary_temp = root_output_dir / "multi_scene_summary.json.tmp"
    summary_temp.write_text(json.dumps(multi_summary, indent=2), encoding="utf-8")
    summary_temp.replace(root_output_dir / "multi_scene_summary.json")
    if failures:
        raise RuntimeError(
            f"{len(failures)} Replica scene(s) failed; see "
            f"{root_output_dir / 'multi_scene_summary.json'}"
        )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render original Replica v1 PTex scenes with Habitat-Sim 0.2.2 and "
            "generate RGB, Z-depth, base-frame point clouds, and six-channel BEV labels."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to original Replica replica.scene_dataset_config.json",
    )
    parser.add_argument("--scene", default="office_1")
    parser.add_argument("--scenes", nargs="+")
    parser.add_argument("--scenes-file")
    parser.add_argument("--split-file", help="JSON with train/val/test scene lists")
    parser.add_argument("--output-dir", default="data/original_replica_robot")
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--num-sweeps", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-version-mismatch", action="store_true")

    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--hfov", type=float, default=120.0)
    parser.add_argument("--zfar", type=float, default=8.0)
    parser.add_argument("--camera-height", type=float, default=1.0)
    parser.add_argument("--camera-pitch-deg", type=float, default=0.0)
    parser.add_argument("--agent-height", type=float, default=1.0)
    parser.add_argument("--agent-radius", type=float, default=0.36)
    parser.add_argument(
        "--agent-max-climb",
        type=float,
        default=0.20,
        help=(
            "Recast mesh-generation tolerance in meters. Runtime stair safety is "
            "controlled separately by --max-floor-height-delta."
        ),
    )
    parser.add_argument(
        "--agent-max-slope",
        type=float,
        default=45.0,
        help=(
            "Recast triangle-slope tolerance in degrees. Runtime stair safety is "
            "controlled separately by --max-floor-height-delta."
        ),
    )
    parser.add_argument(
        "--enable-stair-filter",
        action="store_true",
        help=(
            "Enable the additional local floor-height check for sampled points and "
            "trajectory steps. Disabled by default; navmesh collision remains active."
        ),
    )
    parser.add_argument("--stair-check-radius", type=float, default=0.50)
    parser.add_argument("--max-floor-height-delta", type=float, default=0.03)
    parser.add_argument("--safe-point-max-tries", type=int, default=1000)

    parser.add_argument("--xbound", type=float, nargs=3, default=[0.0, 3.0, 0.02])
    parser.add_argument("--ybound", type=float, nargs=3, default=[-1.5, 1.5, 0.02])
    parser.add_argument("--min-obstacle-height", type=float, default=0.035)
    parser.add_argument("--max-obstacle-height", type=float, default=1.05)

    parser.add_argument("--max-depth", type=float, default=4.0)
    parser.add_argument("--depth-stride", type=int, default=4)
    parser.add_argument("--max-points", type=int, default=20000)
    parser.add_argument("--gt-width", type=int, default=320)
    parser.add_argument("--gt-height", type=int, default=240)
    parser.add_argument("--gt-hfov", type=float, default=100.0)
    parser.add_argument("--gt-depth-stride", type=int, default=4)
    parser.add_argument("--gt-max-points-per-view", type=int, default=20000)
    parser.add_argument("--gt-multiview", dest="gt_multiview", action="store_true", default=True)
    parser.add_argument("--disable-gt-multiview", dest="gt_multiview", action="store_false")
    parser.add_argument("--save-visualization", action="store_true")
    parser.add_argument("--save-ply", action="store_true", help=argparse.SUPPRESS)

    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument("--turn-angle", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timestamp-start", type=int, default=1_000_000)
    parser.add_argument("--timestamp-step", type=int, default=100_000)
    parser.add_argument("--scene-timestamp-stride", type=int, default=10_000_000)

    parser.set_defaults(semantic_sensor=True)
    parser.add_argument("--enable-physics", action="store_true")
    parser.add_argument("--disable-physics", action="store_true")
    parser.add_argument("--physics-config", default="data/default.physics_config.json")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--recompute-navmesh", action="store_true")
    parser.add_argument(
        "--navmesh-cell-size",
        type=float,
        default=0.05,
        help="Horizontal Recast voxel size in meters.",
    )
    parser.add_argument(
        "--navmesh-cell-height",
        type=float,
        default=0.20,
        help=(
            "Vertical Recast voxel size in meters. The 0.20 m default matches the "
            "stable Habitat-Sim 0.2.2 Replica navmesh settings."
        ),
    )
    parser.add_argument("--navmesh-include-static-objects", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.num_frames <= 0:
        raise ValueError("--num-frames must be positive")
    if args.depth_stride <= 0:
        raise ValueError("--depth-stride must be positive")
    if args.gt_depth_stride <= 0:
        raise ValueError("--gt-depth-stride must be positive")
    if min(args.width, args.height, args.gt_width, args.gt_height) <= 0:
        raise ValueError("All sensor resolutions must be positive")
    if not (0.0 < args.hfov < 180.0 and 0.0 < args.gt_hfov < 180.0):
        raise ValueError("Camera HFOV values must be between 0 and 180 degrees")
    if args.agent_max_climb < 0:
        raise ValueError("--agent-max-climb must be non-negative")
    if args.agent_max_slope < 0:
        raise ValueError("--agent-max-slope must be non-negative")
    if args.navmesh_cell_size <= 0.0 or args.navmesh_cell_height <= 0.0:
        raise ValueError("Navmesh cell size and cell height must be positive")
    if 0.0 < args.agent_max_climb < args.navmesh_cell_height:
        raise ValueError(
            "--agent-max-climb is smaller than --navmesh-cell-height and would be "
            "quantized to zero by Recast; decrease --navmesh-cell-height"
        )
    if args.stair_check_radius <= 0:
        raise ValueError("--stair-check-radius must be positive")
    if args.max_floor_height_delta < 0:
        raise ValueError("--max-floor-height-delta must be non-negative")
    if args.safe_point_max_tries <= 0:
        raise ValueError("--safe-point-max-tries must be positive")
    if args.enable_physics and args.disable_physics:
        raise ValueError("--enable-physics and --disable-physics are mutually exclusive")
    if args.save_ply:
        args.save_visualization = True
    generate_all(args)


if __name__ == "__main__":
    main()


# conda activate habitat022

# nohup env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
# python generate_mydata/robot_bev_closed_loop.py \
#   --dataset "$DATASET" \
#   --scenes-file generate_mydata/replica_scenes.txt \
#   --split-file generate_mydata/replica_splits.example.json \
#   --output-dir "$OUT" \
#   --num-frames 1000 \
#   --num-sweeps 10 \
#   --gpu-id 0 \
#   --disable-physics \
#   --recompute-navmesh \
#   > "${OUT}.log" 2>&1 &
