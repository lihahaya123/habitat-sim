import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import habitat_sim

from habitat_sim.utils.common import quat_from_angle_axis

from generate_mydata import robot_bev_closed_loop as generator


class ReplicaGeneratorGeometryTest(unittest.TestCase):
    def test_point_indices_handle_upper_boundary_float32_roundoff(self):
        y = np.nextafter(np.float32(1.5), np.float32(-np.inf))
        points = np.array([[1.0, y, 0.0]], dtype=np.float32)

        rows, cols = generator.point_indices(
            points,
            (0.0, 3.0, 0.02),
            (-1.5, 1.5, 0.02),
        )

        np.testing.assert_array_equal(rows, [50])
        np.testing.assert_array_equal(cols, [149])

    def test_observed_rays_handle_upper_boundary_float32_roundoff(self):
        valid = np.zeros((150, 150), dtype=np.uint8)
        y = np.nextafter(np.float32(1.5), np.float32(-np.inf))
        points = np.array([[1.0, y, 0.0]], dtype=np.float32)

        generator.mark_observed_rays(
            valid,
            points,
            np.array([0.0, 0.0, 0.0], dtype=np.float32),
            (0.0, 3.0, 0.02),
            (-1.5, 1.5, 0.02),
        )

        self.assertEqual(valid[50, 149], 1)

    def test_navmesh_topdown_handles_upper_boundary_float32_roundoff(self):
        cache = generator.NavmeshTopdown(
            grid=np.ones((150, 150), dtype=np.uint8),
            min_x=-1.5,
            min_z=0.0,
            meters_per_pixel=0.02,
        )
        x = np.nextafter(np.float32(1.5), np.float32(-np.inf))
        world = np.array([[x, 0.0, 1.0]], dtype=np.float32)

        sampled = generator.sample_navmesh_topdown(cache, world)

        np.testing.assert_array_equal(sampled, [1])

    def test_disabled_stair_filter_accepts_navmesh_random_point(self):
        class Pathfinder:
            @staticmethod
            def get_random_navigable_point():
                return np.array([1.0, 0.2, 3.0], dtype=np.float32)

            @staticmethod
            def is_navigable(*args, **kwargs):
                raise AssertionError("disabled stair filter must not run extra checks")

        class Simulator:
            pathfinder = Pathfinder()

        args = SimpleNamespace(
            enable_stair_filter=False,
            safe_point_max_tries=10,
            stair_check_radius=0.5,
            max_floor_height_delta=0.03,
        )

        point = generator.sample_safe_navigable_point(Simulator(), args)

        np.testing.assert_allclose(point, [1.0, 0.2, 3.0])

    def test_default_navmesh_build_uses_stable_habitat_022_recast_settings(self):
        args = generator.make_parser().parse_args(["--dataset", "replica.json"])

        self.assertEqual(args.navmesh_cell_size, 0.05)
        self.assertEqual(args.navmesh_cell_height, 0.20)
        self.assertEqual(args.agent_max_climb, 0.20)
        self.assertEqual(args.agent_max_slope, 45.0)
        self.assertFalse(args.enable_stair_filter)

    def test_navmesh_settings_support_habitat_sim_022(self):
        class LegacyNavMeshSettings:
            def set_defaults(self):
                self.defaults_loaded = True

        settings = LegacyNavMeshSettings()
        args = SimpleNamespace(
            navmesh_cell_size=0.05,
            navmesh_cell_height=0.01,
            agent_height=1.0,
            agent_radius=0.36,
            agent_max_climb=0.03,
            agent_max_slope=20.0,
            navmesh_include_static_objects=False,
        )

        generator.configure_navmesh_settings(settings, args)

        self.assertTrue(settings.defaults_loaded)
        self.assertEqual(settings.cell_size, 0.05)
        self.assertEqual(settings.cell_height, 0.01)
        self.assertEqual(settings.agent_height, 1.0)
        self.assertEqual(settings.agent_radius, 0.36)
        self.assertFalse(hasattr(settings, "include_static_objects"))

    def test_optical_camera_to_base_matches_robot_axes(self):
        habitat_matrix = generator.camera_to_base_matrix(0.18, 0.0)
        optical_matrix = generator.camera_optical_to_base_matrix(habitat_matrix)

        right = optical_matrix @ np.array([1.0, 0.0, 0.0, 1.0])
        down = optical_matrix @ np.array([0.0, 1.0, 0.0, 1.0])
        forward = optical_matrix @ np.array([0.0, 0.0, 1.0, 1.0])

        np.testing.assert_allclose(right[:3], [0.0, -1.0, 0.18], atol=1e-6)
        np.testing.assert_allclose(down[:3], [0.0, 0.0, -0.82], atol=1e-6)
        np.testing.assert_allclose(forward[:3], [1.0, 0.0, 0.18], atol=1e-6)

    def test_depth_unprojection_uses_habitat_z_depth(self):
        intrinsic = generator.make_camera_intrinsic(3, 3, 90.0)
        depth = np.zeros((3, 3), dtype=np.float32)
        depth[1, 1] = 2.0
        points, semantic_ids = generator.depth_to_points(
            depth=depth,
            intrinsic=intrinsic,
            t_base_camera_habitat=generator.camera_to_base_matrix(0.18, 0.0),
            max_depth=4.0,
            stride=1,
            max_points=100,
        )

        self.assertIsNone(semantic_ids)
        np.testing.assert_allclose(points[0, :3], [2.0, 0.0, 0.18], atol=1e-6)

    def test_semantic_fallback_is_other_not_obstacle(self):
        self.assertEqual(generator.semantic_category_to_map_class("chair"), "other")
        self.assertEqual(generator.semantic_category_to_map_class("floor mat"), "carpet")
        self.assertEqual(generator.semantic_category_to_map_class("wall"), "wall")

    def test_multihot_other_and_obstacle_share_a_cell(self):
        class Pathfinder:
            @staticmethod
            def is_navigable(point, max_y_delta=0.5):
                return False

        class Simulator:
            pathfinder = Pathfinder()

        state = habitat_sim.AgentState()
        state.position = np.zeros(3, dtype=np.float32)
        state.rotation = quat_from_angle_axis(0.0, np.array([0.0, 1.0, 0.0]))
        points = np.array([[1.2, 0.0, 0.2, 0.0, 0.0]], dtype=np.float32)
        mask, valid = generator.make_bev_mask(
            Simulator(),
            state,
            [(points, np.array([5]), np.zeros(3, dtype=np.float32))],
            {5: "other"},
            (0.0, 2.0, 1.0),
            (-1.0, 1.0, 1.0),
            0.02,
            0.8,
        )

        row, col = 1, 1
        self.assertEqual(mask[generator.MAP_CLASSES.index("other"), row, col], 1)
        self.assertEqual(mask[generator.MAP_CLASSES.index("obstacle"), row, col], 1)
        self.assertEqual(valid[row, col], 1)
        self.assertEqual(mask[generator.MAP_CLASSES.index("unknown"), row, col], 0)

    def test_history_sweep_transforms_into_current_base(self):
        identity = np.eye(4, dtype=np.float32)
        current = identity.copy()
        current[0, 3] = 1.0
        common = {
            "image_path": Path("images/000000.png"),
            "depth_path": Path("depths/000000.png"),
            "semantic_path": Path("semantics/000000.png"),
            "points_path": Path("points/000000.bin"),
            "bev_mask_path": Path("bev_masks/000000.npy"),
            "bev_valid_mask_path": Path("bev_valid_masks/000000.npy"),
        }
        first = dict(common, timestamp=0, T_map_base=identity)
        second = dict(common, timestamp=1, T_map_base=current)
        second["points_path"] = Path("points/000001.bin")
        infos = generator.build_infos(
            [first, second],
            Path("office_1"),
            1,
            generator.make_camera_intrinsic(3, 3, 90.0),
            generator.camera_to_base_matrix(0.18, 0.0),
        )[0]["infos"]

        np.testing.assert_allclose(
            infos[1]["sweeps"][0]["sensor2lidar_translation"],
            [-1.0, 0.0, 0.0],
            atol=1e-6,
        )
        self.assertEqual(infos[0]["token"], "office_1_000000")

    def test_stair_check_does_not_reject_a_flat_point_near_walls(self):
        class Pathfinder:
            @staticmethod
            def is_navigable(point, max_y_delta=0.5):
                return abs(float(point[0])) < 1e-6 and abs(float(point[2])) < 1e-6

            @staticmethod
            def snap_point(point):
                return np.array([point[0], 0.0, point[2]], dtype=np.float32)

        class Simulator:
            pathfinder = Pathfinder()

        self.assertTrue(
            generator.is_floor_level_safe(Simulator(), [0.0, 0.0, 0.0], 0.5, 0.03)
        )

    def test_stair_check_rejects_a_nearby_height_discontinuity(self):
        class Pathfinder:
            @staticmethod
            def is_navigable(point, max_y_delta=0.5):
                return True

            @staticmethod
            def snap_point(point):
                height = 0.1 if float(point[0]) > 0.25 else 0.0
                return np.array([point[0], height, point[2]], dtype=np.float32)

        class Simulator:
            pathfinder = Pathfinder()

        self.assertFalse(
            generator.is_floor_level_safe(Simulator(), [0.0, 0.0, 0.0], 0.5, 0.03)
        )


class ReplicaDatasetValidationTest(unittest.TestCase):
    def make_scene(self, root: Path, descriptor: str = "info_semantic.json") -> Path:
        dataset = root / "replica.scene_dataset_config.json"
        dataset.write_text("{}", encoding="utf-8")
        scene = root / "office_1"
        habitat = scene / "habitat"
        textures = scene / "textures"
        habitat.mkdir(parents=True)
        textures.mkdir()
        (scene / "mesh.ply").write_bytes(b"ply")
        (textures / "parameters.json").write_text("{}", encoding="utf-8")
        (textures / "0-color-ptex.hdr").write_bytes(b"hdr")
        (habitat / "sorted_faces.bin").write_bytes(b"faces")
        (habitat / "mesh_semantic.ply").write_bytes(b"ply")
        (habitat / "info_semantic.json").write_text("{}", encoding="utf-8")
        (habitat / "mesh_semantic.navmesh").write_bytes(b"nav")
        stage = {
            "render_asset": "../mesh.ply",
            "semantic_descriptor_filename": descriptor,
            "semantic_asset": "mesh_semantic.ply",
            "nav_asset": "mesh_semantic.navmesh",
        }
        (habitat / "replica_stage.stage_config.json").write_text(
            json.dumps(stage), encoding="utf-8"
        )
        return dataset

    def test_valid_original_replica_scene_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self.make_scene(Path(directory))
            result = generator.validate_replica_scene(dataset, "office_1")

            self.assertEqual(result.scene_dir.name, "office_1")
            self.assertEqual(result.ptex_atlas_count, 1)

    def test_legacy_txt_semantic_descriptor_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self.make_scene(Path(directory), "info_semantic.txt")
            with self.assertRaisesRegex(RuntimeError, "info_semantic.json"):
                generator.validate_replica_scene(dataset, "office_1")


class ReplicaSplitTest(unittest.TestCase):
    def test_split_file_rejects_scene_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "splits.json"
            path.write_text(
                json.dumps({"train": ["office_1"], "val": ["office_1"]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "more than one split"):
                generator.load_scene_splits(path, ["office_1"])


class ReplicaManifestTest(unittest.TestCase):
    def test_manifest_requires_contiguous_frame_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            generator.append_manifest(path, {"frame_index": 0, "value": "a"})
            generator.append_manifest(path, {"frame_index": 2, "value": "b"})
            with self.assertRaisesRegex(RuntimeError, "Non-contiguous"):
                generator.load_manifest(path)

    def test_atomic_numpy_output_has_no_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.npy"
            expected = np.arange(6, dtype=np.uint8)
            generator.atomic_save_npy(expected, path)
            np.testing.assert_array_equal(np.load(path), expected)
            self.assertFalse(path.with_name(path.name + ".tmp").exists())


if __name__ == "__main__":
    unittest.main()
