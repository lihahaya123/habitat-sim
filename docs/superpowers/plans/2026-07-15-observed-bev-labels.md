# Observed-Region Six-Class BEV Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the learned `unknown` channel with `furniture`, make the external valid mask depend only on the current front RGB-D observation, and prevent schema-v2/v3 resume mixing.

**Architecture:** Split sensor observability from label construction. The same front depth samples produce the saved pseudo-LiDAR, the ray-derived valid mask, and semantic/geometric endpoints; navmesh contributes only the observed `floor` label. Keep six-channel labels and the one-channel valid mask as separate files.

**Tech Stack:** Python 3.8, NumPy, Habitat-Sim 0.2.2, Pillow, unittest/pytest.

## Global Constraints

- Preserve original Replica v1 PTex rendering and Habitat-Sim 0.2.2 compatibility.
- Output labels are `uint8 [6,H,W]` in order `floor, carpet, obstacle, wall, furniture, other`.
- Output valid masks are `uint8 [H,W]` and derive only from current input sensor rays.
- Invalid cells contain zero in all six learned channels; unknown is `1-valid` outside the network.
- Labels remain multi-hot and are trained with sigmoid losses.
- Schema v2 output cannot resume under schema v3.

---

### Task 1: Lock the six-class contract and semantic mapping

**Files:**
- Modify: `tests/test_robot_bev_closed_loop.py`
- Modify: `generate_mydata/robot_bev_closed_loop.py`

**Interfaces:**
- Produces: `MAP_CLASSES`, `FURNITURE_CATEGORIES`, `semantic_category_to_map_class(name) -> Optional[str]`.

- [x] Add failing tests asserting the exact class order, `chair/table/cabinet -> furniture`, `wall-plug -> other`, and `undefined -> None`.

```python
def test_six_learned_map_classes_exclude_unknown(self):
    self.assertEqual(generator.MAP_CLASSES, [
        "floor", "carpet", "obstacle", "wall", "furniture", "other"
    ])

def test_semantic_furniture_and_ignored_category_mapping(self):
    self.assertEqual(generator.semantic_category_to_map_class("chair"), "furniture")
    self.assertEqual(generator.semantic_category_to_map_class("wall-plug"), "other")
    self.assertIsNone(generator.semantic_category_to_map_class("undefined"))
```

- [x] Run the focused `unittest` cases and confirm failure against schema v2.
- [x] Replace `unknown` with `furniture`, add normalized exact category sets, retain `other` as fallback, and update colors/priorities.

```python
MAP_CLASSES = ["floor", "carpet", "obstacle", "wall", "furniture", "other"]
FURNITURE_CATEGORIES = frozenset({"bed", "cabinet", "chair", "desk", "table"})

def semantic_category_to_map_class(category_name: str) -> Optional[str]:
    normalized = normalize_semantic_category(category_name)
    if normalized in IGNORED_SEMANTIC_CATEGORIES:
        return None
    if normalized in FURNITURE_CATEGORIES:
        return "furniture"
    # Exact floor/carpet/wall groups, then other fallback.
```

- [x] Run the focused tests and confirm they pass.

### Task 2: Separate observability from six-channel labels

**Files:**
- Modify: `tests/test_robot_bev_closed_loop.py`
- Modify: `generate_mydata/robot_bev_closed_loop.py`

**Interfaces:**
- Produces: `make_observation_mask(views, xbound, ybound) -> np.ndarray`.
- Produces: `make_bev_labels(sim, state, views, semantic_id_to_class, xbound, ybound, min_obstacle_height, max_obstacle_height, valid_mask, navmesh_topdown=None) -> np.ndarray`.

- [x] Add failing tests proving an all-navigable navmesh cannot make an empty observation valid, a ray does make traversed cells valid, invalid cells have six zero labels, and furniture/obstacle can overlap.

```python
def test_navmesh_does_not_expand_empty_observation(self):
    valid = generator.make_observation_mask([], (0.0, 2.0, 1.0), (-1.0, 1.0, 1.0))
    self.assertFalse(valid.any())

def test_furniture_and_obstacle_share_an_observed_cell(self):
    valid = generator.make_observation_mask([view], xbound, ybound)
    labels = generator.make_bev_labels(
        Simulator(), state, [view], {5: "furniture"}, xbound, ybound,
        0.02, 0.8, valid_mask=valid,
    )
    self.assertEqual(labels[furniture_idx, row, col], 1)
    self.assertEqual(labels[obstacle_idx, row, col], 1)
    self.assertFalse(labels[:, valid == 0].any())
```

- [x] Run the focused tests and confirm the old coupled `make_bev_mask` behavior fails.
- [x] Implement `make_observation_mask` with an all-zero start and existing ray rasterization.

```python
def make_observation_mask(views, xbound, ybound):
    height, width = bev_grid_shape(xbound, ybound)
    valid = np.zeros((height, width), dtype=np.uint8)
    for points, _, origin in views:
        mark_observed_rays(valid, points, origin, xbound, ybound)
    return valid
```

- [x] Implement `make_bev_labels`: intersect navmesh floor with valid, project geometric/semantic endpoints, skip semantic floor as a traversability source, and zero all invalid cells.

```python
floor = sample_navmesh_topdown(navmesh_topdown, world).reshape(height, width)
labels[floor_idx] = floor & valid_mask
# Populate obstacle and semantic endpoints from the same views.
labels *= valid_mask[None]
return labels
```

- [x] Remove the learned unknown assignment and update visualization to accept `valid_mask` separately.
- [x] Run all geometry/semantic tests and confirm they pass.

### Task 3: Align the rendering loop with the observed-region contract

**Files:**
- Modify: `generate_mydata/robot_bev_closed_loop.py`
- Modify: `tests/test_robot_bev_closed_loop.py`

**Interfaces:**
- Consumes: the front `depth_to_points` result and front sensor origin.
- Produces: separate `bev_masks/*.npy` and `bev_valid_masks/*.npy` files.

- [x] Add a configuration test asserting the parser no longer enables multiview GT and the simulator config contains only aligned front sensors.

```python
args = generator.make_parser().parse_args(["--dataset", "replica.json"])
self.assertFalse(hasattr(args, "gt_multiview"))
```

- [x] Remove left/back/right auxiliary GT sensor construction and obsolete GT camera parameters.
- [x] Unproject front depth once with aligned semantic IDs, save those points, build `valid_mask` from the same view, and build six labels from the same points.

```python
points, semantic_ids = depth_to_points(
    depth, intrinsic, front_extrinsic, args.max_depth,
    args.depth_stride, args.max_points, semantic_obs,
)
views = [(points, semantic_ids, front_extrinsic[:3, 3])]
valid_mask = make_observation_mask(views, xbound, ybound)
mask = make_bev_labels(
    sim, state, views, semantic_id_to_class, xbound, ybound,
    args.min_obstacle_height, args.max_obstacle_height,
    valid_mask, navmesh_topdown,
)
```

- [x] Pass `valid_mask` to visualization and preserve manifest/info paths.
- [x] Run parser and dataset-format tests.

### Task 4: Version metadata and reject mixed resume data

**Files:**
- Modify: `generate_mydata/robot_bev_closed_loop.py`
- Modify: `tests/test_robot_bev_closed_loop.py`

**Interfaces:**
- Produces: `GENERATOR_SCHEMA_VERSION = 3` included in generation fingerprints and metadata.

- [x] Add tests proving fingerprints include schema/class mapping and schema v2 metadata is incompatible.

```python
def test_generation_fingerprint_contains_schema_contract(self):
    first = generator.generation_fingerprint(args, "office_1")
    with mock.patch.object(generator, "GENERATOR_SCHEMA_VERSION", 4):
        second = generator.generation_fingerprint(args, "office_1")
    self.assertNotEqual(first, second)
```

- [x] Add the schema constant and static label contract to `generation_fingerprint`.

```python
values["generator_schema_version"] = GENERATOR_SCHEMA_VERSION
values["map_classes"] = MAP_CLASSES
values["semantic_category_groups"] = semantic_category_groups_metadata()
```

- [x] Validate schema before resume fingerprint/split checks and emit a clear new-output-directory error.

```python
if previous_metadata.get("generator_schema_version") != GENERATOR_SCHEMA_VERSION:
    raise RuntimeError("--resume schema mismatch; choose a new output directory")
```

- [x] Rewrite scene/root metadata descriptions for observed masks, furniture, external unknown, and single front sensor.
- [x] Run manifest/fingerprint tests.

### Task 5: Update user documentation and verify

**Files:**
- Modify: `generate_mydata/closed_loop_readme.md`
- Modify: `docs/superpowers/specs/2026-07-14-replica-rgbd-bev-generator-design.md`
- Modify: `docs/superpowers/plans/2026-07-15-observed-bev-labels.md`

**Interfaces:**
- Documents the final schema-v3 disk and loss contract.

- [x] Replace all old unknown/multiview descriptions with the six learned classes plus external mask.
- [x] Document that old output directories cannot resume and show a new output name.
- [x] Run `/home/lihahaya/miniconda3/envs/habitat022/bin/python -m unittest tests.test_robot_bev_closed_loop -v` and confirm all 26 tests pass.
- [x] Run `python -m py_compile generate_mydata/robot_bev_closed_loop.py tests/test_robot_bev_closed_loop.py` and expect no output.
- [x] Run `git diff --check` and expect no output.
- [x] Attempt a one-frame `office_1` smoke test. Preflight passed, but this managed execution sandbox exposes neither `/dev/dxg` nor `/dev/dri`; Habitat renderer creation exits 139 before Python frame generation. Repeat the documented smoke command in a GPU-enabled terminal.
