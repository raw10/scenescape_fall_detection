"""Align a MapAnything mesh and its camera together for SceneScape."""

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def validate_camera(pose, intrinsics, width, height):
  translation = np.asarray(pose["translation"], dtype=float)
  quaternion = np.asarray(pose["rotation"], dtype=float)
  matrix = np.asarray(intrinsics, dtype=float)
  if translation.shape != (3,) or quaternion.shape != (4,) or not np.isfinite(translation).all() or not np.isfinite(quaternion).all():
    raise ValueError("Camera pose must contain finite translation[3] and xyzw quaternion[4]")
  if not np.isclose(np.linalg.norm(quaternion), 1, atol=1e-4):
    raise ValueError("Camera quaternion must be unit length")
  if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or min(matrix[0, 0], matrix[1, 1]) <= 1:
    raise ValueError("Intrinsics must be a finite 3x3 matrix with non-placeholder focal lengths")
  if not np.allclose(matrix[2], [0, 0, 1]) or not np.allclose([matrix[0, 1], matrix[1, 0]], 0):
    raise ValueError("Only zero-skew pinhole intrinsics are supported")
  if not np.isfinite([width, height]).all() or width <= 0 or height <= 0 or not 0 <= matrix[0, 2] <= width or not 0 <= matrix[1, 2] <= height:
    raise ValueError("Intrinsics principal point must be within the original image dimensions")
  return translation, quaternion, matrix


def transform_camera(pose, intrinsics, width, height, transform):
  translation, quaternion, matrix = validate_camera(pose, intrinsics, width, height)
  transform = np.asarray(transform, dtype=float)
  if transform.shape != (4, 4) or not np.isfinite(transform).all() or not np.allclose(transform[3], [0, 0, 0, 1]):
    raise ValueError("Camera transform must be a finite affine 4x4 matrix")
  linear = transform[:3, :3]
  scale = np.cbrt(np.linalg.det(linear))
  if scale <= 0 or not np.allclose((linear / scale).T @ (linear / scale), np.eye(3)):
    raise ValueError("Camera transform must use positive uniform scale and proper rotation")
  camera_translation = linear @ translation + transform[:3, 3]
  if camera_translation[2] <= 1e-6:
    raise ValueError("Aligned camera must lie above the selected floor")
  camera_rotation = (linear / scale) @ Rotation.from_quat(quaternion).as_matrix()
  return {
    "translation": camera_translation.tolist(),
    "rotation": Rotation.from_matrix(camera_rotation).as_quat().tolist(),
    "scale": [1.0, 1.0, 1.0],
    "intrinsics": {"fx": float(matrix[0, 0]), "fy": float(matrix[1, 1]),
                   "cx": float(matrix[0, 2]), "cy": float(matrix[1, 2])},
    "distortion": {"k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0, "k3": 0.0},
    "resolution": [width, height],
  }


def align_geometry(scene, pose, intrinsics, width, height, meters_per_unit=1.0, floor_plane=None):
  scale = float(meters_per_unit)
  if not np.isfinite(scale) or scale <= 0:
    raise ValueError("meters_per_unit must be finite and positive")
  translation, _, _ = validate_camera(pose, intrinsics, width, height)
  mesh = scene.to_geometry() if isinstance(scene, trimesh.Scene) else scene.copy()
  if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) < 3 or len(mesh.faces) == 0:
    raise ValueError("Alignment requires a nonempty triangle mesh")
  if not np.isfinite(mesh.vertices).all():
    raise ValueError("Mesh contains nonfinite vertices")
  if floor_plane is None:
    to_box, extents = trimesh.bounds.oriented_bounds(mesh)
    from_box = np.linalg.inv(to_box)
    axis = int(np.argmin(extents))
    normal = from_box[:3, axis]
    if np.dot(normal, translation - from_box[:3, 3]) < 0:
      normal = -normal
    floor_offset = -float(np.min(mesh.vertices @ normal))
    method = "largest-obb-face-toward-camera"
  else:
    plane = np.asarray(floor_plane, dtype=float)
    if plane.shape != (4,) or not np.isfinite(plane).all() or np.linalg.norm(plane[:3]) < 1e-8:
      raise ValueError("Floor plane must contain finite nx, ny, nz, d with a nonzero normal")
    normal = plane[:3] / np.linalg.norm(plane[:3])
    floor_offset = float(plane[3] / np.linalg.norm(plane[:3]))
    if np.dot(normal, translation) + floor_offset <= 0:
      raise ValueError("Floor normal must point toward the camera above the floor")
    method = "explicit-plane-in-baked-glb-world"
  transform = trimesh.geometry.align_vectors(normal, [0, 0, 1])
  rotation = transform[:3, :3].copy()
  if not np.isclose(np.linalg.det(rotation), 1):
    raise ValueError("Alignment must use a proper rotation, not a reflection")
  transform[:3, :3] *= scale
  rotated = mesh.vertices @ transform[:3, :3].T
  transform[:3, 3] = [-rotated[:, 0].min(), -rotated[:, 1].min(), scale * floor_offset]
  camera = transform_camera(pose, intrinsics, width, height, transform)
  mesh.apply_transform(transform)
  return mesh, camera, {
    "mesh_world_to_scene": transform.tolist(), "meters_per_unit": scale,
    "floor_method": method, "floor_plane": [*normal.tolist(), floor_offset],
    "pose_convention": "xyzw camera-to-world, already in graph-transformed GLB frame",
    "quality": "best_effort", "bounds": mesh.bounds.tolist(),
  }