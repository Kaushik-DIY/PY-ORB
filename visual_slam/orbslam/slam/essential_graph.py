"""
Essential-graph correction utilities for loop closing.
This module builds and optimizes the pose graph used to distribute RGB-D
scale-fixed SE3 loop corrections across the map.
monocular Sim3 parity is not claimed by this implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import g2o
import numpy as np

from visual_slam.orbslam.slam.config_parameters import Parameters


# Store the outcome and graph statistics of one essential-graph optimization.
@dataclass
class EssentialGraphResult:
    success: bool
    before_error: float
    after_error: float
    corrected_keyframes: int
    message: str = ""
    graph_vertices: int = 0
    graph_edges: int = 0
    spanning_tree_edges: int = 0
    covisibility_edges: int = 0
    loop_edges: int = 0
    corrected_points: int = 0
    optimizer_iterations: int = 0
    edge_kinds: dict[str, int] = field(default_factory=dict)
    edge_weights: dict[str, list[float]] = field(default_factory=dict)


# Build and optimize the pose graph used to distribute loop corrections.
class EssentialGraph:
    def __init__(
        self,
        map_object=None,
        keyframes_to_correct=None,
        loop_keyframe=None,
        current_keyframe=None,
        non_corrected_poses=None,
        corrected_poses=None,
        loop_connections=None,
        min_covisibility_weight: int = 100,
    ):
        self.map = map_object
        self.keyframes_to_correct = [
            kf for kf in (keyframes_to_correct or []) if kf is not None and not _is_bad_keyframe(kf)
        ]
        self.loop_keyframe = loop_keyframe
        self.current_keyframe = current_keyframe
        self.non_corrected_poses = {
            kf: _as_matrix(T) for kf, T in (non_corrected_poses or {}).items() if kf is not None
        }
        self.corrected_poses = {
            kf: _as_matrix(T) for kf, T in (corrected_poses or {}).items() if kf is not None
        }
        self.loop_connections = {
            kf: list(connections) for kf, connections in (loop_connections or {}).items() if kf is not None
        }
        self.min_covisibility_weight = int(min_covisibility_weight)

        self.vertices: list = []
        self.edges: list[tuple[str, object, object, np.ndarray]] = []
        self.edge_information: list[np.ndarray] = []
        self.edge_weights: dict[str, list[float]] = {
            "spanning_tree": [],
            "covisibility": [],
            "loop": [],
        }
        self.edge_kinds = {
            "spanning_tree": 0,
            "covisibility": 0,
            "loop": 0,
        }

    def build_from_map(self) -> "EssentialGraph":
        keyframes = []
        if self.map is not None:
            try:
                keyframes.extend(_as_list(self.map.get_keyframes()))
            except Exception:
                pass

        keyframes.extend(self.keyframes_to_correct)
        for keyframe in [self.loop_keyframe, self.current_keyframe]:
            if keyframe is not None:
                keyframes.append(keyframe)
        for keyframe in list(keyframes):
            try:
                parent = keyframe.get_parent()
            except Exception:
                parent = None
            if parent is not None:
                keyframes.append(parent)
            try:
                keyframes.extend(keyframe.get_connected_keyframes())
            except Exception:
                pass
            try:
                keyframes.extend(keyframe.get_loop_edges())
            except Exception:
                pass

        seen = set()
        self.vertices = []
        for keyframe in keyframes:
            if keyframe is None or keyframe in seen or _is_bad_keyframe(keyframe):
                continue
            self.vertices.append(keyframe)
            seen.add(keyframe)

        self.add_loop_edges()
        self.add_spanning_tree_edges()
        self.add_covisibility_edges()
        return self

    def add_spanning_tree_edges(self) -> None:
        for keyframe in self.vertices:
            parent = keyframe.get_parent() if hasattr(keyframe, "get_parent") else None
            if parent is None or parent not in self.vertices or _is_bad_keyframe(parent):
                continue
            self._add_edge("spanning_tree", keyframe, parent, use_corrected=False)

    def add_covisibility_edges(self) -> None:
        inserted = _edge_pair_set(self.edges)
        for keyframe in self.vertices:
            for connected in keyframe.get_covisible_by_weight(self.min_covisibility_weight):
                if connected not in self.vertices or _is_bad_keyframe(connected):
                    continue
                if connected is keyframe.get_parent() or keyframe.has_child(connected):
                    continue
                pair = _edge_pair(keyframe, connected)
                if pair in inserted:
                    continue
                self._add_edge("covisibility", keyframe, connected, use_corrected=False)
                inserted.add(pair)

    def add_loop_edges(self) -> None:
        inserted = set()
        for keyframe, connections in self.loop_connections.items():
            if keyframe not in self.vertices:
                continue
            for connected in connections:
                if connected not in self.vertices or _is_bad_keyframe(connected):
                    continue
                pair = _edge_pair(keyframe, connected)
                if pair in inserted:
                    continue
                self._add_edge("loop", keyframe, connected, use_corrected=True)
                inserted.add(pair)

        for keyframe in self.vertices:
            for connected in keyframe.get_loop_edges():
                if connected not in self.vertices or _is_bad_keyframe(connected):
                    continue
                pair = _edge_pair(keyframe, connected)
                if pair in inserted:
                    continue
                self._add_edge("loop", keyframe, connected, use_corrected=False)
                inserted.add(pair)

    def optimize(self, iterations: int = 20) -> EssentialGraphResult:
        if len(self.vertices) == 0:
            return EssentialGraphResult(False, float("inf"), float("inf"), 0, "no keyframe vertices")
        if len(self.edges) == 0:
            return EssentialGraphResult(False, float("inf"), float("inf"), 0, "no graph edges")

        before_error = _camera_center_error(self.loop_keyframe, self.current_keyframe)
        old_poses = {kf: _pose_of(kf) for kf in self.vertices}

        try:
            optimizer = _make_se3_optimizer()
            pose_vertices = {}
            fixed_keyframe = self._fixed_keyframe()

            for keyframe in self.vertices:
                Tcw = self.corrected_poses.get(keyframe, old_poses[keyframe])
                vertex = g2o.VertexSE3Expmap()
                vertex.set_id(int(keyframe.kid))
                vertex.set_estimate(_se3quat(Tcw))
                vertex.set_fixed(keyframe is fixed_keyframe)
                optimizer.add_vertex(vertex)
                pose_vertices[keyframe] = vertex

            for edge_id, ((_, keyframe, connected, measurement), information) in enumerate(
                zip(self.edges, self.edge_information)
            ):
                edge = g2o.EdgeSE3Expmap()
                edge.set_id(edge_id)
                edge.set_vertex(0, pose_vertices[keyframe])
                edge.set_vertex(1, pose_vertices[connected])
                edge.set_measurement(_se3quat(measurement))
                edge.set_information(information)
                optimizer.add_edge(edge)

            optimizer.initialize_optimization()
            optimizer.compute_active_errors()
            ret = int(optimizer.optimize(int(iterations)))
            optimizer.compute_active_errors()
        except Exception as exc:
            return EssentialGraphResult(False, before_error, float("inf"), 0, f"g2o failed: {exc}")

        if ret <= 0:
            return EssentialGraphResult(False, before_error, float("inf"), 0, "optimizer reported failure")

        optimized_poses = {}
        for keyframe, vertex in pose_vertices.items():
            Tcw = _matrix_from_se3(vertex.estimate())
            optimized_poses[keyframe] = Tcw

        ok, message = self._validate_optimized_poses(old_poses, optimized_poses)
        if not ok:
            return EssentialGraphResult(False, before_error, float("inf"), 0, message)

        corrected_points = self.apply_corrections(old_poses, optimized_poses)
        after_error = _camera_center_error(self.loop_keyframe, self.current_keyframe)

        return EssentialGraphResult(
            success=np.isfinite(after_error),
            before_error=before_error,
            after_error=after_error,
            corrected_keyframes=len(optimized_poses),
            graph_vertices=len(self.vertices),
            graph_edges=len(self.edges),
            spanning_tree_edges=self.edge_kinds["spanning_tree"],
            covisibility_edges=self.edge_kinds["covisibility"],
            loop_edges=self.edge_kinds["loop"],
            corrected_points=corrected_points,
            optimizer_iterations=ret,
            edge_kinds=dict(self.edge_kinds),
            edge_weights={kind: list(weights) for kind, weights in self.edge_weights.items()},
        )

    def apply_corrections(self, old_poses: dict, optimized_poses: dict) -> int:
        corrected = 0
        corrected_points = set()

        for keyframe, Tcw in optimized_poses.items():
            keyframe.update_pose(g2o.Isometry3d(Tcw))

        for keyframe in self.keyframes_to_correct:
            old_Tcw = old_poses.get(keyframe)
            new_Tcw = optimized_poses.get(keyframe)
            if old_Tcw is None or new_Tcw is None:
                continue

            try:
                new_Twc = np.linalg.inv(new_Tcw)
            except np.linalg.LinAlgError:
                continue

            for point in keyframe.get_matched_good_points():
                if point is None or point in corrected_points or point.is_bad():
                    continue
                p_old = point.get_position()
                p_new = (new_Twc @ old_Tcw @ _homogeneous(p_old))[:3]
                if not np.all(np.isfinite(p_new)):
                    continue
                point.update_position(p_new)
                point.update_normal_and_depth()
                point.corrected_by_kf = getattr(self.current_keyframe, "kid", 0)
                point.corrected_reference = getattr(keyframe, "kid", 0)
                corrected_points.add(point)
                corrected += 1

        for keyframe in self.vertices:
            keyframe.update_connections()

        return corrected

    def _fixed_keyframe(self):
        for keyframe in self.vertices:
            if getattr(keyframe, "kid", None) == 0:
                return keyframe
        if self.loop_keyframe in self.vertices:
            return self.loop_keyframe
        return self.vertices[0]

    def _add_edge(self, kind: str, keyframe, connected, use_corrected: bool) -> None:
        poses = self.corrected_poses if use_corrected else self.non_corrected_poses
        Tiw = poses.get(keyframe, _pose_of(keyframe))
        Tjw = poses.get(connected, _pose_of(connected))
        if not (_is_finite_se3(Tiw) and _is_finite_se3(Tjw)):
            return
        try:
            Tji = Tjw @ np.linalg.inv(Tiw)
        except np.linalg.LinAlgError:
            return
        if not _is_finite_se3(Tji):
            return
        information = self._information_for_edge(kind, keyframe, connected)
        self.edges.append((kind, keyframe, connected, Tji))
        self.edge_information.append(information)
        self.edge_weights[kind].append(float(information[0, 0]))
        self.edge_kinds[kind] += 1

    @staticmethod
    def _information_for_edge(kind: str, keyframe, connected) -> np.ndarray:
        if kind == "loop":
            weight = Parameters.kEssentialGraphLoopEdgeWeight
        elif kind == "covisibility":
            try:
                cov_weight = max(keyframe.get_weight(connected), connected.get_weight(keyframe))
            except Exception:
                cov_weight = 0
            weight = float(cov_weight) * Parameters.kEssentialGraphCovisibilityWeightScale
            weight = min(
                max(weight, Parameters.kEssentialGraphCovisibilityWeightMin),
                Parameters.kEssentialGraphCovisibilityWeightMax,
            )
        else:
            weight = Parameters.kEssentialGraphSpanningTreeWeight
        return np.eye(6, dtype=np.float64) * float(weight)

    @staticmethod
    def _validate_optimized_poses(old_poses: dict, optimized_poses: dict) -> tuple[bool, str]:
        max_jump = 50.0
        for keyframe, Tcw in optimized_poses.items():
            if not _is_finite_se3(Tcw):
                return False, f"invalid optimized pose for keyframe {getattr(keyframe, 'kid', None)}"
            R = Tcw[:3, :3]
            if not np.allclose(R.T @ R, np.eye(3), atol=1e-4):
                return False, "optimized rotation is not orthonormal enough"
            det = float(np.linalg.det(R))
            if not np.isfinite(det) or abs(det - 1.0) > 1e-3:
                return False, "optimized rotation determinant is invalid"
            old = old_poses.get(keyframe)
            if old is not None:
                jump = float(np.linalg.norm(Tcw[:3, 3] - old[:3, 3]))
                if not np.isfinite(jump) or jump > max_jump:
                    return False, "optimized translation jump is unreasonable"
        return True, ""


def optimize_essential_graph_se3(
    keyframes_to_correct,
    loop_keyframe,
    current_keyframe,
    correction_T,
    map_object=None,
    corrected_poses=None,
    non_corrected_poses=None,
    loop_connections=None,
) -> EssentialGraphResult:
    correction_T = np.asarray(correction_T, dtype=np.float64).reshape(4, 4)
    if not _is_finite_se3(correction_T):
        return EssentialGraphResult(False, float("inf"), float("inf"), 0, "non-finite correction")

    keyframes_to_correct = [
        kf for kf in _as_list(keyframes_to_correct) if kf is not None and not _is_bad_keyframe(kf)
    ]
    if not keyframes_to_correct:
        return EssentialGraphResult(False, float("inf"), float("inf"), 0, "no keyframes to correct")

    try:
        correction_inv = np.linalg.inv(correction_T)
    except np.linalg.LinAlgError:
        return EssentialGraphResult(False, float("inf"), float("inf"), 0, "singular correction")

    if corrected_poses is None:
        corrected_poses = {}
        for keyframe in keyframes_to_correct:
            corrected_poses[keyframe] = _pose_of(keyframe) @ correction_inv
    if non_corrected_poses is None:
        non_corrected_poses = {kf: _pose_of(kf) for kf in keyframes_to_correct}

    graph = EssentialGraph(
        map_object=map_object,
        keyframes_to_correct=keyframes_to_correct,
        loop_keyframe=loop_keyframe,
        current_keyframe=current_keyframe,
        non_corrected_poses=non_corrected_poses,
        corrected_poses=corrected_poses,
        loop_connections=loop_connections,
    ).build_from_map()
    return graph.optimize()


def apply_correction_to_map_points(keyframes_to_correct, correction_T) -> int:
    correction_T = np.asarray(correction_T, dtype=np.float64).reshape(4, 4)
    if not _is_finite_se3(correction_T):
        return 0

    corrected_points = set()
    for keyframe in _as_list(keyframes_to_correct):
        if keyframe is None or _is_bad_keyframe(keyframe):
            continue
        for point in keyframe.get_matched_good_points():
            if point is None or point in corrected_points or point.is_bad():
                continue
            position = point.get_position()
            corrected = (correction_T @ _homogeneous(position))[:3]
            if not np.all(np.isfinite(corrected)):
                continue
            point.update_position(corrected)
            point.update_normal_and_depth()
            corrected_points.add(point)

    return len(corrected_points)


def _make_se3_optimizer():
    optimizer = g2o.SparseOptimizer()
    linear_solver = g2o.LinearSolverEigenSE3()
    solver = g2o.BlockSolverSE3(linear_solver)
    algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer.set_algorithm(algorithm)
    optimizer.set_verbose(False)
    return optimizer


def _se3quat(Tcw: np.ndarray):
    Tcw = _as_matrix(Tcw)
    return g2o.SE3Quat(Tcw[:3, :3].copy(), Tcw[:3, 3].copy())


def _matrix_from_se3(se3) -> np.ndarray:
    if hasattr(se3, "to_homogeneous_matrix"):
        return np.asarray(se3.to_homogeneous_matrix(), dtype=np.float64).reshape(4, 4)
    if hasattr(se3, "matrix"):
        return np.asarray(se3.matrix(), dtype=np.float64).reshape(4, 4)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = se3.rotation().matrix()
    T[:3, 3] = se3.translation()
    return T


def _pose_of(keyframe) -> np.ndarray:
    return np.asarray(keyframe.Tcw(), dtype=np.float64).reshape(4, 4)


def _as_matrix(value) -> np.ndarray:
    if hasattr(value, "matrix"):
        value = value.matrix()
    elif hasattr(value, "to_homogeneous_matrix"):
        value = value.to_homogeneous_matrix()
    return np.asarray(value, dtype=np.float64).reshape(4, 4)


def _is_finite_se3(T: np.ndarray) -> bool:
    T = np.asarray(T, dtype=np.float64)
    return T.shape == (4, 4) and np.all(np.isfinite(T))


def _camera_center_error(keyframe_a, keyframe_b) -> float:
    if keyframe_a is None or keyframe_b is None:
        return float("inf")
    try:
        return float(np.linalg.norm(keyframe_a.Ow().reshape(3) - keyframe_b.Ow().reshape(3)))
    except Exception:
        return float("inf")


def _homogeneous(point: np.ndarray) -> np.ndarray:
    out = np.ones(4, dtype=np.float64)
    out[:3] = np.asarray(point, dtype=np.float64).reshape(3)
    return out


def _as_list(values) -> list:
    if values is None:
        return []
    if hasattr(values, "to_list"):
        return values.to_list()
    return list(values)


def _is_bad_keyframe(keyframe) -> bool:
    return hasattr(keyframe, "is_bad") and keyframe.is_bad()


def _edge_pair(kf1, kf2):
    return tuple(sorted((int(kf1.kid), int(kf2.kid))))


def _edge_pair_set(edges):
    return {_edge_pair(kf1, kf2) for _, kf1, kf2, _ in edges}
