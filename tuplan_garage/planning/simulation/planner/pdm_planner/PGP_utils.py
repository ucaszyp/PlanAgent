from nuplan.planning.simulation.planner.abstract_planner import PlannerInput, PlannerInitialization

from tuplan_garage.planning.training.preprocessing.feature_builders.pgp.pgp_graph_map_feature_builder import (
    PGPGraphMapFeatureBuilder,
)
from typing import Tuple
import numpy as np

def get_pgp_graph(current_input: PlannerInput,
                  initialization: PlannerInitialization,
                  max_extent=(-20, 80, -50, 50),
                  polyline_resolution=1,
                  polyline_length=20,
                  proximal_edges_dist_tresh=4,
                  proximal_edges_yaw_thresh=0.785
                  ):
    PGPBuilder = PGPGraphMapFeatureBuilder(max_extent,
                                           polyline_resolution,
                                           polyline_length,
                                           proximal_edges_dist_tresh,
                                           proximal_edges_yaw_thresh)
    pgp_graph = PGPBuilder.get_features_from_simulation(current_input, initialization)
    return pgp_graph

def assign_pose_to_node(
    node_poses,
    query_pose,
    dist_thresh=5,
    yaw_thresh=np.pi / 3,
    return_multiple=False,
    route_node_idcs=None,
):
    """
    Assigns a given agent pose to a lane node. Takes into account distance from the lane centerline as well as
    direction of motion.
    """
    dist_vals = []
    yaw_diffs = []

    for i in range(len(node_poses)):
        distances = np.linalg.norm(node_poses[i][:, :2] - query_pose[:2], axis=1)
        dist_vals.append(np.min(distances))
        idx = np.argmin(distances)
        yaw_lane = node_poses[i][idx, 2]
        yaw_query = query_pose[2]
        yaw_diffs.append(
            np.arctan2(np.sin(yaw_lane - yaw_query), np.cos(yaw_lane - yaw_query))
        )

    idcs_yaw = np.where(np.absolute(np.asarray(yaw_diffs)) <= yaw_thresh)[0]
    idcs_dist = np.where(np.asarray(dist_vals) <= dist_thresh)[0]
    idcs = np.intersect1d(idcs_dist, idcs_yaw)
    yaw_candidates = idcs_yaw
    if route_node_idcs is not None:
        idcs = np.intersect1d(idcs, route_node_idcs)
        yaw_candidates = np.intersect1d(yaw_candidates, route_node_idcs)

    if len(idcs) > 0:
        if return_multiple:
            return idcs
        else:
            return idcs[int(np.argmin(np.asarray(dist_vals)[idcs]))]
    elif len(yaw_candidates) > 0:
        # use closest node that statisifies yaw (and route) constraint
        filtered_dist_vals = [dist_vals[idx] for idx in yaw_candidates]
        idx = np.argmin(np.asarray(filtered_dist_vals))
        assigned_node_id = yaw_candidates[idx]
        if return_multiple:
            return np.asarray([assigned_node_id])
        else:
            return assigned_node_id
    else:
        # use closest node as a fallback
        assigned_node_id = np.argmin(np.asarray(dist_vals))
        if return_multiple:
            return np.asarray([assigned_node_id])
        else:
            return assigned_node_id
