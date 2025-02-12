import copy
import math
import random
from typing import List, Optional, Dict

import numpy as np
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, Point2D
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInput,
    PlannerInitialization,
)
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from scipy.spatial.distance import cdist, euclidean

from tuplan_garage.planning.simulation.planner.pdm_planner.GPT_utils_rest import (
    generate_gpt_decision,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.PGP_utils import (
    get_pgp_graph,
    assign_pose_to_node,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.abstract_pdm_planner import (
    AbstractPDMPlanner,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.bev_vis.gpt_vis import (
    get_map,
    plot_scenario,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation import (
    PDMObservation,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.proposal.batch_idm_policy import (
    BatchIDMPolicy,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.proposal.pdm_generator import (
    PDMGenerator,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.proposal.pdm_proposal import (
    PDMProposalManager,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    PDMScorer,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_emergency_brake import (
    PDMEmergencyBrake,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    parallel_discrete_path,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath
from nuplan.common.geometry.convert import absolute_to_relative_poses

from nuplan.common.actor_state.tracked_objects_types import (
    AGENT_TYPES,
    TrackedObjectType,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.graph_search.dijkstra import (
    Dijkstra,
)

from nuplan.common.maps.nuplan_map.lane import NuPlanLane
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from tuplan_garage.planning.simulation.planner.pdm_planner.utils_ import (
    get_choices_test,
    get_ego_info,
    compute_node_position,
    get_around_vehicle,
    get_lane_num,
    get_choices,
    get_around_vehicle_dict,
    get_static_object_info,
)
from shapely.geometry import Point

MAX_DYNAMIC_OBJECTS: Dict[TrackedObjectType, int] = {
    TrackedObjectType.VEHICLE: 50,
    TrackedObjectType.PEDESTRIAN: 25,
    TrackedObjectType.BICYCLE: 10,
}

max_elements = {
    "LANE": 40,
    "ROUTE_LANES": 10,
    "CROSSWALK": 5,
}
map_features = [
    "LANE",
    "ROUTE_LANES",
    "CROSSWALK",
]
max_points = {
    "LANE": 50,
    "ROUTE_LANES": 50,
    "CROSSWALK": 30,
}




class AbstractPDMClosedPlanner(AbstractPDMPlanner):
    """
    Interface for planners incorporating PDM-Closed. Used for PDM-Closed and PDM-Hybrid.
    """

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        proposal_sampling: TrajectorySampling,
        idm_policies: BatchIDMPolicy,
        lateral_offsets: Optional[List[float]],
        map_radius: float,
    ):
        """
        Constructor for AbstractPDMClosedPlanner
        :param trajectory_sampling: Sampling parameters for final trajectory
        :param proposal_sampling: Sampling parameters for proposals
        :param idm_policies: BatchIDMPolicy class
        :param lateral_offsets: centerline offsets for proposals (optional)
        :param map_radius: radius around ego to consider
        """

        super(AbstractPDMClosedPlanner, self).__init__(map_radius)

        assert (
            trajectory_sampling.interval_length == proposal_sampling.interval_length
        ), "AbstractPDMClosedPlanner: Proposals and Trajectory must have equal interval length!"

        # config parameters
        self._trajectory_sampling: int = trajectory_sampling
        self._proposal_sampling: int = proposal_sampling
        self._idm_policies: BatchIDMPolicy = idm_policies
        self._lateral_offsets: Optional[List[float]] = lateral_offsets

        if self._iteration == 0:
            self.last_choice = [-1, -1, -1]
        self.gpt_choice = ["keep", -1, -1, -1]
        self.scenario_analysis = ["keep", "single_direction", []]
        self.has_traffic_light = False

        # observation/forecasting class
        self._observation = PDMObservation(
            trajectory_sampling, proposal_sampling, map_radius
        )

        # proposal/trajectory related classes
        self._generator = PDMGenerator(trajectory_sampling, proposal_sampling)
        self._simulator = PDMSimulator(proposal_sampling)
        self._scorer = PDMScorer(proposal_sampling)
        self._emergency_brake = PDMEmergencyBrake(
            trajectory_sampling, time_to_infraction_threshold=2.0
        )

        # lazy loaded
        self._proposal_manager: Optional[PDMProposalManager] = None

    def _update_proposal_manager(self, chosen_lane, ego_state: EgoState):
        """
        Updates or initializes PDMProposalManager class
        :param ego_state: state of ego-vehicle
        """

        # TODO: Find additional conditions to trigger re-planning
        create_new_proposals = self._iteration == 0

        if create_new_proposals:
            proposal_paths: List[PDMPath] = self._get_proposal_paths(
                chosen_lane, ego_state
            )

            self._proposal_manager = PDMProposalManager(
                lateral_proposals=proposal_paths,
                longitudinal_policies=self._idm_policies,
            )

        # update proposals
        self._proposal_manager.update(chosen_lane.speed_limit_mps)

    def _get_adjacent_lanes(self, ego_state: EgoState, offset=4.0):
        current_lane, current_heading_error = self._get_starting_lane(ego_state)

        # get adjacent lanes

        adjacent_lanes = {"current_lane": current_lane, "left": None, "right": None}
        nearest_pose = current_lane.baseline_path.get_nearest_pose_from_position(
            ego_state.center.point
        )
        theta = ego_state.center.heading + np.pi / 2
        x_left = nearest_pose.x + np.cos(theta) * offset
        y_left = nearest_pose.y + np.sin(theta) * offset
        x_right = nearest_pose.x - np.cos(theta) * offset
        y_right = nearest_pose.y - np.sin(theta) * offset
        pose_left = StateSE2(x_left, y_left, ego_state.center.heading)
        pose_right = StateSE2(x_right, y_right, ego_state.center.heading)
        ego_state_left = EgoState.build_from_center(
            pose_left,
            ego_state.dynamic_car_state.center_velocity_2d,
            ego_state.dynamic_car_state.center_acceleration_2d,
            ego_state.tire_steering_angle,
            ego_state.time_point,
            ego_state.car_footprint.vehicle_parameters,
        )
        ego_state_right = EgoState.build_from_center(
            pose_right,
            ego_state.dynamic_car_state.center_velocity_2d,
            ego_state.dynamic_car_state.center_acceleration_2d,
            ego_state.tire_steering_angle,
            ego_state.time_point,
            ego_state.car_footprint.vehicle_parameters,
        )
        left_lane, left_heading_error = self._get_starting_lane(ego_state_left)
        right_lane, right_heading_error = self._get_starting_lane(ego_state_right)

        if current_heading_error > np.pi / 2:
            if left_heading_error < np.pi / 2:
                current_lane = left_lane
            elif right_heading_error < np.pi / 2:
                current_lane = right_lane
            return {"current_lane": current_lane, "left": None, "right": None}

        current_outgoing_edges = current_lane.outgoing_edges
        left_outgoing_edges = left_lane.outgoing_edges
        right_outgoing_edges = right_lane.outgoing_edges
        left_route_availabe = False
        right_route_availabe = False
        for left_outgoing_edge in left_outgoing_edges:
            for current_outgoing_edge in current_outgoing_edges:
                if current_outgoing_edge.is_same_roadblock(left_outgoing_edge):
                    left_route_availabe = True
                    break
            else:
                continue
            break

        for right_outgoing_edge in right_outgoing_edges:
            for current_outgoing_edge in current_outgoing_edges:
                if current_outgoing_edge.is_same_roadblock(right_outgoing_edge):
                    right_route_availabe = True
                    break
            else:
                continue
            break
        adjacent_lanes["left"] = (
            left_lane if not current_lane == left_lane and left_route_availabe else None
        )
        adjacent_lanes["right"] = (
            right_lane
            if not current_lane == right_lane and right_route_availabe
            else None
        )
        # adjacent_lanes['left'] = left_lane if current_lane.is_same_roadblock(left_lane) \
        #                                       and left_lane.is_left_of(current_lane) else None
        # adjacent_lanes['right'] = right_lane if current_lane.is_same_roadblock(right_lane) \
        #                                         and right_lane.is_right_of(current_lane) else None

        return adjacent_lanes

    def _get_discrete_centerline_change(
        self,
        current_lane: LaneGraphEdgeMapObject,
        ego_state: EgoState,
        mode="regular",
        search_depth: int = 30,
    ) -> List[StateSE2]:
        """
        Applies a Dijkstra search on the lane-graph to retrieve discrete centerline.
        :param ego_state: state of the ego vehicle.
        :param mode: mode of the planner, either 'regular' or 'change_lane'
        :param current_lane: lane object of starting lane.
        :param search_depth: depth of search (for runtime), defaults to 30
        :return: list of discrete states on centerline (x,y,θ)
        """
        if mode == "regular":
            roadblocks = list(self._route_roadblock_dict.values())
            roadblock_ids = list(self._route_roadblock_dict.keys())

            # find current roadblock index
            start_idx = np.argmax(
                np.array(roadblock_ids) == current_lane.get_roadblock_id()
            )
            roadblock_window = roadblocks[start_idx : start_idx + search_depth]

            graph_search = Dijkstra(current_lane, list(self._route_lane_dict.keys()))
            route_plan, path_found = graph_search.search(roadblock_window[-1])

            centerline_discrete_path: List[StateSE2] = []
            for lane in route_plan:
                centerline_discrete_path.extend(lane.baseline_path.discrete_path)
        else:
            roadblocks = list(self._route_roadblock_dict.values())
            roadblock_ids = list(self._route_roadblock_dict.keys())

            nearest_pose = current_lane.baseline_path.get_nearest_pose_from_position(
                ego_state.center.point
            )
            idx = 0
            for idx, discrete_point in enumerate(
                current_lane.baseline_path.discrete_path
            ):
                if (
                    euclidean(
                        [discrete_point.x, discrete_point.y],
                        [nearest_pose.x, nearest_pose.y],
                    )
                    < 0.1
                ):
                    break

            theta = nearest_pose.heading + np.pi / 2
            x_new = nearest_pose.x + np.cos(theta) * 4
            y_new = nearest_pose.y + np.sin(theta) * 4
            pose_adjacent = StateSE2(x_new, y_new, nearest_pose.heading)
            ego_state_new = EgoState.build_from_center(
                pose_adjacent,
                ego_state.dynamic_car_state.center_velocity_2d,
                ego_state.dynamic_car_state.center_acceleration_2d,
                ego_state.tire_steering_angle,
                ego_state.time_point,
                ego_state.car_footprint.vehicle_parameters,
            )

            origin_start_idx = np.argmax(
                np.array(roadblock_ids) == current_lane.get_roadblock_id()
            )

            current_lane_new = current_lane
            closest_distance = np.inf
            for edge in self._route_lane_dict.values():
                distance = edge.polygon.distance(ego_state_new.car_footprint.geometry)
                if distance < closest_distance:
                    current_lane_new = edge
                    closest_distance = distance
            # current_lane_new=self._get_starting_lane(ego_state_new)
            # find current roadblock index
            start_idx = np.argmax(
                np.array(roadblock_ids) == current_lane_new.get_roadblock_id()
            )

            roadblock_window = roadblocks[start_idx : start_idx + search_depth]

            graph_search = Dijkstra(
                current_lane_new, list(self._route_lane_dict.keys())
            )
            route_plan, path_found = graph_search.search(roadblock_window[-1])

            centerline_discrete_path: List[StateSE2] = []
            # centerline_discrete_path.extend(route_lane[idx:])
            for lane in route_plan:
                centerline_discrete_path.extend(lane.baseline_path.discrete_path)

        return centerline_discrete_path

    def _get_proposal_paths(
        self, current_lane: LaneGraphEdgeMapObject, ego_state: EgoState
    ) -> List[PDMPath]:
        """
        Returns a list of path's to follow for the proposals. Inits a centerline.
        :param current_lane: current or starting lane of path-planning
        :return: lists of paths (0-index is centerline)
        """
        centerline_discrete_path = self._get_discrete_centerline(current_lane)
        self._centerline = PDMPath(centerline_discrete_path)
        # 1. save centerline path (necessary for progress metric)
        output_paths: List[PDMPath] = [self._centerline]

        # 2. add additional paths with lateral offset of centerline
        if self._lateral_offsets is not None:
            for lateral_offset in self._lateral_offsets:
                offset_discrete_path = parallel_discrete_path(
                    discrete_path=centerline_discrete_path, offset=lateral_offset
                )
                output_paths.append(PDMPath(offset_discrete_path))

        return output_paths

    def _get_scenario_analysis(self, adjacent_lanes, search_depth=30, offset=4):
        roadblocks = list(self._route_roadblock_dict.values())
        roadblock_ids = list(self._route_roadblock_dict.keys())

        # find current roadblock index
        start_idx = np.argmax(
            np.array(roadblock_ids) == adjacent_lanes["current_lane"].get_roadblock_id()
        )
        roadblock_window = roadblocks[start_idx : start_idx + search_depth]

        graph_search = Dijkstra(
            adjacent_lanes["current_lane"], list(self._route_lane_dict.keys())
        )
        route_plan, path_found = graph_search.search(roadblock_window[-1])
        curve_lane_ids = []
        curve_after_lane = None
        curve_direction = "keep"
        for i in range(len(route_plan)):
            lane_array = route_plan[i].baseline_path.discrete_path
            curve_lane_ids.append(route_plan[i].id)
            heading_diff = lane_array[0].heading - lane_array[-1].heading
            if heading_diff > np.pi / 4 or heading_diff < -np.pi:
                curve_direction = "right"
                curve_after_lane = i + 1
                break
            elif heading_diff < -np.pi / 4 or heading_diff > np.pi:
                curve_direction = "left"
                curve_after_lane = i + 1
                break

        if curve_after_lane:

            arc_lane = route_plan[curve_after_lane]
            arc_point = arc_lane.baseline_path.discrete_path[
                (len(arc_lane.baseline_path.discrete_path)) // 2
            ]
            theta = arc_point.heading + np.pi / 2
            x_left = arc_point.x + np.cos(theta) * offset
            y_left = arc_point.y + np.sin(theta) * offset
            intersecting_lanes = self._drivable_area_map.intersects(
                Point(x_left, y_left)
            )
            # print("intersecting_lanes", intersecting_lanes)
            # is_on_lane = False
            # for lane_id in intersecting_lanes:
            #     if lane_id in self._route_lane_dict.keys():
            #         is_on_lane = True
            #         break

            if not arc_lane.adjacent_edges[0] and intersecting_lanes:
                return curve_direction, "double_direction", curve_lane_ids
        return curve_direction, "single_direction", curve_lane_ids

    def _get_gpt_choice(self, ego_state, adjacent_lanes, current_input, initialization):
        user_message = ""
        # getting surrouding vehicles, change to relative coordinates
        vehicle_tokens = self._observation._object_manager._dynamic_object_tokens[
            TrackedObjectType.VEHICLE
        ]
        _, _, dynamic_object_tokens, dynamic_object_coords, dynamic_object_dxy = (
            self._observation._object_manager.get_nearest_objects(
                ego_state.center.point
            )
        )
        relative_coords = None
        if len(dynamic_object_coords.shape) == 3:
            dynamic_object_coords = dynamic_object_coords[:, -1, :]

            absolute_coords = [ego_state.center]
            for dynamic_object_coord,dxy in zip(dynamic_object_coords,dynamic_object_dxy):
                absolute_coords.append(
                    StateSE2(
                        dynamic_object_coord[0],
                        dynamic_object_coord[1],
                        ego_state.center.heading,
                    )
                )
            relative_coords = absolute_to_relative_poses(absolute_coords)
        elif len(dynamic_object_coords.shape) == 2:
            dynamic_object_coords = dynamic_object_coords[-1, :]
            absolute_coords = [ego_state.center]
            absolute_coords.append(
                StateSE2(
                    dynamic_object_coords[0],
                    dynamic_object_coords[1],
                    ego_state.center.heading,
                )
            )
            relative_coords = absolute_to_relative_poses(absolute_coords)

        # get static object information
        static_object_tokens = self._observation._object_manager._static_object_tokens
        static_object_coords = self._observation._object_manager._static_object_coords
        relative_static_coords = None
        if len(static_object_coords):
            absolute_static_coords = [ego_state.center]
            for static_object_coord in static_object_coords:
                absolute_static_coords.append(
                    StateSE2(
                        static_object_coord[-1][0],
                        static_object_coord[-1][1],
                        ego_state.center.heading,
                    )
                )
            relative_static_coords = absolute_to_relative_poses(absolute_static_coords)

        # get route lane dict
        if self._iteration == 0:
            current_lane_connectors = {}
            current_lane_connectors[adjacent_lanes["current_lane"].id] = adjacent_lanes[
                "current_lane"
            ]
            for outgoing_edge in adjacent_lanes["current_lane"].outgoing_edges:
                current_lane_connectors[outgoing_edge.id] = outgoing_edge
                for outgoing2_edge in outgoing_edge.outgoing_edges:
                    current_lane_connectors[outgoing2_edge.id] = outgoing2_edge
            traffic_light_tokens, traffic_light_polygon = (
                self._observation._get_traffic_light_geometries(
                    current_input.traffic_light_data,
                    current_lane_connectors,
                )
            )
            self.has_traffic_light = True if traffic_light_tokens else False

        # get pgp graph
        pgp_graph = get_pgp_graph(current_input, initialization)
        s_next = pgp_graph.s_next
        edge_type = pgp_graph.edge_type
        node_feats = pgp_graph.lane_node_feats
        node_feat_lens = np.sum(1 - pgp_graph.lane_node_masks[:, :, 0], axis=1)
        node_poses = []
        for i, node_feat in enumerate(node_feats):
            if node_feat_lens[i] != 0:
                node_poses.append(node_feat[: int(node_feat_lens[i]), :3])
        current_node = assign_pose_to_node(node_poses, [0, 0, 0])

        # lane position indicator for the nearest nodes
        lane_position_indicator = compute_node_position(
            s_next, current_node, node_feats
        )

        # get bev
        vehicle_token_mapped={}
        interpolation_method = "linear"
        vector_map = get_map(
            ego_state,
            self._route_roadblock_dict.keys(),
            current_input.traffic_light_data,
            self._map_api,
            map_features,
            max_elements,
            max_points,
            interpolation_method,
            self._map_radius,
        )
        neighbor_agents = []
        _, observation = current_input.history.current_state
        agent_absolute_coords = [ego_state.center]
        agent_shape = []
        i=0
        for object in observation.tracked_objects:
            if object.tracked_object_type == TrackedObjectType.EGO or (
                (
                    self._map_radius
                    and ego_state.center.distance_to(object.center) > self._map_radius
                )
                or (object.track_token in self._observation._collided_track_ids)
            ):
                continue
            agent_shape.append([object._box._length, object._box._width,object.tracked_object_type])
            agent_absolute_coords.append(object._box._center)
            if object.tracked_object_type==TrackedObjectType.VEHICLE:
                vehicle_token_mapped[object.track_token]="%03d" % (i + 1)
            i+=1
        agent_relative_coords = absolute_to_relative_poses(agent_absolute_coords)
        for i in range(len(agent_shape)):
            neighbor_agents.append(
                agent_relative_coords[i + 1].serialize()+agent_shape[i]
            )
        ego_agent=agent_relative_coords[0].serialize()
        bev_data_cache = {
            "ego_agent": np.array(ego_agent),
            "neighbor_agents": np.array(neighbor_agents),
        }
        bev_data_cache.update(vector_map)

        randint=random.randint(1,1000)
        path="" # fill your path
        bev_pic = f"{path}/{self._iteration}"
        plot_scenario(
            bev_data_cache,
            bev_pic,
        )

        # add ego information to user_message
        user_message += get_ego_info(ego_state, self.last_choice)

        # add lane information to user_message
        user_message += get_lane_num(adjacent_lanes)

        # get the nearest vehicle information of ego in seven directions 
        # according to lane_position_indicator
        nearest_vehicles = get_around_vehicle_dict(
            adjacent_lanes,
            lane_position_indicator,
            ego_state,
            node_poses,
            relative_coords,
            dynamic_object_tokens,
            dynamic_object_dxy,
            vehicle_tokens,
            vehicle_token_mapped
        )

        # add the surrounding vehicles info to user_message
        user_message += get_around_vehicle(nearest_vehicles, self.has_traffic_light)

        # add static object to user_message
        static_info, abstastic_flag = get_static_object_info(
            ego_state, adjacent_lanes, static_object_tokens, relative_static_coords
        )
        user_message += static_info

        # use rule base to get the lane choices
        lane_choices = get_choices(ego_state, adjacent_lanes, nearest_vehicles)

        user_message += "Speed limit options: [3.0,9.0,15.0]\n"
        user_message += "Max acceleration options: [1.5,2.5,3.5]\n"
        user_message += "Max deceleration options: [1.0,2.0,3.0] default is 3.0\n"

        # add lane choices to user_message
        user_message += "Lateral decision option: "
        for choice in lane_choices:
            user_message += choice + ", "


        self.gpt_choice = generate_gpt_decision(user_message, self._iteration,bev_pic+".png")
        
        # change centerline according to the gpt_choice
        if self.gpt_choice[0] == "left" and adjacent_lanes["right"] is not None:
            chosen_lane = adjacent_lanes["right"]
        elif self.gpt_choice[0] == "right" and adjacent_lanes["left"] is not None:
            chosen_lane = adjacent_lanes["left"]
        else:
            chosen_lane = adjacent_lanes["current_lane"]
        return chosen_lane, self.gpt_choice, abstastic_flag

    def _get_simulated_proposals(self, ego_state, chosen_lane, abstastic_flag):
        # 2. Centerline extraction and proposal update
        self._update_proposal_manager(chosen_lane, ego_state)

        # 3. Generate/Unroll proposals
        proposals_array = self._generator.generate_proposals(
            ego_state, self._observation, self._proposal_manager
        ) 


        mask = [True] * len(proposals_array)
        original_indices = np.arange(len(proposals_array))

        # gpt change
        if (
            not self.gpt_choice[1] == -1
            and not self.gpt_choice[2] == -1
            and not self.gpt_choice[3] == -1
        ):
            gpt_choice_index = (
                self.gpt_choice[1] * 9 + self.gpt_choice[2] * 3 + self.gpt_choice[3]
            )
            mask = [
                False if not i % 27 == gpt_choice_index else mask[i]
                for i in range(len(original_indices))
            ]

        # mask[:27]=[False]*27
        proposals_array = proposals_array[mask]

        # 4. Simulate proposals
        simulated_proposals_array = self._simulator.simulate_proposals(
            proposals_array, ego_state
        )
        return simulated_proposals_array, original_indices, mask

    def _get_closed_loop_trajectory(
        self,
        current_input: PlannerInput,
        initialization: PlannerInitialization,
    ) -> InterpolatedTrajectory:
        """
        Creates the closed-loop trajectory for PDM-Closed planner.
        :param current_input: planner input
        :return: trajectory
        """

        ego_state, observation = current_input.history.current_state
        # 1. Environment forecast and observation update
        self._observation.update(
            ego_state,
            observation,
            current_input.traffic_light_data,
            self._route_lane_dict,
        )

        # self.gpt_choice = ["keep", -1, -1, -1]
        adjacent_lanes = self._get_adjacent_lanes(ego_state)
        chosen_lane = adjacent_lanes["current_lane"]
        abstastic_flag = 0
        # print("test gpt simulation", self._iteration)
        if self._iteration %20 ==0:
            max_score = 0.0
            threshold = 0.5
            try_times = 3
            while max_score < threshold and try_times:
                chosen_lane, _, abstastic_flag = self._get_gpt_choice(
                    ego_state, adjacent_lanes, current_input, initialization
                )
                simulated_proposals_array, original_indices, mask = (
                    self._get_simulated_proposals(
                        ego_state, chosen_lane, abstastic_flag
                    )
                )

                # 5. Score proposals
                proposal_scores = self._scorer.score_proposals(
                    simulated_proposals_array,
                    ego_state,
                    self._observation,
                    self._centerline,
                    self._route_lane_dict,
                    self._drivable_area_map,
                    self._map_api,
                )
                max_score = proposal_scores.max()
                try_times -= 1
        else:
            simulated_proposals_array, original_indices, mask = (
                self._get_simulated_proposals(ego_state, chosen_lane, abstastic_flag)
            )

            # 5. Score proposals
            proposal_scores = self._scorer.score_proposals(
                simulated_proposals_array,
                ego_state,
                self._observation,
                self._centerline,
                self._route_lane_dict,
                self._drivable_area_map,
                self._map_api,
            )
        # 6.a Apply brake if emergency is expected
        trajectory = self._emergency_brake.brake_if_emergency(
            ego_state, proposal_scores, self._scorer
        )

        # 6.b Otherwise, extend and output best proposal
        if trajectory is None:
            max_arg = np.argmax(proposal_scores)

            max_arg_origin = original_indices[mask][max_arg]
            trajectory = self._generator.generate_trajectory(max_arg_origin)

        return trajectory
