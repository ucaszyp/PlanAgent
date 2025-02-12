import os
import argparse
from tqdm import tqdm
from common_utils import *
from data_utils import *
import matplotlib.pyplot as plt
from nuplan.planning.utils.multithreading.worker_parallel import (
    SingleMachineParallelExecutor,
)
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (
    NuPlanScenarioBuilder,
)
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import (
    ScenarioMapping,
)
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

from vis_nuplan import *


class VizConfig:
    """Config for visualization."""

    front_x: float = 75.0
    back_x: float = 75.0
    front_y: float = 75.0
    back_y: float = 75.0
    px_per_meter: float = 4
    show_agent_id: bool = True
    center_agent_idx: int = -1  # -1 for SDC
    verbose: bool = True


# define data processor
class DataProcessor(object):
    def __init__(self, scenarios):
        self._scenarios = scenarios

        self.past_time_horizon = 2  # [seconds]
        self.num_past_poses = 10 * self.past_time_horizon
        self.future_time_horizon = 8  # [seconds]
        self.num_future_poses = 10 * self.future_time_horizon
        self.num_agents = 20

        self._map_features = [
            "LANE",
            "ROUTE_LANES",
            "CROSSWALK",
        ]  # name of map features to be extracted.
        self._max_elements = {
            "LANE": 40,
            "ROUTE_LANES": 10,
            "CROSSWALK": 5,
        }  # maximum number of elements to extract per feature layer.
        self._max_points = {
            "LANE": 50,
            "ROUTE_LANES": 50,
            "CROSSWALK": 30,
        }  # maximum number of points per feature to extract per feature layer.
        self._radius = 60  # [m] query radius scope relative to the current pose.
        self._interpolation_method = "linear"  # Interpolation method to apply when interpolating to maintain fixed size map elements.

    def get_ego_agent(self):
        self.anchor_ego_state = self.scenario.initial_ego_state

        past_ego_states = self.scenario.get_ego_past_trajectory(
            iteration=0,
            num_samples=self.num_past_poses,
            time_horizon=self.past_time_horizon,
        )

        sampled_past_ego_states = list(past_ego_states) + [self.anchor_ego_state]
        past_ego_states_tensor = sampled_past_ego_states_to_tensor(
            sampled_past_ego_states
        )
        # [x, y, heading, vx, vy, ax, ay]
        past_time_stamps = list(
            self.scenario.get_past_timestamps(
                iteration=0,
                num_samples=self.num_past_poses,
                time_horizon=self.past_time_horizon,
            )
        ) + [self.scenario.start_time]

        past_time_stamps_tensor = sampled_past_timestamps_to_tensor(past_time_stamps)

        return past_ego_states_tensor, past_time_stamps_tensor

    def get_neighbor_agents(self):
        present_tracked_objects = self.scenario.initial_tracked_objects.tracked_objects
        past_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in self.scenario.get_past_tracked_objects(
                iteration=0,
                time_horizon=self.past_time_horizon,
                num_samples=self.num_past_poses,
            )
        ]

        sampled_past_observations = past_tracked_objects + [present_tracked_objects]
        past_tracked_objects_tensor_list, past_tracked_objects_types = (
            sampled_tracked_objects_to_tensor_list(sampled_past_observations)
        )

        return past_tracked_objects_tensor_list, past_tracked_objects_types

    def get_map(self):
        ego_state = self.scenario.initial_ego_state
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        route_roadblock_ids = self.scenario.get_route_roadblock_ids()
        traffic_light_data = self.scenario.get_traffic_light_status_at_iteration(0)

        coords, traffic_light_data = get_neighbor_vector_set_map(
            self.map_api,
            self._map_features,
            ego_coords,
            self._radius,
            route_roadblock_ids,
            traffic_light_data,
        )

        vector_map = map_process(
            ego_state.rear_axle,
            coords,
            traffic_light_data,
            self._map_features,
            self._max_elements,
            self._max_points,
            self._interpolation_method,
        )

        return vector_map

    def get_ego_agent_future(self):
        current_absolute_state = self.scenario.initial_ego_state

        trajectory_absolute_states = self.scenario.get_ego_future_trajectory(
            iteration=0,
            num_samples=self.num_future_poses,
            time_horizon=self.future_time_horizon,
        )

        # Get all future poses of the ego relative to the ego coordinate system
        trajectory_relative_poses = convert_absolute_to_relative_poses(
            current_absolute_state.rear_axle,
            [state.rear_axle for state in trajectory_absolute_states],
        )

        return trajectory_relative_poses

    def get_neighbor_agents_future(self, agent_index):
        current_ego_state = self.scenario.initial_ego_state
        present_tracked_objects = self.scenario.initial_tracked_objects.tracked_objects

        # Get all future poses of of other agents
        future_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in self.scenario.get_future_tracked_objects(
                iteration=0,
                time_horizon=self.future_time_horizon,
                num_samples=self.num_future_poses,
            )
        ]

        sampled_future_observations = [present_tracked_objects] + future_tracked_objects
        future_tracked_objects_tensor_list, _ = sampled_tracked_objects_to_tensor_list(
            sampled_future_observations
        )
        agent_futures = agent_future_process(
            current_ego_state,
            future_tracked_objects_tensor_list,
            self.num_agents,
            agent_index,
        )

        return agent_futures

    """
    def waymax_plot(self, data, vis_dir):
        viz_config = (
            VizConfig() if viz_config is None else VizConfig(**viz_config)
        )
        fig, ax = init_fig_ax(viz_config)

        # 1. Plots trajectory.
        # Shape: (num_objects, num_timesteps).
        ego_traj = data['ego_agent_past']
        x_y_head = ego_traj[:,0:3]
        vehicle_parameters = get_pacifica_parameters()
        ego_width = vehicle_parameters.width
        ego_height = vehicle_parameters.height

        # The current timestep index of observation is the last one in time dimension
        # as observation is toward the past.
        timestep = None
        indices = None

        is_controlled = datatypes.get_control_mask(obs.metadata, highlight_obj)
        plot_trajectory(ax, traj, is_controlled, time_idx=timestep, indices=indices)  # pytype: disable=wrong-arg-types  # jax-ndarray

        # 2. Plots road graph elements.
        # Shape: (num_points,)
        plot_roadgraph_points(ax, obs.roadgraph_static_points, verbose=False)

        # Shape: (num_traffic_lights, num_timesteps).
        # plot_traffic_light_signals_as_points(
        #     ax, obs.traffic_lights, timestep, verbose=False
        # )

        # 3. Gets np img, centered on selected agent's current location.
        # Shape (num_objects, 2).
        current_xy = traj.xy[:, timestep, :]
        if viz_config.center_agent_idx == -1:
            xy = current_xy[obs.metadata.is_sdc]
        else:
            xy = current_xy[viz_config.center_agent_idx]
        origin_x, origin_y = xy[0, :2]
        ax.axis((
            origin_x - viz_config.back_x,
            origin_x + viz_config.front_x,
            origin_y - viz_config.back_y,
            origin_y + viz_config.front_y,
        ))

        return utils.img_from_fig(fig)
    """

    def plot_scenario(self, data, vis_dir):
        viz_config = None
        viz_config = VizConfig() if viz_config is None else VizConfig(**viz_config)
        fig, ax = init_fig_ax(viz_config)
        # fig, ax = plt.subplots()
        # Create map layers
        create_map_raster(ax, data["lanes"], data["crosswalks"], data["route_lanes"])

        # Create agent layers
        create_ego_raster(ax, data["ego_agent_past"][-1])
        create_agents_raster(ax, data["neighbor_agents_past"][:, -1])

        # Draw past and future trajectories
        # draw_trajectory(ax, data['ego_agent_past'], data['neighbor_agents_past'])
        # draw_trajectory(ax, data['ego_agent_future'], data['neighbor_agents_future'])

        plt.gca().set_aspect("equal")
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        plt.savefig(f"1.png", dpi=100, bbox_inches="tight", pad_inches=0)
        # plt.savefig(f"{vis_dir}/{data['map_name']}_{data['token']}.png")
        plt.close()

    def save_to_disk(self, dir, data):
        np.savez(f"{dir}/{data['map_name']}_{data['token']}.npz", **data)

    def work(self, save_dir, vis_dir, debug=False, waymax=False):
        for scenario in tqdm(self._scenarios):
            map_name = scenario._map_name
            token = scenario.token
            self.scenario = scenario
            self.map_api = scenario.map_api

            # get agent past tracks
            ego_agent_past, time_stamps_past = (
                self.get_ego_agent()
            )  # [x, y, heading, vx, vy, ax, ay]
            neighbor_agents_past, neighbor_agents_types = (
                self.get_neighbor_agents()
            )  # [instance_id, x, y, heading, vx, vy, ax, ay]
            ego_agent_past, neighbor_agents_past, neighbor_indices = agent_past_process(
                ego_agent_past,
                time_stamps_past,
                neighbor_agents_past,
                neighbor_agents_types,
                self.num_agents,
            )

            # get vector set map
            vector_map = self.get_map()

            # get agent future tracks
            ego_agent_future = self.get_ego_agent_future()
            neighbor_agents_future = self.get_neighbor_agents_future(neighbor_indices)

            # gather data
            data = {
                "map_name": map_name,
                "token": token,
                "ego_agent_past": ego_agent_past,
                "ego_agent_future": ego_agent_future,
                "neighbor_agents_past": neighbor_agents_past,
                "neighbor_agents_future": neighbor_agents_future,
            }
            data.update(vector_map)

            # visualization
            # if waymax:
            #     self.waymax_plot(data, vis_dir)
            if debug:
                self.plot_scenario(data, vis_dir)

            # save to disk
            self.save_to_disk(save_dir, data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Processing")
    parser.add_argument("--data_path", type=str, help="path to raw data")
    parser.add_argument("--map_path", type=str, help="path to map data")
    parser.add_argument("--save_path", type=str, help="path to save processed data")
    parser.add_argument("--vis_path", type=str, help="path to save visualize data")
    parser.add_argument(
        "--scenarios_per_type",
        type=int,
        default=1000,
        help="number of scenarios per type",
    )
    parser.add_argument(
        "--total_scenarios", default=None, help="limit total number of scenarios"
    )
    parser.add_argument(
        "--shuffle_scenarios", type=bool, default=False, help="shuffle scenarios"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="if visualize the data output",
        default=False,
    )
    parser.add_argument(
        "--waymax",
        action="store_true",
        help="if visualize with waymax style",
        default=False,
    )
    args = parser.parse_args()

    # create save folder
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.vis_path, exist_ok=True)
    # get scenarios
    map_version = "nuplan-maps-v1.0"
    sensor_root = None
    db_files = None
    scenario_mapping = ScenarioMapping(
        scenario_map=get_scenario_map(), subsample_ratio_override=0.5
    )
    builder = NuPlanScenarioBuilder(
        args.data_path,
        args.map_path,
        sensor_root,
        db_files,
        map_version,
        scenario_mapping=scenario_mapping,
    )
    # scenario_filter = ScenarioFilter(*get_filter_parameters(args.scenarios_per_type, args.total_scenarios, args.shuffle_scenarios))
    scenario_filter = ScenarioFilter(
        scenario_types=None,
        scenario_tokens=["003445cf99235331", "0a1adb702c0f5949"],
        log_names=None,
        map_names=None,
        num_scenarios_per_type=None,
        limit_total_scenarios=None,
        timestamp_threshold_s=None,
        ego_displacement_minimum_m=None,
        expand_scenarios=False,
        remove_invalid_goals=True,
        shuffle=False,
        ego_start_speed_threshold=None,
        ego_stop_speed_threshold=None,
        speed_noise_tolerance=None,
    )
    worker = SingleMachineParallelExecutor(use_process_pool=False)
    print("Building Scenarios")
    scenarios = builder.get_scenarios(scenario_filter, worker)
    print(f"Total number of scenarios: {len(scenarios)}")
    # process data
    del worker, builder, scenario_filter, scenario_mapping
    processor = DataProcessor(scenarios)
    processor.work(args.save_path, args.vis_path, debug=args.debug, waymax=args.waymax)
