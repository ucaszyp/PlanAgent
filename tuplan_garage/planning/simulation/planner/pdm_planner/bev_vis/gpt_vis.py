import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.actor_state.tracked_objects_types import (
    AGENT_TYPES,
    TrackedObjectType,
)
from nuplan.common.actor_state.state_representation import Point2D

import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from nuplan.database.nuplan_db.nuplan_scenario_queries import *
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import (
    ScenarioExtractionInfo,
)
from nuplan.planning.training.preprocessing.features.trajectory_utils import (
    convert_absolute_to_relative_poses,
)

from nuplan.planning.training.preprocessing.features.agents import Agents
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.common.geometry.torch_geometry import global_state_se2_tensor_to_local
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import (
    AgentInternalIndex,
    EgoInternalIndex,
    sampled_past_ego_states_to_tensor,
    sampled_past_timestamps_to_tensor,
    compute_yaw_rate_from_state_tensors,
    filter_agents_tensor,
    pack_agents_tensor,
    pad_agent_states,
)

from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.geometry.torch_geometry import vector_set_coordinates_to_local_frame
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import *
from nuplan.planning.training.preprocessing.utils.vector_preprocessing import (
    interpolate_points,
)

from tuplan_garage.planning.simulation.planner.pdm_planner.bev_vis.data_utils import (
    convert_feature_layer_to_fixed_size,
    polyline_process,
)


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


def init_fig_ax_via_size(x_px: float, y_px: float):
    """Initializes a figure with given size in pixel."""
    fig, ax = plt.subplots()
    # Sets output image to pixel resolution.
    dpi = 50
    fig.set_size_inches([x_px / dpi, y_px / dpi])
    fig.set_dpi(dpi)
    fig.set_facecolor("white")
    return fig, ax


def init_fig_ax(
    vis_config: VizConfig = VizConfig(),
):
    """Initializes a figure with vis_config."""
    return init_fig_ax_via_size(
        (vis_config.front_x + vis_config.back_x) * vis_config.px_per_meter,
        (vis_config.front_y + vis_config.back_y) * vis_config.px_per_meter,
    )


def create_map_raster(ax, lanes, crosswalks, route_lanes):
    for i in range(lanes.shape[0]):
        lane = lanes[i]
        if lane[0][0] != 0:
            ax.plot(lane[:, 0], lane[:, 1], "black", linewidth=1)  # plot centerline

    for j in range(crosswalks.shape[0]):
        crosswalk = crosswalks[j]
        if crosswalk[0][0] != 0:
            ax.plot(
                crosswalk[:, 0], crosswalk[:, 1], "b", linewidth=1
            )  # plot crosswalk

    for k in range(route_lanes.shape[0]):
        route_lane = route_lanes[k]
        if route_lane[0][0] != 0:
            ax.plot(
                route_lane[:, 0], route_lane[:, 1], "black", linewidth=1
            )  # plot route_lanes
    # print("draw maps")


def create_ego_raster(ax, vehicle_state):
    # Extract ego vehicle dimensions
    vehicle_parameters = get_pacifica_parameters()
    ego_width = vehicle_parameters.width
    ego_front_length = vehicle_parameters.front_length
    ego_rear_length = vehicle_parameters.rear_length

    # Extract ego vehicle state
    x_center, y_center, heading = vehicle_state[0], vehicle_state[1], vehicle_state[2]
    ego_bottom_left = (x_center - ego_rear_length, y_center - ego_width / 2)

    # Paint the rectangle
    rect = plt.Rectangle(
        ego_bottom_left,
        ego_front_length + ego_rear_length,
        ego_width,
        linewidth=2,
        color="#00CFBF",
        alpha=1.0,
        zorder=3,
        transform=mpl.transforms.Affine2D().rotate_around(
            *(x_center, y_center), heading
        )
        + plt.gca().transData,
    )
    plt.gca().add_patch(rect)
    plt.text(
        (ego_bottom_left[0] + x_center) / 2,
        (ego_bottom_left[1] + y_center) / 2,
        "000",
        color="black",
        fontsize=8,
    )

    plt.arrow(
        x_center,
        y_center,
        (ego_front_length + ego_rear_length) * np.cos(heading),
        (ego_front_length + ego_rear_length) * np.sin(heading),
        head_width=0.6,
        head_length=0.6,
        fc="r",
        ec="r",
        linewidth=2.5,
    )


def create_agents_raster(ax, agents):
    for i in range(agents.shape[0]):
        if agents[i, 0] != 0:
            agent_type = int(agents[i, -1])
            x_center, y_center, heading = agents[i, 0], agents[i, 1], agents[i, 2]
            agent_length, agent_width = agents[i, 3], agents[i, 4]
            agent_bottom_left = (
                # x_center - agent_length*np.cos(heading) / 2-agent_width*np.sin(heading),
                # y_center - agent_length*np.sin(heading) / 2+agent_width*np.cos(heading),
                x_center-agent_length/2,
                y_center-agent_width/2
            )
            agent_bottom_left = (
                x_center - agent_length / 2,
                y_center - agent_width /2,
            )
            color = None

            if agent_type == int(TrackedObjectType.VEHICLE):
                color = "#FF9E00"
            elif agent_type == int(TrackedObjectType.PEDESTRIAN):
                color = "#0000E6"
            elif agent_type == int(TrackedObjectType.BICYCLE):
                color = "#FF3D63"
            elif agent_type == int(TrackedObjectType.TRAFFIC_CONE):
                color = "#E99646"
            elif agent_type == int(TrackedObjectType.BARRIER):
                color = "#A73BDC"
            elif agent_type == int(TrackedObjectType.CZONE_SIGN):
                color = "#314F4F"
            elif agent_type == int(TrackedObjectType.GENERIC_OBJECT):
                color = "#808080"

            # plt.arrow(x_center, y_center, agent_length/2, 0, head_width=0.2, head_length=0.2, fc='r', ec='r', linewidth=2)

            rect = plt.Rectangle(
                agent_bottom_left,
                agent_length,
                agent_width,
                linewidth=2,
                color=color,
                alpha=1,
                zorder=3,
                transform=mpl.transforms.Affine2D().rotate_around(
                    *(x_center, y_center), heading
                )
                + plt.gca().transData,
            )
            plt.gca().add_patch(rect)
            if agent_type == int(TrackedObjectType.VEHICLE):
                plt.text(
                    (agent_bottom_left[0] + x_center) / 2,
                    (agent_bottom_left[1] + y_center) / 2,
                    "%03d" % (i + 1),
                    color="black",
                    fontsize=8,
                )
                plt.arrow(
                    x_center,
                    y_center,
                    (agent_length -1) * np.cos(heading),
                    (agent_length -1) * np.sin(heading),
                    head_width=0.5,
                    head_length=0.5,
                    fc="r",
                    ec="r",
                    linewidth=2,
                )


def get_neighbor_vector_set_map(
    map_api: AbstractMap,
    map_features: List[str],
    point: Point2D,
    radius: float,
    route_roadblock_ids: List[str],
    traffic_light_status_data: List[TrafficLightStatusData],
):

    coords: Dict[str, MapObjectPolylines] = {}
    traffic_light_data: Dict[str, LaneSegmentTrafficLightData] = {}
    feature_layers: List[VectorFeatureLayer] = []

    for feature_name in map_features:
        try:
            feature_layers.append(VectorFeatureLayer[feature_name])
        except KeyError:
            raise ValueError(
                f"Object representation for layer: {feature_name} is unavailable"
            )

    # extract lanes
    if VectorFeatureLayer.LANE in feature_layers:
        lanes_mid, lanes_left, lanes_right, lane_ids = get_lane_polylines(
            map_api, point, radius
        )

        # lane baseline paths
        coords[VectorFeatureLayer.LANE.name] = lanes_mid

        # lane traffic light data
        traffic_light_data[VectorFeatureLayer.LANE.name] = get_traffic_light_encoding(
            lane_ids, traffic_light_status_data
        )

        # lane boundaries
        if VectorFeatureLayer.LEFT_BOUNDARY in feature_layers:
            coords[VectorFeatureLayer.LEFT_BOUNDARY.name] = MapObjectPolylines(
                lanes_left.polylines
            )
        if VectorFeatureLayer.RIGHT_BOUNDARY in feature_layers:
            coords[VectorFeatureLayer.RIGHT_BOUNDARY.name] = MapObjectPolylines(
                lanes_right.polylines
            )

    # extract route
    if VectorFeatureLayer.ROUTE_LANES in feature_layers:
        route_polylines = get_route_lane_polylines_from_roadblock_ids(
            map_api, point, radius, route_roadblock_ids
        )
        coords[VectorFeatureLayer.ROUTE_LANES.name] = route_polylines

    # extract generic map objects
    for feature_layer in feature_layers:
        if feature_layer in VectorFeatureLayerMapping.available_polygon_layers():
            polygons = get_map_object_polygons(
                map_api,
                point,
                radius,
                VectorFeatureLayerMapping.semantic_map_layer(feature_layer),
            )
            coords[feature_layer.name] = polygons

    return coords, traffic_light_data


def map_process(
    anchor_state,
    coords,
    traffic_light_data,
    map_features,
    max_elements,
    max_points,
    interpolation_method,
):
    """
    This function process the data from the raw vector set map data.
    :param anchor_state: The current state of the ego vehicle.
    :param coords: The input data of the vectorized map coordinates.
    :param traffic_light_data: The input data of the traffic light data.
    :return: dict of the map elements.
    """

    # convert data to tensor list
    anchor_state_tensor = torch.tensor(
        [anchor_state.x, anchor_state.y, anchor_state.heading], dtype=torch.float32
    )
    list_tensor_data = {}

    for feature_name, feature_coords in coords.items():
        list_feature_coords = []

        # Pack coords into tensor list
        for element_coords in feature_coords.to_vector():
            list_feature_coords.append(
                torch.tensor(element_coords, dtype=torch.float32)
            )
        list_tensor_data[f"coords.{feature_name}"] = list_feature_coords

        # Pack traffic light data into tensor list if it exists
        if feature_name in traffic_light_data:
            list_feature_tl_data = []

            for element_tl_data in traffic_light_data[feature_name].to_vector():
                list_feature_tl_data.append(
                    torch.tensor(element_tl_data, dtype=torch.float32)
                )
            list_tensor_data[f"traffic_light_data.{feature_name}"] = (
                list_feature_tl_data
            )

    """
    Vector set map data structure, including:
    coords: Dict[str, List[<np.ndarray: num_elements, num_points, 2>]].
            The (x, y) coordinates of each point in a map element across map elements per sample.
    traffic_light_data: Dict[str, List[<np.ndarray: num_elements, num_points, 4>]].
            One-hot encoding of traffic light status for each point in a map element across map elements per sample.
            Encoding: green [1, 0, 0, 0] yellow [0, 1, 0, 0], red [0, 0, 1, 0], unknown [0, 0, 0, 1]
    availabilities: Dict[str, List[<np.ndarray: num_elements, num_points>]].
            Boolean indicator of whether feature data is available for point at given index or if it is zero-padded.
    """

    tensor_output = {}
    traffic_light_encoding_dim = LaneSegmentTrafficLightData.encoding_dim()

    for feature_name in map_features:
        if f"coords.{feature_name}" in list_tensor_data:
            feature_coords = list_tensor_data[f"coords.{feature_name}"]

            feature_tl_data = (
                list_tensor_data[f"traffic_light_data.{feature_name}"]
                if f"traffic_light_data.{feature_name}" in list_tensor_data
                else None
            )

            coords, tl_data, avails = convert_feature_layer_to_fixed_size(
                anchor_state_tensor,
                feature_coords,
                feature_tl_data,
                max_elements[feature_name],
                max_points[feature_name],
                traffic_light_encoding_dim,
                interpolation=(
                    interpolation_method  # apply interpolation only for lane features
                    if feature_name
                    in [
                        VectorFeatureLayer.LANE.name,
                        VectorFeatureLayer.LEFT_BOUNDARY.name,
                        VectorFeatureLayer.RIGHT_BOUNDARY.name,
                        VectorFeatureLayer.ROUTE_LANES.name,
                        VectorFeatureLayer.CROSSWALK.name,
                    ]
                    else None
                ),
            )

            coords = vector_set_coordinates_to_local_frame(
                coords, avails, anchor_state_tensor
            )

            tensor_output[f"vector_set_map.coords.{feature_name}"] = coords
            tensor_output[f"vector_set_map.availabilities.{feature_name}"] = avails

            if tl_data is not None:
                tensor_output[f"vector_set_map.traffic_light_data.{feature_name}"] = (
                    tl_data
                )

    """
    Post-precoss the map elements to different map types. Each map type is a array with the following shape.
    N: number of map elements (fixed for a given map feature)
    P: number of points (fixed for a given map feature)
    F: number of features
    """

    for feature_name in map_features:
        if feature_name == "LANE":
            polylines = tensor_output[f"vector_set_map.coords.{feature_name}"].numpy()
            traffic_light_state = tensor_output[
                f"vector_set_map.traffic_light_data.{feature_name}"
            ].numpy()
            avails = tensor_output[
                f"vector_set_map.availabilities.{feature_name}"
            ].numpy()
            vector_map_lanes = polyline_process(polylines, avails, traffic_light_state)

        elif feature_name == "CROSSWALK":
            polylines = tensor_output[f"vector_set_map.coords.{feature_name}"].numpy()
            avails = tensor_output[
                f"vector_set_map.availabilities.{feature_name}"
            ].numpy()
            vector_map_crosswalks = polyline_process(polylines, avails)

        elif feature_name == "ROUTE_LANES":
            polylines = tensor_output[f"vector_set_map.coords.{feature_name}"].numpy()
            avails = tensor_output[
                f"vector_set_map.availabilities.{feature_name}"
            ].numpy()
            vector_map_route_lanes = polyline_process(polylines, avails)

        else:
            pass

    vector_map_output = {
        "lanes": vector_map_lanes,
        "crosswalks": vector_map_crosswalks,
        "route_lanes": vector_map_route_lanes,
    }

    return vector_map_output


def get_map(
    ego_state,
    route_roadblock_ids,
    traffic_light_data,
    map_api,
    map_features,
    max_elements,
    max_points,
    interpolation_method,
    radius,
):
    ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)

    coords, traffic_light_data = get_neighbor_vector_set_map(
        map_api,
        map_features,
        ego_coords,
        radius,
        route_roadblock_ids,
        traffic_light_data,
    )

    vector_map = map_process(
        ego_state.rear_axle,
        coords,
        traffic_light_data,
        map_features,
        max_elements,
        max_points,
        interpolation_method,
    )

    return vector_map


def plot_scenario(data, vis_dir):
    viz_config = None
    viz_config = VizConfig() if viz_config is None else VizConfig(**viz_config)
    fig, ax = init_fig_ax(viz_config)
    # fig, ax = plt.subplots()
    # Create map layers
    create_map_raster(ax, data["lanes"], data["crosswalks"], data["route_lanes"])

    # Create agent layers
    create_ego_raster(ax, data["ego_agent"])
    create_agents_raster(ax, data["neighbor_agents"])

    # Draw past and future trajectories
    # draw_trajectory(ax, data['ego_agent_past'], data['neighbor_agents_past'])
    # draw_trajectory(ax, data['ego_agent_future'], data['neighbor_agents_future'])

    plt.gca().set_aspect("equal")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    plt.savefig(vis_dir, dpi=100, bbox_inches="tight", pad_inches=0)
    # plt.savefig(f"{vis_dir}/{data['map_name']}_{data['token']}.png")
    plt.close()
