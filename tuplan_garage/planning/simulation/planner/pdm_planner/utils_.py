import math

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.transform_state import (
    get_rear_right_corner,
    get_rear_left_corner,
    get_front_right_corner,
    get_front_left_corner,
)
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from scipy.spatial.distance import cdist
import numpy as np
from scipy.spatial.distance import euclidean

from tuplan_garage.planning.simulation.planner.pdm_planner.PGP_utils import (
    assign_pose_to_node,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.PGP_utils import (
    get_pgp_graph,
    assign_pose_to_node,
)
from nuplan.common.actor_state.state_representation import StateSE2, Point2D

REGION_DICT = {
    1: "rear right",
    2: "right",
    3: "front right",
    4: "behind",
    5: "center",
    6: "front",
    7: "rear left",
    8: "left",
    9: "front left",
}


def get_pre_choice(pre_score,pre_choice):
    message=""
    message+=f"Reference choice: [{pre_choice[0]}.1f,{pre_choice[1]}.1f,{pre_choice[2]}.1f]\n"
    message+=f"Predicted score: {pre_score}.1f\n"
    return message

# 将当前的场景类型加入到user_message中
def get_scenario_type(scenario_type):
    message = "Scenario type: "
    # if scenario_type == "right" or "special_right":
    #     message += (
    #         "You are turning right and the road ahead is a two-direction street\n"
    #     )
    # elif scenario_type == "left":
    #     message += "You are turning left.\n"
    # else:
    #     message += "You are driving normally.\n"
    # return message
    if scenario_type == "special_right":
        message += (
            "You are turning right and the road ahead is a two-direction street\n"
        )
    # elif scenario_type == "left":
    #     message += "You are turning left.\n"
    else:
        message += "You are driving normally.\n"
    return message


# 将静态物体信息加入到user_message中
def get_static_object_info(
    ego_state,
    adjacent_lanes,
    static_object_tokens,
    static_object_coords,
    visible_radius=15.0,
    interval=1.5,
):
    if not static_object_coords or not static_object_tokens:
        return "", 0
    current_lane = adjacent_lanes["current_lane"]
    left_edge = current_lane.left_boundary
    right_edge = current_lane.right_boundary
    user_message = ""
    l, r = False, False
    for object_token, object_cood in zip(static_object_tokens, static_object_coords):
        object_center = Point2D(object_cood.x, object_cood.y)
        to_ego_distance = math.sqrt(
            (object_center.x - ego_state.center.x) ** 2
            + (object_center.y - ego_state.center.y) ** 2
        )
        if to_ego_distance < visible_radius and object_cood.x > 0:
            left_arc = left_edge.get_nearest_pose_from_position(object_center)
            right_arc = right_edge.get_nearest_pose_from_position(object_center)
            left_arc_distance = math.sqrt(
                (left_arc.x - object_center.x) ** 2
                + (left_arc.y - object_center.y) ** 2
            )
            right_arc_distance = math.sqrt(
                (right_arc.x - object_center.x) ** 2
                + (right_arc.y - object_center.y) ** 2
            )
            if left_arc_distance < interval:
                l = True
                user_message += f"There is a small-scale static object on left boundary of the lane. The distance to the ego vehicle is {left_arc_distance}. It's token is {object_token}\n"
            elif right_arc_distance < interval:
                r = True
                user_message += f"There is a small-scale static object on right boundary of the lane. The distance to the ego vehicle is {right_arc_distance}. It's token is {object_token}\n"
    return user_message, (l - r)


# 按照pgp图的方式得到七个方向最近的node节点
def compute_node_position(s_next, current_node, node_feats):
    current_node_len = 0
    for point in node_feats[current_node]:
        if point[0] == 0 and point[1] == 0:
            break
        current_node_len += 1

    lane_position_indicator = {
        "current": [],
        "front": [],
        "front left": [],
        "front right": [],
        "rear left": [],
        "rear right": [],
        "behind": [],
        "left": [],
        "right": [],
    }
    lane_position_indicator["current"].append(current_node)
    for index, node_idx in enumerate(s_next[current_node]):
        if node_idx == 0:
            break
        node_idx = int(node_idx)
        if index == 0:
            lane_position_indicator["front"].append(node_idx)
            continue
        if node_idx >= len(node_feats):
            continue
        node_len = 0
        for point in node_feats[node_idx]:
            if point[0] == 0 and point[1] == 0:
                break
            node_len += 1

        if (
            np.all(
                node_feats[node_idx, :, 0]
                >= node_feats[current_node, current_node_len - 1, 0]
            )
            and abs(np.mean(node_feats[node_idx, :, 1])) < 2
        ):
            lane_position_indicator["front"].append(node_idx)

        elif (
            np.all(node_feats[node_idx, :, 0] <= node_feats[current_node, 0, 0])
            and abs(np.mean(node_feats[node_idx, :, 1])) < 2
        ):
            lane_position_indicator["behind"].append(node_idx)

        elif (
            np.all(node_feats[node_idx, 2:, 0] >= 0)
            and np.all(node_feats[node_idx, : node_len - 5, 1] >= 0)
            and abs(np.mean(node_feats[node_idx, :, 1])) > 2
        ):

            lane_position_indicator["front left"].append(node_idx)

        elif (
            np.all(node_feats[node_idx, 2:, 0] >= 0)
            and np.all(node_feats[node_idx, : node_len - 5, 1] <= 0)
            and abs(np.mean(node_feats[node_idx, :, 1])) > 2
        ):

            lane_position_indicator["front right"].append(node_idx)

        elif (
            np.all(node_feats[node_idx, : node_len - 2, 0] <= 0)
            and np.all(node_feats[node_idx, 5:, 1] >= 0)
            and abs(np.mean(node_feats[node_idx, :, 1])) > 2
        ):

            lane_position_indicator["rear left"].append(node_idx)

        elif (
            np.all(node_feats[node_idx, : node_len - 2, 0] <= 0)
            and np.all(node_feats[node_idx, 5:, 1] <= 0)
            and abs(np.mean(node_feats[node_idx, :, 1])) > 2
        ):

            lane_position_indicator["rear right"].append(node_idx)

        elif (
            np.mean(node_feats[node_idx, :, 1]) > 2
            and euclidean((0, 0), np.mean(node_feats[node_idx, :node_len, :2], axis=0))
            < 7
        ):
            lane_position_indicator["left"].append(node_idx)

        elif (
            np.mean(node_feats[node_idx, :, 1]) < -2
            and euclidean((0, 0), np.mean(node_feats[node_idx, :node_len, :2], axis=0))
            < 7
        ):

            lane_position_indicator["right"].append(node_idx)

    return lane_position_indicator


# 将ego信息加入到user message中
def get_ego_info(ego_state, last_choice):
    ego_velocity = math.sqrt(
        ego_state.dynamic_car_state.center_velocity_2d.x**2
        + ego_state.dynamic_car_state.center_velocity_2d.y**2
    )
    velocity_angle = math.atan2(
        ego_state.dynamic_car_state.center_velocity_2d.y,
        ego_state.dynamic_car_state.center_velocity_2d.x,
    )
    ego_accel = math.sqrt(
        ego_state.dynamic_car_state.center_acceleration_2d.x**2
        + ego_state.dynamic_car_state.center_acceleration_2d.y**2
    )
    accel_angle = math.atan2(
        ego_state.dynamic_car_state.center_acceleration_2d.y,
        ego_state.dynamic_car_state.center_acceleration_2d.x,
    )

    # if abs(velocity_angle - accel_angle) % math.pi < math.pi / 2:
    #     user_message = f"Your motion state: You current speed is {ego_velocity:.2f}m/s, acceleration is {ego_accel :.2f}m/s^2, heading is {(ego_state.center.heading/math.pi):.2f}pi.\n"
    # else:
    #     user_message = f"Your motion state: You current speed is {ego_velocity:.2f}m/s, deceleration is {ego_accel :.2f}m/s^2, heading is {(ego_state.center.heading/math.pi):.2f}pi.\n"
    if abs(velocity_angle - accel_angle) % math.pi < math.pi / 2:
        user_message = f"Your motion state: You current speed is {ego_velocity:.2f}m/s, acceleration is {ego_accel :.2f}m/s^2.\n"
    else:
        user_message = f"Your motion state: You current speed is {ego_velocity:.2f}m/s, deceleration is {ego_accel :.2f}m/s^2.\n"

    # speed_limit_mode = math.floor(last_choice % 5 + 1) * 0.2
    # user_message = f"Your motion state: You current speed is {ego_velocity:.2f}m/s, acceleration is {ego_accel * accel_flag:.2f}m/s^2, heading is {ego_state.center.heading:.2f}pi. Your current limit speed and max acceleration is {last_choice[0]:.1f} and {last_choice[1]:.1f}\n"
    return user_message


# 将车道数量转化成文本形式
def get_lane_num(adjacent_lanes):
    user_message = ""
    if adjacent_lanes["left"] and adjacent_lanes["right"]:
        user_message += "You are driving in the middle lane of a multi-lanes road.\n"
    elif adjacent_lanes["left"]:
        user_message += "You are driving in the rightest lane of a multi-lanes road.\n"
    elif adjacent_lanes["right"]:
        user_message += "You are driving in the leftest lane of a multi-lanes road.\n"
    else:
        user_message += "You are driving in the single lane road.\n"
    return user_message



# 将pgp和坐标两种方式结合得到八个方向最近的车辆信息
def get_around_vehicle_dict(
    adjacent_lanes,
    lane_position_indicator,
    ego_state,
    node_poses,
    relative_coords,
    dynamic_object_tokens,
    dynamic_object_dxy,
    vehicle_tokens,
    vehicle_token_mapped,
    visible_radius=20.0,
):
    nearest_vehicles_position=None
    nearest_vehicles = {
        "front": [],
        "front left": [],
        "front right": [],
        "rear left": [],
        "rear right": [],
        "left": [],
        "right": [],
    }
    if not relative_coords:
        return nearest_vehicles
    left_lane_consistence = (
        False
        if adjacent_lanes["left"]
        and not lane_position_indicator["left"]
        and not lane_position_indicator["front left"]
        and not lane_position_indicator["rear left"]
        else True
    )

    right_lane_consistence = (
        False
        if adjacent_lanes["right"]
        and not lane_position_indicator["right"]
        and not lane_position_indicator["front right"]
        and not lane_position_indicator["rear right"]
        else True
    )
    nearest_vehicles_position = get_around_vehicle_position(
    ego_state,
    dynamic_object_tokens,
    vehicle_tokens,
    vehicle_token_mapped,
    relative_coords,
    dynamic_object_dxy,
    visible_radius,
    )
    if not left_lane_consistence or not right_lane_consistence:
        if not left_lane_consistence:
            if (
                nearest_vehicles_position["front left"]
                and nearest_vehicles_position["front left"][1] < visible_radius
            ):
                nearest_vehicles["front left"].append(
                    nearest_vehicles_position["front left"]
                )
            if nearest_vehicles_position["rear left"]:
                nearest_vehicles["rear left"].append(
                    nearest_vehicles_position["rear left"]
                )
            if (
                nearest_vehicles_position["left"]
                and nearest_vehicles_position["left"][1] < visible_radius
            ):
                nearest_vehicles["left"].append(nearest_vehicles_position["left"])
        if not right_lane_consistence:
            if (
                nearest_vehicles_position["front right"]
                and nearest_vehicles_position["front right"][1] < visible_radius
            ):
                nearest_vehicles["front right"].append(
                    nearest_vehicles_position["front right"]
                )
            if nearest_vehicles_position["rear right"]:
                nearest_vehicles["rear right"].append(
                    nearest_vehicles_position["rear right"]
                )
            if (
                nearest_vehicles_position["right"]
                and nearest_vehicles_position["right"][1] < visible_radius
            ):
                nearest_vehicles["right"].append(nearest_vehicles_position["right"])

    for i in range(1, len(relative_coords)):
        distance = math.sqrt(relative_coords[i].x ** 2 + relative_coords[i].y ** 2)
        if distance > visible_radius * (3 / 2):
            break
        if dynamic_object_tokens[i - 1] not in vehicle_tokens:
            continue
        vehicle_node = assign_pose_to_node(node_poses, relative_coords[i].serialize())
        velocity = math.sqrt(
            dynamic_object_dxy[i - 1][0] ** 2 + dynamic_object_dxy[i - 1][1] ** 2
        )
        # heading=relative_coords[i].heading
        heading = (
            math.atan2(dynamic_object_dxy[i - 1, 1], dynamic_object_dxy[i - 1, 0])
            / math.pi
        )
        # if dynamic_object_dxy[i - 1, 0] >= 0:
        #     heading = "same"
        # else:
        #     heading = "opposite"

        # 对于在后方的车辆,视野看到visible_radius内
        token=vehicle_token_mapped.get(dynamic_object_tokens[i-1]) if vehicle_token_mapped else dynamic_object_tokens[i-1]
        if not token:
            continue
        if (
            vehicle_node in lane_position_indicator["rear left"]
            or vehicle_node in lane_position_indicator["rear right"]
        ):
            if vehicle_node in lane_position_indicator["rear left"]:
                nearest_vehicles["rear left"].append(
                    [token, distance, velocity, heading, "node"]
                )
            else:
                nearest_vehicles["rear right"].append(
                    [token, distance, velocity, heading, "node"]
                )
        # 对于非后面的车辆,视野看到visible_radius*(3/2)内
        # or (
        #         vehicle_node == lane_position_indicator["current"][0]
        #         and relative_coords[i].x > 0
        #     )
        elif distance <= visible_radius:
            if vehicle_node in lane_position_indicator["front"] or (
                vehicle_node == lane_position_indicator["current"][0]
                and relative_coords[i].x > 0
                and abs(relative_coords[i].y) < 4
            ):
                nearest_vehicles["front"].append(
                    [token, distance, velocity, heading, "node"]
                )
            elif vehicle_node in lane_position_indicator["front left"]:
                nearest_vehicles["front left"].append(
                    [token, distance, velocity, heading, "node"]
                )
            elif vehicle_node in lane_position_indicator["front right"]:
                nearest_vehicles["front right"].append(
                    [token, distance, velocity, heading, "node"]
                )
            elif vehicle_node in lane_position_indicator["left"]:
                nearest_vehicles["left"].append(
                    [token, distance, velocity, heading, "node"]
                )
            elif vehicle_node in lane_position_indicator["right"]:
                nearest_vehicles["right"].append(
                    [token, distance, velocity, heading, "node"]
                )
    flag=True
    for vehicle in nearest_vehicles.values():
        if vehicle:
            flag=False
            break
    if flag and nearest_vehicles_position:
        for key,value in nearest_vehicles_position.items():
            if value:
                nearest_vehicles[key].append(value)
    return nearest_vehicles

def get_around_vehicle2(nearest_vehicles, has_traffic_light=False):
    user_message=""
    for key,vehicle in nearest_vehicles.items():
        if vehicle:
            user_message+=f"The vehicle {vehicle[0]} is driving in {key} of you. The distance is {vehicle[1]:.2f}m. Velocity is {vehicle[2]:.2f}m/s."
            if has_traffic_light:
                if vehicle[2]==0:
                    user_message+="He is waiting for the red light."
                elif vehicle[2]<2:
                    user_message+="He is was just waiting for the red light."
            user_message+="\n"
    return user_message


# 将周围车辆信息加入到user_message中
def get_around_vehicle(nearest_vehicles, has_traffic_light=False):
    user_message = ""
    for key, vehicles in nearest_vehicles.items():
        if key == "front":
            for vehicle in vehicles:
                if vehicle[-1] == "node":
                    user_message += f"The vehicle {vehicle[0]} is driving in directly front of you, in the same lane. The distance is {vehicle[1]:.2f}m. Velocity is {vehicle[2]:.2f}m/s.\n"
                else:
                    user_message += f"The vehicle {vehicle[0]} is driving in directly front of you. The distance is {vehicle[1]:.2f}. Velocity is {vehicle[2]:.2f}m/s.\n"
                if vehicle[2] == 0 and has_traffic_light:
                    user_message += "He is waiting for the red light."
                elif vehicle[2] < 2 and has_traffic_light:
                    user_message += "He is was just waiting for the red light."
                user_message += "\n"
        elif "left" in key:
            for vehicle in vehicles:
                if vehicle[-1] == "node":
                    user_message += f"The vehicle {vehicle[0]} is driving in {key} of you, in the left adjacent lane. The distance is {vehicle[1]:.2f}m. Velocity is {vehicle[2]:.2f}m/s.\n"
                else:
                    user_message += f"The vehicle {vehicle[0]} is driving in {key} of you. The distance is {vehicle[1]:.2f}m. Velocity is {vehicle[2]:.2f}m/s.\n"
                if vehicle[2] == 0 and has_traffic_light:
                    user_message += "He is waiting for the red light."
                elif vehicle[2] < 2 and has_traffic_light:
                    user_message += "He is was just waiting for the red light."
                user_message += "\n"
        elif "right" in key:
            for vehicle in vehicles:
                if vehicle[-1] == "node":
                    user_message += f"The vehicle {vehicle[0]} is driving in {key} of you, in the right adjacent lane. The distance is {vehicle[1]:.2f}m. Velocity is {vehicle[2]:.2f}m/s.\n"
                else:
                    user_message += f"The vehicle {vehicle[0]} is driving in {key} of you. The distance is {vehicle[1]:.2f}m. Velocity is {vehicle[2]:.2f}m/s.\n"
                if vehicle[2] == 0 and has_traffic_light:
                    user_message += "He is waiting for the red light."
                elif vehicle[2] < 2 and has_traffic_light:
                    user_message += "He is was just waiting for the red light."
                user_message += "\n"

    return user_message


# 按照坐标形式得到七个方向上距离最近的车辆
# 作为图形式的补充
def get_around_vehicle_position(
    ego_state,
    dynamic_object_tokens,
    vehicle_tokens,
    vehicle_token_mapped,
    relative_coords,
    dynamic_object_dxy,
    visible_radius,
):
    # m1, b1 = calculate_line(rear right.x, rear right.y, rear left.x, rear left.y)
    # m2,b2=calculate_line(front right.x,front right.y,front left.x,front left.y)
    # m3,b3=calculate_line(front left.x,front left.y,rear left.x,rear left.y)
    # m4,b4=calculate_line(front right.x,front right.y,rear right.x,rear right.y)
    indicate = 7
    nearest_vehicles = {
        "front": None,
        "front left": None,
        "front right": None,
        "rear left": None,
        "rear right": None,
        "left": None,
        "right": None,
        # "behind": None,
    }
    if not relative_coords:
        return nearest_vehicles
    rear_right = get_rear_right_corner(
        relative_coords[0],
        ego_state.car_footprint.half_length,
        ego_state.car_footprint.half_width,
    )
    rear_left = get_rear_left_corner(
        relative_coords[0],
        ego_state.car_footprint.half_length,
        ego_state.car_footprint.half_width,
    )
    front_right = get_front_right_corner(
        relative_coords[0],
        ego_state.car_footprint.half_length,
        ego_state.car_footprint.half_width,
    )
    front_left = get_front_left_corner(
        relative_coords[0],
        ego_state.car_footprint.half_length,
        ego_state.car_footprint.half_width,
    )

    for i in range(1, len(relative_coords)):
        distance = math.sqrt(
            (relative_coords[i].x - relative_coords[0].x) ** 2
            + (relative_coords[i].y - relative_coords[0].y) ** 2
        )
        if indicate and distance > visible_radius:
            break
        if dynamic_object_tokens[i - 1] not in vehicle_tokens:
            continue

        velocity = math.sqrt(
            dynamic_object_dxy[i - 1, 0] ** 2 + dynamic_object_dxy[i - 1, 1] ** 2
        )
        heading = (
            math.atan2(dynamic_object_dxy[i - 1, 1], dynamic_object_dxy[i - 1, 0])
            / math.pi
        )
        # heading=relative_coords[i].heading
        # if dynamic_object_dxy[i - 1, 0] >= 0:
        #     heading = "same"
        # else:
        #     heading = "opposite"
        if relative_coords[i].x <= rear_left.x:
            column = 1
        elif rear_left.x < relative_coords[i].x < rear_right.x:
            column = 2
        else:
            column = 3

        if relative_coords[i].y <= rear_left.y:
            row = 1
        elif rear_left.y < relative_coords[i].y < front_left.y:
            row = 2
        else:
            row = 3
        region_number = 3 * (row - 1) + column
        if not nearest_vehicles[REGION_DICT[region_number]]:
            token=vehicle_token_mapped.get(dynamic_object_tokens[i-1]) if vehicle_token_mapped else dynamic_object_tokens[i-1]
            if not token:
                continue
            nearest_vehicles[REGION_DICT[region_number]] = [
                token,
                distance,
                velocity,
                heading,
                "position",
            ]
            indicate -= 1

    return nearest_vehicles


# 将可以选择的车道转化成文本形式
def get_choices(ego_state, adjacent_lanes, nearest_vehicles):
    choices = []
    ego_velocity = math.sqrt(
        ego_state.dynamic_car_state.center_velocity_2d.x**2
        + ego_state.dynamic_car_state.center_velocity_2d.y**2
    )
    # choices.append("change lane to left")
    # choices.append("keep current lane")
    if (
        nearest_vehicles["front"]
        and nearest_vehicles["front"][0][2] < ego_velocity / 2
        and adjacent_lanes["left"]
        and not nearest_vehicles["left"]
        and not nearest_vehicles["front left"]
        and not nearest_vehicles["rear left"]
    ):
        choices.append("change lane to left to overtake the lead vehicle")
        choices.append("follow the lead vehicle")
    elif (
        nearest_vehicles["front"]
        and nearest_vehicles["front"][0][2] < ego_velocity / 2
        and adjacent_lanes["right"]
        and not nearest_vehicles["right"]
        and not nearest_vehicles["front right"]
        and not nearest_vehicles["rear right"]
    ):
        choices.append("change lane to right to overtake the lead vehicle")
        choices.append("follow the lead vehicle")
    elif nearest_vehicles["front"]:
        choices.append("follow the lead vehicle")
    else:
        choices.append("keep the current lane")
    return choices

def get_choices_test():
    choices=[]
    choices.append("change lane to left")
    choices.append("keep current lane")
    return choices

