import numpy as np
from shapely.geometry import Point

from tuplan_garage.planning.simulation.planner.pdm_planner.utils.graph_search.dijkstra import (
    Dijkstra,
)


def get_scenario_analysis(
    route_roadblock_dict,
    route_lane_dict,
    adjacent_lanes,
    drivable_area_map,
    search_depth=30,
    offset=4,
):
    roadblocks = list(route_roadblock_dict.values())
    roadblock_ids = list(route_roadblock_dict.keys())

    # find current roadblock index
    start_idx = np.argmax(
        np.array(roadblock_ids) == adjacent_lanes["current_lane"].get_roadblock_id()
    )
    roadblock_window = roadblocks[start_idx : start_idx + search_depth]

    graph_search = Dijkstra(
        adjacent_lanes["current_lane"], list(route_lane_dict.keys())
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

    if curve_after_lane and curve_after_lane<len(route_plan):

        arc_lane = route_plan[curve_after_lane]
        arc_point = arc_lane.baseline_path.discrete_path[
            (len(arc_lane.baseline_path.discrete_path)) // 2
        ]
        theta = arc_point.heading + np.pi / 2
        x_left = arc_point.x + np.cos(theta) * offset
        y_left = arc_point.y + np.sin(theta) * offset
        intersecting_lanes = drivable_area_map.intersects(Point(x_left, y_left))
        # print("intersecting_lanes", intersecting_lanes)
        # is_on_lane = False
        # for lane_id in intersecting_lanes:
        #     if lane_id in self._route_lane_dict.keys():
        #         is_on_lane = True
        #         break

        if not arc_lane.adjacent_edges[0] and intersecting_lanes:
            return curve_direction, "double_direction", curve_lane_ids
    return curve_direction, "single_direction", curve_lane_ids


def scenario_turn(adjacent_lanes, curve_direction, lane_num, curve_lane_ids):
    scenario_type = "regular"
    if curve_direction == "right":
        if adjacent_lanes["current_lane"].id == curve_lane_ids[-1]:
            if lane_num == "double_direction":
                scenario_type = "special_right"
            else:
                scenario_type = "right"
        elif (len(curve_lane_ids) - 1) and adjacent_lanes[
            "current_lane"
        ].id == curve_lane_ids[-2]:
            scenario_type = "before_right"

    elif curve_direction == "left":
        if adjacent_lanes["current_lane"].id == curve_lane_ids[-1]:
            scenario_type = "left"
        elif (len(curve_lane_ids) - 1) and adjacent_lanes[
            "current_lane"
        ].id == curve_lane_ids[-2]:
            scenario_type = "before_left"

    return scenario_type
