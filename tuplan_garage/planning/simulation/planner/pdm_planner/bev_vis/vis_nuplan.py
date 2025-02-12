import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from immutabledict import immutabledict
# import jax
# from waymax.waymax.visualization import

_RoadGraphShown = (1, 2, 3, 15, 16, 17, 18, 19)
_RoadGraphDefaultColor = (0.9, 0.9, 0.9)

TRAFFIC_LIGHT_COLORS = {
    # Unknown = 0, Arrow_Stop = 1, Arrow_Caution = 2, Arrow_Go = 3, Stop = 4,
    # Caution = 5, Go = 6, Flashing_Stop = 7, Flashing_Caution = 8
    # third_party/waymo_open_dataset/protos/map.proto
    0: [0.75, 0.75, 0.75],
    1: [1.0, 0.0, 0.0],
    2: [1.0, 1.0, 0.0],
    3: [0.0, 1.0, 0.0],
    4: [1.0, 0.0, 0.0],
    5: [1.0, 1.0, 0.0],
    6: [0.0, 1.0, 0.0],
    7: [1.0, 1.0, 0.0],
    8: [1.0, 1.0, 0.0],
}

ROAD_GRAPH_COLORS = {
    # Consistent with MapElementIds
    1: np.array([230, 230, 230]) / 255.0,  # 'LaneCenter-Freeway',
    2: np.array([230, 230, 230]) / 255.0,  # 'LaneCenter-SurfaceStreet',
    3: np.array([230, 230, 230]) / 255.0,  # 'LaneCenter-BikeLane',
    6: np.array([140, 230, 255]) / 255.0,  # 'RoadLine-BrokenSingleWhite',
    7: np.array([89, 219, 255]) / 255.0,  # 'RoadLine-SolidSingleWhite',
    8: np.array([89, 219, 255]) / 255.0,  # 'RoadLine-SolidDoubleWhite',
    9: np.array([241, 153, 255]) / 255.0,  # 'RoadLine-BrokenSingleYellow',
    10: np.array([241, 153, 255]) / 255.0,  # 'RoadLine-BrokenDoubleYellow'
    11: np.array([120, 120, 120]) / 255.0,  # 'RoadLine-SolidSingleYellow',
    12: np.array([120, 120, 120]) / 255.0,  # 'RoadLine-SolidDoubleYellow',
    13: np.array([120, 120, 120]) / 255.0,  # 'RoadLine-PassingDoubleYellow',
    15: np.array([80, 80, 80]) / 255.0,  # 'RoadEdgeBoundary',
    16: np.array([80, 80, 80]) / 255.0,  # 'RoadEdgeMedian',
    17: np.array([255, 0, 0]) / 255.0,  # 'StopSign',  # One point
    18: np.array([200, 200, 200]) / 255.0,  # 'Crosswalk',  # Polygon
    19: np.array([200, 200, 200]) / 255.0,  # 'SpeedBump',  # Polygon
}


COLOR_DICT = immutabledict({
    # RGB color:
    'context': np.array([0.6, 0.6, 0.6]),  # Context agents, grey.
    'controlled': np.array([0, 0.6, 0.8]),  # Modeled agents, dark blue.
    'history': np.array([0.8, 0.8, 0.8]),  # Grey for history.
    'overlap': np.array([1.0, 0.0, 0.0]),  # Red for overlap
})

class VizConfig:
  """Config for visualization."""

  front_x: float = 75.0
  back_x: float = 75.0
  front_y: float = 75.0
  back_y: float = 75.0
  px_per_meter: float = 4.0
  show_agent_id: bool = True
  center_agent_idx: int = -1  # -1 for SDC
  verbose: bool = True


def init_fig_ax_via_size(
    x_px: float, y_px: float
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
  """Initializes a figure with given size in pixel."""
  fig, ax = plt.subplots()
  # Sets output image to pixel resolution.
  dpi = 50
  fig.set_size_inches([x_px / dpi, y_px / dpi])
  fig.set_dpi(dpi)
  fig.set_facecolor('white')
  return fig, ax


def init_fig_ax(
    vis_config: VizConfig = VizConfig(),
):
  """Initializes a figure with vis_config."""
  return init_fig_ax_via_size(
      (vis_config.front_x + vis_config.back_x) * vis_config.px_per_meter,
      (vis_config.front_y + vis_config.back_y) * vis_config.px_per_meter,
  )

def plot_trajectory(
    ax,
    traj,
    is_controlled: np.ndarray,
    time_idx,
    indices,
) -> None:
  """Plots a Trajectory with different color for controlled and context.

  Plots the full bounding_boxes only for time_idx step, overlap is
  highlighted.

  Notation: A: number of agents; T: numbe of time steps; 5 degree of freedom:
  center x, center y, length, width, yaw.

  Args:
    ax: matplotlib axes.
    traj: a Trajectory with shape (A, T).
    is_controlled: binary mask for controlled object, shape (A,).
    time_idx: step index to highlight bbox, -1 for last step. Default(None) for
      not showing bbox.
    indices: ids to show for each agents if not None, shape (A,).
  """
  if len(traj.shape) != 2:
    raise ValueError('traj should have shape (A, T)')

  traj_5dof = np.array(
      traj.stack_fields(['x', 'y', 'length', 'width', 'yaw'])
  )  # Forces to np from jnp

  num_obj, num_steps, _ = traj_5dof.shape
  if time_idx is not None:
    if time_idx == -1:
      time_idx = num_steps - 1
    if time_idx >= num_steps:
      raise ValueError('time_idx is out of range.')

  # Adds id if needed.
  if indices is not None and time_idx is not None:
    for i in range(num_obj):
      if not traj.valid[i, time_idx]:
        continue
      ax.text(
          traj_5dof[i, time_idx, 0] - 2,
          traj_5dof[i, time_idx, 1] + 2,
          f'{indices[i]}',
          zorder=10,
      )
  plot_bounding_boxes(ax, traj_5dof, time_idx, is_controlled, traj.valid)

def plot_bounding_boxes(
    ax,
    traj_5dof,
    time_idx,
    is_controlled,
    valid
) -> None:
  """Helper function to plot multiple bounding boxes across time."""
  # Plots bounding boxes (traj_5dof) with shape: (A, T)
  # is_controlled: (A,)
  # valid: (A, T)
  valid_controlled = is_controlled[:, np.newaxis] & valid
  valid_context = ~is_controlled[:, np.newaxis] & valid

  num_obj = traj_5dof.shape[0]
  time_indices = np.tile(
      np.arange(traj_5dof.shape[1])[np.newaxis, :], (num_obj, 1)
  )
  # Shrinks bounding_boxes for non-current steps.
  traj_5dof[time_indices != time_idx, 2:4] /= 10
  plot_numpy_bounding_boxes(
      ax=ax,
      bboxes=traj_5dof[(time_indices >= time_idx) & valid_controlled],
      color=COLOR_DICT['controlled'],
  )

  plot_numpy_bounding_boxes(
      ax=ax,
      bboxes=traj_5dof[(time_indices < time_idx) & valid],
      color=COLOR_DICT['history'],
      as_center_pts=True,
  )

  plot_numpy_bounding_boxes(
      ax=ax,
      bboxes=traj_5dof[(time_indices >= time_idx) & valid_context],
      color=COLOR_DICT['context'],
  )

  # Shows current overlap
  # (A, A)
  # overlap_fn = jax.jit(geometry.compute_pairwise_overlaps)
  # overlap_mask_matrix = overlap_fn(traj_5dof[:, time_idx])
  # Remove overlap against invalid objects.
  # overlap_mask_matrix = np.where(
  #     valid[None, :, time_idx], overlap_mask_matrix, False
  # )
  # (A,)
  # overlap_mask = np.any(overlap_mask_matrix, axis=1)

  # plot_numpy_bounding_boxes(
  #   ax=ax,
  #   bboxes=traj_5dof[:, time_idx][overlap_mask & valid[:, time_idx]],
  #   color=COLOR_DICT['overlap'],
  # )

def plot_roadgraph_points(
    ax,
    rg_pts,
    verbose: bool = False,
) -> None:
  """Plots road graph as points.

  Args:
    ax: matplotlib axes.
    rg_pts: a RoadgraphPoints with shape (1,)
    verbose: print roadgraph points count if set to True.
  """
  if len(rg_pts.shape) != 1:
    raise ValueError(f'Roadgraph should be rank 1, got {len(rg_pts.shape)}')
  if rg_pts.valid.sum() == 0:
    return
  elif verbose:
    print(f'Roadgraph points count: {rg_pts.valid.sum()}')

  xy = rg_pts.xy[rg_pts.valid]
  rg_type = rg_pts.types[rg_pts.valid]
  for curr_type in np.unique(rg_type):
    if curr_type in _RoadGraphShown:
      p1 = xy[rg_type == curr_type]
      rg_color = ROAD_GRAPH_COLORS.get(curr_type, _RoadGraphDefaultColor)
      ax.plot(p1[:, 0], p1[:, 1], '.', color=rg_color, ms=2)

def plot_traffic_light_signals_as_points(
    ax,
    tls,
    timestep: int = 0,
    verbose: bool = False,
) -> None:
  """Plots traffic lights for timestep.

  Args:
    ax: matplotlib axes.
    tls: a TrafficLightStates to show.
    timestep: draw traffi lights at this given timestep.
    verbose: print traffic lights count if set to True.
  """
  if len(tls.shape) != 2:
    raise ValueError('Traffic light shape wrong.')

  valid = tls.valid[:, timestep]
  if valid.sum() == 0:
    return
  elif verbose:
    print(f'Traffic lights count: {valid.sum()}')

  tls_xy = tls.xy[:, timestep][valid]
  tls_state = tls.state[:, timestep][valid]

  for xy, state in zip(tls_xy, tls_state):
    tl_color = TRAFFIC_LIGHT_COLORS[int(state)]
    ax.plot(xy[0], xy[1], marker='o', color=tl_color, ms=4)

def plot_numpy_bounding_boxes(
    ax,
    bboxes,
    color,
    alpha,
    as_center_pts: bool = False,
) -> None:
  """Plots multiple bounding boxes.

  Args:
    ax: Fig handles.
    bboxes: Shape (num_bbox, 5), with last dimension as (x, y, length, width,
      yaw).
    color: Shape (3,), represents RGB color for drawing.
    alpha: Alpha value for drawing, i.e. 0 means fully transparent.
    as_center_pts: If set to True, bboxes will be drawn as center points,
      instead of full bboxes.
  """
  if bboxes.ndim != 2 or bboxes.shape[1] != 5 or color.shape != (3,):
    raise ValueError(
        (
            'Expect bboxes rank 2, last dimension of bbox 5, color of size 3,'
            ' got{}, {}, {} respectively'
        ).format(bboxes.ndim, bboxes.shape[1], color.shape)
    )

  if as_center_pts:
    ax.plot(bboxes[:, 0], bboxes[:, 1], 'o', color=color, ms=2, alpha=alpha)
  else:
    c = np.cos(bboxes[:, 4])
    s = np.sin(bboxes[:, 4])
    pt = np.array((bboxes[:, 0], bboxes[:, 1]))  # (2, N)
    length, width = bboxes[:, 2], bboxes[:, 3]
    u = np.array((c, s))
    ut = np.array((s, -c))

    # Compute box corner coordinates.
    tl = pt + length / 2 * u - width / 2 * ut
    tr = pt + length / 2 * u + width / 2 * ut
    br = pt - length / 2 * u + width / 2 * ut
    bl = pt - length / 2 * u - width / 2 * ut

    # Compute heading arrow using center left/right/front.
    cl = pt - width / 2 * ut
    cr = pt + width / 2 * ut
    cf = pt + length / 2 * u

    # Draw bboxes.
    ax.plot(
        [tl[0, :], tr[0, :], br[0, :], bl[0, :], tl[0, :]],
        [tl[1, :], tr[1, :], br[1, :], bl[1, :], tl[1, :]],
        color=color,
        zorder=4,
        alpha=alpha,
    )

    # Draw heading arrow.
    ax.plot(
        [cl[0, :], cr[0, :], cf[0, :], cl[0, :]],
        [cl[1, :], cr[1, :], cf[1, :], cl[1, :]],
        color=color,
        zorder=4,
        alpha=alpha,
    )
