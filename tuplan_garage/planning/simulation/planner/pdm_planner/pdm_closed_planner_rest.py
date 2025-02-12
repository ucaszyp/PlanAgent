import gc
import logging
import warnings
from typing import List, Optional, Type

from nuplan.planning.simulation.observation.observation_type import (
    DetectionsTracks,
    Observation,
)
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from tuplan_garage.planning.simulation.planner.pdm_planner.abstract_pdm_closed_planner_rest import (
    AbstractPDMClosedPlanner,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation_utils import (
    get_drivable_area_map,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.proposal.batch_idm_policy import (
    BatchIDMPolicy,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = logging.getLogger(__name__)


class PDMClosedPlanner(AbstractPDMClosedPlanner):
    """PDM-Closed planner class."""

    # Inherited property, see superclass.
    requires_scenario: bool = False

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        proposal_sampling: TrajectorySampling,
        idm_policies: BatchIDMPolicy,
        lateral_offsets: Optional[List[float]],
        map_radius: float,
        # ttc:float=2.0,
    ):
        """
        Constructor for PDMClosedPlanner
        :param trajectory_sampling: Sampling parameters for final trajectory
        :param proposal_sampling: Sampling parameters for proposals
        :param idm_policies: BatchIDMPolicy class
        :param lateral_offsets: centerline offsets for proposals (optional)
        :param map_radius: radius around ego to consider
        """
        super(PDMClosedPlanner, self).__init__(
            trajectory_sampling,
            proposal_sampling,
            idm_policies,
            lateral_offsets,
            map_radius,
            # ttc,
        )
        self._initialization = None

    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass."""
        self._iteration = 0
        self._map_api = initialization.map_api
        self._load_route_dicts(initialization.route_roadblock_ids)
        self._initialization = initialization
        gc.collect()

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def compute_planner_trajectory(
        self, current_input: PlannerInput
    ) -> AbstractTrajectory:
        """Inherited, see superclass."""

        gc.disable()
        ego_state, _ = current_input.history.current_state

        # Apply route correction on first iteration (ego_state required)
        if self._iteration == 0:
            self._route_roadblock_correction(ego_state)

        # Update/Create drivable area polygon map
        self._drivable_area_map = get_drivable_area_map(
            self._map_api, ego_state, self._map_radius
        )

        trajectory = self._get_closed_loop_trajectory(
            current_input, self._initialization
        )
        # planAgent_path="/DATA_EDS2/zyp/Workspace/llmdriver/tuplan_garage_xzb/tuplan_garage/planning/simulation/planner/pdm_planner/token_calculate/planAgent.txt"
        # dilu_path="/DATA_EDS2/zyp/Workspace/llmdriver/tuplan_garage_xzb/tuplan_garage/planning/simulation/planner/pdm_planner/token_calculate/dilu.txt"
        # gptdriver_path="/DATA_EDS2/zyp/Workspace/llmdriver/tuplan_garage_xzb/tuplan_garage/planning/simulation/planner/pdm_planner/token_calculate/gptdriver.txt"
        # assist_path="/DATA_EDS2/zyp/Workspace/llmdriver/tuplan_garage_xzb/tuplan_garage/planning/simulation/planner/pdm_planner/token_calculate/assist.txt"
        # assit_path_low="/DATA_EDS2/zyp/Workspace/llmdriver/tuplan_garage_xzb/tuplan_garage/planning/simulation/planner/pdm_planner/token_calculate/assist_low.txt"
        # if self._iteration==141:
        #     with open(assist_path,'a') as f:
        #         f.write(f"{self.assist_token}\n")
        #     with open(assit_path_low,'a') as f:
        #         f.write(f"{self.assist_low_token}\n")
            # with open(planAgent_path,'a') as f:
            #     f.write(f"{self.planAgent_token}\n")
            # with open(dilu_path,'a') as f:
            #     f.write(f"{self.dilu_token}\n")
            # with open(gptdriver_path,'a') as f:
            #     f.write(f"{self.gptdriver_token}\n")

        self._iteration += 1
        return trajectory
