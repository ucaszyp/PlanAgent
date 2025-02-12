SPLIT=val14_split_rest #val14_split test14_hard
CHALLENGE=closed_loop_nonreactive_agents # closed_loop_nonreactive_agents, closed_loop_reactive_agents

python /DATA_EDS2/zyp/Workspace/llmdriver/nuplan-devkit/nuplan/planning/script/run_simulation.py \
    +simulation=$CHALLENGE \
    planner=pdm_score_planner_rest \
    scenario_filter=$SPLIT \
    scenario_builder=nuplan \
    max_callback_workers=3 \
    experiment_uid=test14_val \
    hydra.searchpath="[pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]" \
    worker.threads_per_node=16

SPLIT=val14_split_multiple #val14_split test14_hard
CHALLENGE=closed_loop_nonreactive_agents # closed_loop_nonreactive_agents, closed_loop_reactive_agents

python /DATA_EDS2/zyp/Workspace/llmdriver/nuplan-devkit/nuplan/planning/script/run_simulation.py \
    +simulation=$CHALLENGE \
    planner=pdm_score_planner_multiple \
    scenario_filter=$SPLIT \
    scenario_builder=nuplan \
    max_callback_workers=3 \
    experiment_uid=test14_val \
    hydra.searchpath="[pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]" \
    worker.threads_per_node=16

SPLIT=val14_split_turn #val14_split test14_hard
CHALLENGE=closed_loop_nonreactive_agents # closed_loop_nonreactive_agents, closed_loop_reactive_agents

python /DATA_EDS2/zyp/Workspace/llmdriver/nuplan-devkit/nuplan/planning/script/run_simulation.py \
    +simulation=$CHALLENGE \
    planner=pdm_score_planner_turn \
    scenario_filter=$SPLIT \
    scenario_builder=nuplan \
    max_callback_workers=3 \
    experiment_uid=test14_val \
    hydra.searchpath="[pkg://tuplan_garage.planning.script.config.common, pkg://tuplan_garage.planning.script.config.simulation, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]" \
    worker.threads_per_node=16