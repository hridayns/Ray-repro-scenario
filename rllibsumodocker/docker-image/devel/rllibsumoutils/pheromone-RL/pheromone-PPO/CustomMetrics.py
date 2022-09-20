"""Example of using RLlib's debug callbacks.
Here we use callbacks to track the average CartPole pole angle magnitude as a
custom metric.
"""

from typing import Dict
import argparse
import pickle
import numpy as np
import os
from copy import deepcopy
import ray
from ray import tune
# from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch

import pathlib

# if os.path.exists('./data/raw_pf_state.npy'):
    # RAW_PF = np.load('./data/raw_pf_state.npy')
# else:
    # RAW_PF = []

ACTIONS_TO_SUMO = {
    0: -1, # right
    1: 0, # stay
    2: +1 # left
}


class MyCallbacks(DefaultCallbacks):
    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Make sure this episode has just been started (only initial obs
        # logged so far).
        assert episode.length == 0, (
            "ERROR: `on_episode_start()` callback should be called right "
            "after env reset!"
        )
        print("episode {} (env-idx={}) started.".format(episode.episode_id, env_index))

        episode.user_data['env'] = base_env.get_unwrapped()[0]# get_sub_environments
        # print(base_env.get_sub_environments())
        episode.hist_data["raw_pf"] = []
        episode.hist_data["lcs_to"] = []
        episode.hist_data["lcs_from"] = []
        # episode.hist_data["raw_pf_pos"] = []
        # episode.hist_data["lcs_to_pos"] = []
        # episode.hist_data["lcs_from_pos"] = []

        # episode.hist_data["dist_lcs_to"] = []
        # episode.hist_data["dist_lcs_from"] = []

        # episode.custom_metrics["successful_lcs_total"] = 0
        # episode.custom_metrics["failed_lcs_total"] = 0
        # episode.custom_metrics["failed_lcs_invalid_action"] = 0
        # episode.custom_metrics["failed_lcs_invalid_lane"] = 0
        
        # episode.hist_data["failed_lcs_invalid_action"] = 0
        # episode.hist_data["failed_lcs_invalid_lane"] = []

        episode.user_data['tot_episode_num_fp'] = os.path.join(pathlib.Path(__file__).parent.absolute(),'data','tot_episode_num.npy')

        if os.path.exists(episode.user_data['tot_episode_num_fp']):
            episode.user_data['tot_episode_num'] = np.load(episode.user_data['tot_episode_num_fp'])
        else:
            episode.user_data['tot_episode_num'] = 0

    def on_episode_step(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Make sure this episode is ongoing.
        assert episode.length > 0, (
            "ERROR: `on_episode_step()` callback should not be called right "
            "after env reset!"
        )

        # for agent in list(episode._agent_to_index.keys()):
        #     invalid_lane = episode.last_info_for(agent)['invalid_lane']
        #     lc_success = episode.last_info_for(agent)['lc_success']

        #     if lc_success and not invalid_lane:
        #         episode.custom_metrics["successful_lcs_total"] += 1
        #     elif not lc_success and not invalid_lane:
        #         episode.custom_metrics["failed_lcs_invalid_action"] += 1
        #     elif not lc_success and invalid_lane:
        #         episode.custom_metrics["failed_lcs_invalid_lane"] += 1
            


        # l = int(episode.user_data['env'].road_len)
        nl = episode.user_data['env'].num_lanes
        nz = episode.user_data['env'].num_zones
        lcs_to = np.zeros( (nl,nz) ).tolist()
        lcs_from = np.zeros( (nl,nz) ).tolist()
        
        # lcs_to_pos = np.zeros( (nl,l) )
        # lcs_from_pos = np.zeros( (nl,l) )
        # pf_pos = np.zeros( (nl,l) )
        pf = episode.user_data['env'].get_pheromone_field() # get values of pheromone_field at this step
        
        # dist_lcs_to = np.zeros( (nl,1) ).tolist()
        # dist_lcs_from = np.zeros( (nl,1) ).tolist()

        for agent in list(episode._agent_to_index.keys()):
            if 'lane_idx' in episode.last_raw_obs_for(agent):
                lane = episode.last_raw_obs_for(agent)['lane_idx']
            else:
                if 'lane_idx' in episode.last_info_for(agent):
                    lane = episode.last_info_for(agent)['lane_idx']
                else:
                    lane = 0
            if episode.last_info_for(agent):
                if 'zone_pos' in episode.last_info_for(agent):
                    zone, pos = episode.last_info_for(agent)['zone_pos']
                    pos = round(pos)
                elif episode.last_done_for(agent):
                    zone = nz - 1
                    pos = 1000 - 1
                else:
                    zone = 0
                    pos = 0

                if 'lc_success' in episode.last_info_for(agent) and 'reason' in episode.last_info_for(agent):
                    info_action = episode.last_info_for(agent)['action']
                    lc_success = episode.last_info_for(agent)['lc_success']
                    reason = episode.last_info_for(agent)['reason']
                    if lc_success and reason in ['can']:#, 'stay']:
                        # print('NUMBER OF LANES: {}'.format(nl))
                        # print('ZONE: {}'.format(zone))
                        # print('LANE THAT CAR IS LEAVING FROM: {}'.format(lane))
                        # print('LANE THAT CAR WILL END UP IN: {}'.format(lane + ACTIONS_TO_SUMO[info_action]))
                        # print('LANE THAT CAR WILL END UP IN AFTER ADJUSTMENT: {}'.format(nl - lane - 1 - ACTIONS_TO_SUMO[info_action]))
                        # print('LANE THAT CAR WILL GO FROM AFTER ADJUSTMENT: {}'.format(nl - lane - 1))
                        lcs_to[nl - lane - 1 - ACTIONS_TO_SUMO[info_action]][zone] += 1
                        lcs_from[nl - lane - 1][zone] += 1

                        # lcs_to_pos[nl - lane - 1 - ACTIONS_TO_SUMO[info_action]][pos] += 1
                        # lcs_from_pos[nl - lane - 1][pos] += 1
                        # pf_pos[nl-lane-1][pos] = pf[nl-lane-1][zone]
                        
        # print('############# LANE CHANGES FIELD #############')
        # for i in range(nl):
        #     for j in range(nz):
        #         print('==={}===|'.format(lcs_to[i][j]), end='')
        #     print()

        episode.hist_data["lcs_to"].append(lcs_to)
        episode.hist_data["lcs_from"].append(lcs_from)
        # episode.hist_data["lcs_to_pos"].append(lcs_to_pos)
        # episode.hist_data["lcs_from_pos"].append(lcs_from_pos)
        # print('episode {}'.format(episode.length))
        episode.hist_data['raw_pf'].append(pf)
        # episode.hist_data['raw_pf_pos'].append(pf_pos)


    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        # Check if there are multiple episodes in a batch, i.e.
        # "batch_mode": "truncate_episodes".
        if worker.policy_config["batch_mode"] == "truncate_episodes":
            # Make sure this episode is really done.
            assert episode.batch_builder.policy_collectors["default_policy"].batches[
                -1
            ]["dones"][-1], (
                "ERROR: `on_episode_end()` should only be called "
                "after episode is done!"
            )
        print(
            "episode {} (env-idx={}) ended with length {} and data {}".format(
                episode.episode_id, env_index, episode.length, len(episode.hist_data)
            )
        )

        episode.user_data['tot_episode_num'] += 1
        # episode.custom_metrics['failed_lcs_total'] = episode.custom_metrics['failed_lcs_invalid_action'] + episode.custom_metrics['failed_lcs_invalid_lane']



        S = episode.user_data['env'].rndgen.bit_generator.state
        with open('rng_state.pickle', 'wb') as handle:
            pickle.dump(S, handle, protocol=pickle.HIGHEST_PROTOCOL)




        with open(episode.user_data['tot_episode_num_fp'], 'wb') as f:
            np.save(f ,np.array(episode.user_data['tot_episode_num']))

        print('TOTAL EPISODES: {}'.format(episode.user_data['tot_episode_num']))

    ###############

        raw_pf_states_dir = os.path.join(pathlib.Path(__file__).parent.absolute(),'data','raw_pf_states')
        os.makedirs(raw_pf_states_dir, exist_ok=True)

        raw_pf_state_fp = os.path.join(pathlib.Path(__file__).parent.absolute(),'data','raw_pf_states','raw_pf_state_{}.npy'.format(episode.user_data['tot_episode_num']))
        
        with open(raw_pf_state_fp, 'wb') as f:
            np.save(f ,np.array(episode.hist_data['raw_pf']))

    ###############
        lcs_to = os.path.join(pathlib.Path(__file__).parent.absolute(),'data','lcs_to')
        os.makedirs(lcs_to, exist_ok=True)

        lcs_to_fp = os.path.join(pathlib.Path(__file__).parent.absolute(),'data','lcs_to','lcs_to_{}.npy'.format(episode.user_data['tot_episode_num']))

        with open(lcs_to_fp, 'wb') as f:
            np.save(f ,np.array(episode.hist_data['lcs_to']))

    ###############

        lcs_from = os.path.join(pathlib.Path(__file__).parent.absolute(),'data','lcs_from')
        os.makedirs(lcs_from, exist_ok=True)

        lcs_from_fp = os.path.join(pathlib.Path(__file__).parent.absolute(),'data','lcs_from','lcs_from_{}.npy'.format(episode.user_data['tot_episode_num']))

        with open(lcs_from_fp, 'wb') as f:
            np.save(f ,np.array(episode.hist_data['lcs_from']))

    # ###############

    #     raw_pf_states_pos_dir = os.path.join(pathlib.Path(__file__).parent.absolute(),'data','raw_pf_states_pos')
    #     os.makedirs(raw_pf_states_pos_dir, exist_ok=True)

    #     raw_pf_state_pos_fp = os.path.join(pathlib.Path(__file__).parent.absolute(),'data','raw_pf_states_pos','raw_pf_state_pos_{}.npy'.format(episode.user_data['tot_episode_num']))
        
    #     with open(raw_pf_state_pos_fp, 'wb') as f:
    #         np.save(f ,np.array(episode.hist_data['raw_pf_pos']))

    # ###############
    #     lcs_to_pos = os.path.join(pathlib.Path(__file__).parent.absolute(),'data','lcs_to_pos')
    #     os.makedirs(lcs_to_pos, exist_ok=True)

    #     lcs_to_pos_fp = os.path.join(pathlib.Path(__file__).parent.absolute(),'data','lcs_to_pos','lcs_to_pos_{}.npy'.format(episode.user_data['tot_episode_num']))

    #     with open(lcs_to_pos_fp, 'wb') as f:
    #         np.save(f ,np.array(episode.hist_data['lcs_to_pos']))

    # ###############

    #     lcs_from_pos = os.path.join(pathlib.Path(__file__).parent.absolute(),'data','lcs_from_pos')
    #     os.makedirs(lcs_from_pos, exist_ok=True)

    #     lcs_from_pos_fp = os.path.join(pathlib.Path(__file__).parent.absolute(),'data','lcs_from_pos','lcs_from_pos_{}.npy'.format(episode.user_data['tot_episode_num']))

    #     with open(lcs_from_pos_fp, 'wb') as f:
    #         np.save(f ,np.array(episode.hist_data['lcs_from_pos']))


    def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch, **kwargs):
        print("returned sample batch of size {}".format(samples.count))

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        print(
            "algorithm.train() result: {} -> {} episodes".format(
                algorithm, result["episodes_this_iter"]
            )
        )
        # you can mutate the result dict to add new fields to return
        result["callback_ok"] = True

        # print(result['info'].keys())

        # print('RESULT mean: {}'.format(result['episode_reward_mean']))
        # print('RESULT: {}'.format(result))

    def on_trial_result(self, iteration, trials, trial, result, **info):
        print(f"Got result: {result['episode_len_mean']}")

    def on_learn_on_batch(
        self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs
    ) -> None:
        result["sum_actions_in_train_batch"] = np.sum(train_batch["actions"])
        print(
            "policy.learn_on_batch() result: {} -> sum actions: {}".format(
                policy, result["sum_actions_in_train_batch"]
            )
        )

    def on_postprocess_trajectory(
        self,
        *,
        worker: RolloutWorker,
        episode: Episode,
        agent_id: str,
        policy_id: str,
        policies: Dict[str, Policy],
        postprocessed_batch: SampleBatch,
        original_batches: Dict[str, SampleBatch],
        **kwargs
    ):
        # print("postprocessed {} steps".format(postprocessed_batch.count))
        if "num_batches" not in episode.custom_metrics:
            episode.custom_metrics["num_batches"] = 0
        episode.custom_metrics["num_batches"] += 1


# if __name__ == "__main__":
#     args = parser.parse_args()

#     ray.init()
#     trials = tune.run(
#         "PG",
#         stop={
#             "training_iteration": args.stop_iters,
#         },
#         config={
#             "env": "CartPole-v0",
#             "num_envs_per_worker": 2,
#             "callbacks": MyCallbacks,
#             "framework": args.framework,
#             # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
#             "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
#         },
#     ).trials

#     # Verify episode-related custom metrics are there.
#     custom_metrics = trials[0].last_result["custom_metrics"]
#     print(custom_metrics)
#     assert "pole_angle_mean" in custom_metrics
#     assert "pole_angle_min" in custom_metrics
#     assert "pole_angle_max" in custom_metrics
#     assert "num_batches_mean" in custom_metrics
#     assert "callback_ok" in trials[0].last_result

#     # Verify `on_learn_on_batch` custom metrics are there (per policy).
#     if args.framework == "torch":
#         info_custom_metrics = custom_metrics["default_policy"]
#         print(info_custom_metrics)
#         assert "sum_actions_in_train_batch" in info_custom_metrics