#!/usr/bin/env python3

""" PPO algorithm
    Author: Hriday N Sanghvi
    This program and the accompanying materials are made available under the
    terms of the Eclipse Public License 2.0 which is available at
    http://www.eclipse.org/legal/epl-2.0.
"""

from copy import deepcopy
import logging
import pathlib
from pprint import pformat
import argparse

# from eval_marlenvironment import DEFAULT_AGENT_CONFIG
# from marlenvironment import DEFAULT_AGENT_CONFIG
# from marlenvironment import DEFAULT_SCENARIO_CONFIG
import ray
from ray import tune
from CustomMetrics import MyCallbacks

from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.ppo.ppo_tf_policy import PPOTF1Policy as PPOTFPolicy

from ray.tune.logger import pretty_print

from rllibsumoutils.sumoutils import sumo_default_config

import marlenvironment

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger('ppotrain')

parser = argparse.ArgumentParser()
parser.add_argument('--fast', help='run without GUI',
                    action='store_true')
parser.add_argument('--nopolicy', action='store_true')
parser.add_argument('-nv', '--num_veh', type=int, default=5)
parser.add_argument('-nl', '--num_lanes', type=int, default=2)
parser.add_argument('-nz', '--num_zones', type=int, default=1)
parser.add_argument('-ls', '--lane_size', type=float, default=1000)
parser.add_argument('-nb', '--num_blockages', type=int, default=1)
parser.add_argument('-bls', '--block_sizes', type=str, default='250')
parser.add_argument('-blp', '--block_pos', type=str, default='750')
parser.add_argument('-bll', '--block_lanes', type=str, default='0')
parser.add_argument('-bdur', '--block_duration', type=str, default='END')
parser.add_argument('-puf', '--pf_update_freq', type=int, default=1)
parser.add_argument('-ec', '--evaporation', type=float, default=0.5)
parser.add_argument('-df', '--diffusion', type=float, default=0.3)
parser.add_argument('--mixed_blockage_training', action='store_true')

args = parser.parse_args()

block_sizes = [float(x) for x in args.block_sizes.split(',') if x.isnumeric()]
block_pos = [float(x) for x in args.block_pos.split(',') if x.isnumeric()]
block_lanes = [int(x) for x in args.block_lanes.split(',') if x.isnumeric()]
block_dur = [int(x) if x.isnumeric() else x for x in args.block_duration.split(',')]

assert len(block_sizes) == len(block_pos) == len(block_lanes) == len(block_dur)

lane_block_settings = []
for i in range(len(block_sizes)):
    lane_block_settings.append( [block_pos[i], block_sizes[i], block_lanes[i], block_dur[i]] )

DEFAULT_SCENARIO_CONFIG = {
    'sumo_config': sumo_default_config(),
    'nopolicy': args.nopolicy,
    'num_vehicles': args.num_veh,
    'num_lanes': args.num_lanes,
    'num_zones': args.num_zones,
    'road_length': args.lane_size,
    'evaporation_coeff': args.evaporation, # % of newly calculated pheromone feedback to be used in update of pheromone field (0 - 1)
    'diffusion_factor': args.diffusion, # 0 - 1 (How much of the calculated pheromone feedback should be diffused to surroundings?)
    'puf': args.pf_update_freq,
    'pheromone_init_val': 0,

    # Training options #
    'mbt': args.mixed_blockage_training,
    ### LANE BLOCK SETTINGS ###
    'lane_block_settings': lane_block_settings,
    'agent_rnd_order': True,
    'log_level': 'WARN',
    'seed': 42,
    'misc': {
        'max_distance': 200, # [m]
    }
}

NUM_AGENTS = DEFAULT_SCENARIO_CONFIG['num_vehicles']
na = NUM_AGENTS
nl = DEFAULT_SCENARIO_CONFIG['num_lanes']
nz = DEFAULT_SCENARIO_CONFIG['num_zones']
rl = DEFAULT_SCENARIO_CONFIG['road_length']
ec = DEFAULT_SCENARIO_CONFIG['evaporation_coeff']
df = DEFAULT_SCENARIO_CONFIG['diffusion_factor']
# pfi = DEFAULT_SCENARIO_CONFIG['pheromone_init_val']
nb = len(DEFAULT_SCENARIO_CONFIG['lane_block_settings'])
bls = DEFAULT_SCENARIO_CONFIG['lane_block_settings'][0][1]
blp = DEFAULT_SCENARIO_CONFIG['lane_block_settings'][0][0]
bll = DEFAULT_SCENARIO_CONFIG['lane_block_settings'][0][2]
bdur = DEFAULT_SCENARIO_CONFIG['lane_block_settings'][0][3]
puf = DEFAULT_SCENARIO_CONFIG['puf']
# param_str = 'nv-{}_nl-{}_nz-{}_ls-{}_nb-{}_bls-{}_blp-{}_bll-{}_bdur-{}_puf-{}_ec-{}_df-{}'.format(na, nl, nz, rl, nb, bls, blp, bll, bdur, puf, ec, df)
param_str = 'nv-{}_nl-{}_nz-{}_ls-{}_nb-{}_bls-{}_blp-{}_bll-{}_bdur-{}_puf-{}_ec-{}_df-{}'.format(na, nl, nz, rl, nb, bls, blp, bll, bdur, puf, ec, df)


def trial_str_creator(trial):
    # timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    global param_str
    cfg = trial.config

    trial_str = '{}({})'.format(trial.trainable_name, param_str)

    return trial_str

def _main():
    """ Training example """

    ray.tune.registry.register_env('pheromone_env', marlenvironment.env_creator)
    # Initialize RAY.
    ray.init()

    # with dashboard alive
    # ray.init(address='auto', _redis_password='5241590000000000')

    # Load default Scenario configuration for the LEARNING ENVIRONMENT
    scenario_config = deepcopy(DEFAULT_SCENARIO_CONFIG)
    scenario_config['seed'] = 42
    scenario_config['log_level'] = 'ERROR'
    
    if args.fast:
        scenario_config['sumo_config']['sumo_connector'] = 'libsumo'
        scenario_config['sumo_config']['sumo_gui'] = False
    else:
        scenario_config['sumo_config']['sumo_connector'] = 'traci'
        scenario_config['sumo_config']['sumo_gui'] = True

    scenario_config['sumo_config']['sumo_cfg'] = '{}/scenario/sumo.cfg.xml'.format(pathlib.Path(__file__).parent.absolute())
    scenario_config['sumo_config']['routes_cfg'] = '{}/scenario/routes.rou.xml'.format(pathlib.Path(__file__).parent.absolute())
    scenario_config['sumo_config']['sumo_params'] = ['--collision.action', 'warn', '--no-step-log', 'true', '--no-warnings']
    scenario_config['sumo_config']['trace_file'] = True # False
    scenario_config['sumo_config']['end_of_sim'] = 3600 # [s]
    scenario_config['sumo_config']['update_freq'] = 1 # number of traci.simulationStep()
                                                       # for each learning step.
    scenario_config['sumo_config']['log_level'] = 'ERROR'
    logger.info('Scenario Configuration: \n %s', pformat(scenario_config))

    agent_ids = []
    for i in range(NUM_AGENTS):
        agent_ids.append('agent_{}'.format(i+1))

    ## MARL Environment Init
    env_config = {
        # 'agent_init': agent_init,
        'agent_ids' : agent_ids,
        'scenario_config': scenario_config,
    }
    marl_env = marlenvironment.SUMOTestMultiAgentEnv(env_config)

    # Model
    model = deepcopy(MODEL_DEFAULTS)
    model['fcnet_hiddens'] = [512, 512] # 256, 256
    model['fcnet_activation'] = 'tanh'
    model['conv_filters'] = None
    model['conv_activation'] = 'relu'
    model['post_fcnet_hiddens'] = []
    model['post_fcnet_activation'] = 'relu'
    model['free_log_std'] = False
    model['no_final_linear'] = False
    model['vf_share_layers'] = False
    model['use_lstm'] = False
    model['max_seq_len'] = 20
    model['lstm_cell_size'] = 256
    model['lstm_use_prev_action'] = False
    model['lstm_use_prev_reward'] = False
    model['_time_major'] = False


    # Config for PPO
    policies = {}
    policies['my_ppo'] = (PPOTFPolicy,
                           marl_env.get_obs_space('agent_1'),
                           marl_env.get_action_space('agent_1'),
                           {})


    # Algorithm.
    # my_trainer1 = PPO(env='CartPole-v0')
    my_ppo_config = PPOConfig()\
        .python_environment()\
        .resources(
            num_gpus=1,
            num_cpus_per_worker=1,
            num_gpus_per_worker=0,
        )\
        .framework(
            framework='tf',
            eager_tracing=False,
        )\
        .environment(
            env='pheromone_env',
            env_config=env_config,
            observation_space=None,
            action_space=None,
            clip_rewards=None,
            normalize_actions=False, # default is True
            clip_actions=False,
            disable_env_checking=True,
        )\
        .rollouts(
            num_rollout_workers = 1,
            num_envs_per_worker = 1,
            # sample_collector = SimpleListCollector,
            # create_env_on_local_worker = False,
            # sample_async = False,
            rollout_fragment_length = 400,
            batch_mode = 'complete_episodes',
            # horizon = None,
            # soft_horizon = False,
            # no_done_at_end = False,
            observation_filter = 'NoFilter',
        )\
        .training(
            gamma=0.99,
            lr=5e-05,
            train_batch_size=4000,
            model=model,
            lr_schedule=None,
            use_critic=True,
            use_gae=True,
            lambda_=1.0,
            kl_coeff=0.2,
            sgd_minibatch_size=128,
            num_sgd_iter=30,
            shuffle_sequences=True,
            vf_loss_coeff=1.0,
            entropy_coeff=0.0,
            entropy_coeff_schedule=None,
            clip_param=0.3,
            vf_clip_param=10,
            grad_clip=None,
            kl_target=0.01,
        )\
        .callbacks(MyCallbacks)\
        .exploration(
            explore=True,
            exploration_config={'type': 'StochasticSampling'}
        )\
        .multi_agent(
            policies = policies,
            policy_map_capacity = 100,
            policy_map_cache = None,
            policy_mapping_fn = lambda agent_id: 'my_ppo',
            policies_to_train = ['my_ppo'],
            observation_fn = None,
            replay_mode = 'independent',
            count_steps_by = 'env_steps',
        )\
        .offline_data(
            # postprocess_inputs=False,
        )\
        .evaluation(
            evaluation_interval = 10,
            evaluation_duration = 10,
            evaluation_duration_unit = 'episodes',
            # evaluation_sample_timeout_s = 180.0,
            evaluation_parallel_to_training = False,
            evaluation_config = {
               'explore': False,
               'exploration_config' : {'type': 'StochasticSampling'}
            },
            evaluation_num_workers = 1,
            # custom_evaluation_function = None
            always_attach_evaluation_results = True,
            # in_evaluation = False,
            # sync_filters_on_rollout_workers_timeout_s = 60.0
            evaluation_sample_timeout_s=7200,
        )\
        .reporting(
            keep_per_episode_custom_metrics = True, # default is False
            metrics_episode_collection_timeout_s = 60.0,
            metrics_num_episodes_for_smoothing = 100,
            min_time_s_per_iteration = 300,
            min_train_timesteps_per_iteration = 0,
            min_sample_timesteps_per_iteration = 0,
        )\
        .debugging(
            log_level='WARN',
            seed=42
        )

    # my_trainer = my_ppo_config.build()
    # my_trainer.train()
    stop = tune.stopper.TrialPlateauStopper(
        metric='episode_reward_mean',
        std=0.01,
        num_results=4,
        grace_period=4,
    )
    
    result = tune.run(
        'PPO',
        name='ma-pheromone-ray2',
        config=my_ppo_config.to_dict(),
        metric='episode_len_mean',
        mode='min',
        trial_name_creator=trial_str_creator,
        keep_checkpoints_num=2,
        checkpoint_score_attr='min-episode_len_mean',
        stop=stop,
        verbose=0, # Set > 0 for more information
        # fail_fast='raise',
        log_to_file=('train.log','error.log'),
        checkpoint_freq=10,
        checkpoint_at_end=True,
        # local_dir='/media/hridayns/Seagate\ Expansion\ Drive/free_space',
        local_dir='~/devel/rllibsumoutils/pheromone-RL/pheromone-PPO/tune-results',
        # restore='~/devel/rllibsumoutils/pheromone-RL/pheromone-PPO/tune-results/ma-pheromone/PPO_nv-1000_nl-3_nz-10_ls-5000.0_nb-1_bls-500.0_blp-2750.0_bll-0_bdur-END_puf-1_ec-0.3_df-0.5__0_2022-09-10_12-31-05/checkpoint_000015/checkpoint-15',        
        resume='LOCAL+ERRORED',
    )

    ray.shutdown()

if __name__ == '__main__':
    _main()