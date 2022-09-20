""" Example MARL Environment for RLLIB + SUMO Utlis
    Author: Lara CODECA
    This program and the accompanying materials are made available under the
    terms of the Eclipse Public License 2.0 which is available at
    http://www.eclipse.org/legal/epl-2.0.
"""

import collections
import logging
import math
import os
from os.path import exists as file_exists

import sys
from copy import deepcopy
from pprint import pformat
from time import time

import gym
import numpy as np
# from numpy.random import RandomState
from ray.rllib.env import MultiAgentEnv
from rllibsumoutils.sumoutils import SUMOUtils
import pickle
from numpy.random import default_rng

# """ Import SUMO library """
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
    import traci.constants as tc
    from traci.exceptions import TraCIException
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

####################################################################################################

logger = logging.getLogger(__name__)

####################################################################################################

def env_creator(config):
    """ Environment creator used in the environment registration. """
    logger.info('Environment creation: SUMOTestMultiAgentEnv')
    return SUMOTestMultiAgentEnv(config)

####################################################################################################

### SIMULATION TIME SETTINGS ###
STEP = 0
START = 0.0
END = 86400.0
ACTION_STEPS = 1

### LANE BLOCK SETTINGS ###
BLOCK_POS = 1750.0
BLOCK_LENGTH = 400.0
BLOCK_LANE = 0

BLOCK_POS2 = 750.0
BLOCK_LENGTH2 = 700.0
BLOCK_LANE2 = 1

BLOCK_POS3 = 1450.0
BLOCK_LENGTH3 = 500.0
BLOCK_LANE3 = 2

### LANE CHANGING SETTINGS ###
# LC_MODE = 1536 # 2560 # 1024 + 513 # 512 # 1541 # 1621
# LC_MODE_DONOCOOPCHANGES = 1617
ACTION_TIMEOUT_VAL = 0
NUMBER_OF_ACTIONS = 4 # stay on same lane, go to the left lane, go to the right lane, do the cooperative move

### SCENARIO SETTINGS ###
LENGTH_OF_ROAD = 1000
NUMBER_OF_VEH = 100
NUMBER_OF_LANES = 3
NUMBER_OF_VEH_PER_HR = 10000

### PHEROMONE SETTINGS ###
NUMBER_OF_ZONES = 4
SIZE_OF_ZONE = LENGTH_OF_ROAD // NUMBER_OF_ZONES
EVAPORATION_COEFF = 0.1 # % of newly calculated pheromone feedback to be used in update of pheromone field (0 - 1)
DIFFUSION_FACTOR = 1 # 0 - 1 (How much of the calculated pheromone feedback should be diffused to surroundings?)
PHEROMONE_INIT_VAL = 0

class SUMOSimulationWrapper(SUMOUtils):
    """ A wrapper for the interaction with the SUMO simulation """

    def _initialize_simulation(self):
        """ Specific simulation initialization. """
        try:
            super()._initialize_simulation()
        except NotImplementedError:
            pass
        
        self.vehicles_tagged = set()

    def _initialize_metrics(self):
        """ Specific metrics initialization """
        try:
            super()._initialize_metrics()
        except NotImplementedError:
            pass
        self.veh_subscriptions = dict()
        self.collisions = collections.defaultdict(int)

    def agent_to_veh_id_mapping_interface(agent):
        pass

    def spawn_blockage(self, lane_block_settings):
        block_id_prefix = 'ghost'
        for i, lbs in enumerate(lane_block_settings):
            if i == 0:
                block_id = block_id_prefix
            else:
                block_id = ''.join([block_id_prefix, str(i+1)])
            
            pos, size, lane, dur = lbs

            self.traci_handler.vehicle.add(vehID=block_id, routeID='r0', typeID=block_id, depart=0, departLane=lane, departPos=pos, departSpeed='random', arrivalLane='current', arrivalPos=pos)

            self.traci_handler.vehicle.setColor(typeID=block_id,color=(255,0,0))
            self.traci_handler.vehicle.setLength(typeID=block_id,length=size)

            if dur == 'END':
                dur = END

            self.traci_handler.vehicle.setStop(vehID=block_id, edgeID='1f2', pos=float(pos), laneIndex=int(lane), duration=float(10), until=float(dur))# flags=0, startPos=0, until=120)
            # self.traci_handler.vehicle.setStop(vehID=block_id, edgeID='1f2', pos=pos, laneIndex=int(lane), duration=10, until=dur)# flags=0, startPos=0, until=120)

    def _default_step_action(self, agents=None):
        """ Specific code to be executed in every simulation step """
        try:
            super()._default_step_action(agents)
        except NotImplementedError:
            pass
        

        # for vh in self.veh_subscriptions.keys():
            # self.agents

        # get collisions
        # collisions = self.traci_handler.simulation.getCollidingVehiclesIDList()
        # logger.debug('Collisions: %s', pformat(collisions))
        # for veh in collisions:
            # self.collisions[veh] += 1
        # get subscriptions
        self.veh_subscriptions = self.traci_handler.vehicle.getAllSubscriptionResults()
        for veh, vals in self.veh_subscriptions.items():
            logger.debug('Subs: %s, %s', pformat(veh), pformat(vals))
        running = set()

        for agent in self.veh_subscriptions:
            running.add(agent)
        
        if len(running) == 0:
            logger.info('All the agent left the simulation..')
            self.end_simulation()
        return True

####################################################################################################
class PheromoneFieldVal(object):
    def __init__(self, m = 0, n = 0):
        self.current_mean = m
        self.n = n
    def update_mean1(self, new_val):
        print('CURRENT MEAN: {}'.format(self.current_mean))
        print('Number of observations: {}'.format(self.n))
        self.n += 1
        print('NEW VALUE BEING ADDED: {}'.format(new_val))
        print('NEW Number of observations: {}'.format(self.n))

        print('NEW MEAN = ( ({} * {}) + {} ) / {}'.format(self.current_mean, self.n-1, new_val, self.n))
        self.current_mean = ((self.current_mean * (self.n - 1) ) + new_val) / self.n
        print('NEW MEAN: {}'.format(self.current_mean))
        # input()
    
    def update_mean(self, new_val):
        self.current_mean = new_val

    def diffuse(self, diffused_val):
        self.current_mean += diffused_val

    def val(self):
        return self.current_mean

class Pheromone(object):
    def __init__(self, percep_h_top = 0, percep_h_bottom = 0, percep_w = 1, default_val = 0.0):
        self.range = {
            'top' : percep_h_top,
            'bot': percep_h_bottom,
            'adj': percep_w,
        }
        self.default_val = default_val
        self.clear()

    def set(self, x, y, val):
        self.val[x, y] = val

    def get(self):
        return self.val

    def clear(self):
        H = self.range['top'] + self.range['bot'] + 1
        W = 2 * self.range['adj'] + 1

        self.val = np.full((W, H), self.default_val)

class SUMOAgent(object):
    """ Agent implementation. """

    def __init__(self, agent, config):
        self.agent_id = agent
        self.config = config
        self.action_to_meaning = self.config['action_to_meaning']

        self.LC_ACTION_TIMEOUT = 0

        self.prev_action = None
        
        self.lane = None
        self.zone = None
        self.prev_lane = None
        self.prev_zone = None
        self.target_lane = None

        self.prev_accel = None
        self.prev_speed = None

        self.action_mapping = dict()
        for pos, action in enumerate(config['actions']):
            self.action_mapping[pos] = config['actions'][action]
        logger.debug('Agent "%s" configuration \n %s', self.agent_id, pformat(self.config))

        self.pheromone = Pheromone()

    def get_id(self):
        return self.ID
    def get_pheromone(self):
        return self.pheromone

    def max_normalized_softmax(self, data):
        # print('RAW DATA : {}'.format(data))
        data -= np.max(data) # largest value shifted to 0; max normalization
        # print('SHIFT TO 0 : {}'.format(data))
        exp = np.exp(data)
        # print('EXP : {}'.format(exp))
        p = exp / np.sum(exp)
        # print('PROBABILITY : {}'.format(p))
        return p.flatten()

    def pf_to_prob(self, x, y):
        pf_percep = self.pheromone.get()

        actions = self.config['actions']
        action_sum = np.full( (len(actions.values()), 1), 0, dtype=np.float ) # [0 for i in range()]
        
        # print(np.sum(pf_percep, axis=1, keepdims=True))

        for action in list(actions.values()):
            if x-action > x: 
                # print('GO RIGHT')
                action_sum[0][0] = np.sum(pf_percep[x+1:])

            elif x-action < x:
                # print('GO LEFT')
                action_sum[2][0] = np.sum(pf_percep[:x])

            else:
                # print('STAY')
                action_sum[1][0] = np.sum(pf_percep[x])

        # print(action_sum)
        return self.max_normalized_softmax(data = action_sum)

    def step(self, action, sumo_handler):
        """ Implements the logic of each specific action passed as input. """
        logger.debug('Agent %s: action %d', self.agent_id, action)
        # Subscriptions EXAMPLE:
        #     {'agent_0': {64: 14.603468282230542, 104: None},
        #      'agent_1': {64: 12.922797055918513,
        #                  104: ('veh.19', 27.239870121802596)}}
        # logger.debug('Subscriptions: %s', pformat(sumo_handler.veh_subscriptions[self.config['veh_id']]))
        
        vh = self.config['veh_id']
        if action == 3:
            # sumo_handler.traci_handler.vehicle.setParameter(vh, param="laneChangeModel.lcCooperative", value=str(1.0))
            pass
        else:
            # sumo_handler.traci_handler.vehicle.setParameter(vh, param="laneChangeModel.lcCooperative", value=str(0.0))
            sumo_handler.traci_handler.vehicle.changeLaneRelative(vehID=vh, indexOffset=self.action_mapping[action], duration=self.config['LC_MAX_DURATION'])

            self.LC_ACTION_TIMEOUT = 5

        return

    def reset(self, sumo_handler):
        """ Resets the agent and return the observation. """

        # sumo_handler.traci_handler.vehicle.add(self.agent_id, route, departLane='best', departSpeed='max')
        # sumo_handler.traci_handler.vehicle.subscribeLeader(self.agent_id)
        sumo_handler.traci_handler.vehicle.subscribe(self.config['veh_id'], varIDs=[tc.VAR_SPEED, tc.VAR_ACCELERATION, tc.VAR_LANEPOSITION, tc.VAR_LANE_INDEX,])
        # logger.info('Agent %s reset done.', self.agent_id)
        return self.agent_id, self.config['start']

####################################################################################################

DEFAULT_AGENT_CONFIG = {
    'origin': '1f2',
    'destination': '1f2',
    'start': 0,
    'actions': {
        0: -1,
        1: 0,
        2: +1,
        # 3: 1.0,
    },
    'action_to_meaning': {
        0 : 'right',
        1 : 'stay',
        2 : 'left',
        # 3: 'cooperate'
    },
    'LC_MAX_DURATION': 5,
    'lon_actions': {
        'acc': +1.0,
        'same': 0.0,
        'dec': -1.0,
    },
}

class SUMOTestMultiAgentEnv(MultiAgentEnv):
    """
    A RLLIB environment for testing MARL environments with SUMO simulations.
    Based on https://github.com/ray-project/ray/blob/master/rllib/tests/test_multi_agent_env.py
    """

    def __init__(self, config):
        """ Initialize the environment. """
        super(SUMOTestMultiAgentEnv, self).__init__()

        self._config = config
        self.agent_tags = deepcopy(self._config['agent_ids'])
        self.agents_on_standby = []
        # logging
        level = logging.getLevelName(config['scenario_config']['log_level'])
        logger.setLevel(level)

        # SUMO Connector
        self.simulation = None

        # Random number generator
        # self.rndgen = RandomState(config['scenario_config']['seed'])
        # self.rndgen = default_rng(config['scenario_config']['seed'])
        self.seed(config['scenario_config']['seed'])
        
        if file_exists('rng_state.pickle'):
            with open('rng_state.pickle', 'rb') as handle:
                self.rndgen.bit_generator.state = pickle.load(handle)

        # Agent initialization
        self.agents = dict()

        # for agent in self._config['agent_ids']:
        #     self.agents[agent] = SUMOAgent(agent, DEFAULT_AGENT_CONFIG)

        # Environment initialization
        self.resetted = True
        self.episodes = 0
        self.steps = 0

        self.pheromone_field = []
        self.nopolicy = self._config['scenario_config']['nopolicy']
        if self.nopolicy:
            self.lc_mode = 1557 # everything except drive right changes
        else:
            self.lc_mode = 1536


        self.num_lanes = self._config['scenario_config']['num_lanes']
        self.num_zones = self._config['scenario_config']['num_zones']
        self.road_len = self._config['scenario_config']['road_length']
        self.zone_size = self.road_len // self.num_zones
        self.evaporation_coeff = self._config['scenario_config']['evaporation_coeff']
        self.diffusion_factor = self._config['scenario_config']['diffusion_factor']
        self.puf = self._config['scenario_config']['puf']

        pf_init_val = self._config['scenario_config']['pheromone_init_val']

        for i in range(self.num_lanes):
            self.pheromone_field.append([])
            for j in range(self.num_zones):
                self.pheromone_field[i].append( PheromoneFieldVal(m=pf_init_val,n=0) )

        self.vehicle_tagging = {}
        self.rng = np.random.default_rng(config['scenario_config']['seed'])

    def seed(self, seed):
        """ Set the seed of a possible random number generator. """
        # self.rndgen = RandomState(seed)
        self.rndgen = default_rng(seed)

    def get_agents(self):
        """ Returns a list of the agents. """
        return self.agents.keys()

    def __del__(self):
        logger.info('Environment destruction: SUMOTestMultiAgentEnv')
        if self.simulation:
            # self.simulation.end_simulation()
            del self.simulation

    def get_agent_pos(self, agent):
        veh_data = self.simulation.veh_subscriptions[self.agents[agent].config['veh_id']]

        lane_pos = veh_data[tc.VAR_LANEPOSITION]
        zone_idx = int(lane_pos // self.zone_size)
        if zone_idx >= self.num_zones:
            zone_idx = self.num_zones - 1

        return (zone_idx ,lane_pos)

    def get_agent_lane(self,agent):
        veh_data = self.simulation.veh_subscriptions[self.agents[agent].config['veh_id']]
        lane_idx = veh_data[tc.VAR_LANE_INDEX]

        return lane_idx


    def get_pheromone_field(self):
        '''
        print('############# PHEROMONE FIELD #############')
    
        for i in range( (self.num_lanes-1), -1, -1):
            for j in range(self.num_zones):
                print('==={}===|'.format(self.pheromone_field[i][j].val()), end='')
            print()
        '''
        raw_pf = deepcopy(self.pheromone_field)
    
        for x, i in enumerate(range( (self.num_lanes-1), -1, -1)):
            for j in range(self.num_zones):
                raw_pf[x][j] = self.pheromone_field[i][j].val()
        
        '''
        print('############# RAW PHEROMONE FIELD #############')
    
        for i in range(self.num_lanes):
            for j in range(self.num_zones):
                print('==={}===|'.format(raw_pf[i][j]), end='')
            print()
        '''
        
        return raw_pf

    ######################################### OBSERVATIONS #########################################

    def get_observation(self, agent):
        """
        Returns the observation of a given agent.
        See http://sumo.sourceforge.net/pydoc/traci._simulation.html
        """
        ret = collections.OrderedDict()

        if self.agents[agent].config['veh_id'] not in self.simulation.veh_subscriptions:
            # ret['lane_idx'] = 0
            ret['pf_prob'] = np.array([0,0,0], dtype=np.float32)
            # ret['speed'] = np.array([0], dtype=np.float32)
            # ret['zone_idx'] = 0
            return ret
            
        veh_data = self.simulation.veh_subscriptions[self.agents[agent].config['veh_id']]

        lane_pos = veh_data[tc.VAR_LANEPOSITION]
        lane_idx = veh_data[tc.VAR_LANE_INDEX]
        zone_idx = int(lane_pos // self.zone_size)
        if zone_idx >= self.num_zones:
            zone_idx = self.num_zones - 1

        A = self.agents[agent]

        top = A.pheromone.range['top'] # 0
        bot = A.pheromone.range['bot'] # 0
        adj = A.pheromone.range['adj'] # 1
        
        center_c_idx = top # check with example, considering Agent is at the center of perception box
        center_r_idx = adj # check with example, considering Agent is at the center of perception box

        A.pheromone.clear()

        # print('AGENT {}'.format(agent))
        pf = np.array(self.get_pheromone_field())
        # print(type(pf))
        # print(pf)
        # z_pf = pf[0:,zone_idx]        
        z_pf = np.array([0.0, 0.0, 0.0])

        if lane_idx == 1:
            z_pf[0] = 0 # right
            z_pf[2] = pf[lane_idx-1, zone_idx] # left
        elif lane_idx == 0:
            z_pf[0] = pf[lane_idx+1, zone_idx] # right
            z_pf[2] = 0 # left
            
        z_pf[1] = pf[lane_idx, zone_idx] # stay
        prob = z_pf
        # print(z_pf)
        # print('LANE IDX ', lane_idx)
        # for a in np.arange(-adj, adj + 1, 1): # -1 0 1
        #     for z in np.arange(-bot, top + 1, 1): # 0
        #         if 0 <= lane_idx - a < self.num_lanes and 0 <= zone_idx + z < self.num_zones:
        #             A.pheromone.set(center_r_idx + a, center_c_idx + z, self.pheromone_field[lane_idx-a][zone_idx+z].val())
        #         else:
        #             A.pheromone.set(center_r_idx + a, center_c_idx + z, -999) # invalid lane probability will evaluate to 0 by setting a high negative number
    
        # prob = A.pf_to_prob(center_r_idx, center_c_idx)

        # ret['lane_idx'] = veh_data[tc.VAR_LANE_INDEX]
        ret['pf_prob'] = prob # right stay left
        # ret['speed'] = np.array([veh_data[tc.VAR_SPEED]], dtype=np.float32)
        # ret['zone_idx'] = zone_idx
        # print('AGENT {}'.format(agent))
        # ret = self.agents[agent].pf_perception.get_val()
        # logger.debug('Agent %s --> Obs: %s', agent, pformat(ret))
        return ret

    def compute_observations(self, agents):
        """ For each agent in the list, return the observation. """
        obs = dict()
        for agent in agents:
            obs[agent] = self.get_observation(agent)
        return obs

    ########################################### REWARDS ############################################

    def isValidAction(self, agent, action):
        if action == 0:
            return (True, 'stay')
        else:
            veh_id = self.agents[agent].config['veh_id']
            lane_idx = self.simulation.veh_subscriptions[veh_id][tc.VAR_LANE_INDEX]
            
            if 0 <= lane_idx + action < self.num_lanes:
                
                lcs = self.simulation.traci_handler.vehicle.getLaneChangeState(vehID=veh_id, direction=action)[1]
                # lcs_pretty = self.simulation.traci_handler.vehicle.getLaneChangeStatePretty(vehID=veh_id, direction=action)[1]
                lc_success = self.simulation.traci_handler.vehicle.wantsAndCouldChangeLane(vehID=veh_id, direction=action, state=lcs)
                
                if lc_success:
                    return (True, 'can')
                else:
                    return (False, 'cannot')
            else:
                return (False, 'invalid_lane')

    def get_reward(self, agent):
        """ Return the reward for a given agent. """
        veh_id = self.agents[agent].config['veh_id']
        if veh_id in self.simulation.veh_subscriptions:
            reward = -1
        else:
            reward = 0

        # Normalize rewards - scale them?
        # 0.01 instead of 0.1
        # Negative reward for doing action that it cannot perform in SUMO as per Lane change state information

        logger.debug('Agent %s --> Reward %d', agent, reward)
        return reward

    def compute_rewards(self, agents):
        """ For each agent in the list, return the rewards. """
        rew = dict()
        for agent in agents:
            rew[agent] = self.get_reward(agent)
        return rew

    ##################################### REST & LEARNING STEP #####################################

    def reset(self):
        """ Resets the env and returns observations from ready agents. """
        self.resetted = True
        self.episodes += 1
        self.steps = 0
        print('---------------------EPISODE {}--------------------------'.format(self.episodes))


        # Reset the SUMO simulation
        if self.simulation:
            del self.simulation

        # self._config['scenario_config']['sumo_config']['sumo_output'] = time()
        self.simulation = SUMOSimulationWrapper(self._config['scenario_config']['sumo_config'])

        self.agent_tags = deepcopy(self._config['agent_ids'])
        self.agents = dict()
        self.agents_on_standby = []
        self.pheromone_field = []
        self.num_lanes = self._config['scenario_config']['num_lanes']
        self.num_zones = self._config['scenario_config']['num_zones']
        self.road_len = self._config['scenario_config']['road_length']
        self.zone_size = self.road_len // self.num_zones
        self.evaporation_coeff = self._config['scenario_config']['evaporation_coeff']
        self.diffusion_factor = self._config['scenario_config']['diffusion_factor']
        self.puf = self._config['scenario_config']['puf']

        pf_init_val = self._config['scenario_config']['pheromone_init_val']

        for i in range(self.num_lanes):
            self.pheromone_field.append([])
            for j in range(self.num_zones):
                self.pheromone_field[i].append( PheromoneFieldVal(m=pf_init_val,n=0) )

        self.vehicle_tagging = {}

        # for vh in self.simulation.traci_handler.vehicle.getIDList():
        #     if vh not in self.simulation.vehicles_tagged:
        #         agent = self.agent_tags.pop(0)
        #         agent_config = deepcopy(DEFAULT_AGENT_CONFIG)
        #         agent_config['veh_id'] = vh
        #         self.agents[agent] = SUMOAgent(agent, agent_config)
        #         agent_id, start = self.agents[agent].reset(self.simulation)
        #         self.simulation.vehicles_tagged.add(vh)

        # for agent in self.agents:
            # agent_id, start = self.agents[agent].reset(self.simulation)
        
        # lbs = self._config['scenario_config']['lane_block_settings']
    
        if self._config['scenario_config']['mbt']:
            NB = self.rndgen.integers(low=0, high=1, endpoint=True) # either 0 or 1
            # NB = self.rng.integers(low=0, high=1, endpoint=True) # either 0 or 1
        else:
            NB = len(self._config['scenario_config']['lane_block_settings'])
        # print('NB {}'.format(NB))
        if NB > 0:

            self.simulation.spawn_blockage(self._config['scenario_config']['lane_block_settings'])
            
            # bsize = self.rng.uniform(low=0.1*self.road_len, high=0.3*self.road_len)
            # bpos = self.rng.uniform(low=0.3*self.road_len, high=self.road_len)
            # blane = self.rng.integers(low=0, high=self.num_lanes-1, endpoint=True)
            # bdur = self.rng.integers(low=50, high=300, endpoint=True)

            # print('BLOCKAGE for episode {}'.format(self.episodes))
            # print('BLOCKAGE SIZE: {}'.format(bsize))
            # print('BLOCKAGE POS: {}'.format(bpos))
            # print('BLOCKAGE LANE: {}'.format(blane))
            # print('BLOCKAGE LANE TYPE: {}'.format(type(blane)))
            # print('BLOCKAGE DURATION: {}'.format(bdur))
            # assert len(block_sizes) == len(block_pos) == len(block_lanes) == len(block_dur)
            # lbs = []
            # lbs.append([bpos, bsize, blane, bdur])
            # self.simulation.spawn_blockage(lbs)
        else:
            # print('NO BLOCKAGE for episode {}'.format(self.episodes))
            pass

        # self.simulation.spawn_blockage(self._config['scenario_config']['lane_block_settings'])


        while len(self.simulation.traci_handler.vehicle.getIDList()) <= 0:
            self.simulation.fast_forward(0)
        
        for vh in self.simulation.traci_handler.vehicle.getIDList():
            if vh not in self.simulation.vehicles_tagged and 'ghost' not in vh:
                agent = self.agent_tags.pop(0)
                self.agents[agent] = SUMOAgent(agent, deepcopy(DEFAULT_AGENT_CONFIG)) # Generate new agent for the new vehicle in simulation
                self.agents[agent].config['veh_id'] = vh # Tag the vehicle ID to that new agent
                self.agents[agent].reset(self.simulation) # subscribe info for that vehicle ID
                self.simulation.vehicles_tagged.add(vh) # tagged ID added to ALREADY tagged


        # Reset the agents
        # waiting_agents = list()
        # for agent in self.agents.values():
            # agent_id, start = agent.reset(self.simulation)
            # waiting_agents.append((start, agent_id))
        # waiting_agents.sort()

        # Move the simulation forward
        # starting_time = waiting_agents[0][0]
        # print(starting_time)
        # input()
        
        self.simulation._default_step_action(self.agents.keys()) # hack to retrieve the subs

        # Observations
        initial_obs = self.compute_observations(self.agents.keys())
        return initial_obs


    def update_pf(self):
        sum_zonewise_speeds = []
        for i in range(self.num_lanes):
            sum_zonewise_speeds.append([])
            for j in range(self.num_zones):
                sum_zonewise_speeds[i].append(0)

        sum_zonewise_accels = []
        for i in range(self.num_lanes):
            sum_zonewise_accels.append([])
            for j in range(self.num_zones):
                sum_zonewise_accels[i].append(0)

        sum_zonewise_lc_from = []
        for i in range(self.num_lanes):
            sum_zonewise_lc_from.append([])
            for j in range(self.num_zones):
                sum_zonewise_lc_from[i].append(0)

        vehicles_per_zone = []
        for i in range(self.num_lanes):
            vehicles_per_zone.append([])
            for j in range(self.num_zones):
                vehicles_per_zone[i].append(0)


        for vh, val in self.simulation.veh_subscriptions.items():
            if 'ghost' in vh or vh in ['ghost', 'ghost2', 'ghost3']:
                continue
            lane_pos = val[tc.VAR_LANEPOSITION]
            lane_idx = val[tc.VAR_LANE_INDEX]

            zone_idx = int(lane_pos // self.zone_size)
            if zone_idx >= self.num_zones:
                zone_idx = self.num_zones - 1
                
            if vh in self.agents:
                agent = self.agents[vh]
                if agent.prev_lane != agent.lane and agent.prev_lane != None:
                    if zone_idx + 1 < self.num_zones:
                        sum_zonewise_lc_from[agent.prev_lane][agent.prev_zone + 1] += 1

                        print('ADDED to zone index {} ON LANE {}'.format(agent.prev_zone+1, agent.prev_lane))
                    
            sum_zonewise_accels[lane_idx][zone_idx] += val[tc.VAR_ACCELERATION]
            sum_zonewise_speeds[lane_idx][zone_idx] += val[tc.VAR_SPEED]
            vehicles_per_zone[lane_idx][zone_idx] += 1


        for i in range( (self.num_lanes-1), -1, -1):
            for j in range(self.num_zones):
                if vehicles_per_zone[i][j] != 0:
                    speed_comp = sum_zonewise_speeds[i][j] / vehicles_per_zone[i][j]
                    accel_comp = sum_zonewise_accels[i][j] / vehicles_per_zone[i][j]
                else:
                    speed_comp = 0
                    accel_comp = 0
                
                lc_from_comp = -sum_zonewise_lc_from[i][j]
                val = 0.25 * speed_comp + 0.25 * accel_comp + 0.5 * lc_from_comp

                # print('val = 0.25 * {} + 0.25 * {} + 0.5 * {} = {}'.format(speed_comp, accel_comp, lc_from_comp, val))
                evaporated_val = (1 - self.evaporation_coeff) * self.pheromone_field[i][j].val() + self.evaporation_coeff * val
                # print('evaporated val = (1 - {}) * {} + {} * {} = {}'.format(self.evaporation_coeff, self.pheromone_field[i][j].val(), self.evaporation_coeff, val, evaporated_val))
                self.pheromone_field[i][j].update_mean(evaporated_val)
                if j > 0:
                    # self.pheromone_field[i][j+1].accept_diffusion(DIFFUSION_FACTOR * val)
                    self.pheromone_field[i][j-1].diffuse(self.diffusion_factor * self.pheromone_field[i][j].val())

        # '''
        # print('############# PHEROMONE FIELD #############')
    
        # for i in range( (self.num_lanes-1), -1, -1):
        #     print('LANE index {}'.format(i), end='')
        #     for j in range(self.num_zones):
        #         print('==={}===|'.format(self.pheromone_field[i][j].val()), end='')
        #     print()
        # input()
        # '''
    def step(self, action_dict):
        """
        Returns observations from ready agents.
        The returns are dicts mapping from agent_id strings to values. The
        number of agents in the env can vary over time.
        Returns
        -------
            obs (dict): New observations for each ready agent.
            rewards (dict): Reward values for each ready agent. If the
                episode is just started, the value will be None.
            dones (dict): Done values for each ready agent. The special key
                "__all__" (required) is used to indicate env termination.
            infos (dict): Optional info values for each agent id.
        """


        self.resetted = False
        self.steps += 1
        # print('steps: {}'.format(self.steps))
        # print('action_dict: {}'.format(action_dict))
        logger.debug('====> [SUMOTestMultiAgentEnv:step] Episode: %d - Step: %d <====',
                     self.episodes, self.steps)
        # print('NUMBER OF STEPS: {}'.format(self.steps))

        dones = {}
        dones['__all__'] = False
        
        if self.steps == 1 or self.steps % self.puf == 0:
            self.update_pf()

        shuffled_agents = sorted(action_dict.keys()) # it may seem not smar to sort something that
                                                     # may need to be shuffled afterwards, but it
                                                     # is a matter of consistency instead of using
                                                     # whatever insertion order was used in the dict

        if self._config['scenario_config']['agent_rnd_order']:
            # randomize the agent order to minimize SUMO's insertion queues impact
            logger.debug('Shuffling the order of the agents.')
        self.rndgen.shuffle(shuffled_agents) # in-place shuffle

        # print('======= STEP {}========'.format(self.steps))
        obs, rewards, infos = {}, {}, {}


        for vh in self.simulation.traci_handler.vehicle.getIDList(): # Set LC mode from the beginning
            if 'ghost' not in vh:
                self.simulation.traci_handler.vehicle.setLaneChangeMode(vh, self.lc_mode)

        for vh in self.simulation.traci_handler.vehicle.getIDList(): # Generate agents based on the vehicles in the simulation
            if vh not in self.simulation.vehicles_tagged and 'ghost' not in vh:
                agent = self.agent_tags.pop(0) # select next tag from tag pool
                self.vehicle_tagging[vh] = agent
                self.agents[agent] = SUMOAgent(agent, deepcopy(DEFAULT_AGENT_CONFIG)) # Generate new agent for the new vehicle in simulation
                self.agents[agent].config['veh_id'] = vh # Tag the vehicle ID to that new agent
                self.agents[agent].reset(self.simulation) # subscribe info for that vehicle ID
                self.simulation.vehicles_tagged.add(vh) # tagged ID added  to ALREADY tagged
                obs[agent] = self.get_observation(agent)
                # print('AGENT {} : Obs {}'.format(agent, obs[agent]))
                # input()
                rewards[agent] = self.get_reward(agent)
                dones[agent] = False
            # else:
                # agent = self.vehicle_tagging[vh]
                # agent.prev_lane = agent.lane
                # agent.lane = l_IDX

        # print('self agents', self.agents.keys())
        # print('agents on standby ', self.agents_on_standby)
        # print('action dict', shuffled_agents)
        # print('veh subs', self.simulation.veh_subscriptions.keys())
        # print('traci veh ID list()', self.simulation.traci_handler.vehicle.getIDList())

        for agent in shuffled_agents:
            if not self.nopolicy:
                self.agents[agent].step(action_dict[agent], self.simulation)
            
            self.agents_on_standby.append(agent)

        logger.debug('Before SUMO')
        ongoing_simulation = self.simulation.step(until_end=False, agents=set(action_dict.keys()))
        logger.debug('After SUMO')

        ## end of the episode
        if not ongoing_simulation:
            logger.info('Reached the end of the SUMO simulation.')
            dones['__all__'] = True

        for agent in self.agents_on_standby:
            dones[agent] = self.agents[agent].config['veh_id'] not in self.simulation.veh_subscriptions
            if dones[agent]:
                self.agents_on_standby.remove(agent)
            else:
                if self.agents[agent].LC_ACTION_TIMEOUT > 0:
                    self.agents[agent].LC_ACTION_TIMEOUT -= 1
                else:
                    self.agents_on_standby.remove(agent)                    
                    obs[agent] = self.get_observation(agent)
                    # rewards[agent] = self.get_reward(agent)
                    # valid_action, reason = self.isValidAction(agent, self.agents[agent].action_mapping[action_dict[agent]])
                    # if not valid_action:
                    #     rewards[agent] += -0.01

                    # infos[agent] = {
                    #     'lc_success': valid_action,
                    #     'reason': reason,
                    #     'zone_pos': self.get_agent_pos(agent),
                    #     'action': action_dict[agent], 
                    # }

        # obs, rewards, infos = {}, {}, {}
        for agent in shuffled_agents:
            dones[agent] = self.agents[agent].config['veh_id'] not in self.simulation.veh_subscriptions
            if not dones[agent]:
                obs[agent] = self.get_observation(agent)
                rewards[agent] = self.get_reward(agent)
                valid_action, reason = self.isValidAction(agent, self.agents[agent].action_mapping[action_dict[agent]])
                # if not valid_action: # remove reward for 'validness' of action if not working out and check
                    # rewards[agent] += -10

                infos[agent] = {
                    'lc_success': valid_action,
                    'reason': reason,
                    'zone_pos': self.get_agent_pos(agent),
                    'lane_idx': self.get_agent_lane(agent),
                    'action': action_dict[agent], 
                }

        logger.debug('Observations: %s', pformat(obs))
        logger.debug('Rewards: %s', pformat(rewards))
        logger.debug('Dones: %s', pformat(dones))
        logger.debug('Info: %s', pformat(infos))
        logger.debug('========================================================')

        # print('Observations for step {}: {}'.format(self.steps, obs))
        return obs, rewards, dones, infos

    ################################## ACTIONS & OBSERATIONS SPACE #################################

    def get_action_space_size(self, agent=None):
        """ Returns the size of the action space. """
        return len(DEFAULT_AGENT_CONFIG['actions'].keys())

    def get_action_space(self, agent):
        """ Returns the action space. """
        return gym.spaces.Discrete(self.get_action_space_size(agent))

    def get_set_of_actions(self, agent):
        """ Returns the set of possible actions for an agent. """
        return set(range(self.get_action_space_size(agent)))

    def get_obs_space_size(self, agent):
        """ Returns the size of the observation space. """
        return 1
        # return ((len(self.agents[agent].config['actions'].keys())+1))

    def get_obs_space(self, agent):
        """ Returns the observation space. """
        return gym.spaces.Dict({
            # 'lane_idx': gym.spaces.Discrete(self.num_lanes),
            # 'speed': gym.spaces.Box(low=0.0, high=100.0, shape=(1,)),
            'pf_prob': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
            # 'zone_idx': gym.spaces.Discrete(self.num_zones),
        })
