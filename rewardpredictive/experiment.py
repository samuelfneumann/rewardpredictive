#
# Copyright (c) 2020 Lucas Lehnert <lucas_lehnert@brown.edu>
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#

import pickle
import multiprocessing as mp
import os
from abc import abstractmethod, ABC
from glob import glob
from itertools import product
from os import path as osp
from tqdm import tqdm
from pathlib import Path
import json
from math import sqrt

import yaml

from .evaluate import eval_reward_predictive
from .evaluate import eval_total_reward
from .utils import SFLearning, EGreedyScheduleUpdate
from .mdp import *
from .utils import set_seeds
from .utils import simulate_episodes

from definitions import ROOT_DIR


def experiment_to_config(exp_set_name):
    return f"{ROOT_DIR}/configs/"+exp_set_name+".json"


def _load_experiment_from_save_dir(save_dir):
    with open(osp.join(save_dir, 'index.yaml'), 'r') as f:
        exp_dict = yaml.load(f, Loader=yaml.Loader)
    experiment_constructor = globals()[exp_dict['class_name']]
    return experiment_constructor.load(save_dir)


def _load_experiment_list(base_dir='./data'):
    index_list = glob(osp.join(base_dir, '**', 'index.yaml'), recursive=True)
    save_dir_list = [osp.split(p)[0] for p in index_list]
    # with mp.Pool() as p:
    #     experiment_list = p.map(_load_experiment_from_save_dir, save_dir_list)
    experiment_list = [_load_experiment_from_save_dir(d) for d in save_dir_list]
    return experiment_list


#LEARNING_RATE_LIST = [0.1, 0.5, 0.9]

class ExperimentHParam(rl.Experiment):
    HP_REPEATS = 'repeats'
    HP_MDP_SIZE = "mdp_size"

    def __init__(self, hparam=None, base_dir='./data'):
        if hparam is None:
            hparam = {}
        self.hparam = self._add_defaults_to_hparam(hparam)
        self._load_config(CONFIG_FILE)
        self.results = {}
        # self.save_dir = self._get_save_dir(self.hparam, base_dir)

    def _load_config(self, file):
        with open(file) as hparam_file:
            hparam = json.loads(hparam_file.read())

        for key in hparam["experiment"].keys():
            self.hparam[key] = hparam["experiment"][key]

    def _add_defaults_to_hparam(self, hparam: dict) -> dict:
        hparam_complete = self.get_default_hparam()
        for k in hparam_complete.keys():
            if k in hparam.keys():
                hparam_complete[k] = hparam[k]
        return hparam_complete

    def get_default_hparam(self) -> dict:
        return {
            ExperimentHParam.HP_REPEATS: 10
        }

    def _get_save_dir(self, hparams, save_dir_base, hparam_keys=None) -> str:
        if hparam_keys is None:
            hparam_keys = list(hparams.keys())
        param_str_list = ['{}_{}'.format(k, hparams[k]) for k in hparam_keys]
        return osp.join(save_dir_base, self.get_class_name(), *param_str_list)

    def _run_experiment(self):
        res_list = [self.run_repeat(i) for i in range(self.hparam[ExperimentHParam.HP_REPEATS])]
        for k in res_list[0].keys():
            self.results[k] = [r[k] for r in res_list]

    @abstractmethod
    def run_repeat(self, rep_idx: int) -> dict:
        pass

    def _save_ndarray_rep(self, rep_list, name):
        fn_list = ['{:03d}_{}.npy'.format(i, name) for i in
                   range(len(rep_list))]
        for rep, fn in zip(rep_list, fn_list):
            np.save(osp.join(self.save_dir, fn), rep, allow_pickle=True)
        return fn_list

    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        results_fn = {}
        for k in self.results.keys():
            results_fn[k] = self._save_ndarray_rep(self.results[k], k)

        exp_dict = {
            'class_name': self.get_class_name(),
            'hparam': self.hparam,
            'results': results_fn
        }
        with open(osp.join(self.save_dir, 'index.yaml'), 'w') as f:
            yaml.dump(exp_dict, f, default_flow_style=False)

    @classmethod
    def load(cls, save_dir: str):
        with open(osp.join(save_dir, 'index.yaml'), 'r') as f:
            exp_dict = yaml.load(f, Loader=yaml.Loader)
        if not issubclass(globals()[exp_dict['class_name']], cls):
            exp_msg = 'Cannot load experiment because class {} is not a sub-class of {}.'
            exp_msg = exp_msg.format(exp_dict['class_name'],
                                     cls.get_class_name())
            raise rl.ExperimentException(exp_msg)
        exp = globals()[exp_dict['class_name']](exp_dict['hparam'])
        exp.results = exp_dict['results']
        for k in exp.results:
            exp.results[k] = [
                np.load(osp.join(exp.save_dir, p), allow_pickle=True) for p in
                exp.results[k]]
        return exp


def _run_repeat(exp, rep_idx):
    return exp.run_repeat(rep_idx)


class ExperimentHParamParallel(ExperimentHParam, ABC):
    def _run_experiment(self):
        num_repeats = self.hparam[ExperimentHParam.HP_REPEATS]
        param_list = [(self, i) for i in range(num_repeats)]
        with mp.Pool() as p:
            res_list = p.starmap(_run_repeat, param_list)
        for k in res_list[0].keys():
            self.results[k] = [r[k] for r in res_list]


class ExperimentSet(object):
    def __init__(self, experiment_list):
        self.experiment_list = experiment_list

    @abstractmethod
    def get_best_experiment(self):
        pass

    def get_experiment_list_by_hparam(self, hparam):
        # print('Retrieving experiment(s) with hyper-parameter(s):')
        # for k, v in hparam.items():
        #     print('\t{}: {}'.format(k, v))

        exp_list = []
        for exp in self.experiment_list:
            if all([exp.hparam[k] == hparam[k] for k in hparam.keys()]):
                exp_list.append(exp)
        return exp_list

    def get_hparam_values(self, hparam_key):
        return np.sort(
            np.unique([exp.hparam[hparam_key] for exp in self.experiment_list]))

    def run(self, seed=12345):
        for i, exp in enumerate(self.experiment_list):
            print(f'Running experiment {i} out of {len(self.experiment_list)}')
            set_seeds(seed)
            exp.run()
            exp.save()

    def run_best(self, seed=12345):
        exp = self.get_best_experiment()
        set_seeds(seed)
        exp.run()
        exp.save()

    @classmethod
    def load(cls, base_dir='./data'):
        return ExperimentSet(_load_experiment_list(base_dir=base_dir))


"""
RANDOM 10x10 GRIDWORLD EXPERIMENTS BELOW
"""


class ExperimentTaskSequenceRandomRewardChange(ExperimentHParamParallel):
# class ExperimentTaskSequenceRandomRewardChange(ExperimentHParam):

    HP_EXPLORATION = 'exploration'
    HP_TASK_SEQUENCE = 'task_sequence'
    HP_NUM_EPISODES = 'episodes'
    HP_EPSILON = "epsilon"
    HP_GAMMA = "gamma"
    HP_NUM_TASKS = "num_tasks"

    def __init__(self, *params, base_dir="./data", **kwargs):

        """
        Experiment task sequence for our random reward change experiments.

        Biggest difference to the normal RewardChange class is how we yield tasks -
        # we use Python generators! Look below as to how that works.
        # We use generators here (instead of instantiating num_tasks environments) in order to
        # potentially generate an endless number of tasks (Don't do this yet things will break!).
        :param params: params to pass in
        :param kwargs: dict params to pass in
        """
        super().__init__(*params, **kwargs)
        # self.num_tasks = num_tasks
        self.task_sequence = self._get_task_sequence()
        self.num_actions = self.task_sequence[0].action_space.n
        self.save_dir = self._get_save_dir(self.hparam, base_dir)

    def get_default_hparam(self) -> dict:
        defaults = super().get_default_hparam()
        defaults[ExperimentTaskSequenceRandomRewardChange.HP_REPEATS] = 20
        defaults[ExperimentTaskSequenceRandomRewardChange.HP_TASK_SEQUENCE] = 'slight'
        defaults[ExperimentTaskSequenceRandomRewardChange.HP_EXPLORATION] = 'optimistic'
        defaults[ExperimentTaskSequenceRandomRewardChange.HP_NUM_EPISODES] = 100
        defaults[ExperimentTaskSequenceRandomRewardChange.HP_EPSILON] = 0.1
        defaults[ExperimentTaskSequenceRandomRewardChange.HP_GAMMA] = 0.9
        defaults[ExperimentTaskSequenceRandomRewardChange.HP_MDP_SIZE] = 8
        defaults[ExperimentTaskSequenceRandomRewardChange.HP_NUM_TASKS] = 10
        # defaults[ExperimentTaskSequenceRandomRewardChange.HP_MDP_SIZE] = self.task_sequence[0].num_states()
        return defaults

    def _get_task_sequence(self):
        data_path = Path(ROOT_DIR, 'data', 'RandomRewardMaze')
        file_path = data_path / 'maze.pkl'
        mdp_seq = []

        if not data_path.is_dir():
            data_path.mkdir()

        if file_path.is_file():
            # We don't need to generate
            print(f"Loading mazes from file {file_path}")
            with open(file_path, 'rb') as f:
                mdp_seq = pickle.load(f)
            mdp_size = int(sqrt(mdp_seq[0].num_states()))
            self.hparam[ExperimentHParam.HP_MDP_SIZE] = mdp_size
        else:
            mdp_size = self.hparam[ExperimentHParam.HP_MDP_SIZE]
            num_tasks = self.hparam[ExperimentTaskSequenceRandomRewardChange.HP_NUM_TASKS]
            pbar = tqdm(range(num_tasks))
            for i in pbar:
                mdp_seq.append(RandomRewardChange(size_maze=mdp_size))
                pbar.set_description(f"Creating env #{i}")

            with open(file_path, 'wb') as f:
                pickle.dump(mdp_seq, f)
            print(f"Dumping mazes to file {file_path}")

        return mdp_seq

    def run_repeat(self, rep_idx: int) -> dict:
        set_seeds(12345 + rep_idx)
        episodes = self.hparam[ExperimentTaskSequenceRandomRewardChange.HP_NUM_EPISODES]
        agent = self._construct_agent()
        ep_len_logger = rl.logging.LoggerEpisodeLength()
        if self.hparam[ExperimentTaskSequenceRandomRewardChange.HP_EXPLORATION] == 'optimistic':
            policy = rl.policy.GreedyPolicy(agent)
            transition_listener = rl.data.transition_listener(agent, ep_len_logger)
        elif self.hparam[ExperimentTaskSequenceRandomRewardChange.HP_EXPLORATION] == 'egreedy':
            epsilon = self.hparam[ExperimentTaskSequenceRandomRewardChange.HP_EPSILON]
            policy = rl.policy.EGreedyPolicy(agent, epsilon)
            exp_schedule = rl.schedule.LinearInterpolatedVariableSchedule([0, 180], [epsilon, epsilon])
            exp_schedule_listener = EGreedyScheduleUpdate(policy, exp_schedule)
            transition_listener = rl.data.transition_listener(agent, exp_schedule_listener, ep_len_logger)

        for task in self.task_sequence:
            simulate_episodes(task, policy, transition_listener, episodes, max_steps=1000)
            self._reset_agent(agent)

        res_dict = {
            'episode_length': np.reshape(ep_len_logger.get_episode_length(), [len(self.task_sequence), -1])
        }
        return res_dict

    @abstractmethod
    def _construct_agent(self):
        pass

    @abstractmethod
    def _reset_agent(self, agent):
        pass


class ExperimentTaskSequenceRandomRewardChangeQLearning(ExperimentTaskSequenceRandomRewardChange):
    HP_LEARNING_RATE = 'lr'

    def get_default_hparam(self) -> dict:
        defaults = super().get_default_hparam()
        defaults[ExperimentTaskSequenceRandomRewardChangeQLearning.HP_LEARNING_RATE] = 0.9
        return defaults

    def _construct_agent(self):
        states = self.hparam[ExperimentHParam.HP_MDP_SIZE] ** 2
        if self.hparam[ExperimentTaskSequenceRandomRewardChange.HP_EXPLORATION] == 'optimistic':
            q_vals = np.ones([self.num_actions, states], dtype=np.float32)
        elif self.hparam[ExperimentTaskSequenceRandomRewardChange.HP_EXPLORATION] == 'egreedy':
            q_vals = np.zeros([self.num_actions, states], dtype=np.float32)
        lr = self.hparam[ExperimentTaskSequenceRandomRewardChangeQLearning.HP_LEARNING_RATE]
        gamma = self.hparam[ExperimentTaskSequenceRandomRewardChange.HP_GAMMA]
        return rl.agent.QLearning(num_states=states, num_actions=self.num_actions, learning_rate=lr, gamma=gamma, init_Q=q_vals)

    def _reset_agent(self, agent):
        agent.reset()


class ExperimentTaskSequenceRandomRewardChangeQTransfer(ExperimentTaskSequenceRandomRewardChangeQLearning):
    def _reset_agent(self, agent):
        pass


class ExperimentSetTaskSequenceRandomRewardChange(ExperimentSet):

    def get_best_experiment(self, exploration='optimistic', task_sequence='slight'):
        exp_list = self.get_experiment_list_by_hparam({
            ExperimentTaskSequenceRandomRewardChange.HP_EXPLORATION: exploration,
            ExperimentTaskSequenceRandomRewardChange.HP_TASK_SEQUENCE: task_sequence
        })
        ep_len_list = [np.mean(exp.results['episode_length']) for exp in exp_list]
        best_idx = np.argmin(ep_len_list)
        exp = exp_list[best_idx]
        for k, v in exp.hparam.items():
            print('{}: {}'.format(k, v))
        return exp

    @classmethod
    def get_lr_dict(cls):
        learning_rates = {"q_learning_rates": [0.1],
                          "sf_learning_rates": [0.1],
                          "r_learning_rates": [0.1]}

        with open(CONFIG_FILE) as config:
            hparam = json.loads(config.read())
            # return hparam["experiment_set"]["learning_rates"]

        for key in hparam["experiment_set"]["learning_rates"].keys():
            learning_rates[key] = hparam["experiment_set"]["learning_rates"][key]

        return learning_rates



class ExperimentSetTaskSequenceRandomRewardChangeQLearning(ExperimentSetTaskSequenceRandomRewardChange):

    def __init__(self, experiment_list):
        lr = ExperimentTaskSequenceRandomRewardChangeQLearning.HP_LEARNING_RATE
        experiment_list.sort(key=lambda e: (e.hparam[lr]))
        super().__init__(experiment_list)

    @classmethod
    def load(cls, base_dir='./data'):
        global CONFIG_FILE
        CONFIG_FILE = experiment_to_config(cls.__name__)
        exp_list = _load_experiment_list(
            osp.join(base_dir, ExperimentTaskSequenceRandomRewardChangeQLearning.get_class_name()))
        return ExperimentSetTaskSequenceRandomRewardChangeQLearning(exp_list)

    @classmethod
    def construct(cls, base_dir='./data'):
        global CONFIG_FILE
        CONFIG_FILE = experiment_to_config(cls.__name__)
        exp_list = []
        # lr_list = [.1, .3, .5, .7, .9]
        lr_list = ExperimentSetTaskSequenceRandomRewardChange.get_lr_dict()["q_learning_rates"]
        for lr, task_seq, expl in product(lr_list, ['significant'], ['egreedy']):
            exp_list.append(ExperimentTaskSequenceRandomRewardChangeQLearning({
                ExperimentTaskSequenceRandomRewardChangeQLearning.HP_LEARNING_RATE: lr,
                ExperimentTaskSequenceRandomRewardChangeQLearning.HP_TASK_SEQUENCE: task_seq,
                ExperimentTaskSequenceRandomRewardChangeQLearning.HP_EXPLORATION: expl
            }, base_dir=base_dir))
        return ExperimentSetTaskSequenceRandomRewardChangeQLearning(exp_list)


class ExperimentSetTaskSequenceRandomRewardChangeQTransfer(ExperimentSetTaskSequenceRandomRewardChange):

    def __init__(self, experiment_list):
        lr = ExperimentTaskSequenceRandomRewardChangeQTransfer.HP_LEARNING_RATE
        experiment_list.sort(key=lambda e: (e.hparam[lr]))
        super().__init__(experiment_list)

    @classmethod
    def load(cls, base_dir='./data'):
        global CONFIG_FILE
        CONFIG_FILE = experiment_to_config(cls.__name__)
        exp_list = _load_experiment_list(
            osp.join(base_dir, ExperimentTaskSequenceRandomRewardChangeQTransfer.get_class_name()))
        return ExperimentSetTaskSequenceRandomRewardChangeQTransfer(exp_list)

    @classmethod
    def construct(cls, base_dir='./data'):
        global CONFIG_FILE
        CONFIG_FILE = experiment_to_config(cls.__name__)
        exp_list = []
        # lr_list = [.1, .3, .5, .7, .9]
        lr_list = ExperimentSetTaskSequenceRandomRewardChange.get_lr_dict()["q_learning_rates"]
        for lr, task_seq, expl in product(lr_list, ['significant'], ['egreedy']):
            exp_list.append(ExperimentTaskSequenceRandomRewardChangeQTransfer({
                ExperimentTaskSequenceRandomRewardChangeQTransfer.HP_LEARNING_RATE: lr,
                ExperimentTaskSequenceRandomRewardChangeQTransfer.HP_TASK_SEQUENCE: task_seq,
                ExperimentTaskSequenceRandomRewardChangeQTransfer.HP_EXPLORATION: expl
            }, base_dir=base_dir))
        return ExperimentSetTaskSequenceRandomRewardChangeQTransfer(exp_list)


# SF LEARNING
class ExperimentTaskSequenceRandomRewardChangeSFLearning(ExperimentTaskSequenceRandomRewardChange):
    HP_LEARNING_RATE_SF = 'lr_sf'
    HP_LEARNING_RATE_REWARD = 'lr_r'

    def get_default_hparam(self) -> dict:
        defaults = super().get_default_hparam()
        defaults[ExperimentTaskSequenceRandomRewardChangeSFLearning.HP_LEARNING_RATE_SF] = 0.5
        defaults[ExperimentTaskSequenceRandomRewardChangeSFLearning.HP_LEARNING_RATE_REWARD] = 0.9
        return defaults

    def _construct_agent(self):
        states = self.hparam[ExperimentHParam.HP_MDP_SIZE] ** 2
        if self.hparam[ExperimentTaskSequenceRandomRewardChangeSFLearning.HP_EXPLORATION] == 'optimistic':
            init_sf_mat = np.eye(states * self.num_actions, dtype=np.float32)
            init_w_vec = np.ones(states * self.num_actions, dtype=np.float32)
        elif self.hparam[ExperimentTaskSequenceRandomRewardChangeSFLearning.HP_EXPLORATION] == 'egreedy':
            init_sf_mat = np.zeros([states * self.num_actions, states * self.num_actions], dtype=np.float32)
            init_w_vec = np.zeros(states * self.num_actions, dtype=np.float32)
        lr_sf = self.hparam[ExperimentTaskSequenceRandomRewardChangeSFLearning.HP_LEARNING_RATE_SF]
        lr_r = self.hparam[ExperimentTaskSequenceRandomRewardChangeSFLearning.HP_LEARNING_RATE_REWARD]
        gamma = self.hparam[ExperimentTaskSequenceRandomRewardChange.HP_GAMMA]
        agent = SFLearning(
            num_states=states,
            num_actions=self.num_actions,
            learning_rate_sf=lr_sf,
            learning_rate_reward=lr_r,
            gamma=gamma,
            init_sf_mat=init_sf_mat,
            init_w_vec=init_w_vec
        )
        return agent

    def _reset_agent(self, agent):
        agent.reset(reset_sf=True, reset_w=True)

class ExperimentTaskSequenceRandomRewardChangeSFTransfer(ExperimentTaskSequenceRandomRewardChangeSFLearning):
    def _reset_agent(self, agent):
        agent.reset(reset_sf=False, reset_w=True)


class ExperimentTaskSequenceRandomRewardChangeSFTransferAll(ExperimentTaskSequenceRandomRewardChangeSFLearning):
    def _reset_agent(self, agent):
        agent.reset(reset_sf=False, reset_w=False)


class ExperimentSetTaskSequenceRandomRewardChangeSFLearning(ExperimentSetTaskSequenceRandomRewardChange):

    def __init__(self, experiment_list):
        lr_sf = ExperimentTaskSequenceRandomRewardChangeSFLearning.HP_LEARNING_RATE_SF
        lr_r = ExperimentTaskSequenceRandomRewardChangeSFLearning.HP_LEARNING_RATE_REWARD
        experiment_list.sort(key=lambda e: (e.hparam[lr_sf], e.hparam[lr_r]))
        super().__init__(experiment_list)

    @classmethod
    def load(cls, base_dir='./data'):
        global CONFIG_FILE
        CONFIG_FILE = experiment_to_config(cls.__name__)
        exp_list = _load_experiment_list(
            osp.join(base_dir, ExperimentTaskSequenceRandomRewardChangeSFLearning.get_class_name()))
        return ExperimentSetTaskSequenceRandomRewardChangeSFLearning(exp_list)

    @classmethod
    def construct(cls, base_dir='./data'):
        global CONFIG_FILE
        CONFIG_FILE = experiment_to_config(cls.__name__)
        exp_list = []
        # lr_list = [.1, .3, .5, .7, .9]
        lr_dict = ExperimentSetTaskSequenceRandomRewardChange.get_lr_dict()
        sf_lr_list = lr_dict["sf_learning_rates"]
        r_lr_list = lr_dict["r_learning_rates"]
        param_it = product(sf_lr_list, r_lr_list, ['significant'], ['egreedy'])
        for lr_sf, lr_r, task_seq, expl in param_it:
            exp_list.append(ExperimentTaskSequenceRandomRewardChangeSFLearning({
                ExperimentTaskSequenceRandomRewardChangeSFLearning.HP_LEARNING_RATE_SF: lr_sf,
                ExperimentTaskSequenceRandomRewardChangeSFLearning.HP_LEARNING_RATE_REWARD: lr_r,
                ExperimentTaskSequenceRandomRewardChangeSFLearning.HP_TASK_SEQUENCE: task_seq,
                ExperimentTaskSequenceRandomRewardChangeSFLearning.HP_EXPLORATION: expl
            }, base_dir=base_dir))
        return ExperimentSetTaskSequenceRandomRewardChangeSFLearning(exp_list)


class ExperimentSetTaskSequenceRandomRewardChangeSFTransfer(ExperimentSetTaskSequenceRandomRewardChange):

    def __init__(self, experiment_list):
        lr_sf = ExperimentTaskSequenceRandomRewardChangeSFTransfer.HP_LEARNING_RATE_SF
        lr_r = ExperimentTaskSequenceRandomRewardChangeSFTransfer.HP_LEARNING_RATE_REWARD
        experiment_list.sort(key=lambda e: (e.hparam[lr_sf], e.hparam[lr_r]))
        super().__init__(experiment_list)

    @classmethod
    def load(cls, base_dir='./data'):
        global CONFIG_FILE
        CONFIG_FILE = experiment_to_config(cls.__name__)
        exp_list = _load_experiment_list(
            osp.join(base_dir, ExperimentTaskSequenceRandomRewardChangeSFTransfer.get_class_name()))
        return ExperimentSetTaskSequenceRandomRewardChangeSFTransfer(exp_list)

    @classmethod
    def construct(cls, base_dir='./data'):
        global CONFIG_FILE
        CONFIG_FILE = experiment_to_config(cls.__name__)
        exp_list = []
        #lr_list = [.1, .3, .5, .7, .9]
        lr_dict = ExperimentSetTaskSequenceRandomRewardChange.get_lr_dict()
        sf_lr_list = lr_dict["sf_learning_rates"]
        r_lr_list = lr_dict["r_learning_rates"]
        it = product(sf_lr_list, r_lr_list, ['significant'], ['egreedy'])
        for lr_sf, lr_r, task_seq, expl in it:
            exp_list.append(ExperimentTaskSequenceRandomRewardChangeSFTransfer({
                ExperimentTaskSequenceRandomRewardChangeSFTransfer.HP_LEARNING_RATE_SF: lr_sf,
                ExperimentTaskSequenceRandomRewardChangeSFTransfer.HP_LEARNING_RATE_REWARD: lr_r,
                ExperimentTaskSequenceRandomRewardChangeSFTransfer.HP_TASK_SEQUENCE: task_seq,
                ExperimentTaskSequenceRandomRewardChangeSFTransfer.HP_EXPLORATION: expl
            }, base_dir=base_dir))
        return ExperimentSetTaskSequenceRandomRewardChangeSFTransfer(exp_list)


class ExperimentSetTaskSequenceRandomRewardChangeSFTransferAll(ExperimentSetTaskSequenceRandomRewardChange):

    def __init__(self, experiment_list):
        lr_sf = ExperimentTaskSequenceRandomRewardChangeSFTransferAll.HP_LEARNING_RATE_SF
        lr_r = ExperimentTaskSequenceRandomRewardChangeSFTransferAll.HP_LEARNING_RATE_REWARD
        experiment_list.sort(key=lambda e: (e.hparam[lr_sf], e.hparam[lr_r]))
        super().__init__(experiment_list)

    @classmethod
    def load(cls, base_dir='./data'):
        global CONFIG_FILE
        CONFIG_FILE = experiment_to_config(cls.__name__)
        exp_list = _load_experiment_list(
            osp.join(base_dir, ExperimentTaskSequenceRandomRewardChangeSFTransferAll.get_class_name()))
        return ExperimentSetTaskSequenceRandomRewardChangeSFTransferAll(exp_list)

    @classmethod
    def construct(cls, base_dir='./data'):
        global CONFIG_FILE
        CONFIG_FILE = experiment_to_config(cls.__name__)
        exp_list = []
        lr_dict = ExperimentSetTaskSequenceRandomRewardChange.get_lr_dict()
        sf_lr_list = lr_dict["sf_learning_rates"]
        r_lr_list = lr_dict["r_learning_rates"]
        it = product(sf_lr_list, r_lr_list, ['significant'], ['egreedy'])
        for lr_sf, lr_r, task_seq, expl in it:
            exp_list.append(ExperimentTaskSequenceRandomRewardChangeSFTransferAll({
                ExperimentTaskSequenceRandomRewardChangeSFTransferAll.HP_LEARNING_RATE_SF: lr_sf,
                ExperimentTaskSequenceRandomRewardChangeSFTransferAll.HP_LEARNING_RATE_REWARD: lr_r,
                ExperimentTaskSequenceRandomRewardChangeSFTransferAll.HP_TASK_SEQUENCE: task_seq,
                ExperimentTaskSequenceRandomRewardChangeSFTransferAll.HP_EXPLORATION: expl
            }, base_dir=base_dir))
        return ExperimentSetTaskSequenceRandomRewardChangeSFTransferAll(exp_list)


