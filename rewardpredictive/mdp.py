#
# Copyright (c) 2020 Lucas Lehnert <lucas_lehnert@brown.edu>
#
# This source code is licensed under an MIT license found in the LICENSE file in the root directory of this project.
#
import pickle
import numpy as np
from tqdm import trange
import rlutils as rl
from itertools import product
from rlutils.environment.gridworld import pt_to_idx, \
    generate_gridworld_transition_function_with_barrier, \
    generate_mdp_from_transition_and_reward_function


class TaskASlightRewardChange(rl.environment.TabularMDP):
    def __init__(self, slip_prob=0.05):
        barrier_idx_list = []
        t_fn = generate_gridworld_transition_function_with_barrier(10, 10, slip_prob, barrier_idx_list)

        start_list_idx = [pt_to_idx((0, 9), (10, 10))]
        goal_list_idx = [pt_to_idx((8, 0), (10, 10))]

        def r_fn(s_1, a, s_2):
            nonlocal goal_list_idx
            if s_2 in goal_list_idx:
                return 1.0
            else:
                return 0.0

        t_mat, r_mat = generate_mdp_from_transition_and_reward_function(100, 4, t_fn, r_fn,
                                                                        reward_matrix=True,
                                                                        dtype=np.float32)
        super().__init__(t_mat, r_mat, start_list_idx, goal_list_idx)

    def __str__(self):
        return 'TaskASlightRewardChange'


class TaskBSlightRewardChange(rl.environment.TabularMDP):
    def __init__(self, slip_prob=0.05):
        barrier_idx_list = []
        t_fn = generate_gridworld_transition_function_with_barrier(10, 10, slip_prob, barrier_idx_list)

        start_list_idx = [pt_to_idx((0, 9), (10, 10))]
        goal_list_idx = [pt_to_idx((9, 1), (10, 10))]

        def r_fn(s_1, a, s_2):
            nonlocal goal_list_idx
            if s_2 in goal_list_idx:
                return 1.0
            else:
                return 0.0

        t_mat, r_mat = generate_mdp_from_transition_and_reward_function(100, 4, t_fn, r_fn,
                                                                        reward_matrix=True,
                                                                        dtype=np.float32)
        super().__init__(t_mat, r_mat, start_list_idx, goal_list_idx)

    def __str__(self):
        return 'TaskBSlightRewardChange'


class TaskASignificantRewardChange(rl.environment.TabularMDP):
    def __init__(self, slip_prob=0.05):
        barrier_idx_list = []
        t_fn = generate_gridworld_transition_function_with_barrier(10, 10, slip_prob, barrier_idx_list)

        start_list_idx = [pt_to_idx((0, 9), (10, 10))]
        goal_list_idx = [pt_to_idx((9, 9), (10, 10))]

        def r_fn(s_1, a, s_2):
            nonlocal goal_list_idx
            if s_2 in goal_list_idx:
                return 1.0
            else:
                return 0.0

        t_mat, r_mat = generate_mdp_from_transition_and_reward_function(100, 4, t_fn, r_fn,
                                                                        reward_matrix=True,
                                                                        dtype=np.float32)
        super().__init__(t_mat, r_mat, start_list_idx, goal_list_idx)

    def __str__(self):
        return 'TaskASignificantRewardChange'


class TaskBSignificantRewardChange(rl.environment.TabularMDP):
    def __init__(self, slip_prob=0.05):
        barrier_idx_list = []
        t_fn = generate_gridworld_transition_function_with_barrier(10, 10, slip_prob, barrier_idx_list)

        start_list_idx = [pt_to_idx((0, 9), (10, 10))]
        goal_list_idx = [pt_to_idx((0, 0), (10, 10))]

        def r_fn(s_1, a, s_2):
            nonlocal goal_list_idx
            if s_2 in goal_list_idx:
                return 1.0
            else:
                return 0.0

        t_mat, r_mat = generate_mdp_from_transition_and_reward_function(100, 4, t_fn, r_fn,
                                                                        reward_matrix=True,
                                                                        dtype=np.float32)
        super().__init__(t_mat, r_mat, start_list_idx, goal_list_idx)

    def __str__(self):
        return 'TaskBSignificantRewardChange'


class FixedTerminalRewardChange(rl.environment.TabularMDP):
    def __init__(self, slip_prob=0.00, size_maze=9):
        barrier_idx_list = []
        self.size_maze = size_maze
        t_fn = generate_gridworld_transition_function_with_barrier(self.size_maze, self.size_maze, slip_prob, barrier_idx_list)

        start_list_idx = [pt_to_idx((0, 0), (self.size_maze, self.size_maze))]
        self.possible_indices = np.arange(0, self.size_maze)
        self.ignore_positions = [(0, 0)]
        middle = self.possible_indices[len(self.possible_indices) // 2]
        idxs = [0, middle, self.possible_indices[-1]]
        self.terminal_positions = list(product(idxs, idxs))
        self.terminal_positions.remove((middle, middle))

        self.goal_position = self.sample_terminal()

        goal_list_idx = [pt_to_idx(self.goal_position, (self.size_maze, self.size_maze))]
        terminal_list_idx = [pt_to_idx(pt, (self.size_maze, self.size_maze)) for pt in self.terminal_positions]

        def r_fn(s_1, a, s_2):
            nonlocal goal_list_idx
            if s_2 in goal_list_idx:
                return 5.0
            else:
                return -0.1

        t_mat, r_mat = generate_mdp_from_transition_and_reward_function(self.size_maze ** 2, 4, t_fn, r_fn,
                                                                        reward_matrix=True,
                                                                        dtype=np.float32)
        super().__init__(t_mat, r_mat, start_list_idx, terminal_list_idx)

    def sample_terminal(self):
        """
        Samples a random position, excluding a given list of positions.
        :param exclude: list of 2-tuples, positions to exclude from sampling
        :return: 2-tuple denoting a position.
        """
        idx = np.random.choice(len(self.terminal_positions))
        pos = self.terminal_positions[idx]
        return pos

    def __str__(self):
        return 'TaskBSignificantRewardChange'


class RandomRewardChange(rl.environment.TabularMDP):
    def __init__(self, slip_prob=0.05, size_maze=10):
        """
        TODO
        :param slip_prob:
        """
        barrier_idx_list = []
        self.size_maze = size_maze
        t_fn = generate_gridworld_transition_function_with_barrier(self.size_maze, self.size_maze, slip_prob, barrier_idx_list)

        start_list_idx = [pt_to_idx((0, 0), (self.size_maze, self.size_maze))]
        self.possible_indices = np.arange(0, self.size_maze)
        self.ignore_positions = [(0, 0)]

        self.goal_position = self.sample_position()

        goal_list_idx = [pt_to_idx(self.goal_position, (self.size_maze, self.size_maze))]

        def r_fn(s_1, a, s_2):
            nonlocal goal_list_idx
            if s_2 in goal_list_idx:
                return 1.0
            else:
                return 0.0

        t_mat, r_mat = generate_mdp_from_transition_and_reward_function(self.size_maze ** 2, 4, t_fn, r_fn,
                                                                        reward_matrix=True,
                                                                        dtype=np.float32)
        super().__init__(t_mat, r_mat, start_list_idx, goal_list_idx)

    def sample_position(self):
        """
        Samples a random position, excluding a given list of positions.
        :param exclude: list of 2-tuples, positions to exclude from sampling
        :return: 2-tuple denoting a position.
        """
        pos = tuple(np.random.choice(self.possible_indices, size=2, replace=True))
        if pos not in self.ignore_positions:
            return pos

        return self.sample_position()

    def __str__(self):
        return 'TaskBSignificantRewardChange'

def mdp_goal_dist(m1, m2, ord=1):
    m1_position = np.array(m1.goal_position)
    m2_position = np.array(m2.goal_position)
    l1_norm = np.linalg.norm(m1_position - m2_position, ord=ord)
    return l1_norm

def generate_l1_tasks(size_maze, num_tasks, lower=None, upper=None,
                             maze_class=RandomRewardChange):
    assert lower is None or upper is None

    mdp_seq = [maze_class(size_maze=size_maze)]

    if lower is None:
        lower = float('inf')

    if upper is None:
        upper = 0

    for t in trange(num_tasks - 1):
        candidate = maze_class(size_maze=size_maze)
        l1_norm = mdp_goal_dist(mdp_seq[-1], candidate)
        while l1_norm > lower or l1_norm < upper or l1_norm == 0:
            candidate = maze_class(size_maze=size_maze)
            l1_norm = mdp_goal_dist(mdp_seq[-1], candidate)

        mdp_seq.append(candidate)

    return mdp_seq

def test_mdps_dist(mdp_seq, dist, bound="lower"):
    assert bound == "lower" or bound == "upper"
    all_positions = [mdp_seq[0].goal_position]
    for i in range(1, len(mdp_seq)):
        l1_norm = mdp_goal_dist(mdp_seq[i - 1], mdp_seq[i])
        if bound == "lower":
            if l1_norm > dist:
                return False
        elif bound == "upper":
            if l1_norm < dist:
                return False
        all_positions.append(mdp_seq[i].goal_position)
    print(f"Tesing for bound {bound}. All positions: {all_positions}")

    return True

if __name__ == "__main__":
    from pathlib import Path
    from definitions import ROOT_DIR

    random_reward_dir = Path(ROOT_DIR, 'data', 'RandomRewardMaze')

    # lower = 3
    # print("Generating similar mdps")
    # lower_mdp_seq = generate_l1_tasks(9, 20, lower=lower, maze_class=FixedTerminalRewardChange)
    # assert test_mdps_dist(lower_mdp_seq, lower, "lower")
    # lower_file_path = random_reward_dir / 'lower_maze.pkl'
    # print(f"Saving to {lower_file_path}")
    # with open(lower_file_path, 'wb') as f:
    #     pickle.dump(lower_mdp_seq, f)

    upper = 10
    print("Generating dissimilar mdps")
    upper_mdp_seq = generate_l1_tasks(9, 20, upper=upper, maze_class=FixedTerminalRewardChange)
    assert test_mdps_dist(upper_mdp_seq, upper, "upper")
    upper_file_path = random_reward_dir / 'upper_maze.pkl'
    with open(upper_file_path, 'wb') as f:
        pickle.dump(upper_mdp_seq, f)
