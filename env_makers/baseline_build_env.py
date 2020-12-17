import sys
import multiprocessing
import os
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import gym
from gym.wrappers import FlattenObservation, FilterObservation
from baselines import logger
from baselines.bench import Monitor
from baselines.common import set_global_seeds
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.wrappers import ClipActionsWrapper
from torchrl_with_routines.routine_wrapper import RoutineWrapper


def build_env(game_name, args, routine_actions=None, eval_mode=False, inside_frame_stack=True):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin':
        ncpu //= 2
    nenv = args.num_env or ncpu
    if eval_mode:
        nenv = 1
    alg = args.alg
    seed = args.seed

    env_type, env_id = "atari", game_name + "NoFrameskip-v4"

    if env_type == 'atari' and alg not in ['deepq', 'trpo_mpi']:
        assert args.reward_scale == 1, "we assume the reward equals to the score in our code."
        if inside_frame_stack:
            wrapper_kwargs = {'frame_stack': True}
        env = make_vec_env(env_id, env_type, nenv, seed, gamestate=args.gamestate, reward_scale=args.reward_scale,
                           routine_actions=routine_actions, wrap_monitor=True, wrapper_kwargs=wrapper_kwargs)
        if not inside_frame_stack:
            env = VecFrameStack(env, nstack=4)
        # the original implementation uses vecframestack, but we use framestack inside vec wrapper.
    else:
        raise NotImplementedError("We remove other specifications.")

    return env


def make_vec_env(env_id, env_type, num_env, seed,
                 wrapper_kwargs=None,
                 env_kwargs=None,
                 start_index=0,
                 reward_scale=1.0,
                 flatten_dict_observations=True,
                 gamestate=None,
                 initializer=None,
                 force_dummy=False,
                 routine_actions=None,
                 wrap_monitor=True):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari and MuJoCo.
    """
    wrapper_kwargs = wrapper_kwargs or {}
    env_kwargs = env_kwargs or {}
    mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    seed = seed + 10000 * mpi_rank if seed is not None else None
    logger_dir = logger.get_dir()

    def make_thunk(rank, initializer=None):
        return lambda: make_env(
            env_id=env_id,
            env_type=env_type,
            mpi_rank=mpi_rank,
            subrank=rank,
            seed=seed,
            reward_scale=reward_scale,
            gamestate=gamestate,
            flatten_dict_observations=flatten_dict_observations,
            wrapper_kwargs=wrapper_kwargs,
            env_kwargs=env_kwargs,
            logger_dir=logger_dir,
            initializer=initializer,
            routine_actions=routine_actions,
            wrap_monitor=wrap_monitor,
        )

    set_global_seeds(seed)
    if not force_dummy and num_env > 1:
        return SubprocVecEnv([make_thunk(i + start_index, initializer=initializer) for i in range(num_env)])
    else:
        return DummyVecEnv([make_thunk(i + start_index, initializer=None) for i in range(num_env)])


def make_env(env_id, env_type, mpi_rank=0, subrank=0, seed=None, reward_scale=1.0, gamestate=None,
             flatten_dict_observations=True, wrapper_kwargs=None, env_kwargs=None, logger_dir=None, initializer=None,
             routine_actions=None, wrap_monitor=True):
    if initializer is not None:
        initializer(mpi_rank=mpi_rank, subrank=subrank)

    wrapper_kwargs = wrapper_kwargs or {}
    env_kwargs = env_kwargs or {}
    if ':' in env_id:
        import re
        import importlib
        module_name = re.sub(':.*','',env_id)
        env_id = re.sub('.*:', '', env_id)
        importlib.import_module(module_name)
    if env_type == 'atari':
        env = make_atari(env_id)
    else:
        raise NotImplementedError("Currently, we only support Atari.")

    if flatten_dict_observations and isinstance(env.observation_space, gym.spaces.Dict):
        env = FlattenObservation(env)

    env.seed(seed + subrank if seed is not None else None)
    if wrap_monitor:
        env = Monitor(env,
                      logger_dir and os.path.join(logger_dir, str(mpi_rank) + '.' + str(subrank)),
                      allow_early_resets=True)

    if env_type == 'atari':
        env = wrap_deepmind(env, **wrapper_kwargs)
    else:
        raise NotImplementedError("Currently, we only support Atari.")

    if isinstance(env.action_space, gym.spaces.Box):
        env = ClipActionsWrapper(env)

    if routine_actions is not None:
        env = RoutineWrapper(env=env, routine_actions=routine_actions)

    if reward_scale != 1:
        raise RuntimeWarning("reward scale is not 1!")
    return env


def make_mujoco_env(env_id, seed, reward_scale=1.0):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.
    """
    rank = MPI.COMM_WORLD.Get_rank()
    myseed = seed + 1000 * rank if seed is not None else None
    set_global_seeds(myseed)
    env = gym.make(env_id)
    logger_path = None if logger.get_dir() is None else os.path.join(logger.get_dir(), str(rank))
    env = Monitor(env, logger_path, allow_early_resets=True)
    env.seed(seed)
    if reward_scale != 1.0:
        from baselines.common.retro_wrappers import RewardScaler
        env = RewardScaler(env, reward_scale)
    return env


def make_robotics_env(env_id, seed, rank=0):
    """
    Create a wrapped, monitored gym.Env for MuJoCo.
    """
    set_global_seeds(seed)
    env = gym.make(env_id)
    env = FlattenObservation(FilterObservation(env, ['observation', 'desired_goal']))
    env = Monitor(
        env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
        info_keywords=('is_success',))
    env.seed(seed)
    return env

