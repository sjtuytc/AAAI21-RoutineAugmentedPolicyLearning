import warnings
warnings.filterwarnings("ignore")
import sys
import os
import os.path as osp
import gym
from collections import defaultdict

import argparse
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.cmd_util import parse_unknown_args
from baselines import logger
from importlib import import_module
from utils.gym_baselines import cleanup_log_dir
from env_makers.baseline_build_env import build_env
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}


def train(args, extra_args):
    env_type, env_id = "atari", args.game_name + "NoFrameskip-v4"
    print('env_type: {}'.format(env_type))

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    learn = get_learn_function(args.alg)

    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)

    env = build_env(args.game_name, args, routine_actions=None, frame_stack=True)
    if args.save_video_interval != 0:
        env = VecVideoRecorder(env, osp.join(logger.get_dir(), "videos"),
                               record_video_trigger=lambda x: x % args.save_video_interval == 0,
                               video_length=args.save_video_length)

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    model = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )

    return model, env


def get_default_network(env_type):
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'


def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs


def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k, v in parse_unknown_args(args).items()}


def configure_logger(log_path, **kwargs):
    # log_path is where to put the ckpt and monitor file.
    if log_path is not None:
        logger.configure(log_path)
    else:
        logger.configure(**kwargs)


def common_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game_name', help='Atari game name', type=str, default='Pong')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--alg', help='Algorithm', type=str, default='a2c')
    parser.add_argument('--num_timesteps', type=float, default=10e6),
    parser.add_argument('--network', help='network type (mlp, cnn, lstm, cnn_lstm, conv_only)', default=None)
    parser.add_argument('--gamestate', help='game state to load (so far only used in retro games)', default=None)
    parser.add_argument('--num_env', help='Number of environment copies being run in parallel. When not specified, set to number of '
                                          'cpus for Atari, and to 1 for Mujoco', default=None, type=int)
    parser.add_argument('--reward_scale', help='Reward scale factor. Default: 1.0', default=1.0, type=float)
    parser.add_argument('--save_video_interval', help='Save video every x steps (0 = disabled)', default=0, type=int)
    parser.add_argument('--save_video_length', help='Length of recorded video. Default: 200', default=200, type=int)

    # routine related information
    parser.add_argument('--eid', type=str, default="a2c_v1", help='current experiment id.')
    parser.add_argument('--log_dir', type=str, default='logs/', help='root folder to save experimental logs.')
    return parser


def main(args):
    # configure logger, disable logging in child MPI processes (with rank > 0)

    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)

    exp_id = args.eid + '_' + args.game_name + "_" + str(args.seed)

    # set up folders to save ckpt and monitor results
    os.makedirs(args.log_dir, exist_ok=True)
    EXP_FOLDER = os.path.join(args.log_dir, exp_id)
    os.makedirs(EXP_FOLDER, exist_ok=True)

    monitor_dir = os.path.join(EXP_FOLDER, 'monitors')
    ckpt_dir = os.path.join(EXP_FOLDER, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(monitor_dir, exist_ok=True)
    cleanup_log_dir(monitor_dir)

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        configure_logger(monitor_dir)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(monitor_dir, format_strs=[])

    model, env = train(args, extra_args)

    model.save(os.path.join(osp.expanduser(ckpt_dir), 'final'))

    env.close()

    return model


if __name__ == '__main__':
    main(sys.argv)
