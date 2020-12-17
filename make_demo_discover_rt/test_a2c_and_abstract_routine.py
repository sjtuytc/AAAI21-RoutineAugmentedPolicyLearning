import sys
import os
import multiprocessing
import os.path as osp
import numpy as np

import argparse
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.logging.FATAL)

from baselines.common.cmd_util import parse_unknown_args
from baselines import logger
from importlib import import_module

from env_makers.baseline_build_env import build_env
from make_demo_discover_rt.baseline_a2c import learn
from make_demo_discover_rt.game_info import get_action_num_by_game_name
from make_demo_discover_rt.game_info import game_names_wo_id
from make_demo_discover_rt.sq_rt_proposal import CmSqRtProposal
from make_demo_discover_rt.ablated_rt_proposal import random_fetch_demonstration, abstract_random_routines
from utils.data_io import save_into_pkl, read_from_pkl, save_into_json


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


def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


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


def run_one_episode(env, model):
    obs = env.reset()

    state = model.initial_state if hasattr(model, 'initial_state') else None
    dones = np.zeros((1,))

    episode_rew_return = -100000

    obs_seq = [obs]
    action_seq = []
    demo_seq = []  # a list of (obs, act, rew, done)
    while True:
        if state is not None:
            actions, _, state, _ = model.step(obs, S=state, M=dones)
        else:
            actions, _, _, _ = model.step(obs)

        obs, rew, done, info = env.step(actions)
        obs_seq.append(obs[0])  # (1, 84, 84, 4)
        action_seq.append(actions[0])
        demo_seq.append((obs[0], actions[0], rew[0], done[0]))

        # env.render()
        done_any = done.any() if isinstance(done, np.ndarray) else done
        if done_any:
            if type(info) == list or type(info) == tuple:
                info = info[0]
            if info is not None and 'episode' in info.keys():
                episode_rew_return = info['episode']['r']
                break
    return obs_seq, action_seq, episode_rew_return, demo_seq


def make_demonstration(game_name, ckpt_path, args, extra_args, save_dir, load_if_exist=True):
    # settings
    env_type, env_id = "atari", game_name + "NoFrameskip-v4"
    seed = args.seed
    demonstration_name = "demonstration_v1"

    alg_kwargs = {}
    alg_kwargs.update(extra_args)

    save_p = os.path.join(save_dir, demonstration_name + ".pkl")

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = 'cnn'
    if load_if_exist and os.path.exists(save_p):
        print("Load from existing demonstration file.")
        demonstration = read_from_pkl(folder=save_dir, name=demonstration_name)
    else:
        print("No existing demonstration is found, making a demonstration.")

        env = build_env(game_name=game_name, args=args)

        print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

        print("Load model from ", ckpt_path)
        model = learn(
            env=env,
            seed=seed,
            total_timesteps=-1,
            load_path=ckpt_path,
            variable_scope='a2c_model',
            **alg_kwargs
        )

        best_rew = -np.inf
        final_obs_seq = []
        final_action_seq = []
        for k in range(10):
            print("Collecting %d th demonstration." % k)
            obs_seq, action_seq, episode_rew, demo_seq = run_one_episode(env, model)
            if episode_rew > best_rew:
                print("Current best score:", episode_rew)
                best_rew = episode_rew
                final_obs_seq = obs_seq
                final_action_seq = action_seq

        demonstration = {'obs_seq': final_obs_seq, "action_seq": final_action_seq, "demo_seq": demo_seq,
                         "score": best_rew}
        if save_dir is not None:
            save_p = save_into_pkl(save_obj=demonstration, folder=save_dir, name=demonstration_name, verbose=False)
            print("Current demonstration is saved at: ", save_p)
            print("The score of current demonstration is:", best_rew)
        env.close()
    obs_seq, action_seq, score = demonstration['obs_seq'], demonstration['action_seq'], demonstration['score']
    return obs_seq, action_seq, score, learn, alg_kwargs


def common_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log_dir', help='Log directory of all expert demonstrations.', type=str,
                        default='logs/')
    parser.add_argument('--expert_eid', type=str, default="a2c_v1", help='Experiment id of the expert.')
    parser.add_argument('--game_name', help='Atari game name', type=str, default='Alien')
    parser.add_argument('--env_type', help='type of environment, used when the environment'
                                           ' type cannot be automatically determined', type=str)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--alg', help='Algorithm', type=str, default='a2c')
    parser.add_argument('--network', help='network type (mlp, cnn, lstm, cnn_lstm, conv_only)', default=None)
    parser.add_argument('--gamestate', help='game state to load (so far only used in retro games)', default=None)
    parser.add_argument('--num_env', help='Number of environment copies being run in parallel. '
                                          'When not specified, set to number of '
                                          'cpus for Atari, and to 1 for Mujoco', default=1, type=int)
    parser.add_argument('--reward_scale', help='Reward scale factor. Default: 1.0', default=1.0, type=float)

    # routine related information
    parser.add_argument('--sim', help='similarity threshold when generating routine proposals', default=2, type=int)
    parser.add_argument('--prop_n', help='proposal number', default=30, type=int)
    return parser


def run_for_one_game(game_name, args, extra_args):
    exp_id = args.expert_eid + '_' + game_name + "_" + str(args.seed)
    expert_exp_folder = os.path.join(args.log_dir, exp_id)

    # set up folders to save ckpt and monitor results
    monitor_dir = os.path.join(expert_exp_folder, 'test_monitors')
    ckpt_dir = os.path.join(expert_exp_folder, 'checkpoints')
    ckpt_path = os.path.join(ckpt_dir, 'final')
    demonstration_dir = os.path.join(expert_exp_folder, 'demonstration')

    os.makedirs(monitor_dir, exist_ok=True)
    os.makedirs(demonstration_dir, exist_ok=True)
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        configure_logger(monitor_dir)  # monitors will be saved at this directory.
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        configure_logger(monitor_dir, format_strs=[])

    obs_seq, action_seq, score, learn_fn, learn_kwargs = \
        make_demonstration(game_name=game_name, ckpt_path=ckpt_path, args=args, extra_args=extra_args,
                           save_dir=demonstration_dir,
                           load_if_exist=True)
    print("Finish making demonstration.")

    # make routine proposals
    print("Make routine proposals.")
    prop = CmSqRtProposal(action_seq, prim_num=get_action_num_by_game_name(game_name))
    rt_proposals = prop.run(similar_thre=args.sim, select_num=args.prop_n)

    routine_root = "abstracted_routines"
    os.makedirs(routine_root, exist_ok=True)

    # make ablated routines
    random_fetch = random_fetch_demonstration(routine_temp=rt_proposals, action_seq=action_seq, seed=args.seed)
    pure_random = abstract_random_routines(routine_temp=rt_proposals, action_seq=action_seq, seed=args.seed)

    result_routines = {'routines': rt_proposals, 'frequencies': prop.rt_frequencies, 'random_fetch': random_fetch,
                       'pure_random': pure_random}
    library_str = exp_id + '_routine_library'
    save_into_json(result_routines, folder=routine_root, file_name=library_str, verbose=True)


def main(args):
    # configure logger, disable logging in child MPI processes (with rank > 0)
    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)
    if args.game_name == 'all':
        for one_game in game_names_wo_id():
            run_for_one_game(one_game, args, extra_args)
    else:
        run_for_one_game(args.game_name, args, extra_args)


if __name__ == '__main__':
    main(sys.argv)
