import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--algo', default='a2c', help='algorithm to use: a2c | ppo | acktr')
    parser.add_argument('--expert_eid', type=str, default="a2c_v1", help='Experiment id of the expert.')
    parser.add_argument(
        '--gail',
        action='store_true',
        default=False,
        help='do imitation learning with gail')
    parser.add_argument(
        '--gail-experts-dir',
        default='./gail_experts',
        help='directory that contains expert demonstrations for gail')
    parser.add_argument(
        '--gail-batch-size',
        type=int,
        default=128,
        help='gail batch size (default: 128)')
    parser.add_argument(
        '--gail-epoch', type=int, default=5, help='gail epochs (default: 5)')
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=False,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.95,
        help='gae lambda parameter (default: 0.95)')
    parser.add_argument(
        '--entropy-coef',
        type=float,
        default=0.01,
        help='entropy term coefficient (default: 0.01)')
    parser.add_argument(
        '--value-loss-coef',
        type=float,
        default=0.5,
        help='value loss coefficient (default: 0.5)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--routine-num',
        type=int,
        default=3,
        help='how many routines chosen')
    parser.add_argument(
        '--num-processes',
        type=int,
        default=16,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=5,
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument(
        '--ppo-epoch',
        type=int,
        default=4,
        help='number of ppo epochs (default: 4)')
    parser.add_argument(
        '--num-mini-batch',
        type=int,
        default=32,
        help='number of batches for ppo (default: 32)')
    parser.add_argument(
        '--clip-param',
        type=float,
        default=0.2,
        help='ppo clip parameter (default: 0.2)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=1000,
        help='log interval, one log per n STEPS (default: 1000)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=1e6,
        help='save interval, one save per n STEPS (default: 1e6)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=None,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=10e6,
        help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--game_name',
        default='Pong',
        help='game name to train on')
    parser.add_argument(
        '--env-name',
        default=None,
        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument(
        '--log_dir',
        default='./logs',
        help='directory to save agent logs')
    parser.add_argument(
        '--save-dir',
        default=None,
        help='directory to save agent ckpt')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--recurrent-policy',
        action='store_true',
        default=False,
        help='use a recurrent policy')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')

    # use routine
    parser.add_argument(
        '--no-rt-update',
        action='store_true',
        default=False,
        help='disables routine value update')
    parser.add_argument(
        '--use-routine',
        action='store_true',
        default=False,
        help='use routine')
    parser.add_argument('--routine_ablation', type=str, default="", help='Name of the ablated routines.')

    # test arguments.
    parser.add_argument(
        '--test',
        action='store_true',
        default=False,
        help='test trained agents')
    parser.add_argument(
        '--vis-num',
        type=int,
        default=3,
        help='num of visualization')
    parser.add_argument(
        '--fps',
        type=int,
        default=20,
        help='frame per seconds')
    args = parser.parse_args()
    if args.test:
        args.num_processes = 1
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.save_interval = int(args.save_interval)
    assert args.algo in ['a2c', 'ppo', 'acktr']
    if args.recurrent_policy:
        assert args.algo in ['a2c', 'ppo'], 'Recurrent policy is not implemented for ACKTR'

    return args


def get_exp_id(env_name, algorithm, version, use_routine, routine_update, seed, routine_ablation):
    if use_routine and routine_update:
        rid = "rtu"
    elif use_routine and not routine_update:
        rid = 'rtn'
    else:
        rid = 'b'
    if routine_ablation == "":
        exp_id = str(algorithm) + "_" + str(version) + "_" + str(env_name) + "_" + str(rid) + "_" + \
                 str(seed)
    else:
        exp_id = str(algorithm) + "_" + str(version) + "_" + str(env_name) + "_" + str(rid) + "_" + \
                 str(seed) + "_" + str(routine_ablation)
    return exp_id
