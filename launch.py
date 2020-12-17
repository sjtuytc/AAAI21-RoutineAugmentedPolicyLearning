import warnings
warnings.filterwarnings("ignore")
import sys
import os
import os.path as osp
import gym
import argparse

GAME_NAMES = [
    ['Amidar', 'Hero', 'Alien', 'Qbert', 'MsPacman'],
    ['Atlantis', 'BankHeist', 'BattleZone', 'BeamRider'],
    ['Bowling', 'Boxing', 'Breakout', 'Centipede', 'Seaquest'],
    ['DemonAttack', 'DoubleDunk', 'Freeway', 'Gopher'],
    ['Gravitar',  'Pitfall', 'PrivateEye', 'Pong'],
    ['RoadRunner', 'Robotank', 'StarGunner'],
    ['Phoenix', 'Frostbite', 'Jamesbond', 'TimePilot', 'WizardOfWor']
]

RAW_LIST = ['adventure', 'air_raid', 'alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis',
    'bank_heist', 'battle_zone', 'beam_rider', 'berzerk', 'bowling', 'boxing', 'breakout', 'carnival',
    'centipede', 'chopper_command', 'crazy_climber', 'defender', 'demon_attack', 'double_dunk',
    'elevator_action', 'enduro', 'fishing_derby', 'freeway', 'frostbite', 'gopher', 'gravitar',
    'hero', 'ice_hockey', 'jamesbond', 'journey_escape', 'kangaroo', 'krull', 'kung_fu_master',
    'montezuma_revenge', 'ms_pacman', 'name_this_game', 'phoenix', 'pitfall', 'pong', 'pooyan',
    'private_eye', 'qbert', 'riverraid', 'road_runner', 'robotank', 'seaquest', 'skiing',
    'solaris', 'space_invaders', 'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down',
    'venture', 'video_pinball', 'wizard_of_wor', 'yars_revenge', 'zaxxon']


def common_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log_dir', type=str, default='logs/', help='root folder to save experimental logs.')
    parser.add_argument('--game_name', type=str, default='', help='run one game only.')
    parser.add_argument('--mode', type=str, help='specify which code to run.')
    parser.add_argument('--gpu_ids', default=[0, 1, 2, 3], nargs='+', help='gpu ids to run different games')
    parser.add_argument('--game_groups', default=[0, 1, 2, 3], nargs='+', help='game groups to run')
    parser.add_argument('--seed', type=int, default='0', help='random seeds for those games')
    parser.add_argument('--routine_ablation', type=str, default="", help='Name of the ablated routines.')
    return parser


def main(args):
    # configure logger, disable logging in child MPI processes (with rank > 0)

    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    if args.mode == 'expert':  # training expert policy.
        code_path = 'make_demo_discover_rt/train_expert_a2c.py'
    elif args.mode == 'abstraction':  # test and abstract routines.
        code_path = 'make_demo_discover_rt/test_a2c_and_abstract_routine.py'
    elif args.mode == 'baseline':  # run baseliens for policy learning
        code_path = 'torchrl_with_routines/train_routine_policy.py'
    elif args.mode == 'routine':  # run routine-augmented policy learning
        code_path = 'torchrl_with_routines/train_routine_policy.py --use-routine'
        if args.routine_ablation != "":
            code_path += ' --routine_ablation ' + args.routine_ablation
    else:
        raise KeyError("Mode not supported!")
    if args.game_name == '':
        assert len(args.game_groups) <= len(args.gpu_ids), "GPU number is not enough to run so many games!"
        all_commands = []
        for idx, game_group in enumerate(args.game_groups):
            cur_game_list = GAME_NAMES[int(game_group)]
            cur_gpu = args.gpu_ids[idx]
            prefix = "CUDA_VISIBLE_DEVICES=" + str(cur_gpu) + ' nohup python ' + code_path + ' --seed ' + str(args.seed)
            for one_g in cur_game_list:
                exp_name = args.mode + "_" + one_g + "_" + str(args.seed)
                log_path = os.path.join(args.log_dir, exp_name + args.routine_ablation + '.txt')
                cur_str = prefix + ' --game_name ' + one_g + ' --log_dir ' + args.log_dir + ' >' + log_path + ' &'
                all_commands.append(cur_str)
    else:
        cur_gpu = args.gpu_ids[0]
        prefix = "CUDA_VISIBLE_DEVICES=" + str(cur_gpu) + ' nohup python ' + code_path + ' --seed ' + str(args.seed)
        exp_name = args.mode + "_" + args.game_name + "_" + str(args.seed)
        log_path = os.path.join(args.log_dir, exp_name + args.routine_ablation + '.txt')
        cur_str = prefix + ' --game_name ' + args.game_name + ' --log_dir ' + args.log_dir + ' >' + log_path + ' &'
        all_commands = [cur_str]
    print("Running all commands are:", all_commands)
    for one_command in all_commands:
        os.system(one_command)


if __name__ == '__main__':
    main(sys.argv)
