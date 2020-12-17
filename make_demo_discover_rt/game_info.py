"""
author: Anonymous Author
contact: Anonymous@anonymous
filename: cm_game_info
description: a collection of all game information in our project.
"""
import gym


def atari_actions_all():
    return ['noop()', 'fire()', 'up()', 'right()', 'left()', 'down()', 'up_right()', 'up_left()', 'down_right()',
            'down_left()', 'up_fire()', 'right_fire()', 'left_fire()', 'down_fire()', 'up_right_fire()',
            'up_left_fire()', 'down_right_fire()', 'down_left_fire()']


def coinrun_actions():
    return ['noop()', 'right()', 'jump()']


def game_names_wo_id():
    names = ['Alien', 'Amidar', 'Assault', 'Asterix', 'Asteroids', 'Atlantis', 'BankHeist', 'BattleZone', 'BeamRider',
             'Bowling', 'Boxing', 'Breakout', 'Centipede', 'ChopperCommand', 'CrazyClimber', 'DemonAttack', 'DoubleDunk',
             'Enduro', 'FishingDerby', 'Freeway', 'Frostbite', 'Gopher', 'Gravitar', 'Hero', 'IceHockey', 'Jamesbond',
             'Kangaroo', 'Krull', 'KungFuMaster', 'MontezumaRevenge', 'MsPacman', 'NameThisGame', 'Pitfall', 'Pong',
             'PrivateEye', 'Qbert', 'Riverraid', 'RoadRunner', 'Robotank', 'Seaquest', 'SpaceInvaders', 'StarGunner',
             'Tennis', 'TimePilot', 'Tutankham', 'UpNDown', 'Venture', 'VideoPinball', 'WizardOfWor', 'Zaxxon']
    return names


def game_names_with_id():
    names = game_names_wo_id()
    name_dict = {}
    for idx, name in enumerate(names):
        name_dict[idx] = name
    return name_dict


def get_atari_game_info():
    from at_config import at_config
    import gym
    for game_id in range(50):
        game_name = at_config.GAME_NAMES[game_id]
        print("Cur game id:", game_id, "; game name:", game_name)
        env_id = at_config.GAME_NAMES[game_id] + "NoFrameskip-v4"
        test_env = gym.make(env_id)
        print(test_env.get_action_meanings())


def get_action_num_by_game_name(game_name):
    env_id = game_name + "NoFrameskip-v4"
    test_env = gym.make(env_id)
    num = len(test_env.get_action_meanings())
    test_env.close()
    return num


if __name__ == '__main__':
    # get_atari_game_info()
    game_ids = game_names_with_id()
    print(game_ids)
