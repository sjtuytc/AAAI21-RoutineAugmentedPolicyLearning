import os
import time
from collections import deque

import numpy as np

from torchrl_with_routines import algo
from torchrl_with_routines.a2c_arguments import get_args
from torchrl_with_routines.model import Policy
from torchrl_with_routines.storage import RolloutStorage
from torchrl_with_routines.a2c_arguments import get_exp_id
from env_makers.torchrl_build_env import make_vec_envs
from utils.data_io import load_routine_action, imgseq2video
from utils.gym_baselines import cleanup_log_dir, get_vec_normalize
from utils.training import update_linear_schedule
from utils.tensor_list import *


def get_trainable_parameter_norm(model):
    with torch.no_grad():
        norm = sum([p.norm(1).item() for p in model.parameters() if p.requires_grad])
    return norm


def main():
    # set up exp folders and settings
    args = get_args()
    env_name = args.game_name + "NoFrameskip-v4" if args.env_name is None else args.env_name

    exp_id = get_exp_id(env_name=env_name, algorithm='a2c', version='v1', use_routine=args.use_routine,
                        routine_update=not args.no_rt_update, seed=args.seed, routine_ablation=args.routine_ablation)
    exp_folder = os.path.join(args.log_dir, exp_id)
    log_dir = os.path.expanduser(exp_folder)
    os.makedirs(log_dir, exist_ok=True)
    ckpt_path = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_path, exist_ok=True)
    ckpt_folder = ckpt_path if args.save_dir is None else args.save_dir
    ckpt_file = os.path.join(ckpt_folder, env_name + ".pt")

    if args.test:
        log_dir = os.path.join(args.log_dir, exp_id, 'test_monitors')
        os.makedirs(log_dir, exist_ok=True)
        cleanup_log_dir(log_dir)
    else:
        cleanup_log_dir(log_dir)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    print("Exp dir is:", log_dir)
    cleanup_log_dir(log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    routine_file_name = args.expert_eid + '_' + str(args.game_name) + "_" + str(args.seed) + '_routine_library'
    routine_folder = "abstracted_routines"
    routine_actions = load_routine_action(routine_folder, routine_file_name, args.routine_num, args.routine_ablation) \
        if args.use_routine else None

    envs = make_vec_envs(env_name=env_name, seed=args.seed, num_processes=args.num_processes,
                         gamma=args.gamma, log_dir=log_dir, device=device, allow_early_resets=False,
                         routine_actions=routine_actions, num_frame_stack=-1, inner_frame_stack=True)

    # raise RuntimeError
    print("Creating environment finished!")

    # Create policy object
    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    if args.test:
        actor_critic = torch.load(ckpt_file)[0]
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C(actor_critic, args.value_loss_coef, args.entropy_coef, lr=args.lr, eps=args.eps,
                         alpha=args.alpha, max_grad_norm=args.max_grad_norm)
    else:
        raise NotImplementedError
    assert not args.recurrent_policy, "we don't support recurrent policy for routine policy learning yet."

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space, args.gamma,
                              actor_critic.recurrent_hidden_state_size)

    use_routine_and_update = (not args.no_rt_update) and args.use_routine
    rt_rollouts = RolloutStorage(args.num_steps, args.num_processes,
                                 envs.observation_space.shape, envs.action_space, args.gamma,
                                 actor_critic.recurrent_hidden_state_size) if use_routine_and_update else None

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)
    if use_routine_and_update:
        rt_rollouts.obs[0].copy_(obs)
        rt_rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    step_counter, last_save, last_log, vis_counter = 0, 0, 0, 0
    episode_video = []
    # main training loop
    while step_counter < int(args.num_env_steps):
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            update_linear_schedule(agent.optimizer, step_counter, args.num_env_steps, agent.optimizer.lr)
        raw_rollouts = [{'in_rt': [], 'obs': [], 'act': [], 'rew': [], 'mask': []} for i in range(envs.num_envs)]
        # add start obs and mask
        for env_idx in range(envs.num_envs):
            raw_rollouts[env_idx]['obs'].append(rollouts.obs[0][env_idx])
            raw_rollouts[env_idx]['mask'].append(rollouts.masks[0][env_idx])
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step], rollouts.masks[step])

            # Step env.
            if args.test:
                episode_video.append(envs.render(mode='rgb_array'))
            obs, reward, done, infos = envs.step(action)
            fit_gammas = []
            for env_idx, info in enumerate(infos):
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    if args.test:
                        use_rt = 'rt' if args.use_routine else 'base'
                        print("Current score:", info['episode']['r'])
                        imgseq2video(episode_video, name=args.game_name + '_' + use_rt + '_' + str(vis_counter), folder=log_dir,
                                     verbose=True, fps=args.fps)
                        vis_counter += 1
                        if vis_counter > args.vis_num:
                            raise RuntimeError("Stop testing.")
                        episode_video = []
                assert routine_actions is None or 'rt_transition' in info.keys(), 'Failed to get rt transition.'

                if 'rt_transition' in info.keys():  # routine_actions are in the action space
                    if len(info['rt_transition']) < 1:  # primitive action
                        raw_rollouts[env_idx]['in_rt'].append(False)
                        raw_rollouts[env_idx]['act'].append(action[env_idx])
                        raw_rollouts[env_idx]['obs'].append(obs[env_idx])
                        raw_rollouts[env_idx]['rew'].append(reward[env_idx])
                        cur_mask = 0.0 if done[env_idx] else 1.0
                        raw_rollouts[env_idx]['mask'].append(cur_mask)
                        fit_gammas.append(args.gamma)
                        step_counter += 1
                        continue
                    else:  # use routine
                        raw_rollouts[env_idx]['in_rt'].append(True)
                        rtt = info['rt_transition']
                        rt, rt_l, rt_obs, rt_act, rt_rew, rt_done, rt_gamma = \
                            rtt['rt'], rtt['rt_l'], rtt['rt_obs'], rtt['rt_act'], rtt['rt_rew'], rtt['rt_done'], rtt['rt_gamma']
                        step_counter += rt_l
                        raw_rollouts[env_idx]['act'] += rt_act
                        raw_rollouts[env_idx]['obs'] += rt_obs
                        raw_rollouts[env_idx]['rew'] += rt_rew
                        rt_masks = [[0.0] if one_done else [1.0] for one_done in rt_done]
                        raw_rollouts[env_idx]['mask'] += rt_masks
                        fit_gammas.append(rt_gamma)
                        # check lengths
                        if not len(rt_obs) == len(rt_masks) == len(rt_rew) == rt_l:
                            value_error = "obs/masks/rew/act lenths are:" + str(len(rt_obs) - 1) + str(len(rt_masks) - 1) \
                                          + str(len(rt_rew)) + str(rt_l)
                            raise RuntimeError(value_error)
                else:
                    step_counter += 1
                    fit_gammas.append(args.gamma)
            # If done, then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
            fit_gammas = torch.FloatTensor(fit_gammas).view(-1, 1)
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks, gammas=fit_gammas)

        if not args.test:
            # get the value prediction of the last states.
            with torch.no_grad():
                next_value = actor_critic.get_value(rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                                                    rollouts.masks[-1]).detach()
                if use_routine_and_update:
                    rt_rollouts.insert_routine_experience(rt_rollouts=raw_rollouts)
                    next_rt_value = actor_critic.get_value(rt_rollouts.obs[-1], None, rt_rollouts.masks[-1]).detach()

            rollouts.compute_returns(next_value, args.use_gae, args.gae_lambda, args.use_proper_time_limits)
            if use_routine_and_update:
                rt_rollouts.compute_returns(next_rt_value, args.use_gae, args.gae_lambda,
                                            args.use_proper_time_limits)
                value_loss, action_loss, dist_entropy, detail_log = agent.update(rollouts, rt_rollouts=rt_rollouts)
            else:
                value_loss, action_loss, dist_entropy, detail_log = agent.update(rollouts, rt_rollouts=None)

            rollouts.after_update()

            # save for every interval-th episode or for the last epoch
            if (step_counter >= last_save or step_counter >= args.num_env_steps - 1) and ckpt_folder != "":
                last_save += args.save_interval
                torch.save([actor_critic, getattr(get_vec_normalize(envs), 'ob_rms', None)], ckpt_file)

            if len(episode_rewards) <= 0:
                episode_rewards = [-1]
            if step_counter >= last_log and len(episode_rewards) > 0:
                norm_value = get_trainable_parameter_norm(actor_critic)
                last_log += args.log_interval
                end = time.time()
                fps = int(step_counter / (end - start))
                time_left = float(args.num_env_steps - step_counter) / fps / 3600
                print(detail_log)
                print("Env steps {}, FPS {}, time Left {:.1f}h, value/act/entro loss {:.3f}/{:.3f}/{:.3f}, norm value {:.3f}"
                      " \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, "
                      "\n".format(step_counter, fps, time_left, value_loss, action_loss, dist_entropy, norm_value,
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards), np.max(episode_rewards)))
                if -1 in episode_rewards:
                    episode_rewards.remove(-1)


if __name__ == "__main__":
    main()
