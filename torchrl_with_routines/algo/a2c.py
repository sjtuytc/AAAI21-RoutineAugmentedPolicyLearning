import torch
import torch.nn as nn
import torch.optim as optim
import random


def clip_loss(loss, clip_range):
    if abs(loss.item()) > clip_range:
        return loss * 0
    else:
        return loss


class A2C:
    def __init__(self,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 alpha=None,
                 max_grad_norm=None,
                 use_smooth_loss=False):

        self.actor_critic = actor_critic

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        self.optimizer = optim.RMSprop(actor_critic.parameters(), lr, eps=eps, alpha=alpha)
        self.smooth_l1 = torch.nn.SmoothL1Loss()
        self.use_smooth = use_smooth_loss
        random.seed(0)

    def update(self, rollouts, rt_rollouts=None):
        # obs.size(): [num_step + 1, num_env, frame_stack, w, h], actions.size(): [num_step, num_env, 1]
        # rewards.size(): [num_step, num_env, 1]
        obs_shape = rollouts.obs.size()[2:]  # [frame_stack, w, h]
        action_shape = rollouts.actions.size()[-1]  # [1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        # Calculate value of states, action probs, distribution entropy.
        values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),  # [num_step * num_env, frame_stack, w, h]
            rollouts.recurrent_hidden_states[0].view(-1, self.actor_critic.recurrent_hidden_state_size),  # [num_env, 1]
            rollouts.masks[:-1].view(-1, 1),  # [num_step * num_env, 1]
            rollouts.actions.view(-1, action_shape))  # [num_step * num_env, 1]

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values  # [n_step, n_env, 1]
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()

        if rt_rollouts is not None:  # calculate losses within routines
            num_steps, num_processes, _ = rt_rollouts.rewards.size()

            # Calculate value of states, action probs, distribution entropy.
            rt_values, rt_action_log_probs, rt_dist_entropy, _ = self.actor_critic.evaluate_actions(
                rt_rollouts.obs[:-1].view(-1, *obs_shape),  # [num_step * num_env, frame_stack, w, h]
                None,
                rt_rollouts.masks[:-1].view(-1, 1),  # [num_step * num_env, 1]
                rt_rollouts.actions.view(-1, action_shape))  # [num_step * num_env, 1]

            rt_values = rt_values.view(num_steps, num_processes, 1)
            rt_action_log_probs = rt_action_log_probs.view(num_steps, num_processes, 1)

            rt_advantages = rt_rollouts.returns[:-1] - rt_values  # [n_step, n_env, 1]
            rt_value_loss = clip_loss(rt_advantages.pow(2).mean(), 10)
            # rt_action_loss = clip_loss(-(rt_advantages.detach() * rt_action_log_probs).mean(), 10)
            value_loss = value_loss * 0.5 + rt_value_loss * 0.5
            # action_loss = rt_action_loss
            # dist_entropy = rt_dist_entropy
            # value_loss += rt_value_loss
            # action_loss += rt_action_loss
            # dist_entropy += rt_dist_entropy
            # if random.random() < 0.5:
            #     value_loss = rt_value_loss
            #     action_loss = rt_action_loss
            #     dist_entropy = rt_dist_entropy
        self.optimizer.zero_grad()
        total_loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

        self.optimizer.step()
        if rt_rollouts is not None:
            detail_log = "Avgs are, values {:.3f}, advantages {:3f}, act_log {:.3f}, dist_entropy {:.3f}, rt_values {:.3f}, " \
                     "rt_advantages {:3f}, rt_act_log {:.3f}, rt_dist_entropy {:.3f}".\
            format(values.mean(), advantages.pow(2).mean(), action_log_probs.mean(), dist_entropy.mean(), rt_values.mean(),
                         rt_advantages.pow(2).mean(), rt_action_log_probs.mean(), rt_dist_entropy.mean())
        else:
            detail_log = ""
        return value_loss.item(), action_loss.item(), dist_entropy.item(), detail_log
