from gym import spaces
import gym


class RoutineWrapper(gym.Wrapper):
    def __init__(self, env, routine_actions, gamma=0.99):
        gym.Wrapper.__init__(self, env)
        self.env, self.routine_actions = env, routine_actions
        self.prim_action_num = self.env.action_space.n
        self.all_action_num = self.prim_action_num + len(self.routine_actions)
        self.action_space = spaces.Discrete(self.all_action_num)
        self.done = False
        self.gamma = gamma

    def step(self, action):
        if action < self.prim_action_num:
            env_frame, cur_reward, cur_done, cur_info = self.env.step(action)
            cur_info['rt_transition'] = []
            return env_frame, cur_reward, cur_done, cur_info
        else:
            cur_routine = self.routine_actions[int(action) - self.prim_action_num]
            return_reward = 0
            cur_done = False
            env_frame = None
            return_info, return_gamma = {}, self.gamma
            rt_obs, rt_act, rt_rew, rt_done = [], [], [], []
            for idx, act in enumerate(cur_routine):
                env_frame, cur_reward, cur_done, cur_info = self.env.step(act)
                return_reward += cur_reward * (self.gamma ** idx)
                return_gamma = self.gamma ** idx
                # return_reward = cur_reward
                # return_gamma = self.gamma
                rt_obs.append(env_frame), rt_act.append(act), rt_rew.append(cur_reward), rt_done.append(cur_done)
                if isinstance(cur_info, dict) and 'episode' in cur_info.keys():
                    assert cur_done, "episode info appears but not done!"
                    return_info = cur_info
                    break
                if cur_done:
                    break
            return_info['rt_transition'] = {'rt': cur_routine, 'rt_l': len(rt_act), 'rt_obs': rt_obs,
                                            'rt_act': rt_act, 'rt_rew': rt_rew, 'rt_done': rt_done,
                                            'rt_gamma': return_gamma}
            return_done = cur_done
            return env_frame, return_reward, return_done, return_info

    def random_step(self):
        random_a = self.action_space.sample()
        print("Taking action: ", random_a)
        return self.step(random_a)

    def reset(self):
        self.done = False
        return self.env.reset()

