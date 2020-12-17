from baselines.a2c.a2c import learn
import numpy as np
import math
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.logging.FATAL)


class CmRolloutRtEvaluator:
    def __init__(self, learn_fn, run_one_fn, env_build_fn, rt_proposals, env_kwargs, learn_kwargs):
        self.learn_fn, self.run_one_fn, self.env_build_fn, self.rt_proposals, self.env_kwargs, self.learn_kwargs = \
            learn_fn, run_one_fn, env_build_fn, rt_proposals, env_kwargs, learn_kwargs
        self.sc = 0

    def eval_one_routine_set(self, ras):
        print("Begin evaluating %d th search." % self.sc)
        print("Current routine actions are:", ras)
        self.sc += 1
        env = self.env_build_fn(routine_actions=ras, **self.env_kwargs)
        vs = 'a2c_' + str(self.sc)
        print("Reset before learning.")
        env.reset()
        trained_model = self.learn_fn(env=env, load_path=None, variable_scope=vs, **self.learn_kwargs)
        env.close()
        print("Training finished, begin evaluation.")

        # eval current model
        total_score, eval_ep = 0, 5
        for i in range(eval_ep):
            print("Eval id:", i, end='\r')
            env = self.env_build_fn(routine_actions=ras, **self.env_kwargs)
            obs_seq, action_seq, score = self.run_one_fn(env=env, model=trained_model)
            total_score += score
        env.close()
        del trained_model
        avg_score = total_score / eval_ep
        print("Current score is:", avg_score)
        return avg_score

    def run_forward_search(self):
        print("Training baseline without routines.")
        base_score = self.eval_one_routine_set(ras=None)
        best_score, best_rt = -math.inf, None
        for idx, proposal in enumerate(self.rt_proposals):
            avg_score = self.eval_one_routine_set(ras=[proposal])
            print("The score of routine ", proposal, " is:")
            print("Advantage:", avg_score - base_score, "; Base:", base_score, "; With Rt:", avg_score)

            if avg_score > best_score:
                print("Getting a best routine:", proposal)
                best_rt = proposal
                best_score = avg_score
                print("Advantage:", best_score - base_score, "; Base:", base_score, "; With Rt:", best_score)

        print("Best routine: ", best_rt)
        print("Best rt score: ", best_score)
        print("Baseline score: ", base_score)
        print("Advantage score: ", best_score - base_score)
        return best_rt, best_score - base_score
