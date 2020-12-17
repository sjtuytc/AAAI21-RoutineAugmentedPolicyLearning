"""
author: Anonymous Author
contact: Anonymous@anonymous
filename: cm_mini_sq
description: minimum routine learning function.
"""
import sys
import numpy as np
from Levenshtein import distance
sys.path.insert(0, "../")
from utils.tensor_list import calculate_rank, whether_a_contain_b, rank_one_by_another, combine_list_to_str, combine_lists
from make_demo_discover_rt.pysequitur.main import Sequencer3, print_grammar, AlphabetsTransformer


class CmSqRtProposal:
    """
    Compress an action seq into routines.
    """
    def __init__(self, all_action_seq, prim_num):
        self.prim_n = prim_num
        self.prim_ids = [[i] for i in range(self.prim_n)]
        self.id2rts = [[i] for i in range(self.prim_n)]
        self.all_action_seq = all_action_seq
        # transformer used in Sequitur algorithm.
        self.a_trans = AlphabetsTransformer()
        self.parsed_result, self.rt_actions_idxs, self.avg_freq, self.rt_frequencies, self.rt_scores = \
            None, None, None, None, None

    def cal_dis(self, action_a, action_b):
        encoded_a = combine_list_to_str(self.a_trans.list_ids2alphabets(action_a))
        encoded_b = combine_list_to_str(self.a_trans.list_ids2alphabets(action_b))
        cur_dis = distance(encoded_a, encoded_b)
        return cur_dis

    def run(self, similar_thre=3, select_num=3, size_weight=0.1):
        """
        Main entrance to the sq routine learning algorithm.
        """
        # run sequitur algorithm and get parsed action sequence.
        self.run_sequitur()
        # calculate frequency of the abstracted routines.
        self.evaluate_routines(similar_thre=similar_thre, select_num=select_num)
        return self.rt_actions_idxs

    def run_sequitur(self):
        print("Begin run sequitur for size:", len(self.all_action_seq))
        # we first encode all action seq to alphabets in order to run sequitur algorithm.
        encoded_action_seq = self.a_trans.list_ids2alphabets(self.all_action_seq)
        structure = Sequencer3(encoded_action_seq)
        self.parsed_result = structure.get()

        # collect results to form routines
        rt_actions = []
        rt_considered_nonterminal = []
        for idx, cur_gram in enumerate(self.parsed_result):
            if cur_gram is None:
                continue
            for jdx, cur_ele in enumerate(cur_gram):
                if type(cur_ele) != str and cur_ele not in rt_considered_nonterminal:
                    rt_considered_nonterminal.append(cur_ele)
                    cur_idx = cur_ele.real
                    cur_raw_actions = self.get_actions_for_routine(cur_idx)
                    cur_actions = self.a_trans.list_alphabets2ids(cur_raw_actions)
                    cur_actions = combine_lists([self.id2rts[routine_id] for routine_id in cur_actions])
                    if cur_actions in rt_actions:
                        continue
                    rt_actions.append(cur_actions)
        self.rt_actions_idxs = rt_actions

    def get_actions_for_routine(self, cur_idx):
        # find until no non-terminal variables are killed.
        cur_ori_repre = self.parsed_result[cur_idx]
        return_actions = []
        for idx, ele in enumerate(cur_ori_repre):
            if type(ele) == str:
                return_actions.append(ele)
            else:
                cur_idx = ele.real
                cur_actions = self.get_actions_for_routine(cur_idx)
                return_actions += cur_actions
        return return_actions

    def evaluate_routines(self, similar_thre, select_num, size_weight=0):
        # calculate frequencies and sizes for all routines.
        rt_freqs = []
        rt_sizes = []
        for idx, cur_rt_action in enumerate(self.rt_actions_idxs):
            cur_rt_freq = 0
            for begin_pos, cur_ele in enumerate(self.all_action_seq):
                end_pos = int(begin_pos + len(cur_rt_action))
                if end_pos >= len(self.all_action_seq):
                    break
                if self.all_action_seq[begin_pos:end_pos] == cur_rt_action:
                    # One routine detected.
                    cur_rt_freq += 1
            rt_freqs.append(cur_rt_freq)
            rt_sizes.append(len(cur_rt_action))
        avg_freq, avg_size = np.mean(rt_freqs), np.mean(rt_sizes)

        # collect rt infos
        rt_and_infos = []
        for idx, cur_rt_action in enumerate(self.rt_actions_idxs):
            cur_len = len(cur_rt_action)
            cur_score = rt_freqs[idx] / avg_freq + size_weight * cur_len / avg_size
            cur_rt_and_info = {'rt': cur_rt_action, 'freq': rt_freqs[idx], 'size': cur_len, 'score': cur_score}
            cur_rt_is_worse = False
            # check whether this routine is too similar in comparison to primitive actions
            for jdx, prim_rt in enumerate(self.prim_ids):
                if self.cal_dis(cur_rt_action, prim_rt) < similar_thre:
                    cur_rt_is_worse = True
                    break
            for jdx, ana_info in enumerate(rt_and_infos):
                if cur_rt_is_worse:
                    break
                ana_rt_action = ana_info['rt']
                ana_score = ana_info['score']
                # too similar routines are detected
                if self.cal_dis(cur_rt_action, ana_rt_action) < similar_thre or whether_a_contain_b(cur_rt_action, ana_rt_action) \
                        or whether_a_contain_b(ana_rt_action, cur_rt_action):
                    # if cur score is larger than one existed routine, replace that routine by cur routine
                    if cur_score > ana_score:
                        rt_and_infos.remove(ana_info)
                        cur_rt_is_worse = False
                    # else ignore current routine
                    else:
                        cur_rt_is_worse = True
                    break
            # if cur routine is not similar and worse, append it to routine library
            if not cur_rt_is_worse:
                rt_and_infos.append(cur_rt_and_info)

        # self.rt_frequencies = [np.round(cur_freq / abs(self.avg_freq), 3) for cur_freq in return_freq]
        # self.rt_frequencies = all_freqs

        # parse results
        rt_action_idxs = [cur_info['rt'] for cur_info in rt_and_infos]
        rt_scores = [cur_info['score'] for cur_info in rt_and_infos]
        rt_freqs = [cur_info['freq'] for cur_info in rt_and_infos]
        rt_action_idxs, re_arranged_score = rank_one_by_another(rt_action_idxs, rt_scores)
        rt_freqs, re_arranged_score = rank_one_by_another(rt_freqs, rt_scores)
        rt_scores = re_arranged_score
        self.rt_actions_idxs = rt_action_idxs[:select_num]
        self.rt_frequencies = rt_freqs[:select_num]
        self.rt_scores = rt_scores[:select_num]

        print("Finish abstracting routine.")
        print("Result routines:", self.rt_actions_idxs)
        print("Result routine frequencies:", self.rt_frequencies)
        print("Result routine scores:", self.rt_scores)

        return self.rt_actions_idxs

    def select_routines(self, size_weight, select_num):
        final_rt_actions, final_rt_freqs, final_rt_scores = [], [], []
        for idx, cur_f in enumerate(self.rt_frequencies):
            actions = self.rt_actions_idxs[idx]
            cur_score = cur_f + size_weight * len(actions)
            final_rt_scores.append(cur_score)
        self.rt_actions_idxs, re_arranged_score = rank_one_by_another(self.rt_actions_idxs, final_rt_scores)
        self.rt_frequencies, re_arranged_score = rank_one_by_another(self.rt_frequencies, final_rt_scores)
        final_rt_scores = re_arranged_score
        self.rt_actions_idxs = self.rt_actions_idxs[:select_num]
        self.rt_frequencies = self.rt_frequencies[:select_num]
        return final_rt_scores[:select_num]

    def transfer_routines(self):
        final_rt_actions = []
        for idx, cur_act in enumerate(self.rt_actions_idxs):
            cur_rt_action = []
            for jdx, cur_idx in enumerate(cur_act):
                sub_routine_names = list(self.asm.id2name(cur_idx))
                cur_rt_action += sub_routine_names
            final_rt_actions.append(cur_rt_action)
        return final_rt_actions

    """
    Display and save routines.
    """

    def d(self):
        for idx, cur_act in enumerate(self.rt_actions_idxs):
            print(idx, "th rt; acts:", cur_act, "; freq:", self.rt_frequencies[idx], "; len:", len(cur_act))


if __name__ == '__main__':
    data_num = 800
