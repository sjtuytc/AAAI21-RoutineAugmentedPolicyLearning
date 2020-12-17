import random


def random_fetch_demonstration(routine_temp, action_seq, seed):
    random.seed(seed)
    result_routines = []
    for idx, one_rt in enumerate(routine_temp):
        start_int = random.randint(0, len(action_seq) - len(one_rt))
        fetched = action_seq[start_int: start_int + len(one_rt)]
        fetched = [int(ele) for ele in fetched]
        result_routines.append(fetched)
    return result_routines


def abstract_random_routines(routine_temp, action_seq, seed):
    random.seed(seed)
    result_routines = []
    for idx, one_rt in enumerate(routine_temp):
        generated_one_rt = []
        for jdx in range(len(one_rt)):
            random_variable = random.choice(list(set(action_seq)))
            random_variable = int(random_variable)
            generated_one_rt.append(random_variable)
        result_routines.append(generated_one_rt)
    return result_routines
