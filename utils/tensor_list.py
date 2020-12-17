import torch


# tensor operation
def np_to_torch_dev(input_array, device):
    return torch(input_array).to(device)


def calculate_rank(one_list, big_first):
    a = {}
    rank = 1
    for num in sorted(one_list, reverse=big_first):
        a[num] = rank
        rank = rank + 1
    return[a[i] for i in one_list]


def whether_a_contain_b(list_a, list_b):
    length_b = len(list_b)
    for k, a_ele in enumerate(list_a):
        if k + length_b > len(list_a):
            return False
        else:
            target_list = list_a[k:k+length_b]
            assert len(target_list) == len(list_b)
            if target_list == list_b:
                return True
    return False


def rank_one_by_another(rank_this_list, value_list, large_to_small=True):
    sorted_tuples = sorted(zip(value_list, rank_this_list), key=lambda pair: pair[0], reverse=large_to_small)
    ranked_list = [cur_pair[1] for cur_pair in sorted_tuples]
    corresponding_value = [cur_pair[0] for cur_pair in sorted_tuples]
    return ranked_list, corresponding_value


def combine_lists(one_list):
    """
    Combine a set of lists into one list.
    """
    final_list = []
    for l in one_list:
        final_list += l
    return final_list


def combine_list_to_str(one_list_of_str):
    """
    Combine a set of alphabets into one string.
    """
    return_str = ""
    for cur_str in one_list_of_str:
        return_str += cur_str
    return return_str
