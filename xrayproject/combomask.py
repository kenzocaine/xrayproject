def combine_left_right(left_mask, right_mask):
    index = list(range(0, (len(left_mask))))
    mask = []
    for i in index:
        combination = np.maximum(left_mask[i], right_mask[i])
        mask.append(combination)
    return mask