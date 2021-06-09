import numpy as np

def combine_left_right(left_mask, right_mask):
    index = list(range(0, (len(left_mask))))
    mask = []
    for i in index:
        combination = np.maximum(left_mask[i], right_mask[i])
        mask.append(combination)
    return mask

def USA_data(path, path_right, path_left):
    left,targets, ID = load_masks(10, get_all = False, get_random = False, balanced = True, path = path_left)
    right, ID = load_train(ID, path_right)
    USA_masks = combine_left_right(left, right)
    USA_images, ID = load_train(ID, path)
    return USA_masks, USA_images