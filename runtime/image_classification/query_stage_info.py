import os
import numpy as np

batch_size = 32

stage_nums = [10, 10, 10, 8]

saved_stage_info_list = np.array([])

with open('data.npy', 'rb') as f:
    saved_stage_info_list = np.load(f)
# insert all zeros vector at the beginning to enable cumsum calculation
saved_stage_info_list = np.insert(
    saved_stage_info_list, 0, np.zeros(saved_stage_info_list[0].shape), axis=0)
saved_stage_info_list = np.array(saved_stage_info_list, dtype=int)

# * return message
query_stage_info_list = []

cumsum_stage_nums = np.cumsum(stage_nums)
for idx in range(len(cumsum_stage_nums) - 1):
    if idx == 0:
        query_stage_info_list.append(
            saved_stage_info_list[cumsum_stage_nums[idx]][0]
        )
    else:
        query_stage_info_list.append(
            saved_stage_info_list[cumsum_stage_nums[idx]][0] -
            saved_stage_info_list[cumsum_stage_nums[idx - 1]][0]
        )
query_stage_info_list.append(
    saved_stage_info_list[-1][0] -
    saved_stage_info_list[cumsum_stage_nums[-2]][0]
)

print(query_stage_info_list)
