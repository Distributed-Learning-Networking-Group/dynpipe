layer_communication_list=[1,2,1,2,1,2,1,2,1,2]
layer_forward_list=[1,1,1,1,1,1,1,1,1,1,3]
layer_backward_list=[2,2,2,2,2,2,2,2,2,2,3]
top_k=1
stage_nums=[1,1,1,1,1,1,1,3]
stage_num=8
layer_communication_list_present=layer_communication_list.copy()#input layer
layer_communication_list_back=layer_communication_list.copy()#critirien
layer_communication_list_present.insert(0,0)
layer_communication_list_back.append(0)
layer_process_time = [x + y+z+m for x, y,z,m in zip(layer_forward_list, layer_backward_list,layer_communication_list_present,layer_communication_list_back)]
straggle_for_stage_=[1,1,1,1]
dp_process_time=[]
dp_process_communication_time=[]
print(layer_process_time)
import heapq
def top_k_max_indices(lst, start, end, k):
    max_indices = heapq.nlargest(k, range(start, end), key=lst.__getitem__)
    max_indices.sort()
    return max_indices
def calculate_stage_end_indices(stage_sizes):
    # 计算每个阶段的结束index
    end_indices = [sum(stage_sizes[:i + 1]) for i in range(len(stage_sizes))]
    return end_indices
stage_index=calculate_stage_end_indices(stage_nums)
stage_index.insert(0,0)
print(stage_index)
max_indexes=[]
for i in range(1,len(stage_index)):
    max_index=top_k_max_indices(layer_process_time,stage_index[i-1],stage_index[i],top_k)
    max_index=[x+1 for x in max_index]
    max_indexes+=max_index
max_indexes.insert(0,0)
max_indexes.append(len(layer_forward_list))
print("max_indexes",max_indexes)
layer_forward_list_new=[]
layer_backward_list_new=[]
layer_communication_list_new=[]
for i in range(1, len(max_indexes) - 1):
    layer_forward_list_new.append(sum(layer_forward_list[max_indexes[i - 1]:max_indexes[i]]))
    layer_backward_list_new.append(sum(layer_backward_list[max_indexes[i - 1]:max_indexes[i]]))
for i in range(1, len(max_indexes) - 2):
    layer_communication_list_new.append(layer_communication_list[max_indexes[i] - 1])
if (max_indexes[-2] - 1) != len(layer_forward_list) - 1:
    layer_forward_list_new[-1] += sum(layer_forward_list[max_indexes[-2]:len(layer_forward_list)])
    layer_backward_list_new[-1] += sum(layer_backward_list[max_indexes[-2]:len(layer_forward_list)])
print("new forward",layer_forward_list_new)
print("new backward",layer_backward_list_new)
print("new communication",layer_communication_list_new)
import numpy as np
record=np.full((len(layer_forward_list_new),)*(stage_num-1),np.inf)
print("record",record.shape)
from itertools import combinations
items = list(range(1, len(layer_forward_list_new)))
a = list(combinations(items, stage_num-1))
for i in a:
    layer_communication_list_new_ = []
    present_stage_forward = []
    present_stage_backward = []
    for j in range(len(i)+1):
        if j==0:
            present_stage_forward.append(sum(layer_forward_list_new[0:i[j]]))
        elif j==len(i):
            present_stage_forward.append(sum(layer_forward_list_new[i[j - 1]:len(layer_backward_list_new)]))
        else:
            present_stage_forward.append(sum(layer_forward_list_new[i[j-1]:i[j]]))
    for j in range(len(i)+1):
        if j==0:
            present_stage_backward.append(sum(layer_backward_list_new[0:i[j]]))
        elif j==len(i):
            present_stage_backward.append(sum(layer_backward_list_new[i[j - 1]:len(layer_backward_list_new)]))
        else:
            present_stage_backward.append(sum(layer_backward_list_new[i[j-1]:i[j]]))
    for j in range(len(i)):
        layer_communication_list_new_.append(layer_communication_list_new[i[j]-1])


    # print(present_stage_forward)
    # print(present_stage_backward)
    # print(layer_communication_list_new_)
min_index=[1,2,3,4,5,6,7]

new_stage_nums=[]
for i in range(len(min_index)+1):
    if i==0:
        new_stage_nums.append(max_indexes[min_index[0]])
    elif i==len(min_index):
        new_stage_nums.append(len(layer_forward_list) - max_indexes[min_index[i-1]])
    else:
        new_stage_nums.append(max_indexes[min_index[i]] - max_indexes[min_index[i - 1]])
print(new_stage_nums)
# for i in range(1, len(layer_forward_list_new)):
#     layer_communication_list_new_ = []
#     present_stage_forward = []
#     present_stage_backward = []
#     stage_information = []
#     new_stage_nums = [max_indexes[i], len(layer_forward_list) - max_indexes[i]]
#     print(new_stage_nums, straggle_for_stage_)
#     present_stage_forward.append(straggle_for_stage_[0] * sum(layer_forward_list_new[0:i]))
#     present_stage_forward.append(straggle_for_stage_[1] * sum(layer_forward_list_new[i:len(layer_forward_list_new)]))
#
#     present_stage_backward.append(straggle_for_stage_[0] * sum(layer_backward_list_new[0:i]))
#     present_stage_backward.append(straggle_for_stage_[1] * sum(layer_backward_list_new[i:len(layer_forward_list_new)]))
#     layer_communication_list_new_.append(layer_communication_list_new[i - 1])
#     record[i] = main(stage_num, present_stage_forward, present_stage_backward, layer_communication_list_new_, 99, 0,
#                      0)