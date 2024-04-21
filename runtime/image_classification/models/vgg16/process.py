import json

# 读取 txt 文件并将数据转换成 list
with open('vgg16_data_1_bac_.txt', 'r') as file:
    data = file.read().splitlines()

# 将数据转换成 list
data_list1 = [float(x) for x in data]
with open('vgg16_data_1_for_.txt', 'r') as file:
    data = file.read().splitlines()

# 将数据转换成 list
data_list2 = [float(x) for x in data]

print(sum(data_list1),sum(data_list2))
