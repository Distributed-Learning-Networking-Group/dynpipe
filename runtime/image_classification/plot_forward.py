import numpy as np
import matplotlib.pyplot as plt

# 定义一个函数，将列表保存到文本文件中
def save_list_to_txt(data, filename):
    np.savetxt(filename, data)

# 定义一个函数，从文本文件中读取列表
def load_list_from_txt(filename):
    return np.loadtxt(filename)
def plot_multiple_curves(data_list):
    i=0
    for data in data_list:
        plt.plot(data)
        i+=1
    plt.show()
# 定义一个函数，绘制曲线
def plot_curve(data):
    plt.plot(data)
    plt.show()


# 从文本文件中读取列表，并绘制曲线
loaded_list4 = load_list_from_txt("data_1_bac.txt")#1 b
loaded_list5 = load_list_from_txt("data_0_bac.txt")#0 b
loaded_list = load_list_from_txt("data_0_for.txt")#0 f
loaded_list1 = load_list_from_txt("data_1_for.txt")#1 f
loaded_list2 = load_list_from_txt("data2.txt")#batch
loaded_list3 = load_list_from_txt("data3.txt")#batch

loaded_list4_ = load_list_from_txt("data_1_bac_.txt")#1 b
loaded_list5_ = load_list_from_txt("data_0_bac_.txt")#0 b
loaded_list_ = load_list_from_txt("data_0_for_.txt")#0 f
loaded_list1_ = load_list_from_txt("data_1_for_.txt")#1 f
loaded_list3_ = load_list_from_txt("data3_0.txt")#batch

loaded_list3_0 = load_list_from_txt("data3_0.txt")
loaded_list6 = load_list_from_txt("data6.txt")

loaded_list7 = load_list_from_txt("dataa.txt")
loaded_list8 = load_list_from_txt("datab.txt")
loaded_list9 = load_list_from_txt("datac.txt")
loaded_list10 = load_list_from_txt("datad.txt")
loaded_list11 = load_list_from_txt("datae.txt")

loaded_list12 = load_list_from_txt("dataf.txt")
loaded_list13 = load_list_from_txt("datag.txt")
loaded_list14 = load_list_from_txt("datah.txt")
loaded_list15 = load_list_from_txt("datai.txt")
loaded_list16 = load_list_from_txt("datak.txt")

loaded_list17 = load_list_from_txt("data_rec_0.txt")
loaded_list18 = load_list_from_txt("data_send_0.txt")
loaded_list19 = load_list_from_txt("data_send_1.txt")
loaded_list20 = load_list_from_txt("data_rec_1.txt")

result1 = [x + y for x, y in zip(loaded_list4, loaded_list1)]
result2 = [x + y for x, y in zip(loaded_list, loaded_list5)]

result1_ = [x + y for x, y in zip(loaded_list4_, loaded_list1_)]
result2_ = [x + y for x, y in zip(loaded_list_, loaded_list5_)]

result1_2=[x+y+z+d for x,y,z,d in zip(loaded_list,loaded_list1,loaded_list4,loaded_list5)]
result4_5 = [x + y for x, y in zip(loaded_list4, loaded_list5)]
result3 = [a+b+c+d+e for a,b,c,d,e in zip(loaded_list7,loaded_list8,loaded_list9,loaded_list10,loaded_list11)]
result4 = [a+b+c+d+e for a,b,c,d,e in zip(loaded_list12,loaded_list13,loaded_list14,loaded_list15,loaded_list16)]

result5 = [x + y+z for x, y,z in zip(loaded_list18, loaded_list17,loaded_list5)]
result6 = [x + y+z for x, y,z in zip(loaded_list19, loaded_list20,loaded_list4)]
def average_between_indices(lst, start_index, end_index):
    if start_index < 0 or start_index >= len(lst) or end_index < 0 or end_index >= len(lst) or start_index > end_index:
        return "Invalid indices"
    else:
        sub_list = lst[start_index:end_index+1]
        average = sum(sub_list) / len(sub_list)
        return average
# plot_curve(result1)
# plot_curve(result2)
plot_multiple_curves([result1,result2,result1_,result2_,loaded_list3,loaded_list3_])

# plot_multiple_curves([loaded_list3,loaded_list3_0])
# plot_curve(loaded_list3_0)

# print("no disturb",average_between_indices(loaded_list3,5,280))
# print("disturb",average_between_indices(loaded_list3_0,5,280))
# plot_curve(loaded_list3)
# plot_curve(loaded_list6)
# plot_curve(loaded_list17)
#plot_multiple_curves([result1,result2])
# plot_curve(loaded_list16)
#plot_multiple_curves([loaded_list17,loaded_list18,loaded_list19,loaded_list20])
#plot_multiple_curves([loaded_list2,loaded_list3])
#plot_multiple_curves([loaded_list7,loaded_list8,loaded_list9,loaded_list10,loaded_list11])
#plot_multiple_curves([loaded_list12,loaded_list13,loaded_list14,loaded_list15,loaded_list16])
#plot_multiple_curves([loaded_list4,loaded_list5])


# print(average_between_indices(result1,20,40))
# print(average_between_indices(result1,80,100)) #1f+1b
# print(average_between_indices(result2,20,40))
# print(average_between_indices(result2,80,100)) #0f+0b
# print(average_between_indices(loaded_list2,20,40))
# print(average_between_indices(loaded_list2,80,100)) #batch
#
# print(average_between_indices(loaded_list17,20,40))
# print(average_between_indices(loaded_list17,80,100))
# print(average_between_indices(loaded_list18,20,40))
# print(average_between_indices(loaded_list18,80,100))
# print(average_between_indices(loaded_list19,20,40))
# print(average_between_indices(loaded_list19,80,100))
# print(average_between_indices(loaded_list20,20,40))
# print(average_between_indices(loaded_list20,80,100))
# plot_curve(loaded_list17)
# plot_curve(loaded_list18)
# plot_curve(loaded_list19)
# plot_curve(loaded_list20)


#plot_multiple_curves([loaded_list2,loaded_list3])
#plot_multiple_curves([result1,result2])
#plot_curve(loaded_list7)#run forward
#plot_curve(loaded_list13)#print
#plot_curve(loaded_list14)#zero grad
#plot_curve(loaded_list10)#backward
#plot_curve(loaded_list16) #optimizer
#plot_curve(loaded_list6)
#plot_multiple_curves([result5,loaded_list15])
#plot_curve(loaded_list17)
#plot_curve(loaded_list4)
#plot_multiple_curves([loaded_list10,loaded_list15])



# load_list1=load_list_from_txt("data_0_for")
# load_list2=load_list_from_txt("data_0_bac")
# load_list3=load_list_from_txt("data_1_for")
# load_list4=load_list_from_txt("data_1_bac")
# load_list5=load_list_from_txt("data_2_for")
# load_list6=load_list_from_txt("data_2_bac")
# load_list7=load_list_from_txt("data_3_for")
# load_list8=load_list_from_txt("data_3_bac")
# load_list9=load_list_from_txt("data_batch")
#
# result1 = [x + y for x, y in zip(load_list1, load_list2)]
# result2 = [x + y for x, y in zip(load_list3, load_list4)]
# result3 = [x + y for x, y in zip(load_list5, load_list6)]
# result4 = [x + y for x, y in zip(load_list7, load_list8)]
# plot_curve(load_list9)