import numpy as np
import matplotlib.pyplot as plt

def load_list_from_txt(filename):
    return np.loadtxt(filename)
def plot_curve(data):
    plt.plot(data)
    plt.show()
gpu0=load_list_from_txt("time_list_0")
gpu1=load_list_from_txt("time_list_1")
# print(round(gpu0[2]-gpu0[0],2))
for i in range(1,len(gpu0)):
    gpu0[i]=round(gpu0[i]-gpu0[0],2)
for i in range(1,len(gpu1)):
    gpu1[i]=round(gpu1[i]-gpu1[0],2)

for i in range(len(gpu0)):
    print(i,gpu0[i])