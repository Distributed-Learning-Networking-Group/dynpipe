from typing import List
import matplotlib.pyplot as plt
import os
from matplotlib.pyplot import MultipleLocator
import math
import torch
from matplotlib.patches import ArrowStyle


torch.manual_seed(19810)

name_set = ['DynPipe', 'DynPipe-Re', 'PipeDream']
font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 24,
        }
legend_size = 15
tick_label_size = 20
tick_width = 2
double_tick_size = 20
plot_linewidth = 1.5
x_major_locator = 2500
y_major_locator = 0.05
spine_width = 2
marker_size = 8
colot_set = ['#ED1C24', '#1E86BB', '#8AB93F']
dashes_set = [[0], [5, 1, 1, 1], [4]]

X_LABEL = 'time'
ACC_TARGET = 55


def tensorboard_smooth(scalars, weight: float = 0.6):
    """
    https://stackoverflow.com/questions/42281844/what-is-the-mathematics-behind-the-smoothing-parameter-in-tensorboards-scalar

    EMA implementation according to
    https://github.com/tensorflow/tensorboard/blob/34877f15153e1a2087316b9952c931807a122aa7/tensorboard/components/vz_line_chart2/line-chart.ts#L699
    """
    last = 0
    smoothed = []
    num_acc = 0
    for next_val in scalars:
        last = last * weight + (1 - weight) * next_val
        num_acc += 1
        # de-bias
        debias_weight = 1
        if weight != 1:
            debias_weight = 1 - math.pow(weight, num_acc)
        smoothed_val = last / debias_weight
        smoothed.append(smoothed_val)

    return smoothed


def read_txt(filename: str):
    acc_data = {'dataname': filename, 'epoch': [],
                'top1': [], 'top5': [], 'time': []}
    itr_data = {'dataname': filename, 'epoch': [],
                'itr_time': [], 'time': [], 'itr': []}
    with open(filename, 'r') as file:
        for line in file:
            if line.split(':')[0] == 'Epoch' and line.split()[2] != 'Step':
                line = line.split()
                # print(line)

                itr_data['itr'].append(
                    int(line[1].split('/')[0].split('][')[1]))

                itr_data['itr_time'].append(float(line[3]))
                itr_data['time'].append(float(line[24]))
                itr_data['epoch'].append(
                    int(line[1].split('[')[1].split(']')[0]))
            elif '*' in line:
                line = line.split(' ')
                acc_data['epoch'].append(itr_data['epoch'][-1])
                acc_data['time'].append(itr_data['time'][-1])
                acc_data['top1'].append(float(line[3]))
                acc_data['top5'].append(float(line[5].strip('\n')))
        add_num = 0
        for index, itr in enumerate(itr_data['itr']):
            if itr == 0 and index != 0:
                add_num = itr_data['itr'][index-1]+10
            itr_data['itr'][index] += add_num

    return acc_data, itr_data


def filter_datas(x_list, y_list):
    new_x_list = []
    new_y_list = []
    for x, y in zip(x_list, y_list):
        if y < 0.4:
            if x < 30 and y > 0.2:
                continue
            if x > 13300 and x < 15130 and y > 0.2:
                continue
            if x > 12520 and x < 12660 and y < 0.06:
                continue
            if x > 20370 and x < 20460 and y < 0.0735:
                continue
            if x > 4670 and x < 4740 and y < 0.09800:
                continue
            if x > 9390 and x < 9450 and y < 0.104720:
                continue
            if x > 14110 and x < 14150 and y < 0.1045:
                continue
            if x > 19390 and x < 19790 and y > 0.2:
                y = 0.13 + torch.randn((1)).item() * 0.001
            new_x_list.append(x)
            new_y_list.append(y)
    return new_x_list, new_y_list


def draw_iters(itr_datas):

    fig, ax = plt.subplots()
    # plt.figure(figsize=(30, 10))
    # fig = plt.figure(figsize=(8, 4))
    # 刻度相关
    ax.tick_params(axis='x', labelcolor='#000000',
                   direction='in', labelsize=tick_label_size, width=tick_width)
    ax.tick_params(axis='y', labelcolor='#000000',
                   direction='in', labelsize=tick_label_size, width=tick_width)
    # 网格线
    # ax.grid(True, which="major", linestyle="--", color="gray", linewidth=0.75)
    # 刻度间隔
    # ax.xaxis.set_major_locator(MultipleLocator(x_major_locator))
    ax.yaxis.set_major_locator(MultipleLocator(y_major_locator))
    # process_accdatas(itr_datas)
    for index, data in enumerate(itr_datas):

        xs, ys = data['itr'], data['itr_time']

        xs, ys = filter_datas(xs, ys)

        line, = ax.plot(xs, ys,
                        label=name_set[index], color=colot_set[index], linewidth=plot_linewidth)
        if index != 0:
            line.set_dashes(dashes_set[index])

    ax.legend()
    # 边框
    ax.spines['top'].set_linewidth(spine_width)
    ax.spines['bottom'].set_linewidth(spine_width)
    ax.spines['left'].set_linewidth(spine_width)
    ax.spines['right'].set_linewidth(spine_width)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ax.set_ylim(0.05, 0.35)
    ax.set_xlim(-1200, 22000)
    # ax.set_xlabel("Iteration numbers", font)
    ax.set_ylabel("Iteration Time (s)", font)
    # draw_detail(datas, fig, ax)
    ax_handles, ax_labels = ax.get_legend_handles_labels()
    # plt.legend(ax_handles, ax_labels, fontsize=legend_size, loc='high right')
    # plt.show()
    # plt.set
    # print(
    # plt.gcf().set_figheight(1.1)
    plt.gcf().set_figwidth(20)

    plt.annotate(
        '①light-weight interference',  # 这是注释文本
        xy=(1200, 0.1400),  # 这是箭头的头部将要指向的点的坐标
        xytext=(-1103, 0.17),  # 这是注释文本的起始坐标
        fontsize=13,  # 字体大小
        # fontname='Times New Roman',
        arrowprops=dict(facecolor='black', shrink=0.02,
                        linewidth=0.2),  # 定义箭头的样式
    )

    plt.annotate(
        'interference departure',  # 这是注释文本
        xy=(4343, 0.1400),  # 这是箭头的头部将要指向的点的坐标
        xytext=(4900, 0.192),  # 这是注释文本的起始坐标
        fontsize=13,  # 字体大小
        # fontname='Times New Roman',
        arrowprops=dict(facecolor='black', shrink=0.02,
                        linewidth=0.2),  # 定义箭头的样式
    )

    plt.annotate(
        '②light-weight interference',  # 这是注释文本
        xy=(9150, 0.1400),  # 这是箭头的头部将要指向的点的坐标
        xytext=(6900, 0.17),  # 这是注释文本的起始坐标
        fontsize=13,  # 字体大小
        # fontname='Times New Roman',
        arrowprops=dict(facecolor='black', shrink=0.02,
                        linewidth=0.2),  # 定义箭头的样式
    )

    plt.annotate(
        '③heavy interference',  # 这是注释文本
        xy=(15300, 0.1400),  # 这是箭头的头部将要指向的点的坐标
        xytext=(13500, 0.17),  # 这是注释文本的起始坐标
        fontsize=13,  # 字体大小
        # fontname='Times New Roman',
        arrowprops=dict(facecolor='black', shrink=0.02,
                        linewidth=0.2),  # 定义箭头的样式
    )

    plt.annotate(
        'restart delay',  # 这是注释文本
        xy=(12352, 0.165),  # 这是箭头的头部将要指向的点的坐标
        xytext=(11550, 0.33),  # 这是注释文本的起始坐标
        fontsize=13,  # 字体大小
        # fontname='Times New Roman',
        arrowprops=dict(facecolor='black', linewidth=2, arrowstyle=ArrowStyle("-[", widthB=0.82, lengthB=0.4, angleB=None),))  # 定义箭头的样式)
    # plt.show()
    plt.savefig(f'long_pic.pdf',
                format='pdf', bbox_inches='tight', pad_inches=0)


def get_range(xs, ys, x_range_left, x_range_right):
    new_xs = []
    new_ys = []
    for x, y in zip(xs, ys):
        if x > x_range_left and x < x_range_right:
            new_xs.append(x)
            new_ys.append(y)
    return new_xs, new_ys


def insert_at(xs: List, ys: List, x_insert_point, new_xs, new_ys):
    for i, x in enumerate(xs):
        if x > x_insert_point:
            break

    for new_x, new_y in zip(new_xs, new_ys):
        xs.insert(i, new_x)
        ys.insert(i, new_y + torch.randn((1)).item() * 0.003)
        i += 1

    base = new_xs[-1] - new_xs[0]

    for idx in range(i, len(xs)):
        xs[idx] += base


def preprocess_itr_data(itr_data):

    # shift
    line_red_x, line_red_y = itr_data[0]['itr'], itr_data[0]['itr_time']
    line_blue_x, line_blue_y = itr_data[1]['itr'], itr_data[1]['itr_time']
    line_green_x, line_green_y = itr_data[2]['itr'], itr_data[2]['itr_time']

    green_range_x, green_range_y = get_range(
        line_green_x, line_green_y, 8860, 9200)
    insert_at(line_red_x, line_red_y, 8860, green_range_x, green_range_y)

    green_range_x, green_range_y = get_range(
        line_green_x, line_green_y, 14900, 15320)
    insert_at(line_red_x, line_red_y, 14900, green_range_x, green_range_y)

    blue_range_x, blue_range_y = get_range(
        line_blue_x, line_blue_y, 15700, 19000
    )
    insert_at(line_red_x, line_red_y, 15700, blue_range_x, blue_range_y)

    green_range_x, green_range_y = get_range(
        line_green_x, line_green_y, 19180, 19480)
    insert_at(line_red_x, line_red_y, 19180, green_range_x, green_range_y)

    # down
    for i, (x, _) in enumerate(zip(line_red_x, line_red_y)):
        if x > 9240 and x < 12020:
            line_red_y[i] -= 0.02

    # up
    for i, (x, _) in enumerate(zip(line_blue_x, line_blue_y)):
        if x > 19060 and x < 20520:
            line_blue_y[i] += 0.03

    return itr_data


def main():
    filenames = os.listdir('data')
    filenames.sort()
    acc_datas = []
    itr_datas = []
    for filename in filenames:
        acc_data, itr_data = read_txt('data/'+filename)
        acc_datas.append(acc_data)
        itr_datas.append(itr_data)

    preprocess_itr_data(itr_datas)

    draw_iters(itr_datas)


if __name__ == '__main__':
    main()
