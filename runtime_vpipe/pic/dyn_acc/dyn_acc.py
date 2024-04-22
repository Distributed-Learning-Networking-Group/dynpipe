import matplotlib.pyplot as plt
import numpy as np
import os
import math

X_LABEL = 'time'

font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 24,
        }

legend_size = 15
tick_label_size = 20
tick_width = 2
double_tick_size = 20
plot_linewidth = 2.5
x_major_locator = 10
y_major_locator = 10
spine_width = 2
marker_size = 8
colot_set = ['#ED1C24', '#1E86BB', '#8AB93F']
dashes_set = [[0], [5, 1, 1, 1], [4]]


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


def data_process(data, accdatas):
    if 'dynamic' in data['dataname']:
        data['dataname'] = 'DynPipe'
        for index, time in enumerate(data['time']):
            data['time'][index] /= 3600  # second to hour
        data['top1'] = tensorboard_smooth(np.array(data['top1']))

    elif 'pipedream' in data['dataname']:
        data['dataname'] = 'PipeDream'
        for index, time in enumerate(data['time']):
            data['time'][index] /= 3600  # second to hour
        data['top1'] = tensorboard_smooth(np.array(data['top1']))

    return data


def read_txt(filename: str):
    acc_data = {'dataname': filename, 'epoch': [],
                'top1': [], 'top5': [], 'time': []}
    loss_data = {'dataname': filename, 'epoch': [],
                 'loss': [], 'avg_loss': [], 'time': [], }
    with open(filename, 'r') as file:
        for line in file:
            if line.split(':')[0] == 'Epoch' and line.split()[2] != 'Step':
                line = line.split()
                # print(line)
                loss_data['loss'].append(float(line[14]))
                loss_data['avg_loss'].append(
                    float(line[15].replace('(', '').replace(')', '')))
                loss_data['time'].append(float(line[24]))
                loss_data['epoch'].append(
                    int(line[1].split('[')[1].split(']')[0]))
            elif '*' in line:
                line = line.split(' ')
                acc_data['epoch'].append(loss_data['epoch'][-1])
                acc_data['time'].append(loss_data['time'][-1])
                acc_data['top1'].append(float(line[3]))
                acc_data['top5'].append(float(line[5].strip('\n')))

    return acc_data, loss_data


def draw_plot(accdatas, lossdatas):
    fig, ax = plt.subplots()
    ax.tick_params(axis='x', labelcolor='#000000',
                   direction='in', labelsize=tick_label_size, width=tick_width)
    ax.tick_params(axis='y', labelcolor='#000000',
                   direction='in', labelsize=tick_label_size, width=tick_width)

    for index, data in enumerate(accdatas):

        data = data_process(data, accdatas)
        line, = ax.plot(data[X_LABEL], data['top1'],
                        label=data['dataname'], color=colot_set[index], linewidth=plot_linewidth)
        if index != 0:
            line.set_dashes(dashes_set[index])

    ax.legend()
    ax.spines['top'].set_linewidth(spine_width)
    ax.spines['bottom'].set_linewidth(spine_width)
    ax.spines['left'].set_linewidth(spine_width)
    ax.spines['right'].set_linewidth(spine_width)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel("Time (hours)", font)
    ax.set_ylabel("Top-1 Accuracy (%)", font)
    ax_handles, ax_labels = ax.get_legend_handles_labels()
    plt.legend(ax_handles, ax_labels, fontsize=legend_size, loc='lower right')
    plt.savefig(f'{os.path.basename(os.getcwd())}.pdf',
                format='pdf', bbox_inches='tight', pad_inches=0)


def main():
    filenames = os.listdir('data')
    filenames.sort()
    acc_datas = []
    loss_datas = []
    for filename in filenames:
        acc_data, loss_data = read_txt('data/'+filename)
        acc_datas.append(acc_data)
        loss_datas.append(loss_data)
    draw_plot(acc_datas, loss_datas)


if __name__ == '__main__':
    main()
