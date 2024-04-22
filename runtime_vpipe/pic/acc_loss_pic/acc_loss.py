import matplotlib.pyplot as plt
import os
from matplotlib.pyplot import MultipleLocator


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


def rename(data):
    if 'DynPipe' in data['dataname']:
        data['dataname'] = 'DynPipe'
    elif 'PipeDream' in data['dataname']:
        data['dataname'] = 'PipeDream'
    elif 'GPipe' in data['dataname']:
        data['dataname'] = 'GPipe'
    return data


def read_txt(filename: str):
    acc_data = {'dataname': filename, 'top1': [], 'time': []}
    loss_data = {'dataname': filename, 'avg_loss': [], 'time': []}
    with open(filename, 'r') as file:
        for line in file:
            acc_data['time'].append(float(line.split()[0]))
            acc_data['top1'].append(float(line.split()[1]))

    return acc_data, loss_data


def draw_plot(accdatas, model_datasets):
    fig, ax = plt.subplots()

    if model_datasets == 'resnet50_cifar10':
        ax.set_ylim(10, 80)
        ax.set_xlim(-0.5, 9)
        ax.set_xlabel("Time (hours)", font)
        ax.set_ylabel("Top-1 Accuracy (%)", font)
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.yaxis.set_major_locator(MultipleLocator(15))
    elif model_datasets == 'vgg16_cifar10':
        ax.set_ylim(15, 90)
        ax.set_xlim(-0.2, 3.5)
        ax.set_xlabel("Time (hours)", font)
        ax.set_ylabel("Top-1 Accuracy (%)", font)
    elif model_datasets == 'vgg16_image64':
        ax.set_ylim(-5, 72)
        ax.set_xlim(-1.5, 24)
        ax.set_xlabel("Time (hours)", font)
        ax.set_ylabel("Top-1 Accuracy (%)", font)
        ax.xaxis.set_major_locator(MultipleLocator(5))
    elif model_datasets == 'resnet50_image64':
        ax.set_ylim(-5, 65)
        ax.set_xlabel("Time (hours)", font)
        ax.set_ylabel("Top-1 Accuracy (%)", font)
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.yaxis.set_major_locator(MultipleLocator(20))
    elif model_datasets == 'bert_bookcorpus':
        ax.set_xlim(-0.3, 6)
        ax.set_xlabel("Time (hours)", font)
        ax.set_ylabel("Loss Value", font)
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.tick_params(axis='x', labelcolor='#000000',
                   direction='in', labelsize=tick_label_size, width=tick_width)
    ax.tick_params(axis='y', labelcolor='#000000',
                   direction='in', labelsize=tick_label_size, width=tick_width)

    for index, data in enumerate(accdatas):
        data = rename(data)
        line, = ax.plot(data['time'], data['top1'],
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

    ax_handles, ax_labels = ax.get_legend_handles_labels()
    plt.legend(ax_handles, ax_labels, fontsize=legend_size)
    plt.savefig(f'{model_datasets}.pdf',
                format='pdf', bbox_inches='tight', pad_inches=0)


def main():
    model_datasets = 'vgg16_cifar10'
    filenames = os.listdir(f'data/{model_datasets}/')
    filenames.sort()
    acc_datas = []
    loss_datas = []
    for filename in filenames:
        acc_data, loss_data = read_txt(f'data/{model_datasets}/'+filename)
        acc_datas.append(acc_data)
        loss_datas.append(loss_data)
    draw_plot(acc_datas, model_datasets)


if __name__ == '__main__':
    main()
