import matplotlib.pyplot as plt
import numpy as np
import os

CHOOSE_DATASET = 'bookcorpus'  # bookcorpus or cifar10 or imagenet64

font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 24,
        }
legend_size = 15
tick_label_size = 20
tick_width = 2
double_tick_size = 20
plot_linewidth = 2.5
x_major_locator = 100
y_major_locator = 3000
spine_width = 2
marker_size = 8
colot_set = ['#ED1C24', '#1E86BB', '#8AB93F']
dashes_set = [[0], [5, 1, 1, 1], [4]]


# data seqs/s or image/s
gpipe = {'bookcorpus': [76.085], 'cifar10': [
    105.4, 32.89], 'imagenet64': [98.77, 32.27]}
pipedream = {'bookcorpus': [150.65], 'cifar10': [
    153.21, 147.12], 'imagenet64': [127.65, 104.27]}
dynpipe = {'bookcorpus': [180.85], 'cifar10': [
    194.85, 179.32], 'imagenet64': [194.33, 177.73]}


def draw_bar():
    if CHOOSE_DATASET == 'bookcorpus':
        bar_width = 0.25
        bar_spacing = 0.1
        x = ['Bert']
        plt.gca().set_ylabel("Throughput (seqs/s)", font)
    elif CHOOSE_DATASET == 'cifar10' or CHOOSE_DATASET == 'imagenet64':
        bar_width = 0.25
        bar_spacing = 0.02
        x = ['ResNet50', 'VGG16']
        plt.gca().set_ylabel("Throughput (image/s)", font)

    index = np.arange(len(x))+1

    bars_value1 = plt.bar(index, gpipe[CHOOSE_DATASET], bar_width,
                          label='GPipe', color="#E6E6E6", edgecolor='black', linewidth=1)
    bars_value2 = plt.bar(index + bar_width+bar_spacing, pipedream[CHOOSE_DATASET], bar_width,
                          label='PipeDream', color="#4472C4", edgecolor='black', linewidth=1)
    bars_value3 = plt.bar(index + bar_width*2+bar_spacing*2, dynpipe[CHOOSE_DATASET], bar_width,
                          label='DynPipe', color="#FF9300", edgecolor='black', linewidth=1)

    for bar in bars_value1:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval,
                 round(yval, 2), ha='center', va='bottom', size=15)

    for bar in bars_value2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval,
                 round(yval, 2), ha='center', va='bottom', size=15)

    for bar in bars_value3:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval,
                 round(yval, 2), ha='center', va='bottom', size=15)

    plt.gca().spines['top'].set_linewidth(spine_width)
    plt.gca().spines['bottom'].set_linewidth(spine_width)
    plt.gca().spines['left'].set_linewidth(spine_width)
    plt.gca().spines['right'].set_linewidth(spine_width)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend(fontsize=legend_size, ncol=3,
               columnspacing=0.7, loc='upper center', bbox_to_anchor=(0.5, 1.15))
    plt.xticks(index + (bar_width*2+bar_spacing)/2, x)
    plt.tick_params(axis='x', labelcolor='#000000', color='#FFFFFF',
                    labelsize=20, width=0, length=0, which='major')
    plt.tick_params(axis='y', labelcolor='#000000',
                    direction='in', labelsize=tick_label_size, width=tick_width, which='major')

    plt.savefig(f'{os.path.basename(os.getcwd())}_{CHOOSE_DATASET}.pdf',
                format='pdf', bbox_inches='tight', pad_inches=0)


def main():
    draw_bar()


if __name__ == '__main__':
    main()
