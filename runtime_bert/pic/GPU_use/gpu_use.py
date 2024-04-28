import matplotlib.pyplot as plt
import os
from matplotlib.pyplot import MultipleLocator

CHOOSE_MODEL = 'vgg16'  # vgg16 ro resnet50

font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 24,
        }
legend_size = 15
tick_label_size = 20
tick_width = 2
double_tick_size = 20
plot_linewidth = 2
x_major_locator = 100
y_major_locator = 20
spine_width = 2
marker_size = 8
colot_set = ['#ED1C24', '#1E86BB', '#8AB93F']
dashes_set = [[0], [5, 1, 1, 1], [4]]
gpu_num = ['2GPUs', '4GPUs', '6GPUs', '8GPUs']

# data
resnet50_cifar_10_dynpipe = [
    59.00657894736842, 52.06333333333333, 35.57683741648107, 24.757929883138564]
resnet50_cifar_10_pipedream = [
    46.473684210526315, 40.15, 15.62, 9.608623548922056]
resnet50_cifar_10_gpipe = [
    42.63333333333333, 14.300653594771243, 10.17353579175705, 7.710355987055016]

vgg16_cifar_10_dynpipe = [72.64238410596026,
                          73.01333333333334, 45.99107142857143, 39.259504132231406]
vgg16_cifar_10_pipedream = [57.51315789473684,
                            41.61, 36.801339285714285, 13.6578073089701]
vgg16_cifar_10_gpipe = [31.8476821192053,
                        10.01628664495114, 7.406926406926407, 5.710610932475884]


def draw_plot():
    fig, ax = plt.subplots()
    ax.tick_params(axis='x', labelcolor='#000000',
                   direction='in', labelsize=tick_label_size, width=tick_width)
    ax.tick_params(axis='y', labelcolor='#000000',
                   direction='in', labelsize=tick_label_size, width=tick_width)
    ax.yaxis.set_major_locator(MultipleLocator(y_major_locator))
    if CHOOSE_MODEL == 'vgg16':
        line, = ax.plot(gpu_num, vgg16_cifar_10_dynpipe,
                        label='DynPipe', color=colot_set[0], linewidth=plot_linewidth, marker="o", ms=8, markerfacecolor='none', markeredgewidth=2)
        line, = ax.plot(gpu_num, vgg16_cifar_10_pipedream,
                        label='PipeDream', color=colot_set[1], linewidth=plot_linewidth, marker="^", ms=8, markerfacecolor='none', markeredgewidth=2)
        line, = ax.plot(gpu_num, vgg16_cifar_10_gpipe,
                        label='GPipe', color=colot_set[2], linewidth=plot_linewidth, marker="x", ms=8, markerfacecolor='none', markeredgewidth=2)

    elif CHOOSE_MODEL == 'resnet50':
        ax.set_ylim(0.1, 70)
        line, = ax.plot(gpu_num, resnet50_cifar_10_dynpipe,
                        label='DynPipe', color=colot_set[0], linewidth=plot_linewidth, marker="o", ms=8, markerfacecolor='none', markeredgewidth=2)
        line, = ax.plot(gpu_num, resnet50_cifar_10_pipedream,
                        label='PipeDream', color=colot_set[1], linewidth=plot_linewidth, marker="^", ms=8, markerfacecolor='none', markeredgewidth=2)
        line, = ax.plot(gpu_num, resnet50_cifar_10_gpipe,
                        label='GPipe', color=colot_set[2], linewidth=plot_linewidth, marker="x", ms=8, markerfacecolor='none', markeredgewidth=2)

    ax.legend()
    ax.spines['top'].set_linewidth(spine_width)
    ax.spines['bottom'].set_linewidth(spine_width)
    ax.spines['left'].set_linewidth(spine_width)
    ax.spines['right'].set_linewidth(spine_width)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_ylabel("GPU Utilization (%)", font)
    ax_handles, ax_labels = ax.get_legend_handles_labels()
    plt.legend(ax_handles, ax_labels, fontsize=legend_size)
    plt.savefig(f'{os.path.basename(os.getcwd())}_{CHOOSE_MODEL}.pdf',
                format='pdf', bbox_inches='tight', pad_inches=0)

    resnet_dynpipe_to_pipedream = []
    resnet_dynpipe_to_gpipe = []
    vgg_dynpipe_to_pipedream = []
    vgg_dynpipe_to_gpipe = []
    for i in range(4):
        resnet_dynpipe_to_pipedream.append(
            resnet50_cifar_10_dynpipe[i] - resnet50_cifar_10_pipedream[i])
        resnet_dynpipe_to_gpipe.append(
            resnet50_cifar_10_dynpipe[i] - resnet50_cifar_10_gpipe[i])
        vgg_dynpipe_to_pipedream.append(
            vgg16_cifar_10_dynpipe[i] - vgg16_cifar_10_pipedream[i])
        vgg_dynpipe_to_gpipe.append(
            vgg16_cifar_10_dynpipe[i]-vgg16_cifar_10_gpipe[i])
    print(
        f'develop with ResNet50 on PipeDream: min: {min(resnet_dynpipe_to_pipedream)} max {max(resnet_dynpipe_to_pipedream)}')
    print(
        f'develop with ResNet50 on GPipe: min: {min(resnet_dynpipe_to_gpipe)} max {max(resnet_dynpipe_to_gpipe)}')
    print(
        f'develop with VGG16 on PipeDream: min: {min(vgg_dynpipe_to_pipedream)} max {max(vgg_dynpipe_to_pipedream)}')
    print(
        f'develop with VGG16 on GPipe: min: {min(vgg_dynpipe_to_gpipe)} max {max(vgg_dynpipe_to_gpipe)}')


def main():
    draw_plot()


if __name__ == '__main__':
    main()
