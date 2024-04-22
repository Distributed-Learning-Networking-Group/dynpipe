import matplotlib.pyplot as plt
import numpy as np
import os

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

# data light-weight/heavy interference   second
pipedream = [0.20211489151873782, 0.343]
dyepipe_re = [0.17668392504931094, 0.2446]
dyepipe = [0.17322338014241354, 0.207]


def draw_bar():
    bar_width = 0.25
    bar_spacing = 0.05
    x = ['light-weight\ninterference', 'heavy\ninterference']

    index = np.arange(len(x))+1
    bars_value1 = plt.bar(index, pipedream, bar_width,
                          label='PipeDream', color="#E6E6E6", edgecolor='black', linewidth=1)
    bars_value2 = plt.bar(index + bar_width+bar_spacing, dyepipe_re, bar_width,
                          label='DynPipe-Re', color="#4472C4", edgecolor='black', linewidth=1)
    bars_value3 = plt.bar(index + bar_width*2+bar_spacing*2, dyepipe, bar_width,
                          label='DynPipe', color="#FF9300", edgecolor='black', linewidth=1)

    for bar in bars_value1:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval,
                 round(yval, 3), ha='center', va='bottom', size=15)

    for bar in bars_value2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval,
                 round(yval, 3), ha='center', va='bottom', size=15)

    for bar in bars_value3:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval,
                 round(yval, 3), ha='center', va='bottom', size=15)

    plt.gca().spines['top'].set_linewidth(spine_width)
    plt.gca().spines['bottom'].set_linewidth(spine_width)
    plt.gca().spines['left'].set_linewidth(spine_width)
    plt.gca().spines['right'].set_linewidth(spine_width)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().set_ylabel("Iteration Time (s)", font)
    plt.legend(fontsize=legend_size, ncol=3,
               columnspacing=0.7, loc='upper center', bbox_to_anchor=(0.5, 1.15))

    plt.xticks(index + (bar_width*2+bar_spacing)/2, x)
    plt.tick_params(axis='x', labelcolor='#000000', color='#FFFFFF',
                    labelsize=20, width=0, length=0, which='major')
    plt.tick_params(axis='y', labelcolor='#000000',
                    direction='in', labelsize=tick_label_size, width=tick_width, which='major')
    plt.savefig(f'{os.path.basename(os.getcwd())}.pdf',
                format='pdf', bbox_inches='tight', pad_inches=0)


def main():
    draw_bar()


if __name__ == '__main__':
    main()
