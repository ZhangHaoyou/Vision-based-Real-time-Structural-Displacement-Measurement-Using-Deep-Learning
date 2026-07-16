import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_template(*args, 
                figsize=(8,5), save_path='plot.svg',
                legend_labels=None,
                xlabel='X-Axis', ylabel='Y-Axis',
                colors=None, linestyles=None, 
                markers=None, markeverys=None,
                xlim=None, major_x_interval=None,
                ylim=None, major_y_interval=None,
                title='', legend_horizontal=False,
                ):
    """
    Line plot template using matplotlib
    Args:
    *args: Variable-length argument list for input data, format: [[x1, y1], [x2, y2], ...]
    figsize: dimension of figure (default: (8, 5) cm)
    save_path: File path to save the plot (default: 'plot.svg')
    ylabel: Label for the x-axis (default: 'X-Axis')
    ylabel: Label for the y-axis (default: 'Y-Axis')
    title: Title of the plot (default: '')
    legend_labels: List of labels for the legend (default: None)
    xlim: Tuple for x-axis limits (default: None)
    ylim: Tuple for y-axis limits (default: None)
    major_x_interval: Interval between major x-ticks (default: None)
    major_y_interval: Interval between major y-ticks (default: None)
    minor_x_ticks: Number of minor ticks between major x-ticks (default: 1)
    minor_y_ticks: Number of minor ticks between major y-ticks (default: 1)
    """
    
    plt.close('all')
    plt.rcParams['font.family'] = 'Times New Roman'
    
    
    # Set figure size: 8cm width, 5cm height, and DPI 200
    width, height = figsize
    fig, ax = plt.subplots(figsize=(width/2.54, height/2.54), dpi=200)  # Convert from cm to inches (1cm = 1/2.54 inches)
    
    # Add title if provided
    if title:
        ax.set_title(title, fontsize=11)
    
    # Plot each line
    for i, data in enumerate(args):
        x, y = data
        label = legend_labels[i] if legend_labels is not None else None
        color = colors[i] if colors is not None else None
        linestyle= linestyles[i] if linestyles is not None else None
        marker = markers[i] if markers is not None else None
        markevery = markeverys[i] if markeverys is not None else None
        ax.plot(x, y, label=label, color=color, linestyle=linestyle, marker=marker, markevery=markevery,
                markersize=4, markeredgewidth=1, markeredgecolor=color, markerfacecolor='white',
                zorder=3) # fillstyle='none'
    
    # Set x_y label and font size for labels
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    
    # Set custom axis limits
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # Set major and minor ticks at custom intervals
    if major_x_interval is not None:
        xticks = [xlim[0]+i*major_x_interval for i in range(int((xlim[1]-xlim[0])/major_x_interval)+1)]
        ax.xaxis.set_ticks(xticks)
        ax.set_xticklabels([str(i) for i in xticks], fontsize=8)
        # Customize minor ticks: set exactly one minor tick between each major tick
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    if major_y_interval is not None:
        yticks = [ylim[0]+i*major_y_interval for i in range(int((ylim[1]-ylim[0])/major_y_interval)+1)]
        ax.yaxis.set_ticks(yticks)
        ax.set_yticklabels([str(i) for i in yticks], fontsize=8)
        # Customize minor ticks: set exactly one minor tick between each major tick
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    
    # Customize tick direction: inward ticks for both major and minor
    ax.tick_params(axis='both', which='major', direction='in', length=4, labelsize=10)
    ax.tick_params(axis='both', which='minor', direction='in', length=3, labelsize=0)
    
    # Add legend if legend labels are provided
    if legend_labels is not None:
        ncol = len(legend_labels) if legend_horizontal else 1
        ax.legend(fontsize=10, frameon=False, ncol=ncol,
                handletextpad=0.3, handlelength=1.2,
                markerscale=0.8, columnspacing=0.5)  # Legend font size 10, no border
    
    # Save plot
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

if __name__ == '__main__':
    
    # Example usage
    # First line's x and y coordinates
    x1 = [0, 1, 2, 3, 4]
    y1 = [0, 1, 4, 9, 16]

    # Second line's x and y coordinates
    x2 = [0, 1, 2, 3, 4]
    y2 = [0, 2, 3, 5, 7]

    plot_template([x1, y1], [x2, y2],
                    xlabel='Time (s)', ylabel='Displacement (mm)', legend_labels=['Square', 'Linear'],
                    colors=['black', 'red'], markers=[None, 's'],
                    xlim=(0,4), ylim=(0,16), major_x_interval=1, major_y_interval=4)
