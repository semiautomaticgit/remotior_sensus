# Remotior Sensus , software to process remote sensing and GIS data.
# Copyright (C) 2022-2023 Luca Congedo.
# Author: Luca Congedo
# Email: ing.congedoluca@gmail.com
#
# This file is part of Remotior Sensus.
# Remotior Sensus is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
# Remotior Sensus is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with Remotior Sensus. If not, see <https://www.gnu.org/licenses/>.

"""
Tools to manage plots
"""


try:
    import matplotlib.pyplot as plt
    import matplotlib.pyplot as mpl_plot
except Exception as error:
    str(error)
    print('plot tools: matplotlib error')


# prepare plot
def prepare_plot(x_label=None, y_label=None):
    if x_label is None:
        x_label = 'Wavelength'
    if y_label is None:
        y_label = 'Values'
    figure, ax = plt.subplots()
    # Set empty ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('auto')
    ax.grid('on')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return ax


# prepare plot
def prepare_scatter_plot(x_label=None, y_label=None):
    if x_label is None:
        x_label = 'Band X'
    if y_label is None:
        y_label = 'Band Y'
    figure, ax = plt.subplots()
    # Set empty ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('auto')
    ax.grid('on')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return ax


# add list of values to plot
def add_lines_to_plot(
        name_list, wavelength_list, value_list, color_list,
        legend_max_chars=15
        ):
    plots = []
    plot_names = []
    v_lines = []
    wavelength_min = 1000000
    wavelength_max = 0
    value_min = 10000000
    value_max = 0
    for _id in range(len(name_list)):
        plot, = plt.plot(
            wavelength_list[_id], value_list[_id], color_list[_id]
        )
        v_lines.extend(wavelength_list[_id])
        wavelength_min = min(min(wavelength_list[_id]), wavelength_min)
        wavelength_max = max(max(wavelength_list[_id]), wavelength_max)
        value_min = min(min(value_list[_id]), value_min)
        value_max = max(max(value_list[_id]), value_max)
        plots.append(plot)
        plot_names.append(name_list[_id][:legend_max_chars])
    x_min = wavelength_min
    x_ticks = [x_min]
    for x in range(10):
        x_min += (wavelength_max - wavelength_min) / 10
        x_ticks.append(x_min)
    y_min = value_min
    y_ticks = [y_min]
    for y in range(10):
        y_min += (value_max - value_min) / 10
        y_ticks.append(y_min)
    return plots, plot_names, x_ticks, y_ticks, set(v_lines)


# create plot
def create_plot(
        ax, plots, plot_names, x_ticks=None, y_ticks=None, v_lines=None
):
    if x_ticks is None:
        x_ticks = [0, 1]
    if y_ticks is None:
        y_ticks = [0, 1]
    if v_lines is not None:
        for x in v_lines:
            ax.axvline(x, color='black', linestyle='dashed')
    ax.legend(
        plots, plot_names, bbox_to_anchor=(0.0, 0.0, 1.1, 1.0), loc=1,
        borderaxespad=0.
        ).set_draggable(True)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    plt.show()


# create plot
def create_scatter_plot(
        ax, plots, plot_names, x_ticks=None, y_ticks=None
):
    if x_ticks is None:
        x_ticks = [0, 1]
    if y_ticks is None:
        y_ticks = [0, 1]
    ax.legend(
        plots, plot_names, bbox_to_anchor=(0.0, 0.0, 1.1, 1.0), loc=1,
        borderaxespad=0.
        ).set_draggable(True)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    plt.show()


# add values to plot
def add_values_to_scatter_plot(histogram, ax):
    pal = mpl_plot.get_cmap('rainbow')
    pal.set_under('w', 0.0)
    plot = ax.imshow(
        histogram[0].T, origin='lower', interpolation='none',
        extent=[histogram[1][0], histogram[1][-1], histogram[2][0],
                histogram[2][-1]], cmap=pal, vmin=0.001
    )
    return plot
