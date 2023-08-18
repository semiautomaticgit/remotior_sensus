from unittest import TestCase

from remotior_sensus.util import plot_tools


class TestPlotTools(TestCase):

    def test_plot(self):
        pass
        """
        ax = plot_tools.prepare_plot()
        plots, plot_names, x_ticks, y_ticks, v_lines = (
            plot_tools.add_lines_to_plot(
                name_list=['test1', 'test2'],
                wavelength_list=[[0.4, 0.5, 0.6], [0.3, 0.5, 0.7]],
                value_list=[[0.7, 0.4, 0.9], [1.1, 1.3, 1.2]],
                color_list=['red', 'blue'])
        )
        plot_tools.create_plot(
            ax=ax, plots=plots, plot_names=plot_names, x_ticks=x_ticks,
            y_ticks=y_ticks, v_lines=v_lines
        )
        """
