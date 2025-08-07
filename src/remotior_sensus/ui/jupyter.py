# Remotior Sensus , software to process remote sensing and GIS data.
# Copyright (C) 2022-2025 Luca Congedo.
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
Tools to manage raster writing and reading
"""

import os
import io
import base64
import json
from copy import deepcopy

from remotior_sensus.core import configurations as cfg

try:
    import ipywidgets as widgets
    from IPython.display import display
    ipywidgets_version = widgets.__version__
except Exception as error:
    try:
        cfg.logger.log.error(str(error))
    except Exception as error:
        str(error)

try:
    from ipyleaflet import (
        Map, Rectangle, Polygon, LayersControl, ImageOverlay, LayerGroup,
        GeoJSON, Marker, WidgetControl
    )
except Exception as error:
    try:
        cfg.logger.log.error(str(error))
    except Exception as error:
        str(error)

try:
    from PIL import Image
except Exception as error:
    try:
        cfg.logger.log.error(str(error))
    except Exception as error:
        str(error)

""" Progress messages """
msg_label_main = msg_label = messages_row = progress_widget = None
tot_remaining = ''

""" Interfaces """
browser_rows = plot_rows = plot_widget = browser_selector_val = None
bandset_rows = download_products_rows = bandcalc_rows = None
clip_raster_rows = image_conversion_rows = masking_rows = None
mosaic_bandsets_rows = reproject_rows = split_raster_rows = None
stack_raster_rows = vector_to_raster_rows = None
classification_accordion = algorithm_selection = None
combination_rows = dilation_rows = erosion_rows = sieve_rows = None
neighbor_rows = pca_rows = accuracy_rows = None
classification_report_rows = raster_to_vector_rows = None
cross_classification_rows = reclassification_rows = None
import_tool_rows = classification_rows = map_label = None
training_dock_rows = working_toolbar_rows = None
plot_button = main_button = map_button = None
preview_point = save_classifier = None
caller_function = directory_list = training_path = input_bandset = None
downloaded = classifier_preview = load_classifier = None
selected_bandsets = []

""" Download products """
aoi_draw = False
layers_control = rectangle = download_table = None
date_from = date_to = advanced = product_table = preview = None
ul_x = ul_y = lr_x = lr_y = None
selected_values = []
points = []
selected_bands = [
    '01', '02', '03', '04', '05', '06', '07', '08', '8A', '09', '10',
    '11', '12'
]

""" Training """
rgb_layer_group = old_layer = last_region_path = temporary_roi = None
classification_preview_group = old_classification_preview = None
training_map = training_data = None
region_growing_max = region_growing_dist = region_growing_min = None
classification_preview_pointer = False
roi_grow = False
roi_manual = False
close_polygon = False
selection_training = []
polygon_coordinates = []
r_point_coordinates = []

""" File browser """
browser_dir = os.getcwd()
file_list = selected_file_paths = browser = new_file_text = None
select_button = open_dir_button = parent_dir_button = cancel_button = None
new_file_button = new_dir_button = new_dir_text = None
selected_files_dirs = []

""" button styles """
selected_interface_style = dict(
    font_weight='bold', font_variant="small-caps", text_color='white',
    text_decoration='underline', button_color='#2f2f2f'
)
unselected_interface_style = dict(
    font_weight='bold', text_color='#000000', button_color='#dddddd'
)
label_style = {
    'text_color': 'white', 'font_weight': 'bold', 'background': '#5a5a5a'
}


class JupyterInterface(object):

    def __init__(self, remotior_sensus_session):
        self.rs = remotior_sensus_session
        self.rs.set(progress_callback=update_progress)
        if cfg.default_catalog is None:
            cfg.default_catalog = self.rs.bandset_catalog()

        global browser_dir, selected_file_paths, browser, new_file_text
        global select_button, open_dir_button, parent_dir_button, cancel_button
        global new_file_button, new_dir_button, new_dir_text
        global selected_files_dirs, ul_x, ul_y, lr_x, lr_y
        global msg_label_main, msg_label, progress_widget, messages_row
        global browser_rows, training_map, map_label, classification_accordion
        global training_dock_rows, working_toolbar_rows, algorithm_selection
        global region_growing_max, region_growing_dist, region_growing_min
        global browser_selector_val, label_style, plot_rows, plot_widget
        global plot_button, main_button, map_button
        global bandset_rows, download_products_rows, bandcalc_rows
        global import_tool_rows, classification_rows
        global clip_raster_rows, image_conversion_rows, masking_rows
        global mosaic_bandsets_rows, reproject_rows, split_raster_rows
        global stack_raster_rows, vector_to_raster_rows
        global combination_rows, dilation_rows, erosion_rows, sieve_rows
        global neighbor_rows, pca_rows, accuracy_rows
        global classification_report_rows, raster_to_vector_rows
        global cross_classification_rows, reclassification_rows
        global preview, download_table
        global date_from, date_to, advanced, product_table
        global selected_values, points, selected_bands
        global rgb_layer_group, old_layer, last_region_path, temporary_roi
        global training_data, classification_preview_pointer, roi_grow
        global roi_manual, close_polygon, selection_training
        global polygon_coordinates, r_point_coordinates
        global preview_point, save_classifier

        """ Messages """
        msg_label_main = widgets.HTML(placeholder='')
        msg_label = widgets.HTML(placeholder='')
        progress_widget = widgets.IntProgress(
            value=0, min=0, max=100, style={'bar_color': 'green'},
            orientation='horizontal'
        )
        messages_row = widgets.HBox(
            [progress_widget, msg_label_main, msg_label]
        )

        def error_message(message):
            msg_label_main.value = (
                '<div style="color: red; font-weight: bold;">%s</div>'
                % message
            )

        def reset_message():
            msg_label_main.value = ''
            msg_label.value = ''
            progress_widget.value = 0

        self._reset_message = reset_message

        """ File browser """
        dir_icon = f'\U0001F4C1'
        file_icon = f'\U0001F4C4'

        # bridge function
        def run_function(selected, function):
            return function(selected)

        # interface selector linked to stack (1=main interface, 2=map, 3=plot)
        browser_selector_val = widgets.IntText(value=1)

        def disable_browser_buttons(state=True):
            global select_button, open_dir_button, parent_dir_button
            global cancel_button, new_file_button, new_file_text
            global new_dir_button, new_dir_text
            select_button.disabled = state
            open_dir_button.disabled = state
            parent_dir_button.disabled = state
            cancel_button.disabled = state
            new_file_button.disabled = state
            new_dir_button.disabled = state
            new_file_text.disabled = state
            new_dir_text.disabled = state

        # select browser
        def activate_browser(_):
            global browser_selector_val, msg_label_main
            msg_label_main.value = 'Open file browser'
            browser_selector_val.value = 0
            disable_browser_buttons(False)
            browse_path()

        # noinspection PyUnresolvedReferences
        def selected_dir_change(_):
            global browser_dir, selected_file_paths
            if os.path.isdir(selected_file_paths.value):
                browser_dir = selected_file_paths.value.replace('//', '/')
                browse_path()

        # get selected files
        def selection_files_change(change):
            global selected_files_dirs
            selected = change['new']
            selected_files_dirs = []
            for s in selected:
                s_path = os.path.join(
                    browser_dir, s.replace(dir_icon, '').replace(file_icon, '')
                )
                if os.path.isdir(s_path):
                    selected_files_dirs = [s_path]
                    break
                selected_files_dirs.append(s_path)

        selected_file_paths = widgets.Text(
            placeholder='Comma separated paths',
            layout=widgets.Layout(width='500px')
        )
        selected_file_paths.observe(selected_dir_change)
        # browser multiple select
        browser = widgets.SelectMultiple(
            options=[], rows=10, layout=widgets.Layout(width='500px')
        )
        browser.observe(selection_files_change, names='value')
        browser_selector = widgets.VBox([selected_file_paths, browser])

        def browse_path():
            global file_list, directory_list
            global selected_file_paths, browser, new_file_text, new_dir_text
            selected_file_paths.value = browser_dir
            new_file_text.value = ''
            new_dir_text.value = ''
            paths = os.listdir(browser_dir)
            file_list = [file_icon + f for f in paths if
                         os.path.isfile(os.path.join(browser_dir, f))]
            directory_list = [dir_icon + d for d in paths if
                              os.path.isdir(os.path.join(browser_dir, d))]
            file_options = sorted(directory_list) + sorted(file_list)
            browser.options = file_options

        def activate_cancel_dir(_):
            global browser_selector_val
            disable_browser_buttons(True)
            browser_selector_val.value = 1

        def activate_select(_):
            global caller_function, selected_files_dirs
            disable_browser_buttons(True)
            run_function(selected_files_dirs, caller_function)

        # noinspection PyUnresolvedReferences
        def activate_new_file(_):
            global caller_function, new_file_text
            disable_browser_buttons(True)
            run_function(new_file_text.value, caller_function)

        # noinspection PyUnresolvedReferences
        def activate_new_dir(_):
            global new_dir_text
            self.rs.files_directories.create_directory(
                '%s/%s' % (browser_dir, new_dir_text.value)
            )
            browse_path()

        def activate_open_dir(_):
            global browser_dir, selected_files_dirs
            if len(selected_files_dirs) > 0:
                if os.path.isdir(selected_files_dirs[0]):
                    browser_dir = selected_files_dirs[0]
                    browse_path()

        def activate_parent_dir(_):
            global browser_dir
            new_dir = self.rs.files_directories.parent_directory(browser_dir)
            if new_dir:
                if len(new_dir) > 0:
                    browser_dir = new_dir
            browse_path()

        select_button = widgets.Button(
            description='Select', button_style='success', icon='check',
            layout=widgets.Layout(width='150px'), disabled=True
            )
        select_button.on_click(activate_select)
        open_dir_button = widgets.Button(
            description='Open dir', button_style='primary', icon='folder-open',
            layout=widgets.Layout(width='150px'), disabled=True
            )
        open_dir_button.on_click(activate_open_dir)
        parent_dir_button = widgets.Button(
            description='Parent dir', button_style='primary', icon='level-up',
            layout=widgets.Layout(width='150px'), disabled=True
            )
        parent_dir_button.on_click(activate_parent_dir)
        cancel_button = widgets.Button(
            description='Cancel', button_style='warning',
            layout=widgets.Layout(width='150px'), disabled=True
            )
        cancel_button.on_click(activate_cancel_dir)
        new_file_button = widgets.Button(
            description='New file', button_style='success', icon='file',
            layout=widgets.Layout(width='150px'), disabled=True
            )
        new_file_button.on_click(activate_new_file)
        new_file_text = widgets.Text(placeholder='New file name')
        new_dir_button = widgets.Button(
            description='Create directory', button_style='success',
            icon='folder', layout=widgets.Layout(width='150px'), disabled=True
            )
        new_dir_button.on_click(activate_new_dir)
        new_dir_text = widgets.Text(placeholder='New directory name')
        browser_buttons = widgets.VBox(
            [open_dir_button, parent_dir_button, cancel_button,
             widgets.HBox([new_file_button, new_file_text]),
             widgets.HBox([new_dir_button, new_dir_text]), select_button]
            )
        """
        browser_label = widgets.Label(
            value='File browser', layout=widgets.Layout(width='97%'),
            style=label_style
        )
        """
        browser_label = widgets.HTML(
            value=f"<div style='color:white; weight:bold; "
                  f"background: #5a5a5a'><b>File browser</b></div>"
        )
        browser_rows = widgets.VBox([
            browser_label,
            widgets.HBox([browser_selector, browser_buttons])
        ])

        """ Plot interface """

        plot_widget = widgets.Output()
        """
        plot_label = widgets.Label(
            value='Signature plot', layout=widgets.Layout(width='97%'),
            style=label_style
        )
        """
        plot_label = widgets.HTML(
            value=f"<div style='color:white; weight:bold; "
                  f"background: #5a5a5a'><b>Signature plot</b></div>"
        )
        plot_rows = widgets.VBox(
            [plot_label,
             widgets.HBox([plot_widget], layout=widgets.Layout(width='98%'))]
        )

        """ Band set """

        def active_bandset_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                cfg.default_catalog.current_bandset = change['new']
                refresh_bandsets()

        def delete_bands_click(_):
            global selected_bands
            for r in reversed(selected_bands):
                cfg.default_catalog.remove_band_in_bandset(
                    band_number=int(r) + 1
                )
            refresh_bandsets()

        def move_up_bands_click(_):
            global selected_bands
            for r in selected_bands:
                cfg.default_catalog.move_band_in_bandset(
                    bandset_number=cfg.default_catalog.current_bandset,
                    band_number_input=int(r) + 1, band_number_output=int(r)
                )
            refresh_bandsets()

        def move_down_bands_click(_):
            global selected_bands
            for r in reversed(selected_bands):
                cfg.default_catalog.move_band_in_bandset(
                    bandset_number=cfg.default_catalog.current_bandset,
                    band_number_input=r + 1, band_number_output=r + 2
                )
            refresh_bandsets()

        # get selected bands
        def selection_bands_change(change):
            global selected_bands
            selected_bands = change['new']

        # get selected bandsets
        def selection_bandsets_change(change):
            global selected_bandsets
            selected_bandsets = change['new']

        def add_bands_click(_):
            global caller_function
            caller_function = bandset_paths
            activate_browser(None)

        def wavelengths_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                satellite_name = cfg.sat_band_list[wavelengths.index]
                cfg.default_catalog.set_satellite_wavelength(
                    satellite_name=satellite_name,
                    bandset_number=cfg.default_catalog.current_bandset
                )
                # create table
                refresh_bandsets()

        def band_rows(bandset):
            sep = '│ '
            bandset_options = []
            max_widths = {}
            bands = bandset.bands
            bands.sort(order='band_number')
            for attribute in ['name', 'wavelength', 'path']:
                max_widths[attribute] = 1
                for band in bands:
                    max_widths[attribute] = max(
                        max_widths[attribute], len(str(band[attribute]))
                    )
            for band in bandset.bands:
                band_text = str(band['band_number']).ljust(4) + sep
                for attribute in ['name', 'wavelength', 'path']:
                    band_text += '%s %s' % (
                        str(band[attribute]).ljust(max_widths[attribute]), sep
                    )
                bandset_options.append(str(band_text))
            return bandset_options

        def refresh_bandsets():
            nonlocal active_bandset, bandsets_table, bandset_table
            bandsets_table.options = []
            bandset_table.options = []
            active_bandset.max = cfg.default_catalog.get_bandset_count()
            bandsets_options = []
            n = 0
            for bandset_number in range(
                    1, cfg.default_catalog.get_bandset_count() + 1
            ):
                n += 1
                bandset_x = cfg.default_catalog.get_bandset_by_number(
                    bandset_number
                )
                if bandset_x.bands is not None:
                    names = str(bandset_x.get_band_attributes('name')).replace(
                        "'", ''
                    ).replace('[', '').replace(']', '')
                    if names == 'None':
                        names = ''
                    bandsets_options.append('%s | %s' % (n, names))
                else:
                    bandsets_options.append('%s | ' % n)
            bandsets_table.options = bandsets_options
            bandset_x = cfg.default_catalog.get_bandset_by_number(
                cfg.default_catalog.current_bandset
            )
            bandset_options = band_rows(bandset_x)
            if len(bandset_options) == 0:
                bandset_options = ' '
            bandset_table.options = bandset_options

        def delete_bandset_click(_):
            global selected_bandsets
            for r in reversed(selected_bandsets):
                cfg.default_catalog.remove_bandset(r + 1)
            refresh_bandsets()

        # noinspection PyUnresolvedReferences
        def bandset_paths(input_files):
            global selected_file_paths
            selected_file_paths.value = str(input_files).replace(
                "'", ''
            ).replace('[', '').replace(']', '')
            files = selected_file_paths.value.split(',')
            for file in files:
                if len(file) > 0:
                    try:
                        cfg.default_catalog.add_band_to_bandset(
                            path=file.strip(), raster_band=1,
                            bandset_number=cfg.default_catalog.current_bandset
                        )
                        error_message('')
                    except Exception as err:
                        str(err)
                        error_message('Select a file.')
            refresh_bandsets()
            browser_selector_val.value = 1

        def add_bandset_click(_):
            cfg.default_catalog.create_bandset(
                insert=True,
                bandset_number=cfg.default_catalog.get_bandset_count() + 1,
                bandset_name='BandSet %s'
                             % (cfg.default_catalog.get_bandset_count() + 1)
            )
            refresh_bandsets()

        """
        bandsets_label = widgets.Label(
            value='Band set table', layout=widgets.Layout(width='97%'),
            style=label_style
        )
        """
        bandsets_label = widgets.HTML(
            value=f"<div style='color:white; weight:bold; "
                  f"background: #5a5a5a'><b>Band set table</b></div>"
        )
        bandsets_table = widgets.SelectMultiple(
            options=[], value=[], rows=10, layout=widgets.Layout(width='97%')
        )
        bandsets_table.observe(selection_bandsets_change, names='index')
        """
        bandset_label = widgets.Label(
            value=(
                'Band set definition (Band number | Name | Wavelength | Path).'
                ' Active band set:'
            ), layout=widgets.Layout(width='90%'),
            style=label_style
        )
        """
        bandset_label = widgets.HTML(
            value=f"<div style='color:white; weight:bold; "
                  f"background: #5a5a5a'><b>Band set definition (Band number "
                  f"| Name | Wavelength | Path). Active band set:</b></div>"
        )
        bandset_table = widgets.SelectMultiple(
            options=[], value=[], rows=10, layout=widgets.Layout(width='99%')
        )
        bandset_table.observe(selection_bands_change, names='index')
        add_bands_button = widgets.Button(
            tooltip='Add bands', button_style='success', icon='plus',
            layout=widgets.Layout(width='40px')
        )
        add_bands_button.on_click(add_bands_click)
        active_bandset = widgets.BoundedIntText(
            value=1, min=1, max=cfg.default_catalog.get_bandset_count(),
            step=1, layout=widgets.Layout(width='10%')
        )
        active_bandset.observe(active_bandset_change)
        move_up_band_button = widgets.Button(
            tooltip='Move up band', button_style='info', icon='arrow-up',
            layout=widgets.Layout(width='40px')
        )
        move_up_band_button.on_click(move_up_bands_click)
        move_down_band_button = widgets.Button(
            tooltip='Move down band', button_style='info', icon='arrow-down',
            layout=widgets.Layout(width='40px')
        )
        move_down_band_button.on_click(move_down_bands_click)
        delete_band_button = widgets.Button(
            tooltip='Delete band', button_style='danger', icon='minus',
            layout=widgets.Layout(width='40px')
        )
        delete_band_button.on_click(delete_bands_click)
        wavelengths = widgets.Dropdown(
            options=cfg.sat_band_list,
            description='Wavelength', layout=widgets.Layout(width='250px')
        )
        wavelengths.observe(wavelengths_change)
        add_bandset_button = widgets.Button(
            tooltip='Add band set', button_style='success', icon='plus',
            layout=widgets.Layout(width='40px')
        )
        add_bandset_button.on_click(add_bandset_click)
        delete_bandset_button = widgets.Button(
            tooltip='Delete band set', button_style='danger', icon='minus',
            layout=widgets.Layout(width='40px')
        )
        delete_bandset_button.on_click(delete_bandset_click)

        bandset_rows = widgets.VBox([
            widgets.HBox([
                widgets.VBox([bandsets_label, bandsets_table,
                              widgets.HBox([widgets.HBox(
                                  [add_bandset_button, delete_bandset_button]
                              )])
                              ], layout=widgets.Layout(width='25%')),
                widgets.VBox([widgets.HBox([bandset_label, active_bandset]),
                              bandset_table, wavelengths],
                             layout=widgets.Layout(width='75%')),
                widgets.VBox([add_bands_button, move_up_band_button,
                              move_down_band_button, delete_band_button])
            ])
        ])

        # populate table
        refresh_bandsets()

        """ Download products """

        def table_to_list(table):
            table_list = []
            if table is not None:
                for row in table:
                    table_list.append(
                        '%s | %s | %s | %s' % (row[0], row[1], row[3], row[4])
                        )
            return table_list

        def activate_draw_polygon(_):
            global browser_selector_val, aoi_draw
            if aoi_draw is True:
                aoi_draw = False
            else:
                aoi_draw = True
                browser_selector_val.value = 2

        def download_button_click(_):
            global caller_function
            caller_function = download_function
            activate_browser(None)

        # noinspection PyUnresolvedReferences
        def download_function(input_files):
            global downloaded, browser_selector_val, selected_file_paths
            browser_selector_val.value = 1
            # TODO implement password
            # TODO implement password
            copernicus_user = copernicus_password = None
            selected_file_paths.value = str(input_files).replace(
                "'", '').replace('[', '').replace(']', '')
            files = selected_file_paths.value.split(',')
            if len(files) == 1:
                if len(files[0]) > 0 and os.path.isdir(files[0]):
                    output_path.value = files[0]
            if len(output_path.value) == 0:
                error_message('Select an output directory.')
            else:
                downloaded = self.rs.download_products.download(
                    product_table=download_table,
                    output_path=output_path.value,
                    band_list=selected_bands,
                    copernicus_user=copernicus_user,
                    copernicus_password=copernicus_password
                )
                if downloaded.check:
                    if preprocess_checkbox.value is True:
                        if downloaded.extra is not None:
                            # preprocess
                            directories = downloaded.extra['directory_paths']
                            for directory in directories:
                                base_directory = (
                                    self.rs.files_directories.parent_directory(
                                        directory
                                    )
                                )
                                process_directory = '%s/RT_%s' % (
                                    base_directory,
                                    self.rs.files_directories.get_base_name(
                                        directory
                                    )
                                )
                                self.rs.preprocess_products.preprocess(
                                    input_path=directory,
                                    output_path=process_directory,
                                    nodata_value=0,
                                    add_bandset=True,
                                    bandset_catalog=cfg.default_catalog,
                                )
                self._reset_message()
                refresh_bandsets()

        # noinspection PyUnresolvedReferences
        def find_button_click(_):
            global date_from, date_to, advanced, product_table, download_table
            if date_from.value is None or date_to.value is None:
                error_message('Select a range of dates.')
            else:
                if advanced.value is None:
                    name = None
                else:
                    if len(str(advanced.value)) == 0:
                        name = None
                    else:
                        name = str(advanced.value)
                if (len(ul_x.value) == 0 or len(ul_x.value) == 0
                        or len(ul_x.value) == 0 or len(ul_x.value) == 0):
                    coordinate_list = None
                else:
                    coordinate_list = [
                        float(ul_x.value), float(ul_y.value),
                        float(lr_x.value), float(lr_y.value)
                    ]
                product_table.options = []
                query_result = self.rs.download_products.search(
                    product=products.value,
                    coordinate_list=coordinate_list,
                    name_filter=name,
                    date_from=date_from.value,
                    date_to=date_to.value,
                    max_cloud_cover=cloud_cover.value,
                    result_number=results.value
                )
                if query_result is not None:
                    product_table_result = query_result.extra['product_table']
                    if download_table is None:
                        download_table = product_table_result
                    else:
                        download_table = (
                            self.rs.table_manager.stack_product_table(
                                product_list=[download_table,
                                              product_table_result]
                            )
                        )
                    table = table_to_list(download_table)
                    product_table.options = [(value, value) for value in table]
                self._reset_message()

        # noinspection PyUnresolvedReferences
        def delete_button_click(_):
            global product_table, selected_values, download_table
            if download_table is not None:
                mask = [value not in selected_values for value in range(
                    len(download_table))]
                download_table = download_table[mask]
                table = table_to_list(download_table)
                product_table.options = [(value, value) for value in table]

        # noinspection PyUnresolvedReferences
        def preview_button_click(_):
            global browser_selector_val, layers_control, training_map
            global selected_values, download_table
            if layers_control is None:
                # noinspection SpellCheckingInspection
                layers_control = LayersControl(position='topright')
                training_map.add(layers_control)
            for value in selected_values:
                name = download_table.image[value]
                bottom = download_table.min_lat[value]
                left = download_table.min_lon[value]
                top = download_table.max_lat[value]
                right = download_table.max_lon[value]
                temp_file = cfg.temp.temporary_file_path(name_suffix='.jpg')
                cfg.multiprocess.multi_download_file(
                    url_list=[download_table.preview[selected_values[0]]],
                    output_path_list=[temp_file]
                )
                temp_file2 = cfg.temp.temporary_file_path(name_suffix='.jpg')
                self.rs.shared_tools.image_to_jpg(
                    input_raster=temp_file, output=temp_file2
                )
                with open(temp_file2, 'rb') as f:
                    base64_image = base64.b64encode(f.read()).decode('utf-8')
                training_map.add(LayerGroup(
                    layers=(ImageOverlay(
                        url=f'data:image/jpeg;base64,{base64_image}',
                        bounds=((bottom, left), (top, right))),), name=name)
                )
                browser_selector_val.value = 2

        # get selected products
        # noinspection PyUnresolvedReferences
        def selection_change(change):
            global selected_values, preview, download_table
            selected_values = change['new']
            if len(selected_values) > 0:
                temp_file = cfg.temp.temporary_file_path(name_suffix='.jpg')
                temp_file2 = cfg.temp.temporary_file_path(name_suffix='.jpg')
                cfg.multiprocess.multi_download_file(
                    url_list=[download_table.preview[selected_values[0]]],
                    output_path_list=[temp_file]
                )
                self.rs.shared_tools.image_to_jpg(input_raster=temp_file,
                                                  output=temp_file2)
                file = open(temp_file2, 'rb')
                preview.value = file.read()
                preview.width = '20%'

        def checkbox_change(change):
            global selected_bands
            if change['type'] == 'change' and change['name'] == 'value':
                if change['new'] is True:
                    selected_bands.append(change['owner'].description)
                else:
                    try:
                        selected_bands.remove(change['owner'].description)
                    except Exception as err:
                        str(err)

        """
        search_label = widgets.Label(value='Search products',
                                     style=label_style)
        """
        search_label = widgets.HTML(
            value=f"<div style='color:white; weight:bold; "
                  f"background: #5a5a5a'><b>Search products</b></div>"
        )
        ul_label = widgets.Label(value='UL')
        ul_x = widgets.Text(
            placeholder='X (Lon)', layout=widgets.Layout(width='15%')
            )
        ul_y = widgets.Text(
            placeholder='Y (Lat)', layout=widgets.Layout(width='15%')
            )
        lr_label = widgets.Label(value='LR')
        lr_x = widgets.Text(
            placeholder='X (Lat)', layout=widgets.Layout(width='15%')
            )
        lr_y = widgets.Text(
            placeholder='Y (Lon)', layout=widgets.Layout(width='15%')
            )
        aoi_button = widgets.Button(
            tooltip='Click in the map', layout=widgets.Layout(width='50px'),
            icon='plus', button_style='warning'
            )
        aoi_button.on_click(activate_draw_polygon)
        products = widgets.Dropdown(
            options=list(cfg.product_description.keys()),
            description='Products', layout=widgets.Layout(width='25%')
        )
        date_from = widgets.DatePicker(
            description='Date from', layout=widgets.Layout(width='20%')
            )
        date_to = widgets.DatePicker(
            description='Date to', layout=widgets.Layout(width='20%')
            )
        cloud_cover = widgets.BoundedIntText(
            value=100, min=0, max=100, description='Max cloud cover (%)',
            step=10, layout=widgets.Layout(width='20%')
        )
        results = widgets.BoundedIntText(
            value=20, min=1, max=200, step=10, description='Results',
            layout=widgets.Layout(width='20%')
        )
        advanced = widgets.Text(
            description='Advanced search', layout=widgets.Layout(width='40%')
        )
        find_button = widgets.Button(description='Find',
                                     button_style='primary', icon='search')
        find_button.on_click(find_button_click)
        """
        product_label = widgets.Label(
            value='Product list (product │ name │ date │ cloud cover)',
            style=label_style
        )
        """
        product_label = widgets.HTML(
            value=f"<div style='color:white; weight:bold; "
                  f"background: #5a5a5a'><b>Product list (product │ name "
                  f"│ date │ cloud cover)</b></div>"
        )
        product_table = widgets.SelectMultiple(
            options=[], value=[], rows=10, layout=widgets.Layout(width='100%')
        )
        product_table.observe(selection_change, names='index')
        preview = widgets.Image(layout=widgets.Layout(width='40%'))
        preview_button = widgets.Button(
            tooltip='Preview', button_style='primary', icon='picture-o',
            layout=widgets.Layout(width='20%')
            )
        preview_button.on_click(preview_button_click)
        delete_button = widgets.Button(
            tooltip='Delete row', button_style='danger', icon='minus',
            layout=widgets.Layout(width='20%')
            )
        delete_button.on_click(delete_button_click)
        """
        download_label = widgets.Label(value='Download', style=label_style)
        """
        download_label = widgets.HTML(
            value=f"<div style='color:white; weight:bold; "
                  f"background: #5a5a5a'><b>Download</b></div>"
        )
        # list of bands
        band_list = ['01', '02', '03', '04', '05', '06', '07', '08',  '8A',
                     '09', '10', '11', '12']
        checkboxes = []
        for b in band_list:
            checkbox = widgets.Checkbox(
                value=True, description=b, indent=False,
                layout=widgets.Layout(width='100px')
            )
            checkbox.observe(checkbox_change)
            checkboxes.append(checkbox)

        # hidden
        output_path = widgets.Text(description='Output path',
                                   layout=widgets.Layout(width='70%'))
        download_button = widgets.Button(
            description='Run', button_style='success', icon='caret-right',
            layout=widgets.Layout(width='10%'))
        download_button.on_click(download_button_click)
        preprocess_checkbox = widgets.Checkbox(
            description='Preprocess images', value=True
        )
        download_products_rows = widgets.VBox([
            search_label,
            widgets.HBox([
                ul_label, ul_x, ul_y, lr_label, lr_x, lr_y, aoi_button
            ]),
            widgets.HBox([products, date_from, date_to, cloud_cover]),
            widgets.HBox([results, advanced, find_button]), product_label,
            widgets.HBox(
                [product_table, preview,
                 widgets.VBox([preview_button, delete_button],
                              layout=widgets.Layout(width='20%'))],
                layout=widgets.Layout(width='100%')
                ),
            download_label, widgets.HBox(checkboxes),
            widgets.HBox([preprocess_checkbox]),
            widgets.HBox(
                children=[download_button],
                layout=widgets.Layout(
                    display='flex', flex_flow='column',
                    align_items='flex-end'
                )
            )
        ])

        """ Tabs """

        """ Basic tools """

        # noinspection PyUnresolvedReferences
        def get_vector_fields(input_files):
            global browser_selector_val, selected_file_paths
            global new_file_text
            nonlocal vector_input_text
            nonlocal vector_mc_id_combo, vector_mc_name_combo
            nonlocal vector_c_id_combo, vector_c_name_combo
            selected_file_paths.value = str(input_files).replace(
                "'", ''
            ).replace('[', '').replace(']', '')
            files = selected_file_paths.value.split(',')
            for file in files:
                if len(file) > 0 and (file.endswith('.shp')
                                      or file.endswith('.gpkg')):
                    try:
                        fields = self.rs.shared_tools.get_vector_fields(file)
                        vector_mc_id_combo.options = list(fields.keys())
                        vector_mc_name_combo.options = list(fields.keys())
                        vector_c_id_combo.options = list(fields.keys())
                        vector_c_name_combo.options = list(fields.keys())
                        vector_path = file.strip()
                        vector_input_text.value = vector_path
                        error_message('')
                        break
                    except Exception as err:
                        str(err)
                        error_message('Select a file.')
                else:
                    error_message('Select a file.')
            new_file_text.value = ''
            browser_selector_val.value = 1

        def open_vector_click(_):
            global caller_function
            caller_function = get_vector_fields
            activate_browser(None)

        def import_button_click(_):
            nonlocal vector_input_text
            nonlocal vector_mc_id_combo, vector_mc_name_combo
            nonlocal vector_c_id_combo, vector_c_name_combo
            if cfg.default_signature_catalog is None:
                error_message('Missing training input.')
            else:
                cfg.default_signature_catalog.import_vector(
                    file_path=vector_input_text.value,
                    macroclass_field=vector_mc_id_combo.value,
                    macroclass_name_field=vector_mc_name_combo.value,
                    class_field=vector_c_id_combo.value,
                    class_name_field=vector_c_name_combo.value,
                    calculate_signature=True
                )
                # display geojson data
                display_training_file()
                refresh_training_list()
                cfg.default_signature_catalog.save(output_path=training_path)
                self._reset_message()

        vector_input_text = widgets.Text(placeholder='Select a vector ',
                                         layout=widgets.Layout(width='80%'))
        open_vector_button = widgets.Button(
            tooltip='Select a vector (*.shp;*.gpkg)', button_style='info',
            icon='folder-open', layout=widgets.Layout(width='15%')
        )
        open_vector_button.on_click(open_vector_click)
        vector_mc_id_combo = widgets.Combobox(
            description='MC ID Field', placeholder='-', options=[],
            continuous_update=True, layout=widgets.Layout(width='24%')
        )
        vector_mc_name_combo = widgets.Combobox(
            description='MC Name Field', placeholder='-', options=[],
            continuous_update=True, layout=widgets.Layout(width='24%')
        )
        vector_c_id_combo = widgets.Combobox(
            description='C ID Field', placeholder='-', options=[],
            continuous_update=True, layout=widgets.Layout(width='24%')
        )
        vector_c_name_combo = widgets.Combobox(
            description='C Name Field', placeholder='-', options=[],
            continuous_update=True, layout=widgets.Layout(width='24%')
        )
        import_button = widgets.Button(
            description='Run', button_style='success',
            icon='caret-right', layout=widgets.Layout(width='10%'))
        import_button.on_click(import_button_click)
        import_tool_rows = widgets.VBox([
            widgets.HBox([vector_input_text, open_vector_button],
                         layout=widgets.Layout(width='99%')),
            widgets.HBox([vector_mc_id_combo, vector_mc_name_combo,
                          vector_c_id_combo, vector_c_name_combo],
                         layout=widgets.Layout(width='99%')),
            widgets.HBox(
                children=[import_button], layout=widgets.Layout(
                    display='flex', flex_flow='column', align_items='flex-end'
                )
            )
        ])

        """ Preprocessing """
        clip_raster_rows = widgets.VBox(
            [widgets.Label(value='To be implemented')]
        )
        image_conversion_rows = widgets.VBox(
            [widgets.Label(value='To be implemented')]
        )
        masking_rows = widgets.VBox(
            [widgets.Label(value='To be implemented')]
        )
        mosaic_bandsets_rows = widgets.VBox(
            [widgets.Label(value='To be implemented')]
        )
        reproject_rows = widgets.VBox(
            [widgets.Label(value='To be implemented')]
        )
        split_raster_rows = widgets.VBox(
            [widgets.Label(value='To be implemented')]
        )
        stack_raster_rows = widgets.VBox(
            [widgets.Label(value='To be implemented')]
        )
        vector_to_raster_rows = widgets.VBox(
            [widgets.Label(value='To be implemented')]
        )

        """ Band processing """
        combination_rows = widgets.VBox(
            [widgets.Label(value='To be implemented')]
        )
        dilation_rows = widgets.VBox(
            [widgets.Label(value='To be implemented')]
        )
        erosion_rows = widgets.VBox(
            [widgets.Label(value='To be implemented')]
        )
        sieve_rows = widgets.VBox(
            [widgets.Label(value='To be implemented')]
        )
        neighbor_rows = widgets.VBox(
            [widgets.Label(value='To be implemented')]
        )
        pca_rows = widgets.VBox(
            [widgets.Label(value='To be implemented')]
        )

        """ Classification tab """

        def linear_radio_change(_):
            nonlocal zscore_radio, linear_radio
            zscore_radio.unobserve(zscore_change, names='value')
            if linear_radio.value is True:
                zscore_radio.value = False
            else:
                zscore_radio.value = True
            zscore_radio.observe(zscore_change, names='value')

        def zscore_change(_):
            nonlocal zscore_radio, linear_radio
            linear_radio.unobserve(linear_radio_change, names='value')
            if zscore_radio.value is True:
                linear_radio.value = False
            else:
                linear_radio.value = True
            linear_radio.observe(linear_radio_change, names='value')

        def mc_radio_change(_):
            nonlocal mc_radio, c_radio
            c_radio.unobserve(c_radio_change, names='value')
            if mc_radio.value is True:
                c_radio.value = False
            else:
                c_radio.value = True
            c_radio.observe(c_radio_change, names='value')

        def c_radio_change(_):
            nonlocal mc_radio, c_radio
            mc_radio.unobserve(mc_radio_change, names='value')
            if c_radio.value is True:
                mc_radio.value = False
            else:
                mc_radio.value = True
            mc_radio.observe(mc_radio_change, names='value')

        def scikit_radio_change(_):
            nonlocal scikit_radio, pytorch_radio
            pytorch_radio.unobserve(pytorch_radio_change, names='value')
            if scikit_radio.value is True:
                pytorch_radio.value = False
            else:
                pytorch_radio.value = True
            pytorch_radio.observe(pytorch_radio_change, names='value')

        def pytorch_radio_change(_):
            nonlocal scikit_radio, pytorch_radio
            scikit_radio.unobserve(scikit_radio_change, names='value')
            if pytorch_radio.value is True:
                scikit_radio.value = False
            else:
                scikit_radio.value = True
            scikit_radio.observe(scikit_radio_change, names='value')

        # perform classification
        def classification_run_button_click(_):
            global caller_function
            caller_function = classification_run
            activate_browser(None)

        # perform classification
        # noinspection PyTypeChecker,PyUnresolvedReferences
        def classification_run(_):
            global training_path, classifier_preview, load_classifier
            global training_map, save_classifier, preview_point, new_file_text
            global classification_preview_group, old_classification_preview
            global classification_accordion, algorithm_selection
            nonlocal classification_bandset, mc_radio
            nonlocal input_normalization_checkbox, zscore_radio
            nonlocal maximum_likelihood_threshold_checkbox
            nonlocal maximum_likelihood_threshold_bounded
            nonlocal maximum_likelihood_signature_raster_checkbox
            nonlocal maximum_likelihood_confidence_raster_checkbox
            nonlocal minimum_distance_threshold_checkbox
            nonlocal minimum_distance_threshold_bounded
            nonlocal minimum_distance_signature_raster_checkbox
            nonlocal minimum_distance_confidence_raster_checkbox
            nonlocal pytorch_radio, mlp_best_estimator_checkbox
            nonlocal mlp_best_estimator_input, mlp_activation_text
            nonlocal mlp_cross_validation_checkbox, mlp_batch_size_text
            nonlocal mlp_training_proportion_bounded, mlp_max_iter_input
            nonlocal mlp_learning_rate_bounded, mlp_alpha_bounded
            nonlocal mlp_hidden_layer_text, mlp_confidence_raster_checkbox
            nonlocal rf_best_estimator_checkbox, rf_best_estimator_input
            nonlocal rf_one_vs_rest_checkbox, rf_balanced_class_weight_checkbox
            nonlocal rf_cross_validation_checkbox, rf_tree_number_input
            nonlocal rf_min_split_input, rm_max_features_text
            nonlocal rf_confidence_raster_checkbox
            nonlocal sam_signature_raster_checkbox, sam_threshold_bounded
            nonlocal sam_confidence_raster_checkbox, sam_threshold_checkbox
            nonlocal svm_regularization_input, svc_gamma_text, svc_kernel_text
            nonlocal svm_best_estimator_checkbox, svm_best_estimator_input
            nonlocal svm_confidence_raster_checkbox
            nonlocal svm_balanced_class_weight_checkbox
            nonlocal svm_cross_validation_checkbox
            threshold = False
            signature_raster = False
            cross_validation = True
            find_best_estimator = False
            classification_confidence = False
            input_normalization = class_weight = None
            rf_max_features = rf_number_trees = rf_min_samples_split = None
            svm_c = svm_gamma = svm_kernel = mlp_hidden_layer_sizes = None
            mlp_training_portion = mlp_alpha = mlp_learning_rate_init = None
            mlp_max_iter = mlp_batch_size = mlp_activation = None
            self.rs.configurations.action = True
            # if not preview ask for output file
            if preview_point is None:
                browser_selector_val.value = 1
                c_output_path = str(
                    '%s/%s' % (browser_dir, new_file_text.value)
                ).replace('//', '/')
                # TODO implement save classifier
                if save_classifier is True:
                    if not c_output_path.lower().endswith(
                            self.rs.configurations.rsmo_suffix):
                        c_output_path += self.rs.configurations.rsmo_suffix
            else:
                # path for preview
                c_output_path = (
                    self.rs.configurations.temp.temporary_file_path(
                        name_suffix='.vrt'
                    )
                )
            if c_output_path.lower().endswith('.vrt'):
                pass
            elif not c_output_path.lower().endswith('.tif'):
                c_output_path += '.tif'
            # TODO implement load classifier
            if len('') > 0:
                load_classifier = 'classifier_path'
            # get bandset
            bandset_number = int(classification_bandset.value)
            if ipywidgets_version[0] != '7':
                classifier_index = classification_accordion.selected_index
            else:
                classifier_index = algorithm_selection.index
            if classifier_index is None:
                classifier_index = 0
            classifier_list = [
                self.rs.configurations.maximum_likelihood,
                self.rs.configurations.minimum_distance,
                self.rs.configurations.multi_layer_perceptron,
                self.rs.configurations.random_forest,
                self.rs.configurations.spectral_angle_mapping,
                self.rs.configurations.support_vector_machine
            ]
            classifier_name = classifier_list[classifier_index]
            if mc_radio.value is True:
                macroclass = True
            else:
                macroclass = False
            if input_normalization_checkbox.value is True:
                if zscore_radio.value is True:
                    input_normalization = self.rs.configurations.z_score
                else:
                    input_normalization = self.rs.configurations.linear_scaling
            if training_path is None:
                error_message('Training input missing.')
                return False
            if (cfg.default_signature_catalog is None
                    or cfg.default_signature_catalog is False):
                error_message('Training input missing.')
                return False
            else:
                signature_catalog = cfg.default_signature_catalog
            if save_classifier is True:
                only_fit = True
            else:
                only_fit = False
            # maximum likelihood
            if classifier_name == self.rs.configurations.maximum_likelihood:
                if maximum_likelihood_threshold_checkbox.value is True:
                    threshold = maximum_likelihood_threshold_bounded.value
                if maximum_likelihood_signature_raster_checkbox.value is True:
                    signature_raster = True
                if maximum_likelihood_confidence_raster_checkbox.value is True:
                    classification_confidence = True
            # minimum distance
            elif classifier_name == self.rs.configurations.minimum_distance:
                if minimum_distance_threshold_checkbox.value is True:
                    threshold = minimum_distance_threshold_bounded.value
                if minimum_distance_signature_raster_checkbox.value is True:
                    signature_raster = True
                if minimum_distance_confidence_raster_checkbox.value is True:
                    classification_confidence = True
            # multi layer perceptron
            elif (classifier_name ==
                  self.rs.configurations.multi_layer_perceptron):
                if pytorch_radio.value is True:
                    classifier_name = (
                        self.rs.configurations.pytorch_multi_layer_perceptron
                    )
                if mlp_best_estimator_checkbox.value is True:
                    find_best_estimator = int(mlp_best_estimator_input.value)
                if mlp_cross_validation_checkbox.value is True:
                    cross_validation = True
                else:
                    cross_validation = False
                mlp_training_portion = mlp_training_proportion_bounded.value
                mlp_learning_rate_init = mlp_learning_rate_bounded.value
                mlp_alpha = mlp_alpha_bounded.value
                mlp_max_iter = int(mlp_max_iter_input.value)
                try:
                    mlp_batch_size = int(mlp_batch_size_text.value)
                except Exception as err:
                    str(err)
                    mlp_batch_size = 'auto'
                mlp_activation = mlp_activation_text.value
                hidden_layers = mlp_hidden_layer_text.value
                try:
                    mlp_hidden_layer_sizes = eval('[%s]' % hidden_layers)
                except Exception as err:
                    mlp_hidden_layer_sizes = [100]
                    error_message('Warning: hidden layer sizes')
                    str(err)
                if mlp_confidence_raster_checkbox.value is True:
                    classification_confidence = True
            # random forest
            elif classifier_name == self.rs.configurations.random_forest:
                if rf_best_estimator_checkbox.value is True:
                    find_best_estimator = int(rf_best_estimator_input.value)
                if rf_one_vs_rest_checkbox.value is True:
                    classifier_name = self.rs.configurations.random_forest_ovr
                if rf_balanced_class_weight_checkbox.value is True:
                    class_weight = 'balanced'
                if rf_cross_validation_checkbox.value is True:
                    cross_validation = True
                else:
                    cross_validation = False
                rf_number_trees = int(rf_tree_number_input.value)
                rf_min_samples_split = int(rf_min_split_input.value)
                if len(rm_max_features_text.value) > 0:
                    if rm_max_features_text.value == 'sqrt':
                        rf_max_features = 'sqrt'
                    else:
                        try:
                            rf_max_features = float(rm_max_features_text.value)
                        except Exception as err:
                            str(err)
                if rf_confidence_raster_checkbox.value is True:
                    classification_confidence = True
            # spectral angle mapping
            elif (classifier_name ==
                  self.rs.configurations.spectral_angle_mapping):
                if sam_threshold_checkbox.value is True:
                    threshold = sam_threshold_bounded.value
                if sam_signature_raster_checkbox.value is True:
                    signature_raster = True
                if sam_confidence_raster_checkbox.value is True:
                    classification_confidence = True
            # SVM
            elif (classifier_name ==
                  self.rs.configurations.support_vector_machine):
                svm_c = svm_regularization_input.value
                if len(svc_gamma_text.value) > 0:
                    if svc_gamma_text.value == 'scale':
                        svm_gamma = 'scale'
                    elif svc_gamma_text.value == 'auto':
                        svm_gamma = 'auto'
                    else:
                        try:
                            svm_gamma = float(svc_gamma_text.value)
                        except Exception as err:
                            str(err)
                if len(svc_kernel_text.value) > 0:
                    svm_kernel = svc_kernel_text.value
                if svm_best_estimator_checkbox.value is True:
                    find_best_estimator = int(svm_best_estimator_input.value)
                if svm_cross_validation_checkbox.value is True:
                    cross_validation = True
                else:
                    cross_validation = False
                if svm_balanced_class_weight_checkbox.value is True:
                    class_weight = 'balanced'
                if svm_confidence_raster_checkbox.value is True:
                    classification_confidence = True
            # get bandset
            bandset_x = cfg.default_catalog.get(bandset_number)
            if bandset_x is None:
                error_message('Bandset not found')
                return
            band_count = bandset_x.get_band_count()
            if band_count == 0:
                'cfg.mx.msg_war_6(bandset_number)'
                return
            # classification
            if preview_point is None:
                bandset = bandset_number
            # classification preview
            else:
                # subset bandset
                preview_size = preview_size_input.value
                # prepare virtual raster of input
                dummy_path = self.rs.configurations.temp.temporary_file_path(
                    name_suffix='.vrt'
                )
                prepared = self.rs.shared_tools.prepare_process_files(
                    input_bands=bandset_number, output_path=dummy_path,
                    bandset_catalog=cfg.default_catalog
                )
                temporary_virtual_raster = prepared['temporary_virtual_raster']
                if type(temporary_virtual_raster) is list:
                    temporary_virtual_raster = temporary_virtual_raster[0]
                # get pixel size
                x_size, y_size = self.rs.shared_tools.get_raster_pixel_size(
                    temporary_virtual_raster
                )
                # calculate preview window
                left = preview_point[0] - (x_size * preview_size) / 2
                top = preview_point[1] + (y_size * preview_size) / 2
                right = preview_point[0] + (x_size * preview_size) / 2
                bottom = preview_point[1] - (y_size * preview_size) / 2
                # copy bandset and subset
                bandset = deepcopy(bandset_x)
                bandset.box_coordinate_list = [left, top, right, bottom]
                # load classifier
                if load_classifier is None:
                    # classifier path
                    classifier_path = (
                        self.rs.configurations.temp.temporary_file_path(
                            name_suffix=self.rs.configurations.rsmo_suffix
                        )
                    )
                    # calculate from training on the whole bandset
                    if classifier_preview is None:
                        # run classification
                        fit_classifier = self.rs.band_classification(
                            only_fit=True, save_classifier=True,
                            input_bands=bandset_number,
                            output_path=classifier_path,
                            spectral_signatures=signature_catalog,
                            macroclass=macroclass,
                            algorithm_name=classifier_name,
                            bandset_catalog=cfg.default_catalog,
                            threshold=threshold,
                            signature_raster=signature_raster,
                            cross_validation=cross_validation,
                            input_normalization=input_normalization,
                            load_classifier=load_classifier,
                            class_weight=class_weight,
                            find_best_estimator=find_best_estimator,
                            rf_max_features=rf_max_features,
                            rf_number_trees=rf_number_trees,
                            rf_min_samples_split=rf_min_samples_split,
                            svm_c=svm_c, svm_gamma=svm_gamma,
                            svm_kernel=svm_kernel,
                            mlp_training_portion=mlp_training_portion,
                            mlp_alpha=mlp_alpha,
                            mlp_learning_rate_init=mlp_learning_rate_init,
                            mlp_max_iter=mlp_max_iter,
                            mlp_batch_size=mlp_batch_size,
                            mlp_activation=mlp_activation,
                            mlp_hidden_layer_sizes=mlp_hidden_layer_sizes,
                            classification_confidence=classification_confidence
                        )
                        if fit_classifier.check:
                            only_fit = False
                            save_classifier = False
                            classifier_preview = fit_classifier.extra[
                                'model_path'
                            ]
                        else:
                            error_message('Classification error.')
                            return
                    # load classifier
                    load_classifier = classifier_preview
            # run classification
            try:
                output = self.rs.band_classification(
                    input_bands=bandset, output_path=c_output_path,
                    spectral_signatures=signature_catalog,
                    macroclass=macroclass, algorithm_name=classifier_name,
                    bandset_catalog=cfg.default_catalog, threshold=threshold,
                    signature_raster=signature_raster,
                    cross_validation=cross_validation,
                    input_normalization=input_normalization,
                    load_classifier=load_classifier, class_weight=class_weight,
                    find_best_estimator=find_best_estimator,
                    rf_max_features=rf_max_features,
                    rf_number_trees=rf_number_trees,
                    rf_min_samples_split=rf_min_samples_split,
                    svm_c=svm_c, svm_gamma=svm_gamma, svm_kernel=svm_kernel,
                    mlp_training_portion=mlp_training_portion,
                    mlp_alpha=mlp_alpha,
                    mlp_learning_rate_init=mlp_learning_rate_init,
                    mlp_max_iter=mlp_max_iter, mlp_batch_size=mlp_batch_size,
                    mlp_activation=mlp_activation,
                    mlp_hidden_layer_sizes=mlp_hidden_layer_sizes,
                    classification_confidence=classification_confidence,
                    only_fit=only_fit, save_classifier=save_classifier
                )
            except Exception as err:
                error_message(str(err))
                output = None
            if output is None:
                error_message('Classification error.')
            elif output.check:
                if save_classifier is not True:
                    output_raster = output.path
                    if output_raster is False:
                        error_message('Raster output not found.')
                    else:
                        # apply symbology
                        classification_jpg = create_raster_symbology(
                            output_raster, macroclass
                        )
                        # add raster to layers
                        with open(classification_jpg, 'rb') as f:
                            base64_i = (
                                base64.b64encode(f.read()).decode('utf-8')
                            )
                        bandset_bbox = cfg.default_catalog.get_bbox()
                        if preview_point is None and bandset_bbox is None:
                            error_message('Error in band set bounding box.')
                            return False
                        elif preview_point is not None:
                            bandset_crs = cfg.default_catalog.get().crs
                            bbox = self.rs.shared_tools.get_raster_bbox(
                                output_raster
                            )
                            preview_bottom_left = (
                                self.rs.shared_tools.project_point_coordinates(
                                    bbox[0], bbox[3], bandset_crs, 4326
                                )
                            )
                            preview_top_right = (
                                self.rs.shared_tools.project_point_coordinates(
                                    bbox[2], bbox[1], bandset_crs, 4326
                                )
                            )
                            rgb_left, rgb_bottom = preview_bottom_left
                            rgb_right, rgb_top = preview_top_right
                            layer = ImageOverlay(
                                url=f'data:image/jpeg;base64,{base64_i}',
                                bounds=((rgb_bottom, rgb_left),
                                        (rgb_top, rgb_right))
                            )
                            classification_preview_group.substitute(
                                old_classification_preview, layer
                            )
                            old_classification_preview = layer
                        else:
                            name = self.rs.files_directories.file_name(
                                output_raster
                            )
                            bandset_crs = cfg.default_catalog.get().crs
                            bandset_bottom_left = (
                                self.rs.shared_tools.project_point_coordinates(
                                    bandset_bbox[0], bandset_bbox[3],
                                    bandset_crs, 4326
                                )
                            )
                            bandset_top_right = (
                                self.rs.shared_tools.project_point_coordinates(
                                    bandset_bbox[2], bandset_bbox[1],
                                    bandset_crs, 4326
                                )
                            )
                            if bandset_bottom_left is not None:
                                rgb_left, rgb_bottom = bandset_bottom_left
                            else:
                                error_message('Create band set.')
                                return False
                            if bandset_top_right is not None:
                                rgb_right, rgb_top = bandset_top_right
                            else:
                                error_message('Create band set.')
                                return False
                            layer = ImageOverlay(
                                url=f'data:image/jpeg;base64,{base64_i}',
                                bounds=((rgb_bottom, rgb_left),
                                        (rgb_top, rgb_right))
                            )
                            training_map.add(
                                LayerGroup(layers=[layer], name=name)
                            )
                    preview_point = None
                    # TODO implement adding algorithm raster
                    if 'algorithm_raster' in output.extra:
                        if output.extra['algorithm_raster'] is not None:
                            # add raster to layers
                            pass
                    if 'signature_rasters' in output.extra:
                        # add raster to layers
                        try:
                            for s in output.extra['signature_rasters']:
                                if s is not None:
                                    pass
                        except Exception as err:
                            str(err)
                    self._reset_message()
            else:
                error_message('Classification error.')
            return output

        self._classification_run = classification_run

        def create_raster_symbology(raster, macroclass=True):
            value_name_dic, value_color_dic = export_symbology_to_rgb_color(
                macroclass=macroclass
            )
            colored_array = self.rs.shared_tools.get_colored_raster(
                raster, value_color_dic
            )
            temp_file = cfg.temp.temporary_file_path(name_suffix='.jpg')
            Image.fromarray(colored_array).save(temp_file)
            return temp_file

        # export symbology colors
        def export_symbology_to_rgb_color(macroclass=True):
            if macroclass is True:
                value_color = (
                    cfg.default_signature_catalog.macroclasses_color_string
                )
                value_color_dictionary = {}
                for i in value_color:
                    value_color_dictionary[i] = tuple(
                        int(value_color[i][c:c + 2], 16) for c in
                        (1, 3, 5)
                    )
                value_name_dictionary = (
                    cfg.default_signature_catalog.macroclasses
                )
            else:
                classes = cfg.default_signature_catalog.table.class_id.tolist()
                names = cfg.default_signature_catalog.table.class_name.tolist()
                colors = cfg.default_signature_catalog.table.color.tolist()
                value_name_dictionary = {}
                value_color_dictionary = {}
                for i, class_i in enumerate(classes):
                    value_name_dictionary[class_i] = names[i]
                    value_color_dictionary[class_i] = tuple(
                        int(colors[i][c:c + 2], 16) for c in (1, 3, 5)
                    )
            return value_name_dictionary, value_color_dictionary

        def classification_accordion_change(_):
            global classifier_preview, load_classifier
            classifier_preview = None
            load_classifier = None

        def algorithm_selection_change(_):
            global classifier_preview, load_classifier
            classifier_preview = None
            load_classifier = None

        """
        input_classification_label = widgets.Label(
            value='Input', layout=widgets.Layout(width='98%'),
            style=label_style
        )
        """
        input_classification_label = widgets.HTML(
            value=f"<div style='color:white; weight:bold; "
                  f"background: #5a5a5a'><b>Input</b></div>"
        )
        classification_bandsets_label = widgets.Label(
            value='Select input bandset', layout=widgets.Layout(width='20%')
        )
        classification_bandset = widgets.BoundedIntText(
            value=1, min=1,
            max=cfg.default_catalog.get_bandset_count(), step=1,
            layout=widgets.Layout(width='5%')
        )
        input_normalization_checkbox = widgets.Checkbox(
            value=False, description='Use input normalization', indent=False,
            layout=widgets.Layout(width='20%')
            )
        zscore_radio = widgets.Checkbox(description='Z-score', value=True,
                                        layout=widgets.Layout(width='25%'))
        zscore_radio.observe(zscore_change, names='value')
        linear_radio = widgets.Checkbox(
            description='Linear scaling', value=False,
            layout=widgets.Layout(width='25%')
        )
        linear_radio.observe(linear_radio_change, names='value')
        classification_training_label = widgets.Label(
            value='Use training', layout=widgets.Layout(width='10%')
        )
        mc_radio = widgets.Checkbox(description='Macroclass ID', value=True,
                                    layout=widgets.Layout(width='30%'))
        mc_radio.observe(mc_radio_change, names='value')
        c_radio = widgets.Checkbox(description='Class ID', value=False,
                                   layout=widgets.Layout(width='20%'))
        c_radio.observe(c_radio_change, names='value')
        """
        algorithm_classification_label = widgets.Label(
            value='Algorithm', layout=widgets.Layout(width='98%'),
            style=label_style
        )
        """
        algorithm_classification_label = widgets.HTML(
            value=f"<div style='color:white; weight:bold; "
                  f"background: #5a5a5a'><b>Algorithm</b></div>"
        )
        maximum_likelihood_use_label = widgets.Label(value='Use')
        maximum_likelihood_threshold_checkbox = widgets.Checkbox(
            description='Single threshold', value=True,
            layout=widgets.Layout(width='30%')
            )
        maximum_likelihood_threshold_bounded = widgets.BoundedFloatText(
            value=0.00, min=0, max=100.0, step=0.1, tooltip='Single threshold',
            continuous_update=True, layout=widgets.Layout(width='10%')
        )
        maximum_likelihood_signature_raster_checkbox = widgets.Checkbox(
            description='Save signature raster', value=False,
            layout=widgets.Layout(width='50%')
            )
        maximum_likelihood_confidence_raster_checkbox = widgets.Checkbox(
            description='Calculate classification confidence raster',
            value=False, ayout=widgets.Layout(width='40%')
            )
        maximum_likelihood_rows = widgets.VBox([
            widgets.HBox([maximum_likelihood_use_label,
                          maximum_likelihood_threshold_checkbox,
                          maximum_likelihood_threshold_bounded]),
            maximum_likelihood_signature_raster_checkbox,
            maximum_likelihood_confidence_raster_checkbox
        ])

        minimum_distance_use_label = widgets.Label(value='Use')
        minimum_distance_threshold_checkbox = widgets.Checkbox(
            description='Single threshold', value=True,
            layout=widgets.Layout(width='30%')
        )
        minimum_distance_threshold_bounded = widgets.BoundedFloatText(
            value=0.00, min=0, max=99999999, tooltip='Single threshold',
            step=1, continuous_update=True, layout=widgets.Layout(width='10%')
        )
        minimum_distance_signature_raster_checkbox = widgets.Checkbox(
            description='Save signature raster', value=False,
            layout=widgets.Layout(width='50%')
        )
        minimum_distance_confidence_raster_checkbox = widgets.Checkbox(
            description='Calculate classification confidence raster',
            value=False, layout=widgets.Layout(width='40%')
        )
        minimum_distance_rows = widgets.VBox([
            widgets.HBox([minimum_distance_use_label,
                          minimum_distance_threshold_checkbox,
                          minimum_distance_threshold_bounded]),
            minimum_distance_signature_raster_checkbox,
            minimum_distance_confidence_raster_checkbox
        ])

        classification_framework_label = widgets.Label(
            value='Use framework', layout=widgets.Layout(width='10%')
        )
        scikit_radio = widgets.Checkbox(description='scikit-learn', value=True,
                                        layout=widgets.Layout(width='20%'))
        scikit_radio.observe(scikit_radio_change, names='value')
        pytorch_radio = widgets.Checkbox(description='PyTorch', value=False,
                                         layout=widgets.Layout(width='20%'))
        pytorch_radio.observe(pytorch_radio_change, names='value')

        mlp_hidden_layer_label = widgets.Label(
            value='Hidden layer sizes', layout=widgets.Layout(width='30%')
        )
        mlp_hidden_layer_text = widgets.Text(
            tooltip='Hidden layer sizes',
            value='100', layout=widgets.Layout(width='68%')
        )
        mlp_max_iter_input = widgets.BoundedIntText(
            value=200, min=1, max=100000, step=1, continuous_update=True,
            description='Max iter', tooltip='Max iter',
            layout=widgets.Layout(width='20%')
        )
        mlp_activation_text = widgets.Text(
            description='Activation', tooltip='Activation',
            value='relu', layout=widgets.Layout(width='15%')
        )
        mlp_alpha_bounded = widgets.BoundedFloatText(
            value=0.0100, min=0.00001, max=99999999, step=1,
            description='Alpha', tooltip='Alpha',
            continuous_update=True, layout=widgets.Layout(width='20%')
        )
        mlp_training_proportion_bounded = widgets.BoundedFloatText(
            value=0.9, min=0.00001, max=99.9, step=1,
            description='Training proportion',
            tooltip='Training proportion',
            continuous_update=True, layout=widgets.Layout(width='30%')
        )
        mlp_batch_size_text = widgets.Text(
            description='Batch size', tooltip='Batch size',
            value='auto', layout=widgets.Layout(width='25%')
        )
        mlp_learning_rate_bounded = widgets.BoundedFloatText(
            value=0.00100, min=0.00001, max=99999999, step=1,
            description='Learning rate init',
            tooltip='Learning rate init',
            continuous_update=True, layout=widgets.Layout(width='30%')
        )
        mlp_cross_validation_checkbox = widgets.Checkbox(
            description='Cross validation', value=True,
            layout=widgets.Layout(width='50%')
        )
        mlp_best_estimator_checkbox = widgets.Checkbox(
            description='Find best estimator with steps', value=False,
            layout=widgets.Layout(width='50%')
        )
        mlp_best_estimator_input = widgets.BoundedIntText(
            value=5, min=1, max=100000, step=1, continuous_update=True,
            tooltip='Find best estimator with steps',
            layout=widgets.Layout(width='20%')
        )
        mlp_confidence_raster_checkbox = widgets.Checkbox(
            description='Calculate classification confidence raster',
            value=False, layout=widgets.Layout(width='40%')
        )
        mlp_rows = widgets.VBox([
            widgets.HBox([classification_framework_label, scikit_radio,
                          pytorch_radio]),
            widgets.HBox([mlp_hidden_layer_label, mlp_hidden_layer_text]),
            widgets.HBox([mlp_max_iter_input, mlp_activation_text,
                          mlp_alpha_bounded]),
            widgets.HBox([mlp_training_proportion_bounded, mlp_batch_size_text,
                          mlp_learning_rate_bounded]),
            mlp_cross_validation_checkbox,
            widgets.HBox(
                [mlp_best_estimator_checkbox, mlp_best_estimator_input]
            ),
            mlp_confidence_raster_checkbox
        ])

        rf_tree_number_input = widgets.BoundedIntText(
            value=10, min=1, max=100000, step=1, continuous_update=True,
            description='Number of trees', tooltip='Number of trees',
            layout=widgets.Layout(width='30%')
        )
        rf_min_split_input = widgets.BoundedIntText(
            value=10, min=1, max=100000, step=1, continuous_update=True,
            description='Minimum number to split',
            tooltip='Minimum number to split',
            layout=widgets.Layout(width='30%')
        )
        rm_max_features_text = widgets.Text(
            description='Max features', tooltip='ax features',
            value='', layout=widgets.Layout(width='35%')
        )
        rf_one_vs_rest_checkbox = widgets.Checkbox(
            description='One-Vs-Rest', value=False,
            layout=widgets.Layout(width='50%')
        )
        rf_cross_validation_checkbox = widgets.Checkbox(
            description='Cross validation', value=True,
            layout=widgets.Layout(width='50%')
        )
        rf_balanced_class_weight_checkbox = widgets.Checkbox(
            description='Balanced class weight', value=False,
            layout=widgets.Layout(width='50%')
        )
        rf_best_estimator_checkbox = widgets.Checkbox(
            description='Find best estimator with steps', value=False,
            layout=widgets.Layout(width='50%')
        )
        rf_best_estimator_input = widgets.BoundedIntText(
            value=5, min=1, max=100000, step=1, continuous_update=False,
            tooltip='Find best estimator with steps',
            layout=widgets.Layout(width='20%')
        )
        rf_confidence_raster_checkbox = widgets.Checkbox(
            description='Calculate classification confidence raster',
            value=False, layout=widgets.Layout(width='40%')
        )
        random_forest_rows = widgets.VBox([
            widgets.HBox([rf_tree_number_input, rf_min_split_input,
                          rm_max_features_text]),
            rf_one_vs_rest_checkbox,
            rf_cross_validation_checkbox,
            rf_balanced_class_weight_checkbox,
            widgets.HBox(
                [rf_best_estimator_checkbox, rf_best_estimator_input]
            ),
            rf_confidence_raster_checkbox
        ])

        sam_use_label = widgets.Label(value='Use')
        sam_threshold_checkbox = widgets.Checkbox(
            description='Single threshold', value=True,
            layout=widgets.Layout(width='30%')
            )
        sam_threshold_bounded = widgets.BoundedFloatText(
            value=0.00, min=0, max=90, step=1, tooltip='Single threshold',
            continuous_update=True, layout=widgets.Layout(width='10%')
        )
        sam_signature_raster_checkbox = widgets.Checkbox(
            description='Save signature raster', value=False,
            layout=widgets.Layout(width='50%')
            )
        sam_confidence_raster_checkbox = widgets.Checkbox(
            description='Calculate classification confidence raster',
            value=False, layout=widgets.Layout(width='40%')
            )
        sam_rows = widgets.VBox([
            widgets.HBox([sam_use_label, sam_threshold_checkbox,
                          sam_threshold_bounded]),
            sam_signature_raster_checkbox, sam_confidence_raster_checkbox
        ])

        svm_regularization_label = widgets.Label(
            value='Regularization parameter C',
            layout=widgets.Layout(width='30%')
        )
        svm_regularization_input = widgets.BoundedFloatText(
            value=1, min=0.0001, max=99999, step=1, continuous_update=True,
            tooltip='Regularization parameter C',
            layout=widgets.Layout(width='15%')
        )
        svc_kernel_text = widgets.Text(
            description='Kernel', tooltip='Kernel',
            value='rbf', layout=widgets.Layout(width='25%')
        )
        svc_gamma_text = widgets.Text(
            description='Gamma', tooltip='Gamma',
            value='scale', layout=widgets.Layout(width='25%')
        )
        svm_cross_validation_checkbox = widgets.Checkbox(
            description='Cross validation', value=True,
            layout=widgets.Layout(width='50%')
        )
        svm_balanced_class_weight_checkbox = widgets.Checkbox(
            description='Balanced class weight', value=False,
            layout=widgets.Layout(width='50%')
        )
        svm_best_estimator_checkbox = widgets.Checkbox(
            description='Find best estimator with steps', value=False,
            layout=widgets.Layout(width='50%')
        )
        svm_best_estimator_input = widgets.BoundedIntText(
            value=5, min=1, max=100000, step=1, continuous_update=False,
            tooltip='Find best estimator with steps',
            layout=widgets.Layout(width='20%')
        )
        svm_confidence_raster_checkbox = widgets.Checkbox(
            description='Calculate classification confidence raster',
            value=False, layout=widgets.Layout(width='40%')
        )
        svm_rows = widgets.VBox([
            widgets.HBox([svm_regularization_label, svm_regularization_input,
                          svc_kernel_text, svc_gamma_text]),
            svm_cross_validation_checkbox,
            svm_balanced_class_weight_checkbox,
            widgets.HBox(
                [svm_best_estimator_checkbox, svm_best_estimator_input]
            ),
            svm_confidence_raster_checkbox
        ])

        if ipywidgets_version[0] != '7':
            classification_accordion = widgets.Accordion(
                children=[
                    maximum_likelihood_rows, minimum_distance_rows, mlp_rows,
                    random_forest_rows, sam_rows, svm_rows
                ],
                titles=(
                    'Maximum Likelihood', 'Minimum Distance',
                    'Multi-layer Perceptron', 'Random Forest',
                    'Spectral Angle Mapping', 'Support Vector Machine'
                ),  selected_index=0
            )
            classification_accordion.observe(
                classification_accordion_change, names='selected_index')
        else:
            # alternative classification selection
            algorithm_selection = widgets.Select(
                options=['Maximum Likelihood', 'Minimum Distance',
                         'Multi-layer Perceptron', 'Random Forest',
                         'Spectral Angle Mapping', 'Support Vector Machine'],
                rows=8
            )
            algorithm_selection.observe(algorithm_selection_change,
                                        names='index')
            classification_accordion = widgets.VBox([
                algorithm_selection,
                widgets.Label(value='> Maximum Likelihood'),
                maximum_likelihood_rows,
                widgets.Label(value='> Minimum Distance'),
                minimum_distance_rows,
                widgets.Label(value='> Multi-layer Perceptron'), mlp_rows,
                widgets.Label(value='> Random Forest'), random_forest_rows,
                widgets.Label(value='> Spectral Angle Mapping'), sam_rows,
                widgets.Label(value='> Support Vector Machine'), svm_rows
            ])

        """
        run_classification_label = widgets.Label(
            value='Run', layout=widgets.Layout(width='98%'), style=label_style
        )
        """
        run_classification_label = widgets.HTML(
            value=f"<div style='color:white; weight:bold; "
                  f"background: #5a5a5a'><b>Run</b></div>"
        )
        classification_run_button = widgets.Button(
            description='Run', button_style='success', icon='caret-right',
            layout=widgets.Layout(width='10%'))
        classification_run_button.on_click(classification_run_button_click)

        classification_rows = widgets.VBox([
            input_classification_label,
            widgets.HBox([
                classification_bandsets_label, classification_bandset,
                input_normalization_checkbox, zscore_radio, linear_radio
            ]),
            widgets.HBox([classification_training_label, mc_radio, c_radio]),
            algorithm_classification_label, classification_accordion,
            run_classification_label,
            widgets.HBox(
                children=[classification_run_button], layout=widgets.Layout(
                    display='flex', flex_flow='column', align_items='flex-end'
                )
            )
        ])

        """ Postprocessing """
        accuracy_rows = widgets.VBox(
            [widgets.Label(value='To be implemented')]
        )
        classification_report_rows = widgets.VBox(
            [widgets.Label(value='To be implemented')]
        )
        raster_to_vector_rows = widgets.VBox(
            [widgets.Label(value='To be implemented')]
        )
        cross_classification_rows = widgets.VBox(
            [widgets.Label(value='To be implemented')]
        )
        reclassification_rows = widgets.VBox(
            [widgets.Label(value='To be implemented')]
        )

        """ Band calc """

        bandcalc_rows = widgets.VBox(
            [widgets.Label(value='To be implemented')]
        )

        """ Training """

        # get selected training
        def selection_training_change(change):
            global selection_training
            selection_training = change['new']

        # delete selected training
        def delete_signature_click(_):
            global selection_training
            signatures = cfg.default_signature_catalog.table
            signatures.sort(order='signature_id')
            ids = signatures['signature_id']
            for selected in reversed(selection_training):
                cfg.default_signature_catalog.remove_signature_by_id(
                    signature_id=ids[selected]
                )
            cfg.default_signature_catalog.save(output_path=training_path)
            display_training_file()
            refresh_training_list()
            self._reset_message()

        # plot selected training
        def plot_signature_click(_):
            global browser_selector_val, plot_widget, plot_button
            global selection_training
            signatures = cfg.default_signature_catalog.table
            signatures.sort(order='signature_id')
            ids = signatures['signature_id']
            selected_ids = []
            for selected in selection_training:
                selected_ids.append(ids[selected])
            plot = cfg.default_signature_catalog.add_signatures_to_plot_by_id(
                signature_id_list=selected_ids, return_plot=True
            )
            fig = plot.gcf()
            fig.set_figwidth(10)
            with plot_widget:
                plot.show()
            self._plot_button_click(None)

        # save training
        # noinspection PyTypeChecker
        def save_training_click(_):
            global training_path, classifier_preview, load_classifier
            nonlocal mc_id_input, c_id_input, mc_name_input_text
            nonlocal c_name_input_text, mc_color_input, c_color_input
            if cfg.default_signature_catalog is None:
                error_message('Missing training input file')
                return
            cfg.default_signature_catalog.import_vector(
                file_path=last_region_path,
                macroclass_value=int(mc_id_input.value),
                class_value=int(c_id_input.value),
                macroclass_name=mc_name_input_text.value,
                class_name=c_name_input_text.value,
                calculate_signature=True, color_string=c_color_input.value
            )
            cfg.default_signature_catalog.set_macroclass_color(
                macroclass_id=int(mc_id_input.value),
                color_string=mc_color_input.value
            )
            # display geojson data
            display_training_file()
            refresh_training_list()
            cfg.default_signature_catalog.save(output_path=training_path)
            classifier_preview = None
            load_classifier = None
            self._reset_message()

        def refresh_training_list():
            nonlocal training_dock
            training_dock_options = training_rows()
            training_dock.options = training_dock_options

        def training_rows():
            sep = '│ '
            training_options = []
            max_widths = {}
            signatures = cfg.default_signature_catalog.table
            macroclasses = cfg.default_signature_catalog.macroclasses
            signatures.sort(order='signature_id')
            for attribute in ['macroclass_id', 'class_id', 'class_name']:
                max_widths[attribute] = 1
                for signature in signatures:
                    max_widths[attribute] = max(
                        max_widths[attribute], len(str(signature[attribute]))
                    )
            max_widths['mc_name'] = 1
            for mc in macroclasses:
                max_widths['mc_name'] = max(
                    max_widths['mc_name'], len(str(macroclasses[mc]))
                )
            for signature in signatures:
                sig_text = '%s %s%s %s' % (
                    str(signature['macroclass_id']).ljust(
                        max_widths['macroclass_id']), sep,
                    str(macroclasses[signature.macroclass_id]).ljust(
                        max_widths['mc_name']), sep
                )
                for attribute in ['class_id', 'class_name']:
                    sig_text += '%s %s' % (
                        str(signature[attribute]).ljust(max_widths[attribute]),
                        sep
                    )
                training_options.append(str(sig_text))
            return training_options

        def display_training_file():
            global training_path, training_data, training_map
            if training_path is not None:
                gpkg_path = cfg.default_signature_catalog.geometry_file
                json_path = cfg.temp.temporary_file_path(name_suffix='.json')
                self.rs.shared_tools.vector_to_json(gpkg_path, json_path)
                with open(json_path, 'r') as f:
                    geojson_data = json.load(f)
                if training_data is None:
                    training_data = GeoJSON(
                        data=geojson_data, name='Training',
                        style={
                            'opacity': 1, 'fillOpacity': 1,
                            'color': '#ffffff', 'fillColor': '#000000',
                            'dashArray': 3, 'weight': 1
                        },
                    )
                    # noinspection PyUnresolvedReferences
                    training_map.add(training_data)
                else:
                    training_data.data = geojson_data

        # noinspection PyUnresolvedReferences
        def new_training_file(_):
            global training_path, input_bandset, browser_selector_val
            global new_file_text
            nonlocal training_label, training_input_text
            browser_selector_val.value = 2
            new_path = str('%s/%s' % (
                browser_dir, new_file_text.value
            )).replace('//', '/')
            if not new_path.endswith(cfg.scpx_suffix):
                new_path += cfg.scpx_suffix
            input_bandset = cfg.default_catalog.current_bandset
            training_label.value = (
                f"<div style='color:white; weight:bold; background: #5a5a5a'>"
                f"<b>ROI & Signature list (band set {input_bandset})</b></div>"
            )
            bandset_x = cfg.default_catalog.get(input_bandset)
            band_count = bandset_x.get_band_count()
            if band_count == 0:
                error_message('Bandset is empty.')
                return
            cfg.default_signature_catalog = (
                self.rs.spectral_signatures_catalog(
                    bandset=cfg.default_catalog.get(input_bandset)
                )
            )
            training_path = new_path
            training_input_text.value = training_path
            new_file_text.value = ''

        # noinspection PyUnresolvedReferences
        def open_training_file(input_files):
            global training_path, input_bandset, browser_selector_val
            global selected_file_paths, new_file_text
            nonlocal training_input_text
            browser_selector_val.value = 2
            selected_file_paths.value = str(input_files).replace(
                "'", '').replace('[', '').replace(']', '')
            files = selected_file_paths.value.split(',')
            for file in files:
                if len(file) > 0 and file.endswith(cfg.scpx_suffix):
                    try:
                        input_bandset = cfg.default_catalog.current_bandset
                        if cfg.default_signature_catalog is None:
                            (cfg.default_signature_catalog
                             ) = self.rs.spectral_signatures_catalog(
                                bandset=cfg.default_catalog.get(input_bandset)
                            )
                        cfg.default_signature_catalog.load(file_path=file)
                        training_path = file.strip()
                        training_input_text.value = training_path
                        error_message('')
                    except Exception as err:
                        str(err)
                        error_message('Select a file.')
                else:
                    error_message('Select a file.')
            new_file_text.value = ''
            # display geojson data
            display_training_file()
            refresh_training_list()

        def new_training_click(_):
            global caller_function
            caller_function = new_training_file
            activate_browser(None)

        def open_training_click(_):
            global caller_function
            caller_function = open_training_file
            activate_browser(None)

        def activate_classification_preview(_):
            global roi_grow, roi_manual, training_map, preview_point, aoi_draw
            global classification_preview_pointer
            training_map.default_style = {'cursor': 'crosshair'}
            if classification_preview_pointer is True:
                classification_preview_pointer = False
                training_map.default_style = {'cursor': 'default'}
            else:
                classification_preview_pointer = True
                roi_grow = False
                roi_manual = False
                preview_point = None
                aoi_draw = False

        def activate_roi(_):
            global roi_grow, roi_manual, training_map, aoi_draw
            global classification_preview_pointer
            training_map.default_style = {'cursor': 'crosshair'}
            if roi_grow is True:
                roi_grow = False
                training_map.default_style = {'cursor': 'default'}
            else:
                roi_grow = True
                roi_manual = False
                classification_preview_pointer = False
                aoi_draw = False

        def activate_manual_roi(_):
            global roi_grow, roi_manual, training_map, aoi_draw
            global classification_preview_pointer
            training_map.default_style = {'cursor': 'crosshair'}
            if roi_manual is True:
                roi_manual = False
                training_map.default_style = {'cursor': 'default'}
            else:
                roi_manual = True
                roi_grow = False
                classification_preview_pointer = False
                aoi_draw = False

        # create RGB color composite
        # noinspection PyUnresolvedReferences
        def create_rgb_color_composite(color_composite):
            global rgb_layer_group, old_layer, training_map
            bandset_x = cfg.default_catalog.get()
            band_count = bandset_x.get_band_count()
            if band_count == 0:
                error_message('Empty band set.')
                return False
            composite = str(color_composite).split(',')
            if len(composite) == 1:
                composite = str(color_composite).split('-')
            if len(composite) == 1:
                composite = str(color_composite).split(';')
            if len(composite) == 1:
                composite = str(color_composite)
            if len(composite) < 3:
                error_message('Create rgb color composite.')
                return False
            try:
                if int(composite[0]) > band_count:
                    composite[0] = band_count
                if int(composite[1]) > band_count:
                    composite[1] = band_count
                if int(composite[2]) > band_count:
                    composite[2] = band_count
            except Exception as err:
                str(err)
                error_message('Create band set.')
                return False
            bandset_bbox = cfg.default_catalog.get_bbox()
            if bandset_bbox is None:
                error_message('Error in band set bounding box.')
            else:
                # remove layer
                if rgb_layer_group is None:
                    add_layer_group = True
                else:
                    add_layer_group = False
                # add layer
                jpg_path = cfg.default_catalog.create_jpg(
                    bands=[int(composite[0]), int(composite[1]),
                           int(composite[2])]
                )
                bandset_crs = cfg.default_catalog.get().crs
                bandset_bottom_left = (
                    self.rs.shared_tools.project_point_coordinates(
                        bandset_bbox[0], bandset_bbox[3], bandset_crs, 4326)
                )
                bandset_top_right = (
                    self.rs.shared_tools.project_point_coordinates(
                        bandset_bbox[2], bandset_bbox[1], bandset_crs, 4326)
                )
                if bandset_bottom_left is not None:
                    rgb_left, rgb_bottom = bandset_bottom_left
                else:
                    error_message('Create band set.')
                    return False
                if bandset_top_right is not None:
                    rgb_right, rgb_top = bandset_top_right
                else:
                    error_message('Create band set.')
                    return False
                rgb_raster_name = 'Virtual band set'
                with open(jpg_path, 'rb') as f:
                    base64_image = base64.b64encode(f.read()).decode('utf-8')
                layer = ImageOverlay(
                            url=f'data:image/jpeg;base64,{base64_image}',
                            bounds=((rgb_bottom, rgb_left),
                                    (rgb_top, rgb_right))
                        )
                if rgb_layer_group is None:
                    rgb_layer_group = LayerGroup(
                        layers=[layer], name=rgb_raster_name
                    )
                    old_layer = layer
                else:
                    # noinspection PyUnresolvedReferences
                    rgb_layer_group.substitute(old_layer, layer)
                    old_layer = layer
                if add_layer_group:
                    training_map.add(rgb_layer_group)
                if rgb_layer_group in training_map.layers:
                    pass
            return True

        def zoom_to_bandset(_):
            global training_map
            bandset_bbox = cfg.default_catalog.get_bbox()
            if bandset_bbox is None:
                error_message('Error in band set bounding box.')
            else:
                bandset_crs = cfg.default_catalog.get().crs
                bandset_bottom_left = (
                    self.rs.shared_tools.project_point_coordinates(
                        bandset_bbox[0], bandset_bbox[3], bandset_crs, 4326
                    )
                )
                bandset_top_right = (
                    self.rs.shared_tools.project_point_coordinates(
                        bandset_bbox[2], bandset_bbox[1], bandset_crs, 4326
                    )
                )
                if bandset_bottom_left is not None:
                    rgb_left, rgb_bottom = bandset_bottom_left
                else:
                    error_message('Create band set.')
                    return False
                if bandset_top_right is not None:
                    rgb_right, rgb_top = bandset_top_right
                else:
                    error_message('Create band set.')
                    return False
                training_map.center = (
                    (rgb_top + rgb_bottom) / 2, (rgb_right + rgb_left) / 2
                )
                training_map.zoom = 10

        def rgb_options(change):
            nonlocal rgb_combo
            composite = change['new']
            options = list(rgb_combo.options)
            if composite not in options and len(composite) > 0:
                try:
                    check = create_rgb_color_composite(composite)
                    if check is True:
                        options.append(composite)
                        rgb_combo.options = options
                        rgb_combo.value = composite
                    else:
                        return False
                except Exception as err:
                    str(err)
                    error_message('Error creating RGB composite.')
                    return False
            else:
                create_rgb_color_composite(rgb_combo.value)

        new_training_button = widgets.Button(
            description='+', tooltip='Create training input',
            button_style='success', icon='file',
            layout=widgets.Layout(width='15%')
        )
        new_training_button.on_click(new_training_click)
        open_training_button = widgets.Button(
            tooltip='Open training input', button_style='info',
            icon='folder-open', layout=widgets.Layout(width='15%')
        )
        open_training_button.on_click(open_training_click)
        training_input_text = widgets.Text(placeholder='Training input',
                                           layout=widgets.Layout(width='68%'))
        delete_signature_button = widgets.Button(
            tooltip='Delete ROI signature', button_style='danger',
            icon='minus', layout=widgets.Layout(width='40px')
        )
        delete_signature_button.on_click(delete_signature_click)
        plot_signature_button = widgets.Button(
            tooltip='Spectral signature plot', button_style='info',
            icon='area-chart', layout=widgets.Layout(width='40px')
        )
        plot_signature_button.on_click(plot_signature_click)
        # training multiple select
        training_dock = widgets.SelectMultiple(
            options=[], rows=10, layout=widgets.Layout(width='80%')
        )
        training_dock.observe(selection_training_change, names='index')
        """
        training_label = widgets.Label(
            value='ROI & Signature list', layout=widgets.Layout(width='98%'),
            style=label_style
        )
        """
        training_label = widgets.HTML(
            value=f"<div style='color:white; weight:bold; "
                  f"background: #5a5a5a'><b>ROI & Signature list</b></div>"
        )
        training_label_2 = widgets.Label(
            value='MC ID │ MC name │ C ID │ C name',
            layout=widgets.Layout(width='85%')
        )
        """
        mc_id_label = widgets.Label(
            value='MC ID', style=label_style,
            layout=widgets.Layout(width='20%')
        )
        """
        mc_id_label = widgets.HTML(
            value=f"<div style='color:white; weight:bold; "
                  f"background: #5a5a5a'><b>MC ID</b></div>"
        )
        mc_id_input = widgets.BoundedIntText(
            value=1, min=1, max=10000, step=1, continuous_update=True,
            tooltip='Macroclass ID', layout=widgets.Layout(width='20%')
        )
        mc_name_input_text = widgets.Text(
            placeholder='Macroclass name', value='Macroclass 1',
            layout=widgets.Layout(width='50%')
        )
        mc_color_input = widgets.ColorPicker(concise=True, value='#96d800')
        mc_color_input_box = widgets.VBox(
            [mc_color_input], layout=widgets.Layout(width='8%')
        )
        """
        c_id_label = widgets.Label(
            value='C ID', style=label_style, layout=widgets.Layout(width='20%')
        )
        """
        c_id_label = widgets.HTML(
            value=f"<div style='color:white; weight:bold; "
                  f"background: #5a5a5a'><b>C ID</b></div>"
        )
        c_id_input = widgets.BoundedIntText(
            value=1, min=1, max=10000, step=1, continuous_update=True,
            tooltip='Class ID', layout=widgets.Layout(width='20%')
        )
        c_name_input_text = widgets.Text(
            placeholder='Class name', value='Class 1',
            layout=widgets.Layout(width='50%')
        )
        c_color_input = widgets.ColorPicker(concise=True, value='#96d800')
        c_color_input_box = widgets.VBox(
            [c_color_input], layout=widgets.Layout(width='8%')
        )
        save_training_button = widgets.Button(
            tooltip='Save to training input', button_style='danger',
            icon='floppy-o', layout=widgets.Layout(width='15%')
        )
        save_training_button.on_click(save_training_click)

        """ Working toolbar """

        zoom_to_bandset_button = widgets.Button(
            icon='search', layout=widgets.Layout(width='5%'),
            tooltip='Zoom to bandset', button_style='danger'
        )
        zoom_to_bandset_button.on_click(zoom_to_bandset)
        """
        region_growing_rgb_label = widgets.Label(
            value='RGB=', style=label_style, layout=widgets.Layout(width='5%')
        )
        """
        region_growing_rgb_label = widgets.HTML(
            value=f"<div style='color:white; weight:bold; "
                  f"background: #5a5a5a'><b>RGB=</b></div>"
        )
        rgb_combo = widgets.Combobox(
            placeholder='-', options=['3-2-1', '4-3-2', '7-3-2'],
            tooltip='Press Enter', continuous_update=False,
            layout=widgets.Layout(width='10%')
        )
        rgb_combo.observe(rgb_options, 'value')
        region_growing_button = widgets.Button(
            description='+', layout=widgets.Layout(width='5%'),
            tooltip='Region growing', button_style='warning'
        )
        region_growing_button.on_click(activate_roi)
        region_manual_button = widgets.Button(
            tooltip='Polygon', layout=widgets.Layout(width='5%'),
            icon='square-o', button_style='warning'
        )
        region_manual_button.on_click(activate_manual_roi)
        """
        region_growing_dist_label = widgets.Label(
            value='Dist', style=label_style, layout=widgets.Layout(width='5%')
        )
        """
        region_growing_dist_label = widgets.HTML(
            value=f"<div style='color:white; weight:bold; "
                  f"background: #5a5a5a'><b>Dist</b></div>"
        )
        region_growing_dist = widgets.BoundedFloatText(
            value=0.01, min=0.000001, max=10000.0, step=0.01, tooltip='Dist',
            continuous_update=True, layout=widgets.Layout(width='7%')
        )
        """
        region_growing_min_label = widgets.Label(
            value='Min', style=label_style, layout=widgets.Layout(width='5%')
        )
        """
        region_growing_min_label = widgets.HTML(
            value=f"<div style='color:white; weight:bold; "
                  f"background: #5a5a5a'><b>Min</b></div>"
        )
        region_growing_min = widgets.BoundedIntText(
            value=60, min=1, max=1000, step=10, continuous_update=True,
            tooltip='Min', layout=widgets.Layout(width='7%')
        )
        """
        region_growing_max_label = widgets.Label(
            value='Max', style=label_style, layout=widgets.Layout(width='5%')
        )
        """
        region_growing_max_label = widgets.HTML(
            value=f"<div style='color:white; weight:bold; "
                  f"background: #5a5a5a'><b>Max</b></div>"
        )
        region_growing_max = widgets.BoundedIntText(
            value=100, min=1, max=10000, step=10, continuous_update=True,
            tooltip='Max', layout=widgets.Layout(width='7%')
        )
        """
        preview_label = widgets.Label(
            value='Preview', style=label_style,
            layout=widgets.Layout(width='7%')
        )
        """
        preview_label = widgets.HTML(
            value=f"<div style='color:white; weight:bold; "
                  f"background: #5a5a5a'><b>Preview</b></div>"
        )
        classification_preview_button = widgets.Button(
            description='+', layout=widgets.Layout(width='5%'),
            tooltip='Preview', button_style='success'
        )
        classification_preview_button.on_click(activate_classification_preview)
        """
        preview_size_max_label = widgets.Label(
            value='S', style=label_style, layout=widgets.Layout(width='5%')
        )
        """
        preview_size_max_label = widgets.HTML(
            value=f"<div style='color:white; weight:bold; "
                  f"background: #5a5a5a'><b>S</b></div>"
        )
        preview_size_input = widgets.BoundedIntText(
            value=200, min=1, max=10000, step=10, continuous_update=True,
            tooltip='Size', layout=widgets.Layout(width='7%')
        )
        """
        map_label = widgets.Label(
            value='Map', layout=widgets.Layout(width='97%'), style=label_style
        )
        """
        map_label = widgets.HTML(
            value=f"<div style='color:white; weight:bold; "
                  f"background: #5a5a5a'><b>Map</b></div>"
        )
        training_dock_rows = widgets.VBox([
            widgets.HBox(
                [new_training_button, open_training_button,
                 training_input_text]
            ), training_label,
            widgets.HBox(
                children=[training_label_2], layout=widgets.Layout(
                    display='flex', flex_flow='column', align_items='flex-end'
                )
            ),
            widgets.HBox(
                [widgets.VBox(
                    [delete_signature_button, plot_signature_button]
                ), training_dock]),
            widgets.HBox([mc_id_label, mc_id_input, mc_name_input_text,
                          mc_color_input_box]),
            widgets.HBox([c_id_label, c_id_input, c_name_input_text,
                          c_color_input_box]),
            widgets.HBox(
                children=[save_training_button],
                layout=widgets.Layout(
                    display='flex', flex_flow='column', align_items='flex-end'
                )
            )
        ], layout=widgets.Layout(width='25%'))
        working_toolbar_rows = widgets.HBox(
            [zoom_to_bandset_button,
             region_growing_rgb_label, rgb_combo,
             region_manual_button,
             region_growing_button, region_growing_dist,
             region_growing_dist_label,
             region_growing_min_label, region_growing_min,
             region_growing_max_label, region_growing_max,
             preview_label, classification_preview_button,
             preview_size_max_label, preview_size_input]
        )

    def full_interface(self):
        """Returns the full interface.

        Requires ipywidgets >= 8.0
        """
        global messages_row, browser_rows
        global training_map, plot_rows, plot_widget
        global training_dock_rows, working_toolbar_rows
        global browser_selector_val
        global bandset_rows, download_products_rows, bandcalc_rows
        global import_tool_rows, classification_rows
        global plot_button, main_button, map_button
        global clip_raster_rows, image_conversion_rows, masking_rows
        global mosaic_bandsets_rows, reproject_rows, split_raster_rows
        global stack_raster_rows, vector_to_raster_rows
        global combination_rows, dilation_rows, erosion_rows, sieve_rows
        global neighbor_rows, pca_rows, accuracy_rows
        global classification_report_rows, raster_to_vector_rows
        global cross_classification_rows, reclassification_rows
        global selected_interface_style, unselected_interface_style

        """ Main tabs """

        """ Basic tools """
        tab_basic_tools = widgets.Tab(children=[import_tool_rows],
                                      titles=['Import signatures'])

        """ Preprocessing """
        tab_preprocessing = widgets.Tab(children=[
            clip_raster_rows, image_conversion_rows, masking_rows,
            mosaic_bandsets_rows, reproject_rows, split_raster_rows,
            stack_raster_rows, vector_to_raster_rows
        ], titles=[
            'Clip raster bands', 'Image conversion', 'Masking bands',
            'Mosaic of band sets', 'Reproject raster bands',
            'Split raster bands', 'Stack raster bands', 'Vector to raster'
        ])

        """ Band processing """
        tab_band_processing = widgets.Tab(children=[
            classification_rows, combination_rows, dilation_rows, erosion_rows,
            sieve_rows, neighbor_rows, pca_rows
        ], titles=[
            'Classification', 'Combination', 'Dilation', 'Erosion',
            'Sieve', 'Neighbor', 'PCA'
        ])

        """ Postprocessing """
        tab_postprocessing = widgets.Tab(
            children=[
                accuracy_rows, classification_report_rows,
                raster_to_vector_rows, cross_classification_rows,
                reclassification_rows
            ], titles=[
                'Accuracy', 'Classification report',
                'Classification to vector', 'Cross classification',
                'Reclassification'
            ]
        )

        tab = widgets.Tab(children=[
            bandset_rows, download_products_rows, tab_basic_tools,
            tab_preprocessing, tab_band_processing, tab_postprocessing,
            bandcalc_rows
        ], layout=widgets.Layout(width='99%'),
            titles=['Band set', 'Download Products', 'Basic tools',
                    'Preprocessing', 'Band Processing', 'Postprocessing',
                    'Band calc']
        )

        main_tabs = widgets.VBox([tab], layout=widgets.Layout(width='100%'))
        self.reset_map()
        training_map_rows = widgets.HBox([
            training_dock_rows,
            widgets.VBox([working_toolbar_rows, training_map],
                         layout=widgets.Layout(width='75%'))]
        )
        # stack interface and file browser
        try:
            stack = widgets.Stack(
                [browser_rows, main_tabs, training_map_rows, plot_rows],
                selected_index=1, layout=widgets.Layout(width='100%')
            )
        except Exception as error_stack:
            str(error_stack)
            # in case of ipywidgets < 8
            stack = widgets.Tab(
                children=[browser_rows, main_tabs, training_map_rows,
                          plot_rows],
                selected_index=1, layout=widgets.Layout(width='99%'),
            )
        try:
            widgets.jslink((browser_selector_val, 'value'),
                           (stack, 'selected_index'))
        except Exception as error_jslink:
            str(error_jslink)
        main_button = widgets.Button(description='Main interface',
                                     icon='table')
        main_button.on_click(self._main_button_click)
        main_button.style = selected_interface_style
        map_button = widgets.Button(description='Map', icon='globe')
        map_button.on_click(self._map_button_click)
        map_button.style = unselected_interface_style
        plot_button = widgets.Button(description='Signature plot',
                                     icon='area-chart')
        plot_button.on_click(self._plot_button_click)
        plot_button.style = unselected_interface_style
        display(
            widgets.VBox([
                widgets.HBox([main_button, map_button, plot_button],
                             layout=widgets.Layout(width='100%')),
                messages_row, stack
            ], layout=widgets.Layout(width='100%'))
        )

    def classification_training_interface(self):
        """Returns the bandset interface.
        """
        global messages_row, bandset_rows, browser_rows, classification_rows
        global map_label, training_map
        global training_dock_rows, working_toolbar_rows
        self.reset_map()
        training_map_rows = widgets.HBox([
            training_dock_rows,
            widgets.VBox([working_toolbar_rows, training_map],
                         layout=widgets.Layout(width='75%'))]
        )
        display(
            widgets.VBox([
                bandset_rows, classification_rows, messages_row,
                map_label, training_map_rows, plot_rows, browser_rows
            ], layout=widgets.Layout(width='100%'))
        )

    """ Bandset tools """

    @staticmethod
    def bandset_interface():
        """Returns the bandset interface.
        """
        global messages_row, bandset_rows, browser_rows
        display(
            widgets.VBox([messages_row, bandset_rows, browser_rows],
                         layout=widgets.Layout(width='100%'))
        )

    """ Download products tools """

    def download_interface(self):
        """Returns the download products interface.
        """
        global messages_row, download_products_rows, training_map, browser_rows
        global map_label
        self.reset_map()
        display(
            widgets.VBox([messages_row, download_products_rows, map_label,
                          training_map, browser_rows],
                         layout=widgets.Layout(width='100%'))
        )

    """ Basic tools """

    @staticmethod
    def import_interface():
        """Returns the import tool interface.
        """
        global import_tool_rows
        display(import_tool_rows)

    """ Preprocessing tools """

    """ Band processing tools """

    @staticmethod
    def classification_interface():
        global classification_rows
        display(classification_rows)

    """ Postprocessing tools """

    @staticmethod
    def plot_interface():
        """Returns the plot interface.
        """
        global plot_rows
        display(plot_rows)

    def map_interface(self):
        """Returns the training map interface.
        """
        global training_map
        self.reset_map()
        display(training_map)

    @staticmethod
    def messages_interface():
        """Returns the messages interface.
        """
        global messages_row
        display(messages_row)

    # select main page
    @staticmethod
    def _main_button_click(_):
        global browser_selector_val
        global plot_button, main_button, map_button
        browser_selector_val.value = 1
        if plot_button is not None:
            main_button.style = selected_interface_style
            map_button.style = unselected_interface_style
            plot_button.style = unselected_interface_style

    # select map page
    @staticmethod
    def _map_button_click(_):
        global browser_selector_val
        global plot_button, main_button, map_button
        browser_selector_val.value = 2
        if plot_button is not None:
            map_button.style = selected_interface_style
            main_button.style = unselected_interface_style
            plot_button.style = unselected_interface_style

    # select plot page
    @staticmethod
    def _plot_button_click(_):
        global browser_selector_val
        global plot_button, main_button, map_button
        browser_selector_val.value = 3
        if plot_button is not None:
            plot_button.style = selected_interface_style
            main_button.style = unselected_interface_style
            map_button.style = unselected_interface_style

    # finish manual roi
    def finish_roi(self, _):
        global last_region_path, close_polygon, r_point_coordinates
        global training_map
        close_polygon = True
        bandset_crs = cfg.default_catalog.get().crs
        r_point_coordinates.append(r_point_coordinates[0])
        last_region_path = self.rs.shared_tools.coordinates_to_polygon(
            r_point_coordinates, bandset_crs
        )
        training_map.default_style = {'cursor': 'default'}

    # ROI map interaction
    # noinspection PyUnresolvedReferences
    def roi_interaction(self, **kwargs):
        global rectangle, points, aoi_draw, polygon_coordinates
        global input_bandset
        global roi_grow, roi_manual, last_region_path, temporary_roi
        global r_point_coordinates, close_polygon, training_map
        global classification_preview_pointer, preview_point
        global region_growing_max, region_growing_dist, region_growing_min
        global ul_x, ul_y, lr_x, lr_y
        if kwargs.get('type') == 'click' and roi_manual is True:
            if kwargs.get('type') == 'click':
                if 'coordinates' in kwargs:
                    if close_polygon:
                        r_point_coordinates = []
                        polygon_coordinates = []
                        close_polygon = False
                        points.clear()
                    coordinates = kwargs.get('coordinates')
                    r_point_coord = (
                        self.rs.shared_tools.project_point_coordinates(
                            coordinates[1], coordinates[0], 4326,
                            cfg.default_catalog.get(input_bandset).crs
                        )
                    )
                    r_point_coordinates.append(r_point_coord)
                    polygon_coordinates.append(
                        [coordinates[1], coordinates[0]]
                    )
                    if len(polygon_coordinates) > 0:
                        poly_layer_coordinates = list(polygon_coordinates)
                        poly_layer_coordinates.append(
                            poly_layer_coordinates[0]
                        )
                        geojson_data = {
                            'type': 'Feature', 'properties': {},
                            'geometry': {
                                'type': 'Polygon',
                                'coordinates': [poly_layer_coordinates]
                            }
                        }
                        if temporary_roi is None:
                            temporary_roi = GeoJSON(
                                data=geojson_data, name='temporary ROI',
                                style={
                                    'opacity': 1, 'fillOpacity': 0.5,
                                    'color': '#56e9e5',
                                    'fillColor': '#ffaa00'
                                },
                            )
                            training_map.add(temporary_roi)
                        else:
                            temporary_roi.data = geojson_data
        elif kwargs.get('type') == 'click' and roi_grow is True:
            if 'coordinates' in kwargs:
                coordinates = kwargs['coordinates']
                roi_point_coord = (
                    self.rs.shared_tools.project_point_coordinates(
                        coordinates[1], coordinates[0], 4326,
                        cfg.default_catalog.get(input_bandset).crs
                    )
                )
                # TODO implement band number selection
                roi_band_number = 1
                last_region_path = (
                    self.rs.shared_tools.region_growing_polygon(
                        coordinate_x=roi_point_coord[0],
                        coordinate_y=roi_point_coord[1],
                        input_bands=input_bandset,
                        band_number=roi_band_number,
                        max_width=region_growing_max.value,
                        max_spectral_distance=region_growing_dist.value,
                        minimum_size=region_growing_min.value,
                        bandset_catalog=cfg.default_catalog
                    )
                )
                json_path = cfg.temp.temporary_file_path(
                    name_suffix='.json'
                )
                self.rs.shared_tools.vector_to_json(last_region_path,
                                                    json_path)
                with open(json_path, 'r') as f:
                    geojson_data = json.load(f)
                if temporary_roi is None:
                    temporary_roi = GeoJSON(
                        data=geojson_data, name='temporary ROI',
                        style={
                            'opacity': 1, 'fillOpacity': 0.5,
                            'color': '#56e9e5', 'fillColor': '#ffaa00'
                        },
                    )
                    training_map.add(temporary_roi)
                else:
                    temporary_roi.data = geojson_data
                self._reset_message()
        elif kwargs.get('type') == 'click' and aoi_draw is True:
            if 'coordinates' in kwargs:
                points.append(kwargs['coordinates'])
                if len(points) == 2:
                    if rectangle is not None:
                        try:
                            training_map.remove(rectangle)
                        except Exception as error_rectangle:
                            str(error_rectangle)
                    rectangle = Rectangle(
                        bounds=(points[0], points[1]),
                        color='#fca45d', fill_color='#fca45d',
                        fill_opacity=0.5, name='AOI'
                        )
                    ul_x.value = str(min([points[0][1], points[1][1]]))
                    ul_y.value = str(max([points[0][0], points[1][0]]))
                    lr_x.value = str(max([points[0][1], points[1][1]]))
                    lr_y.value = str(min([points[0][0], points[1][0]]))
                    training_map.add(rectangle)
                    points.clear()
        elif (kwargs.get('type') == 'click'
              and classification_preview_pointer is True):
            if 'coordinates' in kwargs:
                coordinates = kwargs['coordinates']
                preview_point = (
                    self.rs.shared_tools.project_point_coordinates(
                        coordinates[1], coordinates[0], 4326,
                        cfg.default_catalog.get(input_bandset).crs
                    )
                )
                # perform classification preview
                self._classification_run(None)
                preview_point = None

    def reset_map(self):
        global training_map, layers_control
        global classification_preview_group, old_classification_preview
        # training map
        training_map = Map(center=(10, 0), zoom=2, scroll_wheel_zoom=True,
                           default_style={'cursor': 'default'})
        # create an empty classification preview to be the first layer
        width, height = 3, 3
        empty_image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        image_buffer = io.BytesIO()
        empty_image.save(image_buffer, format='png')
        image_buffer.seek(0)
        bounds = ((-90.0, -180.0), (90, 180))
        base64_empty_image = base64.b64encode(
            image_buffer.getvalue()).decode()
        transparent_image = ImageOverlay(
            url=f'data:image/png;base64,{base64_empty_image}',
            bounds=bounds, opacity=0
        )
        classification_preview_group = LayerGroup(
            layers=[transparent_image], name='Classification preview'
        )
        training_map.add(classification_preview_group)
        old_classification_preview = transparent_image
        finish_button = widgets.Button(
            description='Finish', button_style='warning',
            tooltip='Finish polygon',
            icon='square-o', layout=widgets.Layout(width='100px')
        )
        finish_button.on_click(self.finish_roi)
        # noinspection SpellCheckingInspection
        finish_roi_control = WidgetControl(
            widget=finish_button, position='topleft'
        )
        training_map.add(finish_roi_control)
        # noinspection SpellCheckingInspection
        layers_control = LayersControl(position='topright')
        training_map.add(layers_control)
        training_map.on_interaction(self.roi_interaction)


def update_progress(
        step=None, message=None, process=None, percentage=None,
        elapsed_time=None, previous_step=None, start=None, end=None,
        ping=0
):
    global tot_remaining
    progress_symbols = ['○', '◔', '◑', '◕', '⬤', '⚙']
    colon = [' ◵ ', ' ◷ ']
    if start:
        text = '<div>{} {} {}</div>'.format(
            message, progress_symbols[-1], colon[ping]
        )
        try:
            msg_label.value = text
        except Exception as err:
            str(err)
        if process is not None:
            try:
                msg_label_main.value = f'<div>{process}</div>'
                progress_widget.value = step
            except Exception as err:
                str(err)
    elif end:
        if elapsed_time is not None:
            e_time = (
                '(elapsed: {}min{}sec)'.format(
                    int(elapsed_time / 60), str(
                        int(
                            60 * ((elapsed_time / 60) - int(
                                elapsed_time / 60
                            ))
                        )
                    ).rjust(2, '0')
                )
            )
        else:
            e_time = ''
        text = '<div>{} - {}</div>'.format(progress_symbols[-2], e_time)
        try:
            msg_label.value = text
        except Exception as err:
            str(err)
        try:
            if process is not None:
                msg_label_main.value = f'<div>{process}</div>'
                progress_widget.value = step
        except Exception as err:
            str(err)
    else:
        if not percentage and percentage is not None:
            percentage = -25
        if elapsed_time is not None:
            e_time = (
                'elapsed: {}min{}sec'.format(
                    int(elapsed_time / 60), str(
                        int(
                            60 * ((elapsed_time / 60) - int(
                                elapsed_time / 60
                            ))
                        )
                    ).rjust(2, '0')
                )
            )
            if previous_step < step:
                try:
                    remaining_time = (
                            (100 - int(step)) * elapsed_time / int(step)
                    )
                    minutes = int(remaining_time / 60)
                    seconds = round(
                        60 * ((remaining_time / 60)
                              - int(remaining_time / 60))
                    )
                    if seconds == 60:
                        seconds = 0
                        minutes += 1
                    remaining = '; remaining: {}min{}sec'.format(
                        minutes, str(seconds).rjust(2, '0')
                    )
                    tot_remaining = remaining
                except Exception as err:
                    str(err)
                    remaining = ''
            else:
                remaining = tot_remaining
        else:
            e_time = ''
            remaining = ''
        try:
            text = '<div>{} {} - {}{} {}</div>'.format(
                message, progress_symbols[int(percentage / 25)], e_time,
                remaining, colon[ping]
            )
            msg_label.value = text
            if process is not None:
                msg_label_main.value = f'<div>{process}</div>'
                progress_widget.value = step
        except Exception as err:
            str(err)
            if process is not None:
                try:
                    msg_label_main.value = f'<div>{process}</div>'
                except Exception as err:
                    str(err)
