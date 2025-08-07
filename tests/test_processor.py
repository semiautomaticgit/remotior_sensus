from unittest import TestCase
import numpy as np
from pathlib import Path
import multiprocessing
from remotior_sensus.core import processor
import remotior_sensus
# noinspection PyProtectedMember
from remotior_sensus.core.multiprocess_manager import (
    _calculate_block_size, _compute_raster_pieces
)
from remotior_sensus.core.processor_functions import (
    reclassify_raster, band_calculation
)


class TestProcessor(TestCase):

    def test_processor(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=1000, log_level=10
        )
        cfg = rs.configurations
        cfg.logger.log.debug('>>> test reclassification')
        process_parameters = [
            1, cfg.temp, 500, cfg.gdal_path, multiprocessing.Manager().Queue(),
            cfg.refresh_time, cfg.memory_unit_array_12, cfg.log_level,
            None, None
        ]
        data_path = Path(__file__).parent / 'data'
        raster_path = str(data_path / 'S2_2020-01-01' / 'S2_B02.tif')
        calc_datatype = [np.float32]
        input_nodata_as_value = False
        use_value_as_nodata = [1]
        (raster_x_size, raster_y_size, block_size_x, block_size_y,
         list_range_x, list_range_y, tot_blocks,
         number_of_bands) = _calculate_block_size(
            raster_path, 1, cfg.memory_unit_array_12, 0,
            available_ram=100
        )
        # compute raster pieces
        pieces = _compute_raster_pieces(
            raster_x_size, raster_y_size, block_size_x, block_size_y,
            list_range_y, 1, separate_bands=False,
            unique_section=True
        )
        # use raster files directly (not vrt)
        input_parameters = [[[raster_path]], calc_datatype, None,
                            pieces[0], None, None,
                            use_value_as_nodata, None,
                            input_nodata_as_value, None,
                            0, None, True]
        temp = cfg.temp.temporary_file_path(name_suffix=cfg.tif_suffix)
        compress = cfg.raster_compression
        compress_format = cfg.raster_compression_format
        any_nodata_mask = False
        output_nodata_value = -10
        keep_output_array = True
        keep_output_argument = False
        output_parameters = [[temp], [cfg.raster_data_type],
                             compress, compress_format, any_nodata_mask,
                             [output_nodata_value], [1],
                             keep_output_array, keep_output_argument]
        dtype_list = [(cfg.old_value, 'U1024'), (cfg.new_value, 'U1024')]
        function = reclassify_raster
        function_argument = np.array(
            ([('1', '2'), ('1110', '3')]), dtype=dtype_list
        )
        function_variable = cfg.variable_raster_name
        (output_array_list, out_files, proc_error,
         logger) = processor.function_initiator(
            process_parameters, input_parameters, output_parameters,
            function, [function_argument], [function_variable],
            False, False, False,
            False,
        )
        self.assertEqual(output_array_list[0][0][0, 0], output_nodata_value)
        self.assertTrue(rs.files_directories.is_file(out_files[0][0]))
        cfg.logger.log.debug('>>> test band calc')
        function = band_calculation
        function_argument = '_array_function_placeholder * 2'
        function_variable = cfg.variable_raster_name
        (output_array_list, out_files, proc_error,
         logger) = processor.function_initiator(
            process_parameters, input_parameters, output_parameters,
            function, [function_argument], [function_variable],
            False, False, False,
            False
        )
        self.assertEqual(output_array_list[0][0][0, 0], output_nodata_value)
        self.assertTrue(rs.files_directories.is_file(out_files[0][0]))

        # clear temporary directory
        rs.close()
