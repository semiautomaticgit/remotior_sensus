from pathlib import Path
from unittest import TestCase

import remotior_sensus


class TestRasterZonalStats(TestCase):

    def test_raster_zonal_stats(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=1000, log_level=10
        )
        cfg = rs.configurations
        data_path = Path(__file__).parent / 'data'
        v = str(data_path / 'files' / 'roi.gpkg')
        r = str(data_path / 'S2_2020-01-01' / 'S2_B02.tif')
        temp = cfg.temp.temporary_file_path(name_suffix=cfg.csv_suffix)
        cfg.logger.log.debug('>>> test raster_zonal_stats')
        stats = rs.raster_zonal_stats(
            raster_path=r, reference_path=v, vector_field='class',
            stat_names=['Sum', 'Mean'], output_path=temp,
        )
        self.assertTrue(rs.files_directories.is_file(stats.path))
        self.assertTrue(stats.extra['table'] is not None)
        temp = cfg.temp.temporary_file_path(name_suffix=cfg.csv_suffix)
        cfg.logger.log.debug('>>> test raster_zonal_stats percentile')
        stats2 = rs.raster_zonal_stats(
            raster_path=r, reference_path=v, vector_field='class',
            stat_names=['Percentile', 'Max', 'Min'], stat_percentile=[1, 99],
            output_path=temp,
        )
        self.assertTrue(rs.files_directories.is_file(stats2.path))
        self.assertTrue(stats2.extra['table'] is not None)

        cfg.logger.log.debug('>>> test raster_zonal_stats raster')
        raster = rs.vector_to_raster(vector_path=v, align_raster=r,
                                     vector_field='class')
        temp_2 = cfg.temp.temporary_file_path(name_suffix=cfg.csv_suffix)
        stats = rs.raster_zonal_stats(
            raster_path=r, reference_path=raster.path,
            stat_names=['Sum', 'Mean'], output_path=temp_2,
        )
        self.assertTrue(rs.files_directories.is_file(stats.path))
        self.assertTrue(stats.extra['table'] is not None)

        # clear temporary directory
        rs.close()
