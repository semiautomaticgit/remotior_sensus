from unittest import TestCase

import remotior_sensus
from remotior_sensus.tools import band_calc
from remotior_sensus.util import files_directories


class TestBandCalc(TestCase):

    def test_band_calc(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=1000, log_level=10
            )
        cfg = rs.configurations
        cfg.logger.log.debug('test')
        catalog = rs.bandset_catalog()
        file_list = ['S2_2020-01-01/S2_B02.tif', 'S2_2020-01-01/S2_B03.tif',
                     'S2_2020-01-01/S2_B04.tif', 'S2_2020-01-01/S2_B08.tif']
        date = '2021-01-01'
        root_directory = './data'
        catalog.create_bandset(
            file_list, wavelengths=['Sentinel-2'], date=date, bandset_number=1,
            root_directory=root_directory
            )
        raster_list = ['./data/S2_2020-01-02/S2_B02.tif',
                       './data/S2_2020-01-02/S2_B03.tif']
        name_list = ['raster1', 'raster2']
        band_names = band_calc._band_names_alias(
            raster_list, name_list, catalog
        )
        cfg.logger.log.debug('>>> test bandsets_iterator')
        expression = '%s[2021-01-01, 2:4]' % cfg.forbandsets
        band_list, error_message = band_calc._bandsets_iterator(
            expression, catalog
        )
        self.assertEqual(band_list[0], 1)
        expression = '%s[2021-01-01:2021-01-02]S' % cfg.forbandsets
        band_list, error_message = band_calc._bandsets_iterator(
            expression, catalog
        )
        self.assertEqual(band_list[0], 1)
        expression = '%s[1, 2021-01-02:2021-01-04]S' % cfg.forbandsets
        band_list, error_message = band_calc._bandsets_iterator(
            expression, catalog
        )
        self.assertEqual(band_list[0], 1)
        expression = '%s[1:4, 2021-01-04]S' % cfg.forbandsets
        band_list, error_message = band_calc._bandsets_iterator(
            expression, catalog
        )
        self.assertEqual(band_list[0], 1)
        cfg.logger.log.debug('>>> test replace_operator_names')
        # all bands in current BandSet
        expression = '{}{}{}{}{}{}'.format(
            cfg.variable_band_quotes, cfg.variable_bandset_name,
            cfg.variable_current_bandset, cfg.variable_band_name,
            cfg.variable_all, cfg.variable_band_quotes
        )
        out_exp, error_message = band_calc._replace_operator_names(
            expression, catalog
        )
        self.assertTrue('axis' in out_exp)
        self.assertFalse(error_message)
        # band 1 of BandSets in range of dates
        expression = '%s%s{2021-01-01:2021-01-02}%s1%s' % (
            cfg.variable_band_quotes, cfg.variable_bandset_name,
            cfg.variable_band_name, cfg.variable_band_quotes)
        out_exp, error_message = band_calc._replace_operator_names(
            expression, catalog
        )
        self.assertTrue('axis' in out_exp)
        self.assertFalse(error_message)
        # percentile of all bands in current BandSet
        expression = 'percentile(%s%s%s%s%s%s, 80)' % (
            cfg.variable_band_quotes, cfg.variable_bandset_name,
            cfg.variable_current_bandset, cfg.variable_band_name,
            cfg.variable_all, cfg.variable_band_quotes)
        out_exp, error_message = band_calc._replace_operator_names(
            expression, catalog, bandset_number=1
        )
        self.assertTrue('axis' in out_exp)
        self.assertFalse(error_message)
        # band in all band sets
        expression = '%s%s%s%s1%s' % (
            cfg.variable_band_quotes, cfg.variable_bandset_name,
            cfg.variable_all, cfg.variable_band_name,
            cfg.variable_band_quotes)
        out_exp, error_message = band_calc._replace_operator_names(
            expression, catalog
        )
        self.assertTrue('axis' in out_exp)
        self.assertFalse(error_message)
        # percentile of band in all band sets
        expression = 'percentile(%s%s%s%s1%s, 80)' % (
            cfg.variable_band_quotes, cfg.variable_bandset_name,
            cfg.variable_all, cfg.variable_band_name,
            cfg.variable_band_quotes)
        out_exp, error_message = band_calc._replace_operator_names(
            expression, catalog
        )
        self.assertTrue('axis' in out_exp)
        self.assertFalse(error_message)
        # band in selected band sets
        expression = '%s%s{1,2,3}%s1%s' % (
            cfg.variable_band_quotes, cfg.variable_bandset_name,
            cfg.variable_band_name, cfg.variable_band_quotes)
        out_exp, error_message = band_calc._replace_operator_names(
            expression, catalog
        )
        self.assertTrue('axis' in out_exp)
        self.assertFalse(error_message)
        # percentile of band in selected band sets
        expression = 'percentile(%s%s{1,2,3}%s1%s, 80)' % (
            cfg.variable_band_quotes, cfg.variable_bandset_name,
            cfg.variable_band_name, cfg.variable_band_quotes)
        out_exp, error_message = band_calc._replace_operator_names(
            expression, catalog
        )
        self.assertTrue('axis' in out_exp)
        self.assertFalse(error_message)
        # band in selected band sets
        expression = '%s%s1%s%s%s' % (
            cfg.variable_band_quotes, cfg.variable_bandset_name,
            cfg.variable_band_name, cfg.variable_all, cfg.variable_band_quotes)
        out_exp, error_message = band_calc._replace_operator_names(
            expression, catalog
        )
        self.assertTrue('axis' in out_exp)
        self.assertFalse(error_message)
        # percentile of band in selected band sets
        expression = 'percentile(%s%s1%s%s%s, 80) %s ' \
                     'percentile(%s%s1%s%s%s, 80)' % (
                         cfg.variable_band_quotes, cfg.variable_bandset_name,
                         cfg.variable_band_name, cfg.variable_all,
                         cfg.variable_band_quotes, cfg.new_line,
                         cfg.variable_band_quotes, cfg.variable_bandset_name,
                         cfg.variable_band_name, cfg.variable_all,
                         cfg.variable_band_quotes)
        out_exp, error_message = band_calc._replace_operator_names(
            expression, catalog
        )
        self.assertTrue('axis' in out_exp)
        self.assertFalse(error_message)
        cfg.logger.log.debug('>>> test check expressions')
        # current BandSet
        expression = (
                cfg.variable_band_quotes + cfg.variable_bandset_name
                + cfg.variable_current_bandset
                + cfg.variable_band_name + '1' + cfg.variable_band_quotes)
        out_exp, out_name_list, error_message = band_calc._check_expression(
            expression_string=expression, raster_variables=band_names,
            bandset_catalog=catalog
        )
        self.assertEqual(
            out_exp[0][0],
            cfg.variable_band_quotes + cfg.variable_bandset_name + '1'
            + cfg.variable_band_name + '1' + cfg.variable_band_quotes
        )
        self.assertFalse(error_message)
        # expression alias
        expression = (
                cfg.variable_band_quotes + cfg.variable_ndvi_name
                + cfg.variable_band_quotes)
        out_exp, out_name_list, error_message = band_calc._check_expression(
            expression_string=expression, raster_variables=band_names,
            bandset_catalog=catalog
        )
        self.assertFalse(error_message)
        # iterate band sets
        expression = (
                cfg.forbandsets + '[1]S ' + cfg.new_line
                + cfg.variable_band_quotes + cfg.variable_bandset_name
                + cfg.variable_current_bandset + cfg.variable_band_name
                + '1' + cfg.variable_band_quotes)
        out_exp, out_name_list, error_message = band_calc._check_expression(
            expression_string=expression, raster_variables=band_names,
            bandset_catalog=catalog
        )
        self.assertEqual(
            out_exp[0][0],
            cfg.variable_band_quotes + cfg.variable_bandset_name + '1'
            + cfg.variable_band_name + '1' + cfg.variable_band_quotes
        )
        self.assertFalse(error_message)
        # forbandsets range of dates
        expression = (
                cfg.forbandsets + '[2021-01-01:2021-01-02]S '
                + cfg.new_line + cfg.variable_band_quotes
                + cfg.variable_bandset_name
                + cfg.variable_current_bandset + cfg.variable_band_name
                + '1' + cfg.variable_band_quotes)
        out_exp, out_name_list, error_message = band_calc._check_expression(
            expression_string=expression, raster_variables=band_names,
            bandset_catalog=catalog
        )
        self.assertEqual(
            out_exp[0][0],
            cfg.variable_band_quotes + cfg.variable_bandset_name + '1'
            + cfg.variable_band_name + '1' + cfg.variable_band_quotes
        )
        self.assertFalse(error_message)
        # iterate bands in band sets with output temp path and add output to
        # current BandSet
        expression = (
                cfg.forbandsinbandset + '[1,2]S ' + cfg.new_line
                + cfg.variable_band_quotes + cfg.variable_band
                + cfg.variable_band_quotes + cfg.variable_output_separator
                + cfg.variable_output_temporary
                + cfg.variable_output_separator + 'test'
                + cfg.variable_bandset_number_separator
                + cfg.variable_current_bandset)
        out_exp, out_name_list, error_message = band_calc._check_expression(
            expression_string=expression, raster_variables=band_names,
            bandset_catalog=catalog
        )
        self.assertEqual(
            out_exp[0][0].split(cfg.variable_output_separator)[0].strip(),
            cfg.variable_band_quotes + cfg.variable_bandset_name + '1'
            + cfg.variable_band_name + '1' + cfg.variable_band_quotes
        )
        self.assertFalse(error_message)
        # iterate bands with output temp path and add output to BandSet
        expression = (
                cfg.forbandsinbandset + '[1,2]S ' + cfg.new_line
                + cfg.variable_band_quotes + cfg.variable_band
                + cfg.variable_band_quotes + cfg.variable_output_separator
                + cfg.variable_output_temporary
                + cfg.variable_output_separator + 'test1')
        out_exp, out_name_list, error_message = band_calc._check_expression(
            expression_string=expression, raster_variables=band_names,
            bandset_catalog=catalog
        )
        self.assertEqual(
            out_exp[0][0].split(cfg.variable_output_separator)[0].strip(),
            cfg.variable_band_quotes + cfg.variable_bandset_name + '1'
            + cfg.variable_band_name + '1' + cfg.variable_band_quotes
        )
        self.assertFalse(error_message)
        # iterate bands and add output to current BandSet
        expression = (
                cfg.forbandsinbandset + '[1,2]S ' + cfg.new_line
                + cfg.variable_band_quotes + cfg.variable_band
                + cfg.variable_band_quotes + cfg.variable_output_separator
                + 'test2' + cfg.variable_bandset_number_separator
                + cfg.variable_current_bandset)
        out_exp, out_name_list, error_message = band_calc._check_expression(
            expression_string=expression, raster_variables=band_names,
            bandset_catalog=catalog
        )
        self.assertEqual(
            out_exp[0][0].split(cfg.variable_output_separator)[0].strip(),
            cfg.variable_band_quotes + cfg.variable_bandset_name + '1'
            + cfg.variable_band_name + '1' + cfg.variable_band_quotes
        )
        self.assertFalse(error_message)
        # band output name
        expression = (
                cfg.variable_band_quotes + cfg.variable_bandset_name
                + cfg.variable_current_bandset
                + cfg.variable_band_name + '1' + cfg.variable_band_quotes + ' '
                + cfg.variable_output_separator + 'test_1' + cfg.new_line
                + cfg.variable_band_quotes + 'test_1'
                + cfg.variable_band_quotes + ' '
                + cfg.variable_output_separator + 'test_2')
        out_exp, out_name_list, error_message = band_calc._check_expression(
            expression_string=expression, raster_variables=band_names,
            bandset_catalog=catalog
        )
        self.assertEqual(
            out_exp[0][0].split(cfg.variable_output_separator)[0].strip(),
            cfg.variable_band_quotes + cfg.variable_bandset_name + '1'
            + cfg.variable_band_name + '1' + cfg.variable_band_quotes
        )
        self.assertFalse(error_message)
        # iterate bands and output name
        expression = (
                cfg.forbandsinbandset + '[1,2]S ' + cfg.new_line +
                cfg.variable_band_quotes + cfg.variable_band
                + cfg.variable_band_quotes + ' '
                + cfg.variable_output_separator + 'test3')
        out_exp, out_name_list, error_message = band_calc._check_expression(
            expression_string=expression, raster_variables=band_names,
            bandset_catalog=catalog
        )
        self.assertEqual(
            out_exp[0][0].split(cfg.variable_output_separator)[0].strip(),
            cfg.variable_band_quotes + cfg.variable_bandset_name + '1'
            + cfg.variable_band_name + '1' + cfg.variable_band_quotes
        )
        self.assertFalse(error_message)
        # spectral bands alias input
        expression = ('max(' + cfg.variable_band_quotes
                      + cfg.variable_red_name + cfg.variable_band_quotes + ')')
        out_exp, out_name_list, error_message = band_calc._check_expression(
            expression_string=expression, raster_variables=band_names,
            bandset_catalog=catalog
        )
        # spectral range bands
        (blue_band, green_band, red_band, nir_band, swir_1_band,
         swir_2_band) = catalog.get_bandset(1).spectral_range_bands(
            output_as_number=False
            )
        band_number = catalog.get_bandset_bands_by_attribute(
            1, 'wavelength', attribute_value=red_band.wavelength,
            output_number=True
            )
        self.assertEqual(
            out_exp[0][0],
            'max(' + cfg.variable_band_quotes + cfg.variable_bandset_name + '1'
            + cfg.variable_band_name + str(band_number[0])
            + cfg.variable_band_quotes + ')'
        )
        self.assertFalse(error_message)
        # comment lines and nodata
        expression = (
                '# comment line ' + cfg.new_line +
                cfg.variable_band_quotes + cfg.variable_bandset_name
                + cfg.variable_current_bandset + cfg.variable_band_name
                + '3' + cfg.variable_band_quotes
                + ' == nodata(' + cfg.variable_band_quotes
                + cfg.variable_red_name + cfg.variable_band_quotes
                + ')' + cfg.new_line + ' # comment line ' + cfg.new_line)
        out_exp, out_name_list, error_message = band_calc._check_expression(
            expression_string=expression, raster_variables=band_names,
            bandset_catalog=catalog
        )
        self.assertEqual(
            out_exp[0][0],
            cfg.variable_band_quotes + cfg.variable_bandset_name + '1'
            + cfg.variable_band_name + str(band_number[0])
            + cfg.variable_band_quotes + ' == nodata('
            + cfg.variable_band_quotes
            + cfg.variable_bandset_name + '1' + cfg.variable_band_name
            + str(band_number[0]) + cfg.variable_band_quotes + ')'
        )
        self.assertFalse(error_message)
        # output variables
        expression = (cfg.variable_band_quotes + name_list[0]
                      + cfg.variable_band_quotes + ' '
                      + cfg.variable_output_separator + cfg.temp.dir
                      + cfg.variable_output_separator + 'test.tif')
        out_exp, out_name_list, error_message = band_calc._check_expression(
            expression_string=expression, raster_variables=band_names,
            bandset_catalog=catalog
        )
        self.assertEqual(out_name_list[0], 'test')
        self.assertFalse(error_message)
        # output variable name BandSet
        expression = (cfg.variable_band_quotes + name_list[0]
                      + cfg.variable_band_quotes
                      + cfg.variable_output_separator
                      + cfg.variable_output_name_bandset + '.tif')
        out_exp, out_name_list, error_message = band_calc._check_expression(
            expression_string=expression, raster_variables=band_names,
            bandset_catalog=catalog
        )
        self.assertEqual(
            out_name_list[0], catalog.get_bandset(catalog.current_bandset).name
        )
        self.assertFalse(error_message)
        # output variable name date
        expression = (cfg.variable_band_quotes + name_list[
            0] + cfg.variable_band_quotes + cfg.variable_output_separator
                      + cfg.variable_output_name_date + '.tif')
        out_exp, out_name_list, error_message = band_calc._check_expression(
            expression_string=expression, raster_variables=band_names,
            bandset_catalog=catalog
        )
        self.assertEqual(out_name_list[0][0], '2')
        self.assertFalse(error_message)
        cfg.logger.log.debug('>>> test band calc')
        # simple calculation
        raster_list = ['./data/S2_2020-01-01/S2_B02.tif',
                       './data/S2_2020-01-01/S2_B03.tif']
        name_list = ['raster1', 'raster2']
        temp = cfg.temp.temporary_file_path(name_suffix=cfg.tif_suffix)
        expression = (cfg.variable_band_quotes + name_list[0]
                      + cfg.variable_band_quotes + ' + '
                      + cfg.variable_band_quotes + name_list[1]
                      + cfg.variable_band_quotes)
        output = rs.band_calc(
            input_raster_list=raster_list, output_path=temp,
            expression_string=expression,
            input_name_list=name_list, extent_intersection=False
        )
        self.assertTrue(output.check)
        self.assertTrue(files_directories.is_file(output.paths[0]))
        # calculation with not overlapping rasters
        raster_list = ['./data/S2_2020-01-01/S2_B02.tif',
                       './data/S2_2020-01-03/S2_B03.tif']
        # virtual output
        temp = cfg.temp.temporary_file_path(name_suffix=cfg.tif_suffix)
        expression = (cfg.variable_band_quotes + name_list[0]
                      + cfg.variable_band_quotes + ' + '
                      + cfg.variable_band_quotes + name_list[1]
                      + cfg.variable_band_quotes)
        output = rs.band_calc(
            input_raster_list=raster_list, output_path=temp,
            expression_string=expression,
            input_name_list=name_list, extent_intersection=False,
            output_datatype=cfg.int32_dt, any_nodata_mask=False
        )
        self.assertTrue(output.check)
        self.assertTrue(files_directories.is_file(output.paths[0]))
        # add output to BandSet number defined after
        # variable_bandset_number_separator in the output name
        temp = cfg.temp.temporary_file_path(name_suffix=cfg.tif_suffix)
        expression = (cfg.variable_band_quotes + name_list[0]
                      + cfg.variable_band_quotes
                      + cfg.variable_output_separator
                      + cfg.temp.dir + cfg.variable_output_separator
                      + 'test' + cfg.variable_bandset_number_separator + '1')
        band_count = catalog.get_band_count(1)
        output = rs.band_calc(
            input_raster_list=raster_list, output_path=temp,
            expression_string=expression,
            input_name_list=name_list, bandset_catalog=catalog
        )
        self.assertGreater(catalog.get_band_count(1), band_count)
        self.assertTrue(output.check)
        self.assertTrue(files_directories.is_file(output.paths[0]))
        # bandset calculation
        expression = (
                cfg.variable_band_quotes + cfg.variable_bandset_name
                + cfg.variable_current_bandset + cfg.variable_band_name + '1'
                + cfg.variable_band_quotes
        )
        temp = cfg.temp.temporary_file_path(name_suffix=cfg.tif_suffix)
        output = rs.band_calc(
            output_path=temp, expression_string=expression,
            bandset_catalog=catalog
        )
        self.assertTrue(output.check)
        self.assertTrue(files_directories.is_file(output.paths[0]))
        # NDVI calculation
        expression = (
                '(' + cfg.variable_band_quotes + cfg.variable_nir_name
                + cfg.variable_band_quotes + ' - ' + cfg.variable_band_quotes
                + cfg.variable_red_name + cfg.variable_band_quotes + ') / ('
                + cfg.variable_band_quotes + cfg.variable_nir_name
                + cfg.variable_band_quotes + ' + ' + cfg.variable_band_quotes
                + cfg.variable_red_name + cfg.variable_band_quotes + ')'
        )
        temp = cfg.temp.temporary_file_path(name_suffix=cfg.tif_suffix)
        output = rs.band_calc(
            output_path=temp, expression_string=expression,
            bandset_catalog=catalog
        )
        self.assertTrue(output.check)
        self.assertTrue(files_directories.is_file(output.paths[0]))
        # NDVI calculation with expression alias
        cfg.logger.log.debug('>>> test calculation with expression alias')
        expression = (cfg.variable_band_quotes + cfg.variable_ndvi_name
                      + cfg.variable_band_quotes)
        temp = cfg.temp.temporary_file_path(name_suffix=cfg.tif_suffix)
        output = rs.band_calc(
            output_path=temp, expression_string=expression,
            bandset_catalog=catalog
        )
        self.assertTrue(output.check)
        self.assertTrue(files_directories.is_file(output.paths[0]))
        expression = (
                cfg.variable_band_quotes + cfg.variable_bandset_name
                + cfg.variable_current_bandset
                + cfg.variable_band_name + '1' + cfg.variable_band_quotes + ' '
                + cfg.variable_output_separator + 'test_1' + cfg.new_line
                + cfg.variable_band_quotes + 'test_1'
                + cfg.variable_band_quotes + ' '
                + cfg.variable_output_separator + 'test_2')
        temp = cfg.temp.temporary_file_path(name_suffix=cfg.tif_suffix)
        output = rs.band_calc(
            output_path=temp, expression_string=expression,
            bandset_catalog=catalog
        )
        self.assertTrue(output.check)
        self.assertTrue(files_directories.is_file(output.paths[0]))
        # direct expression
        bandset = catalog.get_bandset(1)
        expression = (
                '(' + cfg.variable_band_quotes + cfg.variable_band_name + '1'
                + cfg.variable_band_quotes + ' + ' + cfg.variable_band_quotes
                + cfg.variable_band_name + '2' + cfg.variable_band_quotes + ')'
        )
        temp = cfg.temp.temporary_file_path(name_suffix=cfg.vrt_suffix)
        calc = bandset.execute(rs.band_calc, output_path=temp,
                               expression_string=expression)
        self.assertTrue(calc.check)
        self.assertTrue(files_directories.is_file(calc.paths[0]))
        # direct expression NDVI calculation
        expression = (
                '(' + cfg.variable_band_quotes + cfg.variable_nir_name
                + cfg.variable_band_quotes + ' - ' + cfg.variable_band_quotes
                + cfg.variable_red_name + cfg.variable_band_quotes + ') / ('
                + cfg.variable_band_quotes + cfg.variable_nir_name
                + cfg.variable_band_quotes + ' + ' + cfg.variable_band_quotes
                + cfg.variable_red_name + cfg.variable_band_quotes + ')'
        )
        cfg.logger.log.debug('>>> test bandset calculation with alias')
        temp = cfg.temp.temporary_file_path(name_suffix=cfg.vrt_suffix)
        calc = bandset.execute(rs.band_calc, output_path=temp,
                               expression_string=expression)
        self.assertTrue(calc.check)
        self.assertTrue(files_directories.is_file(calc.paths[0]))
        cfg.logger.log.debug('>>> test bandset calculation with ndvi')
        expression = (
                cfg.variable_band_quotes + cfg.variable_ndvi_name
                + cfg.variable_band_quotes
        )
        temp = cfg.temp.temporary_file_path(name_suffix=cfg.vrt_suffix)
        calc = bandset.execute(rs.band_calc, output_path=temp,
                               expression_string=expression)
        self.assertTrue(calc.check)
        self.assertTrue(files_directories.is_file(calc.paths[0]))

        # clear temporary directory
        rs.close()
