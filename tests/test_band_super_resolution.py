from unittest import TestCase

import remotior_sensus

"""
class TestBandSuperResolution(TestCase):

    def test_band_super_resolution(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=10000, log_level=10
        )
        cfg = rs.configurations

        file_list = []
        catalog_s2 = rs.bandset_catalog()
        catalog_s2.create_bandset(file_list, wavelengths=['Sentinel-2'])
        s2_bandset = catalog_s2.get(1)
        # B04, B03, B02
        wl_list = [
            cfg.satellites[cfg.satSentinel2][0][3],
            cfg.satellites[cfg.satSentinel2][0][2],
            cfg.satellites[cfg.satSentinel2][0][1],
        ]
        s2_bands = s2_bandset.get_absolute_paths()
        s2_band_list = []
        for wl in wl_list:
            band = s2_bandset.get_band_by_wavelength(wavelength=wl,
                                                     output_as_number=True)
            s2_band_list.append(s2_bands[band - 1])
        catalog_s2.create_bandset(paths=s2_band_list, bandset_number=2)

        temp = cfg.temp.temporary_file_path(name='class')
        classification_p = rs.band_super_resolution(
            input_bands=catalog_s2.get(2),
            output_path=[],
            pretrained_model_path='',
            super_resolution_factor=4,
            pytorch_device='cpu', overwrite=True
        )
        # clear temporary directory
        rs.close()

"""