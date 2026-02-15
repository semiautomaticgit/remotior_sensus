from unittest import TestCase

import remotior_sensus
from remotior_sensus.util import download_tools, files_directories


class TestDownloadTools(TestCase):

    def test_download_tools(self):
        rs = remotior_sensus.Session(
            n_processes=2, available_ram=1000, log_level=10
        )
        cfg = rs.configurations

        version = download_tools.get_latest_rs_version()
        temp_rs = cfg.temp.temporary_file_path(name_suffix='.tar.gz')
        rs_download = download_tools.download_rs_version(version=version,
                                                         output_path=temp_rs)
        self.assertTrue(files_directories.is_file(rs_download))
        cfg.logger.log.debug('>>> test download')
        url = 'https://www.python.org'
        temp = cfg.temp.temporary_file_path(name_suffix='.html')
        download_tools.download_file(url=url, output_path=temp)
        self.assertTrue(files_directories.is_file(temp))
        url_3 = (
            "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?"
            "$filter=Collection/Name%20eq%20'SENTINEL-2'%20and%20ContentDate"
            "/Start%20gt%202025-01-01T00:00:00.000Z%20and%20ContentDate"
            "/Start%20lt%202025-01-30T21:42:55.721Z%20and%20Attributes"
            "/OData.CSC.DoubleAttribute/any(att:att/Name%20eq%20%27cloudCover"
            "%27%20and%20att/OData.CSC.DoubleAttribute/Value%20le%2080)%20and"
            "%20OData.CSC.Intersects(area=geography%27SRID=4326;POLYGON"
            "%20((8%2043,8%2041,10%2041,10%2043,8%2043))%27)&$orderby="
            "ContentDate/Start%20asc&$expand=Attributes&$count=True&$top=5&"
            "$skip=0"
        )
        temp_3 = cfg.temp.temporary_file_path(name_suffix='.json')
        download_tools.download_file(url=url_3, output_path=temp_3)
        self.assertTrue(files_directories.is_file(temp_3))

        # clear temporary directory
        rs.close()
