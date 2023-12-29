Changelog
===============

v0.3.03
________

* Fixed Jupyter interface

v0.3.02
________

* Fixed Jupyter interface

v0.3.01
________

* Added functions for interactive interface in Jupyter environment
* Fixed Sentinel-2 band 8A identification in preprocess products

v0.2.01
________

* In Download Products added the functions to search and download Collections
  from Microsoft Planetary Computer: Sentinel-2, Landsat, ASTER,
  MODIS Surface Reflectance 8-Day, and Copernicus DEM


v0.1.24
________

* Fixed band calc calculation with multiband raster as bandset
* Fixed preview path for Copernicus products

v0.1.23
________

* Minor fixes

v0.1.22
________

* Fixed prepare input function
* Fixed logger for multiprocess


v0.1.21
________

* Fixed requirements


v0.1.20
________

* Fixed Copernicus search and download service


v0.1.19
________

* Fixed Copernicus search and download service

v0.1.18
________

* Added Copernicus download service from
  https://catalogue.dataspace.copernicus.eu
  if copernicus_user and copernicus_password are provided.

v0.1.17
________

* Fixed spectral signature calculation for multiband raster
* Fixed closing multiprocess at exit

v0.1.16
________

* Fixed issue in block size calculation for multiprocess in case of large
  input raster and low RAM;
* Fixed management of bandsets using multiband rasters;
* Minor fixes to multiprocess download;