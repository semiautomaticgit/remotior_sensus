Changelog
===============

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