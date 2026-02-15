Changelog
===============

v0.6.2
________

* Minor bug fixing.

v0.6.0
________

* Added optional dependency Pandas for performance improvement in tabular
  data.
* In tool "Band classification" added option for using PyTorch pretrained
  model. In case a pretrained model is selected, and additional algorithm
  is selected for classification, using the same parameters of the named
  algorithm (e.g. random forest); after executing the pretrained model, the
  additional algorithm is executed on the embeddings for classification.
  Currently, it works with models pretrained by the Allen Institute for
  Artificial Intelligence (SatlasPretrain: https://satlas-pretrain.allen.ai)
  in particular, Sentinel-2 swin-v2-base single-image multispectral and
  swin-v2-tiny single-image multispectral models, and Landsat 8 Landsat 9
  swin-v2-base single-image multispectral model.
  SatlasPretrain model weights are released under the
  Open Data Commons 'Attribution License (ODC-BY).
  The repository code is licensed under the Apache License
  2.0 (https://huggingface.co/allenai/satlas-pretrain). This tool downloads the
  official SatlasPretrain weights (Bastani et al., "SatlasPretrain: A
  Large-Scale Dataset for Remote Sensing Image Understanding", ICCV 2023,
  arXiv:2211.15660, https://doi.org/10.48550/arXiv.2211.15660).
  All model weights remain the property of their respective authors.
* In tool "Band classification" added PyTorch pretreained segmentation models
  for Sentinel-2 (swin-v2-base single-image multispectral model using 3 bands
  or 4 bands) pretrained by DPR Team as part of the DPR Zoo Segmentation Hub
  framework (https://github.com/DPR25/dpr-zoo-segmentation-hub) based on
  SatlasPretrain models.
  The model output classes: background, water, developed, tree, shrub, grass,
  crop, bare, snow, wetland, mangroves, moss.
  The repository code of DPR Zoo models are licensed under the MIT License
  (https://huggingface.co/martinkorelic/dpr-zoo-models).
  This tool downloads the model weights (DPR Team, 2025. Made as part of Arnes
  Hackathon 2025).
  All model weights remain the property of their respective authors.
* In tool "Download products" included the download of Sentinel-2 L2A SCL band.
* In tool "Preprocess products" included the Sentinel-2 L2A SCL band.
* Improved progress monitoring for multiprocess.
* Code optimization and bug fixing.

v0.5.2
________

* Performance improvement for the tool "Vector to raster" with the method
  area_based.
* Improvement of the multiprocess iterator.
* Minor fixes

v0.5.1
________

* Fixed issue with the tool "Vector to raster" where a few polygons were
  randomly skipped using the method area_based.
* Minor fixes

v0.5.0
________

* New tool "Raster label" for calculating the area of contiguous
  patches in a raster. The output is a raster where each pixel value represents
  the pixel count of the patch thereof.
* Performance improvement for the tool "Vector to raster" with the method
  area_based.
* Minor fixes

v0.4.4
________

* Changed raster_zonal_stats to accept raster input as reference_path
* Fixed handling nan value as nodata

v0.4.3
________

* First experimental implementation of Pytorch for band_calc
* Minor fixes

v0.4.2
________

* Minor fixes

v0.4.1
________

* Fixed preprocessing calculation
* Minor fixes

v0.4.0
________

* Added tool "Band clustering" for unsupervised K-means classification of
  bandset
* Added tool "Raster edit" for direct editing of pixel values based on vector
* Added tool "Raster zonal stats" for calculating statistics of a raster
  intersecting a vector.
* Improved the NoData handling for multiprocess calculation
* In "Band clip", "Band dilation", "Band erosion", "Band sieve",
  "Band neighbor", "Band resample" added the option multiple_resolution to
  keep original resolution of individual rasters, or use the resolution of the
  first raster for all the bands
* In "Cross classification" fixed area based accuracy and added kappa hat
  metric
* In "Band combination" added option no_raster_output to avoid the creation of
  output raster, producing only the table of combinations
* In "Band calc" replaced nanpercentile with optimized calculation function
* Improved extraction of ROIs in "Band classification"
* Minor bug fixing and removed Requests dependency

v0.3.5
________

* Fixed Copernicus access token error
* Fixed automatic band wavelength definition in BandSet

v0.3.04
________

* Fixed Jupyter interface

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