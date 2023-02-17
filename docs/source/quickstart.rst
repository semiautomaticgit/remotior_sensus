Quickstart
===========================================

This section provides a basic guide to the use of Remotior Sensus.

Start a session of Remotior Sensus
__________________________________

The first step is to start a Remotior Sensus' 
:doc:`session <../remotior_sensus.core.session>`.
During the start of a session, a temporary directory is created to store
temporary files, which is automatically removed after closing the session.
Also, subprocesses are started for parallel processing.

Optional arguments `n_processes` and `available_ram` respectively define
the number of parallel processes and the available RAM used during processing.
Default is 2 parallel processes using 2048 MB of RAM.

.. code:: python
	
    # import Remotior Sensus
    >>> import remotior_sensus
    # start a Remotior Sensus' session defining the number of parallel processes and RAM
    >>> rs = remotior_sensus.Session(n_processes=4, available_ram=4096)
    # for instance check the number of parallel processes
    >>> print(rs.configurations.n_processes)
    4


Processing time can be reduced by increasing the
number of processes and available RAM according to system availability.
Remotior Sensus attempts to not exceed the RAM value 
defined with `available_ram`, however the actual RAM usage could potentially
exceed this value; therefore, it is recommended to set `available_ram` 
as a value lower than the total RAM of the system (e.g., half of system RAM).
The parameters `n_processes` and `available_ram` can also be set later.

Create a BandSet
________________

Although not required, it can be useful to create a :doc:`BandSet <../remotior_sensus.core.bandset>`
of bands to be processed.

A BandSet is an object that includes information about single bands
(from the file path to the spatial and spectral characteristics).
Bands in a BandSet can be referenced by the properties thereof,
such as order number or center wavelength.

BandSets can be used as input for operations on multiple bands
such as Principal Components Analysis, classification, mosaic,
or band calculation.
Multimple BandSets can be defined and identified by their reference number
in the :doc:`BandSet Catalog <../remotior_sensus.core.bandset_catalog>`.

.. code:: python

    # first initiate the BandSet Catalog that manages all the BandSets
    >>> catalog = rs.bandset_catalog()
    # now create a BandSets with a file list
    >>> file_list = ['file1_b1.tif', 'file1_b2.tif', 'file1_b3.tif']
    >>> catalog.create_bandset(file_list, bandset_number=1)
    # for instance get BandSet count
    >>> print(catalog.get_bandset_count())
    1

Create a BandSet from a file list with files inside a data directory,
setting root_directory, defining the BandSet date.

.. code:: python

    >>> bandset_date = '2021-01-01'
    >>> data_directory = 'path'
    >>> bandset = catalog.create_bandset(
    ... file_list,wavelengths=['Sentinel-2'], date=bandset_date, root_directory=data_directory
    ... )

It is possible to get bands using several functions of
:doc:`BandSet Catalog <../remotior_sensus.core.bandset_catalog>`.
For instance, get a band from the attribute center wavelength.

.. code:: python

    >>> band_number = catalog.get_bandset_bands_by_attribute(
        ... bandset_number=1, attribute='wavelength',
        ... attribute_value=0.443, output_number=True)
        >>> print(band_number)
        1

    BandSets can be used in several tools.
    ... bandset_number=1, attribute='wavelength',
    ... attribute_value=0.443, output_number=True)
    >>> print(band_number)
    1

BandSets can be used in several tools.

Run a Tool
__________

Several :doc:`tools <../api_tools>` are available.
For instance the :doc:`Band calc <../remotior_sensus.tools.band_calc>`
allows for mathematical calculations (pixel by pixel) between
bands or single band rasters.
A new raster file is created as result of calculation.

It is possible to perform a calculation between raster files using custom expression.
The following executes the sum between two files.
The arguments `input_raster_list` defines the path of the input files,
and `input_name_list` defines the variable names used in expression corresponding
to input raster files.

.. code:: python

    >>> # start the process
    >>> output = rs.band_calc(
    ... input_raster_list=['file1.tif', 'file2.tif'], output_path='output.tif',
    ... expression_string='"file1 + file2"', input_name_list=['file1', 'file2']
    ... )

Another example is the tool :doc:`Band calc <../remotior_sensus.tools.band_combination>`.
This tool is intended for combining classifications in order to get a
raster where each value corresponds to a combination of class values.
A unique value is assigned to each combination of values.
The output is a raster made of unique values corresponding to combinations
of values.

Several tools accept both file paths or BandSets as input.
The following performs the band combination of BandSet 1 defined by the number
thereof.

.. code:: python

    >>> combination = combination(
    ...     input_bands=1, output_path='output.tif', bandset_catalog=catalog
    ... )

Output Manager
______________

Tools produce output files.
The modules return an object
:doc:`OutputManager <../remotior_sensus.core.output_manager>`
having several attributes:

* check: True if output is as expected, False if process failed.
* path: path of the first output.
* paths: list of output paths in case of multiple outputs.
* extra: additional output elements depending on the process.

The previous Band combination produced a raster output and a
table output containing combination statistics.
The paths can be retrieved as in the following example.

.. code:: python

    >>> raster_path, table_path = combination.paths
    >>> print(raster_path)

Table Manager
_________________

Considering that several tools produce tables, the functions in
:doc:`Table Manager <../remotior_sensus.core.table_manager>` allow for
opening .csv and .dbf file, and managing table data as NumPy structured arrays.

.. code:: python

    >>> table1 = rs.table_manager.open_file(
    ... table_path, field_names=['value', 'field1', 'field2', 'sum', 'area']
    ... )

It includes functions for field calculation, join and pivot tables.
For instance, it is possible to perform a calculation on fields by defining
an expression which includes the field names, such as "area" in the above
example.

.. code:: python

    >>> # perform a calculation
    >>> calculation = rs.table_manager.calculate(
    ... matrix=table1, expression_string='"area" * 100',
    ... output_field_name='calc'
    ... )

It is possible to export the resulting table to .csv file, selecting fields
and separators.

.. code:: python

    >>> # export the table to csv
    >>> rs.table_manager.export_table(
    ... matrix=calculation, output_path='output.csv',
    ... fields=['field1', 'calc'], separator=';', decimal_separator='.'
    ... )

Close the Session
_________________

A session should be closed at the end of all the processes
to remove the temporary files and stop subprocesses.

.. code:: python

    # close Remotior Sensus' session
    >>> rs.close()
