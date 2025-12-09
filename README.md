# GID_ID10

This is code for the processing of Surface X-ray Scattering data, obtained on the ID10 beamline at ESRF.

## Content:
1) GID class, used for the data analysis of Grazing Incidence Diffraction data obtained with linear detector and Soller collimator
2) rebin function, used for rebinning data, from XRR_ID10_ESRF

## Functionality:
1) Reading h5 files, produced at ID10-SURF
2) All necessary corrections, including: conversion to q, rebinning
3) Calculation of reciprocal space maps, obtained during GID experiment
4) Plotting and saving graphs and data

## Dependencies

The project requires the following Python packages:
*   `h5py`: For reading HDF5 data files.
*   `numpy`: For numerical operations.
*   `matplotlib`: For plotting and visualization.
*   `scipy`: For scientific computing.
*   `lmfit`: For peak fitting and analysis.

You can install them using pip:
```bash
pip install h5py numpy matplotlib scipy lmfit
```

## Documentation

The project documentation is generated using Sphinx. To build the documentation locally:

1.  Ensure you have the required dependencies installed (including `sphinx` and `sphinx_rtd_theme`).
    ```bash
    pip install sphinx sphinx_rtd_theme
    ```
2.  Navigate to the `docs` directory.
    ```bash
    cd docs
    ```
3.  Build the HTML documentation.
    ```bash
    make html
    ```

The generated HTML files will be located in `docs/_build/html`. Open `index.html` in your browser to view the documentation.
