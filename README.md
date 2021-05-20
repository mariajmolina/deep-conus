# deep-conus
Repository containing scripts associated with NCAR Advanced Study Program deep learning project (2019-2021) exploring the robustness of a convolutional neural network (CNN) on classifying North America convection in a changing climate.

------------

Molina, M. J., D. J. Gagne, and A. F. Prein (under review): A benchmark to test generalization capabilities of deep learning methods to classify severe convective storms in a changing climate, Earth and Space Science.

------------

_CONUS1 dataset used in project is available on NCAR's RDA at https://rda.ucar.edu/datasets/ds612.0/ and Cheyenne at /gpfs/fs1/collections/rda/data/ds612.0._

_Dataset used to create journal figures is available at [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4052586.svg)](https://doi.org/10.5281/zenodo.4052586)_

## CONUS1 Variables (3D) Used to train the CNN
| Variable | Description |
| ----------- | ----------- |
| P | Total pressure (Pa). |
| QVAPOR | Water vapor mixing ratio (kg/kg). |
| TK | Air temperature (K). |
| EU | x-wind component (m/s). |
| EV | y-wind component (m/s). |
