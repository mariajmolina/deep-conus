# deep_conus1
Repository containing scripts associated with ASP deep learning project exploring current and future convection of North America.

------------

_CONUS1 dataset used in project is available on NCAR's RDA and Cheyenne at /gpfs/fs1/collections/rda/data/ds612.0._


## CONUS1 Variables (3D) Used
| Variable | Description |
| ----------- | ----------- |
| P | Total pressure (P0+PB) (Pa). |
| QVAPOR | Water vapor mixing ratio (kg kg-1). |
| TK | Air temperature (K). |
| EU | x-wind component (m/s). Used Uearth = U*cosalpha - V*sinalpha to rotate to the earth-relative U. |
| EV | y-wind component (m/s). Used Vearth = V*cosalpha + U*sinalpha to rotate to the earth-relative V. |
| W | Z-wind component (m/s). |  

## Convection Variables Derived from WRF-Python
| Variable | Description |
| ----------- | ----------- |
| UH | Updraft helicity. |
| CTT | Cloud top temperature. |


## General Variables Derived from WRF-Python for Deep Learning
| Variable | Description |
| ----------- | ----------- |
| TK, QVAPOR, EU, EV, P, W | Interpolated onto 1, 3, 5, and 7 km. |

