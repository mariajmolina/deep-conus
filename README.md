# deep_conus1
Repository containing scripts associated with ASP deep learning project exploring current and future convection of North America.

------------

_CONUS1 dataset used in project is available on NCAR's RDA and Cheyenne at /gpfs/fs1/collections/rda/data/ds612.0._

## CONUS1 Variables (3D)

| Variable | Description |
| ----------- | ----------- |
| Z | Geopotential Height (PH + PHB)/9.81 (m). |
| W | Z-wind component (m/s). |
| V | y-wind component (m/s). V is relative to the model grid. Use Vearth = V*cosalpha + U*sinalpha to rotate to the earth-relative V (WRF3D) |
| U | x-wind component (m/s). U is relative to the model grid. Use Uearth = U*cosalpha - V*sinalpha to rotate to the earth-relative U (WRF3D) |
| P | Total pressure (P0+PB) (Pa). |
| QVAPOR | Water vapor mixing ratio (kg kg-1). |
| TK | Air temperature (K). |
| EU | x-wind component (m/s). Used Uearth = U*cosalpha - V*sinalpha to rotate to the earth-relative U. |
| EV | y-wind component (m/s). Used Vearth = V*cosalpha + U*sinalpha to rotate to the earth-relative V. |

## CONUS1 Variables (2D)
| Variable | Description |
| ----------- | ----------- |
| . | . |

## Variables Derivable from WRF-Python
| Variable | Description |
| ----------- | ----------- |
| UH | . |

