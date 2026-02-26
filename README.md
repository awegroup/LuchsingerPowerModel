# Luchsinger Power Model

A configurable power model for pumping kite airborne wind energy (AWE) systems based on the Luchsinger model. Takes awesIO-format YAML configuration files as input and always computes power curves using segmented wind shear profiles.

## Overview

This repository provides a standalone power model for AWE pumping kite systems. It calculates power curves from cut-in to cut-out wind speed, accounting for:

- Aerodynamic forces on the kite
- Ground station generator and storage efficiency losses
- Optimal reeling velocity control
- Wind shear along the tether deployment using segmented calculations
- Three operating regions (below force limit, force-limited, power-limited)

## Purpose

This repository is designed to be:

1. **Used locally** via the scripts in `scripts/` with configuration from `data/`
2. **Integrated into a larger toolchain** where `PowerModel` is instantiated directly with paths to awesIO YAML files

## Features

- **Wind shear**: Tether deployment is divided into 20 segments; each segment uses the local wind speed interpolated from a normalized wind profile
- **Reeling-factor optimization**: Optimal reel-out and reel-in velocity factors calculated per operating region
- **awesIO format**: Input and output follow the awesIO data schema; optional validation on load and export
- **Configurable**: All physical parameters via awesIO-format YAML files

## Installation

### Requirements

- Python >= 3.8
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- PyYAML >= 5.4.0
- Matplotlib >= 3.4.0

### Install dependencies

```bash
pip install -r requirements.txt
```

Or using conda:

```bash
conda install numpy scipy pyyaml matplotlib
```

## Quick Start

### Using the Script

```bash
python scripts/calculate_power_curves.py
```

### Using as a Library

```python
import numpy as np
from src.power_luchsinger import PowerModel

model = PowerModel(
    system_config_path='data/soft_kite_pumping_ground_gen_system.yml',
    wind_resource_path='data/clustered_profiles_wind_resource.yml',
    simulation_settings_path='data/simulation_settings_config.yml',
    validate_file=True,
)

# Generate power curves for all wind profiles in the wind resource file
data = model.generate_power_curves(
    wind_speeds=np.arange(4, 25, 1),   # optional; overrides num_points from settings
    output_path='results/luchsinger_power_curves.yml',
    verbose=True,
    show_plot=True,
    save_plot=True,
    validate_file=True,
)

print(f"Generated {len(data['profiles'])} power curves")
```

## Configuration Files

Three YAML files are required, all following the awesIO schema:

| File | Description |
|------|-------------|
| `soft_kite_pumping_ground_gen_system.yml` | System configuration (kite, tether, ground station, operational envelope) |
| `clustered_profiles_wind_resource.yml` | Wind resource with clustered normalized wind shear profiles |
| `simulation_settings_config.yml` | Simulation settings (e.g. number of power curve points) |

### System Configuration Parameters

| Parameter | Description | Unit |
|-----------|-------------|------|
| `wingArea` | Projected wing area | m² |
| `liftCoefficientOut` | Lift coefficient during reel-out | - |
| `dragCoefficientKiteOut` | Kite drag coefficient during reel-out | - |
| `dragCoefficientKiteIn` | Kite drag coefficient during reel-in | - |
| `tetherMaxLength` | Maximum tether length | m |
| `tetherMinLength` | Minimum tether length | m |
| `airDensity` | Air density | kg/m³ |
| `nominalTetherForce` | Maximum tether force | N |
| `nominalGeneratorPower` | Maximum generator power | W |
| `reelOutSpeedLimit` | Maximum reel-out speed | m/s |
| `reelInSpeedLimit` | Maximum reel-in speed | m/s |
| `cutInWindSpeed` | Cut-in wind speed at reference height | m/s |
| `cutOutWindSpeed` | Cut-out wind speed at reference height | m/s |
| `elevationAngleOut` | Tether elevation angle during reel-out | rad |
| `elevationAngleIn` | Tether elevation angle during reel-in | rad |
| `generatorEfficiency` | Generator efficiency | - |
| `storageEfficiency` | Energy storage round-trip efficiency | - |

## Physical Model

### Luchsinger Pumping Cycle

The model implements a ground-based pumping kite system with two phases:

1. **Reel-out phase**: Kite flies crosswind generating high tether force, pulling out the tether and driving the generator
2. **Reel-in phase**: Kite is depowered, tether is reeled back in consuming energy from storage

### Wind Shear

The reeling length is divided into 20 equal segments. For each segment the average altitude is computed and the local wind speed is interpolated from the normalized wind shear profile, then scaled by the reference wind speed. All forces and energies are integrated over these segments.

Nominal operating wind speeds (force limit, power limit) are recomputed for each wind profile before calculating the power curve.

### Operating Regions

| Region | Description |
|--------|-------------|
| **Region 1** | Below force limit — reel-out and reel-in factors jointly optimized |
| **Region 2** | Force-limited — tether force held at nominal; reel-in factor optimized |
| **Region 3** | Power-limited — generator power held at nominal; reel-in factor optimized |

## Project Structure

```
├── src/
│   └── power_luchsinger/
│       ├── __init__.py
│       ├── power_model.py          # Main PowerModel class
│       ├── calculations.py         # Pure aerodynamic calculation functions
│       ├── config_loader.py        # awesIO YAML loading and parsing
│       └── plotting.py             # Plotting utilities
├── scripts/
│   ├── __init__.py
│   └── calculate_power_curves.py   # Script for generating power curves
├── data/
│   ├── soft_kite_pumping_ground_gen_system.yml
│   ├── clustered_profiles_wind_resource.yml
│   └── simulation_settings_config.yml
├── results/                        # Generated outputs (YAML + plots)
├── requirements.txt
├── README.md
└── LICENSE
```

## References

1. R.H. Luchsinger: "Pumping cycle kite power". In *Airborne Wind Energy*, Springer, 2013. https://doi.org/10.1007/978-3-642-39965-7_3

## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.
