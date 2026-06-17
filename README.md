# Luchsinger Power Model

A configurable power model for pumping kite airborne wind energy (AWE) systems based on the Luchsinger model [1] [2]. Takes awesIO-format YAML configuration files as input and always computes power curves using segmented wind shear profiles.

## Overview

This repository provides a standalone power model for AWE pumping kite systems. It calculates power curves from cut-in to cut-out wind speed, accounting for:

- Aerodynamic forces on the kite
- Ground station generator and storage efficiency losses
- Optimal reeling velocity control
- Three operating regions (below force limit, force-limited, power-limited)

## Purpose

This repository is designed to be:

1. **Used locally** via the scripts in `scripts/` with configuration from `data/`
2. **Integrated into a larger toolchain** where `PowerModel` is instantiated directly with paths to awesIO YAML files

## Features
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
from power_luchsinger import PowerModel

model = PowerModel(
    system_config_path='data/soft_kite_pumping_ground_gen_system.yml',
    wind_resource_path='data/profileed_profiles_wind_resource.yml',
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
| `profileed_profiles_wind_resource.yml` | Wind resource with profileed normalized wind shear profiles |
| `simulation_settings_config.yml` | Simulation settings (e.g. number of power curve points) |

## Physical Model

### Luchsinger Pumping Cycle

The model implements a ground-based pumping kite system with two phases:

1. **Reel-out phase**: Kite flies crosswind generating high tether force, pulling out the tether and driving the generator
2. **Reel-in phase**: Kite is depowered, tether is reeled back in consuming energy from storage

### Operating Regions

| Region | Description |
|--------|-------------|
| **Region 1** | Below force limit — reel-out and reel-in factors jointly optimized |
| **Region 2** | Force-limited — tether force held at nominal; reel-in factor optimized |
| **Region 3** | Power-limited — generator power held at nominal; reel-in factor optimized |

## Model Variants

The simulation setting in `data/simulation_settings_config.yml` selects one of two model formulations:

| Model key | Description |
|-----------|-------------|
| `luchsinger_original` | Original Luchsinger-style pumping-cycle formulation with reel-in force represented using a reel-in elevation angle term. For fixed wing kites.|
| `luchsinger_extended_const_lod_in` | Extended formulation with explicit lift-to-drag-based reel-in force treatment and tether drag contribution in reel-out force factor. For soft wing kites.|

### Practical guidance

- Keep all other inputs identical and change only `model` to compare formulations.
- The same three operating regions are used in both models; the main differences are in force-factor definitions and reel-in/reel-out aerodynamic expressions.
- The comparison workflow in `scripts/plot_power_curves.py` is intended for side-by-side evaluation of outputs from the two model variants.

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
│   ├── profileed_profiles_wind_resource.yml
│   └── simulation_settings_config.yml
├── results/                        # Generated outputs (YAML + plots)
├── requirements.txt
├── README.md
└── LICENSE
```

## References

1. R.H. Luchsinger: "Pumping cycle kite power". In *Airborne Wind Energy*, Springer, 2013. https://doi.org/10.1007/978-3-642-39965-7_3
2. R. Schmehl, M. Rodriguez, L. Ouroumova, and M. Gaunaa: "Airborne Wind Energy for Martian Habitats". In Adaptive On- and Off-Earth Environments, Springer, 2024. https://doi.org/10.1007/978-3-031-50081-7_7



## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.
