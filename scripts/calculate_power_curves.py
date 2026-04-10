"""Calculate and plot comprehensive power curve analysis for an AWE system.

This script loads configuration from an awesIO YAML file, validates it,
calculates power curves with wind shear profiles from clustered wind resource data,
and creates a comprehensive visualization. Results are exported in awesIO format.
"""

import sys
from pathlib import Path
import numpy as np

# Add workspace root to path
workspace_root = Path(__file__).parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from src.power_luchsinger import PowerModel

def main():
    """Main entry point for power curve calculation script."""
    # Use awesIO format configuration files
    systemConfigPath = workspace_root / 'data' / 'kitepower V3_20.yml'
    simulationSettingsPath = workspace_root / 'data' / 'simulation_settings_config.yml'
    windResourcePath = workspace_root / 'data' / 'wind_resource.yml'
    # windResourcePath = workspace_root / 'data' / 'wind_resource.yml'
    OUTPUT_PATH = workspace_root / 'results' / 'luchsinger_power_curves.yml'

    # Load power model (all YAML loading handled internally via config_loader)
    model = PowerModel(
        system_config_path=systemConfigPath,
        wind_resource_path=windResourcePath,
        simulation_settings_path=simulationSettingsPath,
        validate_file=True,
    )
    # Generate power curves with wind shear
    data = model.generate_power_curves(
        output_path=OUTPUT_PATH,
        verbose=True,
        show_plot=True,
        save_plot=True, 
        validate_file=True)
    
    # data = model.simulate_cycle_at_one_wind_speed(ws_ref=10.0,  selected_profiles=[0, 1, 2], verbose=True)
    # print(data)

if __name__ == '__main__':
    main()
