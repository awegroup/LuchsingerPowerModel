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
from src.power_luchsinger.plotting import plot_comprehensive_analysis, extract_model_params


def print_summary(model: PowerModel, data: dict) -> None:
    """Print summary of the power curve calculation with wind shear.

    Args:
        model: The power model instance.
        data: Dictionary with all power curve data for multiple profiles.
    """
    profiles = data['profiles']
    n_profiles = len(profiles)
    
    print("\n" + "=" * 60)
    print("POWER CURVE SUMMARY WITH WIND SHEAR")
    print("=" * 60)
    print(f"\nSystem Parameters:")
    print(f"  Wing Area:              {model.wingArea:.1f} m²")
    print(f"  Air Density:            {model.airDensity:.3f} kg/m³")
    print(f"  Nominal Tether Force:   {model.nominalTetherForce:.0f} N")
    print(f"  Nominal Generator Power:{model.nominalGeneratorPower/1000:.1f} kW")
    print(f"  Tether Length:          {model.tetherMinLength:.0f} - {model.tetherMaxLength:.0f} m")

    print(f"\nOperational Envelope:")
    print(f"  Cut-in Wind Speed:      {model.cutInWindSpeed:.1f} m/s (at reference height)")
    print(f"  Cut-out Wind Speed:     {model.cutOutWindSpeed:.1f} m/s (at reference height)")
    print(f"  Force Limit Wind Speed: {model.nominalWindSpeedForce:.1f} m/s")
    print(f"  Power Limit Wind Speed: {model.nominalWindSpeedPower:.1f} m/s")
    
    print(f"\nWind Shear Configuration:")
    print(f"  Reference Height:       {data['reference_height_m']:.1f} m")
    print(f"  Operational Altitude:   {data['operational_altitude_m']:.1f} m")
    print(f"  Number of Profiles:     {n_profiles}")

    print(f"\nPower Statistics Across Profiles:")
    for i, profile in enumerate(profiles):
        power = profile['power']
        windSpeedRef = profile['windSpeedAtRef']
        windSpeedOp = profile['windSpeedAtOp']
        profile_id = profile['profile_id']
        
        max_power = np.max(power)
        idx_max = np.argmax(power)
        
        print(f"\n  Profile {profile_id}:")
        print(f"    Max Power:            {max_power/1000:.2f} kW")
        print(f"    Wind Speed at Max:    {windSpeedRef[idx_max]:.1f} m/s (ref), "
              f"{windSpeedOp[idx_max]:.1f} m/s (op)")
        print(f"    Avg Speed Ratio:      {np.mean(windSpeedOp/windSpeedRef):.3f}")
    
    print("=" * 60)


def main():
    """Main entry point for power curve calculation script."""
    # Use awesIO format configuration files
    systemConfigPath = workspace_root / 'data' / 'soft_kite_pumping_ground_gen_system.yml'
    simulationSettingsPath = workspace_root / 'data' / 'simulation_settings_config.yml'
    windResourcePath = workspace_root / 'data' / 'clustered_profiles_wind_resource.yml'

    print(f"Loading system configuration from: {systemConfigPath}")
    print(f"Loading simulation settings from: {simulationSettingsPath}")
    print(f"Loading wind resource profiles from: {windResourcePath}")

    # Load power model (all YAML loading handled internally via config_loader)
    model = PowerModel(
        system_config_path=systemConfigPath,
        wind_resource_path=windResourcePath,
        simulation_settings_path=simulationSettingsPath,
        validate_file=True,
    )

    # Wind resource is already loaded in model
    wind_shear_data = model.wind_resource
    print(f"\nWind shear profiles loaded:")
    print(f"  {wind_shear_data['n_clusters']} wind profiles")
    print(f"  Reference height: {wind_shear_data['reference_height_m']} m")
    print(f"  Altitude range: {wind_shear_data['altitudes'][0]}"
          f" - {wind_shear_data['altitudes'][-1]} m")

    # Generate power curves with wind shear (500 points)
    print(f"\nCalculating power curves with wind shear (500 points per profile)...")
    data = model.generate_power_curves_with_shear(
        wind_shear_data=wind_shear_data,
        numPoints=500
    )

    # Print summary
    print_summary(model, data)

    # Export power curves in awesIO format
    output_path = workspace_root / 'results' / 'luchsinger_power_curves.yml'
    print(f"\nExporting power curves to: {output_path}")
    model.export_power_curves_awesio(
        data=data,
        output_path=output_path,
        name="Luchsinger Model Power Curves with Wind Shear",
        description="Power curves for 100kW soft kite pumping ground-gen AWE system with wind shear profiles",
        note="Generated using Luchsinger pumping cycle model with 8 representative wind shear profiles",
        file_validate=True
    )

    # For plotting, use the first profile as representative
    # Convert to the format expected by plotting function
    plot_data = {
        'windSpeed': data['profiles'][0]['windSpeedAtOp'],
        'power': data['profiles'][0]['power'],
        'reelOutPower': data['profiles'][0]['reelOutPower'],
        'reelInPower': data['profiles'][0]['reelInPower'],
        'reelOutTime': data['profiles'][0]['reelOutTime'],
        'reelInTime': data['profiles'][0]['reelInTime'],
        'tetherForceOut': data['profiles'][0]['tetherForceOut'],
        'tetherForceIn': data['profiles'][0]['tetherForceIn'],
        'reelOutSpeed': data['profiles'][0]['reelOutSpeed'],
        'reelInSpeed': data['profiles'][0]['reelInSpeed'],
        'gammaOut': data['profiles'][0]['gammaOut'],
        'gammaIn': data['profiles'][0]['gammaIn'],
    }

    # Extract model parameters
    model_params = extract_model_params(model)

    # Create comprehensive plot with energy subplot (using first profile)
    print("\nGenerating comprehensive analysis plots (Profile 1)...")
    plot_comprehensive_analysis(
        plot_data,
        model_params,
        save_path="results/power_curve_analysis.png",
        show=False
    )
    print("Plots saved to: results/power_curve_analysis.png")


if __name__ == '__main__':
    main()
