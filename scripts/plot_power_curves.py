"""Plot power curves for all wind shear profiles.

This script loads power curve data from a YAML file and creates visualizations
comparing the power curves across different wind shear profiles.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import yaml

# Add workspace root to path
workspace_root = Path(__file__).parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))


def load_power_curves(yaml_path: Path) -> dict:
    """Load power curves from YAML file.
    
    Args:
        yaml_path: Path to power curves YAML file.
        
    Returns:
        Dict with power curve data.
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"Power curves file not found: {yaml_path}")
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    return data


def plot_all_power_curves(data: dict, output_path: Path) -> None:
    """Create comprehensive plot of all power curves.
    
    Args:
        data: Power curve data dictionary.
        output_path: Path to save the plot.
    """
    metadata = data.get('metadata', {})
    reference_wind_speeds = np.array(data['reference_wind_speeds_m_s'])
    power_curves = data['power_curves']
    n_profiles = len(power_curves)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Define color map for profiles
    colors = plt.cm.viridis(np.linspace(0, 1, n_profiles))
    
    # 1. Main plot: All cycle power curves
    ax1 = fig.add_subplot(gs[0, :])
    for i, curve in enumerate(power_curves):
        profile_id = curve['profile_id']
        cycle_power_kw = np.array(curve['cycle_power_w']) / 1000
        ax1.plot(reference_wind_speeds, cycle_power_kw, 
                label=f'Profile {profile_id}', color=colors[i], linewidth=2)
    
    ax1.set_xlabel('Wind Speed at Reference Height (m/s)', fontsize=12)
    ax1.set_ylabel('Cycle Power (kW)', fontsize=12)
    ax1.set_title('Power Curves for All Wind Shear Profiles', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', ncol=4, fontsize=9)
    
    # 2. Reel-out vs Reel-in power comparison
    ax2 = fig.add_subplot(gs[1, 0])
    for i, curve in enumerate(power_curves):
        profile_id = curve['profile_id']
        reel_out_power_kw = np.array(curve['reel_out_power_w']) / 1000
        ax2.plot(reference_wind_speeds, reel_out_power_kw,
                color=colors[i], linewidth=1.5, alpha=0.7)
    
    ax2.set_xlabel('Wind Speed (m/s)', fontsize=11)
    ax2.set_ylabel('Reel-Out Power (kW)', fontsize=11)
    ax2.set_title('Reel-Out Power Generation', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[1, 1])
    for i, curve in enumerate(power_curves):
        profile_id = curve['profile_id']
        reel_in_power_kw = np.array(curve['reel_in_power_w']) / 1000
        ax3.plot(reference_wind_speeds, reel_in_power_kw,
                color=colors[i], linewidth=1.5, alpha=0.7)
    
    ax3.set_xlabel('Wind Speed (m/s)', fontsize=11)
    ax3.set_ylabel('Reel-In Power (kW)', fontsize=11)
    ax3.set_title('Reel-In Power Consumption', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 3. Cycle time breakdown
    ax4 = fig.add_subplot(gs[2, 0])
    for i, curve in enumerate(power_curves):
        profile_id = curve['profile_id']
        cycle_time = np.array(curve['cycle_time_s'])
        ax4.plot(reference_wind_speeds, cycle_time,
                color=colors[i], linewidth=1.5, alpha=0.7)
    
    ax4.set_xlabel('Wind Speed (m/s)', fontsize=11)
    ax4.set_ylabel('Cycle Time (s)', fontsize=11)
    ax4.set_title('Pumping Cycle Duration', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 4. Statistics summary
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    # Calculate statistics for each profile
    stats_text = "Power Statistics by Profile\n" + "="*40 + "\n\n"
    for i, curve in enumerate(power_curves):
        profile_id = curve['profile_id']
        cycle_power_kw = np.array(curve['cycle_power_w']) / 1000
        max_power = np.max(cycle_power_kw)
        idx_max = np.argmax(cycle_power_kw)
        ws_at_max = reference_wind_speeds[idx_max]
        
        # Find average power in operational range (5-15 m/s)
        mask = (reference_wind_speeds >= 5) & (reference_wind_speeds <= 15)
        avg_power = np.mean(cycle_power_kw[mask])
        
        stats_text += f"Profile {profile_id}:\n"
        stats_text += f"  Max Power: {max_power:.1f} kW @ {ws_at_max:.1f} m/s\n"
        stats_text += f"  Avg Power (5-15 m/s): {avg_power:.1f} kW\n\n"
    
    ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Overall title
    fig.suptitle(metadata.get('name', 'Power Curves Analysis'),
                fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    plt.show()


def plot_wind_shear_profiles(data: dict, output_path: Path) -> None:
    """Plot normalized wind shear profiles.
    
    Args:
        data: Power curve data dictionary.
        output_path: Path to save the plot.
    """
    altitudes = np.array(data.get('altitudes_m', []))
    power_curves = data['power_curves']
    n_profiles = len(power_curves)
    
    if len(altitudes) == 0:
        print("No altitude data available for wind shear profiles")
        return
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Define color map for profiles
    colors = plt.cm.viridis(np.linspace(0, 1, n_profiles))
    
    # Plot u_normalized (horizontal wind speed)
    for i, curve in enumerate(power_curves):
        profile_id = curve['profile_id']
        u_norm = np.array(curve['u_normalized'])
        ax1.plot(u_norm, altitudes, label=f'Profile {profile_id}',
                color=colors[i], linewidth=2)
    
    ax1.set_xlabel('Normalized Horizontal Wind Speed', fontsize=12)
    ax1.set_ylabel('Altitude (m)', fontsize=12)
    ax1.set_title('Horizontal Wind Shear Profiles', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=9)
    ax1.axhline(y=data['metadata']['reference_height_m'], color='red',
               linestyle='--', linewidth=1, alpha=0.5, label='Reference Height')
    ax1.axhline(y=data['metadata']['model_config']['operating_altitude_m'],
               color='green', linestyle='--', linewidth=1, alpha=0.5,
               label='Operating Altitude')
    
    # Plot v_normalized (vertical wind speed)
    for i, curve in enumerate(power_curves):
        profile_id = curve['profile_id']
        v_norm = np.array(curve['v_normalized'])
        ax2.plot(v_norm, altitudes, label=f'Profile {profile_id}',
                color=colors[i], linewidth=2)
    
    ax2.set_xlabel('Normalized Vertical Wind Speed', fontsize=12)
    ax2.set_ylabel('Altitude (m)', fontsize=12)
    ax2.set_title('Vertical Wind Shear Profiles', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=9)
    ax2.axhline(y=data['metadata']['reference_height_m'], color='red',
               linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(y=data['metadata']['model_config']['operating_altitude_m'],
               color='green', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    
    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Wind shear profiles plot saved to: {output_path}")
    
    plt.show()


def main():
    """Main entry point for plotting script."""
    # Load power curves
    power_curves_path = workspace_root / 'results' / 'luchsinger_power_curves.yml'
    
    if not power_curves_path.exists():
        print(f"Error: Power curves file not found: {power_curves_path}")
        print("Please run calculate_power_curves.py first to generate the data.")
        sys.exit(1)
    
    print(f"Loading power curves from: {power_curves_path}")
    data = load_power_curves(power_curves_path)
    
    n_profiles = len(data['power_curves'])
    print(f"Loaded {n_profiles} power curve profiles")
    
    # Create plots
    print("\nGenerating power curves comparison plot...")
    output_path_curves = workspace_root / 'results' / 'all_power_curves.png'
    plot_all_power_curves(data, output_path_curves)
    
    print("\nGenerating wind shear profiles plot...")
    output_path_shear = workspace_root / 'results' / 'wind_shear_profiles.png'
    plot_wind_shear_profiles(data, output_path_shear)
    
    print("\n" + "="*60)
    print("Plotting complete!")
    print("="*60)


if __name__ == "__main__":
    main()
