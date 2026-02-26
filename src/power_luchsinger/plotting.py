"""Plotting utilities for AWE power model visualization.

This module provides plotting functions that can be used by external toolchains
to visualize power curve analysis results.
"""

import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend to avoid QPainter errors

from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt


def plot_comprehensive_analysis(
    data: Dict,
    model_params: Optional[Dict] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """Create comprehensive subplot analysis of power curve.

    This function creates a multi-panel figure showing:
    - Power output (cycle, reel-out, reel-in)
    - Cycle times (total, reel-out, reel-in)
    - Tether forces
    - Reel speeds and elevation angles
    - Reeling factors (gamma)
    - Energy per cycle (reel-out, reel-in, net)

    Args:
        data (Dict): Full power curve data dict returned by
            ``generate_power_curves_with_shear``, containing a 'profiles'
            list. The first profile is used for plotting.
        model_params (Dict, optional): Model parameters for annotations.
            Can include: wingArea, nominalGeneratorPower, nominalTetherForce,
            nominalWindSpeedForce, nominalWindSpeedPower, cutOutWindSpeed,
            elevationAngleOut, elevationAngleIn
        save_path (str, optional): Path to save figure. If None, not saved.
        show (bool): Whether to display the figure. Default True.

    Returns:
        plt.Figure: The created figure object.
    """
    profiles = data['profiles']
    n_profiles = len(profiles)

    # Build a colormap: use tab10 for up to 10 profiles, tab20 beyond
    cmap = plt.get_cmap('tab10' if n_profiles <= 10 else 'tab20')
    colors = [cmap(i % cmap.N) for i in range(n_profiles)]

    # Use the reference wind speed axis (same across profiles)
    windSpeedRef = profiles[0]['windSpeedAtRef']

    # Create figure with subplots (4 rows, 2 columns)
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.3)

    def _add_region_lines(ax):
        """Add vertical region-boundary lines if model_params available."""
        if not model_params:
            return
        if 'nominalWindSpeedForce' in model_params:
            ax.axvline(x=model_params['nominalWindSpeedForce'], color='orange',
                       linestyle=':', alpha=0.5)
        if ('nominalWindSpeedPower' in model_params and
                'cutOutWindSpeed' in model_params and
                model_params['nominalWindSpeedPower'] < model_params['cutOutWindSpeed']):
            ax.axvline(x=model_params['nominalWindSpeedPower'], color='red',
                       linestyle=':', alpha=0.5)

    # Subplot 1: Power curves – all profiles
    ax1 = fig.add_subplot(gs[0, 0])
    for i, profile in enumerate(profiles):
        c = colors[i]
        lbl = f'Profile {profile["profile_id"]}'
        ax1.plot(windSpeedRef, profile['power'] / 1000,
                 color=c, linewidth=2, label=lbl)
        ax1.plot(windSpeedRef, profile['reelOutPower'] / 1000,
                 color=c, linewidth=1, linestyle='--', alpha=0.6)
        ax1.plot(windSpeedRef, profile['reelInPower'] / 1000,
                 color=c, linewidth=1, linestyle=':', alpha=0.6)

    if model_params and 'nominalGeneratorPower' in model_params:
        ax1.axhline(y=model_params['nominalGeneratorPower'] / 1000,
                    color='purple', linestyle=':', alpha=0.5, label='Nominal Power')
    _add_region_lines(ax1)

    # Legend: profile colours + line-style guide
    handles, labels = ax1.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles += [
        Line2D([0], [0], color='gray', linewidth=2, label='Cycle power'),
        Line2D([0], [0], color='gray', linewidth=1, linestyle='--', label='Reel-out power'),
        Line2D([0], [0], color='gray', linewidth=1, linestyle=':', label='Reel-in power'),
    ]
    ax1.set_xlabel('Wind Speed at ref. height (m/s)', fontsize=11)
    ax1.set_ylabel('Power (kW)', fontsize=11)
    ax1.set_title('Power Output – All Profiles', fontsize=12, fontweight='bold')
    ax1.legend(handles=handles, loc='upper left', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Cycle times – all profiles
    ax2 = fig.add_subplot(gs[0, 1])
    for i, profile in enumerate(profiles):
        c = colors[i]
        lbl = f'Profile {profile["profile_id"]}'
        cycleTime = profile['reelOutTime'] + profile['reelInTime']
        ax2.plot(windSpeedRef, cycleTime,
                 color=c, linewidth=2, label=lbl)
        ax2.plot(windSpeedRef, profile['reelOutTime'],
                 color=c, linewidth=1, linestyle='--', alpha=0.6)
        ax2.plot(windSpeedRef, profile['reelInTime'],
                 color=c, linewidth=1, linestyle=':', alpha=0.6)
    _add_region_lines(ax2)

    handles2, labels2 = ax2.get_legend_handles_labels()
    handles2 += [
        Line2D([0], [0], color='gray', linewidth=2, label='Total cycle'),
        Line2D([0], [0], color='gray', linewidth=1, linestyle='--', label='Reel-out'),
        Line2D([0], [0], color='gray', linewidth=1, linestyle=':', label='Reel-in'),
    ]
    ax2.set_xlabel('Wind Speed at ref. height (m/s)', fontsize=11)
    ax2.set_ylabel('Time (s)', fontsize=11)
    ax2.set_title('Cycle Times – All Profiles', fontsize=12, fontweight='bold')
    ax2.legend(handles=handles2, loc='upper right', fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Tether forces – all profiles
    ax3 = fig.add_subplot(gs[1, 0])
    for i, profile in enumerate(profiles):
        c = colors[i]
        lbl = f'Profile {profile["profile_id"]}'
        ax3.plot(windSpeedRef, profile['tetherForceOut'] / 1000,
                 color=c, linewidth=2, label=lbl)
        ax3.plot(windSpeedRef, profile['tetherForceIn'] / 1000,
                 color=c, linewidth=1, linestyle='--', alpha=0.6)

    if model_params and 'nominalTetherForce' in model_params:
        ax3.axhline(y=model_params['nominalTetherForce'] / 1000,
                    color='purple', linestyle=':', alpha=0.5, label='Nominal Force')
    _add_region_lines(ax3)

    handles3, labels3 = ax3.get_legend_handles_labels()
    handles3 += [
        Line2D([0], [0], color='gray', linewidth=2, label='Reel-out force'),
        Line2D([0], [0], color='gray', linewidth=1, linestyle='--', label='Reel-in force'),
    ]
    ax3.set_xlabel('Wind Speed at ref. height (m/s)', fontsize=11)
    ax3.set_ylabel('Tether Force (kN)', fontsize=11)
    ax3.set_title('Tether Forces – All Profiles', fontsize=12, fontweight='bold')
    ax3.legend(handles=handles3, loc='upper left', fontsize=8, ncol=2)
    ax3.grid(True, alpha=0.3)

    # Subplot 4: Reel speeds – all profiles
    ax4 = fig.add_subplot(gs[1, 1])
    ax4_twin = ax4.twinx()

    for i, profile in enumerate(profiles):
        c = colors[i]
        lbl = f'Profile {profile["profile_id"]}'
        ax4.plot(windSpeedRef, profile['reelOutSpeed'],
                 color=c, linewidth=2, label=lbl)
        ax4.plot(windSpeedRef, profile['reelInSpeed'],
                 color=c, linewidth=1, linestyle='--', alpha=0.6)
    _add_region_lines(ax4)

    ax4.set_xlabel('Wind Speed at ref. height (m/s)', fontsize=11)
    ax4.set_ylabel('Reel Speed (m/s)', fontsize=11)
    ax4.tick_params(axis='y', labelcolor='black')
    ax4.grid(True, alpha=0.3)

    if model_params and 'elevationAngleOut' in model_params and 'elevationAngleIn' in model_params:
        elevOut_deg = np.rad2deg(model_params['elevationAngleOut'])
        elevIn_deg = np.rad2deg(model_params['elevationAngleIn'])
        l3 = ax4_twin.plot(windSpeedRef, np.ones_like(windSpeedRef) * elevOut_deg,
                           'g--', linewidth=1.5, alpha=0.5,
                           label=f'Elev Out ({elevOut_deg:.1f}°)')
        l4 = ax4_twin.plot(windSpeedRef, np.ones_like(windSpeedRef) * elevIn_deg,
                           'r--', linewidth=1.5, alpha=0.5,
                           label=f'Elev In ({elevIn_deg:.1f}°)')
        ax4_twin.set_ylabel('Elevation Angle (°)', fontsize=11, color='gray')
        ax4_twin.tick_params(axis='y', labelcolor='gray')
        ax4_twin.legend(handles=l3 + l4, loc='center right', fontsize=8)

    handles4, _ = ax4.get_legend_handles_labels()
    handles4 += [
        Line2D([0], [0], color='gray', linewidth=2, label='Reel-out speed'),
        Line2D([0], [0], color='gray', linewidth=1, linestyle='--', label='Reel-in speed'),
    ]
    ax4.legend(handles=handles4, loc='upper left', fontsize=8, ncol=2)
    ax4.set_title('Reel Speeds & Elevation Angles', fontsize=12, fontweight='bold')

    # Subplot 5: Energy per cycle – all profiles
    ax5 = fig.add_subplot(gs[2, 0])
    for i, profile in enumerate(profiles):
        c = colors[i]
        lbl = f'Profile {profile["profile_id"]}'
        energyOut = profile['reelOutPower'] * profile['reelOutTime'] / 3_600_000
        energyIn = profile['reelInPower'] * profile['reelInTime'] / 3_600_000
        cycleTime = profile['reelOutTime'] + profile['reelInTime']
        cycleEnergy = profile['power'] * cycleTime / 3_600_000
        ax5.plot(windSpeedRef, energyOut,
                 color=c, linewidth=1, linestyle='--', alpha=0.6)
        ax5.plot(windSpeedRef, energyIn,
                 color=c, linewidth=1, linestyle=':', alpha=0.6)
        ax5.plot(windSpeedRef, cycleEnergy,
                 color=c, linewidth=2, label=lbl)
    _add_region_lines(ax5)

    handles5, _ = ax5.get_legend_handles_labels()
    handles5 += [
        Line2D([0], [0], color='gray', linewidth=2, label='Net cycle energy'),
        Line2D([0], [0], color='gray', linewidth=1, linestyle='--', label='Reel-out energy'),
        Line2D([0], [0], color='gray', linewidth=1, linestyle=':', label='Reel-in energy'),
    ]
    ax5.set_xlabel('Wind Speed at ref. height (m/s)', fontsize=11)
    ax5.set_ylabel('Energy per Cycle (kWh)', fontsize=11)
    ax5.set_title('Energy per Cycle – All Profiles', fontsize=12, fontweight='bold')
    ax5.legend(handles=handles5, loc='upper left', fontsize=8, ncol=2)
    ax5.grid(True, alpha=0.3)

    # Subplot 6: Reeling factors (gamma) – all profiles
    ax6 = fig.add_subplot(gs[2, 1])
    for i, profile in enumerate(profiles):
        c = colors[i]
        lbl = f'Profile {profile["profile_id"]}'
        ax6.plot(windSpeedRef, profile['gammaOut'],
                 color=c, linewidth=2, label=lbl)
        ax6.plot(windSpeedRef, profile['gammaIn'],
                 color=c, linewidth=1, linestyle='--', alpha=0.6)
    _add_region_lines(ax6)

    if model_params:
        if 'nominalWindSpeedForce' in model_params:
            ax6.axvline(x=model_params['nominalWindSpeedForce'], color='orange',
                        linestyle=':', alpha=0.5,
                        label=f"Force Limit ({model_params['nominalWindSpeedForce']:.1f} m/s)")
        if ('nominalWindSpeedPower' in model_params and
                'cutOutWindSpeed' in model_params and
                model_params['nominalWindSpeedPower'] < model_params['cutOutWindSpeed']):
            ax6.axvline(x=model_params['nominalWindSpeedPower'], color='red',
                        linestyle=':', alpha=0.5,
                        label=f"Power Limit ({model_params['nominalWindSpeedPower']:.1f} m/s)")

    handles6, labels6 = ax6.get_legend_handles_labels()
    handles6 += [
        Line2D([0], [0], color='gray', linewidth=2, label='γ_out'),
        Line2D([0], [0], color='gray', linewidth=1, linestyle='--', label='γ_in'),
    ]
    ax6.set_xlabel('Wind Speed at ref. height (m/s)', fontsize=11)
    ax6.set_ylabel('Reeling Factor (-)', fontsize=11)
    ax6.set_title('Dimensionless Reeling Factors – All Profiles', fontsize=12, fontweight='bold')
    ax6.legend(handles=handles6, loc='upper right', fontsize=8, ncol=2)
    ax6.grid(True, alpha=0.3)
    
    # Overall title
    title = 'AWE Power Curve Analysis (Luchsinger Model)'
    if model_params and 'wingArea' in model_params and 'nominalGeneratorPower' in model_params:
        title += (f"\n{model_params['wingArea']} m² wing"
                  f" – {n_profiles} wind shear profile{'s' if n_profiles != 1 else ''}")
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show if requested
    if show:
        plt.show()
    
    return fig


def extract_model_params(model) -> Dict:
    """Extract model parameters for plotting.

    Args:
        model: PowerModel instance.

    Returns:
        Dict: Dictionary of model parameters.
    """
    return {
        'wingArea': model.wingArea,
        'nominalGeneratorPower': model.nominalGeneratorPower,
        'nominalTetherForce': model.nominalTetherForce,
        'nominalWindSpeedForce': model.nominalWindSpeedForce,
        'nominalWindSpeedPower': model.nominalWindSpeedPower,
        'cutOutWindSpeed': model.cutOutWindSpeed,
        'elevationAngleOut': model.elevationAngleOut,
        'elevationAngleIn': model.elevationAngleIn,
    }
