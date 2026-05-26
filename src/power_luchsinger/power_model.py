"""Core power estimation model for airborne wind energy systems based on the Luchsinger model [1].

This module provides the main PowerModel class for calculating power output
of pumping kite systems. The model is planet-agnostic and configurable via
dictionaries or YAML files. Supports both legacy format and awesIO format.

References:
    [1] R.H. Luchsinger: "Pumping cycle kite power". Springer, 2013.
"""

from typing import Dict, Any, Tuple
from pathlib import Path
from datetime import datetime
import logging
import numbers

import numpy as np
import yaml
from scipy import optimize as op

from .config_loader import (
    load_system_config,
    load_wind_resource,
    load_simulation_settings,
)

# Configure logger
logger = logging.getLogger(__name__)


class PowerModel:
    """Calculate power output for airborne wind energy systems.

    This model is configurable via YAML files or
    dictionaries. All physical parameters must be provided through
    configuration.

    The model implements the Luchsinger pumping cycle model [1].
    """


    def __init__(
        self,
        system_config_path,
        wind_resource_path,
        simulation_settings_path,
        validate_file=True,):
        """Initialize the power model with configuration files.

        Args:
            system_config_path (str or Path): Path to system configuration
                YAML file.
            wind_resource_path (str or Path): Path to wind resource YAML file.
            simulation_settings_path (str or Path): Path to simulation
                settings YAML file.
            validate_file (bool): If True, validate YAML files against the
                awesIO schema. Defaults to True.

        Raises:
            ValueError: If required configuration keys are missing.
            ValueError: If parameter values are physically invalid.
        """
        self.system_config_path = Path(system_config_path)
        self.wind_resource_path = Path(wind_resource_path)
        self.simulation_settings_path = Path(simulation_settings_path)

        # Load wind resource profiles
        self.wind_resource = load_wind_resource(
            self.wind_resource_path, validate_file=validate_file)

        # Load and extract all model parameters from system config
        for key, value in load_system_config(
            self.system_config_path,
            validate_file=validate_file,).items():setattr(self, key, value)

        # Load and extract all model parameters from simulation config
        for key, value in load_simulation_settings(
            self.simulation_settings_path).items():setattr(self, key, value)

        # Validate physical constraints
        self._validate_physical_constraints()

        # Compute derived parameters
        self._compute_derived_parameters()

    def _validate_physical_constraints(self) -> None:
        """Validate that parameter values are physically reasonable.

        Raises:
            ValueError: If values are physically invalid.
        """
        if self.wingArea <= 0:
            raise ValueError("Wing area must be positive")
        if self.liftCoefficientKiteOut <= 0:
            raise ValueError("Lift coefficient must be positive")
        if self.dragCoefficientKiteOut <= 0:
            raise ValueError("Drag coefficient must be positive")

        if self.tetherMaxLength <= self.tetherMinLength:
            raise ValueError("Max tether length must exceed min length")

        if self.airDensity <= 0:
            raise ValueError("Air density must be positive")

        if self.nominalTetherForce <= 0:
            raise ValueError("Nominal tether force must be positive")
        if self.nominalGeneratorPower <= 0:
            raise ValueError("Nominal generator power must be positive")

        if self.cutInWindSpeed >= self.cutOutWindSpeed:
            raise ValueError("Cut-in wind speed must be less than cut-out")

    def _compute_derived_parameters(self) -> None:
        """Compute derived parameters from base configuration."""
        # Operational tether length
        self.operationalLength = (
            (self.tetherMaxLength - self.tetherMinLength) / 2 +
            self.tetherMinLength
        )
        self.reelingLength = self.tetherMaxLength - self.tetherMinLength
        
        # Operational altitude (approximate altitude of kite during operation)
        # Use reel-out elevation angle as representative
        self.operationalAltitude = self.operationalLength * np.sin(self.elevationAngleOut)

        # Force factors
        self.forceFactorOut = self._calculate_force_factor_out()
        self.forceFactorIn = self._calculate_force_factor_in()
        self.e2in = (
            self.liftCoefficientKiteIn / self.dragCoefficientKiteIn
        )**2

    def _get_average_wind_speed(
        self,
        reference_wind_speed: float,
        wind_profile: Dict[str, np.ndarray]) -> float:
        """Calculate average wind speed over operational altitude band.
        
        Args:
            reference_wind_speed: Wind speed at reference height (m/s).
            wind_profile: Dict with 'altitudes' and 'u_normalized' arrays.
            
        Returns:    
            float: Average wind speed over operational altitude band (m/s).
        """
        altitudes = np.asarray(wind_profile['altitudes'], dtype=float)
        u_normalized = np.asarray(wind_profile['u_normalized'], dtype=float)

        altitude_min = self.tetherMinLength * np.sin(self.elevationAngleOut)
        altitude_max = self.tetherMaxLength * np.sin(self.elevationAngleOut)


        samples = np.linspace(altitude_min, altitude_max, 200)
        u_norm_samples = np.interp(samples, altitudes, u_normalized)
        average_u_norm = np.trapezoid(u_norm_samples, samples) / (altitude_max - altitude_min)

        return reference_wind_speed * float(average_u_norm)

    def _get_selected_profiles(self, selected_profiles: list = None) -> list:
        """Return selected wind profiles by index.

        Args:
            selected_profiles: List of profile indices to simulate. If None,
                all profiles are returned.

        Returns:
            list: Selected profile dictionaries from ``self.wind_resource``.

        Raises:
            IndexError: If any requested profile index is out of range.
        """
        profiles = self.wind_resource['profiles']

        if selected_profiles is None:
            return profiles

        selected_profile_data = []
        n_profiles = len(profiles)

        for profile_index in selected_profiles:
            idx = int(profile_index)
            if idx < 0 or idx >= n_profiles:
                raise IndexError(
                    f"Profile index {idx} out of range [0, {n_profiles - 1}]"
                )
            selected_profile_data.append(profiles[idx])

        return selected_profile_data

    def simulate_cycle_at_one_wind_speed(
        self,
        ws_ref: float,
        selected_profiles: list = None,
        verbose: bool = False
    ) -> float:
        """Calculate power output for a single wind speed and profile.

        Args:
            ws_ref: Wind speed at reference height (m/s).
            selected_profiles: Optional list of profile indices. If None,
                simulate all profiles.

        Returns:
            float: Power output (W).
        """
        self._verbose = verbose
        wind_shear_data = self.wind_resource
        profiles = self._get_selected_profiles(selected_profiles)
        # print(profiles[:]['id'])
        results = []
        for profile_data in profiles:
            if self._verbose:
                print(f"Simulating cycle for profile {profile_data['id']} at reference wind speed {ws_ref} m/s")
            profile_id = profile_data['id']

            # Prepare wind profile for interpolation
            wind_profile = {
                'altitudes': wind_shear_data['altitudes'],
                'u_normalized': profile_data['u_normalized']}

            # Initialise regime tracking for this profile.  Wind speeds are
            # processed in ascending order so transitions are detected
            # naturally: after detecting a transition the same wind speed is
            # re-evaluated in the new regime (if/if/if fall-through
            wind_speed_regime = 1
            self.nominalWindSpeedForce = None
            self.nominalGammaOutForce = None 
            self.nominalWindSpeedPower = None
            self.nominalGammaOutPower = None
            self.nominalReelOutSpeed = None


            _zero = {
                'cyclePower': 0.0, 'reelOutPower': 0.0, 'reelInPower': 0.0,
                'reelOutTime': 0.0, 'reelInTime': 0.0,
                'tetherForceOut': 0.0, 'tetherForceIn': 0.0,
                'reelOutSpeed': 0.0, 'reelInSpeed': 0.0,
                'gammaOut': 0.0, 'gammaIn': 0.0, 'elevationAngleOut': 0.0, 'elevationAngleIn': 0.0
            }
            

            
        
            ws_avg = self._get_average_wind_speed(ws_ref, wind_profile)
            if wind_speed_regime == 1:
                result = self._calculate_power_region1(
                    ws_avg
                )
                if result['tetherForceOut'] >= self.nominalTetherForce:
                    wind_speed_regime = 2
                    self.nominalWindSpeedForce = ws_avg
                    self.nominalGammaOutForce = result['gammaOut']

                    logger.debug(
                        'Profile %s: regime 1→2 at %.2f m/s',
                        profile_id, ws_avg
                    )

            if wind_speed_regime == 2:
                result = self._calculate_power_region2(
                    ws_avg
                )
                if result['reelOutPower'] >= self.nominalGeneratorPower:
                    wind_speed_regime = 3
                    self.nominalWindSpeedPower = ws_avg
                    self.nominalGammaOutPower = result['gammaOut']
                    self.nominalReelOutSpeed = result['reelOutSpeed']
                    logger.debug(
                        'Profile %s: regime 2→3 at %.2f m/s',
                        profile_id, ws_avg
                    )

            if wind_speed_regime == 3:
                result = self._calculate_power_region3(ws_avg)

            results.append(result)

        return results


    def generate_power_curves(
        self,
        wind_speeds: np.ndarray = None,
        selected_profiles: list = None,
        output_path: Path = None,
        verbose: bool = False,
        show_plot: bool = False,
        save_plot: bool = False,
        validate_file: bool = True,) -> Dict[str, Any]:
        """Generate power curves for multiple wind shear profiles.

        Calculates power curves for each wind profile/cluster stored in
        ``self.wind_resource``.  Optionally exports results in awesIO
        format, prints a summary, and creates a comprehensive plot.

        Args:
            wind_speeds (np.ndarray): Optional array of reference wind speeds
                (m/s) for which to compute the power curves. When provided,
                overrides the ``num_points`` setting in the simulation settings
                file. When ``None``, a linearly-spaced array is built from
                ``cut_in`` to ``cut_out`` using ``power_curve.num_points``
                from the simulation settings (default 100).
            selected_profiles: Optional list of profile indices. If None,
                simulate all profiles.
            output_path (Path): If given, export power curves to this YAML
                file in awesIO format.
            verbose (bool): If True, print a summary of the results.
                Defaults to False.
            show_plot (bool): If True, display the comprehensive analysis
                plot. Defaults to False.
            save_plot (bool): If True, save the plot next to *output_path*
                (or in ``results/``). Defaults to False.
            validate_file (bool): If True, validate the exported YAML
                against the awesIO schema. Defaults to True.

        Returns:
            Dict with:
                - 'reference_height': Reference altitude for wind speeds
                - 'operational_altitude_m': Operational altitude of kite
                - 'profiles': List of dicts, each containing:
                    - 'profile_id': Profile/cluster ID
                    - 'windSpeedAtRef': Wind speed at reference height (m/s)
                    - 'windSpeedAtOp': Wind speed at operational height (m/s)
                    - 'power': Cycle power (W)
                    - ... other power curve variables
        """
        self._verbose = verbose
 
        wind_shear_data = self.wind_resource
        reference_height = wind_shear_data['reference_height']
        profiles = self._get_selected_profiles(selected_profiles)

        # Wind speeds at reference height
        if wind_speeds is not None:
            windSpeedsAtRef = np.asarray(wind_speeds, dtype=float)
        else:
            windSpeedsAtRef = np.linspace(
                self.cutInWindSpeed,
                self.cutOutWindSpeed,
                self.numPoints
            )

        power_curves = []

        for profile_data in profiles:
            profile_id = profile_data['id']

            # Prepare wind profile for interpolation
            wind_profile = {
                'altitudes': wind_shear_data['altitudes'],
                'u_normalized': profile_data['u_normalized']}

            # Initialise regime tracking for this profile.  Wind speeds are
            # processed in ascending order so transitions are detected
            # naturally: after detecting a transition the same wind speed is
            # re-evaluated in the new regime (if/if/if fall-through
            wind_speed_regime = 1
            self.nominalWindSpeedForce = None
            self.nominalGammaOutForce = None 
            self.nominalWindSpeedPower = None
            self.nominalGammaOutPower = None
            self.nominalReelOutSpeed = None


            _zero = {
                'cyclePower': 0.0, 'reelOutPower': 0.0, 'reelInPower': 0.0,
                'reelOutTime': 0.0, 'reelInTime': 0.0,
                'tetherForceOut': 0.0, 'tetherForceIn': 0.0,
                'reelOutSpeed': 0.0, 'reelInSpeed': 0.0,
                'gammaOut': 0.0, 'gammaIn': 0.0, 'elevationAngleOut': 0.0, 'elevationAngleIn': 0.0
            }
            results = []

            for ws_ref in windSpeedsAtRef:
        
                ws_avg = self._get_average_wind_speed(ws_ref, wind_profile)
                if wind_speed_regime == 1:
                    result = self._calculate_power_region1(
                        ws_avg
                    )
                    if result['tetherForceOut'] >= self.nominalTetherForce:
                        wind_speed_regime = 2
                        self.nominalWindSpeedForce = ws_avg
                        self.nominalGammaOutForce = result['gammaOut']

                        logger.debug(
                            'Profile %s: regime 1→2 at %.2f m/s',
                            profile_id, ws_avg
                        )

                if wind_speed_regime == 2:
                    result = self._calculate_power_region2(
                        ws_avg
                    )
                    if result['reelOutPower'] >= self.nominalGeneratorPower:
                        wind_speed_regime = 3
                        self.nominalWindSpeedPower = ws_avg
                        self.nominalGammaOutPower = result['gammaOut']
                        self.nominalReelOutSpeed = result['reelOutSpeed']
                        logger.debug(
                            'Profile %s: regime 2→3 at %.2f m/s',
                            profile_id, ws_avg
                        )

                if wind_speed_regime == 3:
                    result = self._calculate_power_region3(ws_avg)

                results.append(result)


            # Collect results for this profile
            profile_curve = {
                'profile_id': profile_id,
                'u_normalized': profile_data['u_normalized'],
                'v_normalized': profile_data['v_normalized'],
                'windSpeedAtRef': windSpeedsAtRef,
                'power': np.array([r['cyclePower'] for r in results]),
                'reelOutPower': np.array([r['reelOutPower'] for r in results]),
                'reelInPower': np.array([r['reelInPower'] for r in results]),
                'reelOutTime': np.array([r['reelOutTime'] for r in results]),
                'reelInTime': np.array([r['reelInTime'] for r in results]),
                'tetherForceOut': np.array([r['tetherForceOut'] for r in results]),
                'tetherForceIn': np.array([r['tetherForceIn'] for r in results]),
                'reelOutSpeed': np.array([r['reelOutSpeed'] for r in results]),
                'reelInSpeed': np.array([r['reelInSpeed'] for r in results]),
                'gammaOut': np.array([r['gammaOut'] for r in results]),
                'gammaIn': np.array([r['gammaIn'] for r in results]),
                'elevationAngleOut': np.array([r['elevationAngleOut'] for r in results]),
                'elevationAngleIn': np.array([r['elevationAngleIn'] for r in results])

            }

            power_curves.append(profile_curve)

        data = {
            'reference_height': reference_height,
            'operational_altitude_m': self.operationalAltitude,
            'altitudes': wind_shear_data['altitudes'],
            'profiles': power_curves
        }

        # Print summary
        if verbose:
            self._print_summary(data)

        # Export to awesIO YAML
        if output_path is not None:
            self.export_power_curves_awesio(
                data, output_path, file_validate=validate_file
            )
            if verbose:
                print(f"\nExported power curves to: {output_path}")

        # Plot
        if show_plot or save_plot:
            from .plotting import (
                plot_comprehensive_analysis,
                extract_model_params,
            )
            save_path = None
            if save_plot and output_path is not None:
                save_path = str(
                    output_path.parent / "power_curve_analysis.pdf"
                )
            elif save_plot:
                save_path = "results/power_curve_analysis.pdf"

            plot_comprehensive_analysis(
                data, extract_model_params(self),
                save_path=save_path,
                show=show_plot,
            )
            if verbose and save_path:
                print(f"Plot saved to: {save_path}")

        return data

    def simulate_single_wind_speed(
        self,
        ws_ref: float,
        profile_index: int,
        output_path: Path = None,
        verbose: bool = False,
        show_plot: bool = False,
        save_plot: bool = False,
        validate_file: bool = True,
    ) -> Dict[str, Any]:
        """Calculate a power curve for a single wind speed and a single profile.

        Produces the same output structure as ``generate_power_curves`` but
        for exactly one reference wind speed and one wind shear profile.

        Args:
            ws_ref (float): Reference wind speed (m/s).
            profile_index (int): Index of the wind shear profile to simulate.
            output_path (Path): If given, export the result to this YAML file
                in awesIO format.
            verbose (bool): If True, print a summary of the results.
                Defaults to False.
            show_plot (bool): If True, display the analysis plot.
                Defaults to False.
            save_plot (bool): If True, save the plot next to *output_path*
                (or in ``results/``). Defaults to False.
            validate_file (bool): If True, validate the exported YAML against
                the awesIO schema. Defaults to True.

        Returns:
            Dict with:
                - 'reference_height': Reference altitude for wind speeds
                - 'operational_altitude_m': Operational altitude of kite
                - 'profiles': List with a single dict containing:
                    - 'profile_id': Profile/cluster ID
                    - 'windSpeedAtRef': Array with one wind speed value (m/s)
                    - 'power': Array with one cycle power value (W)
                    - ... other power curve variables

        Raises:
            IndexError: If *profile_index* is out of range.
        """
        self._verbose = verbose

        wind_shear_data = self.wind_resource
        reference_height = wind_shear_data['reference_height']

        # Resolve single profile
        profiles = self.wind_resource['profiles']
        n_profiles = len(profiles)
        idx = int(profile_index)
        if idx < 0 or idx >= n_profiles:
            raise IndexError(
                f"Profile index {idx} out of range [0, {n_profiles - 1}]"
            )
        profile_data = profiles[idx]
        profile_id = profile_data['id']

        wind_profile = {
            'altitudes': wind_shear_data['altitudes'],
            'u_normalized': profile_data['u_normalized'],
        }

        # Initialise regime tracking
        wind_speed_regime = 1
        self.nominalWindSpeedForce = None
        self.nominalGammaOutForce = None
        self.nominalWindSpeedPower = None
        self.nominalGammaOutPower = None
        self.nominalReelOutSpeed = None

        ws_avg = self._get_average_wind_speed(ws_ref, wind_profile)

        if wind_speed_regime == 1:
            result = self._calculate_power_region1(ws_avg)
            if result['tetherForceOut'] >= self.nominalTetherForce:
                wind_speed_regime = 2
                self.nominalWindSpeedForce = ws_avg
                self.nominalGammaOutForce = result['gammaOut']
                logger.debug(
                    'Profile %s: regime 1→2 at %.2f m/s', profile_id, ws_avg
                )

        if wind_speed_regime == 2:
            result = self._calculate_power_region2(ws_avg)
            if result['reelOutPower'] >= self.nominalGeneratorPower:
                wind_speed_regime = 3
                self.nominalWindSpeedPower = ws_avg
                self.nominalGammaOutPower = result['gammaOut']
                self.nominalReelOutSpeed = result['reelOutSpeed']
                logger.debug(
                    'Profile %s: regime 2→3 at %.2f m/s', profile_id, ws_avg
                )

        if wind_speed_regime == 3:
            result = self._calculate_power_region3(ws_avg)

        windSpeedsAtRef = np.array([ws_ref])

        profile_curve = {
            'profile_id': profile_id,
            'u_normalized': profile_data['u_normalized'],
            'v_normalized': profile_data['v_normalized'],
            'windSpeedAtRef': windSpeedsAtRef,
            'power': np.array([result['cyclePower']]),
            'reelOutPower': np.array([result['reelOutPower']]),
            'reelInPower': np.array([result['reelInPower']]),
            'reelOutTime': np.array([result['reelOutTime']]),
            'reelInTime': np.array([result['reelInTime']]),
            'tetherForceOut': np.array([result['tetherForceOut']]),
            'tetherForceIn': np.array([result['tetherForceIn']]),
            'reelOutSpeed': np.array([result['reelOutSpeed']]),
            'reelInSpeed': np.array([result['reelInSpeed']]),
            'gammaOut': np.array([result['gammaOut']]),
            'gammaIn': np.array([result['gammaIn']]),
            'elevationAngleOut': np.array([result['elevationAngleOut']]),
            'elevationAngleIn': np.array([result['elevationAngleIn']]),
        }

        data = {
            'reference_height': reference_height,
            'operational_altitude_m': self.operationalAltitude,
            'altitudes': wind_shear_data['altitudes'],
            'profiles': [profile_curve],
        }

        if verbose:
            self._print_summary(data)

        if output_path is not None:
            self.export_power_curves_awesio(
                data, output_path, file_validate=validate_file
            )
            if verbose:
                print(f"\nExported result to: {output_path}")

        if show_plot or save_plot:
            print('There is no plotting function for single wind speed simulation. Skipping plot.')

        return data

    def _calculate_power_region1(self, windSpeed: float) -> Dict[str, float]:
        """Calculate power in Region 1 (below force limit).

        Args:
            windSpeed (float): Wind speed at reference height in m/s.

        Returns:
            Dict with power and time details.
        """

        gammaOut, gammaIn = self._optimize_gamma_out_in_region1(windSpeed)

        return self._calculate_cycle_results(1,windSpeed, gammaOut, gammaIn)

    def _optimize_gamma_out_in_region1(self,
                                        windSpeed: float) -> Tuple[float, float]:
        """Calculate optimal dimensionless reeling velocity factors.
        
        Optimizes the cycle power factor by finding the optimal reeling
        velocities for both reel-out and reel-in phases.
        
        Args:

            windSpeed (float): Wind speed at reference height in m/s.
            
        Returns:
            Tuple[float, float]: (optimal gamma_out, optimal gamma_in).
        """

        gammaOutMax = min(self.reelOutSpeedLimit / windSpeed, 1.0)


        from scipy import optimize as op
        
        if self.model == 'luchsinger_extended_const_lod_in':
            gammaInMin = max(self.reelInSpeedLimit / windSpeed, -np.sqrt(1+1/self.e2in))
            def objective(x):
                gammaOut, gammaIn = x
                powerFactor = (
                    (np.cos(self.elevationAngleOut) - gammaOut)**2 -
                    (self.forceFactorIn / self.forceFactorOut) *
                    ( (
                        self._extended_sqrt_term(gammaIn) - gammaIn
                    )**2 / (1 + self.e2in))
                ) * ((gammaIn * gammaOut) / (gammaIn - gammaOut))
                return -powerFactor

            bounds = ((0.001, gammaOutMax), (gammaInMin, -0.001))
            start = (0.001, -0.001)
        elif self.model == 'luchsinger_original':
            gammaInMin = self.reelInSpeedLimit / windSpeed
            # In original luchsinger paper, gammaIn is defined as positive during reel-in, 
            # but we define it as negative for consistency. 
            def objective(x):
                gammaOut, gammaIn = x[0], -x[1]
                powerFactor = (
                    (np.cos(self.elevationAngleOut) - gammaOut)**2 -
                    (self.forceFactorIn / self.forceFactorOut) *
                    (1 + 2 * np.cos(self.elevationAngleIn) * gammaIn + gammaIn**2)
                ) * ((gammaOut * gammaIn) / (gammaOut + gammaIn))
                return -powerFactor

            bounds = ((0.001, gammaOutMax), (gammaInMin, -0.001))
            start = (0.001, -0.001)

        result = op.minimize(objective, start, bounds=bounds, method='SLSQP')
  
        return result['x'][0], result['x'][1]

    def _calculate_power_region2(self,
                                  windSpeed: float) -> Dict[str, float]:
        """Calculate power in Region 2 (force-limited, below power limit).

        Args:
            windSpeed (float): Wind speed at reference height in m/s.

        Returns:
            Dict with power and time details.
        """

        mu = windSpeed / self.nominalWindSpeedForce

        gammaOut = (
            np.cos(self.elevationAngleOut) -
            (np.cos(self.elevationAngleOut) - self.nominalGammaOutForce) / mu)
        gammaIn = self._optimize_gamma_in_region2(mu, windSpeed)

        return self._calculate_cycle_results(2, windSpeed, gammaOut, gammaIn)

    def _optimize_gamma_in_region2(
        self,
        mu: float,
        windSpeed: float) -> float:
        """Optimize gamma_in for Region 2 operation.

        Args:
            mu (float): Wind speed ratio to nominal force wind speed.
            windSpeed (float): Wind speed at reference height in m/s.

        Returns:
            float: Optimal gamma_in.
        """
        if self.model == 'luchsinger_extended_const_lod_in':
            gammaInMin = max(self.reelInSpeedLimit / windSpeed, -np.sqrt(1+1/self.e2in))
            def objective(x):
                gammaIn = x[0]
                b = (
                    (mu - 1) * np.cos(self.elevationAngleOut)
                    + self.nominalGammaOutForce
                )
                powerFactor = (
                    ((
                        np.cos(self.elevationAngleOut)
                        - self.nominalGammaOutForce
                    ) / mu)**2
                    - (self.forceFactorIn / self.forceFactorOut)
                    * ((
                        self._extended_sqrt_term(gammaIn) - gammaIn
                    )**2 / (1 + self.e2in))
                ) * gammaIn * b / (mu * gammaIn - b)
                return -powerFactor

            result = op.minimize(
                objective,
                [-0.001],
                bounds=[(gammaInMin, -0.001)],
                method='SLSQP',
            )
        elif self.model == 'luchsinger_original':
            gammaInMin = self.reelInSpeedLimit / windSpeed
            # In original luchsinger paper, gammaIn is defined as positive during reel-in, 
            # but we define it as negative for consistency. 
            def objective(x):
                gammaIn = -x[0]
                gammaOutEff = (
                    mu * np.cos(self.elevationAngleOut) -
                    np.cos(self.elevationAngleOut) +
                    self.nominalGammaOutForce
                )

                powerFactor = (
                    (1 / mu**2) * (np.cos(self.elevationAngleOut) - self.nominalGammaOutForce)**2 -
                    (self.forceFactorIn / self.forceFactorOut) *
                    (1 + 2 * np.cos(self.elevationAngleIn) * gammaIn + gammaIn**2)
                ) * (
                    (gammaIn * gammaOutEff) /
                    (mu * gammaIn + gammaOutEff)
                )
                return -powerFactor

            result = op.minimize(
                objective,
                [-0.001],
                bounds=[(gammaInMin, -0.001)],
                method='SLSQP',
            )
        return result['x'][0]

    def _calculate_power_region3(self, windSpeed: float) -> Dict[str, float]:
        """Calculate power in Region 3 (power-limited).

        Args:
            windSpeed (float): Wind speed at reference height in m/s.

        Returns:
            Dict with power and time details.
        """

        mu = windSpeed / self.nominalWindSpeedPower

        vOut = self.nominalReelOutSpeed
        gammaOut = vOut / windSpeed
        gammaIn = self._optimize_gamma_in_region3(mu, windSpeed)

        return self._calculate_cycle_results(3, windSpeed, gammaOut, gammaIn)

    def _optimize_gamma_in_region3(
        self,
        mu: float,
        windSpeed: float) -> float:
        """Optimize gamma_in for Region 3 operation.

        Args:
            mu (float): Wind speed ratio to nominal power wind speed.
            windSpeed (float): Wind speed at reference height in m/s.

        Returns:
            float: Optimal gamma_in.
        """
        if self.model == 'luchsinger_extended_const_lod_in':
            gammaInMin = max(self.reelInSpeedLimit / windSpeed, -np.sqrt(1+1/self.e2in))
            def objective(x):
                gammaIn = x[0]
                powerFactor = (
                    (
                        np.cos(self.elevationAngleOut)
                        - self.nominalGammaOutPower / mu
                    )**2
                    - (self.forceFactorIn / self.forceFactorOut)
                    * ((
                        self._extended_sqrt_term(gammaIn) - gammaIn
                    )**2 / (1 + self.e2in))
                ) * (
                    gammaIn * self.nominalGammaOutPower /
                    (mu * gammaIn - self.nominalGammaOutPower)
                )
                return -powerFactor

            result = op.minimize(
                objective,
                [-0.001],
                bounds=[(gammaInMin, -0.001)],
                method='SLSQP',
            )
        elif self.model == 'luchsinger_original':
            gammaInMin = self.reelInSpeedLimit / windSpeed
            # In original luchsinger paper, gammaIn is defined as positive during reel-in, 
            # but we define it as negative for consistency. 
            def objective(x):
                gammaIn = -x[0]

                powerFactor = (
                    (1 / mu**2) *
                    (np.cos(self.elevationAngleOut) - self.nominalGammaOutForce)**2 -
                    (self.forceFactorIn / self.forceFactorOut) *
                    (1 + 2 * np.cos(self.elevationAngleIn) * gammaIn + gammaIn**2)
                ) * (
                    (self.nominalGammaOutPower * gammaIn) /
                    (self.nominalGammaOutPower + mu * gammaIn)
                )
                return -powerFactor

            result = op.minimize(
                objective,
                [-0.001],
                bounds=[(gammaInMin, -0.001)],
                method='SLSQP',
            )
        return result['x'][0]



    def _calculate_force_factor_out(self) -> float:
        """Calculate the dimensionless force factor for reel-out phase.
        
        This factor characterizes the kite's power generation capability
        during the reel-out phase.
        
        Args:
        Returns:
            float: Force factor f_out
        """
        liftCoefficient = self.liftCoefficientKiteOut

        rm_out = 0.5 * (self.tetherMinLength + self.tetherMaxLength)
        CD_out = (self.dragCoefficientKiteOut + 0.25 * self.tetherDragCoefficient *
                       self.tetherDiameter * rm_out / self.wingArea)
        if self.model == 'luchsinger_extended_const_lod_in':
            E2out = (self.liftCoefficientKiteOut / CD_out)**2
            force_factor_out = liftCoefficient * np.sqrt(1 + 1/E2out) * (1 + E2out)

        elif self.model == 'luchsinger_original':
            force_factor_out = (liftCoefficient**3) / (CD_out**2)

        return force_factor_out

    def _calculate_force_factor_in(self) -> float:
        """Calculate the dimensionless force factor for reel-in phase.
        
        During reel-in, the kite is depowered and the force is proportional
        to drag only.
        
        Args:
        Returns:
            float: Force factor f_in
        """
        liftCoefficient = self.liftCoefficientKiteIn
        dragCoefficient = self.dragCoefficientKiteIn
        if self.model == 'luchsinger_extended_const_lod_in':
            E2in   = (liftCoefficient  / dragCoefficient)**2
            force_factor_in  = liftCoefficient  * np.sqrt(1+1/E2in)
        elif self.model == 'luchsinger_original':
            force_factor_in = dragCoefficient
            
        return force_factor_in

    def _calculate_tether_force_out(self, windSpeed: float, gammaOut: float) -> float:
        """Calculate tether force during reel-out phase.
        
        Uses the Luchsinger model formulation for tether force.
        
        Args:
            windSpeed (float): Wind speed in m/s.
            gammaOut (float): Dimensionless reel-out velocity (v_out / v_wind).
            
        Returns:
            float: Tether force in N.
        """
        airDensity = self.airDensity
        wingArea = self.wingArea
        elevationAngle = self.elevationAngleOut
        forceFactor = self.forceFactorOut
        if self.model == 'luchsinger_extended_const_lod_in':
            effectiveWindFactor = (np.cos(elevationAngle) - gammaOut)**2
            tetherForce = 0.5 * airDensity * windSpeed**2 * wingArea * effectiveWindFactor * forceFactor

        elif self.model == 'luchsinger_original':
            effectiveWindFactor = (np.cos(elevationAngle) - gammaOut)**2
            tetherForce = 0.5 * airDensity * windSpeed**2 * wingArea * effectiveWindFactor * forceFactor
        
        return max(0.0, tetherForce)

    def _calculate_tether_force_in(self, windSpeed: float, gammaIn: float) -> float:
        """Calculate tether force during reel-in phase.
        
        Uses the Luchsinger model formulation for tether force during retraction.
        
        Args:
            windSpeed (float): Wind speed in m/s.
            gammaIn (float): Dimensionless reel-in velocity (v_in / v_wind).
            
        Returns:
            float: Tether force in N.
        """
        airDensity = self.airDensity
        wingArea = self.wingArea
        forceFactor = self.forceFactorIn
        if self.model == 'luchsinger_extended_const_lod_in':
            sqrt_term = self._extended_sqrt_term(gammaIn)
            effectiveWindFactor = ((sqrt_term - gammaIn)**2) / (1 + self.e2in)
            tetherForce = 0.5 * airDensity * windSpeed**2 * wingArea * effectiveWindFactor * forceFactor

        elif self.model == 'luchsinger_original':
            gammaIn = -gammaIn  # Convert to positive for original model
            effectiveWindFactor = 1 + 2 * np.cos(self.elevationAngleIn) * gammaIn + gammaIn**2
            tetherForce = 0.5 * airDensity * windSpeed**2 * wingArea * effectiveWindFactor * forceFactor
        
        return max(0.0, tetherForce)


    def _calculate_cycle_results(self, region: int, windSpeed: float, gammaOut: float, gammaIn: float) -> dict:
        """Calculate complete cycle results including power, time, and forces.
        
        This function encapsulates all power and time calculations for a
        complete pumping cycle.
        
        Args:
            region (int): The wind speed region.
            windSpeed (float): Wind speed in m/s.
            gammaOut (float): Dimensionless reel-out velocity factor.
            gammaIn (float): Dimensionless reel-in velocity factor.
        Returns:
            dict: Dictionary with complete cycle results.
        """

        vOut = windSpeed * gammaOut
        vIn = windSpeed * gammaIn

        # Reel-out phase
        if region == 1:
            tetherForceOut = self._calculate_tether_force_out(windSpeed, gammaOut)
        else:
            tetherForceOut = self.nominalTetherForce
        if region == 3:
            vOut = self.nominalReelOutSpeed
            gammaOut = vOut / windSpeed
        mechPower = tetherForceOut * vOut
        timeOut = self.reelingLength / vOut if vOut > 0 else float('inf')
        energyOut = mechPower * timeOut

        # Reel-in phase
        tetherForceIn = self._calculate_tether_force_in(windSpeed, gammaIn)
        mechPower = tetherForceIn * vIn
        timeIn = self.reelingLength / abs(vIn) if vIn != 0 else float('inf')
        energyIn = mechPower * timeIn

        elecEnergyOut = energyOut * self.generatorEfficiency
        elecEnergyIn = energyIn / self.generatorEfficiency
        elecPowerOut = elecEnergyOut / timeOut if timeOut > 0 else 0.0
        elecPowerIn = elecEnergyIn / timeIn if timeIn > 0 else 0.0

        cycleTime = timeOut + timeIn
        netEnergy = elecEnergyOut + (elecEnergyIn / self.storageEfficiency)
        cyclePower = netEnergy / cycleTime if cycleTime > 0 else 0.0

        if self.model == 'luchsinger_extended_const_lod_in':
            elevationAngleIn = self._get_extended_beta_in(gammaIn)
        else:
            elevationAngleIn = self.elevationAngleIn

        if self.model == 'luchsinger_extended_const_lod_in':
            # Recalculate gammaIn based on the extended const LoD-in model for reporting
            self.elevationAngleIn = self._get_extended_beta_in(gammaIn)

        return {
            'cyclePower': cyclePower,
            'reelOutPower': elecPowerOut,
            'reelInPower': elecPowerIn,
            'reelOutTime': timeOut,
            'reelInTime': timeIn,
            'tetherForceOut': tetherForceOut,
            'tetherForceIn': tetherForceIn,
            'reelOutSpeed': vOut,
            'reelInSpeed': vIn,
            'gammaOut': gammaOut,
            'gammaIn': gammaIn,
            'elevationAngleOut': self.elevationAngleOut,
            'elevationAngleIn': elevationAngleIn,
        }

    def _extended_sqrt_term(self, gamma_in: float) -> float:
        """Return sqrt term used by the extended const LoD-in equations."""
        value = 1 + self.e2in * (1 - gamma_in**2)
        return np.sqrt(np.maximum(0.0, value))

    def _get_extended_beta_in(self, gamma_in: float) -> float:
        """Compute reel-in elevation angle for constant LoD-in model.

        Args:
            gamma_in (float): Signed reel-in factor (negative during reel-in).

        Returns:
            float: Reel-in elevation angle in radians.
        """
        sqrt_term = self._extended_sqrt_term(gamma_in)
        arg = (sqrt_term + gamma_in * self.e2in) / (1 + self.e2in)
        arg = np.clip(arg, -1.0, 1.0)
        return np.arccos(arg)



    def _print_summary(self, data: Dict[str, Any]) -> None:
        """Print a summary of the power curve calculation.

        Args:
            data (dict): Power curve data from generate_power_curves_with_shear.
        """
        profiles = data['profiles']
        n_profiles = len(profiles)

        print("\n" + "=" * 60)
        print("POWER CURVE SUMMARY WITH WIND SHEAR")
        print("=" * 60)
        print(f"\nSystem Parameters:")
        print(f"  Wing Area:              {self.wingArea:.1f} m²")
        print(f"  Air Density:            {self.airDensity:.3f} kg/m³")
        print(f"  Nominal Tether Force:   {self.nominalTetherForce:.0f} N")
        print(f"  Nominal Generator Power:{self.nominalGeneratorPower/1000:.1f} kW")
        print(f"  Tether Length:          {self.tetherMinLength:.0f}"
              f" - {self.tetherMaxLength:.0f} m")

        print(f"\nOperational Envelope:")
        print(f"  Cut-in Wind Speed:      {self.cutInWindSpeed:.1f} m/s"
              f" (at reference height)")
        print(f"  Cut-out Wind Speed:     {self.cutOutWindSpeed:.1f} m/s"
              f" (at reference height)")
        force_ws = (f"{self.nominalWindSpeedForce:.1f} m/s"
                    if self.nominalWindSpeedForce is not None else "N/A")
        power_ws = (f"{self.nominalWindSpeedPower:.1f} m/s"
                    if self.nominalWindSpeedPower is not None else "N/A")
        print(f"  Force Limit Wind Speed: {force_ws}")
        print(f"  Power Limit Wind Speed: {power_ws}")

        print(f"\nWind Shear Configuration:")
        print(f"  Reference Height:       "
              f"{data['reference_height']:.1f} m")
        print(f"  Operational Altitude:   "
              f"{data['operational_altitude_m']:.1f} m")
        print(f"  Number of Profiles:     {n_profiles}")


        print("=" * 60)

    def export_power_curves_awesio(
        self,
        data: Dict[str, Any],
        output_path: Path,
        name: str = "Luchsinger Power Curves with Wind Shear",
        description: str = "Power curves for pumping ground-gen AWE system with wind shear",
        note: str = "Power curve data generated from Luchsinger model with wind shear profiles",
        file_validate: bool = True,) -> None:
        """Export power curve data with wind shear profiles in awesIO format.

        Args:
            data: Power curve data from generate_power_curves_with_shear().
            output_path: Path to save the output YAML file.
            name: Name for the power curves dataset.
            description: Description of the power curves.
            note: Additional notes about the data.
            file_validate: Whether to validate the output file. Defaults to True.
        """
        output_path = Path(output_path)

        reference_height = data['reference_height']
        altitudes = data.get('altitudes', [])
        profiles = data['profiles']
        wind_resource_metadata = self.wind_resource.get('metadata', {})

        # Get reference wind speeds from first profile (same for all)
        reference_wind_speeds = profiles[0]['windSpeedAtRef']

        wind_resource_info = {
            'reference_height': float(reference_height),
        }
        for key in ('n_clusters', 'location', 'time_range', 'data_source'):
            value = wind_resource_metadata.get(key)
            if value is not None:
                wind_resource_info[key] = value

        # Build power curves list for each profile
        power_curves_list = []
        for profile in profiles:
            profile_id = profile['profile_id']

            # Build wind_speed_data: one entry per reference wind speed
            wind_speed_data = []
            n = len(profile['power'])
            for i in range(n):
                power_val = float(profile['power'][i])
                entry = {
                    'wind_speed': float(reference_wind_speeds[i]),
                    'performance': {
                        'power': {
                            'average_cycle_power': power_val,
                            'average_reel_out_power': float(profile['reelOutPower'][i]),
                            'average_reel_in_power': float(profile['reelInPower'][i]),
                        },
                        'timing': {
                            'reel_out_time': float(profile['reelOutTime'][i]),
                            'reel_in_time': float(profile['reelInTime'][i]),
                            'cycle_time': float(profile['reelOutTime'][i] + profile['reelInTime'][i]),
                        },
                        'forces': {
                            'tether_force_out': float(profile['tetherForceOut'][i]),
                            'tether_force_in': float(profile['tetherForceIn'][i]),
                        },
                        'speeds': {
                            'reel_out_speed': float(profile['reelOutSpeed'][i]),
                            'reel_in_speed': float(profile['reelInSpeed'][i]),
                        },
                        'reel_factors': {
                            'gamma_out': float(profile['gammaOut'][i]),
                            'gamma_in': float(profile['gammaIn'][i]),
                        },
                        'elevation_angles': {
                            'elevation_angle_out_rad': float(profile['elevationAngleOut'][i]),
                            'elevation_angle_in_rad': float(profile['elevationAngleIn'][i]),
                        },

                    },
                }
                wind_speed_data.append(entry)

            power_curve = {
                'profile_id': int(profile_id),
                'probability_weight': 1.0 / len(profiles),
                'wind_profile': {
                    'u_normalized': [float(u) for u in profile['u_normalized']],
                    'v_normalized': [float(v) for v in profile['v_normalized']],
                },
                'wind_speed_data': wind_speed_data,
            }
            power_curves_list.append(power_curve)

        # Build awesIO format output
        output = {
            'metadata': {
                'name': name,
                'description': description,
                'note': note,
                'awesIO_version': '0.1.0',
                'schema': 'power_curves_schema.yml',
                'time_created': datetime.now().isoformat(),
                'model_config': {
                    'wing_area': float(self.wingArea),
                    'nominal_power': float(self.nominalGeneratorPower),
                    'nominal_tether_force': float(self.nominalTetherForce),
                    'operating_altitude': float(self.operationalAltitude),
                    'tether_length_operational': float(self.operationalLength),
                    'cut_in_wind_speed': float(self.cutInWindSpeed),
                    'cut_out_wind_speed': float(self.cutOutWindSpeed),
                },
                'wind_resource': wind_resource_info,
            },
            'altitudes': [float(alt) for alt in altitudes],
            'reference_wind_speeds': [float(v) for v in reference_wind_speeds],
            'power_curves': power_curves_list,
        }

        # Write output file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        class _NumericFlowStyleDumper(yaml.SafeDumper):
            """Dump numeric sequences in flow style while preserving structure."""

        def _represent_sequence_with_numeric_flow(dumper, data):
            is_numeric_sequence = (
                len(data) > 0 and
                all(
                    isinstance(item, numbers.Real) and
                    not isinstance(item, bool)
                    for item in data
                )
            )
            return dumper.represent_sequence(
                'tag:yaml.org,2002:seq',
                data,
                flow_style=is_numeric_sequence,
            )

        _NumericFlowStyleDumper.add_representer(
            list,
            _represent_sequence_with_numeric_flow,
        )
        _NumericFlowStyleDumper.add_representer(
            tuple,
            _represent_sequence_with_numeric_flow,
        )

        with open(output_path, 'w') as f:
            yaml.dump(
                output,
                f,
                Dumper=_NumericFlowStyleDumper,
                default_flow_style=False,
                sort_keys=False,
            )

        # Validate output if requested and awesIO is available
        if file_validate:
            try:
                from awesio.validator import validate as awesio_validate
                awesio_validate(input=output_path)
                print(f"  ✓ {output_path.name} validated against power_curves_schema")
            except ImportError:
                print("  awesIO not installed; skipping validation.")
            except Exception as e:
                print(f"  Validation failed: {e}")
