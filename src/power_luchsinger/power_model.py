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

import numpy as np
import yaml
from scipy import optimize as op

from src.power_luchsinger.config_loader import (
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

    def _get_wind_speed_at_operational_altitude(
        self,
        reference_wind_speed: float,
        wind_profile: Dict[str, np.ndarray]) -> float:
        """Calculate wind speed at operational altitude using wind shear profile.
        
        Args:
            reference_wind_speed: Wind speed at reference height (m/s).
            wind_profile: Dict with 'altitudes' and 'u_normalized' arrays.
            reference_height_m: Reference altitude where wind_profile = 1.0 (m).
            
        Returns:
            float: Wind speed at operational altitude (m/s).
        """
        altitudes = wind_profile['altitudes']
        u_normalized = wind_profile['u_normalized']
        
        # Interpolate normalized wind speed at operational altitude
        u_norm_at_op = np.interp(self.operationalAltitude, altitudes, u_normalized)
             
        # Scale to get actual wind speed at operational altitude
        # reference_wind_speed corresponds to u_norm_at_ref
        wind_speed_at_op = reference_wind_speed * u_norm_at_op
        
        return wind_speed_at_op

    
    def generate_power_curves(
        self,
        wind_speeds: np.ndarray = None,
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
                - 'reference_height_m': Reference altitude for wind speeds
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
        reference_height_m = wind_shear_data['reference_height_m']
        profiles = wind_shear_data['profiles']

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
                'u_normalized': profile_data['u_normalized']
            }

            # Calculate wind speeds at operational altitude 
            windSpeedsAtOp = np.array([
                self._get_wind_speed_at_operational_altitude(
                    ws_ref, wind_profile
                )
                for ws_ref in windSpeedsAtRef
            ])

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
                if ws_ref < self.cutInWindSpeed or ws_ref > self.cutOutWindSpeed:
                    results.append(dict(_zero))
                    continue

                if wind_speed_regime == 1:
                    result = self._calculate_power_region1(
                        ws_ref
                    )
                    if result['tetherForceOut'] >= self.nominalTetherForce:
                        wind_speed_regime = 2
                        self.nominalWindSpeedForce = ws_ref
                        self.nominalGammaOutForce = result['gammaOut']

                        logger.debug(
                            'Profile %s: regime 1→2 at %.2f m/s',
                            profile_id, ws_ref
                        )

                if wind_speed_regime == 2:
                    result = self._calculate_power_region2(
                        ws_ref
                    )
                    if result['reelOutPower'] >= self.nominalGeneratorPower:
                        wind_speed_regime = 3
                        self.nominalWindSpeedPower = ws_ref
                        self.nominalGammaOutPower = result['gammaOut']
                        self.nominalReelOutSpeed = result['reelOutSpeed']
                        logger.debug(
                            'Profile %s: regime 2→3 at %.2f m/s',
                            profile_id, ws_ref
                        )

                if wind_speed_regime == 3:
                    result = self._calculate_power_region3(ws_ref)

                results.append(result)


            # Collect results for this profile
            profile_curve = {
                'profile_id': profile_id,
                'u_normalized': profile_data['u_normalized'],
                'v_normalized': profile_data['v_normalized'],
                'windSpeedAtRef': windSpeedsAtRef,
                'windSpeedAtOp': windSpeedsAtOp,
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
            'reference_height_m': reference_height_m,
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
            from src.power_luchsinger.plotting import (
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
        if self._verbose:
            print(f"Region 1 optimizer: {result.nit} iterations")
        
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
        if self._verbose:
            print(f"Region 2 optimizer: {result.nit} iterations")

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
        if self._verbose:
            print(f"Region 3 optimizer: {result.nit} iterations")

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
        dragCoefficient = self.dragCoefficientKiteOut
        if self.model == 'luchsinger_extended_const_lod_in':
            rm_out = 0.5 * (self.tetherMinLength + self.tetherMaxLength)
            CD_out = (self.dragCoefficientKiteOut + 0.25 * self.tetherDragCoefficient *
                       self.tetherDiameter * rm_out / self.wingArea)
            E2out = (self.liftCoefficientKiteOut / CD_out)**2
            force_factor_out = liftCoefficient * np.sqrt(1 + 1/E2out) * (1 + E2out)

        elif self.model == 'luchsinger_original':
            force_factor_out = (liftCoefficient**3) / (dragCoefficient**2)

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
            'elevationAngleIn': self.elevationAngleIn,
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
        print(f"  Force Limit Wind Speed: "
              f"{self.nominalWindSpeedForce:.1f} m/s")
        print(f"  Power Limit Wind Speed: "
              f"{self.nominalWindSpeedPower:.1f} m/s")

        print(f"\nWind Shear Configuration:")
        print(f"  Reference Height:       "
              f"{data['reference_height_m']:.1f} m")
        print(f"  Operational Altitude:   "
              f"{data['operational_altitude_m']:.1f} m")
        print(f"  Number of Profiles:     {n_profiles}")

        print(f"\nPower Statistics Across Profiles:")
        for profile in profiles:
            power = profile['power']
            windSpeedRef = profile['windSpeedAtRef']
            windSpeedOp = profile['windSpeedAtOp']
            profile_id = profile['profile_id']

            max_power = np.max(power)
            idx_max = np.argmax(power)

            print(f"\n  Profile {profile_id}:")
            print(f"    Max Power:            {max_power/1000:.2f} kW")
            print(f"    Wind Speed at Max:    {windSpeedRef[idx_max]:.1f}"
                  f" m/s (ref), {windSpeedOp[idx_max]:.1f} m/s (op)")
            print(f"    Avg Speed Ratio:      "
                  f"{np.mean(windSpeedOp/windSpeedRef):.3f}")

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

        reference_height_m = data['reference_height_m']
        altitudes = data.get('altitudes', [])
        profiles = data['profiles']

        # Get reference wind speeds from first profile (same for all)
        reference_wind_speeds = profiles[0]['windSpeedAtRef']

        # Build power curves list for each profile
        power_curves_list = []
        for profile in profiles:
            profile_id = profile['profile_id']

            # Build wind_speed_data: one entry per reference wind speed
            wind_speed_data = []
            n = len(profile['power'])
            for i in range(n):
                power_val = float(profile['power'][i])
                success = power_val > 0.0
                entry = {
                    'wind_speed_m_s': float(reference_wind_speeds[i]),
                    'success': success,
                    'performance': {
                        'power': {
                            'average_cycle_power_w': power_val,
                            'average_reel_out_power_w': float(profile['reelOutPower'][i]),
                            'average_reel_in_power_w': float(profile['reelInPower'][i]),
                        },
                        'timing': {
                            'reel_out_time_s': float(profile['reelOutTime'][i]),
                            'reel_in_time_s': float(profile['reelInTime'][i]),
                            'cycle_time_s': float(profile['reelOutTime'][i] + profile['reelInTime'][i]),
                        },
                        'forces': {
                            'tether_force_out_n': float(profile['tetherForceOut'][i]),
                            'tether_force_in_n': float(profile['tetherForceIn'][i]),
                        },
                        'speeds': {
                            'reel_out_speed_m_s': float(profile['reelOutSpeed'][i]),
                            'reel_in_speed_m_s': float(profile['reelInSpeed'][i]),
                        },
                        'reel factors': {
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
                'reference_height_m': float(reference_height_m),
                'model_config': {
                    'wing_area_m2': float(self.wingArea),
                    'nominal_power_w': float(self.nominalGeneratorPower),
                    'nominal_tether_force_n': float(self.nominalTetherForce),
                    'operating_altitude_m': float(self.operationalAltitude),
                    'tether_length_operational_m': float(self.operationalLength),
                    'cut_in_wind_speed_m_s': float(self.cutInWindSpeed),
                    'cut_out_wind_speed_m_s': float(self.cutOutWindSpeed),
                },
            },
            'altitudes_m': [float(alt) for alt in altitudes],
            'reference_wind_speeds_m_s': [float(v) for v in reference_wind_speeds],
            'power_curves': power_curves_list,
        }

        # Write output file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(output, f, default_flow_style=False, sort_keys=False)

        # Validate output if requested and awesIO is available
        if file_validate:
            try:
                from awesio.validator import validate as awesio_validate
                awesio_validate(input=output_path)
                print(f"  ✓ {output_path.name} validated against system_schema")
            except ImportError:
                print("  awesIO not installed; skipping validation.")
            except Exception as e:
                print(f"  Validation failed: {e}")
