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
import copy
import logging

import numpy as np
import yaml
from scipy import optimize as op

try:
    from awesio.validator import validate as awesio_validate
    AWESIO_AVAILABLE = True
except ImportError:
    AWESIO_AVAILABLE = False
    awesio_validate = None

from src.power_luchsinger.calculations import (
    calculate_force_factor_out,
    calculate_force_factor_in,
    calculate_tether_force_out,
    calculate_tether_force_in,
    calculate_cycle_results
)

# Configure logger
logger = logging.getLogger(__name__)


def load_wind_shear_profiles(wind_resource_path: Path) -> Dict[str, Any]:
    """Load wind shear profiles from awesIO wind resource YAML file.
    
    Extracts normalized wind profiles for each cluster, which represent
    different atmospheric conditions and shear characteristics.
    
    Args:
        wind_resource_path: Path to wind resource YAML file.
        
    Returns:
        Dict containing:
            - 'altitudes': Array of altitudes in meters
            - 'reference_height_m': Reference altitude where normalized profiles = 1.0
            - 'profiles': List of dicts, each with:
                - 'id': Profile/cluster ID
                - 'u_normalized': Normalized horizontal wind speed profile
                - 'v_normalized': Normalized vertical wind speed profile
            - 'n_clusters': Number of profiles/clusters
            
    Raises:
        FileNotFoundError: If wind resource file not found.
        ValueError: If required keys are missing from the file.
    """
    if not wind_resource_path.exists():
        raise FileNotFoundError(f"Wind resource file not found: {wind_resource_path}")
    
    with open(wind_resource_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Extract metadata
    metadata = data.get('metadata', {})
    n_clusters = metadata.get('n_clusters')
    reference_height_m = metadata.get('reference_height_m')
    
    if n_clusters is None:
        raise ValueError("'n_clusters' not found in wind resource metadata")
    if reference_height_m is None:
        raise ValueError("'reference_height_m' not found in wind resource metadata")
    
    # Extract altitudes
    altitudes = np.array(data.get('altitudes', []))
    if len(altitudes) == 0:
        raise ValueError("'altitudes' array is empty or missing")
    
    # Extract clusters/profiles
    clusters = data.get('clusters', [])
    if len(clusters) != n_clusters:
        raise ValueError(f"Expected {n_clusters} clusters, found {len(clusters)}")
    
    profiles = []
    for cluster in clusters:
        profile = {
            'id': cluster.get('id'),
            'u_normalized': np.array(cluster.get('u_normalized', [])),
            'v_normalized': np.array(cluster.get('v_normalized', []))
        }
        
        # Validate profile data
        if len(profile['u_normalized']) != len(altitudes):
            raise ValueError(f"Profile {profile['id']}: u_normalized length mismatch")
        if len(profile['v_normalized']) != len(altitudes):
            raise ValueError(f"Profile {profile['id']}: v_normalized length mismatch")
        
        profiles.append(profile)
    
    return {
        'altitudes': altitudes,
        'reference_height_m': reference_height_m,
        'profiles': profiles,
        'n_clusters': n_clusters
    }


class PowerModel:
    """Calculate power output for airborne wind energy systems.

    This model is configurable via YAML files or
    dictionaries. All physical parameters must be provided through
    configuration.

    The model implements the Luchsinger pumping cycle model [1].
    """


    def __init__(self, config: Dict[str, Any], simulation_settings: Dict[str, Any] = None):
        """Initialize power model with configuration parameters.

        Args:
            config: Dictionary containing model parameters (legacy format)
                or awesIO system configuration.
            simulation_settings: Optional simulation settings dictionary
                for awesIO format configs. If None, default values will be used.

        Raises:
            ValueError: If required configuration keys are missing.
            ValueError: If parameter values are physically invalid.
        """
        self.config = config
        
        # Use default simulation settings if none provided
        if simulation_settings is None:
            logger.warning(
                "No simulation settings provided. Using default values: "
                f"{self._get_default_simulation_settings()}"
            )
            self.simulation_settings = self._get_default_simulation_settings()
        else:
            self.simulation_settings = simulation_settings

        # Extract parameters (handles both legacy and awesIO formats)
        self._extract_parameters()

        # Then validate physical constraints
        self._validate_physical_constraints()

        # Finally compute derived parameters
        self._compute_derived_parameters()

    def _validate_physical_constraints(self) -> None:
        """Validate that parameter values are physically reasonable.

        Raises:
            ValueError: If values are physically invalid.
        """
        if self.wingArea <= 0:
            raise ValueError("Wing area must be positive")
        if self.liftCoefficientOut <= 0:
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

    @staticmethod
    def _get_default_simulation_settings() -> Dict[str, Any]:
        """Get default simulation settings.
        
        Returns:
            Dictionary with default operational and atmosphere parameters.
        """
        return {
            'operational': {
                'cut_in_wind_speed_m_s': 4.0,
                'cut_out_wind_speed_m_s': 25.0,
                'elevation_angle_out_deg': 30.0,
                'elevation_angle_in_deg': 45.0,
                'minimum_tether_length_m': 100.0,
            },
            'atmosphere': {
                'air_density_kg_m3': 1.225,
            }
        }

    def _extract_parameters(self) -> None:
        """Extract parameters directly from awesIO format config."""
        components = self.config.get('components', {})

        # Extract operational and atmosphere parameters from simulation settings
        operational = self.simulation_settings.get('operational', {})
        atmosphere = self.simulation_settings.get('atmosphere', {})
        
        # Extract operational parameters
        self.cutInWindSpeed = operational.get('cut_in_wind_speed_m_s')
        self.cutOutWindSpeed = operational.get('cut_out_wind_speed_m_s')
        self.elevationAngleOut = np.radians(operational.get('elevation_angle_out_deg'))
        self.elevationAngleIn = np.radians(operational.get('elevation_angle_in_deg'))
        self.tetherMinLength = operational.get('minimum_tether_length_m')

        # Extract atmosphere parameters
        self.airDensity = atmosphere.get('air_density_kg_m3')

        # Extract wing parameters
        wing = components.get('wing', {})
        wing_structure = wing.get('structure', {})
        wing_aero = wing.get('aerodynamics', {}).get('simple_aero_model', {})
        
        self.wingArea = wing_structure.get('projected_surface_area_m2')
        self.liftCoefficientOut = wing_aero.get('lift_coefficient_reel_out')
        self.dragCoefficientKiteOut = wing_aero.get('drag_coefficient_reel_out')
        self.dragCoefficientKiteIn = wing_aero.get('drag_coefficient_reel_in')

        # Extract tether parameters
        tether = components.get('tether', {})
        tether_structure = tether.get('structure', {})       
        self.tetherMaxLength = tether_structure.get('length_m')


        # Extract ground station parameters
        ground_station = components.get('ground_station', {})
        drum = ground_station.get('drum', {})
        generator = ground_station.get('generator', {})
        storage = ground_station.get('storage', {})
        self.reelOutSpeedLimit = drum.get('max_tether_speed_m_s')
        self.reelInSpeedLimit = drum.get('max_tether_speed_m_s')
        self.nominalTetherForce = drum.get('max_tether_force_n') or tether_structure.get('max_tether_force_n')
        self.nominalGeneratorPower = generator.get('rated_power_kw', 0) * 1000  # kW to W
        self.generatorEfficiency = generator.get('efficiency')
        self.storageEfficiency = storage.get('efficiency')


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

        # Total drag coefficients
        self.dragCoefficientOut = self.dragCoefficientKiteOut
        self.dragCoefficientIn = self.dragCoefficientKiteIn

        # Force factors
        self.forceFactorOut = calculate_force_factor_out(
            self.liftCoefficientOut, self.dragCoefficientOut
        )
        self.forceFactorIn = calculate_force_factor_in(self.dragCoefficientIn)

        # Compute nominal wind speeds for force and power limits
        self._compute_nominal_wind_speeds()

    def _compute_nominal_wind_speeds(self) -> None:
        """Compute wind speeds at which force and power limits are reached."""
        windSpeeds = np.arange(self.cutInWindSpeed, self.cutOutWindSpeed, 0.1)

        self.nominalWindSpeedForce = self.cutOutWindSpeed
        self.nominalAvgWindSpeedForce = self.cutOutWindSpeed  # Same as reference when no shear
        self.nominalGammaOutForce = 0.33

        self.nominalWindSpeedPower = self.cutOutWindSpeed
        self.nominalGammaOutPower = 0.33

        # Find force limit
        for vw in windSpeeds:
            gammaOutMax = self.reelOutSpeedLimit / vw
            gammaInMax = self.reelInSpeedLimit / vw

            gammaOut, gammaIn = self._optimize_gamma_out_in_region1(
                self.elevationAngleOut, self.elevationAngleIn,
                self.forceFactorOut, self.forceFactorIn,
                gammaOutMax, gammaInMax
            )

            tetherForce = calculate_tether_force_out(
                self.airDensity, vw, self.wingArea,
                gammaOut, self.elevationAngleOut, self.forceFactorOut
            )

            if tetherForce >= self.nominalTetherForce:
                self.nominalWindSpeedForce = vw
                self.nominalAvgWindSpeedForce = vw  # Same as reference when no shear
                self.nominalGammaOutForce = gammaOut
                break

        # Find power limit (only for winds above force limit)
        for vw in windSpeeds:
            if vw <= self.nominalWindSpeedForce:
                continue

            mu = vw / self.nominalAvgWindSpeedForce
            gammaOut = (
                np.cos(self.elevationAngleOut) -
                (np.cos(self.elevationAngleOut) - self.nominalGammaOutForce) / mu
            )
            vOut = vw * gammaOut

            # Simple power calculation
            mechPower = self.nominalTetherForce * vOut
            elecPower = mechPower * self.generatorEfficiency

            if elecPower >= self.nominalGeneratorPower:
                self.nominalWindSpeedPower = vw
                self.nominalGammaOutPower = gammaOut
                break

        # Compute nominal reel-out speed for power-limited region
        self.nominalReelOutSpeed = (
            self.nominalGeneratorPower / 
            (self.nominalTetherForce * self.generatorEfficiency)
        )

    def _compute_nominal_wind_speeds_with_shear(
        self,
        wind_profile: Dict[str, np.ndarray],
        reference_height_m: float
    ) -> None:
        """Recompute nominal wind speeds using segmented wind shear calculations.
        
        This method recalculates the force and power limit wind speeds using
        segmented wind speeds along the tether deployment.
        
        Args:
            wind_profile: Dict with 'altitudes' and 'u_normalized' arrays.
            reference_height_m: Reference altitude where wind_profile = 1.0 (m).
        """
        windSpeeds = np.arange(self.cutInWindSpeed, self.cutOutWindSpeed, 0.1)

        self.nominalWindSpeedForce = self.cutOutWindSpeed
        self.nominalAvgWindSpeedForce = self.cutOutWindSpeed  # Average wind speed at force limit
        self.nominalGammaOutForce = 0.33

        self.nominalWindSpeedPower = self.cutOutWindSpeed
        self.nominalGammaOutPower = 0.33

        # Find force limit with segmented calculations
        for vw in windSpeeds:
            # Get segmented wind speeds
            windSpeedsOut = self.get_segmented_wind_speeds(
                vw, wind_profile, reference_height_m, self.elevationAngleOut, n_segments=20
            )
            avgWindSpeed = np.mean(windSpeedsOut)
            
            gammaOutMax = self.reelOutSpeedLimit / avgWindSpeed
            gammaInMax = self.reelInSpeedLimit / avgWindSpeed

            gammaOut, gammaIn = self._optimize_gamma_out_in_region1(
                self.elevationAngleOut, self.elevationAngleIn,
                self.forceFactorOut, self.forceFactorIn,
                gammaOutMax, gammaInMax
            )

            # Calculate average tether force across segments
            tetherForces = np.array([
                calculate_tether_force_out(
                    self.airDensity, ws, self.wingArea,
                    gammaOut, self.elevationAngleOut, self.forceFactorOut
                )
                for ws in windSpeedsOut
            ])
            tetherForce = np.mean(tetherForces)

            if tetherForce >= self.nominalTetherForce:
                self.nominalWindSpeedForce = vw
                self.nominalAvgWindSpeedForce = avgWindSpeed  # Store average wind speed
                self.nominalGammaOutForce = gammaOut
                break

        # Find power limit (only for winds above force limit)
        for vw in windSpeeds:
            if vw <= self.nominalWindSpeedForce:
                continue

            windSpeedsOut = self.get_segmented_wind_speeds(
                vw, wind_profile, reference_height_m, self.elevationAngleOut, n_segments=20
            )
            avgWindSpeed = np.mean(windSpeedsOut)
            
            mu = avgWindSpeed / self.nominalAvgWindSpeedForce  # Use average wind speed
            gammaOut = (
                np.cos(self.elevationAngleOut) -
                (np.cos(self.elevationAngleOut) - self.nominalGammaOutForce) / mu
            )
            vOut = avgWindSpeed * gammaOut

            # Simple power calculation
            mechPower = self.nominalTetherForce * vOut
            elecPower = mechPower * self.generatorEfficiency

            if elecPower >= self.nominalGeneratorPower:
                self.nominalWindSpeedPower = vw
                self.nominalGammaOutPower = gammaOut
                break

        # Compute nominal reel-out speed for power-limited region
        self.nominalReelOutSpeed = (
            self.nominalGeneratorPower / 
            (self.nominalTetherForce * self.generatorEfficiency)
        )

    def get_wind_speed_at_operational_altitude(
        self,
        reference_wind_speed: float,
        wind_profile: Dict[str, np.ndarray],
        reference_height_m: float
    ) -> float:
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

    def get_segmented_wind_speeds(
        self,
        reference_wind_speed: float,
        wind_profile: Dict[str, np.ndarray],
        reference_height_m: float,
        elevation_angle: float,
        n_segments: int = 20
    ) -> np.ndarray:
        """Calculate wind speeds for segmented tether deployment.
        
        Divides the reeling length into n_segments and calculates the wind speed
        at the average altitude of each segment.
        
        Args:
            reference_wind_speed: Wind speed at reference height (m/s).
            wind_profile: Dict with 'altitudes' and 'u_normalized' arrays.
            reference_height_m: Reference altitude where wind_profile = 1.0 (m).
            elevation_angle: Tether elevation angle (radians).
            n_segments: Number of segments to divide the reeling phase into.
            
        Returns:
            np.ndarray: Wind speeds for each segment (m/s).
        """
        altitudes = wind_profile['altitudes']
        u_normalized = wind_profile['u_normalized']
        
        # Tether lengths at segment boundaries (from min to max)
        tether_lengths = np.linspace(self.tetherMinLength, self.tetherMaxLength, n_segments + 1)
        
        # Calculate average altitude for each segment
        segment_altitudes = np.zeros(n_segments)
        for i in range(n_segments):
            # Average tether length for this segment
            avg_tether_length = (tether_lengths[i] + tether_lengths[i+1]) / 2
            # Average altitude = average tether length * sin(elevation angle)
            segment_altitudes[i] = avg_tether_length * np.sin(elevation_angle)
        
        # Interpolate normalized wind speed at each segment altitude
        u_norm_at_segments = np.interp(segment_altitudes, altitudes, u_normalized)
        
        # Interpolate normalized wind speed at reference height
        u_norm_at_ref = np.interp(reference_height_m, altitudes, u_normalized)
        
        # Scale to get actual wind speeds at segment altitudes
        wind_speeds = reference_wind_speed * (u_norm_at_segments / u_norm_at_ref)
        
        return wind_speeds


    def calculate_power(self,
                       windSpeed: float,
                       airDensity: float = None,
                       wind_profile: Dict[str, np.ndarray] = None,
                       reference_height_m: float = None) -> Dict[str, float]:
        """Calculate power output for given wind speed.

        Args:
            windSpeed (float): Wind speed at reference height in m/s.
            airDensity (float): Air density in kg/m³. If None, uses
                atmosphere.airDensity from config.
            wind_profile (Dict): Optional wind shear profile with 'altitudes' 
                and 'u_normalized' arrays. If provided, uses segmented calculation.
            reference_height_m (float): Reference height for wind_profile (m).

        Returns:
            Dict[str, float]: Dictionary with keys:
                - 'cyclePower': Average cycle power (W)
                - 'reelOutPower': Reel-out power (W)
                - 'reelInPower': Reel-in power (W)
                - 'reelOutTime': Reel-out time (s)
                - 'reelInTime': Reel-in time (s)
                - 'tetherForceOut': Tether force during reel-out (N)
                - 'tetherForceIn': Tether force during reel-in (N)
                - 'reelOutSpeed': Reel-out speed (m/s)
                - 'reelInSpeed': Reel-in speed (m/s)
                - 'gammaOut': Reel-out factor (-)
                - 'gammaIn': Reel-in factor (-)
        """
        if airDensity is None:
            airDensity = self.airDensity

        if windSpeed < self.cutInWindSpeed or windSpeed > self.cutOutWindSpeed:
            return {
                'cyclePower': 0.0,
                'reelOutPower': 0.0,
                'reelInPower': 0.0,
                'reelOutTime': 0.0,
                'reelInTime': 0.0,
                'tetherForceOut': 0.0,
                'tetherForceIn': 0.0,
                'reelOutSpeed': 0.0,
                'reelInSpeed': 0.0,
                'gammaOut': 0.0,
                'gammaIn': 0.0,
            }

        if windSpeed < self.nominalWindSpeedForce:
            return self._calculate_power_region1(windSpeed, airDensity, wind_profile, reference_height_m)
        elif windSpeed < self.nominalWindSpeedPower:
            return self._calculate_power_region2(windSpeed, airDensity, wind_profile, reference_height_m)
        else:
            return self._calculate_power_region3(windSpeed, airDensity, wind_profile, reference_height_m)

    def _calculate_power_region1(self,
                                  windSpeed: float,
                                  airDensity: float,
                                  wind_profile: Dict[str, np.ndarray] = None,
                                  reference_height_m: float = None) -> Dict[str, float]:
        """Calculate power in Region 1 (below force limit).

        Args:
            windSpeed (float): Wind speed at reference height in m/s.
            airDensity (float): Air density in kg/m³.
            wind_profile (Dict): Optional wind shear profile.
            reference_height_m (float): Reference height for wind_profile (m).

        Returns:
            Dict with power and time details.
        """
        # Get segmented wind speeds if profile provided
        if wind_profile is not None and reference_height_m is not None:
            n_segments = 20
            windSpeedsOut = self.get_segmented_wind_speeds(
                windSpeed, wind_profile, reference_height_m, self.elevationAngleOut, n_segments=n_segments
            )
            windSpeedsIn = self.get_segmented_wind_speeds(
                windSpeed, wind_profile, reference_height_m, self.elevationAngleIn, n_segments=n_segments
            )
            # Use average wind speed for gamma optimization
            avgWindSpeedOut = np.mean(windSpeedsOut)
            avgWindSpeedIn = np.mean(windSpeedsIn)
        else:
            # Use single wind speed (backward compatibility)
            avgWindSpeedOut = windSpeed
            avgWindSpeedIn = windSpeed
            windSpeedsOut = None
            windSpeedsIn = None
        
        gammaOutMax = self.reelOutSpeedLimit / avgWindSpeedOut
        gammaInMax = self.reelInSpeedLimit / avgWindSpeedIn

        gammaOut, gammaIn = self._optimize_gamma_out_in_region1(
            self.elevationAngleOut, self.elevationAngleIn,
            self.forceFactorOut, self.forceFactorIn,
            gammaOutMax, gammaInMax
        )

        vOut = avgWindSpeedOut * gammaOut
        vIn = avgWindSpeedIn * gammaIn

        # Calculate with segmentation if wind profile provided
        if windSpeedsOut is not None and windSpeedsIn is not None:
            # Segment-by-segment calculation
            segment_length = self.reelingLength / n_segments
            
            # Reel-out phase: calculate energy for each segment
            energyOut = 0.0
            totalForceOut = 0.0
            for ws in windSpeedsOut:
                # Tether force for this segment
                force = calculate_tether_force_out(
                    airDensity, ws, self.wingArea,
                    gammaOut, self.elevationAngleOut, self.forceFactorOut
                )
                totalForceOut += force
                
                # Mechanical power in this segment
                mechPower = force * vOut
                
                # Time in this segment
                time_segment = segment_length / vOut if vOut > 0 else float('inf')
                
                # Energy generated in this segment
                energyOut += mechPower * time_segment
            
            # Reel-in phase: calculate energy for each segment
            energyIn = 0.0
            totalForceIn = 0.0
            for ws in windSpeedsIn:
                # Tether force for this segment
                force = calculate_tether_force_in(
                    airDensity, ws, self.wingArea,
                    gammaIn, self.elevationAngleIn, self.forceFactorIn
                )
                totalForceIn += force
                
                # Mechanical power in this segment (negative - consuming energy)
                mechPower = force * vIn
                
                # Time in this segment
                time_segment = segment_length / vIn if vIn > 0 else float('inf')
                
                # Energy consumed in this segment
                energyIn += mechPower * time_segment
            
            # Average forces for reporting
            tetherForceOut = totalForceOut / n_segments
            tetherForceIn = totalForceIn / n_segments
            
            # Total times
            timeOut = self.reelingLength / vOut if vOut > 0 else float('inf')
            timeIn = self.reelingLength / vIn if vIn > 0 else float('inf')
            
            # Electrical energies
            elecEnergyOut = energyOut * self.generatorEfficiency
            elecEnergyIn = energyIn / self.generatorEfficiency
            
            # Average electrical powers
            elecPowerOut = elecEnergyOut / timeOut if timeOut > 0 else 0.0
            elecPowerIn = elecEnergyIn / timeIn if timeIn > 0 else 0.0
            
            # Cycle power
            cycleTime = timeOut + timeIn
            netEnergy = elecEnergyOut - (elecEnergyIn / self.storageEfficiency)
            cyclePower = netEnergy / cycleTime if cycleTime > 0 else 0.0
            
            return {
                'cyclePower': max(0.0, cyclePower),
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
            }
        else:
            # Original non-segmented calculation
            tetherForceOut = calculate_tether_force_out(
                airDensity, windSpeed, self.wingArea,
                gammaOut, self.elevationAngleOut, self.forceFactorOut
            )
            tetherForceIn = calculate_tether_force_in(
                airDensity, windSpeed, self.wingArea,
                gammaIn, self.elevationAngleIn, self.forceFactorIn
            )

            return calculate_cycle_results(
                tetherForceOut, tetherForceIn, vOut, vIn,
                self.reelingLength, gammaOut, gammaIn,
                self.generatorEfficiency, self.storageEfficiency
            )

    def _optimize_gamma_out_in_region1(self,
                                        elevationAngleOut: float,
                                        elevationAngleIn: float,
                                        forceFactorOut: float,
                                        forceFactorIn: float,
                                        gammaOutMax: float,
                                        gammaInMax: float) -> Tuple[float, float]:
        """Calculate optimal dimensionless reeling velocity factors.
        
        Optimizes the cycle power factor by finding the optimal reeling
        velocities for both reel-out and reel-in phases.
        
        Args:
            elevationAngleOut (float): Elevation angle during reel-out in rad.
            elevationAngleIn (float): Elevation angle during reel-in in rad.
            forceFactorOut (float): Force factor during reel-out.
            forceFactorIn (float): Force factor during reel-in.
            gammaOutMax (float): Maximum gamma_out (v_out_max / v_wind).
            gammaInMax (float): Maximum gamma_in (v_in_max / v_wind).
            
        Returns:
            Tuple[float, float]: (optimal gamma_out, optimal gamma_in).
        """
        from scipy import optimize as op
        
        def objective(x):
            gammaOut, gammaIn = x
            # Cycle power factor from Luchsinger model
            powerFactor = (
                (np.cos(elevationAngleOut) - gammaOut)**2 -
                (forceFactorIn / forceFactorOut) *
                (1 + 2 * np.cos(elevationAngleIn) * gammaIn + gammaIn**2)
            ) * ((gammaOut * gammaIn) / (gammaOut + gammaIn))
            return -powerFactor  # Minimize negative = maximize
        
        bounds = ((0.001, gammaOutMax), (0.001, gammaInMax))
        result = op.minimize(objective, (0.001, 0.001), bounds=bounds, method='SLSQP')
        
        return result['x'][0], result['x'][1]

    def _calculate_power_region2(self,
                                  windSpeed: float,
                                  airDensity: float,
                                  wind_profile: Dict[str, np.ndarray] = None,
                                  reference_height_m: float = None) -> Dict[str, float]:
        """Calculate power in Region 2 (force-limited, below power limit).

        Args:
            windSpeed (float): Wind speed at reference height in m/s.
            airDensity (float): Air density in kg/m³.
            wind_profile (Dict): Optional wind shear profile.
            reference_height_m (float): Reference height for wind_profile (m).

        Returns:
            Dict with power and time details.
        """
        # Get segmented wind speeds if profile provided
        if wind_profile is not None and reference_height_m is not None:
            n_segments = 20
            windSpeedsOut = self.get_segmented_wind_speeds(
                windSpeed, wind_profile, reference_height_m, self.elevationAngleOut, n_segments=n_segments
            )
            windSpeedsIn = self.get_segmented_wind_speeds(
                windSpeed, wind_profile, reference_height_m, self.elevationAngleIn, n_segments=n_segments
            )
            avgWindSpeedOut = np.mean(windSpeedsOut)
            avgWindSpeedIn = np.mean(windSpeedsIn)
        else:
            avgWindSpeedOut = windSpeed
            avgWindSpeedIn = windSpeed
            windSpeedsOut = None
            windSpeedsIn = None
        
        # Use average wind speed for mu calculation (accounts for wind shear)
        nominalAvgWind = getattr(self, 'nominalAvgWindSpeedForce', self.nominalWindSpeedForce)
        mu = avgWindSpeedOut / nominalAvgWind
        gammaInMax = self.reelInSpeedLimit / avgWindSpeedIn

        gammaOut = (
            np.cos(self.elevationAngleOut) -
            (np.cos(self.elevationAngleOut) - self.nominalGammaOutForce) / mu
        )
        vOut = avgWindSpeedOut * gammaOut

        gammaIn = self._optimize_gamma_in_region2(mu, gammaInMax)
        vIn = avgWindSpeedIn * gammaIn

        # Calculate with segmentation if wind profile provided
        if windSpeedsOut is not None and windSpeedsIn is not None:
            segment_length = self.reelingLength / n_segments
            
            # Reel-out phase: force-limited (constant reel-out speed)
            # Calculate actual forces per segment with varying wind speeds
            energyOut = 0.0
            totalForceOut = 0.0
            timeOut = self.reelingLength / vOut if vOut > 0 else float('inf')
            
            for ws in windSpeedsOut:
                # Calculate tether force for this segment
                # gammaOut is constant (vOut / avgWindSpeedOut), but local gamma varies
                gammaOut_local = vOut / ws
                force = calculate_tether_force_out(
                    airDensity, ws, self.wingArea,
                    gammaOut_local, self.elevationAngleOut, self.forceFactorOut
                )
                totalForceOut += force
                
                # Mechanical power in this segment
                mechPower = force * vOut
                
                # Time in this segment (same for all segments since vOut is constant)
                time_segment = segment_length / vOut if vOut > 0 else float('inf')
                
                # Energy generated in this segment
                energyOut += mechPower * time_segment
            
            tetherForceOut = totalForceOut / n_segments
            
            # Reel-in phase: calculate energy for each segment
            energyIn = 0.0
            totalForceIn = 0.0
            for ws in windSpeedsIn:
                # Tether force for this segment
                force = calculate_tether_force_in(
                    airDensity, ws, self.wingArea,
                    gammaIn, self.elevationAngleIn, self.forceFactorIn
                )
                totalForceIn += force
                
                # Mechanical power in this segment
                mechPower = force * vIn
                
                # Time in this segment
                time_segment = segment_length / vIn if vIn > 0 else float('inf')
                
                # Energy consumed in this segment
                energyIn += mechPower * time_segment
            
            tetherForceIn = totalForceIn / n_segments
            timeIn = self.reelingLength / vIn if vIn > 0 else float('inf')
            
            # Electrical energies
            elecEnergyOut = energyOut * self.generatorEfficiency
            elecEnergyIn = energyIn / self.generatorEfficiency
            
            # Average electrical powers
            elecPowerOut = elecEnergyOut / timeOut if timeOut > 0 else 0.0
            elecPowerIn = elecEnergyIn / timeIn if timeIn > 0 else 0.0
            
            # Cycle power
            cycleTime = timeOut + timeIn
            netEnergy = elecEnergyOut - (elecEnergyIn / self.storageEfficiency)
            cyclePower = netEnergy / cycleTime if cycleTime > 0 else 0.0
            
            return {
                'cyclePower': max(0.0, cyclePower),
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
            }
        else:
            # Original non-segmented calculation
            tetherForceOut = self.nominalTetherForce
            tetherForceIn = calculate_tether_force_in(
                airDensity, windSpeed, self.wingArea,
                gammaIn, self.elevationAngleIn, self.forceFactorIn
            )

            return calculate_cycle_results(
                tetherForceOut, tetherForceIn, vOut, vIn,
                self.reelingLength, gammaOut, gammaIn,
                self.generatorEfficiency, self.storageEfficiency
            )

    def _optimize_gamma_in_region2(self, mu: float, gammaInMax: float) -> float:
        """Optimize gamma_in for Region 2 operation.

        Args:
            mu (float): Wind speed ratio to nominal force wind speed.
            gammaInMax (float): Maximum gamma_in.

        Returns:
            float: Optimal gamma_in.
        """
        def objective(x):
            gammaIn = x[0]
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
            objective, [0.001],
            bounds=[(0.001, gammaInMax)],
            method='SLSQP'
        )

        return result['x'][0]

    def _calculate_power_region3(self,
                                  windSpeed: float,
                                  airDensity: float,
                                  wind_profile: Dict[str, np.ndarray] = None,
                                  reference_height_m: float = None) -> Dict[str, float]:
        """Calculate power in Region 3 (power-limited).

        Args:
            windSpeed (float): Wind speed at reference height in m/s.
            airDensity (float): Air density in kg/m³.
            wind_profile (Dict): Optional wind shear profile.
            reference_height_m (float): Reference height for wind_profile (m).

        Returns:
            Dict with power and time details.
        """
        # Get segmented wind speeds if profile provided
        if wind_profile is not None and reference_height_m is not None:
            n_segments = 20
            windSpeedsOut = self.get_segmented_wind_speeds(
                windSpeed, wind_profile, reference_height_m, self.elevationAngleOut, n_segments=n_segments
            )
            windSpeedsIn = self.get_segmented_wind_speeds(
                windSpeed, wind_profile, reference_height_m, self.elevationAngleIn, n_segments=n_segments
            )
            avgWindSpeedOut = np.mean(windSpeedsOut)
            avgWindSpeedIn = np.mean(windSpeedsIn)
        else:
            avgWindSpeedOut = windSpeed
            avgWindSpeedIn = windSpeed
            windSpeedsIn = None
        
        mu = avgWindSpeedOut / self.nominalWindSpeedPower
        gammaInMax = self.reelInSpeedLimit / avgWindSpeedIn

        vOut = self.nominalReelOutSpeed
        gammaOut = vOut / avgWindSpeedOut

        gammaIn = self._optimize_gamma_in_region3(mu, gammaInMax)
        vIn = avgWindSpeedIn * gammaIn

        # Calculate with segmentation if wind profile provided
        if windSpeedsIn is not None:
            segment_length = self.reelingLength / n_segments
            
            # Reel-out phase: power-limited (constant power)
            tetherForceOut = self.nominalTetherForce
            timeOut = self.reelingLength / vOut if vOut > 0 else float('inf')
            elecPowerOut = self.nominalGeneratorPower
            energyOut = elecPowerOut * timeOut
            
            # Reel-in phase: calculate energy for each segment
            energyIn = 0.0
            totalForceIn = 0.0
            for ws in windSpeedsIn:
                # Tether force for this segment
                force = calculate_tether_force_in(
                    airDensity, ws, self.wingArea,
                    gammaIn, self.elevationAngleIn, self.forceFactorIn
                )
                totalForceIn += force
                
                # Mechanical power in this segment
                mechPower = force * vIn
                
                # Time in this segment
                time_segment = segment_length / vIn if vIn > 0 else float('inf')
                
                # Energy consumed in this segment (mechanical)
                energyIn += mechPower * time_segment
            
            tetherForceIn = totalForceIn / n_segments
            timeIn = self.reelingLength / vIn if vIn > 0 else float('inf')
            
            # Electrical energy for reel-in
            elecEnergyIn = energyIn / self.generatorEfficiency
            
            # Average electrical power for reel-in
            elecPowerIn = elecEnergyIn / timeIn if timeIn > 0 else 0.0
            
            # Cycle power
            cycleTime = timeOut + timeIn
            netEnergy = energyOut - (elecEnergyIn / self.storageEfficiency)
            cyclePower = netEnergy / cycleTime if cycleTime > 0 else 0.0
            
            return {
                'cyclePower': max(0.0, cyclePower),
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
            }
        else:
            # Original non-segmented calculation
            tetherForceOut = self.nominalTetherForce
            tetherForceIn = calculate_tether_force_in(
                airDensity, windSpeed, self.wingArea,
                gammaIn, self.elevationAngleIn, self.forceFactorIn
            )

            return calculate_cycle_results(
                tetherForceOut, tetherForceIn, vOut, vIn,
                self.reelingLength, gammaOut, gammaIn,
                self.generatorEfficiency, self.storageEfficiency
            )

    def _optimize_gamma_in_region3(self, mu: float, gammaInMax: float) -> float:
        """Optimize gamma_in for Region 3 operation.

        Args:
            mu (float): Wind speed ratio to nominal power wind speed.
            gammaInMax (float): Maximum gamma_in.

        Returns:
            float: Optimal gamma_in.
        """
        def objective(x):
            gammaIn = x[0]

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
            objective, [0.001],
            bounds=[(0.001, gammaInMax)],
            method='SLSQP'
        )

        return result['x'][0]

    def generate_power_curves_with_shear(
        self,
        wind_shear_data: Dict[str, Any],
        numPoints: int = 100
    ) -> Dict[str, Any]:
        """Generate power curves for multiple wind shear profiles.
        
        Calculates power curves for each wind profile/cluster in the wind
        shear data. Wind speeds at reference height are converted to wind
        speeds at operational altitude using the profile.
        
        Args:
            wind_shear_data: Dictionary from load_wind_shear_profiles() with:
                - 'profiles': List of wind profile dicts
                - 'altitudes': Array of altitudes
                - 'reference_height_m': Reference height
            numPoints: Number of reference wind speed points.
            
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
        reference_height_m = wind_shear_data['reference_height_m']
        profiles = wind_shear_data['profiles']
        
        # Wind speeds at reference height
        windSpeedsAtRef = np.linspace(
            self.cutInWindSpeed,
            self.cutOutWindSpeed,
            numPoints
        )
        
        power_curves = []
        
        for profile_data in profiles:
            profile_id = profile_data['id']
            
            # Prepare wind profile for interpolation
            wind_profile = {
                'altitudes': wind_shear_data['altitudes'],
                'u_normalized': profile_data['u_normalized']
            }
            
            # Recompute nominal wind speeds with this wind profile
            self._compute_nominal_wind_speeds_with_shear(wind_profile, reference_height_m)
            
            # Calculate wind speeds at operational altitude (for reporting)
            windSpeedsAtOp = np.array([
                self.get_wind_speed_at_operational_altitude(
                    ws_ref, wind_profile, reference_height_m
                )
                for ws_ref in windSpeedsAtRef
            ])
            
            # Calculate power using reference wind speed and wind profile
            results = [
                self.calculate_power(
                    ws_ref, 
                    wind_profile=wind_profile, 
                    reference_height_m=reference_height_m
                )
                for ws_ref in windSpeedsAtRef
            ]
            
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
            }
            
            power_curves.append(profile_curve)
        
        return {
            'reference_height_m': reference_height_m,
            'operational_altitude_m': self.operationalAltitude,
            'altitudes': wind_shear_data['altitudes'],
            'profiles': power_curves
        }

    @classmethod
    def from_yaml(
        cls,
        yamlPath: Path,
        simulationSettingsPath: Path = None,
        validate: bool = True
    ) -> 'PowerModel':
        """Load configuration from YAML file and create model instance.

        Supports both legacy format and awesIO format configuration files.
        If the file is in awesIO format, it will be validated before use.

        Args:
            yamlPath: Path to system YAML configuration file.
            simulationSettingsPath: Path to simulation settings YAML file
                containing operational and atmosphere parameters.
                Required for awesIO format system configs.
            validate: Whether to validate awesIO format files. Defaults to True.

        Returns:
            PowerModel: Initialized power model instance.

        Raises:
            FileNotFoundError: If YAML file doesn't exist.
            ValueError: If YAML contains invalid configuration.
            Exception: If awesIO validation fails.
        """
        yamlPath = Path(yamlPath)

        if not yamlPath.exists():
            raise FileNotFoundError(f"Configuration file not found: {yamlPath}")

        with open(yamlPath, 'r') as f:
            config = yaml.safe_load(f)

        # Load simulation settings if provided
        simulation_settings = None
        if simulationSettingsPath is not None:
            simulationSettingsPath = Path(simulationSettingsPath)
            if not simulationSettingsPath.exists():
                raise FileNotFoundError(
                    f"Simulation settings file not found: {simulationSettingsPath}"
                )
            with open(simulationSettingsPath, 'r') as f:
                simulation_settings = yaml.safe_load(f)
            print(f"Loaded simulation settings from: {simulationSettingsPath.name}")

        # Validate using awesIO validator (if schema exists and awesIO is available)
        if validate and AWESIO_AVAILABLE:
            try:
                awesio_validate(
                    input=yamlPath,
                    restrictive=False,
                    defaults=False,
                )
                print(f"  ✓ {yamlPath.name} validated against system_schema")
            except FileNotFoundError:
                print(f"  Note: system_schema not available, skipping validation")
        elif validate and not AWESIO_AVAILABLE:
            print(f"  Note: awesIO not available, skipping validation")

        # Create model with awesIO config and simulation settings
        return cls(config, simulation_settings)

    def export_power_curves_awesio(
        self,
        data: Dict[str, Any],
        output_path: Path,
        name: str = "Luchsinger Power Curves with Wind Shear",
        description: str = "Power curves for pumping ground-gen AWE system with wind shear",
        note: str = "Power curve data generated from Luchsinger model with wind shear profiles",
        validate: bool = True,
    ) -> None:
        """Export power curve data with wind shear profiles in awesIO format.

        Args:
            data: Power curve data from generate_power_curves_with_shear().
            output_path: Path to save the output YAML file.
            name: Name for the power curves dataset.
            description: Description of the power curves.
            note: Additional notes about the data.
            validate: Whether to validate the output file. Defaults to True.
        """
        output_path = Path(output_path)

        reference_height_m = data['reference_height_m']
        operational_altitude_m = data['operational_altitude_m']
        altitudes = data.get('altitudes', [])
        profiles = data['profiles']

        # Get reference wind speeds from first profile (same for all)
        reference_wind_speeds = profiles[0]['windSpeedAtRef']

        # Build power curves list for each profile
        power_curves_list = []
        for profile in profiles:
            profile_id = profile['profile_id']
            
            power_curve = {
                'profile_id': int(profile_id),
                'u_normalized': [float(u) for u in profile['u_normalized']],
                'v_normalized': [float(v) for v in profile['v_normalized']],
                'probability_weight': 1.0 / len(profiles),  # Equal weight for all profiles
                'cycle_power_w': [float(p) for p in profile['power']],
                'reel_out_power_w': [float(p) for p in profile['reelOutPower']],
                'reel_in_power_w': [float(p) for p in profile['reelInPower']],
                'reel_out_time_s': [float(t) for t in profile['reelOutTime']],
                'reel_in_time_s': [float(t) for t in profile['reelInTime']],
                'cycle_time_s': [
                    float(t_out + t_in)
                    for t_out, t_in in zip(profile['reelOutTime'], profile['reelInTime'])
                ],
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
        if validate and AWESIO_AVAILABLE:
            awesio_validate(
                input=output_path,
                restrictive=False,
                defaults=False,
            )
            print(f"  ✓ Output validated against power_curves_schema")
        elif validate and not AWESIO_AVAILABLE:
            print(f"  Note: awesIO not available, skipping output validation")
