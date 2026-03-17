# -*- coding: utf-8 -*-
"""Configuration loader for AWE power model.

Loads system configuration, wind resource data, and simulation settings
from YAML files and returns them as dictionaries.
"""

import numpy as np
import yaml
from pathlib import Path


def load_yaml(file_path):
    """Load a YAML file.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Parsed YAML content.
    """
    with open(file_path, 'r') as f:
        return yaml.full_load(f)


def load_system_config(file_path, validate_file=False):
    """Load and extract model parameters from system configuration YAML file.

    Loads the awesIO system configuration, optionally validates it, and
    extracts all model parameters together with the operational and
    atmosphere parameters from simulation_settings into a flat dictionary.

    Args:
        file_path (str or Path): Path to the system configuration YAML file.
        validate_file (bool): If True, validate the YAML against the awesIO
            schema. Defaults to False.

    Returns:
        dict: Flat parameter dictionary with keys:
            wingArea, liftCoefficientKiteOut, dragCoefficientKiteOut,
            liftCoefficientKiteIn, dragCoefficientKiteIn, tetherMaxLength,
            reelOutSpeedLimit, reelInSpeedLimit, nominalTetherForce,
            nominalGeneratorPower, generatorEfficiency, storageEfficiency.

    Raises:
        FileNotFoundError: If the system configuration file is not found.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"System config file not found: {file_path}"
        )

    if validate_file:
        try:
            from awesio.validator import validate as awesio_validate
            awesio_validate(input=file_path)
            print(f"  ✓ {file_path.name} validated against system_schema")
        except ImportError:
            print("  awesIO not installed; skipping validation.")
        except Exception as e:
            print(f"  Validation failed: {e}")

    config = load_yaml(file_path)
    components = config.get('components', {})

    # Wing parameters
    wing = components.get('wing', {})
    wing_structure = wing.get('structure', {})

    # Tether parameters
    tether = components.get('tether', {})
    tether_structure = tether.get('structure', {})

    # Ground station parameters
    ground_station = components.get('ground_station', {})
    drum = ground_station.get('drum', {})
    generator = ground_station.get('generator', {})
    storage = ground_station.get('storage', {})

    return {
        'wingArea': wing_structure.get('projected_surface_area_m2'),
        'tetherMaxLength': tether_structure.get('length_m'),
        'reelOutSpeedLimit': drum.get('max_tether_speed_m_s'),
        'reelInSpeedLimit': -drum.get('max_tether_speed_m_s'),
        'nominalTetherForce': (
            drum.get('max_tether_force_n')
            or tether_structure.get('max_tether_force_n')
        ),
        'nominalGeneratorPower': (
            generator.get('rated_power_kw') or generator.get('max_power_kw', 0)
        ) * 1000,
        'generatorEfficiency': generator.get('efficiency'),
        'storageEfficiency': storage.get('efficiency'),
    }


def load_wind_resource(file_path, validate_file=False):
    """Load wind resource data from YAML file.

    Extracts normalized wind profiles for each cluster, which represent
    different atmospheric conditions and shear characteristics.

    Args:
        file_path (str or Path): Path to the wind resource YAML file.
        validate_file (bool): If True, validate the YAML against the awesIO
            schema. Defaults to False.

    Returns:
        dict: Wind resource dictionary containing:
            - 'altitudes': Array of altitudes in meters.
            - 'reference_height_m': Reference altitude where profiles equal 1.0.
            - 'profiles': List of dicts with 'id', 'u_normalized',
              'v_normalized'.
            - 'n_clusters': Number of profiles/clusters.

    Raises:
        FileNotFoundError: If wind resource file is not found.
        ValueError: If required keys are missing from the file.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Wind resource file not found: {file_path}"
        )

    if validate_file:
        try:
            from awesio.validator import validate as awesio_validate
            awesio_validate(input=file_path)
            print(f"  ✓ {file_path.name} validated against wind_resource_schema")
        except ImportError:
            print("  awesIO not installed; skipping validation.")
        except Exception as e:
            print(f"  Validation failed: {e}")

    data = load_yaml(file_path)

    # Extract metadata
    metadata = data.get('metadata', {})
    n_clusters = metadata.get('n_clusters')
    reference_height_m = metadata.get('reference_height_m')

    if n_clusters is None:
        raise ValueError("'n_clusters' not found in wind resource metadata")
    if reference_height_m is None:
        raise ValueError(
            "'reference_height_m' not found in wind resource metadata"
        )

    # Extract altitudes
    altitudes = np.array(data.get('altitudes', []))
    if len(altitudes) == 0:
        raise ValueError("'altitudes' array is empty or missing")

    # Extract clusters/profiles
    clusters = data.get('clusters', [])
    if len(clusters) != n_clusters:
        raise ValueError(
            f"Expected {n_clusters} clusters, found {len(clusters)}"
        )

    profiles = []
    for cluster in clusters:
        profile = {
            'id': cluster.get('id'),
            'u_normalized': np.array(cluster.get('u_normalized', [])),
            'v_normalized': np.array(cluster.get('v_normalized', []))
        }

        if len(profile['u_normalized']) != len(altitudes):
            raise ValueError(
                f"Profile {profile['id']}: u_normalized length mismatch"
            )
        if len(profile['v_normalized']) != len(altitudes):
            raise ValueError(
                f"Profile {profile['id']}: v_normalized length mismatch"
            )

        profiles.append(profile)

    return {
        'altitudes': altitudes,
        'reference_height_m': reference_height_m,
        'profiles': profiles,
        'n_clusters': n_clusters
    }


def load_simulation_settings(file_path):
    """Load simulation settings from YAML file.

    Args:
        file_path (str or Path): Path to the simulation settings YAML file.

    Returns:
        dict: Simulation settings dictionary with 'operational' and
            'atmosphere' keys.

    Raises:
        FileNotFoundError: If simulation settings file is not found.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Simulation settings file not found: {file_path}"
        )

    settings = load_yaml(file_path)
    settings = settings.get('settings', {})

    return {
        'model': settings.get('model'),
        'cutInWindSpeed': settings.get('cut_in_wind_speed_m_s'),
        'cutOutWindSpeed': settings.get('cut_out_wind_speed_m_s'),
        'elevationAngleOut': (
            np.radians(settings.get('elevation_angle_out_deg'))),
        'elevationAngleIn': (
            np.radians(settings.get('elevation_angle_in_deg'))),
        'numPoints': settings.get('num_points'),
        'nSegments': settings.get('n_segments'),
        'tetherMinLength': settings.get('minimum_tether_length_m'),
        'airDensity': settings.get('air_density_kg_m3'),
        'liftCoefficientKiteOut': settings.get('lift_coefficient_reel_out'),
        'dragCoefficientKiteOut': settings.get('drag_coefficient_reel_out'),
        'liftCoefficientKiteIn': settings.get('lift_coefficient_reel_in'),
        'dragCoefficientKiteIn': settings.get('drag_coefficient_reel_in'),
    }