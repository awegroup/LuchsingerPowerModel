# -*- coding: utf-8 -*-
"""Configuration loader for AWE power model.

Loads system configuration, wind resource data, and simulation settings
from YAML files and returns them as dictionaries.
"""

import numpy as np
import yaml
from pathlib import Path


def _get_first_available(mapping, *keys):
    """Return the first non-None value found for the given keys.

    Args:
        mapping (dict): Mapping to search.
        *keys: Candidate keys in lookup order.

    Returns:
        Any: First non-None value, or None if no key is present.
    """
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _first_item(value):
    """Return the first item from a list-like value, or the value itself."""
    if isinstance(value, list):
        return value[0] if value else {}
    return value or {}


def _get_component(components, singular_key, plural_key=None):
    """Get a component from either legacy singular or awesIO list format."""
    if plural_key:
        component = _first_item(components.get(plural_key))
        if component:
            return component
    return components.get(singular_key, {}) or {}


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

    # Wing parameters. The current awesIO system schema nests the wing,
    # bridle, and KCU under components.kites[0]; older local configs kept
    # those components flat under components.wing/control_system.
    kite = _get_component(components, 'kite', 'kites')
    wing = kite.get('wing') or components.get('wing', {})
    wing_structure = wing.get('structure', {})

    # Tether parameters
    tether = _get_component(components, 'tether', 'tethers')
    tether_structure = tether.get('structure', {})

    # Ground station parameters
    ground_station = _get_component(
        components,
        'ground_station',
        'ground_stations',
    )
    drum = _first_item(ground_station.get('drums')) or ground_station.get(
        'drum',
        {},
    )
    generator = _first_item(
        ground_station.get('generators')
    ) or ground_station.get('generator', {})
    storage = _first_item(ground_station.get('storages')) or ground_station.get(
        'storage',
        {},
    )

    nominal_generator_power = _get_first_available(
        generator,
        'max_power',
        'rated_power',
    )
    if nominal_generator_power is None:
        legacy_power_kw = _get_first_available(
            generator,
            'rated_power_kw',
            'max_power_kw',
        )
        nominal_generator_power = (
            legacy_power_kw * 1000 if legacy_power_kw is not None else None
        )

    return {
        'wingArea': _get_first_available(
            wing_structure,
            'projected_surface_area',
            'projected_surface_area_m2',
        ),
        'tetherMaxLength': _get_first_available(
            tether_structure,
            'length',
            'length_m',
        ),
        'tetherDiameter': _get_first_available(
            tether_structure,
            'diameter',
            'diameter_m',
        ),
        'reelOutSpeedLimit': _get_first_available(
            drum,
            'max_tether_speed',
            'max_tether_speed_m_s',
        ),
        'reelInSpeedLimit': -_get_first_available(
            drum,
            'max_tether_speed',
            'max_tether_speed_m_s',
        ),
        'nominalTetherForce': (
            _get_first_available(drum, 'max_tether_force', 'max_tether_force_n')
            or _get_first_available(
                tether_structure,
                'max_tether_force',
                'max_tether_force_n',
            )
        ),
        'nominalGeneratorPower': nominal_generator_power,
        'generatorEfficiency': generator.get('efficiency'),
        'storageEfficiency': storage.get('efficiency'),
    }


def load_wind_resource(file_path, validate_file=False):
    """Load wind resource data from YAML file.

    Extracts normalized wind profiles for each profile, which represent
    different atmospheric conditions and shear characteristics.

    Args:
        file_path (str or Path): Path to the wind resource YAML file.
        validate_file (bool): If True, validate the YAML against the awesIO
            schema. Defaults to False.

    Returns:
        dict: Wind resource dictionary containing:
            - 'altitudes': Array of altitudes in meters.
            - 'reference_height': Reference altitude where profiles equal 1.0.
            - 'profiles': List of dicts with 'id', 'u_normalized',
              'v_normalized'.
            - 'n_profiles': Number of profiles/profiles.

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
    n_profiles = metadata.get('n_profiles')
    reference_height = metadata.get('reference_height')

    if n_profiles is None:
        raise ValueError("'n_profiles' not found in wind resource metadata")
    if reference_height is None:
        raise ValueError(
            "'reference_height' not found in wind resource metadata"
        )

    # Extract altitudes
    altitudes = np.array(data.get('altitudes', []))
    if len(altitudes) == 0:
        raise ValueError("'altitudes' array is empty or missing")

    # Extract profiles/profiles
    raw_profiles = data.get('profiles', [])
    if len(raw_profiles) != n_profiles:
        raise ValueError(
            f"Expected {n_profiles} profiles, found {len(raw_profiles)}"
        )

    profiles = []
    for profile in raw_profiles:
        profile = {
            'id': profile.get('id'),
            'u_normalized': np.array(profile.get('u_normalized', [])),
            'v_normalized': np.array(profile.get('v_normalized', []))
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
        'reference_height': reference_height,
        'profiles': profiles,
        'n_profiles': n_profiles,
        'metadata': metadata,
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
        'tetherMinLength': settings.get('minimum_tether_length_m'),
        'airDensity': settings.get('air_density_kg_m3'),
        'liftCoefficientKiteOut': settings.get('lift_coefficient_reel_out'),
        'dragCoefficientKiteOut': settings.get('drag_coefficient_reel_out'),
        'tetherDragCoefficient': settings.get('tether_drag_coefficient'),
        'liftCoefficientKiteIn': settings.get('lift_coefficient_reel_in'),
        'dragCoefficientKiteIn': settings.get('drag_coefficient_reel_in'),
    }
