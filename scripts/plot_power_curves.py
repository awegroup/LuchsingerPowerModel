"""Compare and plot two awesIO power curve YAML files.

This script loads two power curve result files (awesIO schema), converts each
file into one plotting profile, and overlays them using the same comprehensive
multi-panel style defined in ``src.power_luchsinger.plotting``.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import yaml

# Add workspace root to path
workspace_root = Path(__file__).parent.parent
if str(workspace_root) not in sys.path:
	sys.path.insert(0, str(workspace_root))

from src.power_luchsinger.plotting import plot_comprehensive_analysis


# User settings: edit these paths/labels and run the script
yamlPathA = workspace_root / 'results' / 'luchsinger_power_curves_OG.yml'
yamlPathB = workspace_root / 'results' / 'luchsinger_power_curves_extended.yml'
profileLabelA = 'OG'
profileLabelB = 'Extended'
savePath = workspace_root / 'results' / 'power_curve_analysis_comparison.pdf'
showPlot = True


def load_yaml(file_path: Path) -> Dict[str, Any]:
	"""Load and parse a YAML file.

	Args:
		file_path (Path): Path to YAML file.

	Returns:
		Dict[str, Any]: Parsed YAML content.

	Raises:
		ValueError: If the file is empty or invalid.
	"""
	with file_path.open('r', encoding='utf-8') as file_handle:
		data = yaml.safe_load(file_handle)

	if not isinstance(data, dict):
		raise ValueError(f"Invalid or empty YAML content: {file_path}")

	return data


def _extract_series(wind_speed_data: List[Dict[str, Any]], key_path: List[str]) -> np.ndarray:
	"""Extract a numeric series from nested wind speed entries.

	Args:
		wind_speed_data (List[Dict[str, Any]]): Per-wind-speed result entries.
		key_path (List[str]): Nested path to value.

	Returns:
		np.ndarray: Numeric series.
	"""
	values = []
	for entry in wind_speed_data:
		value = entry
		for key in key_path:
			value = value[key]
		values.append(float(value))
	return np.asarray(values, dtype=float)


def awesio_to_profile(data: Dict[str, Any], profile_label: str) -> Dict[str, Any]:
	"""Convert awesIO power curve YAML data to internal plotting profile format.

	Args:
		data (Dict[str, Any]): Parsed awesIO power curve YAML data.
		profile_label (str): Label for legend/profile identifier.

	Returns:
		Dict[str, Any]: Single profile in plotting.py compatible format.

	Raises:
		ValueError: If required awesIO fields are missing.
	"""
	power_curves = data.get('power_curves', [])
	if len(power_curves) == 0:
		raise ValueError('Missing power_curves in awesIO file')

	curve = power_curves[0]
	wind_speed_data = curve.get('wind_speed_data', [])
	if len(wind_speed_data) == 0:
		raise ValueError('Missing wind_speed_data in awesIO file')

	reference_wind_speeds = data.get('reference_wind_speeds')
	if reference_wind_speeds is None:
		reference_wind_speeds = data.get('reference_wind_speeds_m_s')
	if reference_wind_speeds is None:
		reference_wind_speeds = [
			entry.get('wind_speed', entry.get('wind_speed_m_s'))
			for entry in wind_speed_data
		]

	return {
		'profile_id': profile_label,
		'u_normalized': np.asarray(curve['wind_profile']['u_normalized'], dtype=float),
		'v_normalized': np.asarray(curve['wind_profile']['v_normalized'], dtype=float),
		'windSpeedAtRef': np.asarray(reference_wind_speeds, dtype=float),
		'power': _extract_series(wind_speed_data, ['performance', 'power', 'average_cycle_power']),
		'reelOutPower': _extract_series(wind_speed_data, ['performance', 'power', 'average_reel_out_power']),
		'reelInPower': _extract_series(wind_speed_data, ['performance', 'power', 'average_reel_in_power']),
		'reelOutTime': _extract_series(wind_speed_data, ['performance', 'timing', 'reel_out_time']),
		'reelInTime': _extract_series(wind_speed_data, ['performance', 'timing', 'reel_in_time']),
		'tetherForceOut': _extract_series(wind_speed_data, ['performance', 'forces', 'tether_force_out']),
		'tetherForceIn': _extract_series(wind_speed_data, ['performance', 'forces', 'tether_force_in']),
		'reelOutSpeed': _extract_series(wind_speed_data, ['performance', 'speeds', 'reel_out_speed']),
		'reelInSpeed': _extract_series(wind_speed_data, ['performance', 'speeds', 'reel_in_speed']),
		'gammaOut': _extract_series(wind_speed_data, ['performance', 'reel_factors', 'gamma_out']),
		'gammaIn': _extract_series(wind_speed_data, ['performance', 'reel_factors', 'gamma_in']),
		'elevationAngleOut': _extract_series(wind_speed_data, ['performance', 'elevation_angles', 'elevation_angle_out_rad']),
		'elevationAngleIn': _extract_series(wind_speed_data, ['performance', 'elevation_angles', 'elevation_angle_in_rad']),
	}


def build_comparison_dataset(data_a: Dict[str, Any], data_b: Dict[str, Any], label_a: str, label_b: str) -> Dict[str, Any]:
	"""Build combined plotting dataset with two profiles.

	Args:
		data_a (Dict[str, Any]): First awesIO dataset.
		data_b (Dict[str, Any]): Second awesIO dataset.
		label_a (str): Label for first profile.
		label_b (str): Label for second profile.

	Returns:
		Dict[str, Any]: Combined dataset for ``plot_comprehensive_analysis``.
	"""
	profile_a = awesio_to_profile(data_a, label_a)
	profile_b = awesio_to_profile(data_b, label_b)

	metadata_a = data_a.get('metadata', {})
	model_config_a = metadata_a.get('model_config', {})
	wind_resource_a = metadata_a.get('wind_resource', {})

	return {
		'reference_height_m': float(wind_resource_a.get('reference_height', metadata_a.get('reference_height_m', 0.0))),
		'operational_altitude_m': float(model_config_a.get('operating_altitude', model_config_a.get('operating_altitude_m', 0.0))),
		'altitudes': np.asarray(data_a.get('altitudes', data_a.get('altitudes_m', [])), dtype=float),
		'profiles': [profile_a, profile_b],
	}


def main() -> None:
	"""Run YAML comparison plotting script."""
	data_a = load_yaml(yamlPathA)
	data_b = load_yaml(yamlPathB)

	comparison_data = build_comparison_dataset(
		data_a,
		data_b,
		profileLabelA,
		profileLabelB,
	)

	plot_comprehensive_analysis(
		comparison_data,
		model_params=None,
		save_path=str(savePath),
		show=showPlot,
	)

	print('Comparison plot saved to:', savePath)


if __name__ == '__main__':
	main()
