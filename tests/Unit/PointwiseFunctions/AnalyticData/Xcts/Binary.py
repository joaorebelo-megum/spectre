# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np

x_coords = [-5.0, 6.0]
y_offset = 0.1
z_offset = 0.2
masses = [1.1, 0.43]
angular_velocity = 0.02
radial_velocity = 0.01
linear_velocity = [0.1, 0.2, 0.3]
falloff_widths = [7.0, 8.0]


def conformal_metric_bbh_isotropic(x):
    return np.identity(3)


def inv_conformal_metric_bbh_isotropic(x):
    return np.identity(3)


def deriv_conformal_metric_bbh_isotropic(x):
    return np.zeros((3, 3, 3))


def extrinsic_curvature_trace_bbh_isotropic(x):
    return 0.0


def shift_background(x):
    return (
        np.array([-angular_velocity * x[1], angular_velocity * x[0], 0.0])
        + radial_velocity * x
        + np.array(linear_velocity)
    )


def longitudinal_shift_background_bbh_isotropic(x):
    return np.zeros((3, 3))


def conformal_factor_minus_one_bbh_isotropic(x):
    r1 = np.sqrt(
        (x[0] - x_coords[0]) ** 2
        + (x[1] - y_offset) ** 2
        + (x[2] - z_offset) ** 2
    )
    r2 = np.sqrt(
        (x[0] - x_coords[1]) ** 2
        + (x[1] - y_offset) ** 2
        + (x[2] - z_offset) ** 2
    )
    return 0.5 * (
        np.exp(-(r1**2) / falloff_widths[0] ** 2) * masses[0] / r1
        + np.exp(-(r2**2) / falloff_widths[1] ** 2) * masses[1] / r2
    )


def energy_density_bbh_isotropic(x):
    return 0.0


def stress_trace_bbh_isotropic(x):
    return 0.0


def momentum_density_bbh_isotropic(x):
    return np.zeros(3)
