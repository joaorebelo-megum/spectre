# Distributed under the MIT License.
# See LICENSE.txt for details.

import os
import sys
import unittest

import numpy as np
import numpy.testing as npt

import spectre.PointwiseFunctions.Hydro as hydro
from spectre import Informer
from spectre.DataStructures import DataVector
from spectre.DataStructures.Tensor import Scalar, tnsr

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from TestFunctions import *


class TestCoMovingMF(unittest.TestCase):
    def test_comoving_magnetic_field_one_form(self):
        spatial_velocity_one_form = tnsr.i[DataVector, 3](
            num_points=5, fill=1.2
        )
        magnetic_field_one_form = tnsr.i[DataVector, 3](num_points=5, fill=1.1)
        magnetic_field_dot_spatial_velocity = Scalar[DataVector](
            num_points=5, fill=1.7
        )
        lorentz_factor = Scalar[DataVector](num_points=5, fill=2.1)
        shift = tnsr.I[DataVector, 3](num_points=5, fill=1.5)
        lapse = Scalar[DataVector](num_points=5, fill=1.1)

        bindings = hydro.comoving_magnetic_field_one_form(
            spatial_velocity_one_form,
            magnetic_field_one_form,
            magnetic_field_dot_spatial_velocity,
            lorentz_factor,
            shift,
            lapse,
        )
        alternative = comoving_magnetic_field_one_form(
            np.array(spatial_velocity_one_form),
            np.array(magnetic_field_one_form),
            np.array(magnetic_field_dot_spatial_velocity)[0],
            np.array(lorentz_factor)[0],
            np.array(shift),
            np.array(lapse)[0],
        )

        assert type(bindings) == tnsr.a[DataVector, 3]
        np.testing.assert_allclose(bindings, alternative)

    def test_comoving_magnetic_field_squared(self):
        magnetic_field_squared = Scalar[DataVector](num_points=5, fill=1.5)
        magnetic_field_dot_spatial_velocity = Scalar[DataVector](
            num_points=5, fill=1.7
        )
        lorentz_factor = Scalar[DataVector](num_points=5, fill=2.1)

        bindings = hydro.comoving_magnetic_field_squared(
            magnetic_field_squared,
            magnetic_field_dot_spatial_velocity,
            lorentz_factor,
        )

        alternative = comoving_magnetic_field_squared(
            np.array(magnetic_field_squared),
            np.array(magnetic_field_dot_spatial_velocity),
            np.array(lorentz_factor)[0],
        )

        assert type(bindings) == Scalar[DataVector]
        np.testing.assert_allclose(bindings, alternative)


if __name__ == "__main__":
    unittest.main(verbosity=2)
