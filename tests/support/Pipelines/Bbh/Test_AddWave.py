# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import shutil
import unittest
from pathlib import Path

import yaml

from spectre.Informer import unit_test_build_path
from spectre.Pipelines.Bbh.AddWave import add_wave_parameters, add_waves_command
from spectre.Pipelines.Bbh.InitialData import generate_id
from spectre.Pipelines.Bbh.Inspiral import start_inspiral_command
from spectre.support.Logging import configure_logging


class TestAddWave(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(
            unit_test_build_path(), "support/Pipelines/Bbh/AddWave"
        )
        shutil.rmtree(self.test_dir, ignore_errors=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.bin_dir = Path(unit_test_build_path(), "../../bin").resolve()
        generate_id(
            {
                "MassRatio": 1.5,
                "MassA": 0.6,
                "MassB": 0.4,
                "DimensionlessSpinA": [0.0, 0.0, 0.0],
                "DimensionlessSpinB": [0.0, 0.0, 0.0],
            },
            separation=20.0,
            orbital_angular_velocity=0.01,
            radial_expansion_velocity=-1.0e-5,
            refinement_level=1,
            polynomial_order=5,
            run_dir=self.test_dir / "ID",
            scheduler=None,
            submit=False,
            executable=str(self.bin_dir / "SolveXcts"),
        )
        self.id_run_dir = self.test_dir / "ID"
        with open(self.id_run_dir / "InitialData.yaml") as open_input_file:
            self.id_metadata, self.id_input_file = yaml.safe_load_all(
                open_input_file
            )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_add_wave_parameters(self):
        # The initial data was generated without a target eccentricity
        with self.assertLogs(level=logging.WARNING):
            params = add_wave_parameters(
                self.id_input_file, self.id_metadata, self.id_run_dir
            )
        self.assertEqual(
            params["IdFileGlob"],
            str(self.id_run_dir.resolve() / "BbhVolume*.h5"),
        )
        self.assertEqual(params["IdSubfile"], "VolumeData")
        # The excisions of the initial data, not enlarged as in the evolution
        self.assertAlmostEqual(params["ExcisionRadiusA"], 1.116 * 0.82)
        self.assertAlmostEqual(params["ExcisionRadiusB"], 0.744 * 0.82)
        self.assertEqual(params["HorizonLMax"], 20)
        self.assertEqual(params["OrbitalAngularVelocity"], 0.01)
        self.assertEqual(params["RadialExpansionVelocity"], -1.0e-5)
        self.assertEqual(params["Separation"], 20.0)
        self.assertEqual(params["TargetEccentricity"], 0.0)
        self.assertEqual(params["MassLeft"], 0.4)
        self.assertEqual(params["MassRight"], 0.6)
        self.assertAlmostEqual(params["AttenuationWidth"], 6.0)
        # The outer shell of the evolution reaches 600 / 15 * 20 = 800
        self.assertAlmostEqual(params["OuterShellRadius"], 800.0)
        self.assertAlmostEqual(
            params["PastEvolutionDuration"], 1.5 * (800.0 + 20.0)
        )
        self.assertEqual(params["PastEvolutionTimeStep"], 0.5)

        self.id_metadata["TargetParams"]["Eccentricity"] = 0.1
        with self.assertRaisesRegex(ValueError, "eccentricity of zero"):
            add_wave_parameters(
                self.id_input_file, self.id_metadata, self.id_run_dir
            )

    def test_cli(self):
        # Not using `CliRunner.invoke()` because it runs in an isolated
        # environment and doesn't work with MPI in the container.
        try:
            add_waves_command(
                [
                    str(self.id_run_dir / "InitialData.yaml"),
                    "--refinement-level",
                    "1",
                    "--polynomial-order",
                    "5",
                    "-E",
                    str(self.bin_dir / "SolveXcts"),
                    "--no-schedule",
                    # Parsing the options loads the initial data, which this
                    # test doesn't have
                    "--no-validate",
                    "-o",
                    str(self.test_dir / "AddWave"),
                    "--no-submit",
                ]
            )
        except SystemExit as e:
            self.assertEqual(e.code, 0)
        add_wave_input_file_path = self.test_dir / "AddWave/AddWave.yaml"
        with open(add_wave_input_file_path) as open_input_file:
            metadata, add_wave_input_file = yaml.safe_load_all(open_input_file)
        self.assertEqual(
            metadata["TargetParams"], self.id_metadata["TargetParams"]
        )
        self.assertEqual(
            metadata["Next"],
            {
                "Run": "spectre.Pipelines.Bbh.PostprocessId:postprocess_id",
                "With": {
                    "id_input_file_path": "__file__",
                    "id_run_dir": "./",
                    "horizon_l_max": 20,
                    "control": False,
                    "evolve": False,
                },
            },
        )
        background = add_wave_input_file["Background"]["NumericBinaryWithWaves"]
        self.assertEqual(
            background["DataFile"],
            str(self.id_run_dir.resolve() / "BbhVolume*.h5"),
        )
        self.assertEqual(background["Separation"], 20.0)
        add_wave_domain = add_wave_input_file["DomainCreator"][
            "BinaryCompactObject"
        ]
        for object_label in ["ObjectA", "ObjectB"]:
            excision_bc = add_wave_domain[object_label]["Interior"][
                "ExciseWithBoundaryCondition"
            ]["NumericData"]
            self.assertEqual(excision_bc["DataFile"], background["DataFile"])

        # The evolution starts from the data with added waves
        try:
            start_inspiral_command(
                [
                    str(add_wave_input_file_path),
                    "--id-subfile-name",
                    "VolumeData",
                    "-E",
                    str(self.bin_dir / "EvolveGhBinaryBlackHole"),
                    "--no-schedule",
                    "--num-nodes",
                    "1",
                    "--refinement-level",
                    "1",
                    "--polynomial-order",
                    "5",
                    "-o",
                    str(self.test_dir / "Inspiral"),
                    "--no-submit",
                ]
            )
        except SystemExit as e:
            self.assertEqual(e.code, 0)
        with open(self.test_dir / "Inspiral/Inspiral.yaml") as open_input_file:
            _, inspiral_input_file = yaml.safe_load_all(open_input_file)
        numeric_id = inspiral_input_file["InitialData"]["NumericInitialData"]
        self.assertEqual(
            numeric_id["VolumeData"]["FileGlob"],
            str((self.test_dir / "AddWave").resolve() / "AddWaveVolume*.h5"),
        )
        # The background shift of the data with added waves vanishes, so
        # 'ShiftExcess' is the whole shift
        self.assertEqual(numeric_id["Variables"]["Shift"], "ShiftExcess")
        inspiral_domain = inspiral_input_file["DomainCreator"][
            "BinaryCompactObject"
        ]
        time_dependent_maps = inspiral_domain["TimeDependentMaps"]
        self.assertEqual(
            time_dependent_maps["RotationMap"]["InitialAngularVelocity"],
            [0.0, 0.0, 0.01],
        )
        # PyYAML reads '-1e-05' (no decimal point) as a string, SpECTRE as a
        # number
        self.assertEqual(
            [
                float(value)
                for value in time_dependent_maps["ExpansionMap"][
                    "InitialValues"
                ]
            ],
            [1.0, -1.0e-5, 0.0],
        )
        # Same domain as the solve that added the waves, apart from the
        # enlarged excisions and the boundary conditions
        for object_label in ["ObjectA", "ObjectB"]:
            self.assertAlmostEqual(
                inspiral_domain[object_label]["InnerRadius"],
                1.0385 * add_wave_domain[object_label]["InnerRadius"],
            )
            for key in ["OuterRadius", "XCoord", "UseLogarithmicMap"]:
                self.assertEqual(
                    inspiral_domain[object_label][key],
                    add_wave_domain[object_label][key],
                )
        for key in [
            "CenterOfMassOffset",
            "UseEquiangularMap",
            "CubeScale",
            "InitialRefinement",
            "InitialGridPoints",
        ]:
            self.assertEqual(inspiral_domain[key], add_wave_domain[key])
        for block in ["Envelope", "OuterShell"]:
            for key in ["Radius", "RadialDistribution"]:
                self.assertEqual(
                    inspiral_domain[block][key], add_wave_domain[block][key]
                )


if __name__ == "__main__":
    configure_logging(log_level=logging.DEBUG)
    unittest.main(verbosity=2)
