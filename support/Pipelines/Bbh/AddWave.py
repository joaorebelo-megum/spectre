# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
from pathlib import Path
from typing import Optional, Union

import click
import yaml
from rich.pretty import pretty_repr

from spectre.Pipelines.Bbh.Inspiral import INSPIRAL_LEVS, inspiral_parameters
from spectre.support.Schedule import schedule, scheduler_options
from spectre.Visualization.ReadInputFile import find_event

logger = logging.getLogger(__name__)

ADD_WAVE_INPUT_FILE_TEMPLATE = Path(__file__).parent / "AddWave.yaml"

# Inspiral parameters that fix the evolution domain. The solve that adds waves
# runs on this domain, so the waves are resolved where the evolution needs them
# and the evolution can start from this grid.
_EVOLUTION_DOMAIN_KEYS = [
    "IdFileGlob",
    "TargetParams",
    "XCoordA",
    "XCoordB",
    "CenterOfMassOffset_y",
    "CenterOfMassOffset_z",
    "ObjectOuterRadius",
    "EnvelopeRadius",
    "OuterShellRadius",
    "ExtraRadRef",
    "ExtraRadPoints",
    "ExcisionAShapeMass",
    "ExcisionAShapeSpin_x",
    "ExcisionAShapeSpin_y",
    "ExcisionAShapeSpin_z",
    "ExcisionBShapeMass",
    "ExcisionBShapeSpin_x",
    "ExcisionBShapeSpin_y",
    "ExcisionBShapeSpin_z",
]


def add_wave_parameters(
    id_input_file: dict,
    id_metadata: dict,
    id_run_dir: Union[str, Path],
) -> dict:
    """Determine the parameters of the solve that adds waves to initial data.

    These parameters fill the 'ADD_WAVE_INPUT_FILE_TEMPLATE'.

    Arguments:
      id_input_file: Initial data input file as a dictionary.
      id_metadata: Metadata of the initial data input file as a dictionary.
      id_run_dir: Directory of the initial data run. Paths in the input file
        are relative to this directory.
    """
    target_params = id_metadata["TargetParams"]
    id_binary = id_input_file["Background"]["Binary"]
    id_domain_creator = id_input_file["DomainCreator"]["BinaryCompactObject"]
    id_subfile_name = find_event(
        "ObserveFields", "EventsAndTriggersAtIterations", id_input_file
    )["SubfileName"]

    inspiral_params = inspiral_parameters(
        id_input_file,
        id_metadata,
        id_run_dir,
        id_subfile_name=id_subfile_name,
        id_horizons_path=None,
    )
    params = {key: inspiral_params[key] for key in _EVOLUTION_DOMAIN_KEYS}
    params.update(
        {
            "IdSubfile": id_subfile_name,
            # The excisions of the initial data, not the enlarged ones of the
            # evolution: the initial data is imposed on them as boundary
            # condition
            "ExcisionRadiusA": id_domain_creator["ObjectA"]["InnerRadius"],
            "ExcisionRadiusB": id_domain_creator["ObjectB"]["InnerRadius"],
            "HorizonLMax": id_domain_creator["TimeDependentMaps"]["ShapeMapA"][
                "LMax"
            ],
        }
    )

    # Post-Newtonian past evolution. 'XCoords' is [left, right] and object A is
    # the right-hand one, so MassA pairs with the right and MassB with the left.
    x_left, x_right = id_binary["XCoords"]
    separation = x_right - x_left
    eccentricity = target_params.get("Eccentricity")
    if eccentricity is None:
        logger.warning(
            "The initial data has no target eccentricity. Building the"
            " post-Newtonian past evolution on a quasi-circular orbit."
        )
        eccentricity = 0.0
    if eccentricity != 0.0:
        raise ValueError(
            "Only a target eccentricity of zero is supported by"
            f" 'NumericBinaryWithWaves' so far, got {eccentricity}."
        )
    params.update(
        {
            # Helical Killing vector of the initial data (the frame)
            "OrbitalAngularVelocity": float(id_binary["AngularVelocity"]),
            "RadialExpansionVelocity": float(id_binary["Expansion"]),
            "Separation": separation,
            "TargetEccentricity": eccentricity,
            "MassLeft": target_params["MassB"],
            "MassRight": target_params["MassA"],
            "AttenuationWidth": 0.3 * separation,
            # Covers the retarded time of the outermost grid point, whose
            # source is at most 'separation / 2' off center, with margin for
            # the retarded-time root find near the outer boundary
            "PastEvolutionDuration": (
                1.5 * (params["OuterShellRadius"] + separation)
            ),
            "PastEvolutionTimeStep": 0.5,
        }
    )
    return params


def add_waves(
    id_input_file_path: Union[str, Path],
    lev: Optional[int] = None,
    refinement_level: Optional[int] = None,
    polynomial_order: Optional[int] = None,
    id_run_dir: Optional[Union[str, Path]] = None,
    add_wave_input_file_template: Union[
        str, Path
    ] = ADD_WAVE_INPUT_FILE_TEMPLATE,
    **scheduler_kwargs,
):
    """Add post-Newtonian gravitational waves to BBH initial data.

    Point the ID_INPUT_FILE_PATH to the input file of your initial data run,
    i.e. the last iteration of the control loop. Also specify 'id_run_dir' if
    the initial data was run in a different directory than where the input file
    is.

    The initial data is loaded as background and initial guess of a second XCTS
    solve on the evolution domain, with post-Newtonian gravitational waves from
    the past inspiral of the binary added to the conformal metric (see
    'Xcts::AnalyticData::NumericBinaryWithWaves'). Specify the resolution like
    for 'start-inspiral', so the solve runs on the grid of the evolution. Once
    the solve is done, horizons are found in the new data (see
    'postprocess-id'). Evolve the result with 'start-inspiral', pointing it to
    the input file of this run. The remaining options are forwarded to the
    'schedule' command. See 'schedule' docs for details.
    """
    logger.warning(
        "The BBH pipeline is still experimental. Please review the"
        " generated input files."
    )

    # Determine parameters from initial data
    if id_run_dir is None:
        id_run_dir = Path(id_input_file_path).resolve().parent
    with open(id_input_file_path, "r") as open_input_file:
        id_metadata, id_input_file = yaml.safe_load_all(open_input_file)
    add_wave_params = add_wave_parameters(
        id_input_file, id_metadata, id_run_dir
    )

    # Determine resolution in the same way as 'start_inspiral', so the same
    # options select the grid of the evolution
    if lev is not None:
        assert (refinement_level is None) and (polynomial_order is None), (
            "The option 'lev' is mutually exclusive with 'refinement_level' and"
            " 'polynomial_order'."
        )
        selected_lev = INSPIRAL_LEVS[lev]
        refinement_level = selected_lev["refinement_level"]
        polynomial_order = selected_lev["polynomial_order"]
    else:
        assert (refinement_level is not None) and (
            polynomial_order is not None
        ), (
            "Resolution not specified. Provide either 'lev' or both"
            " 'refinement_level' and 'polynomial_order'."
        )
    add_wave_params.update({"L": refinement_level, "P": polynomial_order})

    logger.debug(f"Parameters for adding waves: {pretty_repr(add_wave_params)}")

    # Schedule!
    return schedule(
        add_wave_input_file_template, **add_wave_params, **scheduler_kwargs
    )


@click.command(name="add-waves", help=add_waves.__doc__)
@click.argument(
    "id_input_file_path",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
)
@click.option(
    "-i",
    "--id-run-dir",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        path_type=Path,
    ),
    help=(
        "Directory of the initial data run. Paths in the input file are"
        " relative to this directory."
    ),
    show_default="directory of the ID_INPUT_FILE_PATH",
)
@click.option(
    "--add-wave-input-file-template",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
    default=ADD_WAVE_INPUT_FILE_TEMPLATE,
    help="Input file template for the solve that adds waves.",
    show_default=True,
)
@click.option(
    "--lev",
    type=int,
    help=(
        "Resolution levels defined in terms of h and p refinement, as for"
        " 'start-inspiral'. Mutually exclusive with options '-L' and '-P'."
    ),
)
@click.option(
    "--refinement-level",
    "-L",
    type=int,
    help="h-refinement level.",
)
@click.option(
    "--polynomial-order",
    "-P",
    type=int,
    help="p-refinement level.",
)
@scheduler_options
def add_waves_command(**kwargs):
    _rich_traceback_guard = True  # Hide traceback until here
    add_waves(**kwargs)


if __name__ == "__main__":
    add_waves_command(help_option_names=["-h", "--help"])
