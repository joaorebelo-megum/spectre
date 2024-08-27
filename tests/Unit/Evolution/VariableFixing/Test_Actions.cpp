// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/VariableFixing/Actions.hpp"
#include "Evolution/VariableFixing/RadiallyFallingFloor.hpp"
#include "Evolution/VariableFixing/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {

struct EquationOfStateTag : db::SimpleTag, hydro::Tags::EquationOfStateBase {
  using type = EquationsOfState::PolytropicFluid<true>;
};

template <typename Metavariables>
struct mock_component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = size_t;
  using simple_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::SpecificInternalEnergy<DataVector>,
                 hydro::Tags::SpecificEnthalpy<DataVector>,
                 hydro::Tags::Temperature<DataVector>,
                 hydro::Tags::ElectronFraction<DataVector>,
                 domain::Tags::Coordinates<3, Frame::Inertial>,
                 EquationOfStateTag>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>,
      Parallel::PhaseActions<Parallel::Phase::Testing,
                             tmpl::list<VariableFixing::Actions::FixVariables<
                                 VariableFixing::RadiallyFallingFloor<3>>>>>;
};

struct Metavariables {
  using component_list = tmpl::list<mock_component<Metavariables>>;
};

struct SomeType {};
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.VariableFixing.Actions",
                  "[Unit][Evolution][VariableFixing]") {
  TestHelpers::db::test_simple_tag<Tags::VariableFixer<SomeType>>(
      "VariableFixer");

  using component = mock_component<Metavariables>;
  const DataVector x{-2.0, -1.0, 0.0, 1.0, 2.0};
  const DataVector y{-2.0, -1.0, 0.0, 1.0, 2.0};
  const DataVector z{-2.0, -1.0, 0.0, 1.0, 2.0};

  ActionTesting::MockRuntimeSystem<Metavariables> runner{
      VariableFixing::RadiallyFallingFloor<3>(1.e-4, 1.e-5, -1.5, 1.e-7 / 3.0,
                                              -2.5)};

  EquationsOfState::PolytropicFluid<true> polytrope{1.0, 2.0};

  const double root_three = sqrt(3.0);

  const DataVector fixed_density{
      2.3, 1.e-5 * pow(3, -0.75),
      1.e-10,  // quantities at a radius below
               // `radius_at_which_to_begin_applying_floor`
               // do not get fixed.
      1.e-5 * pow(3, -0.75), 1.e-5 * pow(2.0 * root_three, -1.5)};

  auto fixed_pressure =
      get(polytrope.pressure_from_density(Scalar<DataVector>{fixed_density}));

  // This pressure term will remain the same b/c the radial coordinate is at the
  // origin
  fixed_pressure[2] = 2.0;

  ActionTesting::emplace_component_and_initialize<component>(
      &runner, 0,
      {Scalar<DataVector>{DataVector{2.3, -4.2, 1.e-10, 0.0, -0.1}},
       Scalar<DataVector>{DataVector{0.0, 1.e-8, 2.0, -5.5, 3.2}},
       Scalar<DataVector>{DataVector{1.0, 2.0, 3.0, 4.0, 5.0}},
       Scalar<DataVector>{DataVector{1.0, 2.0, 3.0, 4.0, 5.0}},
       Scalar<DataVector>{DataVector{1.0, 2.0, 3.0, 4.0, 5.0}},
       Scalar<DataVector>{DataVector{1.0, 1.0, 1.0, 1.0, 1.0}},
       tnsr::I<DataVector, 3, Frame::Inertial>{{{x, y, z}}},
       std::move(polytrope)});
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  auto& box = ActionTesting::get_databox<component>(runner, 0);
  runner.next_action<component>(0);

  CHECK_ITERABLE_APPROX(db::get<hydro::Tags::Pressure<DataVector>>(box).get(),
                        fixed_pressure);
  CHECK_ITERABLE_APPROX(
      db::get<hydro::Tags::RestMassDensity<DataVector>>(box).get(),
      fixed_density);
}
