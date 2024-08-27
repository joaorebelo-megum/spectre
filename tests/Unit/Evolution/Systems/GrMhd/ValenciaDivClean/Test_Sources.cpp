// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Sources.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.Sources", "[Unit][GrMhd]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GrMhd/ValenciaDivClean"};

  pypp::check_with_random_values<1>(
      &grmhd::ValenciaDivClean::ComputeSources::apply, "Sources",
      {"source_tilde_tau", "source_tilde_s", "source_tilde_b",
       "source_tilde_phi"},
      {{{0.0, 1.0}}}, DataVector{5});
}
