// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Time/Tags/StepperErrorTolerances.hpp"

namespace {
struct Vars {};

SPECTRE_TEST_CASE("Unit.Time.Tags.StepperErrorTolerances", "[Unit][Time]") {
  TestHelpers::db::test_simple_tag<Tags::StepperErrorTolerances<Vars>>(
      "StepperErrorTolerances(Vars)");
}
}  // namespace
