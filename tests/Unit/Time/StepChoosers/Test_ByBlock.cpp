// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "Time/StepChoosers/ByBlock.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/TimeStepRequest.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
constexpr size_t volume_dim = 2;

struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<StepChooser<StepChooserUse::LtsStep>,
                             tmpl::list<StepChoosers::ByBlock<
                                 StepChooserUse::LtsStep, volume_dim>>>,
                  tmpl::pair<StepChooser<StepChooserUse::Slab>,
                             tmpl::list<StepChoosers::ByBlock<
                                 StepChooserUse::Slab, volume_dim>>>>;
  };
  using component_list = tmpl::list<>;
};

template <typename Use>
void test_by_block() {
  using ByBlock = StepChoosers::ByBlock<Use, volume_dim>;

  const ByBlock by_block({2.5, 3.0, 3.5});
  const std::unique_ptr<StepChooser<Use>> by_block_base =
      std::make_unique<ByBlock>(by_block);

  const double current_step = std::numeric_limits<double>::infinity();
  for (size_t block = 0; block < 3; ++block) {
    const Element<volume_dim> element(ElementId<volume_dim>(block), {});
    auto box = db::create<
        db::AddSimpleTags<Parallel::Tags::MetavariablesImpl<Metavariables>,
                          domain::Tags::Element<volume_dim>>>(Metavariables{},
                                                              element);
    const TimeStepRequest expected{.size_goal =
                                       0.5 * static_cast<double>(block + 5)};

    CHECK(by_block(element, current_step) == std::make_pair(expected, true));
    CHECK(serialize_and_deserialize(by_block)(element, current_step) ==
          std::make_pair(expected, true));

    CHECK(by_block_base->desired_step(current_step, box) ==
          std::make_pair(expected, true));
    CHECK(serialize_and_deserialize(by_block_base)
              ->desired_step(current_step, box) ==
          std::make_pair(expected, true));
  }

  TestHelpers::test_factory_creation<StepChooser<Use>, ByBlock>(
      "ByBlock:\n"
      "  Sizes: [1.0, 2.0]");
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.StepChoosers.ByBlock", "[Unit][Time]") {
  register_factory_classes_with_charm<Metavariables>();

  test_by_block<StepChooserUse::LtsStep>();
  test_by_block<StepChooserUse::Slab>();

  CHECK(StepChoosers::ByBlock<StepChooserUse::Slab, 1>{}.uses_local_data());
}
