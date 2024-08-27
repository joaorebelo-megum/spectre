// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Tags/Metavariables.hpp"
#include "Time/Slab.hpp"
#include "Time/StepChoosers/PreventRapidIncrease.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeStepRequest.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using Frac = Time::rational_t;

struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<StepChooser<StepChooserUse::LtsStep>,
                             tmpl::list<StepChoosers::PreventRapidIncrease<
                                 StepChooserUse::LtsStep>>>,
                  tmpl::pair<StepChooser<StepChooserUse::Slab>,
                             tmpl::list<StepChoosers::PreventRapidIncrease<
                                 StepChooserUse::Slab>>>>;
  };
  using component_list = tmpl::list<>;
};

void check_case(const Frac& expected_frac, const std::vector<Frac>& times) {
  CAPTURE(times);
  CAPTURE(expected_frac);

  const Slab slab(0.25, 1.5);
  for (const auto& direction : {1, -1}) {
    CAPTURE(direction);

    const std::optional<double> expected_size =
        expected_frac == -1
            ? std::nullopt
            : std::optional(
                  (direction * expected_frac * slab.duration()).value());

    const auto make_time_id = [&direction, &slab, &times](const size_t i) {
      Frac frac = -direction * times[i];
      int64_t slab_number = 0;
      Slab time_slab = slab;
      while (frac > 1) {
        time_slab = time_slab.advance();
        frac -= 1;
        slab_number += direction;
      }
      while (frac < 0) {
        time_slab = time_slab.retreat();
        frac += 1;
        slab_number -= direction;
      }
      return TimeStepId(direction > 0, slab_number, Time(time_slab, frac));
    };

    struct Tag : db::SimpleTag {
      using type = double;
    };
    using history_tag = Tags::HistoryEvolvedVariables<Tag>;
    typename history_tag::type lts_history{};
    typename history_tag::type gts_history{};

    const auto make_gts_time_id = [&direction, &make_time_id](const size_t i) {
      const double time = make_time_id(i).substep_time();
      const double next_time =
          i > 0 ? make_time_id(i - 1).substep_time() : time + direction;
      const Slab gts_slab(std::min(time, next_time), std::max(time, next_time));
      return TimeStepId(direction > 0, -static_cast<int64_t>(i),
                        direction > 0 ? gts_slab.start() : gts_slab.end());
    };

    for (size_t i = 1; i < times.size(); ++i) {
      lts_history.insert_initial(make_time_id(i), 0.0, 0.0);
      gts_history.insert_initial(make_gts_time_id(i), 0.0, 0.0);
    }

    const auto check = [&direction, &expected_size](auto use, const auto& box,
                                                    const Time& current_time) {
      using Use = tmpl::type_from<decltype(use)>;
      const auto& history = db::get<history_tag>(box);
      const double current_step =
          history.size() > 0
              ? (current_time - history.back().time_step_id.step_time()).value()
              : direction * std::numeric_limits<double>::infinity();

      const StepChoosers::PreventRapidIncrease<Use> relax{};
      const std::unique_ptr<StepChooser<Use>> relax_base =
          std::make_unique<StepChoosers::PreventRapidIncrease<Use>>(relax);

      const std::pair<TimeStepRequest, bool> expected{{.size = expected_size},
                                                      true};
      CHECK(relax(history, current_step) == expected);
      CHECK(serialize_and_deserialize(relax)(history, current_step) ==
            expected);
      CHECK(relax_base->desired_step(current_step, box) == expected);
      CHECK(serialize_and_deserialize(relax_base)
                ->desired_step(current_step, box) == expected);
    };

    {
      CAPTURE(lts_history);
      const auto box = db::create<db::AddSimpleTags<
          Parallel::Tags::MetavariablesImpl<Metavariables>, history_tag>>(
          Metavariables{}, std::move(lts_history));
      check(tmpl::type_<StepChooserUse::LtsStep>{}, box,
            make_time_id(0).step_time());
      check(tmpl::type_<StepChooserUse::Slab>{}, box,
            make_time_id(0).step_time());
    }

    {
      CAPTURE(gts_history);
      const auto box = db::create<db::AddSimpleTags<
          Parallel::Tags::MetavariablesImpl<Metavariables>, history_tag>>(
          Metavariables{}, std::move(gts_history));
      check(tmpl::type_<StepChooserUse::LtsStep>{}, box,
            make_gts_time_id(0).step_time());
      check(tmpl::type_<StepChooserUse::Slab>{}, box,
            make_gts_time_id(0).step_time());
    }
  }
}

void check_substep_methods() {
  const Slab slab(0.25, 1.5);

  struct Tag : db::SimpleTag {
    using type = double;
  };
  using history_tag = Tags::HistoryEvolvedVariables<Tag>;
  typename history_tag::type history{};

  history.insert(TimeStepId(true, 0, slab.start()), 0.0, 0.0);
  history.insert(TimeStepId(true, 0, slab.start(), 1, slab.duration(),
                            (slab.start() + slab.duration() / 3).value()),
                 0.0, 0.0);
  history.insert(TimeStepId(true, 0, slab.start(), 2, slab.duration(),
                            (slab.start() + slab.duration() / 2).value()),
                 0.0, 0.0);
  const StepChoosers::PreventRapidIncrease<StepChooserUse::Slab> relax{};
  CHECK(relax(history, 3.14) == std::pair(TimeStepRequest{}, true));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.StepChoosers.PreventRapidIncrease",
                  "[Unit][Time]") {
  register_factory_classes_with_charm<Metavariables>();

  // -1 indicates no expected restriction
  check_case(-1, {0});
  check_case(-1, {{5, 8}});
  check_case(-1, {0, {5, 8}});
  check_case(-1, {{4, 8}, {5, 8}, {6, 8}});
  check_case(-1, {{7, 8}, {8, 8}, {9, 8}});
  check_case(-1, {{7, 8}, {8, 8}, {10, 8}});
  check_case({1, 8}, {{7, 8}, {8, 8}, {17, 16}});
  check_case({2, 8}, {{6, 8}, {8, 8}, {9, 8}});
  check_case(-1, {{4, 8}, {5, 8}, {6, 8}, {7, 8}});
  check_case({2, 8}, {{3, 8}, {5, 8}, {6, 8}, {7, 8}});
  check_case({1, 8}, {{4, 8}, {5, 8}, {7, 8}, {8, 8}});

  // Cause roundoff errors
  check_case(-1, {{1, 3}, {2, 3}, {3, 3}});

  check_substep_methods();

  TestHelpers::test_creation<
      std::unique_ptr<StepChooser<StepChooserUse::LtsStep>>, Metavariables>(
      "PreventRapidIncrease");
  TestHelpers::test_creation<std::unique_ptr<StepChooser<StepChooserUse::Slab>>,
                             Metavariables>("PreventRapidIncrease");

  CHECK(not StepChoosers::PreventRapidIncrease<StepChooserUse::Slab>{}
                .uses_local_data());
}
