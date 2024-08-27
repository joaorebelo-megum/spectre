// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/ObservationBox.hpp"
#include "Domain/Amr/Flag.hpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Elliptic/Systems/Punctures/AmrCriteria/RefineAtPunctures.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Criterion.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Tags/Criteria.hpp"
#include "PointwiseFunctions/AnalyticData/Punctures/MultiplePunctures.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace Punctures::AmrCriteria {

namespace {

struct Metavariables {
  static constexpr size_t volume_dim = 3;
  using component_list = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<
        amr::Criterion, tmpl::list<Punctures::AmrCriteria::RefineAtPunctures>>>;
  };
};

}  // namespace

SPECTRE_TEST_CASE("Unit.Punctures.AmrCriteria.RefineAtPunctures",
                  "[Unit][ParallelAlgorithms]") {
  register_factory_classes_with_charm<Metavariables>();

  const auto created =
      TestHelpers::test_creation<std::unique_ptr<amr::Criterion>,
                                 Metavariables>("RefineAtPunctures");
  REQUIRE(dynamic_cast<const RefineAtPunctures*>(created.get()) != nullptr);
  const auto& criterion = serialize_and_deserialize(
      dynamic_cast<const RefineAtPunctures&>(*created));

  {
    INFO("Evaluate");
    using background_tag =
        elliptic::Tags::Background<elliptic::analytic_data::Background>;
    using MultiplePunctures = Punctures::AnalyticData::MultiplePunctures;
    using Puncture = Punctures::AnalyticData::Puncture;

    const domain::creators::Brick domain_creator{{{0., 0., 0.}},
                                                 {{1., 1., 1.}},
                                                 {{2, 2, 2}},
                                                 {{3, 3, 3}},
                                                 {{false, false, false}}};
    const Puncture puncture_within{// Position within an element
                                   {{0.2, 0.2, 0.2}},
                                   // Mass, spin, etc are irrelevant
                                   1.,
                                   {{0.1, 0.2, 0.3}},
                                   {{0.4, 0.5, 0.6}}};
    const Puncture puncture_boundary{// Position on an element boundary
                                     {{0.5, 0.5, 0.5}},
                                     // Mass, spin, etc are irrelevant
                                     1.,
                                     {{0.1, 0.2, 0.3}},
                                     {{0.4, 0.5, 0.6}}};
    auto databox =
        db::create<tmpl::list<background_tag, domain::Tags::Domain<3>>>(
            std::unique_ptr<elliptic::analytic_data::Background>(
                std::make_unique<MultiplePunctures>(
                    std::vector<Puncture>{puncture_within, puncture_boundary})),
            domain_creator.create_domain());
    ObservationBox<
        tmpl::list<>,
        db::DataBox<tmpl::list<background_tag, domain::Tags::Domain<3>>>>
        box{make_not_null(&databox)};
    Parallel::GlobalCache<Metavariables> empty_cache{};

    {
      INFO("Element with puncture within");
      const ElementId<3> element_id{0, {{{2, 0}, {2, 0}, {2, 0}}}};
      const auto expected_flags = make_array<3>(amr::Flag::Split);
      auto flags = criterion.evaluate(box, empty_cache, element_id);
      CHECK(flags == expected_flags);
    }
    {
      INFO("Elements with puncture on boundary");
      const auto expected_flags = make_array<3>(amr::Flag::Split);
      for (const auto& element_id :
           {ElementId<3>{0, {{{2, 1}, {2, 1}, {2, 1}}}},
            ElementId<3>{0, {{{2, 2}, {2, 1}, {2, 1}}}},
            ElementId<3>{0, {{{2, 1}, {2, 2}, {2, 1}}}},
            ElementId<3>{0, {{{2, 2}, {2, 2}, {2, 1}}}},
            ElementId<3>{0, {{{2, 1}, {2, 1}, {2, 2}}}},
            ElementId<3>{0, {{{2, 2}, {2, 1}, {2, 2}}}},
            ElementId<3>{0, {{{2, 1}, {2, 2}, {2, 2}}}},
            ElementId<3>{0, {{{2, 2}, {2, 2}, {2, 2}}}}}) {
        CAPTURE(element_id);
        auto flags = criterion.evaluate(box, empty_cache, element_id);
        CHECK(flags == expected_flags);
      }
    }
    {
      INFO("Elements without puncture");
      const auto expected_flags = make_array<3>(amr::Flag::DoNothing);
      for (const auto& element_id :
           {ElementId<3>{0, {{{2, 1}, {2, 0}, {2, 0}}}},
            ElementId<3>{0, {{{2, 3}, {2, 3}, {2, 3}}}},
            ElementId<3>{0, {{{2, 3}, {2, 0}, {2, 0}}}},
            ElementId<3>{0, {{{2, 0}, {2, 3}, {2, 1}}}}}) {
        auto flags = criterion.evaluate(box, empty_cache, element_id);
        CHECK(flags == expected_flags);
      }
    }
  }
}

}  // namespace Punctures::AmrCriteria
