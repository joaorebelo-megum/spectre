// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <deque>
#include <pup.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Domain/Amr/Flag.hpp"
#include "Domain/Amr/Helpers.hpp"
#include "Domain/Amr/Info.hpp"
#include "Domain/Amr/Tags/Flags.hpp"
#include "Domain/Amr/Tags/NeighborFlags.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/Domain/Amr/RegistrationHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Protocols/RegistrationMetavariables.hpp"
#include "ParallelAlgorithms/Amr/Actions/CollectDataFromChildren.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
const ElementId<2> parent_id{0, std::array{SegmentId{0, 0}, SegmentId{0, 0}}};
const ElementId<2> child_1_id{0, std::array{SegmentId{1, 0}, SegmentId{1, 0}}};
const ElementId<2> child_2_id{0, std::array{SegmentId{1, 1}, SegmentId{1, 0}}};
const ElementId<2> child_3_id{0, std::array{SegmentId{1, 0}, SegmentId{1, 1}}};
const ElementId<2> child_4_id{0, std::array{SegmentId{1, 1}, SegmentId{1, 1}}};
const ElementId<2> neighbor_1_id{1,
                                 std::array{SegmentId{1, 1}, SegmentId{0, 0}}};
const ElementId<2> neighbor_2_id{2,
                                 std::array{SegmentId{0, 0}, SegmentId{1, 0}}};
const ElementId<2> neighbor_3_id{2,
                                 std::array{SegmentId{1, 0}, SegmentId{1, 1}}};

auto child_1_mesh() {
  return Mesh<2>{std::array{3_st, 3_st}, Spectral::Basis::Legendre,
                 Spectral::Quadrature::GaussLobatto};
}
auto child_2_mesh() {
  return Mesh<2>{std::array{3_st, 4_st}, Spectral::Basis::Legendre,
                 Spectral::Quadrature::GaussLobatto};
}
auto child_3_mesh() {
  return Mesh<2>{std::array{4_st, 3_st}, Spectral::Basis::Legendre,
                 Spectral::Quadrature::GaussLobatto};
}
auto child_4_mesh() {
  return Mesh<2>{std::array{4_st, 4_st}, Spectral::Basis::Legendre,
                 Spectral::Quadrature::GaussLobatto};
}

auto neighbor_1_mesh() {
  return Mesh<2>{std::array{5_st, 4_st}, Spectral::Basis::Legendre,
                 Spectral::Quadrature::GaussLobatto};
}

auto neighbor_2_mesh() {
  return Mesh<2>{std::array{2_st, 3_st}, Spectral::Basis::Legendre,
                 Spectral::Quadrature::GaussLobatto};
}

auto neighbor_3_mesh() {
  return Mesh<2>{std::array{3_st, 5_st}, Spectral::Basis::Legendre,
                 Spectral::Quadrature::GaussLobatto};
}

auto child_info() {
  return amr::Info<2>{std::array{amr::Flag::Join, amr::Flag::Join},
                      child_4_mesh()};
}

auto neighbor_1_info() {
  return amr::Info<2>{std::array{amr::Flag::Join, amr::Flag::DoNothing},
                      neighbor_1_mesh()};
}

auto neighbor_2_info() {
  return amr::Info<2>{std::array{amr::Flag::Split, amr::Flag::DoNothing},
                      neighbor_2_mesh()};
}

auto neighbor_3_info() {
  return amr::Info<2>{std::array{amr::Flag::Join, amr::Flag::DoNothing},
                      neighbor_3_mesh()};
}

Element<2> child_1() {
  static Element<2> result{
      child_1_id, DirectionMap<2, Neighbors<2>>{
                      {Direction<2>::lower_xi(),
                       Neighbors<2>{std::unordered_set{neighbor_1_id},
                                    OrientationMap<2>::create_aligned()}},
                      {Direction<2>::upper_xi(),
                       Neighbors<2>{std::unordered_set{child_2_id},
                                    OrientationMap<2>::create_aligned()}},
                      {Direction<2>::lower_eta(),
                       Neighbors<2>{std::unordered_set{child_3_id},
                                    OrientationMap<2>::create_aligned()}},
                      {Direction<2>::upper_eta(),
                       Neighbors<2>{std::unordered_set{child_3_id},
                                    OrientationMap<2>::create_aligned()}}}};
  return result;
}

Element<2> child_2() {
  static Element<2> result{
      child_2_id, DirectionMap<2, Neighbors<2>>{
                      {Direction<2>::lower_xi(),
                       Neighbors<2>{std::unordered_set{child_1_id},
                                    OrientationMap<2>::create_aligned()}},
                      {Direction<2>::upper_xi(),
                       Neighbors<2>{std::unordered_set{neighbor_2_id},
                                    OrientationMap<2>{
                                        std::array{Direction<2>::lower_eta(),
                                                   Direction<2>::upper_xi()}}}},
                      {Direction<2>::lower_eta(),
                       Neighbors<2>{std::unordered_set{child_4_id},
                                    OrientationMap<2>::create_aligned()}},
                      {Direction<2>::upper_eta(),
                       Neighbors<2>{std::unordered_set{child_4_id},
                                    OrientationMap<2>::create_aligned()}}}};
  return result;
}

Element<2> child_3() {
  static Element<2> result{
      child_3_id, DirectionMap<2, Neighbors<2>>{
                      {Direction<2>::lower_xi(),
                       Neighbors<2>{std::unordered_set{neighbor_1_id},
                                    OrientationMap<2>::create_aligned()}},
                      {Direction<2>::upper_xi(),
                       Neighbors<2>{std::unordered_set{child_4_id},
                                    OrientationMap<2>::create_aligned()}},
                      {Direction<2>::lower_eta(),
                       Neighbors<2>{std::unordered_set{child_1_id},
                                    OrientationMap<2>::create_aligned()}},
                      {Direction<2>::upper_eta(),
                       Neighbors<2>{std::unordered_set{child_1_id},
                                    OrientationMap<2>::create_aligned()}}}};
  return result;
}

Element<2> child_4() {
  static Element<2> result{
      child_4_id, DirectionMap<2, Neighbors<2>>{
                      {Direction<2>::lower_xi(),
                       Neighbors<2>{std::unordered_set{child_3_id},
                                    OrientationMap<2>::create_aligned()}},
                      {Direction<2>::upper_xi(),
                       Neighbors<2>{std::unordered_set{neighbor_3_id},
                                    OrientationMap<2>{
                                        std::array{Direction<2>::lower_eta(),
                                                   Direction<2>::upper_xi()}}}},
                      {Direction<2>::lower_eta(),
                       Neighbors<2>{std::unordered_set{child_2_id},
                                    OrientationMap<2>::create_aligned()}},
                      {Direction<2>::upper_eta(),
                       Neighbors<2>{std::unordered_set{child_2_id},
                                    OrientationMap<2>::create_aligned()}}}};
  return result;
}

auto child_1_neighbor_info() {
  return std::unordered_map<ElementId<2>, amr::Info<2>>{
      {neighbor_1_id, neighbor_1_info()},
      {child_2_id, child_info()},
      {child_3_id, child_info()}};
}

auto child_2_neighbor_info() {
  return std::unordered_map<ElementId<2>, amr::Info<2>>{
      {child_1_id, child_info()},
      {child_4_id, child_info()},
      {neighbor_2_id, neighbor_2_info()}};
}

auto child_3_neighbor_info() {
  return std::unordered_map<ElementId<2>, amr::Info<2>>{
      {neighbor_1_id, neighbor_1_info()},
      {child_1_id, child_info()},
      {child_4_id, child_info()}};
}
auto child_4_neighbor_info() {
  return std::unordered_map<ElementId<2>, amr::Info<2>>{
      {child_2_id, child_info()},
      {child_3_id, child_info()},
      {neighbor_3_id, neighbor_3_info()}};
}

struct MockInitializeParent {
  template <typename ParallelComponent, typename DataBox,
            typename Metavariables, typename... Tags>
  static void apply(
      DataBox& /*box*/, const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Metavariables::volume_dim>& mock_parent_id,
      const std::unordered_map<ElementId<Metavariables::volume_dim>,
                               tuples::TaggedTuple<Tags...>>& children_items) {
    CHECK(mock_parent_id == parent_id);
    const auto& child_1_items = children_items.at(child_1_id);
    CHECK(get<domain::Tags::Element<2>>(child_1_items) == child_1());
    CHECK(get<domain::Tags::Mesh<2>>(child_1_items) == child_1_mesh());
    CHECK(get<amr::Tags::Info<2>>(child_1_items) == child_info());
    CHECK(get<amr::Tags::NeighborInfo<2>>(child_1_items) ==
          child_1_neighbor_info());

    const auto& child_2_items = children_items.at(child_2_id);
    CHECK(get<domain::Tags::Element<2>>(child_2_items) == child_2());
    CHECK(get<domain::Tags::Mesh<2>>(child_2_items) == child_2_mesh());
    CHECK(get<amr::Tags::Info<2>>(child_2_items) == child_info());
    CHECK(get<amr::Tags::NeighborInfo<2>>(child_2_items) ==
          child_2_neighbor_info());

    const auto& child_3_items = children_items.at(child_3_id);
    CHECK(get<domain::Tags::Element<2>>(child_3_items) == child_3());
    CHECK(get<domain::Tags::Mesh<2>>(child_3_items) == child_3_mesh());
    CHECK(get<amr::Tags::Info<2>>(child_3_items) == child_info());
    CHECK(get<amr::Tags::NeighborInfo<2>>(child_3_items) ==
          child_3_neighbor_info());

    const auto& child_4_items = children_items.at(child_4_id);
    CHECK(get<domain::Tags::Element<2>>(child_4_items) == child_4());
    CHECK(get<domain::Tags::Mesh<2>>(child_4_items) == child_4_mesh());
    CHECK(get<amr::Tags::Info<2>>(child_4_items) == child_info());
    CHECK(get<amr::Tags::NeighborInfo<2>>(child_4_items) ==
          child_4_neighbor_info());
  }
};

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  static constexpr size_t volume_dim = Metavariables::volume_dim;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<volume_dim>;
  using const_global_cache_tags = tmpl::list<>;
  using simple_tags =
      tmpl::list<domain::Tags::Element<volume_dim>,
                 domain::Tags::Mesh<volume_dim>, amr::Tags::Info<volume_dim>,
                 amr::Tags::NeighborInfo<volume_dim>>;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>>;
  using replace_these_simple_actions =
      tmpl::list<amr::Actions::InitializeParent>;
  using with_these_simple_actions = tmpl::list<MockInitializeParent>;
};

struct Metavariables {
  static constexpr size_t volume_dim = 2;
  using component_list = tmpl::list<Component<Metavariables>,
                                    TestHelpers::amr::Registrar<Metavariables>>;
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) {}

  struct registration
      : tt::ConformsTo<Parallel::protocols::RegistrationMetavariables> {
    using element_registrars =
        tmpl::map<tmpl::pair<Component<Metavariables>,
                             tmpl::list<TestHelpers::amr::RegisterElement>>>;
  };
};

void test() {
  using array_component = Component<Metavariables>;
  using registrar = TestHelpers::amr::Registrar<Metavariables>;

  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}};
  ActionTesting::emplace_component_and_initialize<array_component>(
      &runner, child_1_id,
      {child_1(), child_1_mesh(), child_info(), child_1_neighbor_info()});
  ActionTesting::emplace_component_and_initialize<array_component>(
      &runner, child_2_id,
      {child_2(), child_2_mesh(), child_info(), child_2_neighbor_info()});
  ActionTesting::emplace_component_and_initialize<array_component>(
      &runner, child_3_id,
      {child_3(), child_3_mesh(), child_info(), child_3_neighbor_info()});
  ActionTesting::emplace_component_and_initialize<array_component>(
      &runner, child_4_id,
      {child_4(), child_4_mesh(), child_info(), child_4_neighbor_info()});
  ActionTesting::emplace_component<array_component>(&runner, parent_id);
  ActionTesting::emplace_group_component_and_initialize<registrar>(
      &runner,
      std::unordered_set{child_1_id, child_2_id, child_3_id, child_4_id});

  for (const auto& id :
       std::vector{child_1_id, child_2_id, child_3_id, child_4_id, parent_id}) {
    CHECK(ActionTesting::is_simple_action_queue_empty<array_component>(runner,
                                                                       id));
  }

  ActionTesting::simple_action<array_component,
                               amr::Actions::CollectDataFromChildren>(
      make_not_null(&runner), child_1_id, parent_id,
      std::deque{child_2_id, child_3_id});
  for (const auto& id :
       std::vector{child_1_id, child_3_id, child_4_id, parent_id}) {
    CHECK(ActionTesting::is_simple_action_queue_empty<array_component>(runner,
                                                                       id));
  }
  CHECK(ActionTesting::number_of_queued_simple_actions<array_component>(
            runner, child_2_id) == 1);
  ActionTesting::invoke_queued_simple_action<registrar>(make_not_null(&runner),
                                                        0);
  CHECK(ActionTesting::get_databox_tag<registrar,
                                       TestHelpers::amr::RegisteredElements<2>>(
            runner, 0) ==
        std::unordered_set{child_2_id, child_3_id, child_4_id});

  ActionTesting::invoke_queued_simple_action<array_component>(
      make_not_null(&runner), child_2_id);
  for (const auto& id :
       std::vector{child_1_id, child_2_id, child_4_id, parent_id}) {
    CHECK(ActionTesting::is_simple_action_queue_empty<array_component>(runner,
                                                                       id));
  }
  CHECK(ActionTesting::number_of_queued_simple_actions<array_component>(
            runner, child_3_id) == 1);
  ActionTesting::invoke_queued_simple_action<registrar>(make_not_null(&runner),
                                                        0);
  CHECK(ActionTesting::get_databox_tag<registrar,
                                       TestHelpers::amr::RegisteredElements<2>>(
            runner, 0) == std::unordered_set{child_3_id, child_4_id});

  ActionTesting::invoke_queued_simple_action<array_component>(
      make_not_null(&runner), child_3_id);
  for (const auto& id :
       std::vector{child_1_id, child_2_id, child_3_id, parent_id}) {
    CHECK(ActionTesting::is_simple_action_queue_empty<array_component>(runner,
                                                                       id));
  }
  CHECK(ActionTesting::number_of_queued_simple_actions<array_component>(
            runner, child_4_id) == 1);
  ActionTesting::invoke_queued_simple_action<registrar>(make_not_null(&runner),
                                                        0);
  CHECK(ActionTesting::get_databox_tag<registrar,
                                       TestHelpers::amr::RegisteredElements<2>>(
            runner, 0) == std::unordered_set{child_4_id});

  ActionTesting::invoke_queued_simple_action<array_component>(
      make_not_null(&runner), child_4_id);
  for (const auto& id :
       std::vector{child_1_id, child_2_id, child_3_id, child_4_id}) {
    CHECK(ActionTesting::is_simple_action_queue_empty<array_component>(runner,
                                                                       id));
  }
  CHECK(ActionTesting::number_of_queued_simple_actions<array_component>(
            runner, parent_id) == 1);
  ActionTesting::invoke_queued_simple_action<registrar>(make_not_null(&runner),
                                                        0);
  CHECK(ActionTesting::get_databox_tag<registrar,
                                       TestHelpers::amr::RegisteredElements<2>>(
            runner, 0) == std::unordered_set<ElementId<2>>{});

  ActionTesting::invoke_queued_simple_action<array_component>(
      make_not_null(&runner), parent_id);
  for (const auto& id :
       std::vector{child_1_id, child_2_id, child_3_id, child_4_id, parent_id}) {
    CHECK(ActionTesting::is_simple_action_queue_empty<array_component>(runner,
                                                                       id));
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Amr.Actions.CollectDataFromChildren",
                  "[Unit][ParallelAlgorithms]") {
  test();
}
