// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "Domain/Amr/Flag.hpp"
#include "Domain/Amr/Info.hpp"
#include "Domain/Amr/Tags/Flags.hpp"
#include "Domain/Amr/Tags/NeighborFlags.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/MockRuntimeSystem.hpp"
#include "Framework/MockRuntimeSystemFreeFunctions.hpp"
#include "Helpers/Domain/Amr/RegistrationHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Protocols/RegistrationMetavariables.hpp"
#include "ParallelAlgorithms/Amr/Actions/InitializeChild.hpp"
#include "ParallelAlgorithms/Amr/Protocols/AmrMetavariables.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  static constexpr size_t volume_dim = Metavariables::volume_dim;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<volume_dim>;
  using const_global_cache_tags = tmpl::list<>;
  using simple_tags = tmpl::list<
      domain::Tags::Element<volume_dim>, domain::Tags::Mesh<volume_dim>,
      domain::Tags::NeighborMesh<volume_dim>, amr::Tags::Info<volume_dim>,
      amr::Tags::NeighborInfo<volume_dim>>;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      Parallel::Phase::Initialization,
      tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>>;
};

struct Metavariables {
  static constexpr size_t volume_dim = 2;
  using component_list = tmpl::list<Component<Metavariables>,
                                    TestHelpers::amr::Registrar<Metavariables>>;
  struct registration
      : tt::ConformsTo<Parallel::protocols::RegistrationMetavariables> {
    using element_registrars =
        tmpl::map<tmpl::pair<Component<Metavariables>,
                             tmpl::list<TestHelpers::amr::RegisterElement>>>;
  };

  struct amr : tt::ConformsTo<::amr::protocols::AmrMetavariables> {
    using projectors = tmpl::list<>;
  };
};

void test() {
  const ElementId<2> parent_id{0, std::array{SegmentId{2, 1}, SegmentId{0, 0}}};
  const ElementId<2> parent_lower_neighbor_id{
      0, std::array{SegmentId{2, 0}, SegmentId{0, 0}}};
  const ElementId<2> parent_upper_neighbor_id{
      0, std::array{SegmentId{2, 2}, SegmentId{0, 0}}};
  DirectionMap<2, Neighbors<2>> parent_neighbors{};
  OrientationMap<2> aligned = OrientationMap<2>::create_aligned();
  parent_neighbors.emplace(
      Direction<2>::lower_xi(),
      Neighbors<2>{std::unordered_set{parent_lower_neighbor_id}, aligned});
  parent_neighbors.emplace(
      Direction<2>::upper_xi(),
      Neighbors<2>{std::unordered_set{parent_upper_neighbor_id}, aligned});
  Element<2> parent{parent_id, std::move(parent_neighbors)};
  Mesh<2> parent_mesh{3, Spectral::Basis::Legendre,
                      Spectral::Quadrature::GaussLobatto};
  const Mesh<2> expected_child_mesh{std::array{3_st, 4_st},
                                    Spectral::Basis::Legendre,
                                    Spectral::Quadrature::GaussLobatto};
  amr::Info<2> parent_info{
      std::array{amr::Flag::Split, amr::Flag::IncreaseResolution},
      expected_child_mesh};
  const Mesh<2> neighbor_mesh{std::array{4_st, 5_st}, Spectral::Basis::Legendre,
                              Spectral::Quadrature::GaussLobatto};
  std::unordered_map<ElementId<2>, amr::Info<2>> parent_neighbor_info;
  parent_neighbor_info.emplace(
      parent_lower_neighbor_id,
      amr::Info<2>{std::array{amr::Flag::DoNothing, amr::Flag::DoNothing},
                   neighbor_mesh});
  parent_neighbor_info.emplace(
      parent_upper_neighbor_id,
      amr::Info<2>{std::array{amr::Flag::DoNothing, amr::Flag::Split},
                   neighbor_mesh});
  const tuples::TaggedTuple<domain::Tags::Element<2>, domain::Tags::Mesh<2>,
                            amr::Tags::Info<2>, amr::Tags::NeighborInfo<2>>
      parent_items{std::move(parent), std::move(parent_mesh),
                   std::move(parent_info), std::move(parent_neighbor_info)};

  const ElementId<2> child_id{0, std::array{SegmentId{3, 3}, SegmentId{0, 0}}};
  const ElementId<2> expected_child_upper_neighbor_id_0{
      0, std::array{SegmentId{2, 2}, SegmentId{1, 0}}};
  const ElementId<2> expected_child_upper_neighbor_id_1{
      0, std::array{SegmentId{2, 2}, SegmentId{1, 1}}};
  const ElementId<2> sibling_id{0,
                                std::array{SegmentId{3, 2}, SegmentId{0, 0}}};
  DirectionMap<2, Neighbors<2>> expected_child_neighbors{};
  expected_child_neighbors.emplace(
      Direction<2>::lower_xi(),
      Neighbors<2>{std::unordered_set{sibling_id}, aligned});
  expected_child_neighbors.emplace(
      Direction<2>::upper_xi(),
      Neighbors<2>{std::unordered_set{expected_child_upper_neighbor_id_0,
                                      expected_child_upper_neighbor_id_1},
                   aligned});
  const Element<2> expected_child{child_id,
                                  std::move(expected_child_neighbors)};
  DirectionalIdMap<2, Mesh<2>> expected_child_neighbor_mesh{};
  expected_child_neighbor_mesh.emplace(
      DirectionalId<2>{Direction<2>::lower_xi(), sibling_id},
      expected_child_mesh);
  expected_child_neighbor_mesh.emplace(
      DirectionalId<2>{Direction<2>::upper_xi(),
                       expected_child_upper_neighbor_id_0},
      neighbor_mesh);
  expected_child_neighbor_mesh.emplace(
      DirectionalId<2>{Direction<2>::upper_xi(),
                       expected_child_upper_neighbor_id_1},
      neighbor_mesh);

  const amr::Info<2> expected_child_info{
      std::array{amr::Flag::Undefined, amr::Flag::Undefined}, Mesh<2>{}};
  const std::unordered_map<ElementId<2>, amr::Info<2>>
      expected_child_neighbor_info{};

  using array_component = Component<Metavariables>;
  using registrar = TestHelpers::amr::Registrar<Metavariables>;

  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}};
  ActionTesting::emplace_group_component<registrar>(&runner);
  ActionTesting::emplace_component<array_component>(&runner, child_id);
  CHECK(ActionTesting::get_databox_tag<registrar,
                                       TestHelpers::amr::RegisteredElements<2>>(
            runner, 0)
            .empty());
  ActionTesting::simple_action<array_component, amr::Actions::InitializeChild>(
      make_not_null(&runner), child_id, parent_items);
  CHECK(
      ActionTesting::get_databox_tag<array_component, domain::Tags::Element<2>>(
          runner, child_id) == expected_child);
  CHECK(ActionTesting::get_databox_tag<array_component, domain::Tags::Mesh<2>>(
            runner, child_id) == expected_child_mesh);
  CHECK(ActionTesting::get_databox_tag<array_component,
                                       domain::Tags::NeighborMesh<2>>(
            runner, child_id) == expected_child_neighbor_mesh);
  CHECK(ActionTesting::get_databox_tag<array_component, amr::Tags::Info<2>>(
            runner, child_id) == expected_child_info);
  CHECK(ActionTesting::get_databox_tag<array_component,
                                       amr::Tags::NeighborInfo<2>>(
            runner, child_id) == expected_child_neighbor_info);
  CHECK(ActionTesting::get_databox_tag<registrar,
                                       TestHelpers::amr::RegisteredElements<2>>(
            runner, 0)
            .empty());
  ActionTesting::invoke_queued_simple_action<registrar>(make_not_null(&runner),
                                                        0);
  CHECK(ActionTesting::get_databox_tag<registrar,
                                       TestHelpers::amr::RegisteredElements<2>>(
            runner, 0) == std::unordered_set{child_id});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Amr.Actions.InitializeChild",
                  "[Unit][ParallelAlgorithms]") {
  test();
}
