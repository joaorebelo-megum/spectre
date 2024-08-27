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
#include "ParallelAlgorithms/Amr/Actions/InitializeParent.hpp"
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
  static constexpr size_t volume_dim = 3;
  using component_list = tmpl::list<Component<Metavariables>,
                                    TestHelpers::amr::Registrar<Metavariables>>;
  // [registration_metavariables]
  struct registration
      : tt::ConformsTo<Parallel::protocols::RegistrationMetavariables> {
    using element_registrars =
        tmpl::map<tmpl::pair<Component<Metavariables>,
                             tmpl::list<TestHelpers::amr::RegisterElement>>>;
  };
  // [registration_metavariables]

  struct amr : tt::ConformsTo<::amr::protocols::AmrMetavariables> {
    using projectors = tmpl::list<>;
  };
};

// Test setup showing xi and eta dimensions which are hp-refined; only
// p-refinement in zeta dimension

// Before refinement:                     After refinement:
//
//       |-----------|                         |-----------|
//       |           |                         |           |
//       |    N5     |                         |    N5     |
//       |           |                         |           |
// |-----|-----------|-----------|       |-----|-----------|-----------|
// |     |           |           |       |     |           |           |
// |  N6 |    C3     |           |       |  N6 |           |    N11    |
// |     |           |           |       |     |           |           |
// |--|--|-----|-----|    N4     |       |-----|     P     |-----------|
//    |N7|     |     |           |       |     |           |           |
//    |--|  C1 |  C2 |           |       | N12 |           |    N10    |
//    |N8|     |     |           |       |     |           |           |
//    |--|--|--|-----|-----------|       |-----|--|--|-----|-----------|
//       |  |  |     |                         |     |     |
//       |N1|N2|  N3 |                         |  N9 |  N3 |
//       |  |  |     |                         |     |     |
//       |-----|-----|                         |-----|-----|
//
// Block setup is:
// Elements C1, C2, and C3 are in Block 0
// Element N4 is in Block 1 which is aligned with Block 0
// Element N5 is in Block 2 which is rotated by 90 degrees counter clockwise
// Elments N6, N7, and N8 are in Block 3 which is anti-aligned with Block 0
// Elements N1, N2, and N3 are in Block 4 which is rotated 90 deg. clockwise
void test() {
  OrientationMap<3> aligned = OrientationMap<3>::create_aligned();
  OrientationMap<3> b1_orientation = OrientationMap<3>::create_aligned();
  OrientationMap<3> b2_orientation{std::array{Direction<3>::lower_eta(),
                                              Direction<3>::upper_xi(),
                                              Direction<3>::upper_zeta()}};
  OrientationMap<3> b3_orientation{std::array{Direction<3>::lower_xi(),
                                              Direction<3>::lower_eta(),
                                              Direction<3>::upper_zeta()}};
  OrientationMap<3> b4_orientation{std::array{Direction<3>::upper_eta(),
                                              Direction<3>::lower_xi(),
                                              Direction<3>::upper_zeta()}};

  SegmentId s_00{0, 0};
  SegmentId s_10{1, 0};
  SegmentId s_11{1, 1};
  SegmentId s_20{2, 0};
  SegmentId s_21{2, 1};
  SegmentId s_22{2, 2};
  SegmentId s_23{2, 3};

  const ElementId<3> parent_id{0, std::array{s_00, s_00, s_00}};

  const ElementId<3> child_1_id{0, std::array{s_10, s_10, s_00}};
  const ElementId<3> child_2_id{0, std::array{s_11, s_10, s_00}};
  const ElementId<3> child_3_id{0, std::array{s_00, s_11, s_00}};

  const ElementId<3> neighbor_1_id{4, std::array{s_10, s_20, s_00}};
  const ElementId<3> neighbor_2_id{4, std::array{s_10, s_21, s_00}};
  const ElementId<3> neighbor_3_id{4, std::array{s_10, s_11, s_00}};
  const ElementId<3> neighbor_9_id{4, std::array{s_10, s_10, s_00}};

  const ElementId<3> neighbor_4_id{1, std::array{s_00, s_00, s_00}};
  const ElementId<3> neighbor_10_id{1, std::array{s_00, s_10, s_00}};
  const ElementId<3> neighbor_11_id{1, std::array{s_00, s_11, s_00}};

  const ElementId<3> neighbor_5_id{2, std::array{s_10, s_00, s_00}};

  const ElementId<3> neighbor_6_id{3, std::array{s_10, s_10, s_00}};
  const ElementId<3> neighbor_7_id{3, std::array{s_20, s_22, s_00}};
  const ElementId<3> neighbor_8_id{3, std::array{s_20, s_23, s_00}};
  const ElementId<3> neighbor_12_id{3, std::array{s_10, s_11, s_00}};

  const Mesh<3> expected_parent_mesh{4, Spectral::Basis::Legendre,
                                     Spectral::Quadrature::GaussLobatto};

  DirectionMap<3, Neighbors<3>> child_1_neighbors{};
  child_1_neighbors.emplace(
      Direction<3>::lower_xi(),
      Neighbors<3>{std::unordered_set{neighbor_8_id, neighbor_7_id},
                   b3_orientation});
  child_1_neighbors.emplace(
      Direction<3>::upper_xi(),
      Neighbors<3>{std::unordered_set{child_2_id}, aligned});
  child_1_neighbors.emplace(
      Direction<3>::lower_eta(),
      Neighbors<3>{std::unordered_set{neighbor_1_id, neighbor_2_id},
                   b4_orientation});
  child_1_neighbors.emplace(
      Direction<3>::upper_eta(),
      Neighbors<3>{std::unordered_set{child_3_id}, aligned});
  Element<3> child_1{child_1_id, std::move(child_1_neighbors)};
  Mesh<3> child_1_mesh{std::array{3_st, 4_st, 3_st}, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  amr::Info<3> child_1_info{std::array{amr::Flag::Join, amr::Flag::Join,
                                       amr::Flag::IncreaseResolution},
                            expected_parent_mesh};

  DirectionMap<3, Neighbors<3>> child_2_neighbors{};
  child_2_neighbors.emplace(
      Direction<3>::lower_xi(),
      Neighbors<3>{std::unordered_set{child_1_id}, aligned});
  child_2_neighbors.emplace(
      Direction<3>::upper_xi(),
      Neighbors<3>{std::unordered_set{neighbor_4_id}, b1_orientation});
  child_2_neighbors.emplace(
      Direction<3>::lower_eta(),
      Neighbors<3>{std::unordered_set{neighbor_3_id}, b4_orientation});
  child_2_neighbors.emplace(
      Direction<3>::upper_eta(),
      Neighbors<3>{std::unordered_set{child_3_id}, aligned});
  Element<3> child_2{child_2_id, std::move(child_2_neighbors)};
  Mesh<3> child_2_mesh{std::array{4_st, 3_st, 4_st}, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  amr::Info<3> child_2_info{std::array{amr::Flag::Join, amr::Flag::Join,
                                       amr::Flag::DecreaseResolution},
                            expected_parent_mesh};

  DirectionMap<3, Neighbors<3>> child_3_neighbors{};
  child_3_neighbors.emplace(
      Direction<3>::lower_xi(),
      Neighbors<3>{std::unordered_set{neighbor_6_id}, b3_orientation});
  child_3_neighbors.emplace(
      Direction<3>::upper_xi(),
      Neighbors<3>{std::unordered_set{neighbor_4_id}, b1_orientation});
  child_3_neighbors.emplace(
      Direction<3>::lower_eta(),
      Neighbors<3>{std::unordered_set{child_2_id, child_1_id}, aligned});
  child_3_neighbors.emplace(
      Direction<3>::upper_eta(),
      Neighbors<3>{std::unordered_set{neighbor_5_id}, b2_orientation});
  Element<3> child_3{child_3_id, std::move(child_3_neighbors)};
  Mesh<3> child_3_mesh{std::array{4_st, 4_st, 3_st}, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  amr::Info<3> child_3_info{
      std::array{amr::Flag::DecreaseResolution, amr::Flag::Join,
                 amr::Flag::IncreaseResolution},
      expected_parent_mesh};

  Mesh<3> neighbor_mesh{std::array{5_st, 5_st, 5_st}, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};

  std::unordered_map<ElementId<3>, amr::Info<3>> child_1_neighbor_info{
      {neighbor_1_id,
       amr::Info<3>{std::array{amr::Flag::DoNothing, amr::Flag::Join,
                               amr::Flag::DoNothing},
                    neighbor_mesh}},
      {neighbor_2_id,
       amr::Info<3>{std::array{amr::Flag::DoNothing, amr::Flag::Join,
                               amr::Flag::DoNothing},
                    neighbor_mesh}},
      {child_2_id, child_2_info},
      {child_3_id, child_3_info},
      {neighbor_7_id, amr::Info<3>{std::array{amr::Flag::Join, amr::Flag::Join,
                                              amr::Flag::DoNothing},
                                   neighbor_mesh}},
      {neighbor_8_id, amr::Info<3>{std::array{amr::Flag::Join, amr::Flag::Join,
                                              amr::Flag::DoNothing},
                                   neighbor_mesh}}};

  std::unordered_map<ElementId<3>, amr::Info<3>> child_2_neighbor_info{
      {neighbor_3_id,
       amr::Info<3>{std::array{amr::Flag::DoNothing, amr::Flag::DoNothing,
                               amr::Flag::DoNothing},
                    neighbor_mesh}},
      {neighbor_4_id,
       amr::Info<3>{std::array{amr::Flag::DoNothing, amr::Flag::Split,
                               amr::Flag::DoNothing},
                    neighbor_mesh}},
      {child_2_id, child_2_info},
      {child_1_id, child_1_info}};

  std::unordered_map<ElementId<3>, amr::Info<3>> child_3_neighbor_info{
      {child_1_id, child_1_info},
      {child_2_id, child_2_info},
      {neighbor_4_id,
       amr::Info<3>{std::array{amr::Flag::DoNothing, amr::Flag::Split,
                               amr::Flag::DoNothing},
                    neighbor_mesh}},
      {neighbor_5_id,
       amr::Info<3>{std::array{amr::Flag::DoNothing, amr::Flag::DoNothing,
                               amr::Flag::DoNothing},
                    neighbor_mesh}},
      {neighbor_6_id,
       amr::Info<3>{std::array{amr::Flag::DoNothing, amr::Flag::DoNothing,
                               amr::Flag::DoNothing},
                    neighbor_mesh}}};

  using TaggedTupleType =
      tuples::TaggedTuple<Parallel::Tags::MetavariablesImpl<Metavariables>,
                          Parallel::Tags::ArrayIndexImpl<ElementId<3>>,
                          Parallel::Tags::GlobalCacheImpl<Metavariables>,
                          domain::Tags::Element<3>, domain::Tags::Mesh<3>,
                          domain::Tags::NeighborMesh<3>, amr::Tags::Info<3>,
                          amr::Tags::NeighborInfo<3>>;
  std::unordered_map<ElementId<3>, TaggedTupleType> children_items;
  DirectionalIdMap<3, Mesh<3>> unused_child_neighbor_mesh{};
  children_items.emplace(
      child_1_id,
      TaggedTupleType{Metavariables{}, child_1_id, nullptr, std::move(child_1),
                      std::move(child_1_mesh), unused_child_neighbor_mesh,
                      std::move(child_1_info),
                      std::move(child_1_neighbor_info)});
  children_items.emplace(
      child_2_id,
      TaggedTupleType{Metavariables{}, child_2_id, nullptr, std::move(child_2),
                      std::move(child_2_mesh), unused_child_neighbor_mesh,
                      std::move(child_2_info),
                      std::move(child_2_neighbor_info)});
  children_items.emplace(
      child_3_id,
      TaggedTupleType{Metavariables{}, child_3_id, nullptr, std::move(child_3),
                      std::move(child_3_mesh), unused_child_neighbor_mesh,
                      std::move(child_3_info),
                      std::move(child_3_neighbor_info)});

  DirectionMap<3, Neighbors<3>> expected_parent_neighbors{};
  expected_parent_neighbors.emplace(
      Direction<3>::lower_xi(),
      Neighbors<3>{std::unordered_set{neighbor_12_id, neighbor_6_id},
                   b3_orientation});
  expected_parent_neighbors.emplace(
      Direction<3>::upper_xi(),
      Neighbors<3>{std::unordered_set{neighbor_10_id, neighbor_11_id},
                   b1_orientation});
  expected_parent_neighbors.emplace(
      Direction<3>::lower_eta(),
      Neighbors<3>{std::unordered_set{neighbor_9_id, neighbor_3_id},
                   b4_orientation});
  expected_parent_neighbors.emplace(
      Direction<3>::upper_eta(),
      Neighbors<3>{std::unordered_set{neighbor_5_id}, b2_orientation});
  Element<3> expected_parent{parent_id, std::move(expected_parent_neighbors)};

  DirectionalIdMap<3, Mesh<3>> expected_parent_neighbor_mesh{};
  expected_parent_neighbor_mesh.emplace(
      DirectionalId<3>{Direction<3>::lower_xi(), neighbor_6_id}, neighbor_mesh);
  expected_parent_neighbor_mesh.emplace(
      DirectionalId<3>{Direction<3>::lower_xi(), neighbor_12_id},
      neighbor_mesh);
  expected_parent_neighbor_mesh.emplace(
      DirectionalId<3>{Direction<3>::upper_xi(), neighbor_10_id},
      neighbor_mesh);
  expected_parent_neighbor_mesh.emplace(
      DirectionalId<3>{Direction<3>::upper_xi(), neighbor_11_id},
      neighbor_mesh);
  expected_parent_neighbor_mesh.emplace(
      DirectionalId<3>{Direction<3>::lower_eta(), neighbor_3_id},
      neighbor_mesh);
  expected_parent_neighbor_mesh.emplace(
      DirectionalId<3>{Direction<3>::lower_eta(), neighbor_9_id},
      neighbor_mesh);
  expected_parent_neighbor_mesh.emplace(
      DirectionalId<3>{Direction<3>::upper_eta(), neighbor_5_id},
      neighbor_mesh);

  const amr::Info<3> expected_parent_info{
      std::array{amr::Flag::Undefined, amr::Flag::Undefined,
                 amr::Flag::Undefined},
      Mesh<3>{}};
  const std::unordered_map<ElementId<3>, amr::Info<3>>
      expected_parent_neighbor_info{};

  using array_component = Component<Metavariables>;
  using registrar = TestHelpers::amr::Registrar<Metavariables>;

  ActionTesting::MockRuntimeSystem<Metavariables> runner{{}};
  ActionTesting::emplace_group_component<registrar>(&runner);
  ActionTesting::emplace_component<array_component>(&runner, parent_id);
  CHECK(ActionTesting::get_databox_tag<registrar,
                                       TestHelpers::amr::RegisteredElements<3>>(
            runner, 0)
            .empty());
  ActionTesting::simple_action<array_component, amr::Actions::InitializeParent>(
      make_not_null(&runner), parent_id, children_items);
  CHECK(
      ActionTesting::get_databox_tag<array_component, domain::Tags::Element<3>>(
          runner, parent_id) == expected_parent);
  CHECK(ActionTesting::get_databox_tag<array_component, domain::Tags::Mesh<3>>(
            runner, parent_id) == expected_parent_mesh);
  CHECK(ActionTesting::get_databox_tag<array_component,
                                       domain::Tags::NeighborMesh<3>>(
            runner, parent_id) == expected_parent_neighbor_mesh);
  CHECK(ActionTesting::get_databox_tag<array_component, amr::Tags::Info<3>>(
            runner, parent_id) == expected_parent_info);
  CHECK(ActionTesting::get_databox_tag<array_component,
                                       amr::Tags::NeighborInfo<3>>(
            runner, parent_id) == expected_parent_neighbor_info);
  CHECK(ActionTesting::get_databox_tag<registrar,
                                       TestHelpers::amr::RegisteredElements<3>>(
            runner, 0)
            .empty());
  ActionTesting::invoke_queued_simple_action<registrar>(make_not_null(&runner),
                                                        0);
  CHECK(ActionTesting::get_databox_tag<registrar,
                                       TestHelpers::amr::RegisteredElements<3>>(
            runner, 0) == std::unordered_set{parent_id});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Amr.Actions.InitializeParent",
                  "[Unit][ParallelAlgorithms]") {
  test();
}
