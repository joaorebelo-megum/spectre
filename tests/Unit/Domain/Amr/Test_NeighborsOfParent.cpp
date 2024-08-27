// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Domain/Amr/Flag.hpp"
#include "Domain/Amr/Helpers.hpp"
#include "Domain/Amr/Info.hpp"
#include "Domain/Amr/NeighborsOfParent.hpp"
#include "Domain/Amr/NewNeighborIds.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Structure/Side.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"

namespace {
const SegmentId s_00{0, 0};
const SegmentId s_10{1, 0};
const SegmentId s_11{1, 1};

void test_periodic_interval() {
  const OrientationMap<1> aligned = OrientationMap<1>::create_aligned();
  const ElementId<1> parent_id{0, std::array{s_00}};
  const ElementId<1> child_1_id{0, std::array{s_10}};
  const ElementId<1> child_2_id{0, std::array{s_11}};
  const Mesh<1> parent_mesh{4, Spectral::Basis::Legendre,
                            Spectral::Quadrature::GaussLobatto};
  const Mesh<1> child_1_mesh{3, Spectral::Basis::Legendre,
                             Spectral::Quadrature::GaussLobatto};
  const Mesh<1> child_2_mesh{4, Spectral::Basis::Legendre,
                             Spectral::Quadrature::GaussLobatto};

  DirectionMap<1, Neighbors<1>> child_1_neighbors{};
  child_1_neighbors.emplace(
      Direction<1>::lower_xi(),
      Neighbors<1>{std::unordered_set{child_2_id}, aligned});
  child_1_neighbors.emplace(
      Direction<1>::upper_xi(),
      Neighbors<1>{std::unordered_set{child_2_id}, aligned});
  const Element<1> child_1{child_1_id, std::move(child_1_neighbors)};
  const std::unordered_map<ElementId<1>, amr::Info<1>> child_1_neighbor_info{
      {child_2_id, {{{amr::Flag::Join}}, parent_mesh}}};

  DirectionMap<1, Neighbors<1>> child_2_neighbors{};
  child_2_neighbors.emplace(
      Direction<1>::lower_xi(),
      Neighbors<1>{std::unordered_set{child_1_id}, aligned});
  child_2_neighbors.emplace(
      Direction<1>::upper_xi(),
      Neighbors<1>{std::unordered_set{child_1_id}, aligned});
  const Element<1> child_2{child_2_id, std::move(child_2_neighbors)};
  const std::unordered_map<ElementId<1>, amr::Info<1>> child_2_neighbor_info{
      {child_1_id, {{{amr::Flag::Join}}, parent_mesh}}};

  std::vector<std::tuple<const Element<1>&,
                         const std::unordered_map<ElementId<1>, amr::Info<1>>&>>
      children_elements_and_neighbor_info;
  children_elements_and_neighbor_info.emplace_back(
      std::forward_as_tuple(child_1, child_1_neighbor_info));
  children_elements_and_neighbor_info.emplace_back(
      std::forward_as_tuple(child_2, child_2_neighbor_info));

  const auto [parent_neighbors, parent_neighbors_mesh] =
      amr::neighbors_of_parent(parent_id, children_elements_and_neighbor_info);
  DirectionMap<1, Neighbors<1>> expected_parent_neighbors{};
  expected_parent_neighbors.emplace(
      Direction<1>::lower_xi(),
      Neighbors<1>{std::unordered_set{parent_id}, aligned});
  expected_parent_neighbors.emplace(
      Direction<1>::upper_xi(),
      Neighbors<1>{std::unordered_set{parent_id}, aligned});
  CHECK(parent_neighbors == expected_parent_neighbors);
  DirectionalIdMap<1, Mesh<1>> expected_parent_neighbor_meshes{};
  expected_parent_neighbor_meshes.insert(
      {{Direction<1>::lower_xi(), parent_id}, parent_mesh});
  expected_parent_neighbor_meshes.insert(
      {{Direction<1>::upper_xi(), parent_id}, parent_mesh});
}

void test_interval() {
  const OrientationMap<1> aligned = OrientationMap<1>::create_aligned();
  const OrientationMap<1> flipped{std::array{Direction<1>::lower_xi()}};
  const ElementId<1> parent_id{0, std::array{s_00}};
  const ElementId<1> child_1_id{0, std::array{s_10}};
  const ElementId<1> child_2_id{0, std::array{s_11}};
  const ElementId<1> lower_neighbor_id{1, std::array{s_11}};
  const ElementId<1> upper_neighbor_id{2, std::array{s_00}};
  const Mesh<1> parent_mesh{4, Spectral::Basis::Legendre,
                            Spectral::Quadrature::GaussLobatto};
  const Mesh<1> child_1_mesh{3, Spectral::Basis::Legendre,
                             Spectral::Quadrature::GaussLobatto};
  const Mesh<1> child_2_mesh{4, Spectral::Basis::Legendre,
                             Spectral::Quadrature::GaussLobatto};
  const Mesh<1> lower_neighbor_mesh{5, Spectral::Basis::Legendre,
                                    Spectral::Quadrature::GaussLobatto};
  const Mesh<1> upper_neighbor_mesh{6, Spectral::Basis::Legendre,
                                    Spectral::Quadrature::GaussLobatto};

  DirectionMap<1, Neighbors<1>> child_1_neighbors{};
  child_1_neighbors.emplace(
      Direction<1>::lower_xi(),
      Neighbors<1>{std::unordered_set{lower_neighbor_id}, aligned});
  child_1_neighbors.emplace(
      Direction<1>::upper_xi(),
      Neighbors<1>{std::unordered_set{child_2_id}, aligned});
  const Element<1> child_1{child_1_id, std::move(child_1_neighbors)};
  const std::unordered_map<ElementId<1>, amr::Info<1>> child_1_neighbor_info{
      {lower_neighbor_id, {{{amr::Flag::DoNothing}}, lower_neighbor_mesh}},
      {child_2_id, {{{amr::Flag::Join}}, parent_mesh}}};

  DirectionMap<1, Neighbors<1>> child_2_neighbors{};
  child_2_neighbors.emplace(
      Direction<1>::lower_xi(),
      Neighbors<1>{std::unordered_set{child_1_id}, aligned});
  child_2_neighbors.emplace(
      Direction<1>::upper_xi(),
      Neighbors<1>{std::unordered_set{upper_neighbor_id}, flipped});
  const Element<1> child_2{child_2_id, std::move(child_2_neighbors)};
  const std::unordered_map<ElementId<1>, amr::Info<1>> child_2_neighbor_info{
      {child_1_id, {{{amr::Flag::Join}}, parent_mesh}},
      {upper_neighbor_id, {{{amr::Flag::Split}}, upper_neighbor_mesh}}};

  std::vector<std::tuple<const Element<1>&,
                         const std::unordered_map<ElementId<1>, amr::Info<1>>&>>
      children_elements_and_neighbor_info;
  children_elements_and_neighbor_info.emplace_back(
      std::forward_as_tuple(child_1, child_1_neighbor_info));
  children_elements_and_neighbor_info.emplace_back(
      std::forward_as_tuple(child_2, child_2_neighbor_info));

  const auto [parent_neighbors, parent_neighbors_mesh] =
      amr::neighbors_of_parent(parent_id, children_elements_and_neighbor_info);
  DirectionMap<1, Neighbors<1>> expected_parent_neighbors{};
  expected_parent_neighbors.emplace(
      Direction<1>::lower_xi(),
      Neighbors<1>{std::unordered_set{lower_neighbor_id}, aligned});
  const ElementId<1> split_upper_neighbor_id{2, std::array{s_11}};
  expected_parent_neighbors.emplace(
      Direction<1>::upper_xi(),
      Neighbors<1>{std::unordered_set{split_upper_neighbor_id}, flipped});
  CHECK(parent_neighbors == expected_parent_neighbors);
  DirectionalIdMap<1, Mesh<1>> expected_parent_neighbor_meshes{};
  expected_parent_neighbor_meshes.insert(
      {{Direction<1>::lower_xi(), parent_id}, lower_neighbor_mesh});
  expected_parent_neighbor_meshes.insert(
      {{Direction<1>::upper_xi(), parent_id}, upper_neighbor_mesh});
}

void test_rectangle() {
  const OrientationMap<2> aligned = OrientationMap<2>::create_aligned();
  const OrientationMap<2> rotated{
      std::array{Direction<2>::lower_eta(), Direction<2>::upper_xi()}};
  const Mesh<2> mesh;
  const ElementId<2> parent_id{0, std::array{s_00, s_00}};
  const ElementId<2> child_1_id{0, std::array{s_10, s_10}};
  const ElementId<2> child_2_id{0, std::array{s_11, s_10}};
  const ElementId<2> child_3_id{0, std::array{s_10, s_11}};
  const ElementId<2> child_4_id{0, std::array{s_11, s_11}};
  const ElementId<2> neighbor_1_id{1, std::array{s_11, s_00}};
  const ElementId<2> neighbor_2_id{2, std::array{s_10, s_00}};
  const ElementId<2> neighbor_3_id{2, std::array{s_11, s_11}};

  const std::array join_join{amr::Flag::Join, amr::Flag::Join};
  const std::array neighbor_1_flags{amr::Flag::Join, amr::Flag::DoNothing};
  const std::array neighbor_2_flags{amr::Flag::DoNothing, amr::Flag::Split};
  const std::array neighbor_3_flags{amr::Flag::DoNothing, amr::Flag::Join};

  const ElementId<2> neighbor_4_id{1, std::array{s_00, s_00}};
  const ElementId<2> neighbor_5_id{2, std::array{s_10, s_11}};
  const ElementId<2> neighbor_6_id{2, std::array{s_11, s_00}};

  DirectionMap<2, Neighbors<2>> child_1_neighbors{};
  child_1_neighbors.emplace(
      Direction<2>::lower_xi(),
      Neighbors<2>{std::unordered_set{neighbor_1_id}, aligned});
  child_1_neighbors.emplace(
      Direction<2>::upper_xi(),
      Neighbors<2>{std::unordered_set{child_2_id}, aligned});
  child_1_neighbors.emplace(
      Direction<2>::lower_eta(),
      Neighbors<2>{std::unordered_set{child_3_id}, aligned});
  child_1_neighbors.emplace(
      Direction<2>::upper_eta(),
      Neighbors<2>{std::unordered_set{child_3_id}, aligned});
  const Element<2> child_1{child_1_id, std::move(child_1_neighbors)};
  const std::unordered_map<ElementId<2>, amr::Info<2>> child_1_neighbor_info{
      {neighbor_1_id, amr::Info<2>{neighbor_1_flags, mesh}},
      {child_2_id, amr::Info<2>{join_join, mesh}},
      {child_3_id, amr::Info<2>{join_join, mesh}}};

  DirectionMap<2, Neighbors<2>> child_2_neighbors{};
  child_2_neighbors.emplace(
      Direction<2>::lower_xi(),
      Neighbors<2>{std::unordered_set{child_1_id}, aligned});
  child_2_neighbors.emplace(
      Direction<2>::upper_xi(),
      Neighbors<2>{std::unordered_set{neighbor_2_id}, rotated});
  child_2_neighbors.emplace(
      Direction<2>::lower_eta(),
      Neighbors<2>{std::unordered_set{child_4_id}, aligned});
  child_2_neighbors.emplace(
      Direction<2>::upper_eta(),
      Neighbors<2>{std::unordered_set{child_4_id}, aligned});
  const Element<2> child_2{child_2_id, std::move(child_2_neighbors)};
  const std::unordered_map<ElementId<2>, amr::Info<2>> child_2_neighbor_info{
      {child_1_id, amr::Info<2>{join_join, mesh}},
      {child_4_id, amr::Info<2>{join_join, mesh}},
      {neighbor_2_id, amr::Info<2>{neighbor_2_flags, mesh}}};

  DirectionMap<2, Neighbors<2>> child_3_neighbors{};
  child_3_neighbors.emplace(
      Direction<2>::lower_xi(),
      Neighbors<2>{std::unordered_set{neighbor_1_id}, aligned});
  child_3_neighbors.emplace(
      Direction<2>::upper_xi(),
      Neighbors<2>{std::unordered_set{child_4_id}, aligned});
  child_3_neighbors.emplace(
      Direction<2>::lower_eta(),
      Neighbors<2>{std::unordered_set{child_1_id}, aligned});
  child_3_neighbors.emplace(
      Direction<2>::upper_eta(),
      Neighbors<2>{std::unordered_set{child_1_id}, aligned});
  const Element<2> child_3{child_3_id, std::move(child_3_neighbors)};
  const std::unordered_map<ElementId<2>, amr::Info<2>> child_3_neighbor_info{
      {neighbor_1_id, amr::Info<2>{neighbor_1_flags, mesh}},
      {child_1_id, amr::Info<2>{join_join, mesh}},
      {child_4_id, amr::Info<2>{join_join, mesh}}};

  DirectionMap<2, Neighbors<2>> child_4_neighbors{};
  child_4_neighbors.emplace(
      Direction<2>::lower_xi(),
      Neighbors<2>{std::unordered_set{child_3_id}, aligned});
  child_4_neighbors.emplace(
      Direction<2>::upper_xi(),
      Neighbors<2>{std::unordered_set{neighbor_3_id}, rotated});
  child_4_neighbors.emplace(
      Direction<2>::lower_eta(),
      Neighbors<2>{std::unordered_set{child_2_id}, aligned});
  child_4_neighbors.emplace(
      Direction<2>::upper_eta(),
      Neighbors<2>{std::unordered_set{child_2_id}, aligned});
  const Element<2> child_4{child_4_id, std::move(child_4_neighbors)};
  const std::unordered_map<ElementId<2>, amr::Info<2>> child_4_neighbor_info{
      {child_2_id, amr::Info<2>{join_join, mesh}},
      {child_3_id, amr::Info<2>{join_join, mesh}},
      {neighbor_3_id, amr::Info<2>{neighbor_3_flags, mesh}}};

  std::vector<std::tuple<const Element<2>&,
                         const std::unordered_map<ElementId<2>, amr::Info<2>>&>>
      children_elements_and_neighbor_info;
  children_elements_and_neighbor_info.emplace_back(
      std::forward_as_tuple(child_1, child_1_neighbor_info));
  children_elements_and_neighbor_info.emplace_back(
      std::forward_as_tuple(child_2, child_2_neighbor_info));
  children_elements_and_neighbor_info.emplace_back(
      std::forward_as_tuple(child_3, child_3_neighbor_info));
  children_elements_and_neighbor_info.emplace_back(
      std::forward_as_tuple(child_4, child_4_neighbor_info));

  const auto [parent_neighbors, parent_neighbors_mesh] =
      amr::neighbors_of_parent(parent_id, children_elements_and_neighbor_info);
  DirectionMap<2, Neighbors<2>> expected_parent_neighbors{};
  expected_parent_neighbors.emplace(
      Direction<2>::lower_xi(),
      Neighbors<2>{std::unordered_set{neighbor_4_id}, aligned});
  expected_parent_neighbors.emplace(
      Direction<2>::upper_xi(),
      Neighbors<2>{std::unordered_set{neighbor_5_id, neighbor_6_id}, rotated});
  expected_parent_neighbors.emplace(
      Direction<2>::lower_eta(),
      Neighbors<2>{std::unordered_set{parent_id}, aligned});
  expected_parent_neighbors.emplace(
      Direction<2>::upper_eta(),
      Neighbors<2>{std::unordered_set{parent_id}, aligned});
  CHECK(parent_neighbors == expected_parent_neighbors);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Amr.NeighborsOfParent", "[Domain][Unit]") {
  test_periodic_interval();
  test_interval();
  test_rectangle();
}
