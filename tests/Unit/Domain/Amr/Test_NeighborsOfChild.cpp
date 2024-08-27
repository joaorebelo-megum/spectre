// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "Domain/Amr/Flag.hpp"
#include "Domain/Amr/Helpers.hpp"
#include "Domain/Amr/Info.hpp"
#include "Domain/Amr/NeighborsOfChild.hpp"
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
#include "Helpers/Domain/Amr/NeighborFlagHelpers.hpp"
#include "Helpers/Domain/Structure/NeighborHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/CartesianProduct.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"

namespace {
template <size_t Dim>
std::vector<ElementId<Dim>> element_ids_to_test() {
  static constexpr size_t max_level = 2;
  std::vector<ElementId<Dim>> result{};
  for (size_t lx = 0; lx <= max_level; ++lx) {
    for (size_t ix = 0; ix < two_to_the(lx); ++ix) {
      if constexpr (Dim == 1) {
        result.emplace_back(0, std::array{SegmentId{lx, ix}});
      } else {
        for (size_t ly = 0; ly <= max_level; ++ly) {
          for (size_t iy = 0; iy < two_to_the(ly); ++iy) {
            if constexpr (Dim == 2) {
              result.emplace_back(
                  0, std::array{SegmentId{lx, ix}, SegmentId{ly, iy}});
            } else {
              for (size_t lz = 0; lz <= max_level; ++lz) {
                for (size_t iz = 0; iz < two_to_the(lz); ++iz) {
                  result.emplace_back(
                      0, std::array{SegmentId{lx, ix}, SegmentId{ly, iy},
                                    SegmentId{lz, iz}});
                }
              }
            }
          }
        }
      }
    }
  }
  return result;
}

template <size_t Dim>
std::vector<Element<Dim>> valid_elements(
    const gsl::not_null<std::mt19937*> generator,
    const ElementId<Dim>& element_id) {
  const SegmentId xi_segment = element_id.segment_id(0);
  const double lower_xi_endpoint = xi_segment.endpoint(Side::Lower);
  const auto valid_lower_xi_neighbors = TestHelpers::domain::valid_neighbors(
      generator, element_id, Direction<Dim>::lower_xi(),
      lower_xi_endpoint == -1.0 ? TestHelpers::domain::FaceType::Block
                                : TestHelpers::domain::FaceType::Internal);
  const double upper_xi_endpoint = xi_segment.endpoint(Side::Upper);
  const auto valid_upper_xi_neighbors = TestHelpers::domain::valid_neighbors(
      generator, element_id, Direction<Dim>::upper_xi(),
      upper_xi_endpoint == 1.0 ? TestHelpers::domain::FaceType::Block
                               : TestHelpers::domain::FaceType::Internal);
  std::vector<Element<Dim>> result{};
  for (const auto& [lower_xi_neighbors, upper_xi_neighbors] :
       cartesian_product(valid_lower_xi_neighbors, valid_upper_xi_neighbors)) {
    typename Element<Dim>::Neighbors_t neighbors{};
    neighbors.emplace(Direction<Dim>::lower_xi(), lower_xi_neighbors);
    neighbors.emplace(Direction<Dim>::upper_xi(), upper_xi_neighbors);
    result.emplace_back(element_id, std::move(neighbors));
  }
  return result;
}

template <size_t Dim>
Mesh<Dim> make_lgl_mesh(const std::array<size_t, Dim>& extents) {
  return {extents, Spectral::Basis::Legendre,
          Spectral::Quadrature::GaussLobatto};
}

template <size_t Dim>
std::vector<amr::Info<Dim>> valid_parent_info();

template <>
std::vector<amr::Info<1>> valid_parent_info<1>() {
  return std::vector{
      amr::Info<1>{std::array{amr::Flag::Split}, make_lgl_mesh<1>({{3}})}};
}

template <>
std::vector<amr::Info<2>> valid_parent_info<2>() {
  return std::vector{
      amr::Info<2>{std::array{amr::Flag::Split, amr::Flag::Split},
                   make_lgl_mesh<2>({{3, 3}})},
      amr::Info<2>{std::array{amr::Flag::IncreaseResolution, amr::Flag::Split},
                   make_lgl_mesh<2>({{4, 3}})},
      amr::Info<2>{std::array{amr::Flag::Split, amr::Flag::IncreaseResolution},
                   make_lgl_mesh<2>({{3, 4}})}};
}

template <>
std::vector<amr::Info<3>> valid_parent_info<3>() {
  return std::vector{
      amr::Info<3>{
          std::array{amr::Flag::Split, amr::Flag::Split, amr::Flag::Split},
          make_lgl_mesh<3>({{3, 3, 3}})},
      amr::Info<3>{std::array{amr::Flag::IncreaseResolution, amr::Flag::Split,
                              amr::Flag::Split},
                   make_lgl_mesh<3>({{4, 3, 3}})},
      amr::Info<3>{std::array{amr::Flag::Split, amr::Flag::IncreaseResolution,
                              amr::Flag::Split},
                   make_lgl_mesh<3>({{3, 4, 3}})},
      amr::Info<3>{std::array{amr::Flag::Split, amr::Flag::Split,
                              amr::Flag::IncreaseResolution},
                   make_lgl_mesh<3>({{3, 3, 4}})},
      amr::Info<3>{std::array{amr::Flag::Split, amr::Flag::IncreaseResolution,
                              amr::Flag::IncreaseResolution},
                   make_lgl_mesh<3>({{3, 4, 4}})},
      amr::Info<3>{std::array{amr::Flag::IncreaseResolution, amr::Flag::Split,
                              amr::Flag::IncreaseResolution},
                   make_lgl_mesh<3>({{4, 3, 4}})},
      amr::Info<3>{std::array{amr::Flag::IncreaseResolution,
                              amr::Flag::IncreaseResolution, amr::Flag::Split},
                   make_lgl_mesh<3>({{4, 4, 3}})}};
}

template <size_t Dim>
TestHelpers::amr::valid_info_t<Dim> valid_parent_neighbor_info(
    const Element<Dim>& element,
    const std::array<::amr::Flag, Dim>& element_flags) {
  TestHelpers::amr::valid_info_t<Dim> result{};
  const auto valid_lower_xi_neighbor_info =
      TestHelpers::amr::valid_neighbor_info(
          element.id(), element_flags,
          element.neighbors().at(Direction<Dim>::lower_xi()));
  const auto valid_upper_xi_neighbor_info =
      TestHelpers::amr::valid_neighbor_info(
          element.id(), element_flags,
          element.neighbors().at(Direction<Dim>::upper_xi()));
  for (const auto& lower_xi_neighbor_info : valid_lower_xi_neighbor_info) {
    for (const auto& upper_xi_neighbor_info : valid_upper_xi_neighbor_info) {
      auto joined_flags = lower_xi_neighbor_info;
      for (const auto& flags : upper_xi_neighbor_info) {
        joined_flags.emplace(flags);
      }
      result.emplace_back(joined_flags);
    }
  }
  return result;
}

template <size_t Dim>
void test(const gsl::not_null<std::mt19937*> generator) {
  CAPTURE(Dim);
  for (const auto& parent_id :
       random_sample(3, element_ids_to_test<Dim>(), generator)) {
    CAPTURE(parent_id);
    for (const auto& parent :
         random_sample(3, valid_elements(generator, parent_id), generator)) {
      CAPTURE(parent);
      for (const auto& parent_info : valid_parent_info<Dim>()) {
        CAPTURE(parent_info);
        for (const auto& parent_neighbor_info : random_sample(
                 3, valid_parent_neighbor_info(parent, parent_info.flags),
                 generator)) {
          CAPTURE(parent_neighbor_info);
          for (const auto& child_id :
               amr::ids_of_children(parent_id, parent_info.flags)) {
            CAPTURE(child_id);
            const auto [new_neighbors, new_neighbor_meshes] =
                amr::neighbors_of_child(parent, parent_info,
                                        parent_neighbor_info, child_id);
            for (const auto& direction : std::vector{
                     Direction<Dim>::lower_xi(), Direction<Dim>::upper_xi()}) {
              CAPTURE(direction);
              TestHelpers::domain::check_neighbors(new_neighbors.at(direction),
                                                   child_id, direction);
            }
          }
        }
      }
    }
  }
}
}  // namespace

// [[TimeOut, 30]]
SPECTRE_TEST_CASE("Unit.Domain.Amr.NeighborsOfChild", "[Domain][Unit]") {
  MAKE_GENERATOR(generator);
  test<1>(make_not_null(&generator));
  test<2>(make_not_null(&generator));
  test<3>(make_not_null(&generator));
}
