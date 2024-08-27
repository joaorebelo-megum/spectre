// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Slice.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/SliceData.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/NormalCovectorAndMagnitude.hpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/Reconstructor.hpp"
#include "Evolution/Systems/ForceFree/FiniteDifference/Tags.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

namespace TestHelpers {

/*!
 * \brief Defines functions useful for testing subcell in ForceFree evolution
 * system
 */
namespace ForceFree::fd {

using GhostData = evolution::dg::subcell::GhostData;

template <typename F>
DirectionalIdMap<3, GhostData> compute_ghost_data(
    const Mesh<3>& subcell_mesh,
    const tnsr::I<DataVector, 3, Frame::ElementLogical>& volume_logical_coords,
    const DirectionMap<3, Neighbors<3>>& neighbors,
    const size_t ghost_zone_size, const F& compute_variables_of_neighbor_data) {
  DirectionalIdMap<3, GhostData> ghost_data{};

  for (const auto& [direction, neighbors_in_direction] : neighbors) {
    REQUIRE(neighbors_in_direction.size() == 1);
    const ElementId<3>& neighbor_id = *neighbors_in_direction.begin();
    auto neighbor_logical_coords = volume_logical_coords;
    neighbor_logical_coords.get(direction.dimension()) +=
        direction.sign() * 2.0;
    const auto neighbor_vars_for_reconstruction =
        compute_variables_of_neighbor_data(neighbor_logical_coords);

    DirectionMap<3, bool> directions_to_slice{};
    directions_to_slice[direction.opposite()] = true;
    const auto sliced_data = evolution::dg::subcell::detail::slice_data_impl(
        gsl::make_span(neighbor_vars_for_reconstruction.data(),
                       neighbor_vars_for_reconstruction.size()),
        subcell_mesh.extents(), ghost_zone_size,
        std::unordered_set{direction.opposite()}, 0, {});

    REQUIRE(sliced_data.size() == 1);
    REQUIRE(sliced_data.contains(direction.opposite()));

    ghost_data[DirectionalId<3>{direction, neighbor_id}] = GhostData{1};
    ghost_data.at(DirectionalId<3>{direction, neighbor_id})
        .neighbor_ghost_data_for_reconstruction() =
        sliced_data.at(direction.opposite());
  }
  return ghost_data;
}

template <typename Reconstructor>
void test_reconstructor(const size_t points_per_dimension,
                        const Reconstructor& derived_reconstructor) {
  // 1. create the variables to be reconstructed (evolved variables and current
  //    density TildeJ) being linear to coords
  // 2. send through reconstruction
  // 3. check if evolved variables were reconstructed correctly

  const ::ForceFree::fd::Reconstructor& reconstructor = derived_reconstructor;
  static_assert(tmpl::list_contains_v<
                typename ::ForceFree::fd::Reconstructor::creatable_classes,
                Reconstructor>);

  // create an element and its neighbor elements
  DirectionMap<3, Neighbors<3>> neighbors{};
  for (size_t i = 0; i < 2 * 3; ++i) {
    neighbors[gsl::at(Direction<3>::all_directions(), i)] = Neighbors<3>{
        {ElementId<3>{i + 1, {}}}, OrientationMap<3>::create_aligned()};
  }
  const Element<3> element{ElementId<3>{0, {}}, neighbors};

  using TildeE = ::ForceFree::Tags::TildeE;
  using TildeB = ::ForceFree::Tags::TildeB;
  using TildePsi = ::ForceFree::Tags::TildePsi;
  using TildePhi = ::ForceFree::Tags::TildePhi;
  using TildeQ = ::ForceFree::Tags::TildeQ;
  using TildeJ = ::ForceFree::Tags::TildeJ;

  using cons_tags = tmpl::list<TildeE, TildeB, TildePsi, TildePhi, TildeQ>;

  const Mesh<3> subcell_mesh{points_per_dimension,
                             Spectral::Basis::FiniteDifference,
                             Spectral::Quadrature::CellCentered};
  auto logical_coords = logical_coordinates(subcell_mesh);
  // Make the logical coordinates different in each direction
  for (size_t i = 1; i < 3; ++i) {
    logical_coords.get(i) += 4.0 * i;
  }

  // a simple, linear variables for testing purpose
  const auto compute_solution = [](const auto& coords) {
    Variables<::ForceFree::fd::tags_list_for_reconstruction> vars{
        get<0>(coords).size(), 0.0};
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        get<TildeE>(vars).get(j) += 1.0 * coords.get(i);
        get<TildeB>(vars).get(j) += 2.0 * coords.get(i);
        get<TildeJ>(vars).get(j) += 3.0 * coords.get(i);
      }
      get(get<TildePsi>(vars)) += 4.0 * coords.get(i);
      get(get<TildePhi>(vars)) += 5.0 * coords.get(i);
      get(get<TildeQ>(vars)) += 6.0 * coords.get(i);
    }
    get(get<TildePsi>(vars)) += 2.0;
    get(get<TildePhi>(vars)) += 3.0;
    get(get<TildeQ>(vars)) += 40.0;
    for (size_t j = 0; j < 3; ++j) {
      get<TildeE>(vars).get(j) += 1.0e-2 * (j + 1.0) + 10.0;
      get<TildeB>(vars).get(j) += 1.0e-2 * (j + 2.0) + 20.0;
      get<TildeJ>(vars).get(j) += 1.0e-2 * (j + 3.0) + 30.0;
    }
    return vars;
  };

  const size_t num_subcell_grid_pts = subcell_mesh.number_of_grid_points();

  Variables<::ForceFree::fd::tags_list_for_reconstruction>
      volume_vars_and_tilde_j{num_subcell_grid_pts};
  volume_vars_and_tilde_j.assign_subset(compute_solution(logical_coords));

  Variables<cons_tags> volume_vars =
      volume_vars_and_tilde_j.reference_subset<cons_tags>();
  const tnsr::I<DataVector, 3> volume_tilde_j =
      get<TildeJ>(volume_vars_and_tilde_j);

  // compute ghost data from neighbor
  const DirectionalIdMap<3, GhostData> ghost_data =
      compute_ghost_data(subcell_mesh, logical_coords, element.neighbors(),
                         reconstructor.ghost_zone_size(), compute_solution);

  // create Variables on lower and upper faces to perform reconstruction
  const size_t reconstructed_num_pts =
      (subcell_mesh.extents(0) + 1) *
      subcell_mesh.extents().slice_away(0).product();
  using recons_tags = ::ForceFree::fd::tags_list_for_reconstruction;

  std::array<Variables<recons_tags>, 3> vars_on_lower_face =
      make_array<3>(Variables<recons_tags>(reconstructed_num_pts));
  std::array<Variables<recons_tags>, 3> vars_on_upper_face =
      make_array<3>(Variables<recons_tags>(reconstructed_num_pts));

  // Now we have everything to call the reconstruction
  dynamic_cast<const Reconstructor&>(reconstructor)
      .reconstruct(make_not_null(&vars_on_lower_face),
                   make_not_null(&vars_on_upper_face), volume_vars,
                   volume_tilde_j, element, ghost_data, subcell_mesh);

  for (size_t dim = 0; dim < 3; ++dim) {
    CAPTURE(dim);

    // construct face-centered coordinates
    const auto basis = make_array<3>(Spectral::Basis::FiniteDifference);
    auto quadrature = make_array<3>(Spectral::Quadrature::CellCentered);
    auto extents = make_array<3>(points_per_dimension);
    gsl::at(extents, dim) = points_per_dimension + 1;
    gsl::at(quadrature, dim) = Spectral::Quadrature::FaceCentered;
    const Mesh<3> face_centered_mesh{extents, basis, quadrature};
    auto logical_coords_face_centered = logical_coordinates(face_centered_mesh);
    for (size_t i = 1; i < 3; ++i) {
      logical_coords_face_centered.get(i) =
          logical_coords_face_centered.get(i) + 4.0 * i;
    }

    // check reconstructed values for reconstruct() function
    Variables<recons_tags> expected_face_values{
        face_centered_mesh.number_of_grid_points()};
    expected_face_values.assign_subset(
        compute_solution(logical_coords_face_centered));

    tmpl::for_each<::ForceFree::fd::tags_list_for_reconstruction>(
        [dim, &expected_face_values, &vars_on_lower_face,
         &vars_on_upper_face](auto tag_to_check_v) {
          using tag_to_check = tmpl::type_from<decltype(tag_to_check_v)>;
          CAPTURE(db::tag_name<tag_to_check>());
          CHECK_ITERABLE_APPROX(
              get<tag_to_check>(gsl::at(vars_on_lower_face, dim)),
              get<tag_to_check>(expected_face_values));
          CHECK_ITERABLE_APPROX(
              get<tag_to_check>(gsl::at(vars_on_upper_face, dim)),
              get<tag_to_check>(expected_face_values));
        });

    // Test reconstruct_fd_neighbor
    const size_t num_pts_on_mortar =
        face_centered_mesh.slice_away(dim).number_of_grid_points();

    Variables<recons_tags> upper_side_vars_on_mortar{num_pts_on_mortar};

    dynamic_cast<const Reconstructor&>(reconstructor)
        .reconstruct_fd_neighbor(make_not_null(&upper_side_vars_on_mortar),
                                 volume_vars, volume_tilde_j, element,
                                 ghost_data, subcell_mesh,
                                 Direction<3>{dim, Side::Upper});

    Variables<recons_tags> lower_side_vars_on_mortar{num_pts_on_mortar};

    dynamic_cast<const Reconstructor&>(reconstructor)
        .reconstruct_fd_neighbor(make_not_null(&lower_side_vars_on_mortar),
                                 volume_vars, volume_tilde_j, element,
                                 ghost_data, subcell_mesh,
                                 Direction<3>{dim, Side::Lower});

    tmpl::for_each<cons_tags>(
        [dim, &expected_face_values, &lower_side_vars_on_mortar,
         &face_centered_mesh, &upper_side_vars_on_mortar](auto tag_to_check_v) {
          using tag_to_check = tmpl::type_from<decltype(tag_to_check_v)>;
          CAPTURE(db::tag_name<tag_to_check>());
          CHECK_ITERABLE_APPROX(
              get<tag_to_check>(lower_side_vars_on_mortar),
              data_on_slice(get<tag_to_check>(expected_face_values),
                            face_centered_mesh.extents(), dim, 0));
          CHECK_ITERABLE_APPROX(
              get<tag_to_check>(upper_side_vars_on_mortar),
              data_on_slice(get<tag_to_check>(expected_face_values),
                            face_centered_mesh.extents(), dim,
                            face_centered_mesh.extents(dim) - 1));
        });
  }
}

}  // namespace ForceFree::fd
}  // namespace TestHelpers
