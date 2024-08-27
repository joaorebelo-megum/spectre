// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/Initialization/Mortars.hpp"

#include <array>
#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include "Domain/Structure/CreateInitialMesh.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace evolution::dg::Initialization::detail {
namespace {
template <typename MappedType, size_t Dim>
using MortarMap = DirectionalIdMap<Dim, MappedType>;
}  // namespace

template <size_t Dim>
std::tuple<DirectionalIdMap<Dim, evolution::dg::MortarDataHolder<Dim>>,
           DirectionalIdMap<Dim, Mesh<Dim - 1>>,
           DirectionalIdMap<Dim, std::array<Spectral::MortarSize, Dim - 1>>,
           DirectionalIdMap<Dim, TimeStepId>,
           DirectionMap<Dim, std::optional<Variables<tmpl::list<
                                 evolution::dg::Tags::MagnitudeOfNormal,
                                 evolution::dg::Tags::NormalCovector<Dim>>>>>>
mortars_apply_impl(const std::vector<std::array<size_t, Dim>>& initial_extents,
                   const Spectral::Quadrature quadrature,
                   const Element<Dim>& element,
                   const TimeStepId& next_temporal_id,
                   const Mesh<Dim>& volume_mesh) {
  MortarMap<evolution::dg::MortarDataHolder<Dim>, Dim> mortar_data{};
  MortarMap<Mesh<Dim - 1>, Dim> mortar_meshes{};
  MortarMap<std::array<Spectral::MortarSize, Dim - 1>, Dim> mortar_sizes{};
  MortarMap<TimeStepId, Dim> mortar_next_temporal_ids{};
  DirectionMap<Dim, std::optional<Variables<
                        tmpl::list<evolution::dg::Tags::MagnitudeOfNormal,
                                   evolution::dg::Tags::NormalCovector<Dim>>>>>
      normal_covector_quantities{};
  for (const auto& [direction, neighbors] : element.neighbors()) {
    normal_covector_quantities[direction] = std::nullopt;
    for (const auto& neighbor : neighbors) {
      const DirectionalId<Dim> mortar_id{direction, neighbor};
      mortar_data.emplace(mortar_id, MortarDataHolder<Dim>{});
      mortar_meshes.emplace(
          mortar_id,
          ::dg::mortar_mesh(volume_mesh.slice_away(direction.dimension()),
                            ::domain::Initialization::create_initial_mesh(
                                initial_extents, neighbor, quadrature,
                                neighbors.orientation())
                                .slice_away(direction.dimension())));
      mortar_sizes.emplace(
          mortar_id,
          ::dg::mortar_size(element.id(), neighbor, direction.dimension(),
                            neighbors.orientation()));
      // Since no communication needs to happen for boundary conditions
      // the temporal id is not advanced on the boundary, so we only need to
      // initialize it on internal boundaries
      mortar_next_temporal_ids.insert({mortar_id, next_temporal_id});
    }
  }

  for (const auto& direction : element.external_boundaries()) {
    normal_covector_quantities[direction] = std::nullopt;
  }

  return {std::move(mortar_data), std::move(mortar_meshes),
          std::move(mortar_sizes), std::move(mortar_next_temporal_ids),
          std::move(normal_covector_quantities)};
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                                 \
  template std::tuple<                                                         \
      DirectionalIdMap<DIM(data), evolution::dg::MortarDataHolder<DIM(data)>>, \
      DirectionalIdMap<DIM(data), Mesh<DIM(data) - 1>>,                        \
      DirectionalIdMap<DIM(data),                                              \
                       std::array<Spectral::MortarSize, DIM(data) - 1>>,       \
      DirectionalIdMap<DIM(data), TimeStepId>,                                 \
      DirectionMap<DIM(data),                                                  \
                   std::optional<Variables<tmpl::list<                         \
                       evolution::dg::Tags::MagnitudeOfNormal,                 \
                       evolution::dg::Tags::NormalCovector<DIM(data)>>>>>>     \
  mortars_apply_impl(                                                          \
      const std::vector<std::array<size_t, DIM(data)>>& initial_extents,       \
      const Spectral::Quadrature quadrature,                                   \
      const Element<DIM(data)>& element, const TimeStepId& next_temporal_id,   \
      const Mesh<DIM(data)>& volume_mesh);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace evolution::dg::Initialization::detail
