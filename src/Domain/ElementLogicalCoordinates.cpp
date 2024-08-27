// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/ElementLogicalCoordinates.hpp"

#include <array>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/IdPair.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/Structure/BlockId.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Side.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

template <size_t Dim>
std::optional<tnsr::I<double, Dim, Frame::ElementLogical>>
element_logical_coordinates(
    const tnsr::I<double, Dim, Frame::BlockLogical>& x_block_logical,
    const ElementId<Dim>& element_id) {
  tnsr::I<double, Dim, Frame::ElementLogical> x_element_logical{};
  for (size_t d = 0; d < Dim; ++d) {
    // Check if the point is outside the element
    const double up = element_id.segment_id(d).endpoint(Side::Upper);
    const double lo = element_id.segment_id(d).endpoint(Side::Lower);
    if (x_block_logical.get(d) < lo or x_block_logical.get(d) > up) {
      return std::nullopt;
    }
    // Map to element logical coords
    x_element_logical.get(d) =
        (2.0 * x_block_logical.get(d) - up - lo) / (up - lo);
  }
  return x_element_logical;
}

namespace {
// The segments bounds are binary fractions (i.e. the numerator is an
// integer and the denominator is a power of 2) so these floating point
// comparisons should be safe from roundoff problems
// Need to return true if on upper face of block
// Otherwise return true if point is within the segment or on the lower bound
bool segment_contains(const double x_block_logical,
                      const double lower_bound_block_logical,
                      const double upper_bound_block_logical) {
  if (UNLIKELY(x_block_logical == upper_bound_block_logical)) {
    return (upper_bound_block_logical == 1.0);
  }
  return (x_block_logical >= lower_bound_block_logical and
          x_block_logical < upper_bound_block_logical);
}
}  // namespace

template <size_t Dim>
std::unordered_map<ElementId<Dim>, ElementLogicalCoordHolder<Dim>>
element_logical_coordinates(
    const std::vector<ElementId<Dim>>& element_ids,
    const std::vector<BlockLogicalCoords<Dim>>& block_coord_holders) {
  // Temporarily put results here in data structures that allow
  // push_back, because we don't know the sizes of the output
  // DataVectors ahead of time.
  std::vector<std::array<std::vector<double>, Dim>> x_element_logical(
      element_ids.size());
  std::vector<std::vector<size_t>> offsets(element_ids.size());

  // Loop over points
  for (size_t offset = 0; offset < block_coord_holders.size(); ++offset) {
    // Skip points that are not in any block.
    if (not block_coord_holders[offset].has_value()) {
      continue;
    }

    const auto& block_id = block_coord_holders[offset].value().id;
    const auto& x_block_logical = block_coord_holders[offset].value().data;
    // Need to loop over elements, because the block doesn't know
    // things like the refinement_level of each element.
    for (size_t index = 0; index < element_ids.size(); ++index) {
      const auto& element_id = element_ids[index];
      if (element_id.block_id() == block_id.get_index()) {
        // This element is in this block; now check if the point is in
        // this element.
        const auto x_elem =
            element_logical_coordinates(x_block_logical, element_id);
        if (not x_elem.has_value()) {
          continue;
        }
        // Disambiguate points on shared element boundaries
        bool is_contained = true;
        for (size_t d = 0; d < Dim; ++d) {
          const double up = element_id.segment_id(d).endpoint(Side::Upper);
          const double lo = element_id.segment_id(d).endpoint(Side::Lower);
          const double x_block_log = x_block_logical.get(d);
          if (not segment_contains(x_block_log, lo, up)) {
            is_contained = false;
            break;
          }
        }
        if (is_contained) {
          for (size_t d = 0; d < Dim; ++d) {
            gsl::at(x_element_logical[index], d).push_back(x_elem->get(d));
          }
          offsets[index].push_back(offset);
          // Found a matching element, so we don't need to check other
          // elements.
          break;
        }
      }
    }
  }

  // Now we know how many points are in each element, so we can
  // put the intermediate results into the final data structure.
  std::unordered_map<ElementId<Dim>, ElementLogicalCoordHolder<Dim>> result;
  for (size_t index = 0; index < element_ids.size(); ++index) {
    const size_t num_grid_pts = x_element_logical[index][0].size();
    if (num_grid_pts > 0) {
      tnsr::I<DataVector, Dim, Frame::ElementLogical> tmp(num_grid_pts);
      std::vector<size_t> off(num_grid_pts);
      for (size_t s = 0; s < num_grid_pts; ++s) {
        for (size_t d = 0; d < Dim; ++d) {
          tmp.get(d)[s] = gsl::at(x_element_logical[index], d)[s];
        }
        off[s] = offsets[index][s];
      }
      result.emplace(element_ids[index], ElementLogicalCoordHolder<Dim>{
                                             std::move(tmp), std::move(off)});
    }
  }
  return result;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                  \
  template std::optional<tnsr::I<double, DIM(data), Frame::ElementLogical>>   \
  element_logical_coordinates(                                                \
      const tnsr::I<double, DIM(data), Frame::BlockLogical>& x_block_logical, \
      const ElementId<DIM(data)>& element_id);                                \
  template std::unordered_map<ElementId<DIM(data)>,                           \
                              ElementLogicalCoordHolder<DIM(data)>>           \
  element_logical_coordinates(                                                \
      const std::vector<ElementId<DIM(data)>>& element_ids,                   \
      const std::vector<BlockLogicalCoords<DIM(data)>>& block_coord_holders);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
