// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines class Element.

#pragma once

#include <cstddef>
#include <iosfwd>
#include <unordered_set>

#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

/// \ingroup ComputationalDomainGroup
/// A spectral element with knowledge of its neighbors.
///
/// \tparam VolumeDim the volume dimension.
template <size_t VolumeDim>
class Element {
 public:
  using Neighbors_t = DirectionMap<VolumeDim, Neighbors<VolumeDim>>;

  /// Constructor
  ///
  /// \param id a unique identifier for the Element.
  /// \param neighbors info about the Elements that share an interface
  /// with this Element.
  Element(ElementId<VolumeDim> id, Neighbors_t neighbors);

  /// Default needed for serialization
  Element() = default;

  ~Element() = default;
  Element(const Element<VolumeDim>& /*rhs*/) = default;
  Element(Element<VolumeDim>&& /*rhs*/) = default;
  Element<VolumeDim>& operator=(const Element<VolumeDim>& /*rhs*/) = default;
  Element<VolumeDim>& operator=(Element<VolumeDim>&& /*rhs*/) = default;

  /// The directions of the faces of the Element that are external boundaries.
  const std::unordered_set<Direction<VolumeDim>>& external_boundaries() const {
    return external_boundaries_;
  }

  /// The directions of the faces of the Element that are internal boundaries.
  const std::unordered_set<Direction<VolumeDim>>& internal_boundaries() const {
    return internal_boundaries_;
  }

  /// A unique ID for the Element.
  const ElementId<VolumeDim>& id() const { return id_; }

  /// Information about the neighboring Elements.
  const Neighbors_t& neighbors() const { return neighbors_; }

  /// The number of neighbors this element has
  size_t number_of_neighbors() const { return number_of_neighbors_; }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

 private:
  ElementId<VolumeDim> id_{};
  Neighbors_t neighbors_{};
  size_t number_of_neighbors_{};
  std::unordered_set<Direction<VolumeDim>> external_boundaries_{};
  std::unordered_set<Direction<VolumeDim>> internal_boundaries_{};
};

template <size_t VolumeDim>
std::ostream& operator<<(std::ostream& os, const Element<VolumeDim>& element);

template <size_t VolumeDim>
bool operator==(const Element<VolumeDim>& lhs, const Element<VolumeDim>& rhs);

template <size_t VolumeDim>
bool operator!=(const Element<VolumeDim>& lhs, const Element<VolumeDim>& rhs);
