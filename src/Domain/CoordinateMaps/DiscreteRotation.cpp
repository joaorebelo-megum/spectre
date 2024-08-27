// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/DiscreteRotation.hpp"

#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/Side.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace domain::CoordinateMaps {
template <size_t VolumeDim>
DiscreteRotation<VolumeDim>::DiscreteRotation(
    OrientationMap<VolumeDim> orientation)
    : orientation_(std::move(orientation)),
      is_identity_(orientation_.is_aligned()) {
  if constexpr (VolumeDim > 1) {
    // We allow reversing the direction of the axes in 1d to make testing
    // non-aligned blocks possible in 1d, which is easier than testing them in
    // 2d and 3d.
    ASSERT(get(determinant(discrete_rotation_jacobian(orientation_))) > 0.0,
           "Discrete rotations must be done in such a manner that the sign of "
           "the determinant of the discrete rotation is positive. This is to "
           "preserve handedness of the coordinates.");
  }
}

template <size_t VolumeDim>
template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, VolumeDim>
DiscreteRotation<VolumeDim>::operator()(
    const std::array<T, VolumeDim>& source_coords) const {
  return discrete_rotation(orientation_, source_coords);
}

template <size_t VolumeDim>
std::optional<std::array<double, VolumeDim>>
DiscreteRotation<VolumeDim>::inverse(
    const std::array<double, VolumeDim>& target_coords) const {
  return discrete_rotation(orientation_.inverse_map(), target_coords);
}

template <size_t VolumeDim>
template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, VolumeDim, Frame::NoFrame>
DiscreteRotation<VolumeDim>::jacobian(
    const std::array<T, VolumeDim>& source_coords) const {
  auto jacobian_matrix = make_with_value<
      tnsr::Ij<tt::remove_cvref_wrap_t<T>, VolumeDim, Frame::NoFrame>>(
      dereference_wrapper(source_coords[0]), 0.0);
  for (size_t d = 0; d < VolumeDim; d++) {
    const auto new_direction =
        orientation_(Direction<VolumeDim>(d, Side::Upper));
    jacobian_matrix.get(d, orientation_(d)) =
        new_direction.side() == Side::Upper ? 1.0 : -1.0;
  }
  return jacobian_matrix;
}

template <size_t VolumeDim>
template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, VolumeDim, Frame::NoFrame>
DiscreteRotation<VolumeDim>::inv_jacobian(
    const std::array<T, VolumeDim>& source_coords) const {
  auto inv_jacobian_matrix = make_with_value<
      tnsr::Ij<tt::remove_cvref_wrap_t<T>, VolumeDim, Frame::NoFrame>>(
      dereference_wrapper(source_coords[0]), 0.0);
  for (size_t d = 0; d < VolumeDim; d++) {
    const auto new_direction =
        orientation_(Direction<VolumeDim>(d, Side::Upper));
    inv_jacobian_matrix.get(orientation_(d), d) =
        new_direction.side() == Side::Upper ? 1.0 : -1.0;
  }
  return inv_jacobian_matrix;
}

template <size_t VolumeDim>
void DiscreteRotation<VolumeDim>::pup(PUP::er& p) {
  size_t version = 0;
  p | version;
  // Remember to increment the version number when making changes to this
  // function. Retain support for unpacking data written by previous versions
  // whenever possible. See `Domain` docs for details.
  if (version >= 0) {
    p | orientation_;
    p | is_identity_;
  }
}

template class DiscreteRotation<1>;
template class DiscreteRotation<2>;
template class DiscreteRotation<3>;

// Explicit instantiations
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                           \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, DIM(data)> \
  DiscreteRotation<DIM(data)>::operator()(                             \
      const std::array<DTYPE(data), DIM(data)>& source_coords) const;  \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, DIM(data),   \
                    Frame::NoFrame>                                    \
  DiscreteRotation<DIM(data)>::jacobian(                               \
      const std::array<DTYPE(data), DIM(data)>& source_coords) const;  \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, DIM(data),   \
                    Frame::NoFrame>                                    \
  DiscreteRotation<DIM(data)>::inv_jacobian(                           \
      const std::array<DTYPE(data), DIM(data)>& source_coords) const;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3),
                        (double, DataVector,
                         std::reference_wrapper<const double>,
                         std::reference_wrapper<const DataVector>))

#undef DTYPE
#undef INSTANTIATE
}  // namespace domain::CoordinateMaps
