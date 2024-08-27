// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines metafunctions used by Tensor

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "DataStructures/Tensor/IndexType.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <typename X, typename Symm, typename IndexList>
class Tensor;
/// \endcond

/// \ingroup TensorGroup
/// Contains all metafunctions related to Tensor manipulations
namespace TensorMetafunctions {
namespace detail {
template <unsigned>
struct check_index_symmetry_impl;
// empty typelist or only had a vector to start with
template <>
struct check_index_symmetry_impl<0> {
  template <typename...>
  using f = std::true_type;
};

// found incorrect symmetric index
template <>
struct check_index_symmetry_impl<1> {
  template <typename...>
  using f = std::false_type;
};

// recurse the list
template <>
struct check_index_symmetry_impl<2> {
  template <typename Symm, typename IndexSymm, typename Index0,
            typename... IndexPack>
  using f = typename check_index_symmetry_impl<
      tmpl::has_key<IndexSymm, tmpl::front<Symm>>::value and
              not std::is_same<Index0,
                               tmpl::at<IndexSymm, tmpl::front<Symm>>>::value
          ? 1
          : tmpl::size<Symm>::value == 1 ? 0 : 2>::
      template f<tmpl::pop_front<Symm>,
                 tmpl::insert<IndexSymm, tmpl::pair<tmpl::front<Symm>, Index0>>,
                 IndexPack...>;
};
}  // namespace detail

/*!
 * \ingroup TensorGroup
 * \brief Check that each of symmetric indices is in the same frame and have the
 * same dimensionality.
 */
template <typename Symm, typename... IndexPack>
using check_index_symmetry = typename detail::check_index_symmetry_impl<
    tmpl::size<Symm>::value == 0 or tmpl::size<Symm>::value == 1 ? 0 : 2>::
    template f<Symm, tmpl::map<>, IndexPack...>;
template <typename Symm, typename... IndexPack>
constexpr bool check_index_symmetry_v =
    check_index_symmetry<Symm, IndexPack...>::value;

/*!
 * \ingroup TensorGroup
 * \brief Add a spatial index to the front of a Tensor
 *
 * \tparam TheTensor the tensor type to which the new index is prepended
 * \tparam VolumeDim the volume dimension of the tensor index to prepend
 * \tparam Fr the ::Frame of the tensor index to prepend
 */
template <typename TheTensor, std::size_t VolumeDim, UpLo Ul,
          typename Fr = Frame::Grid>
using prepend_spatial_index = ::Tensor<
    typename TheTensor::type,
    tmpl::push_front<
        typename TheTensor::symmetry,
        tmpl::int32_t<
            1 + tmpl::fold<typename TheTensor::symmetry, tmpl::int32_t<0>,
                           tmpl::max<tmpl::_state, tmpl::_element>>::value>>,
    tmpl::push_front<typename TheTensor::index_list,
                     SpatialIndex<VolumeDim, Ul, Fr>>>;

/*!
 * \ingroup TensorGroup
 * \brief Add a spacetime index to the front of a Tensor
 *
 * \tparam TheTensor the tensor type to which the new index is prepended
 * \tparam VolumeDim the volume dimension of the tensor index to prepend
 * \tparam Fr the ::Frame of the tensor index to prepend
 */
template <typename TheTensor, std::size_t VolumeDim, UpLo Ul,
          typename Fr = Frame::Grid>
using prepend_spacetime_index = ::Tensor<
    typename TheTensor::type,
    tmpl::push_front<
        typename TheTensor::symmetry,
        tmpl::int32_t<
            1 + tmpl::fold<typename TheTensor::symmetry, tmpl::int32_t<0>,
                           tmpl::max<tmpl::_state, tmpl::_element>>::value>>,
    tmpl::push_front<typename TheTensor::index_list,
                     SpacetimeIndex<VolumeDim, Ul, Fr>>>;

/// \ingroup TensorGroup
/// \brief remove the first index of a tensor
/// \tparam TheTensor the tensor type whose first index is removed
template <typename TheTensor>
using remove_first_index =
    ::Tensor<typename TheTensor::type,
             tmpl::pop_front<typename TheTensor::symmetry>,
             tmpl::pop_front<typename TheTensor::index_list>>;

/// \ingroup TensorGroup
/// \brief Swap the valences of all indices on a Tensor
template <typename TheTensor>
using change_all_valences =
    ::Tensor<typename TheTensor::type, typename TheTensor::symmetry,
             tmpl::transform<typename TheTensor::index_list,
                             tmpl::bind<change_index_up_lo, tmpl::_1>>>;

/// \ingroup TensorGroup
/// \brief Swap the data type of a tensor for a new type
/// \tparam NewType the new data type
/// \tparam TheTensor the tensor from which to keep symmetry and index
/// information
template <typename NewType, typename TheTensor>
using swap_type = ::Tensor<NewType, typename TheTensor::symmetry,
                           typename TheTensor::index_list>;

namespace detail {
template <typename T, typename Frame>
using frame_is_the_same = std::is_same<typename T::Frame, Frame>;
}  // namespace detail

/// \ingroup TensorGroup
/// \brief Return tmpl::true_type if any indices of the Tensor are in the
/// frame Frame.
template <typename TheTensor, typename Frame>
using any_index_in_frame =
    tmpl::any<typename TheTensor::index_list,
              tmpl::bind<detail::frame_is_the_same, tmpl::_1, Frame>>;

/// \ingroup TensorGroup
/// \brief Return true if any indices of the Tensor are in the
/// frame Frame.
template <typename TheTensor, typename Frame>
constexpr bool any_index_in_frame_v =
    any_index_in_frame<TheTensor, Frame>::value;

}  // namespace TensorMetafunctions
