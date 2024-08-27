// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

/// \cond
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl

/// \endcond

/// \ingroup SpecialRelativityGroup
/// Holds functions related to special relativity.
namespace sr {
/// @{
/*!
 * \ingroup SpecialRelativityGroup
 * \brief Computes the matrix for a Lorentz boost from a single
 * velocity vector (i.e., not a velocity field).
 *
 * \details Given a spatial velocity vector \f$v^i\f$ (with \f$c=1\f$),
 * compute the matrix \f$\Lambda^{a}{}_{\bar{a}}\f$ for a Lorentz boost with
 * that velocity [e.g. Eq. (2.38) of \cite ThorneBlandford2017]:
 *
 * \f{align}{
 * \Lambda^t{}_{\bar{t}} &= \gamma, \\
 * \Lambda^t{}_{\bar{i}} = \Lambda^i{}_{\bar{t}} &= \gamma v^i, \\
 * \Lambda^i{}_{\bar{j}} = \Lambda^j{}_{\bar{i}} &= [(\gamma - 1)/v^2] v^i v^j
 *                                              + \delta^{ij}.
 * \f}
 *
 * Here \f$v = \sqrt{\delta_{ij} v^i v^j}\f$, \f$\gamma = 1/\sqrt{1-v^2}\f$,
 * and \f$\delta^{ij}\f$ is the Kronecker delta. Note that this matrix boosts
 * a one-form from the unbarred to the barred frame, and its inverse
 * (obtained via \f$v \rightarrow -v\f$) boosts a vector from the barred to
 * the unbarred frame.
 *
 * Note that while the Lorentz boost matrix is symmetric, the returned
 * boost matrix is of type `tnsr::Ab`, because `Tensor` does not support
 * symmetric tensors unless both indices have the same valence.
 */
template <size_t SpatialDim>
tnsr::Ab<double, SpatialDim, Frame::NoFrame> lorentz_boost_matrix(
    const tnsr::I<double, SpatialDim, Frame::NoFrame>& velocity);

template <size_t SpatialDim>
void lorentz_boost_matrix(
    gsl::not_null<tnsr::Ab<double, SpatialDim, Frame::NoFrame>*> boost_matrix,
    const tnsr::I<double, SpatialDim, Frame::NoFrame>& velocity);

template <size_t SpatialDim>
tnsr::Ab<double, SpatialDim, Frame::NoFrame> lorentz_boost_matrix(
    const std::array<double, SpatialDim>& velocity);
/// @}

/// @{
/*!
 * \ingroup SpecialRelativityGroup
 * \brief Apply a Lorentz boost to the spatial part of a vector.
 * \details This requires passing the 0th component of the vector as an
 * additional argument.
 */
template <typename DataType, size_t SpatialDim, typename Frame>
void lorentz_boost(gsl::not_null<tnsr::I<DataType, SpatialDim, Frame>*> result,
                   const tnsr::I<DataType, SpatialDim, Frame>& vector,
                   double vector_component_0,
                   const std::array<double, SpatialDim>& velocity);

/*!
 * \brief Apply a Lorentz boost to a one form.
 */
template <typename DataType, size_t SpatialDim, typename Frame>
void lorentz_boost(
    gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> result,
    const tnsr::a<DataType, SpatialDim, Frame>& one_form,
    const std::array<double, SpatialDim>& velocity);

/*!
 * \brief Apply a Lorentz boost to each component of a rank-2 tensor with
 * lower or covariant indices.
 * \note In the future we might want to write a single function capable to boost
 * a tensor of arbitrary rank.
 */
template <typename DataType, size_t SpatialDim, typename Frame>
void lorentz_boost(gsl::not_null<tnsr::ab<DataType, SpatialDim, Frame>*> result,
                   const tnsr::ab<DataType, SpatialDim, Frame>& tensor,
                   const std::array<double, SpatialDim>& velocity);
/// @}
}  // namespace sr
