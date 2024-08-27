// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace domain {
namespace Tags {
template <size_t Dim, typename Frame>
struct Coordinates;
}  // namespace Tags
}  // namespace domain
class DataVector;
template <typename X, typename Symm, typename IndexList>
class Tensor;
/// \endcond

namespace gh {
/// @{
/*!
 * \ingroup GeneralRelativityGroup
 * \brief Computes time derivative of the spatial metric.
 *
 * \details Let the generalized harmonic conjugate momentum and spatial
 * derivative variables be \f$\Pi_{ab} = -n^c \partial_c g_{ab} \f$ and
 * \f$\Phi_{iab} = \partial_i g_{ab} \f$. As \f$ n_i \equiv 0 \f$. The time
 * derivative of the spatial metric is given by the time derivative of the
 * spatial sector of the spacetime metric, i.e.
 * \f$ \partial_0 \gamma_{ij} = \partial_0 g_{ij} \f$.
 *
 * To compute the latter, we use the evolution equation for \f$ g_{ij} \f$,
 * c.f. eq.(35) of \cite Lindblom2005qh (with \f$\gamma_1 = -1\f$):
 *
 * \f[
 * \partial_0 g_{ab} = - \alpha \Pi_{ab} + \beta^k \Phi_{kab}
 * \f]
 */
template <typename DataType, size_t SpatialDim, typename Frame>
void time_deriv_of_spatial_metric(
    gsl::not_null<tnsr::ii<DataType, SpatialDim, Frame>*> dt_spatial_metric,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi);

template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::ii<DataType, SpatialDim, Frame> time_deriv_of_spatial_metric(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi);
/// @}

namespace Tags {
/*!
 * \brief Compute item to get time derivative of the spatial metric from
 *        generalized harmonic and geometric variables
 *
 * \details See `time_deriv_of_spatial_metric()`. Can be retrieved using
 * `gr::Tags::SpatialMetric` wrapped in `Tags::dt`.
 */
template <size_t SpatialDim, typename Frame>
struct TimeDerivSpatialMetricCompute
    : ::Tags::dt<gr::Tags::SpatialMetric<DataVector, SpatialDim, Frame>>,
      db::ComputeTag {
  using argument_tags =
      tmpl::list<gr::Tags::Lapse<DataVector>,
                 gr::Tags::Shift<DataVector, SpatialDim, Frame>,
                 Phi<DataVector, SpatialDim, Frame>,
                 Pi<DataVector, SpatialDim, Frame>>;

  using return_type = tnsr::ii<DataVector, SpatialDim, Frame>;

  static constexpr auto function = static_cast<void (*)(
      gsl::not_null<tnsr::ii<DataVector, SpatialDim, Frame>*>,
      const Scalar<DataVector>&, const tnsr::I<DataVector, SpatialDim, Frame>&,
      const tnsr::iaa<DataVector, SpatialDim, Frame>&,
      const tnsr::aa<DataVector, SpatialDim, Frame>&)>(
      &time_deriv_of_spatial_metric<DataVector, SpatialDim, Frame>);

  using base =
      ::Tags::dt<gr::Tags::SpatialMetric<DataVector, SpatialDim, Frame>>;
};
}  // namespace Tags
}  // namespace gh
