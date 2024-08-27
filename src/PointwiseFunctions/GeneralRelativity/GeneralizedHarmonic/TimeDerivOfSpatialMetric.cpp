// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/TimeDerivOfSpatialMetric.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace gh {
template <typename DataType, size_t SpatialDim, typename Frame>
void time_deriv_of_spatial_metric(
    const gsl::not_null<tnsr::ii<DataType, SpatialDim, Frame>*>
        dt_spatial_metric,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) {
  if (UNLIKELY(get_size(get<0, 0>(*dt_spatial_metric)) !=
               get_size(get(lapse)))) {
    *dt_spatial_metric =
        tnsr::ii<DataType, SpatialDim, Frame>(get_size(get(lapse)));
  }
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t j = i; j < SpatialDim; ++j) {
      dt_spatial_metric->get(i, j) = -get(lapse) * pi.get(i + 1, j + 1);
      for (size_t k = 0; k < SpatialDim; ++k) {
        dt_spatial_metric->get(i, j) += shift.get(k) * phi.get(k, i + 1, j + 1);
      }
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::ii<DataType, SpatialDim, Frame> time_deriv_of_spatial_metric(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) {
  tnsr::ii<DataType, SpatialDim, Frame> dt_spatial_metric{};
  gh::time_deriv_of_spatial_metric(make_not_null(&dt_spatial_metric), lapse,
                                   shift, phi, pi);
  return dt_spatial_metric;
}
}  // namespace gh

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                              \
  template void gh::time_deriv_of_spatial_metric(                         \
      const gsl::not_null<tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>*> \
          dt_spatial_metric,                                              \
      const Scalar<DTYPE(data)>& lapse,                                   \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,          \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,          \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi);           \
  template tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>                  \
  gh::time_deriv_of_spatial_metric(                                       \
      const Scalar<DTYPE(data)>& lapse,                                   \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,          \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,          \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
