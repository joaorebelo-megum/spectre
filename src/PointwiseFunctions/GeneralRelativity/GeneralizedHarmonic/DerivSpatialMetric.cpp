// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/DerivSpatialMetric.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace gh {
template <typename DataType, size_t SpatialDim, typename Frame>
void deriv_spatial_metric(
    const gsl::not_null<tnsr::ijj<DataType, SpatialDim, Frame>*>
        d_spatial_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) {
  if (UNLIKELY(get_size(get<0, 0, 0>(*d_spatial_metric)) !=
               get_size(get<0, 0, 0>(phi)))) {
    *d_spatial_metric =
        tnsr::ijj<DataType, SpatialDim, Frame>(get_size(get<0, 0, 0>(phi)));
  }
  for (size_t k = 0; k < SpatialDim; ++k) {
    for (size_t i = 0; i < SpatialDim; ++i) {
      for (size_t j = i; j < SpatialDim; ++j) {
        d_spatial_metric->get(k, i, j) = phi.get(k, i + 1, j + 1);
      }
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::ijj<DataType, SpatialDim, Frame> deriv_spatial_metric(
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) {
  tnsr::ijj<DataType, SpatialDim, Frame> d_spatial_metric{};
  gh::deriv_spatial_metric<DataType, SpatialDim, Frame>(
      make_not_null(&d_spatial_metric), phi);
  return d_spatial_metric;
}
}  // namespace gh

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                               \
  template void gh::deriv_spatial_metric(                                  \
      const gsl::not_null<tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>*> \
          d_spatial_metric,                                                \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi);          \
  template tnsr::ijj<DTYPE(data), DIM(data), FRAME(data)>                  \
  gh::deriv_spatial_metric(                                                \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
