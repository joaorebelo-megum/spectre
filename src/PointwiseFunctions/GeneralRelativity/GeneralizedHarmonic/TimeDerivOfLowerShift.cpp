// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/GaugeSource.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/TimeDerivOfShift.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/TimeDerivOfSpatialMetric.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace gh {
namespace {
template <typename DataType, size_t SpatialDim, typename Frame>
struct D0LowerShiftBuffer;

template <size_t SpatialDim, typename Frame>
struct D0LowerShiftBuffer<double, SpatialDim, Frame> {
  explicit D0LowerShiftBuffer(const size_t /*size*/) {}

  tnsr::I<double, SpatialDim, Frame> dt_shift{};
  tnsr::ii<double, SpatialDim, Frame> dt_spatial_metric{};
};

template <size_t SpatialDim, typename Frame>
struct D0LowerShiftBuffer<DataVector, SpatialDim, Frame> {
 private:
  // We make one giant allocation so that we don't thrash the heap.
  Variables<tmpl::list<::Tags::TempI<0, SpatialDim, Frame, DataVector>,
                       ::Tags::Tempii<1, SpatialDim, Frame, DataVector>>>
      buffer_;

 public:
  explicit D0LowerShiftBuffer(const size_t size)
      : buffer_(size),
        dt_shift(get<::Tags::TempI<0, SpatialDim, Frame, DataVector>>(buffer_)),
        dt_spatial_metric(
            get<::Tags::Tempii<1, SpatialDim, Frame, DataVector>>(buffer_)) {}

  tnsr::I<DataVector, SpatialDim, Frame>& dt_shift;
  tnsr::ii<DataVector, SpatialDim, Frame>& dt_spatial_metric;
};
}  // namespace

template <typename DataType, size_t SpatialDim, typename Frame>
void time_deriv_of_lower_shift(
    const gsl::not_null<tnsr::i<DataType, SpatialDim, Frame>*> dt_lower_shift,
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) {
  if (UNLIKELY(get_size(get<0>(*dt_lower_shift)) != get_size(get(lapse)))) {
    *dt_lower_shift =
        tnsr::i<DataType, SpatialDim, Frame>(get_size(get(lapse)));
  }
  // Use a Variables to reduce total number of allocations. This is especially
  // important in a multithreaded environment.
  D0LowerShiftBuffer<DataType, SpatialDim, Frame> buffer(get_size(get(lapse)));
  // get \partial_0 N^j
  const auto inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  gh::time_deriv_of_shift<DataType, SpatialDim, Frame>(
      make_not_null(&buffer.dt_shift), lapse, shift, inverse_spatial_metric,
      spacetime_unit_normal, phi, pi);
  gh::time_deriv_of_spatial_metric<DataType, SpatialDim, Frame>(
      make_not_null(&buffer.dt_spatial_metric), lapse, shift, phi, pi);
  for (size_t i = 0; i < SpatialDim; ++i) {
    dt_lower_shift->get(i) = spatial_metric.get(i, 0) * buffer.dt_shift.get(0) +
                             shift.get(0) * buffer.dt_spatial_metric.get(i, 0);
    for (size_t j = 0; j < SpatialDim; ++j) {
      if (j != 0) {
        dt_lower_shift->get(i) +=
            spatial_metric.get(i, j) * buffer.dt_shift.get(j) +
            shift.get(j) * buffer.dt_spatial_metric.get(i, j);
      }
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
tnsr::i<DataType, SpatialDim, Frame> time_deriv_of_lower_shift(
    const Scalar<DataType>& lapse,
    const tnsr::I<DataType, SpatialDim, Frame>& shift,
    const tnsr::ii<DataType, SpatialDim, Frame>& spatial_metric,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_unit_normal,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) {
  tnsr::i<DataType, SpatialDim, Frame> dt_lower_shift{};
  gh::time_deriv_of_lower_shift(make_not_null(&dt_lower_shift), lapse, shift,
                                spatial_metric, spacetime_unit_normal, phi, pi);
  return dt_lower_shift;
}
}  // namespace gh

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                               \
  template void gh::time_deriv_of_lower_shift(                             \
      const gsl::not_null<tnsr::i<DTYPE(data), DIM(data), FRAME(data)>*>   \
          dt_lower_shift,                                                  \
      const Scalar<DTYPE(data)>& lapse,                                    \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,           \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& spatial_metric, \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                  \
          spacetime_unit_normal,                                           \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,           \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi);            \
  template tnsr::i<DTYPE(data), DIM(data), FRAME(data)>                    \
  gh::time_deriv_of_lower_shift(                                           \
      const Scalar<DTYPE(data)>& lapse,                                    \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& shift,           \
      const tnsr::ii<DTYPE(data), DIM(data), FRAME(data)>& spatial_metric, \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                  \
          spacetime_unit_normal,                                           \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,           \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
