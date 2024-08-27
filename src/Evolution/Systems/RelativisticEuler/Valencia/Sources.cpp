// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/RelativisticEuler/Valencia/Sources.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace {
template <size_t Dim>
void densitized_stress(
    const gsl::not_null<tnsr::II<DataVector, Dim, Frame::Inertial>*> result,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& tilde_s_vector,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& spatial_velocity,
    const tnsr::II<DataVector, Dim, Frame::Inertial>& inv_spatial_metric,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const Scalar<DataVector>& pressure) {
  *result = inv_spatial_metric;
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = i; j < Dim; ++j) {
      result->get(i, j) *= get(sqrt_det_spatial_metric) * get(pressure);
      result->get(i, j) +=
          0.5 * (tilde_s_vector.get(i) * spatial_velocity.get(j) +
                 tilde_s_vector.get(j) * spatial_velocity.get(i));
    }
  }
}
}  // namespace

namespace RelativisticEuler::Valencia {
namespace detail {
template <size_t Dim>
void sources_impl(
    const gsl::not_null<Scalar<DataVector>*> source_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        source_tilde_s,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> tilde_s_M,
    const gsl::not_null<tnsr::II<DataVector, Dim, Frame::Inertial>*> tilde_s_MN,
    const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_tau,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& tilde_s,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& pressure, const Scalar<DataVector>& lapse,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& d_lapse,
    const tnsr::iJ<DataVector, Dim, Frame::Inertial>& d_shift,
    const tnsr::ijj<DataVector, Dim, Frame::Inertial>& d_spatial_metric,
    const tnsr::II<DataVector, Dim, Frame::Inertial>& inv_spatial_metric,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::ii<DataVector, Dim, Frame::Inertial>& extrinsic_curvature) {
  raise_or_lower_index(tilde_s_M, tilde_s, inv_spatial_metric);
  densitized_stress(tilde_s_MN, *tilde_s_M, spatial_velocity,
                    inv_spatial_metric, sqrt_det_spatial_metric, pressure);

  // unroll contributions from m=0 and n=0 to avoid initializing
  // source_tilde_tau to zero
  get(*source_tilde_tau) =
      get(lapse) * get<0, 0>(extrinsic_curvature) * get<0, 0>(*tilde_s_MN) -
      get<0>(*tilde_s_M) * get<0>(d_lapse);
  for (size_t m = 1; m < Dim; ++m) {
    get(*source_tilde_tau) +=
        get(lapse) * (extrinsic_curvature.get(m, 0) * tilde_s_MN->get(m, 0) +
                      extrinsic_curvature.get(0, m) * tilde_s_MN->get(0, m)) -
        tilde_s_M->get(m) * d_lapse.get(m);
    for (size_t n = 1; n < Dim; ++n) {
      get(*source_tilde_tau) +=
          get(lapse) * extrinsic_curvature.get(m, n) * tilde_s_MN->get(m, n);
    }
  }

  for (size_t i = 0; i < Dim; ++i) {
    source_tilde_s->get(i) = -(get(tilde_d) + get(tilde_tau)) * d_lapse.get(i);
    for (size_t m = 0; m < Dim; ++m) {
      source_tilde_s->get(i) += tilde_s.get(m) * d_shift.get(i, m);
      for (size_t n = 0; n < Dim; ++n) {
        source_tilde_s->get(i) += 0.5 * get(lapse) *
                                  d_spatial_metric.get(i, m, n) *
                                  tilde_s_MN->get(m, n);
      }
    }
  }
}
}  // namespace detail

template <size_t Dim>
void ComputeSources<Dim>::apply(
    const gsl::not_null<Scalar<DataVector>*> source_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        source_tilde_s,
    const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_tau,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& tilde_s,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& pressure, const Scalar<DataVector>& lapse,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& d_lapse,
    const tnsr::iJ<DataVector, Dim, Frame::Inertial>& d_shift,
    const tnsr::ijj<DataVector, Dim, Frame::Inertial>& d_spatial_metric,
    const tnsr::II<DataVector, Dim, Frame::Inertial>& inv_spatial_metric,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::ii<DataVector, Dim, Frame::Inertial>& extrinsic_curvature) {
  tnsr::I<DataVector, Dim, Frame::Inertial> tilde_s_M{
      get(sqrt_det_spatial_metric).size()};
  tnsr::II<DataVector, Dim, Frame::Inertial> tilde_s_MN{
      get(sqrt_det_spatial_metric).size()};

  detail::sources_impl(
      source_tilde_tau, source_tilde_s, make_not_null(&tilde_s_M),
      make_not_null(&tilde_s_MN), tilde_d, tilde_tau, tilde_s, spatial_velocity,
      pressure, lapse, d_lapse, d_shift, d_spatial_metric, inv_spatial_metric,
      sqrt_det_spatial_metric, extrinsic_curvature);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data)                                                 \
  template class ComputeSources<DIM(data)>;                                    \
  template void detail::sources_impl(                                          \
      const gsl::not_null<Scalar<DataVector>*> source_tilde_tau,               \
      const gsl::not_null<tnsr::i<DataVector, DIM(data), Frame::Inertial>*>    \
          source_tilde_s,                                                      \
      const gsl::not_null<tnsr::I<DataVector, DIM(data), Frame::Inertial>*>    \
          tilde_s_M,                                                           \
      const gsl::not_null<tnsr::II<DataVector, DIM(data), Frame::Inertial>*>   \
          tilde_s_MN,                                                          \
      const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_tau,  \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& tilde_s,          \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& spatial_velocity, \
      const Scalar<DataVector>& pressure, const Scalar<DataVector>& lapse,     \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>& d_lapse,          \
      const tnsr::iJ<DataVector, DIM(data), Frame::Inertial>& d_shift,         \
      const tnsr::ijj<DataVector, DIM(data), Frame::Inertial>&                 \
          d_spatial_metric,                                                    \
      const tnsr::II<DataVector, DIM(data), Frame::Inertial>&                  \
          inv_spatial_metric,                                                  \
      const Scalar<DataVector>& sqrt_det_spatial_metric,                       \
      const tnsr::ii<DataVector, DIM(data), Frame::Inertial>&                  \
          extrinsic_curvature);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM

}  // namespace RelativisticEuler::Valencia
