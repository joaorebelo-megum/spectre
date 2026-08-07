// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/Xcts/NumericBinaryWithWaves.hpp"

#include <cstddef>
#include <string>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/CommonVariables.tpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"

namespace Xcts::AnalyticData {

namespace detail {

template <typename DataType>
tnsr::I<DataType, 3> NumericBinaryWithWavesVariables<DataType>::killing_vector()
    const {
  auto result = make_with_value<tnsr::I<DataType, 3>>(get<0>(x), 0.);
  get<0>(result) = -angular_velocity * get<1>(x) + expansion * get<0>(x);
  get<1>(result) = angular_velocity * get<0>(x) + expansion * get<1>(x);
  get<2>(result) = expansion * get<2>(x);
  return result;
}

template <typename DataType>
tnsr::iJ<DataType, 3>
NumericBinaryWithWavesVariables<DataType>::deriv_killing_vector() const {
  auto result = make_with_value<tnsr::iJ<DataType, 3>>(get<0>(x), 0.);
  get<0, 0>(result) = expansion;
  get<1, 1>(result) = expansion;
  get<2, 2>(result) = expansion;
  get<1, 0>(result) = -angular_velocity;
  get<0, 1>(result) = angular_velocity;
  return result;
}

template <typename DataType>
void NumericBinaryWithWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3>*> spatial_metric,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::SpatialMetric<DataType, 3> /*meta*/) const {
  *spatial_metric = get<gr::Tags::SpatialMetric<DataType, 3>>(loaded_vars);
}

template <typename DataType>
void NumericBinaryWithWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::II<DataType, 3>*> inv_spatial_metric,
    const gsl::not_null<Cache*> cache,
    gr::Tags::InverseSpatialMetric<DataType, 3> /*meta*/) const {
  const auto& spatial_metric =
      cache->get_var(*this, gr::Tags::SpatialMetric<DataType, 3>{});
  *inv_spatial_metric = determinant_and_inverse(spatial_metric).second;
}

template <typename DataType>
void NumericBinaryWithWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> lapse,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::Lapse<DataType> /*meta*/) const {
  *lapse = get<gr::Tags::Lapse<DataType>>(loaded_vars);
}

template <typename DataType>
void NumericBinaryWithWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3>*> extrinsic_curvature,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::ExtrinsicCurvature<DataType, 3> /*meta*/) const {
  *extrinsic_curvature =
      get<gr::Tags::ExtrinsicCurvature<DataType, 3>>(loaded_vars);
}

template <typename DataType>
void NumericBinaryWithWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3>*> conformal_metric,
    const gsl::not_null<Cache*> cache,
    Tags::ConformalMetric<DataType, 3, Frame::Inertial> /*meta*/) const {
  // Unimodular conformal split: det(conformal metric) = 1, so
  // psi = det(gamma)^(1/12) and conformal metric = det(gamma)^(-1/3) gamma.
  const auto& spatial_metric =
      cache->get_var(*this, gr::Tags::SpatialMetric<DataType, 3>{});
  const auto det_spatial_metric = determinant_and_inverse(spatial_metric).first;
  const DataType conformal_scaling = pow(get(det_spatial_metric), -1. / 3.);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      conformal_metric->get(i, j) =
          conformal_scaling * spatial_metric.get(i, j);
    }
  }
  // The post-Newtonian wave content is added here once implemented:
  //   conformal_metric->get(i, j) += attenuation * h_TT.get(i, j);
}

template <typename DataType>
void NumericBinaryWithWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ijj<DataType, 3>*> deriv_conformal_metric,
    const gsl::not_null<Cache*> cache,
    ::Tags::deriv<Tags::ConformalMetric<DataType, 3, Frame::Inertial>,
                  tmpl::size_t<3>, Frame::Inertial> /*meta*/) const {
  if constexpr (std::is_same_v<DataType, DataVector>) {
    ASSERT(this->mesh.has_value() and this->inv_jacobian.has_value(),
           "Need a mesh and an inverse Jacobian for numeric differentiation.");
    const auto& conformal_metric = cache->get_var(
        *this, Tags::ConformalMetric<DataType, 3, Frame::Inertial>{});
    partial_derivative(deriv_conformal_metric, conformal_metric,
                       this->mesh->get(), this->inv_jacobian->get());
  } else {
    (void)deriv_conformal_metric;
    (void)cache;
    ERROR(
        "Numeric differentiation requires a grid, so it only works with "
        "DataVectors.");
  }
}

template <typename DataType>
void NumericBinaryWithWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3>*> dt_conformal_metric,
    const gsl::not_null<Cache*> cache,
    ::Tags::dt<Tags::ConformalMetric<DataType, 3, Frame::Inertial>> /*meta*/)
    const {
  // The previous solve is stationary in the frame corotating with the binary,
  // so in the inertial frame the conformal metric is Lie-dragged along the
  // helical Killing vector: dt(gbar) = -Lie_xi(gbar). No finite differencing.
  // Once the wave content is added, its analytic time derivative is added here.
  const auto& conformal_metric = cache->get_var(
      *this, Tags::ConformalMetric<DataType, 3, Frame::Inertial>{});
  const auto& deriv_conformal_metric = cache->get_var(
      *this, ::Tags::deriv<Tags::ConformalMetric<DataType, 3, Frame::Inertial>,
                           tmpl::size_t<3>, Frame::Inertial>{});
  const auto xi = killing_vector();
  const auto deriv_xi = deriv_killing_vector();
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      dt_conformal_metric->get(i, j) = make_with_value<DataType>(get<0>(x), 0.);
      for (size_t k = 0; k < 3; ++k) {
        dt_conformal_metric->get(i, j) -=
            xi.get(k) * deriv_conformal_metric.get(k, i, j) +
            conformal_metric.get(k, j) * deriv_xi.get(i, k) +
            conformal_metric.get(i, k) * deriv_xi.get(j, k);
      }
    }
  }
}

template <typename DataType>
void NumericBinaryWithWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> extrinsic_curvature_trace,
    const gsl::not_null<Cache*> cache,
    gr::Tags::TraceExtrinsicCurvature<DataType> /*meta*/) const {
  // Taken directly from the loaded extrinsic curvature, K = gamma^ij K_ij.
  // This involves no derivatives and is frame independent, so it is exactly
  // the trace of the previous solve.
  const auto& extrinsic_curvature =
      cache->get_var(*this, gr::Tags::ExtrinsicCurvature<DataType, 3>{});
  const auto& inv_spatial_metric =
      cache->get_var(*this, gr::Tags::InverseSpatialMetric<DataType, 3>{});
  get(*extrinsic_curvature_trace) = make_with_value<DataType>(get<0>(x), 0.);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      get(*extrinsic_curvature_trace) +=
          inv_spatial_metric.get(i, j) * extrinsic_curvature.get(i, j);
    }
  }
}

template <typename DataType>
void NumericBinaryWithWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> dt_extrinsic_curvature_trace,
    const gsl::not_null<Cache*> cache,
    ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataType>> /*meta*/) const {
  // K is a scalar, so the Lie drag reduces to dt(K) = -xi^k d_k K. The spatial
  // derivative is provided spectrally by `CommonVariables`.
  const auto& deriv_extrinsic_curvature_trace = cache->get_var(
      *this, ::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataType>,
                           tmpl::size_t<3>, Frame::Inertial>{});
  const auto xi = killing_vector();
  get(*dt_extrinsic_curvature_trace) = make_with_value<DataType>(get<0>(x), 0.);
  for (size_t k = 0; k < 3; ++k) {
    get(*dt_extrinsic_curvature_trace) -=
        xi.get(k) * deriv_extrinsic_curvature_trace.get(k);
  }
}

template <typename DataType>
void NumericBinaryWithWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*> shift_background,
    const gsl::not_null<Cache*> /*cache*/,
    Tags::ShiftBackground<DataType, 3, Frame::Inertial> /*meta*/) const {
  // Inertial frame: the whole shift is carried by the excess shift, which is
  // the asymptotically small `ShiftExcess` of the previous solve.
  std::fill(shift_background->begin(), shift_background->end(), 0.);
}

template <typename DataType>
void NumericBinaryWithWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::iJ<DataType, 3>*> deriv_shift_background,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::deriv<Tags::ShiftBackground<DataType, 3, Frame::Inertial>,
                  tmpl::size_t<3>, Frame::Inertial> /*meta*/) const {
  std::fill(deriv_shift_background->begin(), deriv_shift_background->end(), 0.);
}

template <typename DataType>
void NumericBinaryWithWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::II<DataType, 3, Frame::Inertial>*>
        longitudinal_shift_background,
    const gsl::not_null<Cache*> cache,
    Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
        DataType, 3, Frame::Inertial> /*meta*/) const {
  // With a vanishing background shift this is just -ubar^ij, where ubar is the
  // trace-free part of dt(conformal metric) with indices raised. Because
  // Lie_xi(gbar)_ij = Dbar_i xi_j + Dbar_j xi_i, this equals +(Lbar xi)^ij,
  // which is exactly what the corotating previous solve used as
  // (Lbar beta_background)^ij with ubar = 0.
  const auto& conformal_metric = cache->get_var(
      *this, Tags::ConformalMetric<DataType, 3, Frame::Inertial>{});
  const auto& inv_conformal_metric = cache->get_var(
      *this, Tags::InverseConformalMetric<DataType, 3, Frame::Inertial>{});
  const auto& dt_conformal_metric = cache->get_var(
      *this, ::Tags::dt<Tags::ConformalMetric<DataType, 3, Frame::Inertial>>{});

  auto trace_dt_conformal_metric = make_with_value<DataType>(get<0>(x), 0.);
  for (size_t k = 0; k < 3; ++k) {
    for (size_t l = 0; l < 3; ++l) {
      trace_dt_conformal_metric +=
          inv_conformal_metric.get(k, l) * dt_conformal_metric.get(k, l);
    }
  }

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      longitudinal_shift_background->get(i, j) =
          make_with_value<DataType>(get<0>(x), 0.);
      for (size_t k = 0; k < 3; ++k) {
        for (size_t l = 0; l < 3; ++l) {
          longitudinal_shift_background->get(i, j) -=
              inv_conformal_metric.get(i, k) * inv_conformal_metric.get(j, l) *
              (dt_conformal_metric.get(k, l) - (1. / 3.) *
                                                   trace_dt_conformal_metric *
                                                   conformal_metric.get(k, l));
        }
      }
    }
  }
}

template <typename DataType>
void NumericBinaryWithWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> conformal_energy_density,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::Conformal<gr::Tags::EnergyDensity<DataType>, 0> /*meta*/) const {
  get(*conformal_energy_density) = make_with_value<DataType>(get<0>(x), 0.);
}

template <typename DataType>
void NumericBinaryWithWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> conformal_stress_trace,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::Conformal<gr::Tags::StressTrace<DataType>, 0> /*meta*/) const {
  get(*conformal_stress_trace) = make_with_value<DataType>(get<0>(x), 0.);
}

template <typename DataType>
void NumericBinaryWithWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*> conformal_momentum_density,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::Conformal<gr::Tags::MomentumDensity<DataType, 3>, 0> /*meta*/)
    const {
  std::fill(conformal_momentum_density->begin(),
            conformal_momentum_density->end(), 0.);
}

template <typename DataType>
void NumericBinaryWithWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> conformal_factor_minus_one,
    const gsl::not_null<Cache*> cache,
    Tags::ConformalFactorMinusOne<DataType> /*meta*/) const {
  const auto& spatial_metric =
      cache->get_var(*this, gr::Tags::SpatialMetric<DataType, 3>{});
  const auto det_spatial_metric = determinant_and_inverse(spatial_metric).first;
  get(*conformal_factor_minus_one) =
      pow(get(det_spatial_metric), 1. / 12.) - 1.;
}

template <typename DataType>
void NumericBinaryWithWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*>
        lapse_times_conformal_factor_minus_one,
    const gsl::not_null<Cache*> cache,
    Tags::LapseTimesConformalFactorMinusOne<DataType> /*meta*/) const {
  const auto& lapse = cache->get_var(*this, gr::Tags::Lapse<DataType>{});
  const auto& conformal_factor_minus_one =
      cache->get_var(*this, Tags::ConformalFactorMinusOne<DataType>{});
  get(*lapse_times_conformal_factor_minus_one) =
      get(lapse) * (get(conformal_factor_minus_one) + 1.) - 1.;
}

template <typename DataType>
void NumericBinaryWithWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*> shift_excess,
    const gsl::not_null<Cache*> /*cache*/,
    Tags::ShiftExcess<DataType, 3, Frame::Inertial> /*meta*/) const {
  // The background shift vanishes, so the excess shift is the whole shift. It
  // is loaded as `ShiftExcess` from the previous solve, which is asymptotically
  // small; the full `Shift` grows like r and must not be used here.
  *shift_excess =
      get<Tags::ShiftExcess<DataType, 3, Frame::Inertial>>(loaded_vars);
}

template class NumericBinaryWithWavesVariables<DataVector>;

}  // namespace detail

NumericBinaryWithWaves::NumericBinaryWithWaves(
    std::string data_file, std::string subgroup, const int observation_step,
    const bool extrapolate_into_excisions, const double angular_velocity,
    const double expansion)
    : data_file_(std::move(data_file)),
      subgroup_(std::move(subgroup)),
      observation_step_(observation_step),
      extrapolate_into_excisions_(extrapolate_into_excisions),
      angular_velocity_(angular_velocity),
      expansion_(expansion) {
  load_interpolator();
}

NumericBinaryWithWaves::NumericBinaryWithWaves(
    const NumericBinaryWithWaves& rhs)
    : NumericBinaryWithWaves(rhs.data_file_, rhs.subgroup_,
                             rhs.observation_step_,
                             rhs.extrapolate_into_excisions_,
                             rhs.angular_velocity_, rhs.expansion_) {}

NumericBinaryWithWaves& NumericBinaryWithWaves::operator=(
    const NumericBinaryWithWaves& rhs) {
  if (this != &rhs) {
    *this = NumericBinaryWithWaves(rhs);
  }
  return *this;
}

void NumericBinaryWithWaves::load_interpolator() {
  // Reads the volume data into memory once. Constructing the interpolator opens
  // the H5 files and is not thread safe unless HDF5 was built with
  // thread-safety support (it is not, on the machines this runs on), so this
  // must not be called concurrently. It is safe here because the background is
  // a `const_global_cache_tag` and `Parallel::GlobalCache` is a Nodegroup, so
  // exactly one instance exists per node and it is unpacked on a single thread.
  // Interpolating from the loaded data afterwards *is* thread safe, so all
  // elements on the node share this one copy.
  if (data_file_.empty()) {
    return;
  }
  interpolator_ = spectre::Exporter::PointwiseInterpolator<3, Frame::Inertial>{
      data_file_, subgroup_,
      spectre::Exporter::ObservationStep{observation_step_},
      spectre::Exporter::get_tensor_components<
          detail::numeric_load_tags<DataVector>>()};
}

void NumericBinaryWithWaves::pup(PUP::er& p) {
  elliptic::analytic_data::Background::pup(p);
  elliptic::analytic_data::InitialGuess::pup(p);
  p | data_file_;
  p | subgroup_;
  p | observation_step_;
  p | extrapolate_into_excisions_;
  p | angular_velocity_;
  p | expansion_;
  // The loaded volume data is not serialized; it is reloaded here instead, so
  // that every node reads the file once rather than shipping it around.
  if (p.isUnpacking()) {
    load_interpolator();
  }
}

bool operator==(const NumericBinaryWithWaves& lhs,
                const NumericBinaryWithWaves& rhs) {
  return lhs.data_file() == rhs.data_file() and
         lhs.subgroup() == rhs.subgroup() and
         lhs.observation_step() == rhs.observation_step() and
         lhs.extrapolate_into_excisions() ==
             rhs.extrapolate_into_excisions() and
         lhs.angular_velocity() == rhs.angular_velocity() and
         lhs.expansion() == rhs.expansion();
}

bool operator!=(const NumericBinaryWithWaves& lhs,
                const NumericBinaryWithWaves& rhs) {
  return not(lhs == rhs);
}

PUP::able::PUP_ID NumericBinaryWithWaves::my_PUP_ID = 0;  // NOLINT

}  // namespace Xcts::AnalyticData

template class Xcts::AnalyticData::CommonVariables<
    DataVector, typename Xcts::AnalyticData::detail::
                    NumericBinaryWithWavesVariables<DataVector>::Cache>;
