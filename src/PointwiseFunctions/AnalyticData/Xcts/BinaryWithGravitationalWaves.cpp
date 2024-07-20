// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/Xcts/BinaryWithGravitationalWaves.hpp"

#include <boost/math/interpolators/cubic_hermite.hpp>
#include <cstddef>

#include "DataStructures/BoostMultiArray.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/ExtractPoint.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/EagerMath/Trace.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/Integration/GslQuadAdaptive.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/CommonVariables.tpp"
#include "PointwiseFunctions/Elasticity/Strain.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/SpecialRelativity/LorentzBoostMatrix.hpp"
#include "PointwiseFunctions/Xcts/LongitudinalOperator.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

// Boost MultiArray is used internally in odeint, so odeint must be included
// later
#include <boost/numeric/odeint.hpp>

namespace Xcts::AnalyticData {

namespace detail {

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> distance_left,
    const gsl::not_null<Cache*> /*cache*/,
    detail::Tags::DistanceLeft<DataType> /*meta*/) const {
  DataType present_time(get_size(get<0>(x)), max_time_interpolator);
  const auto distance_left_aux = get_t_distance_left(present_time);
  *distance_left = distance_left_aux;
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> distance_right,
    const gsl::not_null<Cache*> /*cache*/,
    detail::Tags::DistanceRight<DataType> /*meta*/) const {
  DataType present_time(get_size(get<0>(x)), max_time_interpolator);
  const auto distance_right_aux = get_t_distance_right(present_time);
  *distance_right = distance_right_aux;
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::i<DataType, 3>*> deriv_one_over_distance_left,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::deriv<detail::Tags::OneOverDistanceLeft<DataType>,
                  tmpl::size_t<Dim>, Frame::Inertial> /*meta*/) const {
  DataType present_time(get_size(get<0>(x)), max_time_interpolator);
  const auto deriv_one_over_distance_left_aux =
      get_t_deriv_one_over_distance_left(present_time);
  for (size_t i = 0; i < 3; ++i) {
    deriv_one_over_distance_left->get(i) =
        deriv_one_over_distance_left_aux.get(i);
  }
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::i<DataType, 3>*> deriv_one_over_distance_right,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::deriv<detail::Tags::OneOverDistanceRight<DataType>,
                  tmpl::size_t<Dim>, Frame::Inertial> /*meta*/) const {
  DataType present_time(get_size(get<0>(x)), max_time_interpolator);
  const auto deriv_one_over_distance_right_aux =
      get_t_deriv_one_over_distance_right(present_time);
  for (size_t i = 0; i < 3; ++i) {
    deriv_one_over_distance_right->get(i) =
        deriv_one_over_distance_right_aux.get(i);
  }
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    gsl::not_null<tnsr::ijk<DataType, 3>*> deriv_3_distance_left,
    gsl::not_null<Cache*> /*cache*/,
    ::Tags::deriv<
        ::Tags::deriv<::Tags::deriv<detail::Tags::DistanceLeft<DataType>,
                                    tmpl::size_t<Dim>, Frame::Inertial>,
                      tmpl::size_t<Dim>, Frame::Inertial>,
        tmpl::size_t<Dim>, Frame::Inertial> /*meta*/) const {
  DataType present_time(get_size(get<0>(x)), max_time_interpolator);
  const auto deriv_3_distance_left_aux =
      get_t_deriv_3_distance_left(present_time);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        deriv_3_distance_left->get(i, j, k) =
            deriv_3_distance_left_aux.get(i, j, k);
      }
    }
  }
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    gsl::not_null<tnsr::ijk<DataType, 3>*> deriv_3_distance_right,
    gsl::not_null<Cache*> /*cache*/,
    ::Tags::deriv<
        ::Tags::deriv<::Tags::deriv<detail::Tags::DistanceRight<DataType>,
                                    tmpl::size_t<Dim>, Frame::Inertial>,
                      tmpl::size_t<Dim>, Frame::Inertial>,
        tmpl::size_t<Dim>, Frame::Inertial> /*meta*/) const {
  DataType present_time(get_size(get<0>(x)), max_time_interpolator);
  const auto deriv_3_distance_right_aux =
      get_t_deriv_3_distance_right(present_time);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        deriv_3_distance_right->get(i, j, k) =
            deriv_3_distance_right_aux.get(i, j, k);
      }
    }
  }
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*> normal_left,
    const gsl::not_null<Cache*> /*cache*/,
    detail::Tags::NormalLeft<DataType> /*meta*/) const {
  DataType present_time(get_size(get<0>(x)), max_time_interpolator);
  const auto normal_left_aux = get_t_normal_left(present_time);
  for (size_t i = 0; i < 3; ++i) {
    normal_left->get(i) = normal_left_aux.get(i);
  }
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*> normal_right,
    const gsl::not_null<Cache*> /*cache*/,
    detail::Tags::NormalRight<DataType> /*meta*/) const {
  DataType present_time(get_size(get<0>(x)), max_time_interpolator);
  const auto normal_right_aux = get_t_normal_right(present_time);
  for (size_t i = 0; i < 3; ++i) {
    normal_right->get(i) = normal_right_aux.get(i);
  }
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3>*> radiative_term,
    const gsl::not_null<Cache*> /*cache*/,
    detail::Tags::RadiativeTerm<DataType> /*meta*/) const {
  DataType present_time(get_size(get<0>(x)), max_time_interpolator);
  const auto radiative_term_aux = get_t_radiative_term(present_time);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      radiative_term->get(i, j) = radiative_term_aux.get(i, j);
    }
  }
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3>*> near_zone_term,
    const gsl::not_null<Cache*> /*cache*/,
    detail::Tags::NearZoneTerm<DataType> /*meta*/) const {
  DataType present_time(get_size(get<0>(x)), max_time_interpolator);
  const auto near_zone_term_aux = get_t_near_zone_term(present_time);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      near_zone_term->get(i, j) = near_zone_term_aux.get(i, j);
    }
  }
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3>*> present_term,
    const gsl::not_null<Cache*> /*cache*/,
    detail::Tags::PresentTerm<DataType> /*meta*/) const {
  DataType present_time(get_size(get<0>(x)), max_time_interpolator);
  const auto present_term_aux = get_t_present_term(present_time);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      present_term->get(i, j) = present_term_aux.get(i, j);
    }
  }
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3>*> past_term,
    const gsl::not_null<Cache*> /*cache*/,
    detail::Tags::PastTerm<DataType> /*meta*/) const {
  DataType present_time(get_size(get<0>(x)), max_time_interpolator);
  const auto past_term_aux = get_t_past_term(present_time);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      past_term->get(i, j) = past_term_aux.get(i, j);
    }
  }
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3>*> integral_term,
    const gsl::not_null<Cache*> /*cache*/,
    detail::Tags::IntegralTerm<DataType> /*meta*/) const {
  DataType present_time(get_size(get<0>(x)), max_time_interpolator);
  const auto integral_term_aux = get_t_integral_term(present_time);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      integral_term->get(i, j) = integral_term_aux.get(i, j);
    }
  }
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3>*> pn_conjugate_momentum3,
    const gsl::not_null<Cache*> /*cache*/,
    detail::Tags::PostNewtonianConjugateMomentum3<DataType> /*meta*/) const {
  DataType present_time(get_size(get<0>(x)), max_time_interpolator);
  const auto pn_conjugate_momentum3_aux =
      get_t_pn_conjugate_momentum3(present_time);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      pn_conjugate_momentum3->get(i, j) = pn_conjugate_momentum3_aux.get(i, j);
    }
  }
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3>*> pn_extrinsic_curvature,
    const gsl::not_null<Cache*> /*cache*/,
    detail::Tags::PostNewtonianExtrinsicCurvature<DataType> /*meta*/) const {
  DataType present_time(get_size(get<0>(x)), max_time_interpolator);
  const auto pn_extrinsic_curvature_aux =
      get_t_pn_extrinsic_curvature(present_time);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      pn_extrinsic_curvature->get(i, j) = pn_extrinsic_curvature_aux.get(i, j);
    }
  }
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> retarded_time_left,
    const gsl::not_null<Cache*> /*cache*/,
    detail::Tags::RetardedTimeLeft<DataType> /*meta*/) const {
  DataType present_time(get_size(x.get(0)), max_time_interpolator);
  get(*retarded_time_left) = find_retarded_time_left(present_time);
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> retarded_time_right,
    const gsl::not_null<Cache*> /*cache*/,
    detail::Tags::RetardedTimeRight<DataType> /*meta*/) const {
  DataType present_time(get_size(x.get(0)), max_time_interpolator);
  get(*retarded_time_right) = find_retarded_time_right(present_time);
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> rootfinder_bracket_time_lower,
    const gsl::not_null<Cache*> /*cache*/,
    detail::Tags::RootFinderBracketTimeLower<DataType> /*meta*/) const {
  get(*rootfinder_bracket_time_lower) = past_time.front();
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> rootfinder_bracket_time_upper,
    const gsl::not_null<Cache*> /*cache*/,
    detail::Tags::RootFinderBracketTimeUpper<DataType> /*meta*/) const {
  get(*rootfinder_bracket_time_upper) = past_time.back();
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3>*> conformal_metric,
    const gsl::not_null<Cache*> /*cache*/,
    Xcts::Tags::ConformalMetric<DataType, 3, Frame::Inertial> /*meta*/) const {
  DataType present_time(get_size(get<0>(x)), max_time_interpolator);
  const auto& conformal_metric_aux = get_t_conformal_metric(present_time);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      conformal_metric->get(i, j) = conformal_metric_aux.get(i, j);
    }
  }
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ijj<DataType, 3>*> deriv_conformal_metric,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::deriv<Xcts::Tags::ConformalMetric<DataType, 3, Frame::Inertial>,
                  tmpl::size_t<Dim>, Frame::Inertial> /*meta*/) const {
  std::fill(deriv_conformal_metric->begin(), deriv_conformal_metric->end(), 0.);
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> trace_extrinsic_curvature,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::TraceExtrinsicCurvature<DataType> /*meta*/) const {
  DataType present_time(get_size(get<0>(x)), max_time_interpolator);
  ASSERT(mesh.has_value() and inv_jacobian.has_value(),
         "Need a mesh and a Jacobian for numeric differentiation.");
  if constexpr (std::is_same_v<DataType, DataVector>) {
    get(*trace_extrinsic_curvature) =
        get(get_t_trace_extrinsic_curvature(present_time));
  } else {
    (void)trace_extrinsic_curvature;
    ERROR(
        "Numeric differentiation only works with DataVectors because it needs "
        "a grid.");
  }
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> dt_trace_extrinsic_curvature,
    const gsl::not_null<Cache*> cache,
    ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataType>> /*meta*/) const {
  const auto& trace_extrinsic_curvature =
      cache->get_var(*this, gr::Tags::TraceExtrinsicCurvature<DataType>{});
  double time_displacement = 0.1;
  DataType time_back(get(trace_extrinsic_curvature).size(), -time_displacement);
  /*
  DataType present_time(get_size(get<0>(x)), max_time_interpolator);
  const auto distance_left = get_t_distance_left(present_time);
  const auto distance_right = get_t_distance_right(present_time);
  const auto areal_distance_left = find_areal_distance_left(get(distance_left));
  const auto areal_distance_right =
      find_areal_distance_right(get(distance_right));
  const auto normal_left = get_t_normal_left(present_time);
  const auto normal_right = get_t_normal_right(present_time);
  const auto momentum_left = get_t_momentum_left(present_time);
  const auto momentum_right = get_t_momentum_right(present_time);
  */
  Scalar<DataType> trace_extrinsic_curvature_back;
  ASSERT(mesh.has_value() and inv_jacobian.has_value(),
         "Need a mesh and a Jacobian for numeric differentiation.");
  if constexpr (std::is_same_v<DataType, DataVector>) {
    trace_extrinsic_curvature_back = get_t_trace_extrinsic_curvature(time_back);
  } else {
    (void)dt_trace_extrinsic_curvature;
    (void)cache;
    ERROR(
        "Numeric differentiation only works with DataVectors because it needs "
        "a grid.");
  }

  get(*dt_trace_extrinsic_curvature) =
      /*square(
          1. +
          .75 * sqrt(3) * square(mass_left) * normal_left.get(1) *
              (momentum_left.get(1) / mass_left) /
              ((1. - 2. * mass_left / areal_distance_left) *
               sqrt(square(areal_distance_left) * square(areal_distance_left) -
                    2. * mass_left * cube(areal_distance_left) + 1.6875)) +
          .75 * sqrt(3) * square(mass_right) * normal_right.get(1) *
              (momentum_right.get(1) / mass_right) /
              ((1. - 2. * mass_right / areal_distance_right) *
               sqrt(square(areal_distance_right) *
                        square(areal_distance_right) -
                    2. * mass_right * cube(areal_distance_right) + 1.6875))) *
       */
      (get(trace_extrinsic_curvature) - get(trace_extrinsic_curvature_back)) /
      time_displacement;
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*> shift_background,
    const gsl::not_null<Cache*> /*cache*/,
    Xcts::Tags::ShiftBackground<DataType, 3, Frame::Inertial> /*meta*/) const {
  std::fill(shift_background->begin(), shift_background->end(), 0.);
  // PN shift
  /*
  DataType present_time(get_size(get<0>(x)), max_time_interpolator);
  const auto distance_left_t = get_t_distance_left(present_time);
  const auto distance_right_t = get_t_distance_right(present_time);
  const auto separation_t = get_t_separation(present_time);
  const auto momentum_left_t = get_t_momentum_left(present_time);
  const auto momentum_right_t = get_t_momentum_right(present_time);
  const auto normal_left_t = get_t_normal_left(present_time);
  const auto normal_right_t = get_t_normal_right(present_time);
  for (size_t i = 0; i < 3; ++i) {
    shift_background->get(i) -=
        4. * (momentum_left_t.get(i) / get(distance_left_t) +
              momentum_right_t.get(i) / get(distance_right_t));
    for (size_t j = 0; j < 3; ++j) {
      shift_background->get(i) +=
          .5 * momentum_left_t.get(j) *
              (-normal_left_t.get(i) * normal_left_t.get(j) /
               get(distance_left_t)) +
          .5 * momentum_right_t.get(j) *
              (-normal_right_t.get(i) * normal_right_t.get(j) /
               get(distance_right_t));
    }
    shift_background->get(i) +=
        0.5 * momentum_left_t.get(i) / get(distance_left_t) +
        0.5 * momentum_right_t.get(i) / get(distance_right_t);
  }
  */
  // Horizon Penetrating Shift
  /*
  DataType present_time(get_size(get<0>(x)), max_time_interpolator);
  const auto distance_left_t = get_t_distance_left(present_time);
  const auto distance_right_t = get_t_distance_right(present_time);
  const auto areal_distance_left =
      find_areal_distance_left(get(distance_left_t));
  const auto areal_distance_right =
      find_areal_distance_right(get(distance_right_t));
  const auto normal_left = get_t_normal_left(present_time);
  const auto normal_right = get_t_normal_right(present_time);
  const DataType shift_r_left = .75 * sqrt(3) * square(mass_left) *
                                get(distance_left_t) /
                                pow(areal_distance_left, 3);
  const DataType shift_r_right = .75 * sqrt(3) * square(mass_right) *
                                 get(distance_right_t) /
                                 pow(areal_distance_right, 3);
  for (size_t i = 0; i < 3; ++i) {
    shift_background->get(i) +=
        shift_r_left * normal_left.get(i) + shift_r_right * normal_right.get(i);
  }
  */
  // Co-rotaing shift
  /*
  const double total_mass = mass_left + mass_right;
  const double reduced_mass = mass_left * mass_right / total_mass;
  const auto separation = get_t_separation(present_time);
  const auto angular_velocity = sqrt(total_mass / cube(get(separation))) *
                                (1. + .5 * (reduced_mass / total_mass - 3.) *
                                          total_mass / get(separation));
  get<0>(*shift_background) += -angular_velocity * get<1>(x);
  get<1>(*shift_background) += angular_velocity * get<0>(x);
  */
  // Boosted Horizon Penetrating superposed shift
  /*
  DataType present_time(get_size(get<0>(x)), max_time_interpolator);
  const auto superposed_spacetime_metric =
      get_t_superposed_spacetime_metric(present_time);
  const auto& inv_conformal_metric = cache->get_var(
      *this,
      ::Xcts::Tags::InverseConformalMetric<DataType, Dim, Frame::Inertial>{});
  gr::shift(shift_background, superposed_spacetime_metric,
  inv_conformal_metric);
  */
  /*
  DataType present_time(get_size(get<0>(x)), max_time_interpolator);
  const auto& shift_background_aux = get_t_shift(present_time);
  for (size_t i = 0; i < 3; ++i) {
    shift_background->get(i) = shift_background_aux.get(i);
  }
  */
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::II<DataType, 3, Frame::Inertial>*>
        longitudinal_shift_background_minus_dt_conformal_metric,
    const gsl::not_null<Cache*> cache,
    Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
        DataType, 3, Frame::Inertial> /*meta*/) const {
  // LongitudinalShiftBackground
  const auto& shift_background = cache->get_var(
      *this, ::Xcts::Tags::ShiftBackground<DataType, Dim, Frame::Inertial>{});
  const auto& deriv_shift_background = cache->get_var(
      *this, ::Tags::deriv<
                ::Xcts::Tags::ShiftBackground<DataType, Dim, Frame::Inertial>,
                tmpl::size_t<Dim>, Frame::Inertial>{});
  const auto& inv_conformal_metric = cache->get_var(
      *this,
      ::Xcts::Tags::InverseConformalMetric<DataType, Dim, Frame::Inertial>{});
  const auto& conformal_christoffel_second_kind = cache->get_var(
      *this, ::Xcts::Tags::ConformalChristoffelSecondKind<DataType, Dim,
                                                          Frame::Inertial>{});
  Xcts::longitudinal_operator(
      longitudinal_shift_background_minus_dt_conformal_metric,
      shift_background,
      deriv_shift_background, inv_conformal_metric,
      conformal_christoffel_second_kind);
  // DtConformalMetric (finite difference 1st order)

  const auto& conformal_metric = cache->get_var(
      *this, Xcts::Tags::ConformalMetric<DataType, Dim, Frame::Inertial>{});
  double time_displacement = 0.1;
  DataType time_back(get_size(x.get(0)), -time_displacement);
  const auto conformal_metric_back = get_t_conformal_metric(time_back);

  tnsr::ii<DataType, 3> dt_conformal_metric{get_size(x.get(0))};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      dt_conformal_metric.get(i, j) =
          (conformal_metric.get(i, j) - conformal_metric_back.get(i, j)) /
          time_displacement;
    }
  }

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        for (size_t l = 0; l < 3; ++l) {
          longitudinal_shift_background_minus_dt_conformal_metric->get(i, j) -=
              inv_conformal_metric.get(i, k) * inv_conformal_metric.get(j, l) *
              (dt_conformal_metric.get(k, l) -
               (1. / 3.) *
                   get(trace(dt_conformal_metric, inv_conformal_metric)) *
                   conformal_metric.get(k, l));
        }
      }
    }
  }
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    gsl::not_null<tnsr::iJ<DataType, Dim>*> deriv_shift_background,
    gsl::not_null<Cache*> cache,
    ::Tags::deriv<Xcts::Tags::ShiftBackground<DataType, Dim, Frame::Inertial>,
                  tmpl::size_t<Dim>, Frame::Inertial> /*meta*/) const {
  ASSERT(mesh.has_value() and inv_jacobian.has_value(),
         "Need a mesh and a Jacobian for numeric differentiation.");
  if constexpr (std::is_same_v<DataType, DataVector>) {
    const auto& shift_background = cache->get_var(
        *this, Xcts::Tags::ShiftBackground<DataType, Dim, Frame::Inertial>{});
    partial_derivative(deriv_shift_background, shift_background, mesh->get(),
                       inv_jacobian->get());
  } else {
    (void)deriv_shift_background;
    (void)cache;
    ERROR(
        "Numeric differentiation only works with DataVectors because it needs "
        "a grid.");
  }
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> conformal_energy_density,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::Conformal<gr::Tags::EnergyDensity<DataType>, 0> /*meta*/) const {
  get(*conformal_energy_density) = 0;
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> conformal_stress_trace,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::Conformal<gr::Tags::StressTrace<DataType>, 0> /*meta*/) const {
  get(*conformal_stress_trace) = 0.;
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, Dim>*> conformal_momentum_density,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::Conformal<gr::Tags::MomentumDensity<DataType, Dim>, 0> /*meta*/)
    const {
  std::fill(conformal_momentum_density->begin(),
            conformal_momentum_density->end(), 0.);
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> conformal_factor_minus_one,
    const gsl::not_null<Cache*> /*cache*/,
    Xcts::Tags::ConformalFactorMinusOne<DataType> /*meta*/) const {
  DataType present_time(get_size(get<0>(x)), max_time_interpolator);
  const auto conformal_factor_t = get_t_conformal_factor(present_time);
  get(*conformal_factor_minus_one) = get(conformal_factor_t) - 1.0;
  // get(*conformal_factor_minus_one) = 0.;
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*>
        lapse_times_conformal_factor_minus_one,
    const gsl::not_null<Cache*> /*cache*/,
    Xcts::Tags::LapseTimesConformalFactorMinusOne<DataType> /*meta*/) const {
  DataType present_time(get_size(get<0>(x)), max_time_interpolator);
  const auto lapse = get_t_lapse(present_time);
  const auto conformal_factor = get_t_conformal_factor(present_time);
  get(*lapse_times_conformal_factor_minus_one) =
      get(lapse) * get(conformal_factor) - 1.;
  // get(lapse) - 1.;
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, Dim>*> shift_excess,
    const gsl::not_null<Cache*> /*cache*/,
    Xcts::Tags::ShiftExcess<DataType, Dim, Frame::Inertial> /*meta*/) const {
  std::fill(shift_excess->begin(), shift_excess->end(), 0.);
  DataType present_time(get_size(get<0>(x)), max_time_interpolator);
  const auto shift_excess_aux = get_t_shift(present_time);
  for (size_t i = 0; i < 3; ++i) {
    shift_excess->get(i) += shift_excess_aux.get(i);
  }
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> rest_mass_density,
    const gsl::not_null<Cache*> /*cache*/,
    hydro::Tags::RestMassDensity<DataType> /*meta*/) const {
  get(*rest_mass_density) = 0.;
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> specific_enthalpy,
    const gsl::not_null<Cache*> /*cache*/,
    hydro::Tags::SpecificEnthalpy<DataType> /*meta*/) const {
  get(*specific_enthalpy) = 0.;
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> pressure,
    const gsl::not_null<Cache*> /*cache*/,
    hydro::Tags::Pressure<DataType> /*meta*/) const {
  get(*pressure) = 0.;
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*> spatial_velocity,
    const gsl::not_null<Cache*> /*cache*/,
    hydro::Tags::SpatialVelocity<DataType, 3> /*meta*/) const {
  std::fill(spatial_velocity->begin(), spatial_velocity->end(), 0.);
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> lorentz_factor,
    const gsl::not_null<Cache*> /*cache*/,
    hydro::Tags::LorentzFactor<DataType> /*meta*/) const {
  get(*lorentz_factor) = 0.;
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*> magnetic_field,
    const gsl::not_null<Cache*> /*cache*/,
    hydro::Tags::MagneticField<DataType, 3> /*meta*/) const {
  std::fill(magnetic_field->begin(), magnetic_field->end(), 0.);
}

template <typename DataType>
void BinaryWithGravitationalWavesVariables<
    DataType>::interpolate_past_history() {
  // Now interpolate the past history
  using boost::math::interpolators::cardinal_cubic_hermite;

  for (size_t i = 0; i < 3; ++i) {
    // static_cast being used because boost requires non-const arrays
    interpolation_position_left.at(i) = cardinal_cubic_hermite(
        static_cast<std::vector<double>>(past_position_left.at(i)),
        static_cast<std::vector<double>>(past_dt_position_left.at(i)),
        past_time.front(), std::abs(past_time[0] - past_time[1]));
    interpolation_position_right.at(i) = cardinal_cubic_hermite(
        static_cast<std::vector<double>>(past_position_right.at(i)),
        static_cast<std::vector<double>>(past_dt_position_right.at(i)),
        past_time.front(), std::abs(past_time[0] - past_time[1]));
    interpolation_momentum_left.at(i) = cardinal_cubic_hermite(
        static_cast<std::vector<double>>(past_momentum_left.at(i)),
        static_cast<std::vector<double>>(past_dt_momentum_left.at(i)),
        past_time.front(), std::abs(past_time[0] - past_time[1]));
    interpolation_momentum_right.at(i) = cardinal_cubic_hermite(
        static_cast<std::vector<double>>(past_momentum_right.at(i)),
        static_cast<std::vector<double>>(past_dt_momentum_right.at(i)),
        past_time.front(), std::abs(past_time[0] - past_time[1]));
  }
  // Get the domain of the interpolation to not trigger domain error on 0
  // (zero). The maximum time varies by machine roundoff. The interpolation is
  // done again because above is casted as std::function which does not have
  // access to the domain.
  max_time_interpolator =
      cardinal_cubic_hermite(
          static_cast<std::vector<double>>(past_position_left.at(2)),
          static_cast<std::vector<double>>(past_dt_position_left.at(2)),
          past_time.front(), std::abs(past_time[0] - past_time[1]))
          .domain()
          .second;
}

template <typename DataType>
DataType
BinaryWithGravitationalWavesVariables<DataType>::find_retarded_time_left(
    DataType t0) const {
  DataType bracket_lower(get_size(t0), past_time.front());
  DataType bracket_upper(get_size(t0), max_time_interpolator);
  return RootFinder::toms748<true>(
      [this, t0](const auto time, const size_t i) {
        tnsr::I<double, 3> v;
        for (size_t j = 0; j < 3; ++j) {
          v.get(j) =
              this->x.get(j)[i] - this->interpolation_position_left.at(j)(time);
        }

        return get(magnitude(v)) + time - t0[i];
      },
      bracket_lower, bracket_upper, 1e-8, 1e-10);
}

template <typename DataType>
DataType
BinaryWithGravitationalWavesVariables<DataType>::find_retarded_time_right(
    DataType t0) const {
  DataType bracket_lower(get_size(t0), past_time.front());
  DataType bracket_upper(get_size(t0), max_time_interpolator);
  return RootFinder::toms748<true>(
      [this, t0](const auto time, const size_t i) {
        tnsr::I<double, 3> v;
        for (size_t j = 0; j < 3; ++j) {
          v.get(j) = this->x.get(j)[i] -
                     this->interpolation_position_right.at(j)(time);
        }

        return get(magnitude(v)) + time - t0[i];
      },
      bracket_lower, bracket_upper, 1e-8, 1e-10);
}

template <typename DataType>
DataType
BinaryWithGravitationalWavesVariables<DataType>::find_areal_distance_left(
    DataType r) const {
  DataType bracket_lower(get_size(r), 1.5 * mass_left);
  DataType bracket_upper(get_size(r), 1e9);
  return RootFinder::toms748<true>(
      [this, r](const auto areal_radius, const size_t i) {
        double iso_radius =
            .25 *
            (2. * areal_radius + mass_left +
             sqrt(4. * square(areal_radius) + 4. * areal_radius * mass_left +
                  3. * square(mass_left))) *
            std::pow((4. + 3. * sqrt(2.)) *
                         (2. * areal_radius - 3. * mass_left) /
                         (8. * areal_radius + 6. * mass_left +
                          3. * sqrt(8. * square(areal_radius) +
                                    8. * areal_radius * mass_left +
                                    6. * square(mass_left))),
                     1. / sqrt(2.));

        return iso_radius - r[i];
      },
      bracket_lower, bracket_upper, 1e-8, 1e-10);
}

template <typename DataType>
DataType
BinaryWithGravitationalWavesVariables<DataType>::find_areal_distance_right(
    DataType r) const {
  DataType bracket_lower(get_size(r), 1.5 * mass_right);
  DataType bracket_upper(get_size(r), 1e9);
  return RootFinder::toms748<true>(
      [this, r](const auto areal_radius, const size_t i) {
        double iso_radius =
            .25 *
            (2. * areal_radius + mass_right +
             sqrt(4. * square(areal_radius) + 4. * areal_radius * mass_right +
                  3. * square(mass_right))) *
            std::pow((4. + 3. * sqrt(2.)) *
                         (2. * areal_radius - 3. * mass_right) /
                         (8. * areal_radius + 6. * mass_right +
                          3. * sqrt(8. * square(areal_radius) +
                                    8. * areal_radius * mass_right +
                                    6. * square(mass_right))),
                     1. / sqrt(2.));

        return iso_radius - r[i];
      },
      bracket_lower, bracket_upper, 1e-8, 1e-10);
}

template <typename DataType>
Scalar<DataType>
BinaryWithGravitationalWavesVariables<DataType>::get_t_distance_left(
    const DataType time) const {
  tnsr::I<DataType, 3> v = x;
  for (size_t i = 0; i < time.size(); ++i) {
    for (size_t j = 0; j < 3; ++j) {
      v.get(j)[i] = x.get(j)[i] - interpolation_position_left.at(j)(time[i]);
    }
  }
  return magnitude(v);
}

template <typename DataType>
Scalar<DataType>
BinaryWithGravitationalWavesVariables<DataType>::get_t_distance_right(
    const DataType time) const {
  tnsr::I<DataType, 3> v = x;
  for (size_t i = 0; i < time.size(); ++i) {
    for (size_t j = 0; j < 3; ++j) {
      v.get(j)[i] = x.get(j)[i] - interpolation_position_right.at(j)(time[i]);
    }
  }
  return magnitude(v);
}

template <typename DataType>
Scalar<DataType>
BinaryWithGravitationalWavesVariables<DataType>::get_t_separation(
    const DataType time) const {
  tnsr::I<DataType, 3> v = x;
  for (size_t i = 0; i < time.size(); ++i) {
    for (size_t j = 0; j < 3; ++j) {
      v.get(j)[i] = interpolation_position_right.at(j)(time[i]) -
                    interpolation_position_left.at(j)(time[i]);
    }
  }
  return magnitude(v);
}

template <typename DataType>
tnsr::I<DataType, 3>
BinaryWithGravitationalWavesVariables<DataType>::get_t_momentum_left(
    const DataType time) const {
  tnsr::I<DataType, 3> v = x;
  for (size_t i = 0; i < time.size(); ++i) {
    for (size_t j = 0; j < 3; ++j) {
      v.get(j)[i] = interpolation_momentum_left.at(j)(time[i]);
    }
  }
  return v;
}

template <typename DataType>
tnsr::I<DataType, 3>
BinaryWithGravitationalWavesVariables<DataType>::get_t_momentum_right(
    const DataType time) const {
  tnsr::I<DataType, 3> v = x;
  for (size_t i = 0; i < time.size(); ++i) {
    for (size_t j = 0; j < 3; ++j) {
      v.get(j)[i] = interpolation_momentum_right.at(j)(time[i]);
    }
  }
  return v;
}

template <typename DataType>
tnsr::I<DataType, 3>
BinaryWithGravitationalWavesVariables<DataType>::get_t_normal_left(
    const DataType time) const {
  tnsr::I<DataType, 3> v = x;
  Scalar<DataType> distance_left = get_t_distance_left(time);
  for (size_t i = 0; i < time.size(); ++i) {
    for (size_t j = 0; j < 3; ++j) {
      v.get(j)[i] = (x.get(j)[i] - interpolation_position_left.at(j)(time[i])) /
                    get(distance_left)[i];
    }
  }
  return v;
}

template <typename DataType>
tnsr::I<DataType, 3>
BinaryWithGravitationalWavesVariables<DataType>::get_t_normal_right(
    const DataType time) const {
  tnsr::I<DataType, 3> v = x;
  Scalar<DataType> distance_right = get_t_distance_right(time);
  for (size_t i = 0; i < time.size(); ++i) {
    for (size_t j = 0; j < 3; ++j) {
      v.get(j)[i] =
          (x.get(j)[i] - interpolation_position_right.at(j)(time[i])) /
          get(distance_right)[i];
    }
  }

  return v;
}

template <typename DataType>
tnsr::I<DataType, 3>
BinaryWithGravitationalWavesVariables<DataType>::get_t_normal_lr(
    const DataType time) const {
  tnsr::I<DataType, 3> v = x;
  Scalar<DataType> past_separation = get_t_separation(time);
  for (size_t i = 0; i < time.size(); ++i) {
    for (size_t j = 0; j < 3; ++j) {
      v.get(j)[i] = (interpolation_position_left.at(j)(time[i]) -
                     interpolation_position_right.at(j)(time[i])) /
                    get(past_separation)[i];
    }
  }
  return v;
}

template <typename DataType>
DataType BinaryWithGravitationalWavesVariables<DataType>::integrate_term(
    const DataType time, const size_t i, const size_t j, const int left_right,
    const double t0) const {
  DataType result{x.get(0).size()};
  std::array<std::function<double(double)>, 3> this_interpolation_position{};
  std::array<std::function<double(double)>, 3> this_interpolation_momentum{};
  if (left_right == -1) {
    for (size_t l = 0; l < 3; ++l) {
      this_interpolation_position.at(l) = interpolation_position_left.at(l);
      this_interpolation_momentum.at(l) = interpolation_momentum_left.at(l);
    }
  } else if (left_right == 1) {
    for (size_t l = 0; l < 3; ++l) {
      this_interpolation_position.at(l) = interpolation_position_right.at(l);
      this_interpolation_momentum.at(l) = interpolation_momentum_right.at(l);
    }
  }
  for (size_t k = 0; k < x.get(0).size(); ++k) {
    // double error;
    using boost::math::quadrature::trapezoidal;
    // boost::math::quadrature::gauss_kronrod<double, 31> integration;
    // const integration::GslQuadAdaptive<
    //     integration::GslIntegralType::StandardGaussKronrod>
    //     integration{100};
    result[k] = trapezoidal(
        [this, left_right, i, j, k](const double t) {
          std::array<double, 3> u1{};
          std::array<double, 3> u2{};
          const double this_distance_at_t =
              sqrt(pow(this_interpolation_position.at(0)(t) - x.get(0)[k], 2) +
                   pow(this_interpolation_position.at(1)(t) - x.get(1)[k], 2) +
                   pow(this_interpolation_position.at(2)(t) - x.get(2)[k], 2));
          const double separation_at_t =
              sqrt(pow(interpolation_position_left.at(0)(t) -
                           interpolation_position_right.at(0)(t),
                       2) +
                   pow(interpolation_position_left.at(1)(t) -
                           interpolation_position_right.at(1)(t),
                       2) +
                   pow(interpolation_position_left.at(2)(t) -
                           interpolation_position_right.at(2)(t),
                       2));
          const std::array<double, 3> this_momentum_at_t = {
              this_interpolation_momentum.at(0)(t),
              this_interpolation_momentum.at(1)(t),
              this_interpolation_momentum.at(2)(t)};
          const std::array<double, 3> this_normal_at_t = {
              (x.get(0)[k] - this_interpolation_position.at(0)(t)) /
                  this_distance_at_t,
              (x.get(1)[k] - this_interpolation_position.at(1)(t)) /
                  this_distance_at_t,
              (x.get(2)[k] - this_interpolation_position.at(2)(t)) /
                  this_distance_at_t};
          const std::array<double, 3> normal_lr_at_t = {
              (interpolation_position_left.at(0)(t) -
               interpolation_position_right.at(0)(t)) /
                  separation_at_t,
              (interpolation_position_left.at(1)(t) -
               interpolation_position_right.at(1)(t)) /
                  separation_at_t,
              (interpolation_position_left.at(2)(t) -
               interpolation_position_right.at(2)(t)) /
                  separation_at_t};
          const std::array<std::array<double, 3>, 3> delta{
              {{{1., 0., 0.}}, {{0., 1., 0.}}, {{0., 0., 1.}}}};
          for (size_t l = 0; l < 3; ++l) {
            u1.at(l) = this_momentum_at_t.at(l) / std::sqrt(mass_left);
            u2.at(l) = sqrt(mass_left * mass_right / (2 * separation_at_t)) *
                       normal_lr_at_t.at(l);
          }
          const double term1 =
              t /
              (this_distance_at_t * this_distance_at_t * this_distance_at_t) *
              ((-5. * dot(u1, u1) +
                9. * dot(u1, this_normal_at_t) * dot(u1, this_normal_at_t)) *
                   delta.at(i).at(j) +
               6. * u1.at(i) * u1.at(j) -
               6. * dot(u1, this_normal_at_t) *
                   (u1.at(i) * this_normal_at_t.at(j) +
                    u1.at(j) * this_normal_at_t.at(i)) +
               (9. * dot(u1, u1) -
                15. * dot(u1, this_normal_at_t) * dot(u1, this_normal_at_t)) *
                   this_normal_at_t.at(i) * this_normal_at_t.at(j));
          const double term2 =
              t * t * t /
              (this_distance_at_t * this_distance_at_t * this_distance_at_t *
               this_distance_at_t * this_distance_at_t) *
              ((dot(u1, u1) -
                5. * dot(u1, this_normal_at_t) * dot(u1, this_normal_at_t)) *
                   delta.at(i).at(j) +
               2. * u1.at(i) * u1.at(j) -
               10. * dot(u1, this_normal_at_t) *
                   (u1.at(i) * this_normal_at_t.at(j) +
                    u1.at(j) * this_normal_at_t.at(i)) +
               (-5. * dot(u1, u1) * dot(u1, u1) +
                35. * dot(u1, this_normal_at_t) * dot(u1, this_normal_at_t)) *
                   this_normal_at_t.at(i) * this_normal_at_t.at(j));
          const double term3 =
              t /
              (this_distance_at_t * this_distance_at_t * this_distance_at_t) *
              ((-5. * dot(u2, u2) +
                9. * dot(u2, this_normal_at_t) * dot(u2, this_normal_at_t)) *
                   delta.at(i).at(j) +
               6. * u2.at(i) * u2.at(j) -
               6. * dot(u2, this_normal_at_t) *
                   (u2.at(i) * this_normal_at_t.at(j) +
                    u2.at(j) * this_normal_at_t.at(i)) +
               (9. * dot(u2, u2) -
                15. * dot(u2, this_normal_at_t) * dot(u2, this_normal_at_t)) *
                   this_normal_at_t.at(i) * this_normal_at_t.at(j));
          const double term4 =
              t * t * t /
              (this_distance_at_t * this_distance_at_t * this_distance_at_t *
               this_distance_at_t * this_distance_at_t) *
              ((dot(u2, u2) -
                5. * dot(u2, this_normal_at_t) * dot(u2, this_normal_at_t)) *
                   delta.at(i).at(j) +
               2. * u2.at(i) * u2.at(j) -
               10. * dot(u2, this_normal_at_t) *
                   (u2.at(i) * this_normal_at_t.at(j) +
                    u2.at(j) * this_normal_at_t.at(i)) +
               (-5. * dot(u2, u2) * dot(u2, u2) +
                35. * dot(u2, this_normal_at_t) * dot(u2, this_normal_at_t)) *
                   this_normal_at_t.at(i) * this_normal_at_t.at(j));
          return term1 + term2 + term3 + term4;
          // return 0.0;
        },
        time[k], 0.0, 1., max_time_interpolator, 1e-8);
  }
  return result;
}

template <typename DataType>
tnsr::i<DataType, 3> BinaryWithGravitationalWavesVariables<
    DataType>::get_t_deriv_one_over_distance_left(DataType t) const {
  const auto distance_left_t = get_t_distance_left(t);
  const auto normal_left_t = get_t_normal_left(t);
  tnsr::i<DataType, 3> deriv_one_over_distance_left_t{t.size()};
  for (size_t i = 0; i < 3; ++i) {
    deriv_one_over_distance_left_t.get(i) =
        -normal_left_t.get(i) / (get(distance_left_t) * get(distance_left_t));
  }
  return deriv_one_over_distance_left_t;
}

template <typename DataType>
tnsr::i<DataType, 3> BinaryWithGravitationalWavesVariables<
    DataType>::get_t_deriv_one_over_distance_right(DataType t) const {
  const auto distance_right_t = get_t_distance_right(t);
  const auto normal_right_t = get_t_normal_right(t);
  tnsr::i<DataType, 3> deriv_one_over_distance_right_t{t.size()};
  for (size_t i = 0; i < 3; ++i) {
    deriv_one_over_distance_right_t.get(i) =
        -normal_right_t.get(i) /
        (get(distance_right_t) * get(distance_right_t));
  }
  return deriv_one_over_distance_right_t;
}

template <typename DataType>
tnsr::ijk<DataType, 3>
BinaryWithGravitationalWavesVariables<DataType>::get_t_deriv_3_distance_left(
    DataType t) const {
  const auto distance_left_t = get_t_distance_left(t);
  const auto normal_left_t = get_t_normal_left(t);
  std::array<std::array<double, 3>, 3> delta{
      {{{1., 0., 0.}}, {{0., 1., 0.}}, {{0., 0., 1.}}}};
  tnsr::ijk<DataType, 3> deriv_3_distance_left_t{t.size()};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        deriv_3_distance_left_t.get(i, j, k) =
            (-normal_left_t.get(i) * delta.at(j).at(k) -
             normal_left_t.get(j) * delta.at(i).at(k) -
             normal_left_t.get(k) * delta.at(i).at(j) +
             3 * normal_left_t.get(i) * normal_left_t.get(j) *
                 normal_left_t.get(k)) /
            (get(distance_left_t) * get(distance_left_t));
      }
    }
  }
  return deriv_3_distance_left_t;
}

template <typename DataType>
tnsr::ijk<DataType, 3>
BinaryWithGravitationalWavesVariables<DataType>::get_t_deriv_3_distance_right(
    DataType t) const {
  const auto distance_right_t = get_t_distance_right(t);
  const auto normal_right_t = get_t_normal_right(t);
  std::array<std::array<double, 3>, 3> delta{
      {{{1., 0., 0.}}, {{0., 1., 0.}}, {{0., 0., 1.}}}};
  tnsr::ijk<DataType, 3> deriv_3_distance_right_t{t.size()};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        deriv_3_distance_right_t.get(i, j, k) =
            (-normal_right_t.get(i) * delta.at(j).at(k) -
             normal_right_t.get(j) * delta.at(i).at(k) -
             normal_right_t.get(k) * delta.at(i).at(j) +
             3 * normal_right_t.get(i) * normal_right_t.get(j) *
                 normal_right_t.get(k)) /
            (get(distance_right_t) * get(distance_right_t));
      }
    }
  }
  return deriv_3_distance_right_t;
}

template <typename DataType>
tnsr::ii<DataType, 3>
BinaryWithGravitationalWavesVariables<DataType>::get_t_pn_conjugate_momentum3(
    DataType t) const {
  const auto deriv_one_over_distance_left_t =
      get_t_deriv_one_over_distance_left(t);
  const auto deriv_one_over_distance_right_t =
      get_t_deriv_one_over_distance_right(t);
  const auto deriv_3_distance_left_t = get_t_deriv_3_distance_left(t);
  const auto deriv_3_distance_right_t = get_t_deriv_3_distance_right(t);
  const auto momentum_left_t = get_t_momentum_left(t);
  const auto momentum_right_t = get_t_momentum_right(t);
  std::array<std::array<double, 3>, 3> delta{
      {{{1., 0., 0.}}, {{0., 1., 0.}}, {{0., 0., 1.}}}};
  tnsr::ii<DataType, 3> pn_conjugate_momentum3_t{t.size()};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      pn_conjugate_momentum3_t.get(i, j) = 0.;
      for (size_t k = 0; k < 3; ++k) {
        pn_conjugate_momentum3_t.get(i, j) +=
            momentum_left_t.get(k) *
                (2 * (delta.at(i).at(k) *
                          deriv_one_over_distance_left_t.get(j) +
                      delta.at(j).at(k) *
                          deriv_one_over_distance_left_t.get(i)) -
                 delta.at(i).at(j) * deriv_one_over_distance_left_t.get(k) -
                 0.5 * deriv_3_distance_left_t.get(i, j, k)) +
            momentum_right_t.get(k) *
                (2 * (delta.at(i).at(k) *
                          deriv_one_over_distance_right_t.get(j) +
                      delta.at(j).at(k) *
                          deriv_one_over_distance_right_t.get(i)) -
                 delta.at(i).at(j) * deriv_one_over_distance_right_t.get(k) -
                 0.5 * deriv_3_distance_right_t.get(i, j, k));
      }
    }
  }
  return pn_conjugate_momentum3_t;
}

template <typename DataType>
tnsr::ii<DataType, 3>
BinaryWithGravitationalWavesVariables<DataType>::get_t_pn_extrinsic_curvature(
    DataType t) const {
  const auto pn_conjugate_momentum3_t = get_t_pn_conjugate_momentum3(t);
  const auto conformal_factor_t = get_t_conformal_factor(t);
  tnsr::ii<DataType, 3> pn_extrinsic_curvature_t{t.size()};
  const auto one_over_comformal_factor_to_ten_t =
      1. / (get(conformal_factor_t) * get(conformal_factor_t) *
            get(conformal_factor_t) * get(conformal_factor_t) *
            get(conformal_factor_t) * get(conformal_factor_t) *
            get(conformal_factor_t) * get(conformal_factor_t) *
            get(conformal_factor_t) * get(conformal_factor_t));
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      pn_extrinsic_curvature_t.get(i, j) = -one_over_comformal_factor_to_ten_t *
                                           pn_conjugate_momentum3_t.get(i, j);
    }
  }
  return pn_extrinsic_curvature_t;
}

template <typename DataType>
Scalar<DataType> BinaryWithGravitationalWavesVariables<
    DataType>::get_t_trace_extrinsic_curvature(DataType t) const {
  // PN Trace Extrinsic Curvatfure
  /*
  const auto pn_extrinsic_curvature_t = get_t_pn_extrinsic_curvature(t);
  const auto conformal_metric_t = get_t_conformal_metric(t);
  tnsr::ii<DataType, 3> inv_conformal_metric_t{t.size()};
  const auto det_and_inv = determinant_and_inverse(conformal_metric_t);
  return trace(pn_extrinsic_curvature_t, det_and_inv.second);
  */
  // Boosted Horizon Penetrating superposed trace extrinsic curvature
  tnsr::ii<DataType, 3> extrinsic_curvature{t.size()};
  std::fill(extrinsic_curvature.begin(), extrinsic_curvature.end(), 0.);

  double time_displacement = 0.1;
  DataType time_back(get_size(x.get(0)), t[0] - time_displacement);
  const auto conformal_metric = get_t_conformal_metric(t);
  const auto inv_conformal_metric =
      determinant_and_inverse(conformal_metric).second;
  const auto conformal_metric_back = get_t_conformal_metric(time_back);
  const auto lapse = get_t_lapse(t);

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      extrinsic_curvature.get(i, j) +=
          (conformal_metric.get(i, j) - conformal_metric_back.get(i, j)) /
          (time_displacement * (-2.) * get(lapse));
    }
  }

  const auto shift = get_t_shift(t);
  const auto shift_down = raise_or_lower_index(shift, conformal_metric);
  const auto deriv_shift =
      partial_derivative(shift_down, mesh->get(), inv_jacobian->get());
  const auto deriv_conformal_metric =
      partial_derivative(conformal_metric, mesh->get(), inv_jacobian->get());
  const auto christoffel_second_kind =
      gr::christoffel_second_kind(deriv_conformal_metric, inv_conformal_metric);
  const auto covariant_deriv_shift_contribution = tenex::evaluate<ti::i, ti::j>(
      deriv_shift(ti::i, ti::j) -
      christoffel_second_kind(ti::K, ti::i, ti::j) * shift_down(ti::k) +
      deriv_shift(ti::j, ti::i) -
      christoffel_second_kind(ti::K, ti::j, ti::i) * shift_down(ti::k));

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      extrinsic_curvature.get(i, j) +=
          covariant_deriv_shift_contribution.get(i, j) / (2. * get(lapse));
    }
  }
  return trace(extrinsic_curvature, inv_conformal_metric);
}

template <typename DataType>
tnsr::ii<DataType, 3>
BinaryWithGravitationalWavesVariables<DataType>::get_t_conformal_metric(
    DataType t) const {
  // Boosted Horizon Penetrating superposed spatial metric
  const auto conformal_factor_t = get_t_conformal_factor(t);
  const auto radiative_term = get_t_radiative_term(t);
  const auto superposed_spacetime_metric_t =
      get_t_superposed_spacetime_metric(t);
  auto conformal_metric = gr::spatial_metric(superposed_spacetime_metric_t);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      conformal_metric.get(i, j) +=
          radiative_term.get(i, j) /
          (get(conformal_factor_t) * get(conformal_factor_t) *
           get(conformal_factor_t) * get(conformal_factor_t));
    }
  }
  return conformal_metric;
}

template <typename DataType>
tnsr::ii<DataType, 3>
BinaryWithGravitationalWavesVariables<DataType>::get_t_radiative_term(
    DataType t) const {
  const auto distance_left_t = get_t_distance_left(t);
  const auto distance_right_t = get_t_distance_right(t);
  const auto near_zone_term_t = get_t_near_zone_term(t);
  const auto present_term_t = get_t_present_term(t);
  const auto past_term_t = get_t_past_term(t);
  const auto integral_term_t = get_t_integral_term(t);
  double turn_off = .5;
  if (attenuation_parameter == 0) {
    turn_off = 1.;
  }
  tnsr::ii<DataType, 3> radiative_term_t{t.size()};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      radiative_term_t.get(i, j) =
          (turn_off + .5 * tanh(attenuation_parameter *
                                (get(distance_left_t) - attenuation_radius))) *
          (turn_off + .5 * tanh(attenuation_parameter *
                                (get(distance_right_t) - attenuation_radius))) *
          (near_zone_term_t.get(i, j) + present_term_t.get(i, j) +
           past_term_t.get(i, j) + integral_term_t.get(i, j));
    }
  }
  return radiative_term_t;
}

template <typename DataType>
tnsr::ii<DataType, 3>
BinaryWithGravitationalWavesVariables<DataType>::get_t_near_zone_term(
    DataType t) const {
  const auto distance_left_t = get_t_distance_left(t);
  const auto distance_right_t = get_t_distance_right(t);
  const auto normal_left_t = get_t_normal_left(t);
  const auto normal_right_t = get_t_normal_right(t);
  const auto separation_t = get_t_separation(t);
  const auto momentum_left_t = get_t_momentum_left(t);
  const auto momentum_right_t = get_t_momentum_right(t);
  const auto s =
      get(distance_left_t) + get(distance_right_t) + get(separation_t);
  const auto normal_lr_t = get_t_normal_lr(t);
  tnsr::ii<DataType, 3> near_zone_term_t{t.size()};
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      near_zone_term_t.get(i, j) =
          0.25 / (mass_left * get(distance_left_t)) *
              (2. * momentum_left_t.get(i) * momentum_left_t.get(j) +
               (3. * get(dot_product(normal_left_t, momentum_left_t)) *
                    get(dot_product(normal_left_t, momentum_left_t)) -
                5. * get(dot_product(momentum_left_t, momentum_left_t))) *
                   normal_left_t.get(i) * normal_left_t.get(j) +
               6. * get(dot_product(normal_left_t, momentum_left_t)) *
                   (normal_left_t.get(i) * momentum_left_t.get(j) +
                    normal_left_t.get(j) * momentum_left_t.get(i))) +
          0.25 / (mass_right * get(distance_right_t)) *
              (2. * momentum_right_t.get(i) * momentum_right_t.get(j) +
               (3. * get(dot_product(normal_right_t, momentum_right_t)) *
                    get(dot_product(normal_right_t, momentum_right_t)) -
                5. * get(dot_product(momentum_right_t, momentum_right_t))) *
                   normal_right_t.get(i) * normal_right_t.get(j) +
               6. * get(dot_product(normal_right_t, momentum_right_t)) *
                   (normal_right_t.get(i) * momentum_right_t.get(j) +
                    normal_right_t.get(j) * momentum_right_t.get(i))) +
          0.125 * (mass_left * mass_right) *
              (-32. / s * (1. / get(separation_t) + 1. / s) *
                   normal_lr_t.get(i) * normal_lr_t.get(j) +
               2. *
                   ((get(distance_left_t) + get(distance_right_t)) /
                        (get(separation_t) * get(separation_t) *
                         get(separation_t)) +
                    12. / (s * s)) *
                   normal_left_t.get(i) * normal_right_t.get(j) +
               16. *
                   (2. / (s * s) -
                    1. / (get(separation_t) * get(separation_t))) *
                   (normal_left_t.get(i) * normal_lr_t.get(j) +
                    normal_left_t.get(j) * normal_lr_t.get(i)) +
               (5. / (get(separation_t) * get(distance_left_t)) -
                1. /
                    (get(separation_t) * get(separation_t) *
                     get(separation_t)) *
                    ((get(distance_right_t) * get(distance_right_t)) /
                         get(distance_left_t) +
                     3. * get(distance_left_t)) -
                8. / s * (1. / get(distance_left_t) + 1. / s)) *
                   normal_left_t.get(i) * normal_left_t.get(j) -
               32. / s * (1. / get(separation_t) + 1. / s) *
                   normal_lr_t.get(i) * normal_lr_t.get(j) +
               2. *
                   ((get(distance_left_t) + get(distance_right_t)) /
                        (get(separation_t) * get(separation_t) *
                         get(separation_t)) +
                    12. / (s * s)) *
                   normal_right_t.get(i) * normal_left_t.get(j) -
               16. *
                   (2. / (s * s) -
                    1. / (get(separation_t) * get(separation_t))) *
                   (normal_right_t.get(i) * normal_lr_t.get(j) +
                    normal_right_t.get(j) * normal_lr_t.get(i)) +
               (5. / (get(separation_t) * get(distance_right_t)) -
                1. /
                    (get(separation_t) * get(separation_t) *
                     get(separation_t)) *
                    ((get(distance_left_t) * get(distance_left_t)) /
                         get(distance_right_t) +
                     3. * get(distance_right_t)) -
                8. / s * (1. / get(distance_right_t) + 1. / s)) *
                   normal_right_t.get(i) * normal_right_t.get(j));
    }
    near_zone_term_t.get(i, i) +=
        0.25 / (mass_left * get(distance_left_t)) *
            (get(dot_product(momentum_left_t, momentum_left_t)) -
             5. * get(dot_product(normal_left_t, momentum_left_t)) *
                 get(dot_product(normal_left_t, momentum_left_t))) +
        0.25 / (mass_right * get(distance_right_t)) *
            (get(dot_product(momentum_right_t, momentum_right_t)) -
             5. * get(dot_product(normal_right_t, momentum_right_t)) *
                 get(dot_product(normal_right_t, momentum_right_t))) +
        0.125 * (mass_left * mass_right) *
            (5. * get(distance_left_t) /
                 (get(separation_t) * get(separation_t) * get(separation_t)) *
                 (get(distance_left_t) / get(distance_right_t) - 1.) -
             17. / (get(separation_t) * get(distance_left_t)) +
             4. / (get(distance_left_t) * get(distance_right_t)) +
             8. / s * (1. / get(distance_left_t) + 4. / get(separation_t)) +
             5. * get(distance_right_t) /
                 (get(separation_t) * get(separation_t) * get(separation_t)) *
                 (get(distance_right_t) / get(distance_left_t) - 1.) -
             17. / (get(separation_t) * get(distance_right_t)) +
             4. / (get(distance_left_t) * get(distance_right_t)) +
             8. / s * (1. / get(distance_right_t) + 4. / get(separation_t)));
  }
  return near_zone_term_t;
}

template <typename DataType>
tnsr::ii<DataType, 3>
BinaryWithGravitationalWavesVariables<DataType>::get_t_present_term(
    DataType t) const {
  const auto distance_left_t = get_t_distance_left(t);
  const auto distance_right_t = get_t_distance_right(t);
  const auto normal_left_t = get_t_normal_left(t);
  const auto normal_right_t = get_t_normal_right(t);
  const auto momentum_left_t = get_t_momentum_left(t);
  const auto momentum_right_t = get_t_momentum_right(t);
  const auto separation_t = get_t_separation(t);
  const auto normal_lr_t = get_t_normal_lr(t);
  tnsr::ii<DataType, 3> present_term_t{t.size()};
  tnsr::I<DataType, 3> u1_1(x);
  tnsr::I<DataType, 3> u1_2(x);
  tnsr::I<DataType, 3> u2(x);
  for (size_t i = 0; i < 3; ++i) {
    u1_1.get(i) = momentum_left_t.get(i) / sqrt(mass_left);
    u1_2.get(i) = momentum_right_t.get(i) / sqrt(mass_right);
    u2.get(i) = sqrt(mass_left * mass_right / (2. * get(separation_t))) *
                normal_lr_t.get(i);
  }
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      present_term_t.get(i, j) =
          -0.25 / get(distance_left_t) *
              (2. * u1_1.get(i) * u1_1.get(j) +
               (3. * get(dot_product(u1_1, normal_left_t)) *
                    get(dot_product(u1_1, normal_left_t)) -
                5. * get(dot_product(u1_1, u1_1))) *
                   normal_left_t.get(i) * normal_left_t.get(j) +
               6. * get(dot_product(u1_1, normal_left_t)) *
                   (normal_left_t.get(i) * u1_1.get(j) +
                    normal_left_t.get(j) * u1_1.get(i)) +
               2. * u2.get(i) * u2.get(j) +
               (3. * get(dot_product(u2, normal_left_t)) *
                    get(dot_product(u2, normal_left_t)) -
                5. * get(dot_product(u2, u2))) *
                   normal_left_t.get(i) * normal_left_t.get(j) +
               6. * get(dot_product(u2, normal_left_t)) *
                   (normal_left_t.get(i) * u2.get(j) +
                    normal_left_t.get(j) * u2.get(i))) -
          0.25 / get(distance_right_t) *
              (2. * u1_2.get(i) * u1_2.get(j) +
               (3. * get(dot_product(u1_2, normal_right_t)) *
                    get(dot_product(u1_2, normal_right_t)) -
                5. * get(dot_product(u1_2, u1_2))) *
                   normal_right_t.get(i) * normal_right_t.get(j) +
               6. * get(dot_product(u1_2, normal_right_t)) *
                   (normal_right_t.get(i) * u1_2.get(j) +
                    normal_right_t.get(j) * u1_2.get(i)) +
               2. * u2.get(i) * u2.get(j) +
               (3. * get(dot_product(u2, normal_right_t)) *
                    get(dot_product(u2, normal_right_t)) -
                5. * get(dot_product(u2, u2))) *
                   normal_right_t.get(i) * normal_right_t.get(j) +
               6. * get(dot_product(u2, normal_right_t)) *
                   (normal_right_t.get(i) * u2.get(j) +
                    normal_right_t.get(j) * u2.get(i)));
    }
    present_term_t.get(i, i) +=
        -0.25 / get(distance_left_t) *
            (get(dot_product(u1_1, u1_1)) -
             5. * get(dot_product(u1_1, normal_left_t)) *
                 get(dot_product(u1_1, normal_left_t)) +
             get(dot_product(u2, u2)) -
             5. * get(dot_product(u2, normal_left_t)) *
                 get(dot_product(u2, normal_left_t))) -
        0.25 / get(distance_right_t) *
            (get(dot_product(u1_2, u1_2)) -
             5. * get(dot_product(u1_2, normal_right_t)) *
                 get(dot_product(u1_2, normal_right_t)) +
             get(dot_product(u2, u2)) -
             5. * get(dot_product(u2, normal_right_t)) *
                 get(dot_product(u2, normal_right_t)));
  }
  return present_term_t;
}

template <typename DataType>
tnsr::ii<DataType, 3>
BinaryWithGravitationalWavesVariables<DataType>::get_t_past_term(
    DataType t) const {
  const auto retarded_time_left_t = find_retarded_time_left(t);
  const auto retarded_time_right_t = find_retarded_time_right(t);
  DataType distance_left_at_retarded_time_left =
      get(get_t_distance_left(retarded_time_left_t));
  DataType distance_right_at_retarded_time_right =
      get(get_t_distance_right(retarded_time_right_t));
  DataType separation_at_retarded_time_left =
      get(get_t_separation(retarded_time_left_t));
  DataType separation_at_retarded_time_right =
      get(get_t_separation(retarded_time_right_t));
  tnsr::I<DataType, 3> momentum_left_at_retarded_time_left =
      get_t_momentum_left(retarded_time_left_t);
  tnsr::I<DataType, 3> momentum_right_at_retarded_time_right =
      get_t_momentum_right(retarded_time_right_t);
  tnsr::I<DataType, 3> normal_left_at_retarded_time_left =
      get_t_normal_left(retarded_time_left_t);
  tnsr::I<DataType, 3> normal_right_at_retarded_time_right =
      get_t_normal_right(retarded_time_right_t);
  tnsr::I<DataType, 3> normal_lr_at_retarded_time_left =
      get_t_normal_lr(retarded_time_left_t);
  tnsr::I<DataType, 3> normal_lr_at_retarded_time_right =
      get_t_normal_lr(retarded_time_right_t);
  tnsr::I<DataType, 3> u1_1{t.size()};
  tnsr::I<DataType, 3> u1_2{t.size()};
  tnsr::I<DataType, 3> u2_1{t.size()};
  tnsr::I<DataType, 3> u2_2{t.size()};
  for (size_t i = 0; i < 3; ++i) {
    u1_1.get(i) =
        momentum_left_at_retarded_time_left.get(i) / std::sqrt(mass_left);
    u2_1.get(i) =
        sqrt(mass_left * mass_right / (2. * separation_at_retarded_time_left)) *
        normal_lr_at_retarded_time_left.get(i);

    u1_2.get(i) =
        momentum_right_at_retarded_time_right.get(i) / std::sqrt(mass_right);
    u2_2.get(i) = sqrt(mass_left * mass_right /
                       (2. * separation_at_retarded_time_right)) *
                  normal_lr_at_retarded_time_right.get(i);
  }
  tnsr::ii<DataType, 3> past_term_t{t.size()};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      past_term_t.get(i, j) =
          -1. / (distance_left_at_retarded_time_left) *
              (4. * u1_1.get(i) * u1_1.get(j) +
               (2. * get(dot_product(u1_1, u1_1)) +
                2. * get(dot_product(u1_1, normal_left_at_retarded_time_left)) *
                    get(dot_product(u1_1, normal_left_at_retarded_time_left))) *
                   normal_left_at_retarded_time_left.get(i) *
                   normal_left_at_retarded_time_left.get(j) -
               4. * get(dot_product(u1_1, normal_left_at_retarded_time_left)) *
                   (normal_left_at_retarded_time_left.get(i) * u1_1.get(j) +
                    normal_left_at_retarded_time_left.get(j) * u1_1.get(i))) -
          1. / (distance_right_at_retarded_time_right) *
              (4. * u1_2.get(i) * u1_2.get(j) +
               (2. * get(dot_product(u1_2, u1_2)) +
                2. *
                    get(dot_product(u1_2,
                                    normal_right_at_retarded_time_right)) *
                    get(dot_product(u1_2,
                                    normal_right_at_retarded_time_right))) *
                   normal_right_at_retarded_time_right.get(i) *
                   normal_right_at_retarded_time_right.get(j) -
               4. *
                   get(dot_product(u1_2, normal_right_at_retarded_time_right)) *
                   (normal_right_at_retarded_time_right.get(i) * u1_2.get(j) +
                    normal_right_at_retarded_time_right.get(j) * u1_2.get(i))) -
          1. / (distance_left_at_retarded_time_left) *
              (4. * u2_1.get(i) * u2_1.get(j) +
               (2. * get(dot_product(u2_1, u2_1)) +
                2. * get(dot_product(u2_1, normal_left_at_retarded_time_left)) *
                    get(dot_product(u2_1, normal_left_at_retarded_time_left))) *
                   normal_left_at_retarded_time_left.get(i) *
                   normal_left_at_retarded_time_left.get(j) -
               4. * get(dot_product(u2_1, normal_left_at_retarded_time_left)) *
                   (normal_left_at_retarded_time_left.get(i) * u2_1.get(j) +
                    normal_left_at_retarded_time_left.get(j) * u2_1.get(i))) -
          1. / (distance_right_at_retarded_time_right) *
              (4. * u2_2.get(i) * u2_2.get(j) +
               (2. * get(dot_product(u2_2, u2_2)) +
                2. *
                    get(dot_product(u2_2,
                                    normal_right_at_retarded_time_right)) *
                    get(dot_product(u2_2,
                                    normal_right_at_retarded_time_right))) *
                   normal_right_at_retarded_time_right.get(i) *
                   normal_right_at_retarded_time_right.get(j) -
               4. *
                   get(dot_product(u2_2, normal_right_at_retarded_time_right)) *
                   (normal_right_at_retarded_time_right.get(i) * u2_2.get(j) +
                    normal_right_at_retarded_time_right.get(j) * u2_2.get(i)));
    }
    past_term_t.get(i, i) +=
        -1. / (distance_left_at_retarded_time_left) *
            (-2. * get(dot_product(u1_1, u1_1)) +
             2. * get(dot_product(u1_1, normal_left_at_retarded_time_left)) *
                 get(dot_product(u1_1, normal_left_at_retarded_time_left))) -
        1. / (distance_right_at_retarded_time_right) *
            (-2. * get(dot_product(u1_2, u1_2)) +
             2. * get(dot_product(u1_2, normal_right_at_retarded_time_right)) *
                 get(dot_product(u1_2, normal_right_at_retarded_time_right))) -
        1. / (distance_left_at_retarded_time_left) *
            (-2. * get(dot_product(u2_1, u2_1)) +
             2. * get(dot_product(u2_1, normal_left_at_retarded_time_left)) *
                 get(dot_product(u2_1, normal_left_at_retarded_time_left))) -
        1. / (distance_right_at_retarded_time_right) *
            (-2. * get(dot_product(u2_2, u2_2)) +
             2. * get(dot_product(u2_2, normal_right_at_retarded_time_right)) *
                 get(dot_product(u2_2, normal_right_at_retarded_time_right)));
  }
  return past_term_t;
}

template <typename DataType>
tnsr::ii<DataType, 3>
BinaryWithGravitationalWavesVariables<DataType>::get_t_integral_term(
    DataType t) const {
  const auto retarded_time_left_t = find_retarded_time_left(t);
  const auto retarded_time_right_t = find_retarded_time_right(t);
  const auto t0 = t[0];
  tnsr::ii<DataType, 3> integral_term_t{t.size()};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      integral_term_t.get(i, j) =
          integrate_term(retarded_time_left_t, i, j, -1, t0) +
          integrate_term(retarded_time_right_t, i, j, 1, t0);
    }
  }
  return integral_term_t;
}

template <typename DataType>
Scalar<DataType>
BinaryWithGravitationalWavesVariables<DataType>::get_t_conformal_factor(
    DataType t) const {
  const auto distance_left_t = get_t_distance_left(t);
  const auto distance_right_t = get_t_distance_right(t);
  Scalar<DataType> conformal_factor_t{t.size()};

  // PN conformal factor
  /*
  const auto separation_t = get_t_separation(t);
  const auto momentum_left_t = get_t_momentum_left(t);
  const auto momentum_right_t = get_t_momentum_right(t);
  const DataType E_left_t =
      mass_left +
      get(dot_product(momentum_left_t, momentum_left_t)) /
          (2. * mass_left) -
      mass_left * mass_right / (2. * get(separation_t));
  const DataType E_right_t =
      mass_right +
      get(dot_product(momentum_right_t, momentum_right_t)) /
          (2. * mass_right) -
      mass_left * mass_right / (2. * get(separation_t));
  const auto pn_conformal_factor_t =
      1. + E_left_t / (2. * get(distance_left_t)) +
      E_right_t / (2. * get(distance_right_t));
  get(conformal_factor_t) = pn_conformal_factor_t;
  */
  // Horizon Penetrating conformal factor
  /*
  const auto areal_distance_left =
      find_areal_distance_left(get(distance_left_t));
  const auto areal_distance_right =
      find_areal_distance_right(get(distance_right_t));
  DataType conformal_factor_left{areal_distance_right.size()};
  DataType conformal_factor_right{areal_distance_right.size()};
  for (size_t i = 0; i < conformal_factor_left.size(); ++i) {
    conformal_factor_left[i] =
        sqrt(4. * areal_distance_left[i] /
             (2. * areal_distance_left[i] + mass_left +
              sqrt(4. * square(areal_distance_left[i]) +
                   4. * areal_distance_left[i] * mass_left +
                   3. * square(mass_left)))) *
        std::pow((8. * areal_distance_left[i] + 6. * mass_left +
                  3. * sqrt(8. * square(areal_distance_left[i]) +
                            8. * areal_distance_left[i] * mass_left +
                            6. * square(mass_left))) /
                     ((4. + 3. * sqrt(2.)) *
                      (2. * areal_distance_left[i] - 3. * mass_left)),
                 1. / (2. * sqrt(2.)));
    conformal_factor_right[i] =
        sqrt(4. * areal_distance_right[i] /
             (2. * areal_distance_right[i] + mass_right +
              sqrt(4. * square(areal_distance_right[i]) +
                   4. * areal_distance_right[i] * mass_right +
                   3. * square(mass_right)))) *
        std::pow((8. * areal_distance_right[i] + 6. * mass_right +
                  3. * sqrt(8. * square(areal_distance_right[i]) +
                            8. * areal_distance_right[i] * mass_right +
                            6. * square(mass_right))) /
                     ((4. + 3. * sqrt(2.)) *
                      (2. * areal_distance_right[i] - 3. * mass_right)),
                 1. / (2. * sqrt(2.)));
  }
  get(conformal_factor_t) =
      -1. + conformal_factor_left + conformal_factor_right;
  */
  // Boosted superposed Horizon Penetrating conformal factor
  get(conformal_factor_t) = 1.;
  return conformal_factor_t;
}

template <typename DataType>
Scalar<DataType> BinaryWithGravitationalWavesVariables<DataType>::get_t_lapse(
    DataType t) const {
  Scalar<DataType> lapse_t{t.size()};

  // PN Lapse
  /*
  const auto conformal_factor_t = get_t_conformal_factor(t);
  get(lapse_t) = (2. - get(conformal_factor_t)) /
  get(conformal_factor_t);
  */

  // Horizon Penetrating Lapse
  /*
  const auto distance_left_t = get_t_distance_left(t);
  const auto distance_right_t = get_t_distance_right(t);
  const auto areal_distance_left =
      find_areal_distance_left(get(distance_left_t));
  const auto areal_distance_right =
      find_areal_distance_right(get(distance_right_t));
  DataType lapse_left{areal_distance_right.size()};
  DataType lapse_right{areal_distance_right.size()};
  for (size_t i = 0; i < lapse_left.size(); ++i) {
    lapse_left[i] = sqrt(1. - 2. * mass_left / areal_distance_left[i] +
                         27. * square(square(mass_left)) /
                             (16. * square(square(areal_distance_left[i]))));
    lapse_right[i] = sqrt(1. - 2. * mass_right / areal_distance_right[i] +
                          27. * square(square(mass_right)) /
                              (16. * square(square(areal_distance_right[i]))));
  }
  get(lapse_t) = -1. + lapse_left + lapse_right;
  */
  // Boosted superposed Horizon Penetrating Lapse
  const auto superposed_spacetime_metric = get_t_superposed_spacetime_metric(t);
  const auto shift = get_t_shift(t);
  gr::lapse(make_not_null(&lapse_t), shift, superposed_spacetime_metric);

  return lapse_t;
}

template <typename DataType>
tnsr::I<DataType, 3>
BinaryWithGravitationalWavesVariables<DataType>::get_t_shift(DataType t) const {
  // tnsr::I<DataType, 3> shift_t{t.size()};
  // std::fill(shift_t.begin(), shift_t.end(), 0.);

  // PN shift
  /*
  DataType present_time(get_size(get<0>(x)), max_time_interpolator);
  const auto distance_left_t = get_t_distance_left(present_time);
  const auto distance_right_t = get_t_distance_right(present_time);
  const auto separation_t = get_t_separation(present_time);
  const auto momentum_left_t = get_t_momentum_left(present_time);
  const auto momentum_right_t = get_t_momentum_right(present_time);
  const auto normal_left_t = get_t_normal_left(present_time);
  const auto normal_right_t = get_t_normal_right(present_time);
  for (size_t i = 0; i < 3; ++i) {
    shift_t.get(i) -=
        4. * (momentum_left_t.get(i) / get(distance_left_t) +
              momentum_right_t.get(i) / get(distance_right_t));
    for (size_t j = 0; j < 3; ++j) {
      shift_t.get(i) +=
          .5 * momentum_left_t.get(j) *
              (-normal_left_t.get(i) * normal_left_t.get(j) /
               get(distance_left_t)) +
          .5 * momentum_right_t.get(j) *
              (-normal_right_t.get(i) * normal_right_t.get(j) /
               get(distance_right_t));
    }
    shift_t.get(i) +=
        0.5 * momentum_left_t.get(i) / get(distance_left_t) +
        0.5 * momentum_right_t.get(i) / get(distance_right_t);
  }
  */
  // Horizon Penetrating shift
  /*
  DataType present_time(get_size(get<0>(x)), max_time_interpolator);
  const auto distance_left_t = get_t_distance_left(present_time);
  const auto distance_right_t = get_t_distance_right(present_time);
  const auto areal_distance_left =
      find_areal_distance_left(get(distance_left_t));
  const auto areal_distance_right =
      find_areal_distance_right(get(distance_right_t));
  const auto normal_left = get_t_normal_left(present_time);
  const auto normal_right = get_t_normal_right(present_time);
  const DataType shift_r_left = .75 * sqrt(3) * square(mass_left) *
                                get(distance_left_t) /
                                pow(areal_distance_left, 3);
  const DataType shift_r_right = .75 * sqrt(3) * square(mass_right) *
                                 get(distance_right_t) /
                                 pow(areal_distance_right, 3);
  for (size_t i = 0; i < 3; ++i) {
    shift_t.get(i) +=
        shift_r_left * normal_left.get(i) + shift_r_right * normal_right.get(i);
  }
  */
  // Boosted superposed Horizon Penetrating Shift
  const auto superposed_spacetime_metric = get_t_superposed_spacetime_metric(t);
  const auto conformal_metric = get_t_conformal_metric(t);
  const auto inv_conformal_metric =
      determinant_and_inverse(conformal_metric).second;
  return gr::shift(superposed_spacetime_metric, inv_conformal_metric);
}

template <typename DataType>
Scalar<DataType>
BinaryWithGravitationalWavesVariables<DataType>::get_t_boosted_distance_left(
    const DataType t) const {
  // Boosted Coordinates
  tnsr::I<DataType, 3> v_boosted = x;
  for (size_t i = 0; i < t.size(); ++i) {
    for (size_t j = 0; j < 3; ++j) {
      v_boosted.get(j)[i] -=
          interpolation_position_left.at(j)(max_time_interpolator);
    }
  }

  // Horizon Penetrating Coordinates
  const auto momentum_left_t = get_t_momentum_left(t);
  tnsr::I<DataType, 3, Frame::NoFrame> boost_velocity{t.size()};
  for (size_t i = 0; i < 3; ++i) {
    boost_velocity.get(i) = -momentum_left_t.get(i) / mass_left;
  }
  const DataType velocity_squared{
      get(dot_product(boost_velocity, boost_velocity))};
  const DataType lorentz_factor{1.0 / sqrt(1.0 - velocity_squared)};
  DataType kinetic_energy_per_v_squared{square(lorentz_factor) /
                                        (1.0 + lorentz_factor)};
  tnsr::Ab<DataType, 3> boost_matrix{t.size()};
  boost_matrix.get(0, 0) = lorentz_factor;
  for (size_t i = 0; i < 3; ++i) {
    boost_matrix.get(0, i + 1) = boost_velocity.get(i) * lorentz_factor;
    boost_matrix.get(i + 1, 0) = boost_velocity.get(i) * lorentz_factor;
    for (size_t j = 0; j < 3; ++j) {
      boost_matrix.get(i + 1, j + 1) = boost_velocity.get(i) *
                                       boost_velocity.get(j) *
                                       kinetic_energy_per_v_squared;
    }
    boost_matrix.get(i + 1, i + 1) += 1.0;
  }

  tnsr::I<DataType, 3> v{t.size()};
  for (size_t i = 0; i < 3; ++i) {
    v.get(i) = boost_matrix.get(i + 1, 0) * t;
    for (size_t j = 0; j < 3; ++j) {
      v.get(i) += boost_matrix.get(i + 1, j + 1) * v_boosted.get(j);
    }
  }

  return magnitude(v);
}

template <typename DataType>
Scalar<DataType>
BinaryWithGravitationalWavesVariables<DataType>::get_t_boosted_distance_right(
    const DataType t) const {
  // Boosted Coordinates
  tnsr::I<DataType, 3> v_boosted = x;
  for (size_t i = 0; i < t.size(); ++i) {
    for (size_t j = 0; j < 3; ++j) {
      v_boosted.get(j)[i] -=
          interpolation_position_right.at(j)(max_time_interpolator);
    }
  }

  // Horizon Penetrating Coordinates (boost with negative velocity)
  const auto momentum_right_t = get_t_momentum_right(t);
  tnsr::I<DataType, 3, Frame::NoFrame> boost_velocity{t.size()};
  for (size_t i = 0; i < 3; ++i) {
    boost_velocity.get(i) = -momentum_right_t.get(i) / mass_right;
  }
  const DataType velocity_squared{
      get(dot_product(boost_velocity, boost_velocity))};
  const DataType lorentz_factor{1.0 / sqrt(1.0 - velocity_squared)};
  DataType kinetic_energy_per_v_squared{square(lorentz_factor) /
                                        (1.0 + lorentz_factor)};
  tnsr::Ab<DataType, 3> boost_matrix{t.size()};
  boost_matrix.get(0, 0) = lorentz_factor;
  for (size_t i = 0; i < 3; ++i) {
    boost_matrix.get(0, i + 1) = boost_velocity.get(i) * lorentz_factor;
    boost_matrix.get(i + 1, 0) = boost_velocity.get(i) * lorentz_factor;
    for (size_t j = 0; j < 3; ++j) {
      boost_matrix.get(i + 1, j + 1) = boost_velocity.get(i) *
                                       boost_velocity.get(j) *
                                       kinetic_energy_per_v_squared;
    }
    boost_matrix.get(i + 1, i + 1) += 1.0;
  }

  tnsr::I<DataType, 3> v{t.size()};
  for (size_t i = 0; i < 3; ++i) {
    v.get(i) = boost_matrix.get(i + 1, 0) * t;
    for (size_t j = 0; j < 3; ++j) {
      v.get(i) += boost_matrix.get(i + 1, j + 1) * v_boosted.get(j);
    }
  }

  return magnitude(v);
}

template <typename DataType>
tnsr::I<DataType, 3>
BinaryWithGravitationalWavesVariables<DataType>::get_t_boosted_normal_left(
    const DataType t) const {
  // Boosted Coordinates
  tnsr::I<DataType, 3> v_boosted = x;
  for (size_t i = 0; i < t.size(); ++i) {
    for (size_t j = 0; j < 3; ++j) {
      v_boosted.get(j)[i] -=
          interpolation_position_left.at(j)(max_time_interpolator);
    }
  }

  // Horizon Penetrating Coordinates
  const auto momentum_left_t = get_t_momentum_left(t);
  tnsr::I<DataType, 3, Frame::NoFrame> boost_velocity{t.size()};
  for (size_t i = 0; i < 3; ++i) {
    boost_velocity.get(i) = -momentum_left_t.get(i) / mass_left;
  }
  const DataType velocity_squared{
      get(dot_product(boost_velocity, boost_velocity))};
  const DataType lorentz_factor{1.0 / sqrt(1.0 - velocity_squared)};
  DataType kinetic_energy_per_v_squared{square(lorentz_factor) /
                                        (1.0 + lorentz_factor)};
  tnsr::Ab<DataType, 3> boost_matrix{t.size()};
  boost_matrix.get(0, 0) = lorentz_factor;
  for (size_t i = 0; i < 3; ++i) {
    boost_matrix.get(0, i + 1) = boost_velocity.get(i) * lorentz_factor;
    boost_matrix.get(i + 1, 0) = boost_velocity.get(i) * lorentz_factor;
    for (size_t j = 0; j < 3; ++j) {
      boost_matrix.get(i + 1, j + 1) = boost_velocity.get(i) *
                                       boost_velocity.get(j) *
                                       kinetic_energy_per_v_squared;
    }
    boost_matrix.get(i + 1, i + 1) += 1.0;
  }

  tnsr::I<DataType, 3> v{t.size()};
  const auto distance_left_t = get_t_boosted_distance_left(t);
  for (size_t i = 0; i < 3; ++i) {
    v.get(i) = boost_matrix.get(i + 1, 0) * t;
    for (size_t j = 0; j < 3; ++j) {
      v.get(i) += boost_matrix.get(i + 1, j + 1) * v_boosted.get(j) /
                  get(distance_left_t);
    }
  }

  return v;
}

template <typename DataType>
tnsr::I<DataType, 3>
BinaryWithGravitationalWavesVariables<DataType>::get_t_boosted_normal_right(
    const DataType t) const {
  // Boosted Coordinates
  tnsr::I<DataType, 3> v_boosted = x;
  for (size_t i = 0; i < t.size(); ++i) {
    for (size_t j = 0; j < 3; ++j) {
      v_boosted.get(j)[i] -=
          interpolation_position_right.at(j)(max_time_interpolator);
    }
  }

  // Horizon Penetrating Coordinates (boost with negative velocity)
  const auto momentum_right_t = get_t_momentum_right(t);
  tnsr::I<DataType, 3, Frame::NoFrame> boost_velocity{t.size()};
  for (size_t i = 0; i < 3; ++i) {
    boost_velocity.get(i) = -momentum_right_t.get(i) / mass_right;
  }
  const DataType velocity_squared{
      get(dot_product(boost_velocity, boost_velocity))};
  const DataType lorentz_factor{1.0 / sqrt(1.0 - velocity_squared)};
  DataType kinetic_energy_per_v_squared{square(lorentz_factor) /
                                        (1.0 + lorentz_factor)};
  tnsr::Ab<DataType, 3> boost_matrix{t.size()};
  boost_matrix.get(0, 0) = lorentz_factor;
  for (size_t i = 0; i < 3; ++i) {
    boost_matrix.get(0, i + 1) = boost_velocity.get(i) * lorentz_factor;
    boost_matrix.get(i + 1, 0) = boost_velocity.get(i) * lorentz_factor;
    for (size_t j = 0; j < 3; ++j) {
      boost_matrix.get(i + 1, j + 1) = boost_velocity.get(i) *
                                       boost_velocity.get(j) *
                                       kinetic_energy_per_v_squared;
    }
    boost_matrix.get(i + 1, i + 1) += 1.0;
  }

  tnsr::I<DataType, 3> v{t.size()};
  const auto distance_right_t = get_t_boosted_distance_right(t);
  for (size_t i = 0; i < 3; ++i) {
    v.get(i) = boost_matrix.get(i + 1, 0) * t;
    for (size_t j = 0; j < 3; ++j) {
      v.get(i) += boost_matrix.get(i + 1, j + 1) * v_boosted.get(j) /
                  get(distance_right_t);
    }
  }

  return v;
}

template <typename DataType>
tnsr::aa<DataType, 3> BinaryWithGravitationalWavesVariables<
    DataType>::get_t_boosted_spacetime_metric_left(DataType t) const {
  const auto distance_left_t = get_t_boosted_distance_left(t);
  const auto areal_distance_left =
      find_areal_distance_left(get(distance_left_t));
  const auto momentum_left_t = get_t_momentum_left(t);
  const auto normal_left_t = get_t_boosted_normal_left(t);

  // Schwarzschild Horizon Penetrating lapse
  Scalar<DataType> lapse{t.size()};
  get(lapse) = sqrt(1. - 2. * mass_left / areal_distance_left +
                    27. * square(square(mass_left)) /
                        (16. * square(square(areal_distance_left))));

  // Schwarzschild Horizon Penetrating shift
  tnsr::I<DataType, 3> shift{t.size()};
  for (size_t i = 0; i < 3; ++i) {
    shift.get(i) = .75 * sqrt(3) * square(mass_left) * get(distance_left_t) /
                   pow(areal_distance_left, 3) * normal_left_t.get(i);
  }
  // Schwarzschild Horizon Penetrating spatial metric
  tnsr::ii<DataType, 3> spatial_metric{t.size()};
  std::fill(spatial_metric.begin(), spatial_metric.end(), 0.);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t k = 0; k < areal_distance_left.size(); ++k) {
      spatial_metric.get(i, i)[k] +=
          pow(sqrt(4. * areal_distance_left[k] /
                   (2. * areal_distance_left[k] + mass_left +
                    sqrt(4. * square(areal_distance_left[k]) +
                         4. * areal_distance_left[k] * mass_left +
                         3. * square(mass_left)))) *
                  std::pow((8. * areal_distance_left[k] + 6. * mass_left +
                            3. * sqrt(8. * square(areal_distance_left[k]) +
                                      8. * areal_distance_left[k] * mass_left +
                                      6. * square(mass_left))) /
                               ((4. + 3. * sqrt(2.)) *
                                (2. * areal_distance_left[k] - 3. * mass_left)),
                           1. / (2. * sqrt(2.))),
              4);
    }
  }
  tnsr::aa<DataType, 3> spacetime_metric{t.size()};
  gr::spacetime_metric(make_not_null(&spacetime_metric), lapse, shift,
                       spatial_metric);

  tnsr::I<DataType, 3, Frame::NoFrame> boost_velocity{t.size()};
  for (size_t i = 0; i < 3; ++i) {
    boost_velocity.get(i) = -momentum_left_t.get(i) / mass_left;
  }
  const DataType velocity_squared{
      get(dot_product(boost_velocity, boost_velocity))};
  const DataType lorentz_factor{1.0 / sqrt(1.0 - velocity_squared)};
  DataType kinetic_energy_per_v_squared{square(lorentz_factor) /
                                        (1.0 + lorentz_factor)};
  tnsr::Ab<DataType, 3> boost_matrix{t.size()};
  boost_matrix.get(0, 0) = lorentz_factor;
  for (size_t i = 0; i < 3; ++i) {
    boost_matrix.get(0, i + 1) = boost_velocity.get(i) * lorentz_factor;
    boost_matrix.get(i + 1, 0) = boost_velocity.get(i) * lorentz_factor;
    for (size_t j = 0; j < 3; ++j) {
      boost_matrix.get(i + 1, j + 1) = boost_velocity.get(i) *
                                       boost_velocity.get(j) *
                                       kinetic_energy_per_v_squared;
    }
    boost_matrix.get(i + 1, i + 1) += 1.0;
  }

  tnsr::aa<DataType, 3> boosted_spacetime_metric{t.size()};
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      boosted_spacetime_metric.get(i, j) = 0.;
      for (size_t k = 0; k < 4; ++k) {
        for (size_t l = 0; l < 4; ++l) {
          boosted_spacetime_metric.get(i, j) += boost_matrix.get(k, i) *
                                                boost_matrix.get(l, j) *
                                                spacetime_metric.get(k, l);
        }
      }
    }
  }

  return boosted_spacetime_metric;
}

template <typename DataType>
tnsr::aa<DataType, 3> BinaryWithGravitationalWavesVariables<
    DataType>::get_t_boosted_spacetime_metric_right(DataType t) const {
  const auto distance_right_t = get_t_boosted_distance_right(t);
  const auto areal_distance_right =
      find_areal_distance_right(get(distance_right_t));
  const auto momentum_right_t = get_t_momentum_right(t);
  const auto normal_right_t = get_t_boosted_normal_right(t);

  // Schwarzschild Horizon Penetrating lapse
  Scalar<DataType> lapse{t.size()};
  get(lapse) = sqrt(1. - 2. * mass_right / areal_distance_right +
                    27. * square(square(mass_right)) /
                        (16. * square(square(areal_distance_right))));

  // Schwarzschild Horizon Penetrating shift
  tnsr::I<DataType, 3> shift{t.size()};
  for (size_t i = 0; i < 3; ++i) {
    shift.get(i) = .75 * sqrt(3) * square(mass_right) * get(distance_right_t) /
                   pow(areal_distance_right, 3) * normal_right_t.get(i);
  }
  // Schwarzschild Horizon Penetrating spatial metric
  tnsr::ii<DataType, 3> spatial_metric{t.size()};
  std::fill(spatial_metric.begin(), spatial_metric.end(), 0.);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t k = 0; k < areal_distance_right.size(); ++k) {
      spatial_metric.get(i, i)[k] += pow(
          sqrt(4. * areal_distance_right[k] /
               (2. * areal_distance_right[k] + mass_right +
                sqrt(4. * square(areal_distance_right[k]) +
                     4. * areal_distance_right[k] * mass_right +
                     3. * square(mass_right)))) *
              std::pow((8. * areal_distance_right[k] + 6. * mass_right +
                        3. * sqrt(8. * square(areal_distance_right[k]) +
                                  8. * areal_distance_right[k] * mass_right +
                                  6. * square(mass_right))) /
                           ((4. + 3. * sqrt(2.)) *
                            (2. * areal_distance_right[k] - 3. * mass_right)),
                       1. / (2. * sqrt(2.))),
          4);
    }
  }
  tnsr::aa<DataType, 3> spacetime_metric{t.size()};
  gr::spacetime_metric(make_not_null(&spacetime_metric), lapse, shift,
                       spatial_metric);

  tnsr::I<DataType, 3, Frame::NoFrame> boost_velocity{t.size()};
  for (size_t i = 0; i < 3; ++i) {
    boost_velocity.get(i) = -momentum_right_t.get(i) / mass_right;
  }
  const DataType velocity_squared{
      get(dot_product(boost_velocity, boost_velocity))};
  const DataType lorentz_factor{1.0 / sqrt(1.0 - velocity_squared)};
  DataType kinetic_energy_per_v_squared{square(lorentz_factor) /
                                        (1.0 + lorentz_factor)};
  tnsr::Ab<DataType, 3> boost_matrix{t.size()};
  boost_matrix.get(0, 0) = lorentz_factor;
  for (size_t i = 0; i < 3; ++i) {
    boost_matrix.get(0, i + 1) = boost_velocity.get(i) * lorentz_factor;
    boost_matrix.get(i + 1, 0) = boost_velocity.get(i) * lorentz_factor;
    for (size_t j = 0; j < 3; ++j) {
      boost_matrix.get(i + 1, j + 1) = boost_velocity.get(i) *
                                       boost_velocity.get(j) *
                                       kinetic_energy_per_v_squared;
    }
    boost_matrix.get(i + 1, i + 1) += 1.0;
  }

  tnsr::aa<DataType, 3> boosted_spacetime_metric{t.size()};
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      boosted_spacetime_metric.get(i, j) = 0.;
      for (size_t k = 0; k < 4; ++k) {
        for (size_t l = 0; l < 4; ++l) {
          boosted_spacetime_metric.get(i, j) += boost_matrix.get(k, i) *
                                                boost_matrix.get(l, j) *
                                                spacetime_metric.get(k, l);
        }
      }
    }
  }

  return boosted_spacetime_metric;
}

template <typename DataType>
tnsr::aa<DataType, 3> BinaryWithGravitationalWavesVariables<
    DataType>::get_t_superposed_spacetime_metric(DataType t) const {
  const auto spacetime_metric_left = get_t_boosted_spacetime_metric_left(t);
  const auto spacetime_metric_right = get_t_boosted_spacetime_metric_right(t);
  tnsr::aa<DataType, 3> superposed_spacetime_metric{t.size()};
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      superposed_spacetime_metric.get(i, j) =
          spacetime_metric_left.get(i, j) + spacetime_metric_right.get(i, j);
    }
    superposed_spacetime_metric.get(i, i) -= 1.;
  }

  // Lapse correction
  const auto spatial_metric_left = gr::spatial_metric(spacetime_metric_left);
  const auto inv_spatial_metric_left =
      determinant_and_inverse(spatial_metric_left).second;
  const auto shift_left =
      gr::shift(spacetime_metric_left, inv_spatial_metric_left);
  const auto shift_left_down =
      raise_or_lower_index(shift_left, spatial_metric_left);
  const auto spatial_metric_right = gr::spatial_metric(spacetime_metric_right);
  const auto inv_spatial_metric_right =
      determinant_and_inverse(spatial_metric_right).second;
  const auto shift_right =
      gr::shift(spacetime_metric_right, inv_spatial_metric_right);
  const auto shift_right_down =
      raise_or_lower_index(shift_right, spatial_metric_right);

  superposed_spacetime_metric.get(0, 0) +=
      2. + get(dot_product(shift_left, shift_right_down)) +
      get(dot_product(shift_right, shift_left_down));

  return superposed_spacetime_metric;
}

template class BinaryWithGravitationalWavesVariables<DataVector>;

}  // namespace detail


void BinaryWithGravitationalWaves::reserve_vector_capacity() {
  for (size_t i = 0; i < 3; ++i) {
    past_position_left_.at(i).reserve(number_of_steps);
    past_position_right_.at(i).reserve(number_of_steps);
    past_momentum_left_.at(i).reserve(number_of_steps);
    past_momentum_right_.at(i).reserve(number_of_steps);
    past_dt_position_left_.at(i).reserve(number_of_steps);
    past_dt_position_right_.at(i).reserve(number_of_steps);
    past_dt_momentum_left_.at(i).reserve(number_of_steps);
    past_dt_momentum_right_.at(i).reserve(number_of_steps);
  }
  past_time_.reserve(number_of_steps);
}

void BinaryWithGravitationalWaves::reverse_vector() {
  std::reverse(past_time_.begin(), past_time_.end());
  for (size_t i = 0; i < 3; ++i) {
    reverse(past_position_left_.at(i).begin(), past_position_left_.at(i).end());
    reverse(past_position_right_.at(i).begin(),
            past_position_right_.at(i).end());
    reverse(past_momentum_left_.at(i).begin(), past_momentum_left_.at(i).end());
    reverse(past_momentum_right_.at(i).begin(),
            past_momentum_right_.at(i).end());
    reverse(past_dt_position_left_.at(i).begin(),
            past_dt_position_left_.at(i).end());
    reverse(past_dt_position_right_.at(i).begin(),
            past_dt_position_right_.at(i).end());
    reverse(past_dt_momentum_left_.at(i).begin(),
            past_dt_momentum_left_.at(i).end());
    reverse(past_dt_momentum_right_.at(i).begin(),
            past_dt_momentum_right_.at(i).end());
  }
}

void BinaryWithGravitationalWaves::initialize() {
  double separation = xcoord_right() - xcoord_left();

  total_mass = mass_left() + mass_right();
  reduced_mass = mass_left() * mass_right() / total_mass;
  reduced_mass_over_total_mass = reduced_mass / total_mass;

  double p_circular_squared =
      reduced_mass * reduced_mass * total_mass / separation +
      4. * reduced_mass * reduced_mass * total_mass * total_mass /
          (separation * separation) +
      (74. - 43. * reduced_mass / total_mass) * reduced_mass * reduced_mass *
          total_mass * total_mass * total_mass /
          (8. * separation * separation * separation);
  ymomentum_left_ = sqrt(p_circular_squared);
  ymomentum_right_ = -sqrt(p_circular_squared);

  initial_state_position = {{separation / total_mass, 0., 0.}};
  initial_state_momentum = {{0., ymomentum_right_ / reduced_mass, 0.}};

  time_step = .1;
  initial_time = 0.;
  final_time = std::round(-2 * outer_radius() / time_step) * time_step;
  number_of_steps =
      static_cast<size_t>(std::round((initial_time - final_time) / time_step));
}

void BinaryWithGravitationalWaves::hamiltonian_system(
    const BinaryWithGravitationalWaves::state_type& x,
    BinaryWithGravitationalWaves::state_type& dpdt) {
  // H = H_Newt + H_1PN + H_2PN + H_3PN

  double pdotp = x[3] * x[3] + x[4] * x[4] + x[5] * x[5];
  double qdotq = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
  double qdotp = x[0] * x[3] + x[1] * x[4] + x[2] * x[5];

  double dH_dp0_Newt = x[3];
  double dH_dp0_1 =
      0.5 * x[3] * pdotp * (-1 + 3 * reduced_mass_over_total_mass) -
      (x[0] * (x[4] * x[1] + x[5] * x[2]) * reduced_mass_over_total_mass +
       x[3] * (x[1] * x[1] + x[2] * x[2]) * (3 + reduced_mass_over_total_mass) +
       x[3] * x[0] * x[0] * (3 + 2 * reduced_mass_over_total_mass)) /
          std::sqrt(qdotq * qdotq * qdotq);
  double dH_dp0_2 =
      0.125 *
      (3 * x[3] * pdotp * pdotp *
           (1 - 5 * reduced_mass_over_total_mass +
            5 * reduced_mass_over_total_mass * reduced_mass_over_total_mass) +
       (8 * (3 * x[0] * qdotp * reduced_mass_over_total_mass +
             x[3] * qdotq * (5 + 8 * reduced_mass_over_total_mass))) /
           (qdotq * qdotq) +
       (1 / std::sqrt(qdotq)) *
           (-(12 * x[0] * qdotp * qdotp * qdotp * reduced_mass_over_total_mass *
              reduced_mass_over_total_mass) /
                (qdotq * qdotq) -
            (4 * pdotp * x[0] * qdotp * reduced_mass_over_total_mass *
             reduced_mass_over_total_mass) /
                qdotq -
            (4 * x[3] * qdotp * qdotp * reduced_mass_over_total_mass *
             reduced_mass_over_total_mass) /
                qdotq -
            4 * x[3] * pdotp *
                (-5 + 20 * reduced_mass_over_total_mass +
                 3 * reduced_mass_over_total_mass *
                     reduced_mass_over_total_mass)));
  double dH_dp0_3 =
      0.0625 *
      (5 * x[3] * pdotp * pdotp * pdotp *
           (-1 + 7 * (-1 + reduced_mass_over_total_mass) *
                     (-1 + reduced_mass_over_total_mass) *
                     reduced_mass_over_total_mass) +
       1.0 / (3 * qdotq * qdotq * qdotq) * 2 *
           (3 * pdotp * x[0] * qdotp * qdotq * reduced_mass_over_total_mass *
                (17 + 30 * reduced_mass_over_total_mass) +
            3 * x[3] * qdotp * qdotp * qdotq * reduced_mass_over_total_mass *
                (17 + 30 * reduced_mass_over_total_mass) +
            8 * x[0] * qdotp * qdotp * qdotp * reduced_mass_over_total_mass *
                (5 + 43 * reduced_mass_over_total_mass) +
            6 * x[3] * pdotp * qdotq * qdotq *
                (-27 + reduced_mass_over_total_mass *
                           (136 + 109 * reduced_mass_over_total_mass))) -
       (3 * x[0] * qdotp * reduced_mass_over_total_mass *
            (340 + 3 * M_PI * M_PI + 112 * reduced_mass_over_total_mass) +
        x[3] * qdotq *
            (600 + reduced_mass_over_total_mass *
                       (1340 - 3 * M_PI * M_PI +
                        552 * reduced_mass_over_total_mass))) /
           (6 * sqrt(qdotq * qdotq * qdotq * qdotq * qdotq)) +
       2 / (sqrt(qdotq * qdotq * qdotq * qdotq * qdotq * qdotq * qdotq)) *
           (pdotp * pdotp * x[0] * qdotp * qdotq * qdotq *
                (2 - 3 * reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass +
            2 * x[3] * pdotp * qdotp * qdotp * qdotq * qdotq *
                (2 - 3 * reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            6 * pdotp * x[0] * qdotp * qdotp * qdotp * qdotq *
                (-1 + reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            3 * x[3] * qdotp * qdotp * qdotp * qdotp * qdotq *
                (-1 + reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            15 * x[0] * qdotp * qdotp * qdotp * qdotp * qdotp *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass *
                reduced_mass_over_total_mass -
            3 * x[3] * pdotp * pdotp * qdotq * qdotq * qdotq *
                (7 + reduced_mass_over_total_mass *
                         (-42 + reduced_mass_over_total_mass *
                                    (53 + 5 * reduced_mass_over_total_mass)))));

  double dH_dp1_Newt = x[4];
  double dH_dp1_1 =
      0.5 * x[4] * pdotp * (-1 + 3 * reduced_mass_over_total_mass) -
      (x[1] * (x[3] * x[0] + x[5] * x[2]) * reduced_mass_over_total_mass +
       x[4] * (x[0] * x[0] + x[2] * x[2]) * (3 + reduced_mass_over_total_mass) +
       x[4] * x[1] * x[1] * (3 + 2 * reduced_mass_over_total_mass)) /
          std::sqrt(qdotq * qdotq * qdotq);
  double dH_dp1_2 =
      0.125 *
      (3 * x[4] * pdotp * pdotp *
           (1 - 5 * reduced_mass_over_total_mass +
            5 * reduced_mass_over_total_mass * reduced_mass_over_total_mass) +
       (8 * (3 * x[1] * qdotp * reduced_mass_over_total_mass +
             x[4] * qdotq * (5 + 8 * reduced_mass_over_total_mass))) /
           (qdotq * qdotq) +
       (1 / std::sqrt(qdotq)) *
           (-(12 * x[1] * qdotp * qdotp * qdotp * reduced_mass_over_total_mass *
              reduced_mass_over_total_mass) /
                (qdotq * qdotq) -
            (4 * pdotp * x[1] * qdotp * reduced_mass_over_total_mass *
             reduced_mass_over_total_mass) /
                qdotq -
            (8 * x[4] * qdotp * qdotp * reduced_mass_over_total_mass *
             reduced_mass_over_total_mass) /
                qdotq -
            4 * x[4] * pdotp *
                (-5 + 20 * reduced_mass_over_total_mass +
                 3 * reduced_mass_over_total_mass *
                     reduced_mass_over_total_mass)));
  double dH_dp1_3 =
      0.0625 *
      (5 * x[4] * pdotp * pdotp * pdotp *
           (-1 + 7 * (-1 + reduced_mass_over_total_mass) *
                     (-1 + reduced_mass_over_total_mass) *
                     reduced_mass_over_total_mass) +
       1.0 / (3 * qdotq * qdotq * qdotq) * 2 *
           (3 * pdotp * x[1] * qdotp * qdotq * reduced_mass_over_total_mass *
                (17 + 30 * reduced_mass_over_total_mass) +
            3 * x[4] * qdotp * qdotp * qdotq * reduced_mass_over_total_mass *
                (17 + 30 * reduced_mass_over_total_mass) +
            8 * x[1] * qdotp * qdotp * qdotp * reduced_mass_over_total_mass *
                (5 + 43 * reduced_mass_over_total_mass) +
            6 * x[4] * pdotp * qdotq * qdotq *
                (-27 + reduced_mass_over_total_mass *
                           (136 + 109 * reduced_mass_over_total_mass))) -
       (3 * x[1] * qdotp * reduced_mass_over_total_mass *
            (340 + 3 * M_PI * M_PI + 112 * reduced_mass_over_total_mass) +
        x[4] * qdotq *
            (600 + reduced_mass_over_total_mass *
                       (1340 - 3 * M_PI * M_PI +
                        552 * reduced_mass_over_total_mass))) /
           (6 * sqrt(qdotq * qdotq * qdotq * qdotq * qdotq)) +
       2 / (sqrt(qdotq * qdotq * qdotq * qdotq * qdotq * qdotq * qdotq)) *
           (pdotp * pdotp * x[1] * qdotp * qdotq * qdotq *
                (2 - 3 * reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass +
            2 * x[4] * pdotp * qdotp * qdotp * qdotq * qdotq *
                (2 - 3 * reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            6 * pdotp * x[1] * qdotp * qdotp * qdotp * qdotq *
                (-1 + reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            3 * x[4] * qdotp * qdotp * qdotp * qdotp * qdotq *
                (-1 + reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            15 * x[1] * qdotp * qdotp * qdotp * qdotp * qdotp *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass *
                reduced_mass_over_total_mass -
            3 * x[4] * pdotp * pdotp * qdotq * qdotq * qdotq *
                (7 + reduced_mass_over_total_mass *
                         (-42 + reduced_mass_over_total_mass *
                                    (53 + 5 * reduced_mass_over_total_mass)))));

  double dH_dp2_Newt = x[5];
  double dH_dp2_1 =
      0.5 * x[5] * pdotp * (-1 + 3 * reduced_mass_over_total_mass) -
      (x[2] * (x[4] * x[1] + x[3] * x[0]) * reduced_mass_over_total_mass +
       x[5] * (x[0] * x[0] + x[1] * x[1]) * (3 + reduced_mass_over_total_mass) +
       x[5] * x[2] * x[2] * (3 + 2 * reduced_mass_over_total_mass)) /
          std::sqrt(qdotq * qdotq * qdotq);
  double dH_dp2_2 =
      0.125 *
      (3 * x[5] * pdotp * pdotp *
           (1 - 5 * reduced_mass_over_total_mass +
            5 * reduced_mass_over_total_mass * reduced_mass_over_total_mass) +
       (8 * (3 * x[2] * qdotp * reduced_mass_over_total_mass +
             x[5] * qdotq * (5 + 8 * reduced_mass_over_total_mass))) /
           (qdotq * qdotq) +
       (1 / std::sqrt(qdotq)) *
           (-(12 * x[2] * qdotp * qdotp * qdotp * reduced_mass_over_total_mass *
              reduced_mass_over_total_mass) /
                (qdotq * qdotq) -
            (4 * pdotp * x[2] * qdotp * reduced_mass_over_total_mass *
             reduced_mass_over_total_mass) /
                qdotq -
            (8 * x[5] * qdotp * qdotp * reduced_mass_over_total_mass *
             reduced_mass_over_total_mass) /
                qdotq -
            4 * x[5] * pdotp *
                (-5 + 20 * reduced_mass_over_total_mass +
                 3 * reduced_mass_over_total_mass *
                     reduced_mass_over_total_mass)));
  double dH_dp2_3 =
      0.0625 *
      (5 * x[5] * pdotp * pdotp * pdotp *
           (-1 + 7 * (-1 + reduced_mass_over_total_mass) *
                     (-1 + reduced_mass_over_total_mass) *
                     reduced_mass_over_total_mass) +
       1.0 / (3 * qdotq * qdotq * qdotq) * 2 *
           (3 * pdotp * x[2] * qdotp * qdotq * reduced_mass_over_total_mass *
                (17 + 30 * reduced_mass_over_total_mass) +
            3 * x[5] * qdotp * qdotp * qdotq * reduced_mass_over_total_mass *
                (17 + 30 * reduced_mass_over_total_mass) +
            8 * x[2] * qdotp * qdotp * qdotp * reduced_mass_over_total_mass *
                (5 + 43 * reduced_mass_over_total_mass) +
            6 * x[5] * pdotp * qdotq * qdotq *
                (-27 + reduced_mass_over_total_mass *
                           (136 + 109 * reduced_mass_over_total_mass))) -
       (3 * x[2] * qdotp * reduced_mass_over_total_mass *
            (340 + 3 * M_PI * M_PI + 112 * reduced_mass_over_total_mass) +
        x[5] * qdotq *
            (600 + reduced_mass_over_total_mass *
                       (1340 - 3 * M_PI * M_PI +
                        552 * reduced_mass_over_total_mass))) /
           (6 * sqrt(qdotq * qdotq * qdotq * qdotq * qdotq)) +
       2 / (sqrt(qdotq * qdotq * qdotq * qdotq * qdotq * qdotq * qdotq)) *
           (pdotp * pdotp * x[2] * qdotp * qdotq * qdotq *
                (2 - 3 * reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass +
            2 * x[5] * pdotp * qdotp * qdotp * qdotq * qdotq *
                (2 - 3 * reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            6 * pdotp * x[2] * qdotp * qdotp * qdotp * qdotq *
                (-1 + reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            3 * x[5] * qdotp * qdotp * qdotp * qdotp * qdotq *
                (-1 + reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            15 * x[2] * qdotp * qdotp * qdotp * qdotp * qdotp *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass *
                reduced_mass_over_total_mass -
            3 * x[5] * pdotp * pdotp * qdotq * qdotq * qdotq *
                (7 + reduced_mass_over_total_mass *
                         (-42 + reduced_mass_over_total_mass *
                                    (53 + 5 * reduced_mass_over_total_mass)))));

  double dH_dq0_Newt = x[0] / std::sqrt(qdotq * qdotq * qdotq);
  double dH_dq0_1 =
      (-2 * x[0] * std::sqrt(qdotq) +
       3 * x[0] * qdotp * qdotp * reduced_mass_over_total_mass -
       2 * x[3] * qdotp * qdotq * reduced_mass_over_total_mass +
       pdotp * x[0] * qdotq * (3 + reduced_mass_over_total_mass)) /
      (2 * std::sqrt(qdotq * qdotq * qdotq * qdotq * qdotq));
  double dH_dq0_2 =
      (-48 * x[0] * qdotp * qdotp * std::sqrt(qdotq) *
           reduced_mass_over_total_mass +
       24 * x[3] * qdotp * std::sqrt(qdotq * qdotq * qdotq) *
           reduced_mass_over_total_mass +
       15 * x[0] * qdotp * qdotp * qdotp * qdotp *
           reduced_mass_over_total_mass * reduced_mass_over_total_mass +
       6 * pdotp * x[0] * qdotp * qdotp * qdotq * reduced_mass_over_total_mass *
           reduced_mass_over_total_mass -
       12 * x[3] * qdotp * qdotp * qdotp * qdotq *
           reduced_mass_over_total_mass * reduced_mass_over_total_mass -
       4 * x[3] * pdotp * qdotp * qdotq * qdotq * reduced_mass_over_total_mass *
           reduced_mass_over_total_mass +
       6 * x[0] * qdotq * (1 + 3 * reduced_mass_over_total_mass) -
       8 * pdotp * x[0] * std::sqrt(qdotq * qdotq * qdotq) *
           (5 + 8 * reduced_mass_over_total_mass) +
       pdotp * pdotp * x[0] * qdotq * qdotq *
           (-5 + reduced_mass_over_total_mass *
                     (20 + 3 * reduced_mass_over_total_mass))) /
      (8 * std::sqrt(qdotq * qdotq * qdotq * qdotq * qdotq * qdotq * qdotq));
  double dH_dq0_3 =
      (3 / 2 * qdotp * qdotq *
           (x[0] * (x[1] * x[4] + x[2] * x[5]) -
            x[3] * (x[1] * x[1] + x[2] * x[2])) *
           reduced_mass_over_total_mass *
           (340 + 3 * M_PI * M_PI + 112 * reduced_mass_over_total_mass) +
       2 * x[0] * std::sqrt(qdotq * qdotq * qdotq) *
           (-12 + (-872 + 63 * M_PI * M_PI) * reduced_mass_over_total_mass) -
       6 * qdotp * reduced_mass_over_total_mass * reduced_mass_over_total_mass *
           (pdotp * pdotp * x[0] * qdotp * qdotq * qdotq *
                (2 - 3 * reduced_mass_over_total_mass) -
            6 * pdotp * x[0] * qdotp * qdotp * qdotp * qdotq *
                (-1 + reduced_mass_over_total_mass) +
            6 * x[3] * pdotp * qdotp * qdotp * qdotq * qdotq *
                (-1 + reduced_mass_over_total_mass) -
            15 * x[0] * qdotp * qdotp * qdotp * qdotp * qdotp *
                reduced_mass_over_total_mass +
            15 * x[3] * qdotp * qdotp * qdotp * qdotp * qdotq *
                reduced_mass_over_total_mass +
            x[3] * pdotp * pdotp * qdotq * qdotq * qdotq *
                (-2 + 3 * reduced_mass_over_total_mass)) -
       2 * qdotp * std::sqrt(qdotq) * reduced_mass_over_total_mass *
           (3 * pdotp * x[0] * qdotp * qdotq *
                (17 + 30 * reduced_mass_over_total_mass) -
            3 * x[3] * pdotp * qdotq * qdotq *
                (17 + 30 * reduced_mass_over_total_mass) +
            8 * x[0] * qdotp * qdotp * qdotp *
                (5 + 43 * reduced_mass_over_total_mass) -
            8 * x[3] * qdotp * qdotp * qdotq *
                (5 + 43 * reduced_mass_over_total_mass)) -
       2 * x[0] * std::sqrt(qdotq) *
           (3 * pdotp * qdotp * qdotp * qdotq * reduced_mass_over_total_mass *
                (17 + 30 * reduced_mass_over_total_mass) +
            4 * qdotp * qdotp * qdotp * qdotp * reduced_mass_over_total_mass *
                (5 + 43 * reduced_mass_over_total_mass) +
            3 * pdotp * pdotp * qdotq * qdotq *
                (-27 + reduced_mass_over_total_mass *
                           (136 + 109 * reduced_mass_over_total_mass))) +
       0.75 * x[0] * qdotq *
           (3 * qdotp * qdotp * reduced_mass_over_total_mass *
                (340 + 3 * M_PI * M_PI + 112 * reduced_mass_over_total_mass) +
            pdotp * qdotq *
                (600 + reduced_mass_over_total_mass *
                           (1340 - 3 * M_PI * M_PI +
                            552 * reduced_mass_over_total_mass))) -
       3 * x[0] *
           (pdotp * pdotp * qdotp * qdotp * qdotq * qdotq *
                (2 - 3 * reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            3 * pdotp * qdotp * qdotp * qdotp * qdotp * qdotq *
                (-1 + reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            5 * qdotp * qdotp * qdotp * qdotp * qdotp * qdotp *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass *
                reduced_mass_over_total_mass -
            pdotp * pdotp * pdotp * qdotq * qdotq * qdotq *
                (7 +
                 reduced_mass_over_total_mass *
                     (-42 + reduced_mass_over_total_mass *
                                (53 + 5 * reduced_mass_over_total_mass))))) /
      (48 * std::sqrt(qdotq * qdotq * qdotq * qdotq * qdotq * qdotq * qdotq *
                      qdotq * qdotq));

  double dH_dq1_Newt = x[1] / std::sqrt(qdotq * qdotq * qdotq);
  double dH_dq1_1 =
      (-2 * x[1] * std::sqrt(qdotq) +
       3 * x[1] * qdotp * qdotp * reduced_mass_over_total_mass -
       2 * x[4] * qdotp * qdotq * reduced_mass_over_total_mass +
       pdotp * x[1] * qdotq * (3 + reduced_mass_over_total_mass)) /
      (2 * std::sqrt(qdotq * qdotq * qdotq * qdotq * qdotq));
  double dH_dq1_2 =
      (-48 * x[1] * qdotp * qdotp * std::sqrt(qdotq) *
           reduced_mass_over_total_mass +
       24 * x[4] * qdotp * std::sqrt(qdotq * qdotq * qdotq) *
           reduced_mass_over_total_mass +
       15 * x[1] * qdotp * qdotp * qdotp * qdotp *
           reduced_mass_over_total_mass * reduced_mass_over_total_mass +
       6 * pdotp * x[1] * qdotp * qdotp * qdotq * reduced_mass_over_total_mass *
           reduced_mass_over_total_mass -
       12 * x[4] * qdotp * qdotp * qdotp * qdotq *
           reduced_mass_over_total_mass * reduced_mass_over_total_mass -
       4 * x[4] * pdotp * qdotp * qdotq * qdotq * reduced_mass_over_total_mass *
           reduced_mass_over_total_mass +
       6 * x[1] * qdotq * (1 + 3 * reduced_mass_over_total_mass) -
       8 * pdotp * x[1] * std::sqrt(qdotq * qdotq * qdotq) *
           (5 + 8 * reduced_mass_over_total_mass) +
       pdotp * pdotp * x[1] * qdotq * qdotq *
           (-5 + reduced_mass_over_total_mass *
                     (20 + 3 * reduced_mass_over_total_mass))) /
      (8 * std::sqrt(qdotq * qdotq * qdotq * qdotq * qdotq * qdotq * qdotq));
  double dH_dq1_3 =
      (3 / 2 * qdotp * qdotq *
           (x[1] * (x[0] * x[3] + x[2] * x[5]) -
            x[4] * (x[0] * x[0] + x[2] * x[2])) *
           reduced_mass_over_total_mass *
           (340 + 3 * M_PI * M_PI + 112 * reduced_mass_over_total_mass) +
       2 * x[1] * std::sqrt(qdotq * qdotq * qdotq) *
           (-12 + (-872 + 63 * M_PI * M_PI) * reduced_mass_over_total_mass) -
       6 * qdotp * reduced_mass_over_total_mass * reduced_mass_over_total_mass *
           (pdotp * pdotp * x[1] * qdotp * qdotq * qdotq *
                (2 - 3 * reduced_mass_over_total_mass) -
            6 * pdotp * x[1] * qdotp * qdotp * qdotp * qdotq *
                (-1 + reduced_mass_over_total_mass) +
            6 * x[4] * pdotp * qdotp * qdotp * qdotq * qdotq *
                (-1 + reduced_mass_over_total_mass) -
            15 * x[1] * qdotp * qdotp * qdotp * qdotp * qdotp *
                reduced_mass_over_total_mass +
            15 * x[4] * qdotp * qdotp * qdotp * qdotp * qdotq *
                reduced_mass_over_total_mass +
            x[4] * pdotp * pdotp * qdotq * qdotq * qdotq *
                (-2 + 3 * reduced_mass_over_total_mass)) -
       2 * qdotp * std::sqrt(qdotq) * reduced_mass_over_total_mass *
           (3 * pdotp * x[1] * qdotp * qdotq *
                (17 + 30 * reduced_mass_over_total_mass) -
            3 * x[4] * pdotp * qdotq * qdotq *
                (17 + 30 * reduced_mass_over_total_mass) +
            8 * x[1] * qdotp * qdotp * qdotp *
                (5 + 43 * reduced_mass_over_total_mass) -
            8 * x[4] * qdotp * qdotp * qdotq *
                (5 + 43 * reduced_mass_over_total_mass)) -
       2 * x[1] * std::sqrt(qdotq) *
           (3 * pdotp * qdotp * qdotp * qdotq * reduced_mass_over_total_mass *
                (17 + 30 * reduced_mass_over_total_mass) +
            4 * qdotp * qdotp * qdotp * qdotp * reduced_mass_over_total_mass *
                (5 + 43 * reduced_mass_over_total_mass) +
            3 * pdotp * pdotp * qdotq * qdotq *
                (-27 + reduced_mass_over_total_mass *
                           (136 + 109 * reduced_mass_over_total_mass))) +
       0.75 * x[1] * qdotq *
           (3 * qdotp * qdotp * reduced_mass_over_total_mass *
                (340 + 3 * M_PI * M_PI + 112 * reduced_mass_over_total_mass) +
            pdotp * qdotq *
                (600 + reduced_mass_over_total_mass *
                           (1340 - 3 * M_PI * M_PI +
                            552 * reduced_mass_over_total_mass))) -
       3 * x[1] *
           (pdotp * pdotp * qdotp * qdotp * qdotq * qdotq *
                (2 - 3 * reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            3 * pdotp * qdotp * qdotp * qdotp * qdotp * qdotq *
                (-1 + reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            5 * qdotp * qdotp * qdotp * qdotp * qdotp * qdotp *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass *
                reduced_mass_over_total_mass -
            pdotp * pdotp * pdotp * qdotq * qdotq * qdotq *
                (7 +
                 reduced_mass_over_total_mass *
                     (-42 + reduced_mass_over_total_mass *
                                (53 + 5 * reduced_mass_over_total_mass))))) /
      (48 * std::sqrt(qdotq * qdotq * qdotq * qdotq * qdotq * qdotq * qdotq *
                      qdotq * qdotq));

  double dH_dq2_Newt = x[2] / std::sqrt(qdotq * qdotq * qdotq);
  double dH_dq2_1 =
      (-2 * x[2] * std::sqrt(qdotq) +
       3 * x[2] * qdotp * qdotp * reduced_mass_over_total_mass -
       2 * x[5] * qdotp * qdotq * reduced_mass_over_total_mass +
       pdotp * x[2] * qdotq * (3 + reduced_mass_over_total_mass)) /
      (2 * std::sqrt(qdotq * qdotq * qdotq * qdotq * qdotq));
  double dH_dq2_2 =
      (-48 * x[2] * qdotp * qdotp * std::sqrt(qdotq) *
           reduced_mass_over_total_mass +
       24 * x[5] * qdotp * std::sqrt(qdotq * qdotq * qdotq) *
           reduced_mass_over_total_mass +
       15 * x[2] * qdotp * qdotp * qdotp * qdotp *
           reduced_mass_over_total_mass * reduced_mass_over_total_mass +
       6 * pdotp * x[2] * qdotp * qdotp * qdotq * reduced_mass_over_total_mass *
           reduced_mass_over_total_mass -
       12 * x[5] * qdotp * qdotp * qdotp * qdotq *
           reduced_mass_over_total_mass * reduced_mass_over_total_mass -
       4 * x[5] * pdotp * qdotp * qdotq * qdotq * reduced_mass_over_total_mass *
           reduced_mass_over_total_mass +
       6 * x[2] * qdotq * (1 + 3 * reduced_mass_over_total_mass) -
       8 * pdotp * x[2] * std::sqrt(qdotq * qdotq * qdotq) *
           (5 + 8 * reduced_mass_over_total_mass) +
       pdotp * pdotp * x[2] * qdotq * qdotq *
           (-5 + reduced_mass_over_total_mass *
                     (20 + 3 * reduced_mass_over_total_mass))) /
      (8 * std::sqrt(qdotq * qdotq * qdotq * qdotq * qdotq * qdotq * qdotq));
  double dH_dq2_3 =
      (3 / 2 * qdotp * qdotq *
           (x[2] * (x[0] * x[3] + x[1] * x[3]) -
            x[5] * (x[0] * x[0] + x[1] * x[1])) *
           reduced_mass_over_total_mass *
           (340 + 3 * M_PI * M_PI + 112 * reduced_mass_over_total_mass) +
       2 * x[2] * std::sqrt(qdotq * qdotq * qdotq) *
           (-12 + (-872 + 63 * M_PI * M_PI) * reduced_mass_over_total_mass) -
       6 * qdotp * reduced_mass_over_total_mass * reduced_mass_over_total_mass *
           (pdotp * pdotp * x[2] * qdotp * qdotq * qdotq *
                (2 - 3 * reduced_mass_over_total_mass) -
            6 * pdotp * x[2] * qdotp * qdotp * qdotp * qdotq *
                (-1 + reduced_mass_over_total_mass) +
            6 * x[5] * pdotp * qdotp * qdotp * qdotq * qdotq *
                (-1 + reduced_mass_over_total_mass) -
            15 * x[2] * qdotp * qdotp * qdotp * qdotp * qdotp *
                reduced_mass_over_total_mass +
            15 * x[5] * qdotp * qdotp * qdotp * qdotp * qdotq *
                reduced_mass_over_total_mass +
            x[5] * pdotp * pdotp * qdotq * qdotq * qdotq *
                (-2 + 3 * reduced_mass_over_total_mass)) -
       2 * qdotp * std::sqrt(qdotq) * reduced_mass_over_total_mass *
           (3 * pdotp * x[2] * qdotp * qdotq *
                (17 + 30 * reduced_mass_over_total_mass) -
            3 * x[5] * pdotp * qdotq * qdotq *
                (17 + 30 * reduced_mass_over_total_mass) +
            8 * x[2] * qdotp * qdotp * qdotp *
                (5 + 43 * reduced_mass_over_total_mass) -
            8 * x[5] * qdotp * qdotp * qdotq *
                (5 + 43 * reduced_mass_over_total_mass)) -
       2 * x[2] * std::sqrt(qdotq) *
           (3 * pdotp * qdotp * qdotp * qdotq * reduced_mass_over_total_mass *
                (17 + 30 * reduced_mass_over_total_mass) +
            4 * qdotp * qdotp * qdotp * qdotp * reduced_mass_over_total_mass *
                (5 + 43 * reduced_mass_over_total_mass) +
            3 * pdotp * pdotp * qdotq * qdotq *
                (-27 + reduced_mass_over_total_mass *
                           (136 + 109 * reduced_mass_over_total_mass))) +
       0.75 * x[2] * qdotq *
           (3 * qdotp * qdotp * reduced_mass_over_total_mass *
                (340 + 3 * M_PI * M_PI + 112 * reduced_mass_over_total_mass) +
            pdotp * qdotq *
                (600 + reduced_mass_over_total_mass *
                           (1340 - 3 * M_PI * M_PI +
                            552 * reduced_mass_over_total_mass))) -
       3 * x[2] *
           (pdotp * pdotp * qdotp * qdotp * qdotq * qdotq *
                (2 - 3 * reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            3 * pdotp * qdotp * qdotp * qdotp * qdotp * qdotq *
                (-1 + reduced_mass_over_total_mass) *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass -
            5 * qdotp * qdotp * qdotp * qdotp * qdotp * qdotp *
                reduced_mass_over_total_mass * reduced_mass_over_total_mass *
                reduced_mass_over_total_mass -
            pdotp * pdotp * pdotp * qdotq * qdotq * qdotq *
                (7 +
                 reduced_mass_over_total_mass *
                     (-42 + reduced_mass_over_total_mass *
                                (53 + 5 * reduced_mass_over_total_mass))))) /
      (48 * std::sqrt(qdotq * qdotq * qdotq * qdotq * qdotq * qdotq * qdotq *
                      qdotq * qdotq));

  double L = total_mass * reduced_mass *
             sqrt((x[1] * x[5] - x[2] * x[4]) * (x[1] * x[5] - x[2] * x[4]) +
                  (x[2] * x[3] - x[0] * x[5]) * (x[2] * x[3] - x[0] * x[5]) +
                  (x[0] * x[4] - x[1] * x[3]) * (x[0] * x[4] - x[1] * x[3]));
  double w = L / (square(total_mass) * qdotq);
  double vw = std::cbrt(total_mass * w);
  double gamma_Euler = 0.57721566490153286060651209008240243104215933593992;

  double f2 = -(1247 / 336) - (35 / 12) * reduced_mass_over_total_mass;
  double f3 = 4 * M_PI;
  double f4 =
      -(44711 / 9072) + (9271 / 504) * reduced_mass_over_total_mass +
      (65 / 18) * reduced_mass_over_total_mass * reduced_mass_over_total_mass;
  double f5 = -(8191 / 672 + 583 / 24 * reduced_mass_over_total_mass) * M_PI;
  double f6 = (6643739519 / 69854400) + (16 / 3) * M_PI * M_PI -
              (1712 / 105) * gamma_Euler +
              (-134543 / 7776 + (41 / 48) * M_PI * M_PI) *
                  reduced_mass_over_total_mass -
              (94403 / 3024) * reduced_mass_over_total_mass *
                  reduced_mass_over_total_mass -
              (775 / 324) * reduced_mass_over_total_mass *
                  reduced_mass_over_total_mass * reduced_mass_over_total_mass;
  double fl6 = -1712 / 105;
  double f7 = (-16285 / 504 + 214745 / 1728 * reduced_mass_over_total_mass +
               193385 / 3024 * reduced_mass_over_total_mass *
                   reduced_mass_over_total_mass) *
              M_PI;

  double dE_dt =
      -(32 / 5) * reduced_mass_over_total_mass * reduced_mass_over_total_mass *
      vw * vw * vw * vw * vw * vw * vw * vw * vw * vw *
      (1 + f2 * vw * vw + f3 * vw * vw * vw + f4 * vw * vw * vw * vw +
       f5 * vw * vw * vw * vw * vw + f6 * vw * vw * vw * vw * vw * vw +
       fl6 * vw * vw * vw * vw * vw * vw * std::log(4 * vw) +
       f7 * vw * vw * vw * vw * vw * vw * vw);

  std::array<double, 3> F;
  F[0] = 1 / (w * L) * dE_dt * x[3];
  F[1] = 1 / (w * L) * dE_dt * x[4];
  F[2] = 1 / (w * L) * dE_dt * x[5];

  dpdt[0] = (1 / total_mass) *
            (dH_dp0_Newt + dH_dp0_1 + dH_dp0_2 + dH_dp0_3);  // dX0/dt = dH/dP0
  dpdt[1] = (1 / total_mass) *
            (dH_dp1_Newt + dH_dp1_1 + dH_dp1_2 + dH_dp1_3);  // dX1/dt = dH/dP1
  dpdt[2] = (1 / total_mass) *
            (dH_dp2_Newt + dH_dp2_1 + dH_dp2_2 + dH_dp2_3);  // dX2/dt = dH/dP2

  dpdt[3] = -(1 / total_mass) * (dH_dq0_Newt + dH_dq0_1 + dH_dq0_2 + dH_dq0_3) +
            F[0];  // dP0/dt = -dH/dX0 + F0
  dpdt[4] = -(1 / total_mass) * (dH_dq1_Newt + dH_dq1_1 + dH_dq1_2 + dH_dq1_3) +
            F[1];  // dP1/dt = -dH/dX1 + F1
  dpdt[5] = -(1 / total_mass) * (dH_dq2_Newt + dH_dq2_1 + dH_dq2_2 + dH_dq2_3) +
            F[2];  // dP2/dt = -dH/dX2 + F2
}

void BinaryWithGravitationalWaves::observer_vector(
    const BinaryWithGravitationalWaves::state_type& x, const double t) {
  past_time_.push_back(t);

  std::array<double, 3> x_cm = {
      (xcoord_right() * mass_right() + xcoord_left() * mass_left()) /
          total_mass,
      0., 0.};

  for (size_t i = 0; i < 3; ++i) {
    past_position_left_.at(i).push_back(x_cm.at(i) - mass_right() * x.at(i));
    past_position_right_.at(i).push_back(x_cm.at(i) + mass_left() * x.at(i));
  }
  for (size_t i = 3; i < 6; ++i) {
    past_momentum_left_.at(i - 3).push_back(-x.at(i) * reduced_mass);
    past_momentum_right_.at(i - 3).push_back(x.at(i) * reduced_mass);
  }

  state_type dxdt;
  hamiltonian_system(x, dxdt);

  for (size_t i = 0; i < 3; ++i) {
    past_dt_position_left_.at(i).push_back(-dxdt.at(i) * reduced_mass);
    past_dt_position_right_.at(i).push_back(dxdt.at(i) * reduced_mass);
  }
  for (size_t i = 3; i < 6; ++i) {
    past_dt_momentum_left_.at(i - 3).push_back(-dxdt.at(i) * reduced_mass);
    past_dt_momentum_right_.at(i - 3).push_back(dxdt.at(i) * reduced_mass);
  }
}

void BinaryWithGravitationalWaves::integrate_hamiltonian_system() {
  BinaryWithGravitationalWaves::state_type ini = {
      initial_state_position.at(0),
      initial_state_position.at(1),
      initial_state_position.at(2),
      initial_state_momentum.at(0),
      initial_state_momentum.at(1),
      initial_state_momentum.at(2)};  // initial conditions

  // Bind the hamiltonian_system function to this object
  auto hamiltonian_system_bound =
      std::bind(&BinaryWithGravitationalWaves::hamiltonian_system, this,
                std::placeholders::_1, std::placeholders::_2);

  // Bind the observer function to this object
  auto observer_bound =
      std::bind(&BinaryWithGravitationalWaves::observer_vector, this,
                std::placeholders::_1, std::placeholders::_2);

  // Integrate the Hamiltonian system
  boost::numeric::odeint::integrate_const(
      boost::numeric::odeint::runge_kutta4<
          BinaryWithGravitationalWaves::state_type>(),
      hamiltonian_system_bound, ini, initial_time, final_time, -time_step,
      observer_bound);
}

void BinaryWithGravitationalWaves::write_evolution_to_file() const {
  if (write_evolution_option()) {
    std::ofstream file;
    file.open("PastHistoryEvolution.txt");
    file << "time, position_left_x, position_left_y, position_left_z, "
            "momentum_left_x, momentum_left_y, momentum_left_z, "
            "position_right_x, position_right_y, position_right_z, "
            "momentum_right_x, momentum_right_y, momentum_right_z, "
            "dt_momentum_left_x, dt_momentum_left_y, dt_momentum_left_z, "
            "dt_momentum_right_x, dt_momentum_right_y, dt_momentum_right_z, "
         << std::endl;
    for (size_t i = 0; i < number_of_steps; i++) {
      file << past_time_.at(i) << ", ";
      for (size_t j = 0; j < 3; ++j) {
        file << past_position_left_.at(j).at(i) << ", ";
      }
      for (size_t j = 0; j < 3; ++j) {
        file << past_momentum_left_.at(j).at(i) << ", ";
      }
      for (size_t j = 0; j < 3; ++j) {
        file << past_position_right_.at(j).at(i) << ", ";
      }
      for (size_t j = 0; j < 3; ++j) {
        file << past_momentum_right_.at(j).at(i) << ", ";
      }
      for (size_t j = 0; j < 3; ++j) {
        file << past_dt_momentum_left_.at(j).at(i) << ", ";
      }
      for (size_t j = 0; j < 3; ++j) {
        file << past_dt_momentum_right_.at(j).at(i) << ", ";
      }
      file << std::endl;
    }
    file.close();
  }
}

PUP::able::PUP_ID BinaryWithGravitationalWaves::my_PUP_ID = 0;  // NOLINT

}  // namespace Xcts::AnalyticData

template class Xcts::AnalyticData::CommonVariables<
    DataVector, typename Xcts::AnalyticData::detail::
                    BinaryWithGravitationalWavesVariables<DataVector>::Cache>;
