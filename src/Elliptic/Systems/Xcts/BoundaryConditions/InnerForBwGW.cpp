// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Xcts/BoundaryConditions/InnerForBwGW.hpp"

#include <algorithm>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Elliptic/Systems/Xcts/Geometry.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/BinaryWithGravitationalWaves.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace Xcts::BoundaryConditions {

template <Xcts::Geometry ConformalGeometry>
InnerForBwGW<ConformalGeometry>::InnerForBwGW(
    double mass_left, double mass_right, double xcoord_left,
    double xcoord_right, double attenuation_parameter,
    double attenuation_radius, double outer_radius,
    elliptic::BoundaryConditionType boundary,
    const Options::Context& /*context*/)
    : mass_left_(mass_left),
      mass_right_(mass_right),
      xcoord_left_(xcoord_left),
      xcoord_right_(xcoord_right),
      attenuation_parameter_(attenuation_parameter),
      attenuation_radius_(attenuation_radius),
      outer_radius_(outer_radius),
      boundary_(boundary) {
  solution_ =
      std::make_unique<Xcts::AnalyticData::BinaryWithGravitationalWaves>(
          mass_left, mass_right, xcoord_left, xcoord_right,
          attenuation_parameter, attenuation_radius, outer_radius, false);
}

namespace {

template <Xcts::Geometry ConformalGeometry>
void implement_apply_dirichlet(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_minus_one,
    const gsl::not_null<Scalar<DataVector>*>
        lapse_times_conformal_factor_minus_one,
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess,
    const std::optional<
        std::unique_ptr<Xcts::AnalyticData::BinaryWithGravitationalWaves>>&
        solution,
    const tnsr::I<DataVector, 3>& x) {
  using analytic_tags =
      tmpl::list<Xcts::Tags::ConformalFactorMinusOne<DataVector>,
                 Xcts::Tags::LapseTimesConformalFactorMinusOne<DataVector>,
                 Xcts::Tags::ShiftExcess<DataVector, 3, Frame::Inertial>>;
  const auto solution_vars =
      variables_from_tagged_tuple((*solution)->variables(x, analytic_tags{}));
  *conformal_factor_minus_one =
      get<Xcts::Tags::ConformalFactorMinusOne<DataVector>>(solution_vars);
  *lapse_times_conformal_factor_minus_one =
      get<Xcts::Tags::LapseTimesConformalFactorMinusOne<DataVector>>(
          solution_vars);
  for (size_t i = 0; i < 3; ++i) {
    shift_excess->get(i) =
        get<Xcts::Tags::ShiftExcess<DataVector, 3, Frame::Inertial>>(
            solution_vars)
            .get(i);
  }
}

template <Xcts::Geometry ConformalGeometry>
void implement_apply_neumann(
    const gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_lapse_times_conformal_factor_gradient,
    const gsl::not_null<tnsr::I<DataVector, 3>*>
        n_dot_longitudinal_shift_excess,
    const tnsr::i<DataVector, 3>& face_normal) {
  get(*n_dot_conformal_factor_gradient) = 0.;
  get(*n_dot_lapse_times_conformal_factor_gradient) = 0.;
  for (size_t i = 0; i < 3; ++i) {
    n_dot_longitudinal_shift_excess->get(i) = 0.;
  }
}

}  // namespace

template <Xcts::Geometry ConformalGeometry>
void InnerForBwGW<ConformalGeometry>::apply(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_minus_one,
    const gsl::not_null<Scalar<DataVector>*>
        lapse_times_conformal_factor_minus_one,
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess,
    const gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_lapse_times_conformal_factor_gradient,
    const gsl::not_null<tnsr::I<DataVector, 3>*>
        n_dot_longitudinal_shift_excess,
    const tnsr::i<DataVector, 3>& deriv_conformal_factor_correction,
    const tnsr::i<DataVector, 3>& deriv_lapse_times_conformal_factor_correction,
    const tnsr::iJ<DataVector, 3>& deriv_shift_excess_correction,
    const tnsr::I<DataVector, 3>& x,
    const tnsr::i<DataVector, 3>& face_normal) const {
  if (boundary_ == elliptic::BoundaryConditionType::Dirichlet) {
    implement_apply_dirichlet<ConformalGeometry>(
        conformal_factor_minus_one, lapse_times_conformal_factor_minus_one,
        shift_excess, solution_, x);
  } else if (boundary_ == elliptic::BoundaryConditionType::Neumann) {
    implement_apply_neumann<ConformalGeometry>(
        n_dot_conformal_factor_gradient,
        n_dot_lapse_times_conformal_factor_gradient,
        n_dot_longitudinal_shift_excess, face_normal);
  }
}

template <Xcts::Geometry ConformalGeometry>
void InnerForBwGW<ConformalGeometry>::apply_linearized(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
    const gsl::not_null<Scalar<DataVector>*>
        lapse_times_conformal_factor_correction,
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess_correction,
    const gsl::not_null<Scalar<DataVector>*>
    /*n_dot_conformal_factor_gradient_correction*/,
    const gsl::not_null<Scalar<DataVector>*>
    /*n_dot_lapse_times_conformal_factor_gradient_correction*/,
    const gsl::not_null<tnsr::I<DataVector, 3>*>
    /*n_dot_longitudinal_shift_excess_correction*/,
    const tnsr::i<DataVector, 3>& /*deriv_conformal_factor_correction*/,
    const tnsr::i<DataVector, 3>&
    /*deriv_lapse_times_conformal_factor_correction*/,
    const tnsr::iJ<DataVector, 3>& /*deriv_shift_excess_correction*/) const {
  get(*conformal_factor_correction) = 0.;
  get(*lapse_times_conformal_factor_correction) = 0.;
  std::fill(shift_excess_correction->begin(), shift_excess_correction->end(),
            0.);
}

template <Xcts::Geometry ConformalGeometry>
bool operator==(const InnerForBwGW<ConformalGeometry>& /*lhs*/,
                const InnerForBwGW<ConformalGeometry>& /*rhs*/) {
  return true;
}

template <Xcts::Geometry ConformalGeometry>
bool operator!=(const InnerForBwGW<ConformalGeometry>& lhs,
                const InnerForBwGW<ConformalGeometry>& rhs) {
  return not(lhs == rhs);
}

template <Xcts::Geometry ConformalGeometry>
void InnerForBwGW<ConformalGeometry>::pup(PUP::er& p) {
  Base::pup(p);
  p | mass_left_;
  p | mass_right_;
  p | xcoord_left_;
  p | xcoord_right_;
  p | attenuation_parameter_;
  p | outer_radius_;
  p | solution_;
}

template <Xcts::Geometry ConformalGeometry>
PUP::able::PUP_ID InnerForBwGW<ConformalGeometry>::my_PUP_ID = 0;  // NOLINT

template class InnerForBwGW<Xcts::Geometry::FlatCartesian>;
template class InnerForBwGW<Xcts::Geometry::Curved>;

}  // namespace Xcts::BoundaryConditions
