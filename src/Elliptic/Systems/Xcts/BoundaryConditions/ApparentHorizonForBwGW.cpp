// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Xcts/BoundaryConditions/ApparentHorizonForBwGW.hpp"

#include <algorithm>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/LeviCivitaIterator.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Elliptic/Systems/Xcts/Geometry.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/BinaryWithGravitationalWaves.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/Xcts/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/Xcts/LongitudinalOperator.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace Xcts::BoundaryConditions {

template <Xcts::Geometry ConformalGeometry>
ApparentHorizonForBwGW<ConformalGeometry>::ApparentHorizonForBwGW(
    double mass_left, double mass_right, double xcoord_left,
    double xcoord_right, double attenuation_parameter,
    double attenuation_radius, double outer_radius, bool left_or_right,
    const Options::Context& /*context*/)
    : mass_left_(mass_left),
      mass_right_(mass_right),
      xcoord_left_(xcoord_left),
      xcoord_right_(xcoord_right),
      attenuation_parameter_(attenuation_parameter),
      attenuation_radius_(attenuation_radius),
      outer_radius_(outer_radius),
      left_or_right_(left_or_right) {
  solution_ =
      std::make_unique<Xcts::AnalyticData::BinaryWithGravitationalWaves>(
          mass_left, mass_right, xcoord_left, xcoord_right,
          attenuation_parameter, attenuation_radius, outer_radius, false);
}

namespace {

// Compute the expansion Theta = m^ij (D_i s_j - K_ij) and the shift-correction
// epsilon = shift_orthogonal - lapse
void negative_expansion_quantities(
    const gsl::not_null<Scalar<DataVector>*> expansion,
    const gsl::not_null<Scalar<DataVector>*> beta_orthogonal_correction,
    const tnsr::i<DataVector, 3>& conformal_face_normal,
    const tnsr::ij<DataVector, 3>& deriv_conformal_face_normal,
    const Scalar<DataVector>& face_normal_magnitude,
    const tnsr::II<DataVector, 3>& inv_spatial_metric,
    const tnsr::I<DataVector, 3>& face_normal_raised,
    const tnsr::Ijj<DataVector, 3>& spatial_christoffel_second_kind,
    const tnsr::ii<DataVector, 3>& extrinsic_curvature,
    const tnsr::I<DataVector, 3>& shift, const Scalar<DataVector>& lapse) {
  // Compute Theta = m^ij (D_i s_j - K_ij)
  // Use second output buffer as temporary memory
  DataVector& projection = get(*beta_orthogonal_correction);
  get(*expansion) = 0.;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      projection = inv_spatial_metric.get(i, j) -
                   face_normal_raised.get(i) * face_normal_raised.get(j);
      get(*expansion) += projection * deriv_conformal_face_normal.get(i, j) /
                         get(face_normal_magnitude);
      for (size_t k = 0; k < 3; ++k) {
        get(*expansion) -= projection * conformal_face_normal.get(k) *
                           spatial_christoffel_second_kind.get(k, i, j);
      }
      // Additional extrinsic-curvature term
      get(*expansion) += projection * extrinsic_curvature.get(i, j);
    }
  }
  get(*expansion) *= -1.;
  // Compute epsilon = shift_orthogonal - lapse
  normal_dot_flux(beta_orthogonal_correction, conformal_face_normal, shift);
  get(*beta_orthogonal_correction) *= -1.;
  get(*beta_orthogonal_correction) -= get(lapse);
}

// This sets the output buffer to: \bar{m}^{ij} \bar{\nabla}_i n_j
void normal_gradient_term(
    const gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
    const tnsr::i<DataVector, 3>& face_normal,
    const tnsr::I<DataVector, 3>& face_normal_raised,
    const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
    const Scalar<DataVector>& face_normal_magnitude,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind) {
  // Write directly into the output buffer
  DataVector& projected_normal_gradient = get(*n_dot_conformal_factor_gradient);

  *projected_normal_gradient = 0.;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      projected_normal_gradient +=
          (inv_conformal_metric.get(i, j) -
           face_normal_raised.get(i) * face_normal_raised.get(j)) *
          deriv_unnormalized_face_normal.get(i, j) / get(face_normal_magnitude);
      for (size_t k = 0; k < 3; ++k) {
        projected_normal_gradient -=
            (inv_conformal_metric.get(i, j) -
             face_normal_raised.get(i) * face_normal_raised.get(j)) *
            face_normal.get(k) * conformal_christoffel_second_kind.get(k, i, j);
      }
    }
  }
}

void compute_deriv_solution(
    const tnsr::I<DataVector, 3>& x,
    const std::optional<
        std::unique_ptr<Xcts::AnalyticData::BinaryWithGravitationalWaves>>&
        solution,
    const gsl::not_null<Scalar<DataVector>*> trace_extrinsic_curvature,
    const gsl::not_null<tnsr::iJ<DataVector, 3>*> deriv_shift,
    const gsl::not_null<tnsr::i<DataVector, 3>*>
        deriv_lapse_times_conformal_factor_minus_one,
    const gsl::not_null<tnsr::ijj<DataVector, 3>*> deriv_conformal_metric) {
  using deriv_tags = tmpl::list<
      gr::Tags::TraceExtrinsicCurvature<DataVector>,
      ::Tags::deriv<Xcts::Tags::ShiftExcess<DataVector, 3, Frame::Inertial>,
                    tmpl::size_t<3>, Frame::Inertial>,
      ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactorMinusOne<DataVector>,
                    tmpl::size_t<3>, Frame::Inertial>,
      //::Xcts::Tags::ConformalChristoffelSecondKind<DataVector, 3,
      //                                           Frame::Inertial>,
      ::Tags::deriv<
          ::Xcts::Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
          tmpl::size_t<3>, Frame::Inertial>>;
  for (size_t k = 0; k < x.get(0).size(); ++k) {
    using Affine = domain::CoordinateMaps::Affine;
    using Affine3D =
        domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
    // Setup grid
    const size_t num_points_1d_k = 5;
    const std::array<double, 3> lower_bound_k{
        {x.get(0)[k] - 0.001, x.get(1)[k] - 0.001, x.get(2)[k] - 0.001}};
    const std::array<double, 3> upper_bound_k{
        {x.get(0)[k] + 0.001, x.get(1)[k] + 0.001, x.get(2)[k] + 0.001}};
    Mesh<3> mesh_k{num_points_1d_k, Spectral::Basis::Legendre,
                   Spectral::Quadrature::GaussLobatto};
    const auto coord_map_k =
        domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
            Affine3D{
                Affine{-1., 1., lower_bound_k[0], upper_bound_k[0]},
                Affine{-1., 1., lower_bound_k[1], upper_bound_k[1]},
                Affine{-1., 1., lower_bound_k[2], upper_bound_k[2]},
            });
    const size_t num_points_3d_k =
        num_points_1d_k * num_points_1d_k * num_points_1d_k;
    const DataVector used_for_size_k = DataVector(
        num_points_3d_k, std::numeric_limits<double>::signaling_NaN());
    // Setup coordinates
    const auto x_logical_k = logical_coordinates(mesh_k);
    const auto x_inertial_k = coord_map_k(x_logical_k);
    const auto inv_jacobian_k = coord_map_k.inv_jacobian(x_logical_k);
    // Get variables
    const auto solution_deriv =
        variables_from_tagged_tuple((*solution)->variables(
            x_inertial_k, mesh_k, inv_jacobian_k, deriv_tags{}));
    const auto trace_extrinsic_curvature_aux =
        get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(solution_deriv);
    const auto deriv_shift_aux = get<
        ::Tags::deriv<Xcts::Tags::ShiftExcess<DataVector, 3, Frame::Inertial>,
                      tmpl::size_t<3>, Frame::Inertial>>(solution_deriv);
    const auto deriv_lapse_times_conformal_factor_minus_one_aux = get<
        ::Tags::deriv<Xcts::Tags::LapseTimesConformalFactorMinusOne<DataVector>,
                      tmpl::size_t<3>, Frame::Inertial>>(solution_deriv);
    const auto deriv_conformal_metric_aux = get<::Tags::deriv<
        ::Xcts::Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
        tmpl::size_t<3>, Frame::Inertial>>(solution_deriv);
    for (size_t l = 0; l < x_inertial_k.get(0).size(); ++l) {
      if (x.get(0)[k] == x_inertial_k.get(0)[l] &&
          x.get(1)[k] == x_inertial_k.get(1)[l] &&
          x.get(2)[k] == x_inertial_k.get(2)[l]) {
        get(*trace_extrinsic_curvature)[k] =
            get(trace_extrinsic_curvature_aux)[l];
        for (size_t i = 0; i < 3; ++i) {
          for (size_t j = 0; j < 3; ++j) {
            for (size_t m = 0; m < 3; ++m) {
              deriv_conformal_metric->get(i, j, m)[k] =
                  deriv_conformal_metric_aux.get(i, j, m)[l];
            }
            deriv_shift->get(i, j)[k] = deriv_shift_aux.get(i, j)[l];
          }
          deriv_lapse_times_conformal_factor_minus_one->get(i)[k] =
              deriv_lapse_times_conformal_factor_minus_one_aux.get(i)[l];
        }
        break;
      }
    }
  }
}

}  // namespace

template <Xcts::Geometry ConformalGeometry>
void ApparentHorizonForBwGW<ConformalGeometry>::apply(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_minus_one,
    const gsl::not_null<Scalar<DataVector>*>
        lapse_times_conformal_factor_minus_one,
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess,
    const gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_lapse_times_conformal_factor_gradient,
    const gsl::not_null<tnsr::I<DataVector, 3>*>
    /*n_dot_longitudinal_shift_excess*/,
    const tnsr::i<DataVector, 3>& /*deriv_conformal_factor_correction1*/,
    const tnsr::i<DataVector,
                  3>& /*deriv_lapse_times_conformal_factor_correction1*/,
    const tnsr::iJ<DataVector, 3>& /*deriv_shift_excess_correction1*/,
    const tnsr::I<DataVector, 3>& x, const tnsr::i<DataVector, 3>& face_normal,
    const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
    const Scalar<DataVector>& face_normal_magnitude) const {
  // Compute quantities from solution
  using analytic_tags = tmpl::list<
      ::Xcts::Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
      ::Xcts::Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>,
      Xcts::Tags::LapseTimesConformalFactorMinusOne<DataVector>,
      Xcts::Tags::ConformalFactorMinusOne<DataVector>,
      Xcts::Tags::ShiftExcess<DataVector, 3, Frame::Inertial>,
      Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
          DataVector, 3, Frame::Inertial>>;
  const auto solution_vars =
      variables_from_tagged_tuple((*solution_)->variables(x, analytic_tags{}));
  const auto& spatial_metric =
      get<::Xcts::Tags::ConformalMetric<DataVector, 3, Frame::Inertial>>(
          solution_vars);
  const auto& inv_spatial_metric =
      get<::Xcts::Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>>(
          solution_vars);
  const auto& lapse_times_conformal_factor_minus_one_solution =
      get<Xcts::Tags::LapseTimesConformalFactorMinusOne<DataVector>>(
          solution_vars);
  const auto& conformal_factor_minus_one_solution =
      get<Xcts::Tags::ConformalFactorMinusOne<DataVector>>(solution_vars);
  const auto& shift =
      get<Xcts::Tags::ShiftExcess<DataVector, 3, Frame::Inertial>>(
          solution_vars);
  const auto& minus_dt_conformal_metric =
      get<Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
          DataVector, 3, Frame::Inertial>>(solution_vars);

  // Allocate temporary memory
  TempBuffer<tmpl::list<
      ::Tags::TempScalar<0>, ::Tags::Tempii<1, 3>, ::Tags::TempII<2, 3>,
      ::Tags::TempScalar<3>, ::Tags::TempI<4, 3>, ::Tags::TempScalar<5>,
      ::Tags::TempScalar<6>, ::Tags::TempScalar<7>, ::Tags::TempiJ<8, 3>,
      ::Tags::Tempi<9, 3>, ::Tags::TempIjj<10, 3>, ::Tags::Tempijj<11, 3>,
      ::Tags::TempI<12, 3>>>
      buffer{x.begin()->size()};

  Scalar<DataVector>& trace_extrinsic_curvature =
      get<::Tags::TempScalar<7>>(buffer);
  tnsr::iJ<DataVector, 3>& deriv_shift = get<::Tags::TempiJ<8, 3>>(buffer);
  tnsr::i<DataVector, 3>& deriv_lapse_times_conformal_factor_minus_one =
      get<::Tags::Tempi<9, 3>>(buffer);
  tnsr::Ijj<DataVector, 3>& spatial_christoffel_second_kind =
      get<::Tags::TempIjj<10, 3>>(buffer);
  tnsr::ijj<DataVector, 3>& deriv_conformal_metric =
      get<::Tags::Tempijj<11, 3>>(buffer);
  compute_deriv_solution(
      x, solution_, make_not_null(&trace_extrinsic_curvature),
      make_not_null(&deriv_shift),
      make_not_null(&deriv_lapse_times_conformal_factor_minus_one),
      make_not_null(&deriv_conformal_metric));
  raise_or_lower_first_index(make_not_null(&spatial_christoffel_second_kind),
                             gr::christoffel_first_kind(deriv_conformal_metric),
                             inv_spatial_metric);

  auto& lapse = get<::Tags::TempScalar<0>>(buffer);
  auto& conformal_factor = get<::Tags::TempScalar<3>>(buffer);
  get(conformal_factor) = 1. + get(conformal_factor_minus_one_solution);
  get(lapse) = (1. + get(lapse_times_conformal_factor_minus_one_solution)) /
               get(conformal_factor);
  auto& longitudinal_shift_excess_minus_dt_conformal_metric =
      get<::Tags::TempII<2, 3>>(buffer);
  Xcts::longitudinal_operator(
      make_not_null(&longitudinal_shift_excess_minus_dt_conformal_metric),
      shift, deriv_shift, inv_spatial_metric, spatial_christoffel_second_kind);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < i; ++j) {
      longitudinal_shift_excess_minus_dt_conformal_metric.get(i, j) +=
          minus_dt_conformal_metric.get(i, j);
    }
  }
  auto& extrinsic_curvature = get<::Tags::Tempii<1, 3>>(buffer);
  Xcts::extrinsic_curvature(make_not_null(&extrinsic_curvature),
                            conformal_factor, lapse, spatial_metric,
                            longitudinal_shift_excess_minus_dt_conformal_metric,
                            trace_extrinsic_curvature);
  auto& face_normal_raised = get<::Tags::TempI<4, 3>>(buffer);
  raise_or_lower_index(make_not_null(&face_normal_raised), face_normal,
                       inv_spatial_metric);

  // Compute quantities for negative-expansion boundary conditions.
  Scalar<DataVector>& expansion_of_solution =
      get<::Tags::TempScalar<5>>(buffer);
  Scalar<DataVector>& beta_orthogonal = get<::Tags::TempScalar<6>>(buffer);
  negative_expansion_quantities(
      make_not_null(&expansion_of_solution), make_not_null(&beta_orthogonal),
      face_normal, deriv_unnormalized_face_normal, face_normal_magnitude,
      inv_spatial_metric, face_normal_raised, spatial_christoffel_second_kind,
      extrinsic_curvature, shift, lapse);

  // Shift
  get(beta_orthogonal) /= square(get(*conformal_factor_minus_one) + 1.);
  get(beta_orthogonal) += (get(*lapse_times_conformal_factor_minus_one) + 1.) /
                          cube(get(*conformal_factor_minus_one) + 1.);
  for (size_t i = 0; i < 3; ++i) {
    shift_excess->get(i) = -get(beta_orthogonal) * face_normal_raised.get(i);
  }
  // This is wrong: one should solve the Killing equation for the
  // shift_parallel, with the source term of dt_conformal_metric This is just an
  // approximation to the Killing equation whithout the source term
  const double total_mass = mass_left_ + mass_right_;
  const double reduced_mass = mass_left_ * mass_right_ / total_mass;
  const double separation = xcoord_right_ - xcoord_left_;
  const double angular_velocity =
      sqrt(total_mass / cube(separation)) *
      (1. + .5 * (reduced_mass / total_mass - 3.) * total_mass / separation);
  const std::array<double, 3> rotation = {0., 0., -angular_velocity};
  tnsr::I<DataVector, 3>& x_centered = get<::Tags::TempI<12, 3>>(buffer);
  x_centered = x;
  if (left_or_right_ == false) {
    x_centered.get(0) = x.get(0) - xcoord_right_;
  } else if (left_or_right_ == true) {
    x_centered.get(0) = x.get(0) - xcoord_left_;
  }
  for (LeviCivitaIterator<3> it; it; ++it) {
    shift_excess->get(it[0]) +=
        it.sign() * gsl::at(rotation, it[1]) * x_centered.get(it[2]);
  }

  // Conformal factor
  normal_gradient_term(n_dot_conformal_factor_gradient, face_normal,
                       face_normal_raised, deriv_unnormalized_face_normal,
                       face_normal_magnitude, inv_spatial_metric,
                       spatial_christoffel_second_kind);
  get(*n_dot_conformal_factor_gradient) -=
      0.25 * cube(get(*conformal_factor_minus_one) + 1.) *
      get(expansion_of_solution);
  tnsr::I<DataVector, 3>& n_dot_longitudinal_shift =
      get<::Tags::TempI<4, 3>>(buffer);  // reuse buffer of face_normal_raised
  normal_dot_flux(make_not_null(&n_dot_longitudinal_shift), face_normal,
                  longitudinal_shift_excess_minus_dt_conformal_metric);
  Scalar<DataVector>& nn_dot_longitudinal_shift =
      get<::Tags::TempScalar<6>>(buffer);
  normal_dot_flux(make_not_null(&nn_dot_longitudinal_shift), face_normal,
                  n_dot_longitudinal_shift);
  get(*n_dot_conformal_factor_gradient) +=
      -get(trace_extrinsic_curvature) *
          cube(get(*conformal_factor_minus_one) + 1.) / 6. +
      pow<4>(get(*conformal_factor_minus_one) + 1.) / 8. /
          (get(*lapse_times_conformal_factor_minus_one) + 1.) *
          get(nn_dot_longitudinal_shift);

  // Lapse
  get(*n_dot_lapse_times_conformal_factor_gradient) = 0.;
  for (size_t i = 0; i < 3; ++i) {
    get(*n_dot_lapse_times_conformal_factor_gradient) +=
        face_normal.get(i) *
        deriv_lapse_times_conformal_factor_minus_one.get(i);
  }
}

template <Xcts::Geometry ConformalGeometry>
void ApparentHorizonForBwGW<ConformalGeometry>::apply_linearized(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
    const gsl::not_null<Scalar<DataVector>*>
        lapse_times_conformal_factor_correction,
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess_correction,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_conformal_factor_gradient_correction,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_lapse_times_conformal_factor_gradient_correction,
    const gsl::not_null<tnsr::I<DataVector, 3>*>
        n_dot_longitudinal_shift_excess_correction,
    const tnsr::i<DataVector, 3>& /*deriv_conformal_factor_correction*/,
    const tnsr::i<DataVector, 3>&
    /*deriv_lapse_times_conformal_factor_correction*/,
    const tnsr::iJ<DataVector, 3>& /*deriv_shift_excess_correction*/,
    const tnsr::I<DataVector, 3>& x, const tnsr::i<DataVector, 3>& face_normal,
    const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
    const Scalar<DataVector>& face_normal_magnitude) const {
  // Compute quantities from solution

  using analytic_tags = tmpl::list<
      ::Xcts::Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
      ::Xcts::Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>,
      Xcts::Tags::LapseTimesConformalFactorMinusOne<DataVector>,
      Xcts::Tags::ConformalFactorMinusOne<DataVector>,
      Xcts::Tags::ShiftExcess<DataVector, 3, Frame::Inertial>,
      Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
          DataVector, 3, Frame::Inertial>>;
  const auto solution_vars =
      variables_from_tagged_tuple((*solution_)->variables(x, analytic_tags{}));
  const auto& spatial_metric =
      get<::Xcts::Tags::ConformalMetric<DataVector, 3, Frame::Inertial>>(
          solution_vars);
  const auto& inv_spatial_metric =
      get<::Xcts::Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>>(
          solution_vars);
  const auto& lapse_times_conformal_factor_minus_one_solution =
      get<Xcts::Tags::LapseTimesConformalFactorMinusOne<DataVector>>(
          solution_vars);
  const auto& conformal_factor_minus_one_solution =
      get<Xcts::Tags::ConformalFactorMinusOne<DataVector>>(solution_vars);
  const auto& shift =
      get<Xcts::Tags::ShiftExcess<DataVector, 3, Frame::Inertial>>(
          solution_vars);
  const auto& minus_dt_conformal_metric =
      get<Xcts::Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
          DataVector, 3, Frame::Inertial>>(solution_vars);

  // Allocate temporary memory
  TempBuffer<tmpl::list<
      ::Tags::TempScalar<0>, ::Tags::Tempii<1, 3>, ::Tags::TempII<2, 3>,
      ::Tags::TempScalar<3>, ::Tags::TempI<4, 3>, ::Tags::TempScalar<5>,
      ::Tags::TempScalar<6>, ::Tags::TempScalar<7>, ::Tags::TempiJ<8, 3>,
      ::Tags::Tempi<9, 3>, ::Tags::TempIjj<10, 3>, ::Tags::Tempijj<11, 3>>>
      buffer{x.begin()->size()};

  Scalar<DataVector>& trace_extrinsic_curvature =
      get<::Tags::TempScalar<7>>(buffer);
  tnsr::iJ<DataVector, 3>& deriv_shift = get<::Tags::TempiJ<8, 3>>(buffer);
  tnsr::i<DataVector, 3>& deriv_lapse_times_conformal_factor_minus_one =
      get<::Tags::Tempi<9, 3>>(buffer);
  tnsr::Ijj<DataVector, 3>& spatial_christoffel_second_kind =
      get<::Tags::TempIjj<10, 3>>(buffer);
  tnsr::ijj<DataVector, 3>& deriv_conformal_metric =
      get<::Tags::Tempijj<11, 3>>(buffer);
  compute_deriv_solution(
      x, solution_, make_not_null(&trace_extrinsic_curvature),
      make_not_null(&deriv_shift),
      make_not_null(&deriv_lapse_times_conformal_factor_minus_one),
      make_not_null(&deriv_conformal_metric));
  raise_or_lower_first_index(make_not_null(&spatial_christoffel_second_kind),
                             gr::christoffel_first_kind(deriv_conformal_metric),
                             inv_spatial_metric);

  auto& lapse = get<::Tags::TempScalar<0>>(buffer);
  auto& conformal_factor = get<::Tags::TempScalar<3>>(buffer);
  get(conformal_factor) = 1. + get(conformal_factor_minus_one_solution);
  get(lapse) = (1. + get(lapse_times_conformal_factor_minus_one_solution)) /
               get(conformal_factor);
  auto& longitudinal_shift_excess_minus_dt_conformal_metric =
      get<::Tags::TempII<2, 3>>(buffer);
  Xcts::longitudinal_operator(
      make_not_null(&longitudinal_shift_excess_minus_dt_conformal_metric),
      shift, deriv_shift, inv_spatial_metric, spatial_christoffel_second_kind);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < i; ++j) {
      longitudinal_shift_excess_minus_dt_conformal_metric.get(i, j) +=
          minus_dt_conformal_metric.get(i, j);
    }
  }
  auto& extrinsic_curvature = get<::Tags::Tempii<1, 3>>(buffer);
  Xcts::extrinsic_curvature(make_not_null(&extrinsic_curvature),
                            conformal_factor, lapse, spatial_metric,
                            longitudinal_shift_excess_minus_dt_conformal_metric,
                            trace_extrinsic_curvature);
  auto& face_normal_raised = get<::Tags::TempI<4, 3>>(buffer);
  raise_or_lower_index(make_not_null(&face_normal_raised), face_normal,
                       inv_spatial_metric);

  // Compute quantities for negative-expansion boundary conditions
  Scalar<DataVector>& expansion_of_solution =
      get<::Tags::TempScalar<5>>(buffer);
  Scalar<DataVector>& beta_orthogonal_correction =
      get<::Tags::TempScalar<6>>(buffer);
  negative_expansion_quantities(
      make_not_null(&expansion_of_solution),
      make_not_null(&beta_orthogonal_correction), face_normal,
      deriv_unnormalized_face_normal, face_normal_magnitude, inv_spatial_metric,
      face_normal_raised, spatial_christoffel_second_kind, extrinsic_curvature,
      shift, lapse);

  // Shift
  get(beta_orthogonal_correction) *=
      -2. * get(*conformal_factor_correction) /
      cube(get(conformal_factor_minus_one_solution) + 1.);
  get(beta_orthogonal_correction) +=
      get(*lapse_times_conformal_factor_correction) /
          cube(get(conformal_factor_minus_one_solution) + 1.) -
      3. * (get(lapse_times_conformal_factor_minus_one_solution) + 1.) /
          pow<4>(get(conformal_factor_minus_one_solution) + 1.) *
          get(*conformal_factor_correction);
  for (size_t i = 0; i < 3; ++i) {
    shift_excess_correction->get(i) =
        -get(beta_orthogonal_correction) * face_normal_raised.get(i);
  }

  // Conformal factor
  normal_gradient_term(n_dot_conformal_factor_gradient_correction, face_normal,
                       face_normal_raised, deriv_unnormalized_face_normal,
                       face_normal_magnitude, inv_spatial_metric,
                       spatial_christoffel_second_kind);
  get(*n_dot_conformal_factor_gradient_correction) *=
      -0.25 * get(*conformal_factor_correction);
  get(*n_dot_conformal_factor_gradient_correction) -=
      0.75 * square(get(conformal_factor_minus_one_solution) + 1.) *
      get(expansion_of_solution) * get(*conformal_factor_correction);
  tnsr::I<DataVector, 3>& n_dot_longitudinal_shift =
      get<::Tags::TempI<4, 3>>(buffer);  // reuse buffer of face_normal_raised
  normal_dot_flux(make_not_null(&n_dot_longitudinal_shift), face_normal,
                  longitudinal_shift_excess_minus_dt_conformal_metric);
  Scalar<DataVector>& nn_dot_longitudinal_shift =
      get<::Tags::TempScalar<6>>(buffer);
  normal_dot_flux(make_not_null(&nn_dot_longitudinal_shift), face_normal,
                  n_dot_longitudinal_shift);
  get(*n_dot_conformal_factor_gradient_correction) +=
      -0.5 * get(trace_extrinsic_curvature) *
          square(get(conformal_factor_minus_one_solution) + 1.) *
          get(*conformal_factor_correction) +
      0.5 * pow<3>(get(conformal_factor_minus_one_solution) + 1.) /
          (get(lapse_times_conformal_factor_minus_one_solution) + 1.) *
          get(nn_dot_longitudinal_shift) * get(*conformal_factor_correction) -
      0.125 * pow<4>(get(conformal_factor_minus_one_solution) + 1.) /
          square(get(lapse_times_conformal_factor_minus_one_solution) + 1.) *
          get(nn_dot_longitudinal_shift) *
          get(*lapse_times_conformal_factor_correction);
  Scalar<DataVector>& nn_dot_longitudinal_shift_correction =
      get<::Tags::TempScalar<6>>(buffer);
  normal_dot_flux(make_not_null(&nn_dot_longitudinal_shift_correction),
                  face_normal, *n_dot_longitudinal_shift_excess_correction);
  get(*n_dot_conformal_factor_gradient_correction) +=
      0.125 * pow<4>(get(conformal_factor_minus_one_solution) + 1.) /
      (get(lapse_times_conformal_factor_minus_one_solution) + 1.) *
      get(nn_dot_longitudinal_shift_correction);

  // Lapse
  get(*n_dot_lapse_times_conformal_factor_gradient_correction) = 0.;
}

template <Xcts::Geometry ConformalGeometry>
bool operator==(const ApparentHorizonForBwGW<ConformalGeometry>& /*lhs*/,
                const ApparentHorizonForBwGW<ConformalGeometry>& /*rhs*/) {
  return true;
}

template <Xcts::Geometry ConformalGeometry>
bool operator!=(const ApparentHorizonForBwGW<ConformalGeometry>& lhs,
                const ApparentHorizonForBwGW<ConformalGeometry>& rhs) {
  return not(lhs == rhs);
}

template <Xcts::Geometry ConformalGeometry>
void ApparentHorizonForBwGW<ConformalGeometry>::pup(PUP::er& p) {
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
PUP::able::PUP_ID ApparentHorizonForBwGW<ConformalGeometry>::my_PUP_ID =
    0;  // NOLINT

template class ApparentHorizonForBwGW<Xcts::Geometry::FlatCartesian>;
template class ApparentHorizonForBwGW<Xcts::Geometry::Curved>;

}  // namespace Xcts::BoundaryConditions
