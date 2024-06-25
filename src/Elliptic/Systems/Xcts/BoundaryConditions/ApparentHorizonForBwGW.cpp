// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Xcts/BoundaryConditions/ApparentHorizonForBwGW.hpp"

#include <array>
#include <cstddef>
#include <optional>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/LeviCivitaIterator.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/Systems/Xcts/Geometry.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/BinaryWithGravitationalWaves.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Factory.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/InitialDataUtilities/AnalyticSolution.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"

namespace Xcts::BoundaryConditions {

template <Xcts::Geometry ConformalGeometry>
ApparentHorizonForBwGW<ConformalGeometry>::ApparentHorizonForBwGW(
    double mass_left, double mass_right, double xcoord_left,
    double xcoord_right, double attenuation_parameter, double outer_radius,
    bool solution_for_lapse, bool solution_for_negative_expansion,
    const Options::Context& /*context*/)
    : mass_left_(mass_left),
      mass_right_(mass_right),
      xcoord_left_(xcoord_left),
      xcoord_right_(xcoord_right),
      attenuation_parameter_(attenuation_parameter),
      outer_radius_(outer_radius),
      solution_for_lapse_(solution_for_lapse),
      solution_for_negative_expansion_(solution_for_negative_expansion) {
  if (solution_for_lapse) {
    lapse_solution_ =
        std::make_unique<Xcts::AnalyticData::BinaryWithGravitationalWaves>(
            mass_left, mass_right, xcoord_left, xcoord_right,
            attenuation_parameter, outer_radius, write_evolution_option_);
  } else {
    lapse_solution_ = std::nullopt;
  }
  if (solution_for_negative_expansion) {
    negative_expansion_solution_ =
        std::make_unique<Xcts::AnalyticData::BinaryWithGravitationalWaves>(
            mass_left, mass_right, xcoord_left, xcoord_right,
            attenuation_parameter, outer_radius, write_evolution_option_);
  } else {
    negative_expansion_solution_ = std::nullopt;
  }
}

namespace {
// This sets the output buffer to: \bar{m}^{ij} \bar{\nabla}_i n_j
void normal_gradient_term_flat_cartesian(
    const gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
    const tnsr::i<DataVector, 3>& face_normal,
    const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
    const Scalar<DataVector>& face_normal_magnitude) {
  // Write directly into the output buffer
  DataVector& projected_normal_gradient = get(*n_dot_conformal_factor_gradient);
  projected_normal_gradient = (1. - square(get<0>(face_normal))) *
                                  get<0, 0>(deriv_unnormalized_face_normal) -
                              get<0>(face_normal) * get<1>(face_normal) *
                                  get<0, 1>(deriv_unnormalized_face_normal) -
                              get<0>(face_normal) * get<2>(face_normal) *
                                  get<0, 2>(deriv_unnormalized_face_normal);
  for (size_t i = 1; i < 3; ++i) {
    projected_normal_gradient += deriv_unnormalized_face_normal.get(i, i);
    for (size_t j = 0; j < 3; ++j) {
      projected_normal_gradient -= face_normal.get(i) * face_normal.get(j) *
                                   deriv_unnormalized_face_normal.get(i, j);
    }
  }
  projected_normal_gradient /= get(face_normal_magnitude);
}

// This sets the output buffer to: \bar{m}^{ij} \bar{\nabla}_i n_j
void normal_gradient_term_curved(
    const gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
    const gsl::not_null<DataVector*> temp_buffer,
    const tnsr::i<DataVector, 3>& face_normal,
    const tnsr::I<DataVector, 3>& face_normal_raised,
    const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
    const Scalar<DataVector>& face_normal_magnitude,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind) {
  // Write directly into the output buffer
  DataVector& projected_normal_gradient = get(*n_dot_conformal_factor_gradient);
  DataVector& projection = *temp_buffer;
  // Possible performance optimization: unroll the first iteration of the loop
  // to avoid zeroing the buffer. It's very verbose to do that though.
  *projected_normal_gradient = 0.;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      projection = inv_conformal_metric.get(i, j) -
                   face_normal_raised.get(i) * face_normal_raised.get(j);
      projected_normal_gradient += projection *
                                   deriv_unnormalized_face_normal.get(i, j) /
                                   get(face_normal_magnitude);
      for (size_t k = 0; k < 3; ++k) {
        projected_normal_gradient -=
            projection * face_normal.get(k) *
            conformal_christoffel_second_kind.get(k, i, j);
      }
    }
  }
}

// Compute the expansion Theta = m^ij (D_i s_j - K_ij) and the shift-correction
// epsilon = shift_orthogonal - lapse of an analytic solution
void negative_expansion_quantities(
    const gsl::not_null<Scalar<DataVector>*> expansion,
    const gsl::not_null<Scalar<DataVector>*> beta_orthogonal_correction,
    const std::unique_ptr<Xcts::AnalyticData::BinaryWithGravitationalWaves>&
        solution,
    const tnsr::I<DataVector, 3>& x,
    const tnsr::i<DataVector, 3>& conformal_face_normal,
    const Scalar<DataVector>& unnormalized_conformal_face_normal_magnitude,
    const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal) {
  using analytic_tags =
      tmpl::list<Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>,
                 //::Tags::deriv<Xcts::Tags::ConformalMetric<DataVector, 3,
                 //: Frame::Inertial>,
                 //   tmpl::size_t<3>, Frame::Inertial>,
                 Xcts::Tags::LapseTimesConformalFactorMinusOne<DataVector>,
                 Xcts::Tags::ShiftExcess<DataVector, 3, Frame::Inertial>,
                 Xcts::AnalyticData::detail::Tags::
                     PostNewtonianExtrinsicCurvature<DataVector>>;
  const auto solution_vars =
      variables_from_tagged_tuple(solution->variables(x, analytic_tags{}));
  const auto& inv_spatial_metric =
      get<Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>>(
          solution_vars);
  // const auto& deriv_spatial_metric =
  // get<::Tags::deriv<Xcts::Tags::ConformalMetric<DataVector, 3,
  // Frame::Inertial>,
  //               tmpl::size_t<3>, Frame::Inertial>>(solution_vars);
  const auto& lapse_times_conformal_factor_minus_one =
      get<Xcts::Tags::LapseTimesConformalFactorMinusOne<DataVector>>(
          solution_vars);
  const auto& shift =
      get<Xcts::Tags::ShiftExcess<DataVector, 3, Frame::Inertial>>(
          solution_vars);
  const auto& extrinsic_curvature =
      get<Xcts::AnalyticData::detail::Tags::PostNewtonianExtrinsicCurvature<
          DataVector>>(solution_vars);
  // The passed-in face normal is normalized with the full conformal metric
  // (typically a superposition of isolated metrics). Therefore, we need to
  // normalized the face normal again with the isolated Kerr metric. Possible
  // optimization if this turns out to have a significant computational cost:
  // We could just re-use the face normal and the projected normal gradient
  // from the full conformal metric, since it's probably sufficiently close to
  // the isolated Kerr metric. However, the difference between the two metric
  // means that the expansion is not exactly zero when the excision surface
  // coincides with the isolated Kerr horizon, which users would probably
  // expect.
  TempBuffer<tmpl::list<::Tags::Tempi<0, 3>, ::Tags::TempScalar<1>,
                        ::Tags::TempI<2, 3>, ::Tags::TempIjj<3, 3>>>
      buffer{x.begin()->size()};
  auto& face_normal = get<::Tags::Tempi<0, 3>>(buffer);
  auto& face_normal_magnitude = get<::Tags::TempScalar<1>>(buffer);
  auto& face_normal_raised = get<::Tags::TempI<2, 3>>(buffer);
  auto& spatial_christoffel_second_kind = get<::Tags::TempIjj<3, 3>>(buffer);
  magnitude(make_not_null(&face_normal_magnitude), conformal_face_normal,
            inv_spatial_metric);
  face_normal = conformal_face_normal;
  for (size_t d = 0; d < 3; ++d) {
    face_normal.get(d) /= get(face_normal_magnitude);
  }
  get(face_normal_magnitude) *=
      get(unnormalized_conformal_face_normal_magnitude);
  raise_or_lower_index(make_not_null(&face_normal_raised), face_normal,
                       inv_spatial_metric);
  // gr::christoffel_second_kind(
  //make_not_null(&spatial_christoffel_second_kind),
  //                             deriv_spatial_metric, inv_spatial_metric);
  std::fill(spatial_christoffel_second_kind.begin(),
            spatial_christoffel_second_kind.end(), 0.);
  // Compute Theta = m^ij (D_i s_j - K_ij)
  // Use second output buffer as temporary memory
  DataVector& projection = get(*beta_orthogonal_correction);
  get(*expansion) = 0.;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      projection = inv_spatial_metric.get(i, j) -
                   face_normal_raised.get(i) * face_normal_raised.get(j);
      // This first part is the same as `normal_gradient_term_curved` above
      get(*expansion) += projection * deriv_unnormalized_face_normal.get(i, j) /
                         get(face_normal_magnitude);
      for (size_t k = 0; k < 3; ++k) {
        get(*expansion) -= projection * face_normal.get(k) *
                           spatial_christoffel_second_kind.get(k, i, j);
      }
      // Additional extrinsic-curvature term
      get(*expansion) += projection * extrinsic_curvature.get(i, j);
    }
  }
  get(*expansion) *= -1.;
  // Compute epsilon = shift_orthogonal - lapse
  normal_dot_flux(beta_orthogonal_correction, face_normal, shift);
  get(*beta_orthogonal_correction) *= -1.;
  get(*beta_orthogonal_correction) -=
      get(lapse_times_conformal_factor_minus_one) +
      1.;  // WRONG: should be "... -= get(lapse);"
  // or maybe it should be "... -= get(lapse_times_conformal_factor_minus_one)
  // + 1.;" because we want the conformal factor to be 1
  //  CHECK LATTER AGAIN
}

template <Xcts::Geometry ConformalGeometry>
void apparent_horizon_impl(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_minus_one,
    const gsl::not_null<Scalar<DataVector>*>
        lapse_times_conformal_factor_minus_one,
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess,
    const gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_lapse_times_conformal_factor_gradient,
    const gsl::not_null<tnsr::I<DataVector, 3>*>
        n_dot_longitudinal_shift_excess,
    const std::optional<
        std::unique_ptr<Xcts::AnalyticData::BinaryWithGravitationalWaves>>&
        lapse_solution,
    const std::optional<
        std::unique_ptr<Xcts::AnalyticData::BinaryWithGravitationalWaves>>&
        negative_expansion_solution,
    const tnsr::i<DataVector, 3>& face_normal,
    const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
    const Scalar<DataVector>& face_normal_magnitude,
    const tnsr::I<DataVector, 3>& x,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const tnsr::I<DataVector, 3>& shift_background,
    const tnsr::II<DataVector, 3>& longitudinal_shift_background,
    [[maybe_unused]] const std::optional<
        std::reference_wrapper<const tnsr::II<DataVector, 3>>>
        inv_conformal_metric,
    [[maybe_unused]] const std::optional<
        std::reference_wrapper<const tnsr::Ijj<DataVector, 3>>>
        conformal_christoffel_second_kind) {
  // Allocate some temporary memory
  TempBuffer<tmpl::list<::Tags::TempI<0, 3>, ::Tags::TempScalar<1>,
                        ::Tags::TempI<2, 3>, ::Tags::TempScalar<3>>>
      buffer{face_normal.begin()->size()};
  // Note that the face normal points _out_ of the computational domain, i.e.
  // _into_ the excised region. It is opposite the conformal unit normal to the
  // horizon surface: \bar{s}_i = -n_i.
  tnsr::I<DataVector, 3>& face_normal_raised = get<::Tags::TempI<0, 3>>(buffer);
  if constexpr (ConformalGeometry == Xcts::Geometry::FlatCartesian) {
    get<0>(face_normal_raised) = get<0>(face_normal);
    get<1>(face_normal_raised) = get<1>(face_normal);
    get<2>(face_normal_raised) = get<2>(face_normal);
  } else {
    raise_or_lower_index(make_not_null(&face_normal_raised), face_normal,
                         inv_conformal_metric->get());
  }

  // Compute quantities for negative-expansion boundary conditions. We collect
  // all calls into the analytic solution in one place so it doesn't have to
  // compute intermediate quantities multiple times. Possible optimization:
  // Store the result in the DataBox and use it in repeated applications of the
  // boundary conditions.
  Scalar<DataVector>& expansion_of_solution =
      get<::Tags::TempScalar<3>>(buffer);
  Scalar<DataVector>& beta_orthogonal = get<::Tags::TempScalar<1>>(buffer);
  if (negative_expansion_solution.has_value()) {
    negative_expansion_quantities(
        make_not_null(&expansion_of_solution), make_not_null(&beta_orthogonal),
        *negative_expansion_solution, x, face_normal, face_normal_magnitude,
        deriv_unnormalized_face_normal);
  }

  // Shift
  {
    if (negative_expansion_solution.has_value()) {
      get(beta_orthogonal) /= square(get(*conformal_factor_minus_one) + 1.);
    } else {
      get(beta_orthogonal) = 0.;
    }
    get(beta_orthogonal) +=
        (get(*lapse_times_conformal_factor_minus_one) + 1.) /
        cube(get(*conformal_factor_minus_one) + 1.);
    for (size_t i = 0; i < 3; ++i) {
      shift_excess->get(i) = -get(beta_orthogonal) * face_normal_raised.get(i) -
                             shift_background.get(i);
    }
  }
  // At this point we're done with `beta_orthogonal`, so we can re-purpose the
  // memory buffer.
  for (LeviCivitaIterator<3> it; it; ++it) {
    shift_excess->get(it[0]) += 0.;
  }

  // Conformal factor
  if constexpr (ConformalGeometry == Xcts::Geometry::FlatCartesian) {
    normal_gradient_term_flat_cartesian(
        n_dot_conformal_factor_gradient, face_normal,
        deriv_unnormalized_face_normal, face_normal_magnitude);
  } else {
    normal_gradient_term_curved(
        n_dot_conformal_factor_gradient,
        make_not_null(&get(get<::Tags::TempScalar<1>>(buffer))), face_normal,
        face_normal_raised, deriv_unnormalized_face_normal,
        face_normal_magnitude, *inv_conformal_metric,
        *conformal_christoffel_second_kind);
  }
  // At this point we're done with `face_normal_raised`, so we can re-purpose
  // the memory buffer.
  get(*n_dot_conformal_factor_gradient) *=
      -0.25 * (get(*conformal_factor_minus_one) + 1.);
  if (negative_expansion_solution.has_value()) {
    get(*n_dot_conformal_factor_gradient) -=
        0.25 * cube(get(*conformal_factor_minus_one) + 1.) *
        get(expansion_of_solution);
  }
  {
    tnsr::I<DataVector, 3>& n_dot_longitudinal_shift =
        get<::Tags::TempI<0, 3>>(buffer);
    normal_dot_flux(make_not_null(&n_dot_longitudinal_shift), face_normal,
                    longitudinal_shift_background);
    for (size_t i = 0; i < 3; ++i) {
      n_dot_longitudinal_shift.get(i) +=
          n_dot_longitudinal_shift_excess->get(i);
    }
    Scalar<DataVector>& nn_dot_longitudinal_shift =
        get<::Tags::TempScalar<1>>(buffer);
    normal_dot_flux(make_not_null(&nn_dot_longitudinal_shift), face_normal,
                    n_dot_longitudinal_shift);
    get(*n_dot_conformal_factor_gradient) +=
        -get(extrinsic_curvature_trace) *
            cube(get(*conformal_factor_minus_one) + 1.) / 6. +
        pow<4>(get(*conformal_factor_minus_one) + 1.) / 8. /
            (get(*lapse_times_conformal_factor_minus_one) + 1.) *
            get(nn_dot_longitudinal_shift);
  }

  // Lapse
  if (lapse_solution.has_value()) {
    *lapse_times_conformal_factor_minus_one = get<
        Xcts::Tags::LapseTimesConformalFactorMinusOne<DataVector>>(
        variables_from_tagged_tuple(
            (*lapse_solution)
                ->variables(
                    x, tmpl::list<Xcts::Tags::LapseTimesConformalFactorMinusOne<
                           DataVector>>{})));
  } else {
    get(*n_dot_lapse_times_conformal_factor_gradient) = 0.;
  }
}

template <Xcts::Geometry ConformalGeometry>
void linearized_apparent_horizon_impl(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
    const gsl::not_null<Scalar<DataVector>*>
        lapse_times_conformal_factor_correction,
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift_correction,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_conformal_factor_gradient_correction,
    const gsl::not_null<Scalar<DataVector>*>
        n_dot_lapse_times_conformal_factor_gradient_correction,
    const gsl::not_null<tnsr::I<DataVector, 3>*>
        n_dot_longitudinal_shift_correction,
    const std::optional<
        std::unique_ptr<Xcts::AnalyticData::BinaryWithGravitationalWaves>>&
        lapse_solution,
    const std::optional<
        std::unique_ptr<Xcts::AnalyticData::BinaryWithGravitationalWaves>>&
        negative_expansion_solution,
    const tnsr::i<DataVector, 3>& face_normal,
    const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
    const Scalar<DataVector>& face_normal_magnitude,
    const tnsr::I<DataVector, 3>& x,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const tnsr::II<DataVector, 3>& longitudinal_shift_background,
    const Scalar<DataVector>& conformal_factor_minus_one,
    const Scalar<DataVector>& lapse_times_conformal_factor_minus_one,
    const tnsr::I<DataVector, 3>& n_dot_longitudinal_shift_excess,
    [[maybe_unused]] const std::optional<
        std::reference_wrapper<const tnsr::II<DataVector, 3>>>
        inv_conformal_metric,
    [[maybe_unused]] const std::optional<
        std::reference_wrapper<const tnsr::Ijj<DataVector, 3>>>
        conformal_christoffel_second_kind) {
  // Allocate some temporary memory
  TempBuffer<tmpl::list<::Tags::TempI<0, 3>, ::Tags::TempScalar<1>,
                        ::Tags::TempScalar<2>>>
      buffer{face_normal.begin()->size()};

  // Negative-expansion quantities
  Scalar<DataVector>& expansion_of_solution =
      get<::Tags::TempScalar<2>>(buffer);
  Scalar<DataVector>& beta_orthogonal_correction =
      get<::Tags::TempScalar<1>>(buffer);
  if (negative_expansion_solution.has_value()) {
    negative_expansion_quantities(make_not_null(&expansion_of_solution),
                                  make_not_null(&beta_orthogonal_correction),
                                  *negative_expansion_solution, x, face_normal,
                                  face_normal_magnitude,
                                  deriv_unnormalized_face_normal);
  }

  tnsr::I<DataVector, 3>& face_normal_raised = get<::Tags::TempI<0, 3>>(buffer);
  if constexpr (ConformalGeometry == Xcts::Geometry::FlatCartesian) {
    get<0>(face_normal_raised) = get<0>(face_normal);
    get<1>(face_normal_raised) = get<1>(face_normal);
    get<2>(face_normal_raised) = get<2>(face_normal);
  } else {
    raise_or_lower_index(make_not_null(&face_normal_raised), face_normal,
                         inv_conformal_metric->get());
  }

  // Shift
  {
    if (negative_expansion_solution.has_value()) {
      get(beta_orthogonal_correction) *=
          -2. * get(*conformal_factor_correction) /
          cube(get(conformal_factor_minus_one) + 1.);
    } else {
      get(beta_orthogonal_correction) = 0.;
    }
    get(beta_orthogonal_correction) +=
        get(*lapse_times_conformal_factor_correction) /
            cube(get(conformal_factor_minus_one) + 1.) -
        3. * (get(lapse_times_conformal_factor_minus_one) + 1.) /
            pow<4>(get(conformal_factor_minus_one) + 1.) *
            get(*conformal_factor_correction);
    for (size_t i = 0; i < 3; ++i) {
      shift_correction->get(i) =
          -get(beta_orthogonal_correction) * face_normal_raised.get(i);
    }
  }
  // At this point we're done with `beta_orthogonal_correction`, so we can
  // re-purpose the memory buffer.

  // Conformal factor
  if constexpr (ConformalGeometry == Xcts::Geometry::FlatCartesian) {
    normal_gradient_term_flat_cartesian(
        n_dot_conformal_factor_gradient_correction, face_normal,
        deriv_unnormalized_face_normal, face_normal_magnitude);
  } else {
    normal_gradient_term_curved(
        n_dot_conformal_factor_gradient_correction,
        make_not_null(&get(get<::Tags::TempScalar<1>>(buffer))), face_normal,
        face_normal_raised, deriv_unnormalized_face_normal,
        face_normal_magnitude, *inv_conformal_metric,
        *conformal_christoffel_second_kind);
  }
  // At this point we're done with `face_normal_raised`, so we can re-purpose
  // the memory buffer.
  get(*n_dot_conformal_factor_gradient_correction) *=
      -0.25 * get(*conformal_factor_correction);
  if (negative_expansion_solution.has_value()) {
    get(*n_dot_conformal_factor_gradient_correction) -=
        0.75 * square(get(conformal_factor_minus_one) + 1.) *
        get(expansion_of_solution) * get(*conformal_factor_correction);
  }
  {
    tnsr::I<DataVector, 3>& n_dot_longitudinal_shift =
        get<::Tags::TempI<0, 3>>(buffer);
    normal_dot_flux(make_not_null(&n_dot_longitudinal_shift), face_normal,
                    longitudinal_shift_background);
    for (size_t i = 0; i < 3; ++i) {
      n_dot_longitudinal_shift.get(i) += n_dot_longitudinal_shift_excess.get(i);
    }
    Scalar<DataVector>& nn_dot_longitudinal_shift =
        get<::Tags::TempScalar<1>>(buffer);
    normal_dot_flux(make_not_null(&nn_dot_longitudinal_shift), face_normal,
                    n_dot_longitudinal_shift);
    get(*n_dot_conformal_factor_gradient_correction) +=
        -0.5 * get(extrinsic_curvature_trace) *
            square(get(conformal_factor_minus_one) + 1.) *
            get(*conformal_factor_correction) +
        0.5 * pow<3>(get(conformal_factor_minus_one) + 1.) /
            (get(lapse_times_conformal_factor_minus_one) + 1.) *
            get(nn_dot_longitudinal_shift) * get(*conformal_factor_correction) -
        0.125 * pow<4>(get(conformal_factor_minus_one) + 1.) /
            square(get(lapse_times_conformal_factor_minus_one) + 1.) *
            get(nn_dot_longitudinal_shift) *
            get(*lapse_times_conformal_factor_correction);
  }
  {
    Scalar<DataVector>& nn_dot_longitudinal_shift_correction =
        get<::Tags::TempScalar<1>>(buffer);
    normal_dot_flux(make_not_null(&nn_dot_longitudinal_shift_correction),
                    face_normal, *n_dot_longitudinal_shift_correction);
    get(*n_dot_conformal_factor_gradient_correction) +=
        0.125 * pow<4>(get(conformal_factor_minus_one) + 1.) /
        (get(lapse_times_conformal_factor_minus_one) + 1.) *
        get(nn_dot_longitudinal_shift_correction);
  }

  // Lapse
  if (lapse_solution.has_value()) {
    get(*lapse_times_conformal_factor_correction) = 0.;
  } else {
    get(*n_dot_lapse_times_conformal_factor_gradient_correction) = 0.;
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
        n_dot_longitudinal_shift_excess,
    const tnsr::i<DataVector, 3>& /*deriv_conformal_factor*/,
    const tnsr::i<DataVector, 3>& /*deriv_lapse_times_conformal_factor*/,
    const tnsr::iJ<DataVector, 3>& /*deriv_shift_excess*/,
    const tnsr::i<DataVector, 3>& face_normal,
    const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
    const Scalar<DataVector>& face_normal_magnitude,
    const tnsr::I<DataVector, 3>& x,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const tnsr::I<DataVector, 3>& shift_background,
    const tnsr::II<DataVector, 3>& longitudinal_shift_background) const {
  apparent_horizon_impl<ConformalGeometry>(
      conformal_factor_minus_one, lapse_times_conformal_factor_minus_one,
      shift_excess, n_dot_conformal_factor_gradient,
      n_dot_lapse_times_conformal_factor_gradient,
      n_dot_longitudinal_shift_excess, lapse_solution_,
      negative_expansion_solution_, face_normal, deriv_unnormalized_face_normal,
      face_normal_magnitude, x, extrinsic_curvature_trace, shift_background,
      longitudinal_shift_background, std::nullopt, std::nullopt);
}

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
        n_dot_longitudinal_shift_excess,
    const tnsr::i<DataVector, 3>& /*deriv_conformal_factor*/,
    const tnsr::i<DataVector, 3>& /*deriv_lapse_times_conformal_factor*/,
    const tnsr::iJ<DataVector, 3>& /*deriv_shift_excess*/,
    const tnsr::i<DataVector, 3>& face_normal,
    const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
    const Scalar<DataVector>& face_normal_magnitude,
    const tnsr::I<DataVector, 3>& x,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const tnsr::I<DataVector, 3>& shift_background,
    const tnsr::II<DataVector, 3>& longitudinal_shift_background,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind) const {
  apparent_horizon_impl<ConformalGeometry>(
      conformal_factor_minus_one, lapse_times_conformal_factor_minus_one,
      shift_excess, n_dot_conformal_factor_gradient,
      n_dot_lapse_times_conformal_factor_gradient,
      n_dot_longitudinal_shift_excess, lapse_solution_,
      negative_expansion_solution_, face_normal, deriv_unnormalized_face_normal,
      face_normal_magnitude, x, extrinsic_curvature_trace, shift_background,
      longitudinal_shift_background, inv_conformal_metric,
      conformal_christoffel_second_kind);
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
    const tnsr::i<DataVector, 3>& /*deriv_conformal_factor*/,
    const tnsr::i<DataVector, 3>& /*deriv_lapse_times_conformal_factor*/,
    const tnsr::iJ<DataVector, 3>& /*deriv_shift_excess*/,
    const tnsr::i<DataVector, 3>& face_normal,
    const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
    const Scalar<DataVector>& face_normal_magnitude,
    const tnsr::I<DataVector, 3>& x,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const tnsr::II<DataVector, 3>& longitudinal_shift_background,
    const Scalar<DataVector>& conformal_factor_minus_one,
    const Scalar<DataVector>& lapse_times_conformal_factor_minus_one,
    const tnsr::I<DataVector, 3>& n_dot_longitudinal_shift_excess) const {
  linearized_apparent_horizon_impl<ConformalGeometry>(
      conformal_factor_correction, lapse_times_conformal_factor_correction,
      shift_excess_correction, n_dot_conformal_factor_gradient_correction,
      n_dot_lapse_times_conformal_factor_gradient_correction,
      n_dot_longitudinal_shift_excess_correction, lapse_solution_,
      negative_expansion_solution_, face_normal, deriv_unnormalized_face_normal,
      face_normal_magnitude, x, extrinsic_curvature_trace,
      longitudinal_shift_background, conformal_factor_minus_one,
      lapse_times_conformal_factor_minus_one, n_dot_longitudinal_shift_excess,
      std::nullopt, std::nullopt);
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
    const tnsr::i<DataVector, 3>& /*deriv_conformal_factor*/,
    const tnsr::i<DataVector, 3>& /*deriv_lapse_times_conformal_factor*/,
    const tnsr::iJ<DataVector, 3>& /*deriv_shift_excess*/,
    const tnsr::i<DataVector, 3>& face_normal,
    const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
    const Scalar<DataVector>& face_normal_magnitude,
    const tnsr::I<DataVector, 3>& x,
    const Scalar<DataVector>& extrinsic_curvature_trace,
    const tnsr::II<DataVector, 3>& longitudinal_shift_background,
    const Scalar<DataVector>& conformal_factor_minus_one,
    const Scalar<DataVector>& lapse_times_conformal_factor_minus_one,
    const tnsr::I<DataVector, 3>& n_dot_longitudinal_shift_excess,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind) const {
  linearized_apparent_horizon_impl<ConformalGeometry>(
      conformal_factor_correction, lapse_times_conformal_factor_correction,
      shift_excess_correction, n_dot_conformal_factor_gradient_correction,
      n_dot_lapse_times_conformal_factor_gradient_correction,
      n_dot_longitudinal_shift_excess_correction, lapse_solution_,
      negative_expansion_solution_, face_normal, deriv_unnormalized_face_normal,
      face_normal_magnitude, x, extrinsic_curvature_trace,
      longitudinal_shift_background, conformal_factor_minus_one,
      lapse_times_conformal_factor_minus_one, n_dot_longitudinal_shift_excess,
      inv_conformal_metric, conformal_christoffel_second_kind);
}

template <Xcts::Geometry ConformalGeometry>
void ApparentHorizonForBwGW<ConformalGeometry>::pup(PUP::er& p) {
  Base::pup(p);
  p | mass_left_;
  p | mass_right_;
  p | xcoord_left_;
  p | xcoord_right_;
  p | attenuation_parameter_;
  p | lapse_solution_;
  p | negative_expansion_solution_;
}

template <Xcts::Geometry ConformalGeometry>
PUP::able::PUP_ID ApparentHorizonForBwGW<ConformalGeometry>::my_PUP_ID =
    0;  // NOLINT

template class ApparentHorizonForBwGW<Xcts::Geometry::FlatCartesian>;
template class ApparentHorizonForBwGW<Xcts::Geometry::Curved>;

}  // namespace Xcts::BoundaryConditions
