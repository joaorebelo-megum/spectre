// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>
#include <vector>

#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/FaceNormal.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Elliptic/Systems/Xcts/Geometry.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Options/Auto.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/BinaryWithGravitationalWaves.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/CommonVariables.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace Xcts::BoundaryConditions {

/*!
 * \brief Impose the surface is a quasi-equilibrium apparent horizon.
 *
 * These boundary conditions on the conformal factor \f$\psi\f$, the lapse
 * \f$\alpha\f$ and the shift \f$\beta^i\f$ impose the surface is an apparent
 * horizon, i.e. that the expansion on the surface vanishes: \f$\Theta=0\f$.
 * Specifically, we impose:
 *
 * \f{align}
 * \label{eq:ah_psi}
 * \bar{s}^k\bar{D}_k\psi &= -\frac{\psi^3}{8\alpha}\bar{s}_i\bar{s}_j\left(
 * (\bar{L}\beta)^{ij} - \bar{u}^{ij}\right)
 * - \frac{\psi}{4}\bar{m}^{ij}\bar{\nabla}_i\bar{s}_j + \frac{1}{6}K\psi^3
 * \\
 * \label{eq:ah_beta}
 * \beta_\mathrm{excess}^i &= \frac{\alpha}{\psi^2}\bar{s}^i
 * + \epsilon_{ijk}\Omega^j x^k - \beta_\mathrm{background}^i
 * \f}
 *
 * following section 7.2 of \cite Pfeiffer2005zm, section 12.3.2 of
 * \cite BaumgarteShapiro or section II.B.1 of \cite Varma2018sqd. In these
 * equations \f$\bar{s}_i\f$ is the conformal surface normal to the apparent
 * horizon, \f$\bar{m}^{ij}=\bar{\gamma}^{ij}-\bar{s}^i\bar{s}^j\f$ is the
 * induced conformal surface metric (denoted \f$\tilde{h}^{ij}\f$ in
 * \cite Pfeiffer2005zm and \cite Varma2018sqd) and \f$\bar{D}\f$ is the
 * covariant derivative w.r.t. to the conformal metric \f$\bar{\gamma}_{ij}\f$.
 * Note that e.g. in \cite Varma2018sqd Eq. (16) appears the surface-normal
 * \f$s^i\f$, not the _conformal_ surface normal \f$\bar{s}^i = \psi^2 s^i\f$.
 * To incur a spin on the apparent horizon we can freely choose the rotational
 * parameters \f$\boldsymbol{\Omega}\f$. Note that for a Kerr solution with
 * dimensionless spin \f$\boldsymbol{\chi}\f$ the rotational parameters at the
 * outer horizon are \f$\boldsymbol{\Omega} =
 * -\frac{\boldsymbol{\chi}}{2r_+}\f$, where \f$r_+ / M = 1 + \sqrt{1 -
 * \chi^2}\f$ (see e.g. Eq. (8) in \cite Ossokine2015yla). \f$\epsilon_{ijk}\f$
 * is the flat-space Levi-Civita symbol.
 *
 * Note that the quasi-equilibrium conditions don't restrict the boundary
 * condition for the lapse. The choice for the lapse boundary condition is made
 * by the `Lapse` input-file options. Currently implemented choices are:
 * - A zero von-Neumann boundary condition:
 *   \f$\bar{s}^k\bar{D}_k(\alpha\psi) = 0\f$
 * - A Dirichlet boundary condition imposing the lapse of an analytic solution.
 *
 * \par Negative-expansion boundary conditions:
 * This class also supports negative-expansion boundary conditions following
 * section II.B.2 of \cite Varma2018sqd by taking a Kerr solution as additional
 * parameter and computing its expansion at the excision surface. Choosing an
 * excision surface _within_ the apparent horizon of the Kerr solution will
 * result in a negative expansion that is added to the boundary condition for
 * the conformal factor. Therefore, the excision surface will lie within an
 * apparent horizon. Specifically, we add the quantity
 * \f$\frac{\psi^3}{4}\Theta_\mathrm{Kerr}\f$ to (\f$\ref{eq:ah_psi}\f$), where
 * \f$\Theta_\mathrm{Kerr}\f$ is the expansion of the specified Kerr solution at
 * the excision surface, and we add the quantity \f$\epsilon = s_i
 * \beta_\mathrm{Kerr}^i - \alpha_\mathrm{Kerr}\f$ to the orthogonal part
 * \f$s_i\beta_\mathrm{excess}^i\f$ of (\f$\ref{eq:ah_beta}\f$).
 *
 * \note When negative-expansion boundary conditions are selected, the Kerr
 * solution gets evaluated at every boundary-condition application. This may
 * turn out to incur a significant computational cost, in which case a future
 * optimization might be to pre-compute the negative-expansion quantities at
 * initialization time and store them in the DataBox.
 */
template <Xcts::Geometry ConformalGeometry>
class ApparentHorizonForBwGW
    : public elliptic::BoundaryConditions::BoundaryCondition<3> {
 private:
  using Base = elliptic::BoundaryConditions::BoundaryCondition<3>;

 public:
  static constexpr Options::String help =
      "Impose the boundary is a quasi-equilibrium apparent horizon.";
  struct MassLeft {
    static constexpr Options::String help = "The mass of the left black hole.";
    using type = double;
  };
  struct MassRight {
    static constexpr Options::String help = "The mass of the right black hole.";
    using type = double;
  };
  struct XCoordsLeft {
    static constexpr Options::String help =
        "The coordinates on the x-axis of the left black hole.";
    using type = double;
  };
  struct XCoordsRight {
    static constexpr Options::String help =
        "The coordinates on the x-axis of the right black hole.";
    using type = double;
  };
  struct AttenuationParameter {
    static constexpr Options::String help =
        "The parameter controlling the width of the attenuation function.";
    using type = double;
  };
  struct OuterRadius {
    static constexpr Options::String help =
        "The radius of the outer boundary of the computational domain.";
    using type = double;
  };
  struct Lapse {
    using type = bool;
    static constexpr Options::String help =
        "Specify an analytic solution to impose a Dirichlet condition on the "
        "lapse. The analytic solution will be evaluated at coordinates "
        "centered at the apparent horizon. "
        "Alternatively, set this option to 'None' "
        "to impose a zero von-Neumann boundary condition on the lapse. Note "
        "that the latter will not result in the standard Kerr-Schild slicing "
        "for a single black hole.";
  };
  struct NegativeExpansion {
    using type = bool;
    static constexpr Options::String help =
        "If the excision surface lies within the solution's "
        "apparent horizon, the imposed expansion will be negative and thus the "
        "excision surface will lie within an apparent horizon. Alternatively, "
        "set this option to 'False' to impose the expansion is zero at the "
        "excision surface, meaning the excision surface _is_ an apparent "
        "horizon.";
  };

  using options =
      tmpl::list<MassLeft, MassRight, XCoordsLeft, XCoordsRight,
                 AttenuationParameter, OuterRadius, Lapse, NegativeExpansion>;

  ApparentHorizonForBwGW() = default;
  ApparentHorizonForBwGW(const ApparentHorizonForBwGW&) = delete;
  ApparentHorizonForBwGW& operator=(const ApparentHorizonForBwGW&) = delete;
  ApparentHorizonForBwGW(ApparentHorizonForBwGW&&) = default;
  ApparentHorizonForBwGW& operator=(ApparentHorizonForBwGW&&) = default;
  ~ApparentHorizonForBwGW() override = default;

  /// \cond
  explicit ApparentHorizonForBwGW(CkMigrateMessage* m) : Base(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ApparentHorizonForBwGW);
  /// \endcond

  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> get_clone()
      const override {
    return std::make_unique<ApparentHorizonForBwGW>(
        mass_left_, mass_right_, xcoord_left_, xcoord_right_,
        attenuation_parameter_, outer_radius_, solution_for_lapse_,
        solution_for_negative_expansion_);
  }

  ApparentHorizonForBwGW(double mass_left, double mass_right,
                         double xcoord_left, double xcoord_right,
                         double attenuation_parameter, double outer_radius,
                         bool solution_for_lapse,
                         bool solution_for_negative_expansion,
                         const Options::Context& context = {});

  std::vector<elliptic::BoundaryConditionType> boundary_condition_types()
      const override {
    return {// Conformal factor
            elliptic::BoundaryConditionType::Neumann,
            // Lapse times conformal factor
            this->lapse_solution_.has_value()
                ? elliptic::BoundaryConditionType::Dirichlet
                : elliptic::BoundaryConditionType::Neumann,
            // Shift
            elliptic::BoundaryConditionType::Dirichlet,
            elliptic::BoundaryConditionType::Dirichlet,
            elliptic::BoundaryConditionType::Dirichlet};
  }

  using argument_tags = tmpl::flatten<tmpl::list<
      domain::Tags::FaceNormal<3>,
      ::Tags::deriv<domain::Tags::UnnormalizedFaceNormal<3>, tmpl::size_t<3>,
                    Frame::Inertial>,
      domain::Tags::UnnormalizedFaceNormalMagnitude<3>,
      domain::Tags::Coordinates<3, Frame::Inertial>,
      gr::Tags::TraceExtrinsicCurvature<DataVector>,
      Tags::ShiftBackground<DataVector, 3, Frame::Inertial>,
      Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<DataVector, 3,
                                                              Frame::Inertial>,
      tmpl::conditional_t<ConformalGeometry == Xcts::Geometry::Curved,
                          tmpl::list<Tags::InverseConformalMetric<
                                         DataVector, 3, Frame::Inertial>,
                                     Tags::ConformalChristoffelSecondKind<
                                         DataVector, 3, Frame::Inertial>>,
                          tmpl::list<>>>>;
  using volume_tags = tmpl::list<>;

  void apply(
      gsl::not_null<Scalar<DataVector>*> conformal_factor_minus_one,
      gsl::not_null<Scalar<DataVector>*> lapse_times_conformal_factor_minus_one,
      gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess,
      gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_lapse_times_conformal_factor_gradient,
      gsl::not_null<tnsr::I<DataVector, 3>*> n_dot_longitudinal_shift_excess,
      const tnsr::i<DataVector, 3>& deriv_conformal_factor_correction,
      const tnsr::i<DataVector, 3>&
          deriv_lapse_times_conformal_factor_correction,
      const tnsr::iJ<DataVector, 3>& deriv_shift_excess_correction,
      const tnsr::i<DataVector, 3>& face_normal,
      const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
      const Scalar<DataVector>& face_normal_magnitude,
      const tnsr::I<DataVector, 3>& x,
      const Scalar<DataVector>& extrinsic_curvature_trace,
      const tnsr::I<DataVector, 3>& shift_background,
      const tnsr::II<DataVector, 3>& longitudinal_shift_background) const;

  void apply(
      gsl::not_null<Scalar<DataVector>*> conformal_factor_minus_one,
      gsl::not_null<Scalar<DataVector>*> lapse_times_conformal_factor_minus_one,
      gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess,
      gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_lapse_times_conformal_factor_gradient,
      gsl::not_null<tnsr::I<DataVector, 3>*> n_dot_longitudinal_shift_excess,
      const tnsr::i<DataVector, 3>& deriv_conformal_factor_correction,
      const tnsr::i<DataVector, 3>&
          deriv_lapse_times_conformal_factor_correction,
      const tnsr::iJ<DataVector, 3>& deriv_shift_excess_correction,
      const tnsr::i<DataVector, 3>& face_normal,
      const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
      const Scalar<DataVector>& face_normal_magnitude,
      const tnsr::I<DataVector, 3>& x,
      const Scalar<DataVector>& extrinsic_curvature_trace,
      const tnsr::I<DataVector, 3>& shift_background,
      const tnsr::II<DataVector, 3>& longitudinal_shift_background,
      const tnsr::II<DataVector, 3>& inv_conformal_metric,
      const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind) const;

  using argument_tags_linearized = tmpl::flatten<tmpl::list<
      ::Tags::Normalized<
          domain::Tags::UnnormalizedFaceNormal<3, Frame::Inertial>>,
      ::Tags::deriv<domain::Tags::UnnormalizedFaceNormal<3, Frame::Inertial>,
                    tmpl::size_t<3>, Frame::Inertial>,
      ::Tags::Magnitude<
          domain::Tags::UnnormalizedFaceNormal<3, Frame::Inertial>>,
      domain::Tags::Coordinates<3, Frame::Inertial>,
      gr::Tags::TraceExtrinsicCurvature<DataVector>,
      Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<DataVector, 3,
                                                              Frame::Inertial>,
      Tags::ConformalFactorMinusOne<DataVector>,
      Tags::LapseTimesConformalFactorMinusOne<DataVector>,
      ::Tags::NormalDotFlux<Tags::ShiftExcess<DataVector, 3, Frame::Inertial>>,
      tmpl::conditional_t<ConformalGeometry == Xcts::Geometry::Curved,
                          tmpl::list<Tags::InverseConformalMetric<
                                         DataVector, 3, Frame::Inertial>,
                                     Tags::ConformalChristoffelSecondKind<
                                         DataVector, 3, Frame::Inertial>>,
                          tmpl::list<>>>>;
  using volume_tags_linearized = tmpl::list<>;

  void apply_linearized(
      gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
      gsl::not_null<Scalar<DataVector>*>
          lapse_times_conformal_factor_correction,
      gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess_correction,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_conformal_factor_gradient_correction,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_lapse_times_conformal_factor_gradient_correction,
      gsl::not_null<tnsr::I<DataVector, 3>*>
          n_dot_longitudinal_shift_excess_correction,
      const tnsr::i<DataVector, 3>& deriv_conformal_factor_correction,
      const tnsr::i<DataVector, 3>&
          deriv_lapse_times_conformal_factor_correction,
      const tnsr::iJ<DataVector, 3>& deriv_shift_excess_correction,
      const tnsr::i<DataVector, 3>& face_normal,
      const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
      const Scalar<DataVector>& face_normal_magnitude,
      const tnsr::I<DataVector, 3>& x,
      const Scalar<DataVector>& extrinsic_curvature_trace,
      const tnsr::II<DataVector, 3>& longitudinal_shift_background,
      const Scalar<DataVector>& conformal_factor_minus_one,
      const Scalar<DataVector>& lapse_times_conformal_factor_minus_one,
      const tnsr::I<DataVector, 3>& n_dot_longitudinal_shift_excess) const;

  void apply_linearized(
      gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
      gsl::not_null<Scalar<DataVector>*>
          lapse_times_conformal_factor_correction,
      gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess_correction,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_conformal_factor_gradient_correction,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_lapse_times_conformal_factor_gradient_correction,
      gsl::not_null<tnsr::I<DataVector, 3>*>
          n_dot_longitudinal_shift_excess_correction,
      const tnsr::i<DataVector, 3>& deriv_conformal_factor_correction,
      const tnsr::i<DataVector, 3>&
          deriv_lapse_times_conformal_factor_correction,
      const tnsr::iJ<DataVector, 3>& deriv_shift_excess_correction,
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
      const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  double mass_left_ = std::numeric_limits<double>::signaling_NaN();
  double mass_right_ = std::numeric_limits<double>::signaling_NaN();
  double xcoord_left_ = std::numeric_limits<double>::signaling_NaN();
  double xcoord_right_ = std::numeric_limits<double>::signaling_NaN();
  double attenuation_parameter_ = std::numeric_limits<double>::signaling_NaN();
  double outer_radius_ = std::numeric_limits<double>::signaling_NaN();
  bool write_evolution_option_ = false;
  bool solution_for_lapse_ = false;
  bool solution_for_negative_expansion_ = true;

  std::optional<
      std::unique_ptr<Xcts::AnalyticData::BinaryWithGravitationalWaves>>
      lapse_solution_{};

  std::optional<
      std::unique_ptr<Xcts::AnalyticData::BinaryWithGravitationalWaves>>
      negative_expansion_solution_{};
};

}  // namespace Xcts::BoundaryConditions
