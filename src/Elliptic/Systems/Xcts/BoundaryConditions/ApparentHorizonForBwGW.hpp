// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>
#include <pup.h>
#include <string>
#include <vector>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/FaceNormal.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Elliptic/Systems/Xcts/FluxesAndSources.hpp"
#include "Elliptic/Systems/Xcts/Geometry.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Options/Auto.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/BinaryWithGravitationalWaves.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace Xcts::BoundaryConditions {

/*!
 * \brief Impose apparent horizon boundary conditions to
 * BinaryWithGravitationalWaves
 *
 */
template <Xcts::Geometry ConformalGeometry>
class ApparentHorizonForBwGW
    : public elliptic::BoundaryConditions::BoundaryCondition<3> {
 private:
  using Base = elliptic::BoundaryConditions::BoundaryCondition<3>;

 public:
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
  struct AttenuationRadius {
    static constexpr Options::String help =
        "The parameter controlling the transition center of the attenuation "
        "function.";
    using type = double;
  };
  struct OuterRadius {
    static constexpr Options::String help =
        "The radius of the outer boundary of the computational domain.";
    using type = double;
  };
  struct LeftOrRight {
    static constexpr Options::String help =
        "'True' for left black hole boundary and 'False' for right black hole "
        "boundary.";
    using type = bool;
  };
  using options = tmpl::list<MassLeft, MassRight, XCoordsLeft, XCoordsRight,
                             AttenuationParameter, AttenuationRadius,
                             OuterRadius, LeftOrRight>;
  static constexpr Options::String help =
      "Impose Schwarzschild isotropic in Inner Boundary.";

  ApparentHorizonForBwGW() = default;
  ApparentHorizonForBwGW(const ApparentHorizonForBwGW&) = delete;
  ApparentHorizonForBwGW& operator=(const ApparentHorizonForBwGW&) = delete;
  ApparentHorizonForBwGW(ApparentHorizonForBwGW&&) = default;
  ApparentHorizonForBwGW& operator=(ApparentHorizonForBwGW&&) = default;
  ~ApparentHorizonForBwGW() = default;

  /// \cond
  explicit ApparentHorizonForBwGW(CkMigrateMessage* m) : Base(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(ApparentHorizonForBwGW);
  /// \endcond

  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> get_clone()
      const override {
    return std::make_unique<ApparentHorizonForBwGW>(
        mass_left_, mass_right_, xcoord_left_, xcoord_right_,
        attenuation_parameter_, attenuation_radius_, outer_radius_,
        left_or_right_);
  }

  ApparentHorizonForBwGW(double mass_left, double mass_right,
                         double xcoord_left, double xcoord_right,
                         double attenuation_parameter,
                         double attenuation_radius, double outer_radius,
                         bool left_or_right,
                         const Options::Context& context = {});

  std::vector<elliptic::BoundaryConditionType> boundary_condition_types()
      const override {
    return {// Conformal factor
            elliptic::BoundaryConditionType::Neumann,
            // Lapse times conformal factor
            elliptic::BoundaryConditionType::Neumann,
            // Shift
            elliptic::BoundaryConditionType::Dirichlet,
            elliptic::BoundaryConditionType::Dirichlet,
            elliptic::BoundaryConditionType::Dirichlet};
  }

  using argument_tags =
      tmpl::list<domain::Tags::Coordinates<3, Frame::Inertial>,
                 domain::Tags::FaceNormal<3>,
                 ::Tags::deriv<domain::Tags::UnnormalizedFaceNormal<3>,
                               tmpl::size_t<3>, Frame::Inertial>,
                 domain::Tags::UnnormalizedFaceNormalMagnitude<3>>;
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
      const tnsr::I<DataVector, 3>& x,
      const tnsr::i<DataVector, 3>& face_normal,
      const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
      const Scalar<DataVector>& face_normal_magnitude) const;

  using argument_tags_linearized =
      tmpl::list<domain::Tags::Coordinates<3, Frame::Inertial>,
                 domain::Tags::FaceNormal<3>,
                 ::Tags::deriv<domain::Tags::UnnormalizedFaceNormal<3>,
                               tmpl::size_t<3>, Frame::Inertial>,
                 domain::Tags::UnnormalizedFaceNormalMagnitude<3>>;
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
      const tnsr::I<DataVector, 3>& x,
      const tnsr::i<DataVector, 3>& face_normal,
      const tnsr::ij<DataVector, 3>& deriv_unnormalized_face_normal,
      const Scalar<DataVector>& face_normal_magnitude) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  double mass_left_{std::numeric_limits<double>::signaling_NaN()};
  double mass_right_{std::numeric_limits<double>::signaling_NaN()};
  double xcoord_left_{std::numeric_limits<double>::signaling_NaN()};
  double xcoord_right_{std::numeric_limits<double>::signaling_NaN()};
  double attenuation_parameter_{std::numeric_limits<double>::signaling_NaN()};
  double attenuation_radius_{std::numeric_limits<double>::signaling_NaN()};
  double outer_radius_{std::numeric_limits<double>::signaling_NaN()};
  bool left_or_right_{false};

  std::optional<
      std::unique_ptr<Xcts::AnalyticData::BinaryWithGravitationalWaves>>
      solution_{};
};

}  // namespace Xcts::BoundaryConditions
