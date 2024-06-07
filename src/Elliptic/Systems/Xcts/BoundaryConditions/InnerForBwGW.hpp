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
 * \brief Impose flat spacetime at this boundary
 *
 * Impose \f$\psi=1\f$, \f$\alpha\psi=1\f$, \f$\beta_\mathrm{excess}^i=0\f$ on
 * this boundary, where \f$\psi\f$ is the conformal factor, \f$\alpha\f$ is the
 * lapse and \f$\beta_\mathrm{excess}^i=\beta^i-\beta_\mathrm{background}^i\f$
 * is the shift excess (see `Xcts::Tags::ShiftExcess` for details on the split
 * of the shift in background and excess). Note that this choice only truly
 * represents InnerForBwGW if the conformal background metric is flat.
 *
 */
template <Xcts::Geometry ConformalGeometry>
class InnerForBwGW : public elliptic::BoundaryConditions::BoundaryCondition<3> {
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
  struct OuterRadius {
    static constexpr Options::String help =
        "The radius of the outer boundary of the computational domain.";
    using type = double;
  };
  struct BoundaryCondition {
    static constexpr Options::String help = "Boundary Condition Type.";
    using type = elliptic::BoundaryConditionType;
  };
  using options =
      tmpl::list<MassLeft, MassRight, XCoordsLeft, XCoordsRight,
                 AttenuationParameter, OuterRadius, BoundaryCondition>;
  static constexpr Options::String help =
      "Impose Schwarzschild isotropic in Inner Boundary.";

  InnerForBwGW() = default;
  InnerForBwGW(const InnerForBwGW&) = delete;
  InnerForBwGW& operator=(const InnerForBwGW&) = delete;
  InnerForBwGW(InnerForBwGW&&) = default;
  InnerForBwGW& operator=(InnerForBwGW&&) = default;
  ~InnerForBwGW() = default;

  /// \cond
  explicit InnerForBwGW(CkMigrateMessage* m) : Base(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(InnerForBwGW);
  /// \endcond

  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> get_clone()
      const override {
    return std::make_unique<InnerForBwGW>(mass_left_, mass_right_, xcoord_left_,
                                          xcoord_right_, attenuation_parameter_,
                                          outer_radius_, boundary_);
  }

  InnerForBwGW(double mass_left, double mass_right, double xcoord_left,
               double xcoord_right, double attenuation_parameter,
               double outer_radius, elliptic::BoundaryConditionType boundary,
               const Options::Context& context = {});

  std::vector<elliptic::BoundaryConditionType> boundary_condition_types()
      const override {
    return {// Conformal factor
            boundary_,
            // Lapse times conformal factor
            boundary_,
            // Shift
            boundary_, boundary_, boundary_};
  }

  using argument_tags =
      tmpl::list<domain::Tags::Coordinates<3, Frame::Inertial>,
                 ::Tags::Normalized<
                     domain::Tags::UnnormalizedFaceNormal<3, Frame::Inertial>>>;
  using volume_tags = tmpl::list<>;

  void apply(
      gsl::not_null<Scalar<DataVector>*> conformal_factor_minus_one,
      gsl::not_null<Scalar<DataVector>*> lapse_times_conformal_factor_minus_one,
      gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess,
      gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_lapse_times_conformal_factor_gradient,
      gsl::not_null<tnsr::I<DataVector, 3>*> n_dot_longitudinal_shift_excess,
      const tnsr::I<DataVector, 3>& x,
      const tnsr::i<DataVector, 3>& face_normal) const;

  using argument_tags_linearized = tmpl::list<>;
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
          n_dot_longitudinal_shift_excess_correction) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  double mass_left_{std::numeric_limits<double>::signaling_NaN()};
  double mass_right_{std::numeric_limits<double>::signaling_NaN()};
  double xcoord_left_{std::numeric_limits<double>::signaling_NaN()};
  double xcoord_right_{std::numeric_limits<double>::signaling_NaN()};
  double attenuation_parameter_{std::numeric_limits<double>::signaling_NaN()};
  double outer_radius_{std::numeric_limits<double>::signaling_NaN()};
  elliptic::BoundaryConditionType boundary_{};

  std::optional<
      std::unique_ptr<Xcts::AnalyticData::BinaryWithGravitationalWaves>>
      solution_{};
};

}  // namespace Xcts::BoundaryConditions
