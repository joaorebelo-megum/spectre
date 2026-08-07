// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <pup.h>
#include <string>
#include <vector>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Elliptic/Systems/Xcts/FluxesAndSources.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "IO/Exporter/PointwiseInterpolator.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace Xcts::BoundaryConditions {

namespace detail {
/// Fields read from the volume data of a previous solve to build the boundary
/// values. Only pointwise data is needed -- no derivatives -- because this
/// boundary condition is purely Dirichlet.
using numeric_boundary_load_tags =
    tmpl::list<gr::Tags::SpatialMetric<DataVector, 3>,
               gr::Tags::Lapse<DataVector>,
               Xcts::Tags::ShiftExcess<DataVector, 3, Frame::Inertial>>;
}  // namespace detail

/*!
 * \brief Impose the solution of a previous XCTS solve on this boundary
 *
 * Imposes Dirichlet conditions on all solved-for fields, taking their values
 * from the volume data of a previous solve:
 *
 * \f{align}
 * \psi &= \left(\det\gamma_{ij}\right)^{1/12} \\
 * \alpha\psi &= \alpha \left(\det\gamma_{ij}\right)^{1/12} \\
 * \beta_\mathrm{excess}^i &= \beta_\mathrm{excess}^i \big|_\mathrm{loaded}
 * \f}
 *
 * where \f$\gamma_{ij}\f$, \f$\alpha\f$ and \f$\beta_\mathrm{excess}^i\f$ are
 * read from the file. The conformal factor follows from the unimodular gauge
 * \f$\det\bar{\gamma}_{ij}=1\f$, matching
 * `Xcts::AnalyticData::NumericBinaryWithWaves`, so that the boundary values are
 * consistent with the background built from the same data.
 *
 * This is the boundary counterpart of loading a previous solve as background
 * and initial guess. It is useful where the excision surfaces are no longer
 * apparent horizons of the new free data -- for instance once gravitational
 * wave content is added -- so that `Xcts::BoundaryConditions::ApparentHorizon`
 * would impose a condition inconsistent with the data being loaded.
 *
 * \note Only pointwise values are needed, so unlike
 * `elliptic::BoundaryConditions::AnalyticSolution` this does not require the
 * fluxes and therefore works without a mesh on the face.
 *
 * \warning The loaded `ShiftExcess` is imposed on \f$\beta_\mathrm{excess}\f$
 * directly, so the background shift of this solve must match the one of the
 * solve that produced the data. Otherwise the *full* shift at the boundary
 * differs, which shows up as a spurious spin on the horizon.
 *
 * \tparam EnabledEquations The subset of XCTS equations that are being solved
 */
template <Xcts::Equations EnabledEquations>
class NumericData : public elliptic::BoundaryConditions::BoundaryCondition<3> {
 private:
  using Base = elliptic::BoundaryConditions::BoundaryCondition<3>;

 public:
  struct DataFile {
    static constexpr Options::String help =
        "Path or glob pattern to the volume data of the previous solve";
    using type = std::string;
  };
  struct Subgroup {
    static constexpr Options::String help =
        "The subgroup within the volume data file, excluding extensions";
    using type = std::string;
  };
  struct ObservationStep {
    static constexpr Options::String help =
        "The observation step at which to read the data. Use -1 for the last.";
    using type = int;
    static int suggested_value() { return -1; }
  };
  struct ExtrapolateIntoExcisions {
    static constexpr Options::String help =
        "Whether to extrapolate the loaded data into excised regions. The "
        "boundary points sit on the excision surface itself, so this is "
        "usually only needed if the surfaces of the two solves differ.";
    using type = bool;
    static bool suggested_value() { return false; }
  };
  using options =
      tmpl::list<DataFile, Subgroup, ObservationStep, ExtrapolateIntoExcisions>;
  static constexpr Options::String help =
      "Impose the solution of a previous XCTS solve on this boundary.";

  NumericData() = default;
  // The interpolator holds a `Domain`, which is move-only, so copies reload
  // the data from file. Same pattern as `ray_tracing::NumericData`.
  NumericData(const NumericData& rhs);
  NumericData& operator=(const NumericData& rhs);
  NumericData(NumericData&&) = default;
  NumericData& operator=(NumericData&&) = default;
  ~NumericData() = default;

  NumericData(std::string data_file, std::string subgroup, int observation_step,
              bool extrapolate_into_excisions);

  /// \cond
  explicit NumericData(CkMigrateMessage* m) : Base(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(NumericData);
  /// \endcond

  std::unique_ptr<domain::BoundaryConditions::BoundaryCondition> get_clone()
      const override {
    return std::make_unique<NumericData>(*this);
  }

  std::vector<elliptic::BoundaryConditionType> boundary_condition_types()
      const override {
    return {[]() {
              if constexpr (EnabledEquations == Xcts::Equations::Hamiltonian) {
                return 1;
              } else if constexpr (EnabledEquations ==
                                   Xcts::Equations::HamiltonianAndLapse) {
                return 2;
              } else {
                return 5;
              }
            }(),
            elliptic::BoundaryConditionType::Dirichlet};
  }

  using argument_tags =
      tmpl::list<domain::Tags::Coordinates<3, Frame::Inertial>>;
  using volume_tags = tmpl::list<>;

  void apply(gsl::not_null<Scalar<DataVector>*> conformal_factor_minus_one,
             gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
             const tnsr::i<DataVector, 3>& deriv_conformal_factor,
             const tnsr::I<DataVector, 3>& x) const;

  void apply(
      gsl::not_null<Scalar<DataVector>*> conformal_factor_minus_one,
      gsl::not_null<Scalar<DataVector>*> lapse_times_conformal_factor_minus_one,
      gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_lapse_times_conformal_factor_gradient,
      const tnsr::i<DataVector, 3>& deriv_conformal_factor,
      const tnsr::i<DataVector, 3>& deriv_lapse_times_conformal_factor,
      const tnsr::I<DataVector, 3>& x) const;

  void apply(
      gsl::not_null<Scalar<DataVector>*> conformal_factor_minus_one,
      gsl::not_null<Scalar<DataVector>*> lapse_times_conformal_factor_minus_one,
      gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess,
      gsl::not_null<Scalar<DataVector>*> n_dot_conformal_factor_gradient,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_lapse_times_conformal_factor_gradient,
      gsl::not_null<tnsr::I<DataVector, 3>*> n_dot_longitudinal_shift_excess,
      const tnsr::i<DataVector, 3>& deriv_conformal_factor,
      const tnsr::i<DataVector, 3>& deriv_lapse_times_conformal_factor,
      const tnsr::iJ<DataVector, 3>& deriv_shift_excess,
      const tnsr::I<DataVector, 3>& x) const;

  using argument_tags_linearized = tmpl::list<>;
  using volume_tags_linearized = tmpl::list<>;

  static void apply_linearized(
      gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_conformal_factor_gradient_correction,
      const tnsr::i<DataVector, 3>& deriv_conformal_factor_correction);

  static void apply_linearized(
      gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
      gsl::not_null<Scalar<DataVector>*>
          lapse_times_conformal_factor_correction,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_conformal_factor_gradient_correction,
      gsl::not_null<Scalar<DataVector>*>
          n_dot_lapse_times_conformal_factor_gradient_correction,
      const tnsr::i<DataVector, 3>& deriv_conformal_factor_correction,
      const tnsr::i<DataVector, 3>&
          deriv_lapse_times_conformal_factor_correction);

  static void apply_linearized(
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
      const tnsr::iJ<DataVector, 3>& deriv_shift_excess_correction);

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

  const std::string& data_file() const { return data_file_; }
  const std::string& subgroup() const { return subgroup_; }
  int observation_step() const { return observation_step_; }
  bool extrapolate_into_excisions() const {
    return extrapolate_into_excisions_;
  }

 private:
  /// The boundary values reconstructed from the loaded data at the points `x`
  struct BoundaryValues {
    Scalar<DataVector> conformal_factor_minus_one;
    Scalar<DataVector> lapse_times_conformal_factor_minus_one;
    tnsr::I<DataVector, 3> shift_excess;
  };
  BoundaryValues boundary_values(const tnsr::I<DataVector, 3>& x) const;

  /// Read the volume data into memory. Not thread safe unless HDF5 was built
  /// with thread-safety support, so this happens once per node when the object
  /// is unpacked.
  void load_interpolator();

  std::string data_file_{};
  std::string subgroup_{};
  int observation_step_{-1};
  bool extrapolate_into_excisions_{false};
  // Deliberately not serialized; reloaded on unpacking.
  spectre::Exporter::PointwiseInterpolator<3, Frame::Inertial> interpolator_{};
};

template <Xcts::Equations EnabledEquations>
bool operator==(const NumericData<EnabledEquations>& lhs,
                const NumericData<EnabledEquations>& rhs);

template <Xcts::Equations EnabledEquations>
bool operator!=(const NumericData<EnabledEquations>& lhs,
                const NumericData<EnabledEquations>& rhs);

}  // namespace Xcts::BoundaryConditions
