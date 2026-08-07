// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "DataStructures/CachedTempBuffer.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "IO/Exporter/PointwiseInterpolator.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/CommonVariables.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags/Conformal.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialGuess.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Xcts::AnalyticData {

namespace detail {

/*!
 * \brief Fields loaded from the volume data of a previous XCTS solve.
 *
 * \note `Xcts::Tags::ShiftExcess` is loaded rather than `gr::Tags::Shift`. The
 * observed `Shift` is the *full* corotating shift
 * \f$\beta^i = \beta^i_\mathrm{background} + \beta^i_\mathrm{excess}\f$
 * (see `Xcts::SpacetimeQuantities`), which grows like \f$\propto r\f$ and is
 * therefore badly resolved in the outer shell. The excess shift is
 * asymptotically small, which is exactly why the previous solve writes it.
 */
template <typename DataType>
using numeric_load_tags =
    tmpl::list<gr::Tags::SpatialMetric<DataType, 3>, gr::Tags::Lapse<DataType>,
               Tags::ShiftExcess<DataType, 3, Frame::Inertial>,
               gr::Tags::ExtrinsicCurvature<DataType, 3>>;

template <typename DataType>
using NumericBinaryWithWavesVariablesCache =
    cached_temp_buffer_from_typelist<tmpl::append<
        common_tags<DataType>,
        tmpl::list<
            // Physical fields loaded from the previous solve
            gr::Tags::SpatialMetric<DataType, 3>,
            gr::Tags::InverseSpatialMetric<DataType, 3>,
            gr::Tags::Lapse<DataType>,
            gr::Tags::ExtrinsicCurvature<DataType, 3>,
            // Time derivative of the conformal metric, from the helical
            // Killing vector (see report 004)
            ::Tags::dt<Tags::ConformalMetric<DataType, 3, Frame::Inertial>>,
            // Vacuum matter sources
            gr::Tags::Conformal<gr::Tags::EnergyDensity<DataType>, 0>,
            gr::Tags::Conformal<gr::Tags::StressTrace<DataType>, 0>,
            gr::Tags::Conformal<gr::Tags::MomentumDensity<DataType, 3>, 0>,
            // Initial guesses
            Tags::ConformalFactorMinusOne<DataType>,
            Tags::LapseTimesConformalFactorMinusOne<DataType>,
            Tags::ShiftExcess<DataType, 3, Frame::Inertial>>,
        hydro_tags<DataType>>>;

template <typename DataType>
struct NumericBinaryWithWavesVariables
    : CommonVariables<DataType,
                      NumericBinaryWithWavesVariablesCache<DataType>> {
  static constexpr size_t Dim = 3;
  using Cache = NumericBinaryWithWavesVariablesCache<DataType>;
  using Base = CommonVariables<DataType, Cache>;
  using Base::operator();

  NumericBinaryWithWavesVariables(
      std::optional<std::reference_wrapper<const Mesh<Dim>>> local_mesh,
      std::optional<std::reference_wrapper<const InverseJacobian<
          DataType, Dim, Frame::ElementLogical, Frame::Inertial>>>
          local_inv_jacobian,
      const tnsr::I<DataType, Dim, Frame::Inertial>& local_x,
      double local_angular_velocity, double local_expansion,
      tuples::tagged_tuple_from_typelist<numeric_load_tags<DataType>>
          local_loaded_vars)
      : Base(std::move(local_mesh), std::move(local_inv_jacobian)),
        x(local_x),
        angular_velocity(local_angular_velocity),
        expansion(local_expansion),
        loaded_vars(std::move(local_loaded_vars)) {}

  const tnsr::I<DataType, Dim, Frame::Inertial>& x;
  double angular_velocity;
  double expansion;
  tuples::tagged_tuple_from_typelist<numeric_load_tags<DataType>> loaded_vars;

  // Fields taken straight from the loaded data
  void operator()(gsl::not_null<tnsr::ii<DataType, Dim>*> spatial_metric,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::SpatialMetric<DataType, Dim> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::II<DataType, Dim>*> inv_spatial_metric,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::InverseSpatialMetric<DataType, Dim> /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataType>*> lapse,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::Lapse<DataType> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::ii<DataType, Dim>*> extrinsic_curvature,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::ExtrinsicCurvature<DataType, Dim> /*meta*/) const;

  // Time derivative of the conformal metric from the helical Killing vector
  void operator()(gsl::not_null<tnsr::ii<DataType, Dim>*> dt_conformal_metric,
                  gsl::not_null<Cache*> cache,
                  ::Tags::dt<Tags::ConformalMetric<
                      DataType, Dim, Frame::Inertial>> /*meta*/) const;

  // The six quantities CommonVariables leaves pure virtual
  void operator()(
      gsl::not_null<tnsr::ii<DataType, Dim>*> conformal_metric,
      gsl::not_null<Cache*> cache,
      Tags::ConformalMetric<DataType, Dim, Frame::Inertial> /*meta*/)
      const override;
  void operator()(
      gsl::not_null<tnsr::ijj<DataType, Dim>*> deriv_conformal_metric,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<Tags::ConformalMetric<DataType, Dim, Frame::Inertial>,
                    tmpl::size_t<Dim>, Frame::Inertial> /*meta*/)
      const override;
  void operator()(
      gsl::not_null<Scalar<DataType>*> extrinsic_curvature_trace,
      gsl::not_null<Cache*> cache,
      gr::Tags::TraceExtrinsicCurvature<DataType> /*meta*/) const override;
  void operator()(
      gsl::not_null<Scalar<DataType>*> dt_extrinsic_curvature_trace,
      gsl::not_null<Cache*> cache,
      ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataType>> /*meta*/)
      const override;
  void operator()(
      gsl::not_null<tnsr::I<DataType, Dim>*> shift_background,
      gsl::not_null<Cache*> cache,
      Tags::ShiftBackground<DataType, Dim, Frame::Inertial> /*meta*/)
      const override;
  void operator()(
      gsl::not_null<tnsr::iJ<DataType, Dim>*> deriv_shift_background,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<Tags::ShiftBackground<DataType, Dim, Frame::Inertial>,
                    tmpl::size_t<Dim>, Frame::Inertial> /*meta*/)
      const override;
  void operator()(gsl::not_null<tnsr::II<DataType, Dim, Frame::Inertial>*>
                      longitudinal_shift_background,
                  gsl::not_null<Cache*> cache,
                  Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
                      DataType, Dim, Frame::Inertial> /*meta*/) const override;

  // Vacuum matter sources
  void operator()(
      gsl::not_null<Scalar<DataType>*> conformal_energy_density,
      gsl::not_null<Cache*> cache,
      gr::Tags::Conformal<gr::Tags::EnergyDensity<DataType>, 0> /*meta*/) const;
  void operator()(
      gsl::not_null<Scalar<DataType>*> conformal_stress_trace,
      gsl::not_null<Cache*> cache,
      gr::Tags::Conformal<gr::Tags::StressTrace<DataType>, 0> /*meta*/) const;
  void operator()(
      gsl::not_null<tnsr::I<DataType, Dim>*> conformal_momentum_density,
      gsl::not_null<Cache*> cache,
      gr::Tags::Conformal<gr::Tags::MomentumDensity<DataType, Dim>, 0> /*meta*/)
      const;

  // Initial guesses, taken from the loaded solution
  void operator()(gsl::not_null<Scalar<DataType>*> conformal_factor_minus_one,
                  gsl::not_null<Cache*> cache,
                  Tags::ConformalFactorMinusOne<DataType> /*meta*/) const;
  void operator()(
      gsl::not_null<Scalar<DataType>*> lapse_times_conformal_factor_minus_one,
      gsl::not_null<Cache*> cache,
      Tags::LapseTimesConformalFactorMinusOne<DataType> /*meta*/) const;
  void operator()(
      gsl::not_null<tnsr::I<DataType, Dim>*> shift_excess,
      gsl::not_null<Cache*> cache,
      Tags::ShiftExcess<DataType, Dim, Frame::Inertial> /*meta*/) const;

  // Vacuum hydro quantities. `Xcts::Tags::HydroQuantitiesCompute` requests
  // these from every background, so they must be provided even though this is
  // a vacuum system.
  void operator()(const gsl::not_null<Scalar<DataType>*> rest_mass_density,
                  const gsl::not_null<Cache*> /*cache*/,
                  hydro::Tags::RestMassDensity<DataType> /*meta*/) const {
    get(*rest_mass_density) = make_with_value<DataType>(get<0>(x), 0.);
  }
  void operator()(const gsl::not_null<Scalar<DataType>*> specific_enthalpy,
                  const gsl::not_null<Cache*> /*cache*/,
                  hydro::Tags::SpecificEnthalpy<DataType> /*meta*/) const {
    get(*specific_enthalpy) = make_with_value<DataType>(get<0>(x), 1.);
  }
  void operator()(const gsl::not_null<Scalar<DataType>*> pressure,
                  const gsl::not_null<Cache*> /*cache*/,
                  hydro::Tags::Pressure<DataType> /*meta*/) const {
    get(*pressure) = make_with_value<DataType>(get<0>(x), 0.);
  }
  void operator()(const gsl::not_null<tnsr::I<DataType, Dim>*> spatial_velocity,
                  const gsl::not_null<Cache*> /*cache*/,
                  hydro::Tags::SpatialVelocity<DataType, Dim> /*meta*/) const {
    std::fill(spatial_velocity->begin(), spatial_velocity->end(), 0.);
  }
  void operator()(const gsl::not_null<Scalar<DataType>*> lorentz_factor,
                  const gsl::not_null<Cache*> /*cache*/,
                  hydro::Tags::LorentzFactor<DataType> /*meta*/) const {
    get(*lorentz_factor) = make_with_value<DataType>(get<0>(x), 1.);
  }
  void operator()(const gsl::not_null<tnsr::I<DataType, Dim>*> magnetic_field,
                  const gsl::not_null<Cache*> /*cache*/,
                  hydro::Tags::MagneticField<DataType, Dim> /*meta*/) const {
    std::fill(magnetic_field->begin(), magnetic_field->end(), 0.);
  }

  /// The helical Killing vector \f$\xi^i\f$ of the previous solve, evaluated at
  /// the grid points. See `deriv_killing_vector` for its (constant) gradient.
  tnsr::I<DataType, Dim> killing_vector() const;
  /// \f$\partial_i \xi^j\f$, a constant matrix
  tnsr::iJ<DataType, Dim> deriv_killing_vector() const;
};

}  // namespace detail

/*!
 * \brief XCTS background built from the volume data of a previous solve, with
 * post-Newtonian gravitational waves added
 *
 * This class implements the second stage of an incremental construction of
 * binary-black-hole initial data with realistic wave content:
 *
 * 1. A standard BBH XCTS solve produces a solution in corotating coordinates.
 * 2. This class loads that solution and uses it as both background and initial
 *    guess, adding post-Newtonian wave content on top of the conformal metric,
 *    and the XCTS equations are solved again.
 *
 * ## Frame
 *
 * The previous solve is performed in coordinates that corotate with the binary,
 * so that \f$\partial_t\bar{\gamma}_{ij} = 0\f$ and \f$\partial_t K = 0\f$,
 * with a background shift
 *
 * \f{equation}
 * \xi^i = \left(-\Omega y + \dot{a} x,\; \Omega x + \dot{a} y,\; \dot{a}
 * z\right)
 * \f}
 *
 * This class instead works in the **inertial** frame, where
 * \f$\beta^i_\mathrm{background} = 0\f$ and the solved-for shift is the excess
 * shift of the previous solve. The two are equivalent for the XCTS equations,
 * which see the shift only through
 * \f$\left(\bar{L}\beta\right)^{ij} +
 * \left(\bar{L}\beta_\mathrm{bg}\right)^{ij}
 * - \bar{u}^{ij}\f$: since \f$\mathcal{L}_\xi\bar{\gamma}_{ij} =
 * \bar{D}_i\xi_j + \bar{D}_j\xi_i\f$, the trace-free raised part of
 * \f$-\partial_t\bar{\gamma}\f$ is exactly \f$-\left(\bar{L}\xi\right)^{ij}\f$,
 * so the inertial-frame combination reproduces the corotating one.
 *
 * \warning Load `ShiftExcess`, not `Shift`. The observed `Shift` is the full
 * corotating shift \f$\beta^i_\mathrm{bg} + \beta^i_\mathrm{excess}\f$, which
 * grows \f$\propto r\f$ and is badly resolved in the outer shell; using it as
 * the excess shift also double-counts \f$\left(\bar{L}\xi\right)^{ij}\f$.
 *
 * ## Conformal decomposition
 *
 * The physical fields \f$\gamma_{ij}\f$, \f$\alpha\f$,
 * \f$\beta^i_\mathrm{excess}\f$ and \f$K_{ij}\f$ are loaded from the volume
 * data. The conformal decomposition is fixed by the unimodular gauge
 * \f$\det\bar{\gamma}_{ij} = 1\f$, which determines
 *
 * \f{align}
 * \psi = \left(\det\gamma_{ij}\right)^{1/12}, \quad
 * \bar{\gamma}_{ij} = \left(\det\gamma_{ij}\right)^{-1/3}\gamma_{ij}
 * \f}
 *
 * uniquely from the loaded data. The wave content is then added as
 * \f$\bar{\gamma}_{ij} \to \bar{\gamma}_{ij} + f_\mathrm{att} h^{TT}_{ij}\f$.
 *
 * The trace of the extrinsic curvature is taken directly from the loaded
 * \f$K_{ij}\f$ as \f$K = \gamma^{ij}K_{ij}\f$, which involves no derivatives
 * and is frame independent.
 *
 * \note The wave content is not implemented yet; this class currently
 * reproduces the loaded solution. That makes it testable on its own: solving
 * the XCTS equations with this background should return the data that was
 * loaded.
 */
class NumericBinaryWithWaves : public elliptic::analytic_data::Background,
                               public elliptic::analytic_data::InitialGuess {
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
        "Whether to extrapolate the loaded data into excised regions";
    using type = bool;
    static bool suggested_value() { return false; }
  };
  struct AngularVelocity {
    static constexpr Options::String help =
        "Orbital angular velocity of the previous solve. Together with "
        "'Expansion' this defines the helical Killing vector along which the "
        "previous solution is stationary, and hence the time derivative of "
        "the conformal metric in the inertial frame.";
    using type = double;
  };
  struct Expansion {
    static constexpr Options::String help =
        "Radial expansion velocity of the previous solve";
    using type = double;
  };
  using options =
      tmpl::list<DataFile, Subgroup, ObservationStep, ExtrapolateIntoExcisions,
                 AngularVelocity, Expansion>;
  static constexpr Options::String help =
      "Binary black hole initial data built from a previous XCTS solve, with "
      "post-Newtonian gravitational waves added.";

  NumericBinaryWithWaves() = default;
  // The interpolator holds a `Domain`, which is move-only, so it cannot be
  // copied. Copies reload the data from file instead, following
  // `ray_tracing::NumericData`.
  NumericBinaryWithWaves(const NumericBinaryWithWaves& rhs);
  NumericBinaryWithWaves& operator=(const NumericBinaryWithWaves& rhs);
  NumericBinaryWithWaves(NumericBinaryWithWaves&&) = default;
  NumericBinaryWithWaves& operator=(NumericBinaryWithWaves&&) = default;
  ~NumericBinaryWithWaves() override = default;

  NumericBinaryWithWaves(std::string data_file, std::string subgroup,
                         int observation_step, bool extrapolate_into_excisions,
                         double angular_velocity, double expansion);

  explicit NumericBinaryWithWaves(CkMigrateMessage* m)
      : elliptic::analytic_data::Background(m),
        elliptic::analytic_data::InitialGuess(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(NumericBinaryWithWaves);

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<RequestedTags...> /*meta*/) const {
    return variables_impl<DataType>(x, std::nullopt, std::nullopt,
                                    tmpl::list<RequestedTags...>{});
  }

  template <typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x, const Mesh<3>& mesh,
      const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                            Frame::Inertial>& inv_jacobian,
      tmpl::list<RequestedTags...> /*meta*/) const {
    return variables_impl<DataVector>(x, mesh, inv_jacobian,
                                      tmpl::list<RequestedTags...>{});
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

  const std::string& data_file() const { return data_file_; }
  const std::string& subgroup() const { return subgroup_; }
  int observation_step() const { return observation_step_; }
  bool extrapolate_into_excisions() const {
    return extrapolate_into_excisions_;
  }
  double angular_velocity() const { return angular_velocity_; }
  double expansion() const { return expansion_; }

 private:
  /// Read the volume data into memory. Expensive, and not thread safe unless
  /// HDF5 was built with thread-safety support, so it must happen exactly once
  /// per node and on a single thread. See `load_interpolator` in the .cpp.
  void load_interpolator();

  std::string data_file_{};
  std::string subgroup_{};
  int observation_step_{-1};
  bool extrapolate_into_excisions_{false};
  double angular_velocity_ = std::numeric_limits<double>::signaling_NaN();
  double expansion_ = std::numeric_limits<double>::signaling_NaN();
  // Holds the loaded volume data in memory. Deliberately not serialized: it is
  // reloaded on unpacking, so that each node loads it once.
  spectre::Exporter::PointwiseInterpolator<3, Frame::Inertial> interpolator_{};

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables_impl(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      std::optional<std::reference_wrapper<const Mesh<3>>> mesh,
      std::optional<std::reference_wrapper<const InverseJacobian<
          DataType, 3, Frame::ElementLogical, Frame::Inertial>>>
          inv_jacobian,
      tmpl::list<RequestedTags...> /*meta*/) const {
    static_assert(
        std::is_same_v<DataType, DataVector>,
        "NumericBinaryWithWaves interpolates volume data onto a grid, so it "
        "only works with DataVectors.");
    using VarsComputer = detail::NumericBinaryWithWavesVariables<DataType>;
    // Interpolate all loaded fields from the already-resident volume data. This
    // is thread safe, unlike constructing the interpolator. `num_threads` is
    // pinned to 1 because this is already called from one Charm++ thread per
    // element; letting it spawn its own OpenMP team would oversubscribe.
    std::vector<DataVector> interpolated_data{};
    interpolator_.interpolate_to_points(
        make_not_null(&interpolated_data), x, extrapolate_into_excisions_,
        /*error_on_missing_points=*/true, /*num_threads=*/1);
    auto loaded_vars = spectre::Exporter::make_tagged_tuple<
        detail::numeric_load_tags<DataType>>(std::move(interpolated_data));
    typename VarsComputer::Cache cache{get_size(*x.begin())};
    const VarsComputer computer{
        std::move(mesh), std::move(inv_jacobian), x, angular_velocity_,
        expansion_,      std::move(loaded_vars)};
    return {cache.get_var(computer, RequestedTags{})...};
  }
};

bool operator==(const NumericBinaryWithWaves& lhs,
                const NumericBinaryWithWaves& rhs);
bool operator!=(const NumericBinaryWithWaves& lhs,
                const NumericBinaryWithWaves& rhs);

}  // namespace Xcts::AnalyticData
