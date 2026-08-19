// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/H5/VolumeData.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/NumericBinaryWithWaves.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"

namespace Xcts::AnalyticData {

namespace {

const std::string h5_file_name{
    "Unit.PointwiseFunctions.AnalyticData.Xcts.NumericBinaryWithWaves.h5"};

// A constant, anisotropic spatial metric. Constant data is represented exactly
// on the grid, so the interpolation is exact and every expected value below is
// analytic. The metric is deliberately not proportional to the identity, so the
// unimodular split and the Lie drag are both exercised non-trivially.
constexpr double gxx = 1.0;
constexpr double gyy = 4.0;
constexpr double gzz = 9.0;
constexpr double det_g = gxx * gyy * gzz;  // 36
constexpr double lapse_value = 2.0;
// A constant, non-trivial extrinsic curvature, so that the trace
// K = gamma^ij K_ij is a real test rather than 0 == 0.
constexpr double kxx = 0.1;
constexpr double kyy = 0.2;
constexpr double kzz = 0.3;
constexpr double angular_velocity = 0.3;
constexpr double expansion = 0.0;

// Unimodular split: psi = (det g)^(1/12), conformal metric = (det g)^(-1/3) g
double conformal_factor() { return std::pow(det_g, 1.0 / 12.0); }
double conformal_scaling() { return std::pow(det_g, -1.0 / 3.0); }

void write_volume_file() {
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
  domain::creators::register_derived_with_charm();
  domain::creators::time_dependence::register_derived_with_charm();
  domain::FunctionsOfTime::register_derived_with_charm();
  const domain::creators::Brick domain_creator{
      {{-1.0, -1.0, -1.0}}, {{1.0, 1.0, 1.0}}, {{0, 0, 0}}, {{3, 3, 3}}};
  const auto domain = domain_creator.create_domain();
  const Mesh<3> mesh{3, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const size_t num_points = mesh.number_of_grid_points();
  const DataVector zeros{num_points, 0.0};
  h5::H5File<h5::AccessType::ReadWrite> h5_file{h5_file_name};
  auto& volume_data = h5_file.insert<h5::VolumeData>("/element_data", 0);
  volume_data.write_volume_data(
      0, 0.0,
      {ElementVolumeData{
          ElementId<3>{0},
          {TensorComponent{"SpatialMetric_xx", DataVector{num_points, gxx}},
           TensorComponent{"SpatialMetric_yx", zeros},
           TensorComponent{"SpatialMetric_zx", zeros},
           TensorComponent{"SpatialMetric_yy", DataVector{num_points, gyy}},
           TensorComponent{"SpatialMetric_zy", zeros},
           TensorComponent{"SpatialMetric_zz", DataVector{num_points, gzz}},
           TensorComponent{"Lapse", DataVector{num_points, lapse_value}},
           TensorComponent{"ShiftExcess_x", zeros},
           TensorComponent{"ShiftExcess_y", zeros},
           TensorComponent{"ShiftExcess_z", zeros},
           TensorComponent{"ExtrinsicCurvature_xx",
                           DataVector{num_points, kxx}},
           TensorComponent{"ExtrinsicCurvature_yx", zeros},
           TensorComponent{"ExtrinsicCurvature_zx", zeros},
           TensorComponent{"ExtrinsicCurvature_yy",
                           DataVector{num_points, kyy}},
           TensorComponent{"ExtrinsicCurvature_zy", zeros},
           TensorComponent{"ExtrinsicCurvature_zz",
                           DataVector{num_points, kzz}}},
          mesh}},
      serialize(domain));
}

// The unimodular reconstruction of psi and the conformal metric.
void test_conformal_split(const NumericBinaryWithWaves& background,
                          const tnsr::I<DataVector, 3, Frame::Inertial>& x) {
  INFO("Unimodular conformal split");
  const auto vars = background.variables(
      x, tmpl::list<Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
                    Tags::ConformalFactorMinusOne<DataVector>,
                    Tags::LapseTimesConformalFactorMinusOne<DataVector>,
                    Tags::ShiftExcess<DataVector, 3, Frame::Inertial>>{});
  const auto& conformal_metric =
      get<Tags::ConformalMetric<DataVector, 3, Frame::Inertial>>(vars);
  const size_t num_points = get<0>(x).size();
  const double scaling = conformal_scaling();
  CHECK_ITERABLE_APPROX((get<0, 0>(conformal_metric)),
                        (DataVector{num_points, scaling * gxx}));
  CHECK_ITERABLE_APPROX((get<1, 1>(conformal_metric)),
                        (DataVector{num_points, scaling * gyy}));
  CHECK_ITERABLE_APPROX((get<2, 2>(conformal_metric)),
                        (DataVector{num_points, scaling * gzz}));
  CHECK_ITERABLE_APPROX((get<1, 0>(conformal_metric)),
                        (DataVector{num_points, 0.0}));

  // The defining property of the split: det(conformal metric) == 1
  const DataVector det_conformal_metric = get<0, 0>(conformal_metric) *
                                          get<1, 1>(conformal_metric) *
                                          get<2, 2>(conformal_metric);
  CHECK_ITERABLE_APPROX(det_conformal_metric, (DataVector{num_points, 1.0}));

  const double psi = conformal_factor();
  CHECK_ITERABLE_APPROX(
      get(get<Tags::ConformalFactorMinusOne<DataVector>>(vars)),
      (DataVector{num_points, psi - 1.0}));
  CHECK_ITERABLE_APPROX(
      get(get<Tags::LapseTimesConformalFactorMinusOne<DataVector>>(vars)),
      (DataVector{num_points, lapse_value * psi - 1.0}));
  // The background shift vanishes in the inertial frame, so the excess shift is
  // the loaded `ShiftExcess` -- not the loaded `Shift`, which is the full
  // corotating shift and grows like r.
  const auto& shift_excess =
      get<Tags::ShiftExcess<DataVector, 3, Frame::Inertial>>(vars);
  CHECK_ITERABLE_APPROX((get<0>(shift_excess)), (DataVector{num_points, 0.0}));
}

// The time derivative from the helical Killing vector. For a constant metric
// the transport term drops out and only the gradient of the Killing vector
// survives, so for gamma = diag(gxx, gyy, gzz) and pure rotation
//   dt(gamma)_xy = -Omega (gyy - gxx),
// with every other component vanishing.
void test_killing_vector_time_derivative(
    const NumericBinaryWithWaves& background, const Mesh<3>& mesh,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>& inv_jacobian,
    const tnsr::I<DataVector, 3, Frame::Inertial>& x) {
  INFO("Time derivative from the helical Killing vector");
  const auto vars = background.variables(
      x, mesh, inv_jacobian,
      tmpl::list<gr::Tags::TraceExtrinsicCurvature<DataVector>,
                 ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataVector>>,
                 Tags::ShiftBackground<DataVector, 3, Frame::Inertial>,
                 Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
                     DataVector, 3, Frame::Inertial>>{});
  const size_t num_points = get<0>(x).size();
  const double scaling = conformal_scaling();

  // The background shift vanishes in the inertial frame.
  const auto& shift_background =
      get<Tags::ShiftBackground<DataVector, 3, Frame::Inertial>>(vars);
  for (size_t i = 0; i < 3; ++i) {
    CHECK_ITERABLE_APPROX(shift_background.get(i),
                          (DataVector{num_points, 0.0}));
  }

  // K is taken straight from the loaded extrinsic curvature as gamma^ij K_ij.
  // The metric is diagonal, so this is just the sum of K_ii / gamma_ii.
  const double expected_trace_k = kxx / gxx + kyy / gyy + kzz / gzz;
  CHECK_ITERABLE_APPROX(
      get(get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(vars)),
      (DataVector{num_points, expected_trace_k}));
  // K is constant here, so its Lie drag along the Killing vector vanishes.
  CHECK_ITERABLE_APPROX(
      get(get<::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataVector>>>(vars)),
      (DataVector{num_points, 0.0}));

  // ubar^xy = gbar^xx gbar^yy dt(gbar)_xy with
  // dt(gbar)_xy = -Omega (gbar_yy - gbar_xx) = -Omega scaling (gyy - gxx),
  // and the reported quantity is -ubar^ij.
  const double dt_conformal_metric_xy =
      -angular_velocity * scaling * (gyy - gxx);
  const double expected_longitudinal_xy =
      -dt_conformal_metric_xy / (scaling * gxx * scaling * gyy);
  const auto& longitudinal_shift =
      get<Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
          DataVector, 3, Frame::Inertial>>(vars);
  CHECK_ITERABLE_APPROX((get<0, 1>(longitudinal_shift)),
                        (DataVector{num_points, expected_longitudinal_xy}));
  CHECK_ITERABLE_APPROX((get<0, 0>(longitudinal_shift)),
                        (DataVector{num_points, 0.0}));
  CHECK_ITERABLE_APPROX((get<2, 2>(longitudinal_shift)),
                        (DataVector{num_points, 0.0}));
}

// ---------------------------------------------------------------------------
// Post-Newtonian past evolution
// ---------------------------------------------------------------------------

// Equal masses, so eta = 1/4. The reference values below are for this case.
constexpr double pn_eta = 0.25;

// Reduced energy and angular momentum of a state, used as conserved quantities.
double reduced_energy(const std::array<double, 6>& state, const double eta) {
  const std::array<double, 3> separation{{state[0], state[1], state[2]}};
  const std::array<double, 3> momentum{{state[3], state[4], state[5]}};
  const double q =
      std::sqrt(separation[0] * separation[0] + separation[1] * separation[1] +
                separation[2] * separation[2]);
  const double s = momentum[0] * momentum[0] + momentum[1] * momentum[1] +
                   momentum[2] * momentum[2];
  const double n_dot_p =
      (separation[0] * momentum[0] + separation[1] * momentum[1] +
       separation[2] * momentum[2]) /
      q;
  return detail::reduced_hamiltonian(q, s, n_dot_p * n_dot_p, eta);
}

double reduced_angular_momentum(const std::array<double, 6>& state) {
  const double lx = state[1] * state[5] - state[2] * state[4];
  const double ly = state[2] * state[3] - state[0] * state[5];
  const double lz = state[0] * state[4] - state[1] * state[3];
  return std::sqrt(lx * lx + ly * ly + lz * lz);
}

double state_radius(const std::array<double, 6>& state) {
  return std::sqrt(state[0] * state[0] + state[1] * state[1] +
                   state[2] * state[2]);
}

// Complex step must agree with a central difference to the truncation error of
// the latter -- the complex-step value being the accurate one.
void test_complex_step_derivatives() {
  INFO("Complex-step derivatives of the reduced Hamiltonian");
  const double q = 11.3;
  const double s = 0.031;
  const double z = 0.0007;
  const auto derivs = detail::hamiltonian_derivatives(q, s, z, pn_eta);

  const double dq = 1.0e-6;
  const double fd_dq = (detail::reduced_hamiltonian(q + dq, s, z, pn_eta) -
                        detail::reduced_hamiltonian(q - dq, s, z, pn_eta)) /
                       (2.0 * dq);
  const double ds = 1.0e-9;
  const double fd_ds = (detail::reduced_hamiltonian(q, s + ds, z, pn_eta) -
                        detail::reduced_hamiltonian(q, s - ds, z, pn_eta)) /
                       (2.0 * ds);
  const double dz = 1.0e-11;
  const double fd_dz = (detail::reduced_hamiltonian(q, s, z + dz, pn_eta) -
                        detail::reduced_hamiltonian(q, s, z - dz, pn_eta)) /
                       (2.0 * dz);
  CHECK(derivs[0] == approx(fd_dq).epsilon(1.0e-8));
  CHECK(derivs[1] == approx(fd_ds).epsilon(1.0e-6));
  CHECK(derivs[2] == approx(fd_dz).epsilon(1.0e-4));

  // At large separation and small momentum the Hamiltonian reduces to the
  // Newtonian one, whose derivatives are exact and analytic.
  const double q_far = 1.0e6;
  const double s_far = 1.0e-6;
  const auto far = detail::hamiltonian_derivatives(q_far, s_far, 0.0, pn_eta);
  CHECK(far[0] == approx(1.0 / (q_far * q_far)).epsilon(1.0e-5));
  CHECK(far[1] == approx(0.5).epsilon(1.0e-5));
}

// The quasi-circular sequence, checked against a value SpECTRE itself uses:
// tests/InputFiles/Xcts/BinaryBlackHole.yaml pairs D = 16 with Omega = 0.0144.
void test_circular_orbit() {
  INFO("Quasi-circular sequence");
  const auto at_16 = detail::circular_orbit(16.0, pn_eta);
  CHECK(at_16[1] == approx(0.0144).epsilon(5.0e-3));
  // Newtonian limit: s -> 1/q and omega -> q^(-3/2).
  const double q_far = 1.0e5;
  const auto far = detail::circular_orbit(q_far, pn_eta);
  CHECK(far[0] == approx(1.0 / q_far).epsilon(1.0e-4));
  CHECK(far[1] == approx(std::pow(q_far, -1.5)).epsilon(1.0e-4));
}

// Any error in the three scalar derivatives or in the chain rule that builds
// dH/dq and dH/dp from them breaks the symplectic structure and makes the
// energy drift *linearly*. Conservation to round-off is therefore the sharp
// test of the whole differentiation machinery at once.
void test_conservative_flow() {
  INFO("Energy and angular momentum are conserved without radiation reaction");
  const auto circular = detail::circular_orbit(12.0, pn_eta);
  const double omega = circular[1];
  const double period = 2.0 * M_PI / omega;
  const auto trajectory = detail::evolve_binary_backwards(
      12.0, omega, 0.0, pn_eta, 1.5 * period, 4.0,
      /*with_radiation_reaction=*/false);

  const double energy_0 = reduced_energy(trajectory.state[0], pn_eta);
  const double angular_momentum_0 =
      reduced_angular_momentum(trajectory.state[0]);
  const double radius_0 = state_radius(trajectory.state[0]);
  double max_energy_drift = 0.0;
  double max_angular_momentum_drift = 0.0;
  double max_radius_drift = 0.0;
  for (const auto& state : trajectory.state) {
    max_energy_drift = std::max(
        max_energy_drift, std::abs(reduced_energy(state, pn_eta) - energy_0) /
                              std::abs(energy_0));
    max_angular_momentum_drift = std::max(
        max_angular_momentum_drift,
        std::abs(reduced_angular_momentum(state) - angular_momentum_0) /
            angular_momentum_0);
    // The orbit started exactly circular, so it must stay at fixed radius.
    max_radius_drift = std::max(
        max_radius_drift, std::abs(state_radius(state) - radius_0) / radius_0);
    // Planar motion: the orbit never leaves the z = 0 plane.
    CHECK(std::abs(state[2]) < 1.0e-12);
    CHECK(std::abs(state[5]) < 1.0e-12);
  }
  CHECK(max_energy_drift < 1.0e-10);
  CHECK(max_angular_momentum_drift < 1.0e-10);
  CHECK(max_radius_drift < 1.0e-10);
}

// The BCD form of the radiation-reaction force is normalised so that, for a
// circular orbit, dHhat/dt is exactly the flux divided by eta.
void test_flux_energy_balance() {
  INFO("Radiation reaction removes energy at the rate given by the flux");
  const auto circular = detail::circular_orbit(12.0, pn_eta);
  const double s = circular[0];
  const double omega = circular[1];
  const std::array<double, 6> state{{12.0, 0.0, 0.0, 0.0, std::sqrt(s), 0.0}};
  const auto rhs = detail::inspiral_rhs(state, pn_eta, true);

  // dHhat/dt = dH/dq . qdot + dH/dp . pdot, evaluated from the same pieces the
  // right-hand side is built from.
  const std::array<double, 3> separation{{state[0], state[1], state[2]}};
  const std::array<double, 3> momentum{{state[3], state[4], state[5]}};
  const auto dh_dq =
      detail::hamiltonian_deriv_separation(separation, momentum, pn_eta);
  const auto dh_dp =
      detail::hamiltonian_deriv_momentum(separation, momentum, pn_eta);
  double dh_dt = 0.0;
  for (size_t i = 0; i < 3; ++i) {
    dh_dt += gsl::at(dh_dq, i) * gsl::at(rhs, i) +
             gsl::at(dh_dp, i) * gsl::at(rhs, i + 3);
  }
  const double expected =
      detail::energy_flux(std::cbrt(omega), pn_eta) / pn_eta;
  CHECK(dh_dt == approx(expected).epsilon(1.0e-10));
  // The flux is an energy loss.
  CHECK(dh_dt < 0.0);

  // The frequency of the state matches the one the sequence reported.
  CHECK(detail::orbital_frequency(separation, momentum, pn_eta) ==
        approx(omega));
}

// The trajectory reproduces the orbital state it was given, and the Hermite
// interpolation is exact at the samples and converges between them.
void test_trajectory_sampling() {
  INFO("Trajectory initial state and interpolation");
  // A physical inspiral: the quasi-circular frequency at this separation, with
  // the radial expansion the post-Newtonian sequence predicts there. A
  // non-zero adot is used on purpose, so the momentum inversion is exercised
  // with a genuinely two-component velocity.
  const double separation = 12.0;
  const double omega = detail::circular_orbit(separation, pn_eta)[1];
  const double adot = -1.324e-4;
  const double time_step = 2.0;
  const auto trajectory = detail::evolve_binary_backwards(
      separation, omega, adot, pn_eta, 100.0, time_step);

  // The present state is the one that was asked for: q = (D, 0, 0) and
  // qdot = (adot D, Omega D, 0).
  const auto& present = trajectory.state[0];
  CHECK(present[0] == approx(separation));
  CHECK(present[1] == approx(0.0));
  const auto& present_velocity = trajectory.dt_state[0];
  CHECK(present_velocity[0] == approx(adot * separation));
  CHECK(present_velocity[1] == approx(omega * separation));

  CHECK(trajectory.earliest_time() == approx(-100.0));

  // Interpolation is exact at the sample points.
  for (size_t sample = 0; sample < trajectory.state.size(); ++sample) {
    const double time = -time_step * static_cast<double>(sample);
    const auto interpolated = trajectory.state_at(time);
    for (size_t i = 0; i < 6; ++i) {
      CHECK(gsl::at(interpolated, i) ==
            approx(gsl::at(trajectory.state[sample], i)));
    }
  }

  // Cubic Hermite is fourth-order accurate, so halving the sample spacing must
  // cut the interpolation error by about sixteen. Checking the *order* rather
  // than an absolute tolerance keeps this independent of how fast the chosen
  // orbit happens to vary.
  //
  // A quasi-circular orbit is used here on purpose. For an eccentric one the
  // error is set by the periapsis timescale, not by the orbital period: the
  // configuration used above sweeps from r = 12 down to r = 3 in 100 M, where
  // the angular velocity is some twenty times larger, so a step that is
  // generous at apoapsis is marginal at periapsis. `PastEvolutionTimeStep` has
  // to be chosen against the fastest part of the orbit.
  const double circular_omega = detail::circular_orbit(12.0, pn_eta)[1];
  const double reference_step = 0.25;
  const auto reference = detail::evolve_binary_backwards(
      12.0, circular_omega, 0.0, pn_eta, 100.0, reference_step);
  const auto interpolation_error = [&reference, &reference_step,
                                    &circular_omega](const double step) {
    const auto coarse = detail::evolve_binary_backwards(
        12.0, circular_omega, 0.0, pn_eta, 100.0, step);
    double worst = 0.0;
    for (size_t sample = 0; sample < reference.state.size(); ++sample) {
      const double time = -reference_step * static_cast<double>(sample);
      const auto interpolated = coarse.state_at(time);
      for (size_t i = 0; i < 6; ++i) {
        worst = std::max(worst, std::abs(gsl::at(interpolated, i) -
                                         gsl::at(reference.state[sample], i)));
      }
    }
    return worst;
  };
  const double error_coarse = interpolation_error(4.0);
  const double error_fine = interpolation_error(2.0);
  CHECK(error_coarse > 0.0);
  CHECK(error_fine > 0.0);
  // A generous window around the ideal factor of sixteen.
  CHECK(error_coarse / error_fine > 8.0);
}

// Going into the past, radiation reaction *adds* energy to the binary, so a
// quasi-circular orbit must widen monotonically. This is the check that the
// backwards integration runs in the direction it claims to: the trajectory is
// integrated in tau = -t with dY/dtau = -RHS(Y), and a sign slip there would
// show up here as a binary that tightens into the past.
//
// The contrast with `test_eccentric_configuration` is the point. There the
// separation *falls* into the past, but that is the radial oscillation of an
// eccentric orbit, whose period (~134 M) is far shorter than the timescale on
// which radiation reaction moves the semi-major axis.
void test_past_widens_for_circular_orbit() {
  INFO("A quasi-circular binary was wider in the past");
  const double circular_omega = detail::circular_orbit(12.0, pn_eta)[1];
  const auto trajectory = detail::evolve_binary_backwards(
      12.0, circular_omega, 0.0, pn_eta, 400.0, 4.0);
  const double radius_now = state_radius(trajectory.state.front());
  const double radius_then = state_radius(trajectory.state.back());
  CHECK(radius_now == approx(12.0).epsilon(1.0e-10));
  CHECK(radius_then > radius_now);
  // The quasi-circular inspiral rate at D = 12 is rdot = -1.6e-3, so over
  // 400 M the separation should grow by of order 0.6 M. Bounded loosely,
  // because the rate itself falls as the orbit widens.
  CHECK(radius_then - radius_now > 0.1);
  CHECK(radius_then - radius_now < 3.0);

  // Energy and angular momentum are the clean monotonic diagnostics. The flux
  // is negative at every point of a bound prograde orbit, so going into the
  // past both increase at *every* step. The separation is not monotonic: see
  // below.
  for (size_t sample = 1; sample < trajectory.state.size(); ++sample) {
    CHECK(reduced_energy(trajectory.state[sample], pn_eta) >
          reduced_energy(trajectory.state[sample - 1], pn_eta));
    CHECK(reduced_angular_momentum(trajectory.state[sample]) >
          reduced_angular_momentum(trajectory.state[sample - 1]));
  }

  // `circular_orbit` returns the circular orbit of the *conservative*
  // Hamiltonian, which has n.p = 0. The adiabatic inspiral of the dissipative
  // system wants a small non-zero radial momentum instead, so starting from
  // n.p = 0 injects a residual eccentricity of order
  // rdot / (omega r) = 1.6e-3 / (0.0215 * 12) ~ 6e-3. That shows up as a slow
  // radial wobble on top of the secular widening, with an amplitude of a few
  // hundredths of M -- small, but enough that the separation itself is not
  // monotonic over a radial period.
  //
  // This is why the `Expansion` option exists: a genuinely low-eccentricity
  // start needs rdot from the inspiral rate, not rdot = 0.
  double max_radius = radius_now;
  double wobble = 0.0;
  for (const auto& state : trajectory.state) {
    max_radius = std::max(max_radius, state_radius(state));
    wobble = std::max(wobble, max_radius - state_radius(state));
  }
  CHECK(wobble < 0.15);
}

// Solve 1's own parameters. See report 010 section 6: D = 12 with
// Omega = 0.016 is not a quasi-circular configuration but an orbit of
// eccentricity 0.6, and the code must reproduce that rather than silently
// circularise it.
void test_eccentric_configuration() {
  INFO("An eccentric orbital state is reproduced, not silently circularised");
  // Over a short window the separation already falls measurably. The window is
  // kept short on purpose: continue much further and this orbit reaches the
  // strong field, which `test_strong_field_is_refused` covers.
  const auto eccentric =
      detail::evolve_binary_backwards(12.0, 0.016, 0.001, pn_eta, 20.0, 2.0,
                                      /*with_radiation_reaction=*/false);
  double min_radius = state_radius(eccentric.state[0]);
  for (const auto& state : eccentric.state) {
    min_radius = std::min(min_radius, state_radius(state));
  }
  CHECK(min_radius < 11.5);

  // The contrast: the quasi-circular orbit at the same separation holds 12 to
  // round-off over the same window.
  const double circular_omega = detail::circular_orbit(12.0, pn_eta)[1];
  const auto circular = detail::evolve_binary_backwards(
      12.0, circular_omega, 0.0, pn_eta, 20.0, 2.0,
      /*with_radiation_reaction=*/false);
  for (const auto& state : circular.state) {
    CHECK(state_radius(state) == approx(12.0).epsilon(1.0e-10));
  }
}

// The orbital state the past evolution actually starts from: derived from the
// separation and the target eccentricity, not copied from the loaded data's
// AngularVelocity/Expansion.
void test_quasi_circular_state() {
  INFO("Quasi-circular orbital state derived from the separation");
  const auto state = detail::quasi_circular_state(12.0, pn_eta);
  // Omega agrees with the circular sequence, and adot is the inspiral rate:
  // negative, and of the size the quadrupole formula gives.
  CHECK(state[0] == approx(detail::circular_orbit(12.0, pn_eta)[1]));
  CHECK(state[1] < 0.0);
  CHECK(state[1] == approx(-1.324e-4).epsilon(0.05));
  // Newtonian scaling adot ~ -(64/5) eta / q^4: widening the orbit slows the
  // drift steeply.
  const auto wider = detail::quasi_circular_state(16.0, pn_eta);
  CHECK(wider[1] < 0.0);
  CHECK(wider[1] > state[1]);
  const double ratio = state[1] / wider[1];
  CHECK(ratio > 2.0);
  CHECK(ratio < 6.0);

  // The point of a non-zero adot: it is what makes the start non-eccentric.
  // Compare the residual radial wobble against a start with adot = 0.
  const double omega = state[0];
  const auto with_drift = detail::evolve_binary_backwards(12.0, omega, state[1],
                                                          pn_eta, 400.0, 4.0);
  const auto without_drift =
      detail::evolve_binary_backwards(12.0, omega, 0.0, pn_eta, 400.0, 4.0);
  const auto wobble = [](const auto& trajectory) {
    double peak = 0.0;
    double worst = 0.0;
    for (const auto& state_at_sample : trajectory.state) {
      peak = std::max(peak, state_radius(state_at_sample));
      worst = std::max(worst, peak - state_radius(state_at_sample));
    }
    return worst;
  };
  CHECK(wobble(with_drift) < wobble(without_drift));
}

// The whole point of deriving the orbit: unlike the loaded data's own
// parameters, it survives the full history the wave content needs.
void test_derived_orbit_covers_the_domain() {
  INFO("The derived orbit stays post-Newtonian over the full past evolution");
  const auto state = detail::quasi_circular_state(12.0, pn_eta);
  // 738 M is what `make_addwave.py` requests for an outer boundary at 480.
  const auto trajectory = detail::evolve_binary_backwards(
      12.0, state[0], state[1], pn_eta, 738.0, 2.0);
  double min_radius = state_radius(trajectory.state.front());
  for (const auto& sample : trajectory.state) {
    min_radius = std::min(min_radius, state_radius(sample));
  }
  // Never approaches the strong field, and was wider in the past.
  CHECK(min_radius > 11.9);
  CHECK(state_radius(trajectory.state.back()) >
        state_radius(trajectory.state.front()));
  CHECK(trajectory.earliest_time() == approx(-738.0));
}

// Continuing that trajectory reaches periapsis at r ~ 3, where v/c ~ 0.55 and
// the 3.5PN flux series is far outside its domain of convergence. Integrating
// on regardless inflates the orbit to hundreds of M, so the trajectory would be
// meaningless as wave-generating history. It must fail loudly instead.
void test_strong_field_is_refused() {
  INFO("Leaving the post-Newtonian regime is an error, not a silent result");
  CHECK_THROWS_WITH(
      detail::evolve_binary_backwards(12.0, 0.016, 0.001, pn_eta, 400.0, 4.0),
      Catch::Matchers::ContainsSubstring("post-Newtonian expansion is not"));
  // The quasi-circular configuration at the same separation is fine over a
  // span long enough to cover the outer boundary of the evolution domain.
  const double circular_omega = detail::circular_orbit(12.0, pn_eta)[1];
  CHECK_NOTHROW(detail::evolve_binary_backwards(12.0, circular_omega, 0.0,
                                                pn_eta, 738.0, 2.0));
}

// ---------------------------------------------------------------------------
// Post-Newtonian wave content
// ---------------------------------------------------------------------------

// Independent transcription of the *uncombined* pieces, straight from report
// 003, used to check the closed form of `htt_instantaneous`. Written in terms
// of the raw quantities rather than the kernel, so that it exercises the
// algebra of report 003 step 4 rather than repeating it.
namespace reference {

struct Geometry {
  DataVector radius;
  std::array<DataVector, 3> normal;
};

Geometry geometry_of(const tnsr::I<DataVector, 3, Frame::Inertial>& x,
                     const std::array<double, 3>& position) {
  std::array<DataVector, 3> offset{
      {x.get(0) - position[0], x.get(1) - position[1], x.get(2) - position[2]}};
  DataVector radius = sqrt(offset[0] * offset[0] + offset[1] * offset[1] +
                           offset[2] * offset[2]);
  return {radius,
          {{offset[0] / radius, offset[1] / radius, offset[2] / radius}}};
}

// The second (mass-mass) sum of JS_htt4: (m1 m2 / 8) sum_A T_A^ij.
tnsr::ii<DataVector, 3> near_zone_mass_term(
    const tnsr::I<DataVector, 3, Frame::Inertial>& x,
    const std::array<double, 3>& position_1,
    const std::array<double, 3>& position_2, const double mass_1,
    const double mass_2) {
  const std::array<Geometry, 2> bodies{
      {geometry_of(x, position_1), geometry_of(x, position_2)}};
  const std::array<double, 3> separation{{position_2[0] - position_1[0],
                                          position_2[1] - position_1[1],
                                          position_2[2] - position_1[2]}};
  const double r12 =
      std::sqrt(separation[0] * separation[0] + separation[1] * separation[1] +
                separation[2] * separation[2]);
  const std::array<double, 3> n12{
      {separation[0] / r12, separation[1] / r12, separation[2] / r12}};

  auto result = make_with_value<tnsr::ii<DataVector, 3>>(x.get(0), 0.0);
  for (size_t body = 0; body < 2; ++body) {
    const auto& near = gsl::at(bodies, body);
    const auto& far = gsl::at(bodies, 1 - body);
    const double sigma = (body == 0) ? 1.0 : -1.0;
    const DataVector& ra = near.radius;
    const DataVector& rb = far.radius;
    const DataVector s = ra + rb + r12;

    const DataVector coefficient_n12 = -32.0 / s * (1.0 / r12 + 1.0 / s);
    const DataVector coefficient_cross =
        2.0 * ((ra + rb) / (r12 * r12 * r12) + 12.0 / (s * s));
    const DataVector coefficient_mixed =
        32.0 * sigma * (2.0 / (s * s) - 1.0 / (r12 * r12));
    const DataVector coefficient_nana =
        5.0 / (r12 * ra) - (rb * rb / ra + 3.0 * ra) / (r12 * r12 * r12) -
        8.0 / s * (1.0 / ra + 1.0 / s);
    const DataVector coefficient_delta =
        5.0 * ra / (r12 * r12 * r12) * (ra / rb - 1.0) - 17.0 / (r12 * ra) +
        4.0 / (ra * rb) + 8.0 / s * (1.0 / ra + 4.0 / r12);

    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        DataVector value =
            coefficient_n12 * gsl::at(n12, i) * gsl::at(n12, j) +
            coefficient_nana * gsl::at(near.normal, i) *
                gsl::at(near.normal, j) +
            0.5 * coefficient_cross *
                (gsl::at(near.normal, i) * gsl::at(far.normal, j) +
                 gsl::at(near.normal, j) * gsl::at(far.normal, i)) +
            0.5 * coefficient_mixed *
                (gsl::at(near.normal, i) * gsl::at(n12, j) +
                 gsl::at(near.normal, j) * gsl::at(n12, i));
        if (i == j) {
          value += coefficient_delta;
        }
        result.get(i, j) += value;
      }
    }
  }
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      result.get(i, j) *= 0.125 * mass_1 * mass_2;
    }
  }
  return result;
}

// The momentum sum of JS_htt4, written with p_A rather than through the
// kernel, so the 1/m_A bookkeeping of report 003 step 1 is checked.
tnsr::ii<DataVector, 3> near_zone_momentum_term(
    const tnsr::I<DataVector, 3, Frame::Inertial>& x,
    const std::array<std::array<double, 3>, 2>& positions,
    const std::array<std::array<double, 3>, 2>& momenta,
    const std::array<double, 2>& masses) {
  auto result = make_with_value<tnsr::ii<DataVector, 3>>(x.get(0), 0.0);
  for (size_t body = 0; body < 2; ++body) {
    const auto here = geometry_of(x, gsl::at(positions, body));
    const auto& p = gsl::at(momenta, body);
    const double mass = gsl::at(masses, body);
    const double p_squared = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
    DataVector p_dot_n = make_with_value<DataVector>(here.radius, 0.0);
    for (size_t i = 0; i < 3; ++i) {
      p_dot_n += gsl::at(p, i) * gsl::at(here.normal, i);
    }
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        DataVector value = 2.0 * gsl::at(p, i) * gsl::at(p, j) +
                           (3.0 * p_dot_n * p_dot_n - 5.0 * p_squared) *
                               gsl::at(here.normal, i) *
                               gsl::at(here.normal, j) +
                           6.0 * p_dot_n *
                               (gsl::at(p, i) * gsl::at(here.normal, j) +
                                gsl::at(p, j) * gsl::at(here.normal, i));
        if (i == j) {
          value += p_squared - 5.0 * p_dot_n * p_dot_n;
        }
        result.get(i, j) += 0.25 * value / (mass * here.radius);
      }
    }
  }
  return result;
}

}  // namespace reference

// Field points away from the punctures and off the symmetry axes.
tnsr::I<DataVector, 3, Frame::Inertial> wave_test_points() {
  return tnsr::I<DataVector, 3, Frame::Inertial>{
      {{DataVector{3.1, -8.4, 20.0, 0.7, 55.0},
        DataVector{4.7, 2.2, -13.0, -9.1, 31.0},
        DataVector{-2.3, 6.8, 7.5, 4.4, -18.0}}}};
}

// Report 003 step 1: the momentum sum of the near-zone term is exactly
// +1/4 sum_A Phi(p_A / sqrt(m_A); A). Checking this pins down the 1/m_A
// bookkeeping of the substitution.
void test_kernel_reproduces_momentum_sum() {
  INFO("The kernel reproduces the near-zone momentum sum");
  const auto x = wave_test_points();
  const std::array<std::array<double, 3>, 2> positions{
      {{{6.0, 0.0, 0.0}}, {{-6.0, 0.0, 0.0}}}};
  const std::array<std::array<double, 3>, 2> momenta{
      {{{0.02, 0.13, 0.005}}, {{-0.02, -0.13, -0.005}}}};
  const std::array<double, 2> masses{{0.5, 0.5}};

  auto from_kernel = make_with_value<tnsr::ii<DataVector, 3>>(x.get(0), 0.0);
  for (size_t body = 0; body < 2; ++body) {
    const double sqrt_mass = std::sqrt(gsl::at(masses, body));
    const std::array<double, 3> u{{gsl::at(momenta, body)[0] / sqrt_mass,
                                   gsl::at(momenta, body)[1] / sqrt_mass,
                                   gsl::at(momenta, body)[2] / sqrt_mass}};
    tnsr::ii<DataVector, 3> kernel{};
    detail::post_newtonian_kernel(make_not_null(&kernel), x,
                                  gsl::at(positions, body), u);
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        from_kernel.get(i, j) += 0.25 * kernel.get(i, j);
      }
    }
  }
  const auto reference_value =
      reference::near_zone_momentum_term(x, positions, momenta, masses);
  CHECK_ITERABLE_APPROX(from_kernel, reference_value);
}

// The heart of report 003: the closed form must equal the near-zone mass term
// plus the present-time w contribution computed separately,
//   h_inst = h^(4),mm - sum_A H[w; t] = h^(4),mm + 1/4 sum_A Phi(w; A).
// This is what validates the collection of terms in step 4 -- in particular
// the two cancellations, and the sigma_A sign on the only term that is odd
// under exchanging the bodies.
void test_instantaneous_matches_uncombined() {
  INFO("The closed form equals the uncombined near-zone plus present pieces");
  const auto x = wave_test_points();
  // Deliberately unequal masses and an off-axis separation, so that no
  // symmetry can hide a mistake in the sigma_A term or in the A <-> B
  // bookkeeping.
  const std::array<double, 3> position_1{{5.0, 1.0, -0.5}};
  const std::array<double, 3> position_2{{-7.0, -2.0, 0.5}};
  const double mass_1 = 0.6;
  const double mass_2 = 0.4;

  tnsr::ii<DataVector, 3> combined{};
  detail::htt_instantaneous(make_not_null(&combined), x, position_1, position_2,
                            mass_1, mass_2);

  auto uncombined =
      reference::near_zone_mass_term(x, position_1, position_2, mass_1, mass_2);
  // w = sqrt(m1 m2 / (2 r12)) n12, and -sum_A H[w] = +1/4 sum_A Phi(w; A).
  const std::array<double, 3> separation{{position_2[0] - position_1[0],
                                          position_2[1] - position_1[1],
                                          position_2[2] - position_1[2]}};
  const double r12 =
      std::sqrt(separation[0] * separation[0] + separation[1] * separation[1] +
                separation[2] * separation[2]);
  const double w_magnitude = std::sqrt(mass_1 * mass_2 / (2.0 * r12));
  const std::array<double, 3> w{{w_magnitude * separation[0] / r12,
                                 w_magnitude * separation[1] / r12,
                                 w_magnitude * separation[2] / r12}};
  for (const auto& position : {position_1, position_2}) {
    tnsr::ii<DataVector, 3> kernel{};
    detail::post_newtonian_kernel(make_not_null(&kernel), x, position, w);
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        uncombined.get(i, j) += 0.25 * kernel.get(i, j);
      }
    }
  }
  CHECK_ITERABLE_APPROX(combined, uncombined);
}

// Structural properties that hold whatever the algebra: symmetry, the overall
// mass scaling, and the fact that swapping the two bodies (labels *and*
// masses) leaves the result unchanged, since the sum runs over both.
void test_instantaneous_structure() {
  INFO("Symmetry and scaling of the instantaneous term");
  const auto x = wave_test_points();
  const std::array<double, 3> position_1{{5.0, 1.0, -0.5}};
  const std::array<double, 3> position_2{{-7.0, -2.0, 0.5}};

  tnsr::ii<DataVector, 3> forward{};
  detail::htt_instantaneous(make_not_null(&forward), x, position_1, position_2,
                            0.6, 0.4);
  tnsr::ii<DataVector, 3> swapped{};
  detail::htt_instantaneous(make_not_null(&swapped), x, position_2, position_1,
                            0.4, 0.6);
  CHECK_ITERABLE_APPROX(forward, swapped);

  // Linear in the product of the masses.
  tnsr::ii<DataVector, 3> doubled{};
  detail::htt_instantaneous(make_not_null(&doubled), x, position_1, position_2,
                            1.2, 0.4);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      CHECK_ITERABLE_APPROX(DataVector{2.0 * forward.get(i, j)},
                            doubled.get(i, j));
    }
  }
}

// Report 003 warned that the surviving n_A n_A coefficient contains
// -(1/r_12^3)(r_B^2/r_A + 3 r_A) -> -4r/r_12^3, growing linearly, and flagged
// that as a numerical worry for the outer domain. That warning was wrong: it
// looked at one coefficient in isolation. The n_A^i n_B^j term carries
// +2(r_A + r_B)/r_12^3 -> +4r/r_12^3 per body, and at large r the two normals
// coincide, so summed over both bodies the -8r/r_12^3 and +8r/r_12^3 cancel.
//
// This test measures the actual falloff. It is deliberately strict, because the
// cancellation is exactly the kind of thing a later refactor could break.
void test_instantaneous_falls_off() {
  INFO("The instantaneous term decays at least as fast as 1/r");
  const std::array<double, 3> position_1{{6.0, 0.0, 0.0}};
  const std::array<double, 3> position_2{{-6.0, 0.0, 0.0}};
  const DataVector radii{50.0, 100.0, 200.0, 400.0, 800.0};
  // A generic direction, so that no component is accidentally special and the
  // n_A n_A structure is not aligned with a single tensor slot.
  const double direction_x = 0.5;
  const double direction_y = 0.7;
  const double direction_z = 0.5099019513592785;
  const tnsr::I<DataVector, 3, Frame::Inertial> x{
      {{radii * direction_x, radii * direction_y, radii * direction_z}}};
  tnsr::ii<DataVector, 3> value{};
  detail::htt_instantaneous(make_not_null(&value), x, position_1, position_2,
                            0.5, 0.5);

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      const DataVector& component = value.get(i, j);
      for (size_t k = 1; k < radii.size(); ++k) {
        // Local power-law slope between successive radii.
        const double slope =
            std::log(std::abs(component[k] / component[k - 1])) /
            std::log(radii[k] / radii[k - 1]);
        CAPTURE(i);
        CAPTURE(j);
        CAPTURE(radii[k]);
        CAPTURE(slope);
        CHECK(slope < -0.9);
      }
    }
  }
}

// A trajectory for the wave tests: quasi-circular at D = 12, long enough to
// cover the retarded time of everything inside r ~ 300.
detail::PnInspiral wave_test_trajectory() {
  const double omega = detail::circular_orbit(12.0, pn_eta)[1];
  return detail::evolve_binary_backwards(12.0, omega, 0.0, pn_eta, 400.0, 2.0);
}

// The defining equation of the retarded time, checked as a residual rather than
// against a precomputed number: -t_r must equal the distance to where the body
// *was* at t_r, not where it is now.
void test_retarded_time() {
  INFO("Retarded time solves t - t_r - r_a(t_r) = 0");
  const auto trajectory = wave_test_trajectory();
  const double mass = 0.5;
  double max_reduced_separation = 0.0;
  for (const auto& sample : trajectory.state) {
    max_reduced_separation =
        std::max(max_reduced_separation,
                 std::sqrt(sample[0] * sample[0] + sample[1] * sample[1] +
                           sample[2] * sample[2]));
  }
  const double max_body_distance = mass * max_reduced_separation;

  for (const auto& field_point : {std::array<double, 3>{{30.0, 10.0, -5.0}},
                                  std::array<double, 3>{{-120.0, 40.0, 25.0}},
                                  std::array<double, 3>{{0.0, 0.0, 250.0}}}) {
    for (size_t body = 0; body < 2; ++body) {
      const double time = detail::retarded_time(
          field_point, trajectory, body, mass, mass, max_body_distance, 0.0);
      CAPTURE(body);
      CAPTURE(time);
      // Strictly in the past.
      CHECK(time < 0.0);
      const auto configuration =
          detail::configuration_at(trajectory, time, mass, mass);
      const auto& position = gsl::at(configuration.positions, body);
      const double distance =
          std::sqrt(std::pow(field_point[0] - position[0], 2) +
                    std::pow(field_point[1] - position[1], 2) +
                    std::pow(field_point[2] - position[2], 2));
      // The residual itself, to round-off.
      CHECK(-time == approx(distance).epsilon(1.0e-10));
      // Retardation is a real effect here: the light-crossing time differs
      // from the naive one using the *present* position, because the body
      // moved. Confirms we are not accidentally solving the static problem.
      const auto now = detail::configuration_at(trajectory, 0.0, mass, mass);
      const auto& position_now = gsl::at(now.positions, body);
      const double naive =
          std::sqrt(std::pow(field_point[0] - position_now[0], 2) +
                    std::pow(field_point[1] - position_now[1], 2) +
                    std::pow(field_point[2] - position_now[2], 2));
      CHECK(std::abs(distance - naive) > 1.0e-6);
    }
  }
}

// Unlike the instantaneous term, the retarded piece carries radiation and must
// fall off like 1/r.
//
// The amplitude *envelope* is what obeys 1/r, not any single component along a
// single ray: the wavelength here is pi/Omega ~ 146 M, so between the sample
// radii the wave goes through more than a full cycle and an individual
// component can sit near a zero crossing. Averaging |h|^2 over a sphere removes
// the phase and leaves the envelope.
double rms_over_sphere(const detail::PnInspiral& trajectory,
                       const double radius) {
  // A deterministic quasi-uniform covering of the sphere (Fibonacci spiral).
  const size_t num_points = 200;
  DataVector px{num_points};
  DataVector py{num_points};
  DataVector pz{num_points};
  const double golden_angle = M_PI * (3.0 - std::sqrt(5.0));
  for (size_t k = 0; k < num_points; ++k) {
    const double cos_theta =
        1.0 - 2.0 * (static_cast<double>(k) + 0.5) / num_points;
    const double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
    const double phi = golden_angle * static_cast<double>(k);
    px[k] = radius * sin_theta * std::cos(phi);
    py[k] = radius * sin_theta * std::sin(phi);
    pz[k] = radius * cos_theta;
  }
  const tnsr::I<DataVector, 3, Frame::Inertial> x{{{px, py, pz}}};
  tnsr::ii<DataVector, 3> value{};
  const auto times = detail::retarded_times(x, trajectory, 0.5, 0.5, 0.0);
  detail::htt_retarded(make_not_null(&value), x, trajectory, 0.5, 0.5, times);
  double sum_of_squares = 0.0;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      const double weight = (i == j) ? 1.0 : 2.0;
      for (size_t k = 0; k < num_points; ++k) {
        sum_of_squares += weight * square(value.get(i, j)[k]);
      }
    }
  }
  return std::sqrt(sum_of_squares / num_points);
}

void test_retarded_falls_off_like_one_over_r() {
  INFO("The retarded term decays like 1/r");
  const auto trajectory = wave_test_trajectory();
  const double amplitude_100 = rms_over_sphere(trajectory, 100.0);
  const double amplitude_200 = rms_over_sphere(trajectory, 200.0);
  const double amplitude_300 = rms_over_sphere(trajectory, 300.0);
  CAPTURE(amplitude_100);
  CAPTURE(amplitude_200);
  CAPTURE(amplitude_300);

  // Something non-trivial came out.
  CHECK(amplitude_100 > 1.0e-6);

  // r * amplitude is constant to within a modest band. The residual drift is
  // physical: the amplitude also depends on the orbital frequency at the
  // retarded time, which was lower further in the past.
  const double scaled_100 = 100.0 * amplitude_100;
  const double scaled_200 = 200.0 * amplitude_200;
  const double scaled_300 = 300.0 * amplitude_300;
  CAPTURE(scaled_100);
  CAPTURE(scaled_200);
  CAPTURE(scaled_300);
  CHECK(scaled_200 / scaled_100 > 0.5);
  CHECK(scaled_200 / scaled_100 < 2.0);
  CHECK(scaled_300 / scaled_100 > 0.5);
  CHECK(scaled_300 / scaled_100 < 2.0);

  // And it is definitely not the steeper-than-1/r falloff of the
  // instantaneous term: over a factor of three in radius, a 1/r^3 decay would
  // drop r*h by a factor of nine.
  CHECK(scaled_300 / scaled_100 > 0.3);
}

// The interval integrands carry 1/r_a(tau)^5 along the *past* trajectory. A
// field point inside the orbit can be far from both bodies now and yet have had
// one sweep right through it earlier, which makes the integral blow up like
// b^-4 in the impact parameter. This measures how close that gets, in the
// orbital plane where it is worst.
void test_close_approach_in_the_near_zone() {
  INFO("Closest retarded approach, and its effect on the interval integrals");
  const auto trajectory = wave_test_trajectory();
  const double mass = 0.5;
  // Points on the +y axis, i.e. on the orbital track itself at radius 6.
  for (const double radius : {3.0, 6.0, 9.0, 20.0, 60.0}) {
    const std::array<double, 3> field_point{{0.0, radius, 0.0}};
    const double closest =
        detail::closest_retarded_approach(field_point, trajectory, mass, mass);
    const tnsr::I<DataVector, 3, Frame::Inertial> x{
        {{DataVector{1, 0.0}, DataVector{1, radius}, DataVector{1, 0.0}}}};
    tnsr::ii<DataVector, 3> interval{};
    const auto times = detail::retarded_times(x, trajectory, mass, mass, 0.0);
    detail::htt_interval(make_not_null(&interval), x, trajectory, mass, mass,
                         times, 0.0);
    double largest = 0.0;
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        largest = std::max(largest, std::abs(interval.get(i, j)[0]));
      }
    }
    CAPTURE(radius);
    CAPTURE(closest);
    CAPTURE(largest);
    // No close approach ever happens, and the reason is geometric rather than
    // lucky: the integration span is |t^r_a| = r_a, so a nearby field point
    // gets a short window in which the binary barely rotates, while a window
    // long enough for a body to sweep round -- of order a quarter period,
    // ~73 M here -- belongs to a field point far outside the orbit. The two
    // conditions exclude each other for any bound, slowly moving binary.
    CHECK(largest < 0.02);
    // The measurement itself is always finite.
    CHECK(std::isfinite(largest));
    CHECK(closest > 5.0);
  }
}

// dt(h^TT) is taken by differencing the whole construction in the present
// time, so there is nothing hand-differentiated to check against. Two things
// are checked instead: that the difference converges at the expected order, and
// that its magnitude is the one a wave oscillating at 2*Omega must have.
void test_dt_htt() {
  INFO("Time derivative of the wave");
  const auto trajectory = wave_test_trajectory();
  const double omega = detail::circular_orbit(12.0, pn_eta)[1];
  const tnsr::I<DataVector, 3, Frame::Inertial> x{
      {{DataVector{60.0, -90.0, 30.0}, DataVector{40.0, 20.0, -70.0},
        DataVector{15.0, 35.0, 50.0}}}};

  tnsr::ii<DataVector, 3> coarse{};
  tnsr::ii<DataVector, 3> fine{};
  tnsr::ii<DataVector, 3> finer{};
  detail::dt_htt(make_not_null(&coarse), x, trajectory, 0.5, 0.5, 2.0);
  detail::dt_htt(make_not_null(&fine), x, trajectory, 0.5, 0.5, 1.0);
  detail::dt_htt(make_not_null(&finer), x, trajectory, 0.5, 0.5, 0.5);

  // Second-order backward differences: halving the step should cut the change
  // between successive refinements by about four.
  double change_coarse = 0.0;
  double change_fine = 0.0;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      change_coarse =
          std::max(change_coarse, max(abs(coarse.get(i, j) - fine.get(i, j))));
      change_fine =
          std::max(change_fine, max(abs(fine.get(i, j) - finer.get(i, j))));
    }
  }
  CAPTURE(change_coarse);
  CAPTURE(change_fine);
  CHECK(change_fine > 0.0);
  CHECK(change_coarse / change_fine > 2.5);

  // The wave of a circular binary oscillates at twice the orbital frequency,
  // so |dt h| should sit near 2 Omega |h|. A wide band, since the phase varies
  // across the sample points.
  tnsr::ii<DataVector, 3> value{};
  detail::htt_total(make_not_null(&value), x, trajectory, 0.5, 0.5, 0.0);
  double largest_value = 0.0;
  double largest_rate = 0.0;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      largest_value = std::max(largest_value, max(abs(value.get(i, j))));
      largest_rate = std::max(largest_rate, max(abs(finer.get(i, j))));
    }
  }
  const double implied_frequency = largest_rate / largest_value;
  CAPTURE(implied_frequency);
  CAPTURE(2.0 * omega);
  CHECK(implied_frequency > 0.3 * 2.0 * omega);
  CHECK(implied_frequency < 3.0 * 2.0 * omega);
}

void test_wave_attenuation() {
  INFO("Attenuation vanishes at the punctures and saturates far away");
  const std::array<double, 3> position_1{{6.0, 0.0, 0.0}};
  const std::array<double, 3> position_2{{-6.0, 0.0, 0.0}};
  const double width = 3.6;
  const tnsr::I<DataVector, 3, Frame::Inertial> x{
      {{DataVector{6.0, -6.0, 0.0, 300.0}, DataVector{0.0, 0.0, 0.0, 300.0},
        DataVector{0.0, 0.0, 0.0, 300.0}}}};
  Scalar<DataVector> attenuation{};
  detail::wave_attenuation(make_not_null(&attenuation), x, position_1,
                           position_2, width);
  // Exactly zero on each puncture.
  CHECK(get(attenuation)[0] == approx(0.0));
  CHECK(get(attenuation)[1] == approx(0.0));
  // Strictly between zero and one in the middle, and saturated far away.
  CHECK(get(attenuation)[2] > 0.0);
  CHECK(get(attenuation)[2] < 1.0);
  CHECK(get(attenuation)[3] == approx(1.0));

  // The product with h^TT must be finite at the puncture: h diverges like
  // 1/r_A while the attenuation vanishes like r_A^2.
  const DataVector tiny{1.0e-3, 1.0e-4, 1.0e-5};
  const tnsr::I<DataVector, 3, Frame::Inertial> near_puncture{
      {{6.0 + tiny, DataVector{3, 0.0}, DataVector{3, 0.0}}}};
  Scalar<DataVector> attenuation_near{};
  detail::wave_attenuation(make_not_null(&attenuation_near), near_puncture,
                           position_1, position_2, width);
  tnsr::ii<DataVector, 3> wave{};
  detail::htt_instantaneous(make_not_null(&wave), near_puncture, position_1,
                            position_2, 0.5, 0.5);
  const DataVector product = get(attenuation_near) * get<0, 0>(wave);
  // Approaching the puncture the product goes to zero, not to infinity.
  CHECK(std::abs(product[2]) < std::abs(product[1]));
  CHECK(std::abs(product[1]) < std::abs(product[0]));
}

// The wave reaches the conformal metric, and adding it does not break the
// unimodular gauge that defines this class's conformal split.
void test_wave_enters_conformal_metric(
    const NumericBinaryWithWaves& with_waves,
    const NumericBinaryWithWaves& without_waves,
    const tnsr::I<DataVector, 3, Frame::Inertial>& x) {
  INFO("The wave is added to the conformal metric, preserving det = 1");
  using conformal_metric_tag =
      Tags::ConformalMetric<DataVector, 3, Frame::Inertial>;
  const auto with = get<conformal_metric_tag>(
      with_waves.variables(x, tmpl::list<conformal_metric_tag>{}));
  const auto without = get<conformal_metric_tag>(
      without_waves.variables(x, tmpl::list<conformal_metric_tag>{}));

  // Something was actually added.
  double largest_difference = 0.0;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      largest_difference = std::max(
          largest_difference, max(abs(with.get(i, j) - without.get(i, j))));
    }
  }
  CHECK(largest_difference > 1.0e-6);

  // Only the part of h^TT that is trace free with respect to the loaded
  // conformal metric is added, so det(gamma-bar) = 1 survives up to O(h^2).
  const auto determinant = determinant_and_inverse(with).first;
  const DataVector deviation = abs(get(determinant) - 1.0);
  CAPTURE(largest_difference);
  CAPTURE(max(deviation));
  // The bound is quadratic rather than an absolute number on purpose. These
  // test points sit only ~6 M from bodies of mass 0.5, so the wave amplitude
  // here is ~3e-2 -- far larger than anything in a real wave zone -- and any
  // fixed tolerance would be meaningless. What must hold is that the deviation
  // is O(h^2): if the trace projection were dropped it would be O(h), which
  // this bound rejects by more than an order of magnitude.
  CHECK(max(deviation) < 2.0 * square(largest_difference));
  CHECK(max(deviation) < 0.1 * largest_difference);
}

// dt(ConformalMetric) must now treat the two parts differently: Lie drag for
// the loaded data, a difference in the present time for the wave. The test
// background uses Omega = 0.3, while the wave rotates at the post-Newtonian
// 2*Omega ~ 0.043, so if the wave were still being dragged along the frame's
// Killing vector its contribution to dt would be roughly seven times larger.
void test_wave_time_derivative_is_not_the_frame_drag(
    const NumericBinaryWithWaves& with_waves,
    const NumericBinaryWithWaves& without_waves, const Mesh<3>& mesh,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>& inv_jacobian,
    const tnsr::I<DataVector, 3, Frame::Inertial>& x) {
  INFO("The wave's time derivative is not the frame's Lie drag");
  using metric_tag = Tags::ConformalMetric<DataVector, 3, Frame::Inertial>;
  using dt_tag = ::Tags::dt<metric_tag>;
  const auto vars_with = with_waves.variables(x, mesh, inv_jacobian,
                                              tmpl::list<metric_tag, dt_tag>{});
  const auto vars_without = without_waves.variables(
      x, mesh, inv_jacobian, tmpl::list<metric_tag, dt_tag>{});

  double wave_size = 0.0;
  double wave_rate = 0.0;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      wave_size = std::max(wave_size,
                           max(abs(get<metric_tag>(vars_with).get(i, j) -
                                   get<metric_tag>(vars_without).get(i, j))));
      wave_rate =
          std::max(wave_rate, max(abs(get<dt_tag>(vars_with).get(i, j) -
                                      get<dt_tag>(vars_without).get(i, j))));
    }
  }
  CAPTURE(wave_size);
  CAPTURE(wave_rate);
  CHECK(wave_size > 1.0e-6);
  // Doing something.
  CHECK(wave_rate > 0.0);
  // But not the frame drag, which would give ~Omega * wave_size.
  CHECK(wave_rate < 0.5 * angular_velocity * wave_size);
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticData.Xcts.NumericBinaryWithWaves",
    "[PointwiseFunctions][Unit]") {
  test_complex_step_derivatives();
  test_circular_orbit();
  test_conservative_flow();
  test_flux_energy_balance();
  test_trajectory_sampling();
  test_past_widens_for_circular_orbit();
  test_quasi_circular_state();
  test_derived_orbit_covers_the_domain();
  test_eccentric_configuration();
  test_strong_field_is_refused();

  test_kernel_reproduces_momentum_sum();
  test_instantaneous_matches_uncombined();
  test_instantaneous_structure();
  test_instantaneous_falls_off();
  test_retarded_time();
  test_retarded_falls_off_like_one_over_r();
  test_close_approach_in_the_near_zone();
  test_dt_htt();
  test_wave_attenuation();

  write_volume_file();
  // The wave is effectively switched off in this one: the test points sit at
  // r_A ~ 6 from each body, where an attenuation width of 1e6 gives
  // f_att ~ 1e-21, far below round-off. That lets the loading, the unimodular
  // split and the Killing-vector time derivative be checked in isolation.
  const NumericBinaryWithWaves background{h5_file_name,
                                          "element_data",
                                          0,
                                          false,
                                          angular_velocity,
                                          expansion,
                                          /*separation=*/12.0,
                                          /*target_eccentricity=*/0.0,
                                          /*mass_left=*/0.5,
                                          /*mass_right=*/0.5,
                                          /*attenuation_width=*/1.0e6,
                                          /*past_evolution_duration=*/50.0,
                                          /*past_evolution_time_step=*/2.0};
  // The same data with a realistic attenuation width, so the wave is present.
  const NumericBinaryWithWaves background_with_waves{
      h5_file_name,
      "element_data",
      0,
      false,
      angular_velocity,
      expansion,
      /*separation=*/12.0,
      /*target_eccentricity=*/0.0,
      /*mass_left=*/0.5,
      /*mass_right=*/0.5,
      /*attenuation_width=*/3.6,
      /*past_evolution_duration=*/50.0,
      /*past_evolution_time_step=*/2.0};

  // Serialization drops the loaded data and the trajectory and rebuilds both,
  // so this checks that the round trip is faithful.
  {
    INFO("Serialization round trip");
    const auto deserialized = serialize_and_deserialize(background);
    CHECK(deserialized == background);
    CHECK(deserialized.pn_inspiral() == background.pn_inspiral());
    CHECK(not background.pn_inspiral().state.empty());
  }

  // Interior points of the brick; boundary points are ambiguous on
  // Gauss-Lobatto grids.
  const tnsr::I<DataVector, 3, Frame::Inertial> x{
      {{DataVector{0.0, 0.5, -0.25}, DataVector{0.0, 0.0, 0.5},
        DataVector{0.0, 0.0, 0.75}}}};
  test_conformal_split(background, x);

  // The brick maps the logical cube to itself, so the inverse Jacobian is the
  // identity. All the loaded data is constant, so the spectral derivatives
  // vanish and the expected values stay analytic.
  const Mesh<3> mesh{3, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const tnsr::I<DataVector, 3, Frame::Inertial> x_grid{
      {{DataVector{mesh.number_of_grid_points(), 0.0},
        DataVector{mesh.number_of_grid_points(), 0.0},
        DataVector{mesh.number_of_grid_points(), 0.0}}}};
  auto inv_jacobian = make_with_value<
      InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>>(
      get<0>(x_grid), 0.0);
  for (size_t i = 0; i < 3; ++i) {
    inv_jacobian.get(i, i) = 1.0;
  }
  test_killing_vector_time_derivative(background, mesh, inv_jacobian, x_grid);
  test_wave_enters_conformal_metric(background_with_waves, background, x);
  test_wave_time_derivative_is_not_the_frame_drag(
      background_with_waves, background, mesh, inv_jacobian, x_grid);

  file_system::rm(h5_file_name, true);
}

}  // namespace Xcts::AnalyticData
