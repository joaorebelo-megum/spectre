// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/Xcts/NumericBinaryWithWaves.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/OdeIntegration/OdeIntegration.hpp"
#include "NumericalAlgorithms/RootFinding/GslMultiRoot.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/CommonVariables.tpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"

namespace Xcts::AnalyticData {

namespace detail {

namespace {

// Perturbation for complex-step differentiation. Any value small enough that
// h^2 is negligible works, because there is no subtractive cancellation; 1e-100
// leaves a truncation error of 1e-200 while staying far above the smallest
// normal double (~2.2e-308), so no intermediate underflows.
constexpr double complex_step = 1.0e-100;

// Euler-Mascheroni constant, needed by the 3PN logarithmic flux coefficient.
constexpr double euler_gamma = 0.57721566490153286060651209008240243;

// Below this separation the post-Newtonian expansion parameter v/c ~ 1/sqrt(r)
// exceeds 0.4 and the 3.5PN flux series stops converging usefully. A trajectory
// that gets here is not usable as wave-generating history.
constexpr double minimum_post_newtonian_separation = 6.0;

std::array<double, 3> cross_product(const std::array<double, 3>& a,
                                    const std::array<double, 3>& b) {
  return {{a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
           a[0] * b[1] - a[1] * b[0]}};
}

double dot_product(const std::array<double, 3>& a,
                   const std::array<double, 3>& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

double magnitude(const std::array<double, 3>& a) {
  return sqrt(dot_product(a, a));
}

// Split the six-component state into the reduced separation and momentum.
std::array<double, 3> separation_of(const std::array<double, 6>& state) {
  return {{state[0], state[1], state[2]}};
}
std::array<double, 3> momentum_of(const std::array<double, 6>& state) {
  return {{state[3], state[4], state[5]}};
}

}  // namespace

template <typename DataType>
DataType reduced_hamiltonian(const DataType& q, const DataType& s,
                             const DataType& z, const double eta) {
  const double eta2 = eta * eta;
  const double eta3 = eta2 * eta;
  const double pi2 = M_PI * M_PI;
  const DataType u = 1.0 / q;
  const DataType u2 = u * u;
  const DataType u3 = u2 * u;
  const DataType u4 = u3 * u;
  const DataType s2 = s * s;
  const DataType s3 = s2 * s;
  const DataType s4 = s3 * s;
  const DataType z2 = z * z;
  const DataType z3 = z2 * z;

  const DataType newtonian = 0.5 * s - u;

  const DataType one_pn = 0.125 * (3.0 * eta - 1.0) * s2 -
                          0.5 * u * ((3.0 + eta) * s + eta * z) + 0.5 * u2;

  const DataType two_pn = (1.0 / 16.0) * (1.0 - 5.0 * eta + 5.0 * eta2) * s3 +
                          0.125 * u *
                              ((5.0 - 20.0 * eta - 3.0 * eta2) * s2 -
                               2.0 * eta2 * z * s - 3.0 * eta2 * z2) +
                          0.5 * u2 * ((5.0 + 8.0 * eta) * s + 3.0 * eta * z) -
                          0.25 * u3 * (1.0 + 3.0 * eta);

  const DataType three_pn =
      (1.0 / 128.0) * (-5.0 + 35.0 * eta - 70.0 * eta2 + 35.0 * eta3) * s4 +
      (1.0 / 16.0) * u *
          ((-7.0 + 42.0 * eta - 53.0 * eta2 - 5.0 * eta3) * s3 +
           (2.0 - 3.0 * eta) * eta2 * z * s2 +
           3.0 * (1.0 - eta) * eta2 * z2 * s - 5.0 * eta3 * z3) +
      (1.0 / 16.0) * u2 *
          ((-27.0 + 136.0 * eta + 109.0 * eta2) * s2 +
           (17.0 + 30.0 * eta) * eta * z * s +
           (4.0 / 3.0) * (5.0 + 43.0 * eta) * eta * z2) +
      u3 * ((-25.0 / 8.0 + (pi2 / 64.0 - 335.0 / 48.0) * eta -
             23.0 / 8.0 * eta2) *
                s +
            (-85.0 / 16.0 - 3.0 * pi2 / 64.0 - 7.0 / 4.0 * eta) * eta * z) +
      u4 * (0.125 + (109.0 / 12.0 - 21.0 * pi2 / 32.0) * eta);

  return newtonian + one_pn + two_pn + three_pn;
}

template double reduced_hamiltonian(const double& q, const double& s,
                                    const double& z, double eta);
template std::complex<double> reduced_hamiltonian(const std::complex<double>& q,
                                                  const std::complex<double>& s,
                                                  const std::complex<double>& z,
                                                  double eta);

std::array<double, 3> hamiltonian_derivatives(const double q, const double s,
                                              const double z,
                                              const double eta) {
  using Complex = std::complex<double>;
  const Complex qc{q, 0.0};
  const Complex sc{s, 0.0};
  const Complex zc{z, 0.0};
  const Complex step{0.0, complex_step};
  return {{imag(reduced_hamiltonian(qc + step, sc, zc, eta)) / complex_step,
           imag(reduced_hamiltonian(qc, sc + step, zc, eta)) / complex_step,
           imag(reduced_hamiltonian(qc, sc, zc + step, eta)) / complex_step}};
}

std::array<double, 3> hamiltonian_deriv_momentum(
    const std::array<double, 3>& separation,
    const std::array<double, 3>& momentum, const double eta) {
  const double q = magnitude(separation);
  const std::array<double, 3> normal{
      {separation[0] / q, separation[1] / q, separation[2] / q}};
  const double s = dot_product(momentum, momentum);
  const double n_dot_p = dot_product(normal, momentum);
  const auto derivs = hamiltonian_derivatives(q, s, n_dot_p * n_dot_p, eta);
  const double d_ds = derivs[1];
  const double d_dz = derivs[2];
  std::array<double, 3> result{};
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(result, i) = 2.0 * d_ds * gsl::at(momentum, i) +
                         2.0 * d_dz * n_dot_p * gsl::at(normal, i);
  }
  return result;
}

std::array<double, 3> hamiltonian_deriv_separation(
    const std::array<double, 3>& separation,
    const std::array<double, 3>& momentum, const double eta) {
  const double q = magnitude(separation);
  const std::array<double, 3> normal{
      {separation[0] / q, separation[1] / q, separation[2] / q}};
  const double s = dot_product(momentum, momentum);
  const double n_dot_p = dot_product(normal, momentum);
  const auto derivs = hamiltonian_derivatives(q, s, n_dot_p * n_dot_p, eta);
  const double d_dq = derivs[0];
  const double d_dz = derivs[2];
  std::array<double, 3> result{};
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(result, i) =
        d_dq * gsl::at(normal, i) +
        2.0 * d_dz * n_dot_p / q *
            (gsl::at(momentum, i) - n_dot_p * gsl::at(normal, i));
  }
  return result;
}

double orbital_frequency(const std::array<double, 3>& separation,
                         const std::array<double, 3>& momentum,
                         const double eta) {
  const auto velocity = hamiltonian_deriv_momentum(separation, momentum, eta);
  const double q = magnitude(separation);
  return magnitude(cross_product(separation, velocity)) / (q * q);
}

double energy_flux(const double v_omega, const double eta) {
  const double eta2 = eta * eta;
  const double eta3 = eta2 * eta;
  const double pi2 = M_PI * M_PI;
  const double f2 = -1247.0 / 336.0 - 35.0 / 12.0 * eta;
  const double f3 = 4.0 * M_PI;
  const double f4 =
      -44711.0 / 9072.0 + 9271.0 / 504.0 * eta + 65.0 / 18.0 * eta2;
  const double f5 = -(8191.0 / 672.0 + 583.0 / 24.0 * eta) * M_PI;
  const double f_l6 = -1712.0 / 105.0;
  const double f6 = 6643739519.0 / 69854400.0 + 16.0 / 3.0 * pi2 -
                    1712.0 / 105.0 * euler_gamma +
                    (-134543.0 / 7776.0 + 41.0 / 48.0 * pi2) * eta -
                    94403.0 / 3024.0 * eta2 - 775.0 / 324.0 * eta3;
  const double f7 =
      (-16285.0 / 504.0 + 214745.0 / 1728.0 * eta + 193385.0 / 3024.0 * eta2) *
      M_PI;
  const double v2 = v_omega * v_omega;
  const double v3 = v2 * v_omega;
  const double v4 = v2 * v2;
  const double v5 = v4 * v_omega;
  const double v6 = v3 * v3;
  const double v7 = v6 * v_omega;
  const double v10 = v5 * v5;
  const double series = 1.0 + f2 * v2 + f3 * v3 + f4 * v4 + f5 * v5 +
                        (f6 + f_l6 * log(4.0 * v_omega)) * v6 + f7 * v7;
  return -32.0 / 5.0 * eta2 * v10 * series;
}

std::array<double, 6> inspiral_rhs(const std::array<double, 6>& state,
                                   const double eta,
                                   const bool with_radiation_reaction) {
  const auto separation = separation_of(state);
  const auto momentum = momentum_of(state);
  const auto velocity = hamiltonian_deriv_momentum(separation, momentum, eta);
  const auto force = hamiltonian_deriv_separation(separation, momentum, eta);
  std::array<double, 6> result{};
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(result, i) = gsl::at(velocity, i);
    gsl::at(result, i + 3) = -gsl::at(force, i);
  }
  if (not with_radiation_reaction) {
    return result;
  }
  // F_i = (dE/dt) p_i / (eta * omega * |q x p|). The normalisation is fixed by
  // requiring dHhat/dt = (dE/dt) / eta for circular orbits.
  const double q = magnitude(separation);
  const double omega = magnitude(cross_product(separation, velocity)) / (q * q);
  const double reduced_angular_momentum =
      magnitude(cross_product(separation, momentum));
  const double flux = energy_flux(cbrt(omega), eta);
  const double prefactor = flux / (eta * omega * reduced_angular_momentum);
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(result, i + 3) += prefactor * gsl::at(momentum, i);
  }
  return result;
}

std::array<double, 2> circular_orbit(const double separation,
                                     const double eta) {
  // Radial balance at n.p = 0: d(n.p)/dt = 2 Hhat_s s / q - Hhat_q = 0.
  const auto radial_balance = [&separation, &eta](const double s) {
    const auto derivs = hamiltonian_derivatives(separation, s, 0.0, eta);
    return 2.0 * derivs[1] * s / separation - derivs[0];
  };
  // The Newtonian solution is s = 1/q; bracket generously around it.
  const double newtonian_s = 1.0 / separation;
  const double s = RootFinder::toms748(radial_balance, 0.1 * newtonian_s,
                                       10.0 * newtonian_s, 1.0e-15, 1.0e-15);
  const auto derivs = hamiltonian_derivatives(separation, s, 0.0, eta);
  return {{s, 2.0 * derivs[1] * sqrt(s) / separation}};
}

std::array<double, 2> quasi_circular_state(const double separation,
                                           const double eta) {
  const auto circular = circular_orbit(separation, eta);
  const double omega = circular[1];
  // dEhat/dq along the sequence. The sequence is defined through a root find,
  // so complex step cannot be threaded through it; a central difference is
  // used instead. The sequence is smooth and Ehat is O(0.01), so a step of
  // 1e-5 leaves a truncation error far below the accuracy that adot needs.
  const double step = 1.0e-5;
  const auto energy_at = [&eta](const double q) {
    return reduced_hamiltonian(q, circular_orbit(q, eta)[0], 0.0, eta);
  };
  const double denergy_dq =
      (energy_at(separation + step) - energy_at(separation - step)) /
      (2.0 * step);
  const double radial_velocity =
      (energy_flux(cbrt(omega), eta) / eta) / denergy_dq;
  return {{omega, radial_velocity / separation}};
}

double PnInspiral::earliest_time() const {
  return state.empty() ? 0.0
                       : -time_step * static_cast<double>(state.size() - 1);
}

std::array<double, 6> PnInspiral::state_at(const double time) const {
  ASSERT(not state.empty(), "The past evolution has not been computed.");
  ASSERT(time <= 0.0 and time >= earliest_time() * (1.0 + 1.0e-12),
         "Time " << time << " is outside the evolved interval ["
                 << earliest_time() << ", 0].");
  // Samples run backwards from the present: sample n is at t_n = -n * dt.
  const double sample_coordinate = -time / time_step;
  const size_t lower =
      std::min(static_cast<size_t>(sample_coordinate), state.size() - 2);
  const double theta = sample_coordinate - static_cast<double>(lower);
  // Cubic Hermite on [t_lower, t_lower+1]. The interval width in *time* is
  // negative because the samples march into the past.
  const double interval = -time_step;
  const double theta2 = theta * theta;
  const double theta3 = theta2 * theta;
  const double basis_value_lower = 2.0 * theta3 - 3.0 * theta2 + 1.0;
  const double basis_slope_lower = theta3 - 2.0 * theta2 + theta;
  const double basis_value_upper = -2.0 * theta3 + 3.0 * theta2;
  const double basis_slope_upper = theta3 - theta2;
  const auto& value_lower = state[lower];
  const auto& value_upper = state[lower + 1];
  const auto& slope_lower = dt_state[lower];
  const auto& slope_upper = dt_state[lower + 1];
  std::array<double, 6> result{};
  for (size_t i = 0; i < 6; ++i) {
    gsl::at(result, i) =
        basis_value_lower * gsl::at(value_lower, i) +
        basis_slope_lower * interval * gsl::at(slope_lower, i) +
        basis_value_upper * gsl::at(value_upper, i) +
        basis_slope_upper * interval * gsl::at(slope_upper, i);
  }
  return result;
}

void PnInspiral::pup(PUP::er& p) {
  p | time_step;
  p | eta;
  p | state;
  p | dt_state;
}

bool operator==(const PnInspiral& lhs, const PnInspiral& rhs) {
  return lhs.time_step == rhs.time_step and lhs.eta == rhs.eta and
         lhs.state == rhs.state and lhs.dt_state == rhs.dt_state;
}

bool operator!=(const PnInspiral& lhs, const PnInspiral& rhs) {
  return not(lhs == rhs);
}

PnInspiral evolve_binary_backwards(const double separation,
                                   const double angular_velocity,
                                   const double expansion, const double eta,
                                   const double duration,
                                   const double time_step,
                                   const bool with_radiation_reaction) {
  PnInspiral result{};
  result.time_step = time_step;
  result.eta = eta;

  // The present state, taken exactly as the previous solve specifies it. The
  // velocity is prescribed; the canonical momentum follows from inverting
  // qdot = dHhat/dp, which is a 2-D problem because the motion is planar.
  const std::array<double, 3> initial_separation{{separation, 0.0, 0.0}};
  const std::array<double, 3> initial_velocity{
      {expansion * separation, angular_velocity * separation, 0.0}};
  const auto momentum_residual =
      [&initial_separation, &initial_velocity,
       &eta](const std::array<double, 2>& trial) -> std::array<double, 2> {
    const std::array<double, 3> momentum{{trial[0], trial[1], 0.0}};
    const auto velocity =
        hamiltonian_deriv_momentum(initial_separation, momentum, eta);
    return {
        {velocity[0] - initial_velocity[0], velocity[1] - initial_velocity[1]}};
  };
  // At leading order qdot = p, so the velocity itself is a good starting guess.
  const std::array<double, 2> guess{{initial_velocity[0], initial_velocity[1]}};
  const auto initial_momentum = RootFinder::gsl_multiroot(
      momentum_residual, guess,
      RootFinder::StoppingConditions::Residual{1.0e-13}, 100);

  std::array<double, 6> current{{initial_separation[0], initial_separation[1],
                                 initial_separation[2], initial_momentum[0],
                                 initial_momentum[1], 0.0}};

  // Integrate in tau = -t so the stepper always runs forwards. The trajectory
  // satisfies the same equations, so dY/dtau = -RHS(Y).
  const auto system = [&eta, &with_radiation_reaction](
                          const std::array<double, 6>& y,
                          std::array<double, 6>& dy_dtau,
                          const double /*tau*/) {
    const auto rhs = inspiral_rhs(y, eta, with_radiation_reaction);
    for (size_t i = 0; i < 6; ++i) {
      gsl::at(dy_dtau, i) = -gsl::at(rhs, i);
    }
  };

  const size_t num_samples =
      static_cast<size_t>(std::ceil(duration / time_step)) + 1;
  result.state.reserve(num_samples);
  result.dt_state.reserve(num_samples);
  const auto record = [&result, &eta, &with_radiation_reaction,
                       &separation](const std::array<double, 6>& y) {
    // Refuse to return a trajectory that has left the post-Newtonian regime.
    // Without this the failure is silent: an eccentric configuration reaches
    // the strong field at periapsis, where v_omega ~ 0.5 puts the 3.5PN flux
    // series far outside its domain of convergence, and the backwards
    // integration then pumps in a spurious amount of energy on every passage.
    const double radius = magnitude(std::array<double, 3>{{y[0], y[1], y[2]}});
    if (radius < minimum_post_newtonian_separation) {
      ERROR("The past evolution reached a separation of "
            << radius
            << " M, inside the strong-field region where the "
               "post-Newtonian expansion is not valid (v/c ~ "
            << 1.0 / sqrt(radius) << "). The binary started at a separation of "
            << separation
            << " M, so the orbit is far from circular: check that "
               "'AngularVelocity' and 'Expansion' are the quasi-circular "
               "values for this separation. Reduce 'PastEvolutionDuration' "
               "if a shorter history is enough.");
    }
    result.state.push_back(y);
    result.dt_state.push_back(inspiral_rhs(y, eta, with_radiation_reaction));
  };
  record(current);

  boost::numeric::odeint::dense_output_runge_kutta<
      boost::numeric::odeint::controlled_runge_kutta<
          boost::numeric::odeint::runge_kutta_dopri5<std::array<double, 6>>>>
      stepper = boost::numeric::odeint::make_dense_output(
          1.0e-12, 1.0e-12,
          boost::numeric::odeint::runge_kutta_dopri5<std::array<double, 6>>{});
  stepper.initialize(current, 0.0, time_step);
  for (size_t sample = 1; sample < num_samples; ++sample) {
    const double target_tau = time_step * static_cast<double>(sample);
    while (stepper.current_time() < target_tau) {
      stepper.do_step(system);
    }
    stepper.calc_state(target_tau, current);
    record(current);
  }
  return result;
}

namespace {

// Field-point quantities relative to one body: r_A, 1/r_A and the unit vector
// n_A. Shared by the kernel and the instantaneous term.
template <typename DataType>
struct BodyGeometry {
  DataType radius{};
  DataType inverse_radius{};
  std::array<DataType, 3> normal{};
};

template <typename DataType>
BodyGeometry<DataType> body_geometry(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    const std::array<double, 3>& body_position) {
  BodyGeometry<DataType> result{};
  std::array<DataType, 3> offset{};
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(offset, i) = x.get(i) - gsl::at(body_position, i);
  }
  result.radius =
      sqrt(square(offset[0]) + square(offset[1]) + square(offset[2]));
  result.inverse_radius = 1.0 / result.radius;
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(result.normal, i) = gsl::at(offset, i) * result.inverse_radius;
  }
  return result;
}

// Every piece of h^TT -- present, retarded and both interval integrands --
// has the same tensor structure
//   c_delta delta^ij + c_uu u^i u^j + c_nn n^i n^j + c_un u^(i n^j),
// differing only in the coefficients. Assembling it once means the four sets of
// coefficients transcribed from the paper are the only thing that can be wrong.
// Written for a single point so it can be used inside the per-point loop of the
// retarded terms, where each point has its own retarded time.
void add_wave_structure(const gsl::not_null<tnsr::ii<DataVector, 3>*> result,
                        const size_t point, const double prefactor,
                        const double coefficient_delta,
                        const double coefficient_uu,
                        const double coefficient_nn,
                        const double coefficient_un,
                        const std::array<double, 3>& u,
                        const std::array<double, 3>& normal) {
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      double value = coefficient_uu * gsl::at(u, i) * gsl::at(u, j) +
                     coefficient_nn * gsl::at(normal, i) * gsl::at(normal, j) +
                     // u^(i n^j) with the 1/2 symmetrisation convention.
                     0.5 * coefficient_un *
                         (gsl::at(u, i) * gsl::at(normal, j) +
                          gsl::at(u, j) * gsl::at(normal, i));
      if (i == j) {
        value += coefficient_delta;
      }
      result->get(i, j)[point] += prefactor * value;
    }
  }
}

}  // namespace

template <typename DataType>
void post_newtonian_kernel(const gsl::not_null<tnsr::ii<DataType, 3>*> result,
                           const tnsr::I<DataType, 3, Frame::Inertial>& x,
                           const std::array<double, 3>& body_position,
                           const std::array<double, 3>& u) {
  const auto geometry = body_geometry(x, body_position);
  const double u_squared = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
  DataType u_dot_n = make_with_value<DataType>(geometry.radius, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    u_dot_n += gsl::at(u, i) * gsl::at(geometry.normal, i);
  }
  const DataType u_dot_n_squared = square(u_dot_n);
  const DataType delta_coefficient = u_squared - 5.0 * u_dot_n_squared;
  const DataType normal_coefficient = 3.0 * u_dot_n_squared - 5.0 * u_squared;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      // 12 u^(i n^j) with X^(i Y^j) = (X^i Y^j + X^j Y^i) / 2.
      DataType value = 2.0 * gsl::at(u, i) * gsl::at(u, j) +
                       normal_coefficient * gsl::at(geometry.normal, i) *
                           gsl::at(geometry.normal, j) +
                       6.0 * u_dot_n *
                           (gsl::at(u, i) * gsl::at(geometry.normal, j) +
                            gsl::at(u, j) * gsl::at(geometry.normal, i));
      if (i == j) {
        value += delta_coefficient;
      }
      result->get(i, j) = value * geometry.inverse_radius;
    }
  }
}

template <typename DataType>
void htt_instantaneous(const gsl::not_null<tnsr::ii<DataType, 3>*> result,
                       const tnsr::I<DataType, 3, Frame::Inertial>& x,
                       const std::array<double, 3>& position_1,
                       const std::array<double, 3>& position_2,
                       const double mass_1, const double mass_2) {
  const std::array<BodyGeometry<DataType>, 2> bodies{
      {body_geometry(x, position_1), body_geometry(x, position_2)}};

  // n_12 points from body 1 to body 2, and sigma_A = +1, -1 for A = 1, 2, so
  // that n_AB = sigma_A n_12. This is the only place the orientation matters:
  // every other appearance of n_12 is quadratic.
  std::array<double, 3> separation_vector{};
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(separation_vector, i) =
        gsl::at(position_2, i) - gsl::at(position_1, i);
  }
  const double binary_separation =
      sqrt(square(separation_vector[0]) + square(separation_vector[1]) +
           square(separation_vector[2]));
  std::array<double, 3> binary_normal{};
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(binary_normal, i) =
        gsl::at(separation_vector, i) / binary_separation;
  }
  const double inverse_binary_separation = 1.0 / binary_separation;
  const double inverse_separation_cubed = cube(inverse_binary_separation);

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      result->get(i, j) = make_with_value<DataType>(x.get(0), 0.0);
    }
  }

  for (size_t body = 0; body < 2; ++body) {
    const auto& near = gsl::at(bodies, body);
    const auto& far = gsl::at(bodies, 1 - body);
    const double sigma = (body == 0) ? 1.0 : -1.0;

    DataType cos_angle = make_with_value<DataType>(near.radius, 0.0);
    for (size_t i = 0; i < 3; ++i) {
      cos_angle += gsl::at(binary_normal, i) * gsl::at(near.normal, i);
    }
    const DataType cos_angle_squared = square(cos_angle);
    // s = r_A + r_B + r_12, symmetric under A <-> B.
    const DataType perimeter = near.radius + far.radius + binary_separation;
    const DataType inverse_perimeter = 1.0 / perimeter;

    const DataType near_normal_coefficient =
        3.0 * cos_angle_squared * near.inverse_radius *
            inverse_binary_separation -
        inverse_separation_cubed *
            (square(far.radius) * near.inverse_radius + 3.0 * near.radius) -
        8.0 * inverse_perimeter * (near.inverse_radius + inverse_perimeter);

    const DataType delta_coefficient =
        5.0 * near.radius * inverse_separation_cubed *
            (near.radius / far.radius - 1.0) -
        (16.0 + 5.0 * cos_angle_squared) * near.inverse_radius *
            inverse_binary_separation +
        4.0 * near.inverse_radius / far.radius +
        8.0 * inverse_perimeter *
            (near.inverse_radius + 4.0 * inverse_binary_separation);

    const DataType binary_normal_coefficient =
        2.0 * near.inverse_radius * inverse_binary_separation -
        32.0 * inverse_perimeter *
            (inverse_binary_separation + inverse_perimeter);

    const DataType cross_coefficient =
        2.0 * ((near.radius + far.radius) * inverse_separation_cubed +
               12.0 * square(inverse_perimeter));

    const DataType mixed_coefficient =
        32.0 * sigma *
            (2.0 * square(inverse_perimeter) -
             square(inverse_binary_separation)) +
        12.0 * cos_angle * near.inverse_radius * inverse_binary_separation;

    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        DataType value =
            near_normal_coefficient * gsl::at(near.normal, i) *
                gsl::at(near.normal, j) +
            binary_normal_coefficient * gsl::at(binary_normal, i) *
                gsl::at(binary_normal, j) +
            // n_A^i n_B^j is not symmetric on its own; the sum over A
            // symmetrises it, so accumulate only its symmetric half here.
            0.5 * cross_coefficient *
                (gsl::at(near.normal, i) * gsl::at(far.normal, j) +
                 gsl::at(near.normal, j) * gsl::at(far.normal, i)) +
            // n_A^(i n_12^j) with the 1/2 symmetrisation convention.
            0.5 * mixed_coefficient *
                (gsl::at(near.normal, i) * gsl::at(binary_normal, j) +
                 gsl::at(near.normal, j) * gsl::at(binary_normal, i));
        if (i == j) {
          value += delta_coefficient;
        }
        result->get(i, j) += value;
      }
    }
  }

  const double prefactor = 0.125 * mass_1 * mass_2;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      result->get(i, j) *= prefactor;
    }
  }
}

template <typename DataType>
void htt_total(const gsl::not_null<tnsr::ii<DataType, 3>*> result,
               const tnsr::I<DataType, 3, Frame::Inertial>& x,
               const PnInspiral& inspiral, const double mass_1,
               const double mass_2, const double present_time) {
  const auto configuration =
      configuration_at(inspiral, present_time, mass_1, mass_2);
  htt_instantaneous(result, x, configuration.positions[0],
                    configuration.positions[1], mass_1, mass_2);
  const auto times = retarded_times(x, inspiral, mass_1, mass_2, present_time);
  tnsr::ii<DataType, 3> retarded{};
  htt_retarded(make_not_null(&retarded), x, inspiral, mass_1, mass_2, times);
  tnsr::ii<DataType, 3> interval{};
  htt_interval(make_not_null(&interval), x, inspiral, mass_1, mass_2, times,
               present_time);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      result->get(i, j) += retarded.get(i, j) + interval.get(i, j);
    }
  }
}

template <typename DataType>
void dt_htt(const gsl::not_null<tnsr::ii<DataType, 3>*> result,
            const tnsr::I<DataType, 3, Frame::Inertial>& x,
            const PnInspiral& inspiral, const double mass_1,
            const double mass_2, const double step) {
  // Second-order backward difference. Backward because the trajectory only
  // exists for t <= 0: there is no future to centre a stencil on.
  std::array<tnsr::ii<DataType, 3>, 3> samples{};
  for (size_t k = 0; k < 3; ++k) {
    htt_total(make_not_null(&gsl::at(samples, k)), x, inspiral, mass_1, mass_2,
              -step * static_cast<double>(k));
  }
  const double inverse_step = 1.0 / (2.0 * step);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      result->get(i, j) = (3.0 * samples[0].get(i, j) -
                           4.0 * samples[1].get(i, j) + samples[2].get(i, j)) *
                          inverse_step;
    }
  }
}

template void htt_total(gsl::not_null<tnsr::ii<DataVector, 3>*> result,
                        const tnsr::I<DataVector, 3, Frame::Inertial>& x,
                        const PnInspiral& inspiral, double mass_1,
                        double mass_2, double present_time);
template void dt_htt(gsl::not_null<tnsr::ii<DataVector, 3>*> result,
                     const tnsr::I<DataVector, 3, Frame::Inertial>& x,
                     const PnInspiral& inspiral, double mass_1, double mass_2,
                     double step);

template <typename DataType>
void wave_attenuation(const gsl::not_null<Scalar<DataType>*> result,
                      const tnsr::I<DataType, 3, Frame::Inertial>& x,
                      const std::array<double, 3>& position_1,
                      const std::array<double, 3>& position_2,
                      const double width) {
  const double inverse_width_squared = 1.0 / (width * width);
  get(*result) = make_with_value<DataType>(x.get(0), 1.0);
  for (const auto& position : {position_1, position_2}) {
    DataType radius_squared = make_with_value<DataType>(x.get(0), 0.0);
    for (size_t i = 0; i < 3; ++i) {
      radius_squared += square(x.get(i) - gsl::at(position, i));
    }
    get(*result) *= 1.0 - exp(-radius_squared * inverse_width_squared);
  }
}

BinaryConfiguration configuration_at(const PnInspiral& inspiral,
                                     const double time, const double mass_1,
                                     const double mass_2) {
  const double total_mass = mass_1 + mass_2;
  const double reduced_mass = mass_1 * mass_2 / total_mass;
  // The trajectory is stored in reduced variables and in units M = 1.
  const auto state = inspiral.state_at(time / total_mass);
  BinaryConfiguration result{};
  for (size_t i = 0; i < 3; ++i) {
    const double separation = total_mass * gsl::at(state, i);
    const double momentum = reduced_mass * gsl::at(state, i + 3);
    gsl::at(result.positions[0], i) = mass_2 / total_mass * separation;
    gsl::at(result.positions[1], i) = -mass_1 / total_mass * separation;
    gsl::at(result.momenta[0], i) = momentum;
    gsl::at(result.momenta[1], i) = -momentum;
  }
  return result;
}

double retarded_time(const std::array<double, 3>& field_point,
                     const PnInspiral& inspiral, const size_t body,
                     const double mass_1, const double mass_2,
                     const double max_body_distance,
                     const double present_time) {
  const double total_mass = mass_1 + mass_2;
  // r_a <= |x| + d_max, so t - tau = r_a can never reach below this.
  const double lower_bound =
      present_time - (magnitude(field_point) + max_body_distance);
  const double earliest = inspiral.earliest_time() * total_mass;
  if (lower_bound < earliest) {
    ERROR("The retarded time of a point at radius "
          << magnitude(field_point) << " M needs the trajectory back to t = "
          << lower_bound << " M, but the past evolution only reaches "
          << earliest << " M. Increase 'PastEvolutionDuration' to at least "
          << -lower_bound << ".");
  }
  const auto residual = [&field_point, &inspiral, &body, &mass_1, &mass_2,
                         &present_time](const double time) {
    const auto configuration = configuration_at(inspiral, time, mass_1, mass_2);
    const auto& position = gsl::at(configuration.positions, body);
    std::array<double, 3> offset{};
    for (size_t i = 0; i < 3; ++i) {
      gsl::at(offset, i) = gsl::at(field_point, i) - gsl::at(position, i);
    }
    return present_time - time - magnitude(offset);
  };
  return RootFinder::toms748(residual, lower_bound, present_time, 1.0e-13,
                             1.0e-13);
}

template <typename DataType>
void htt_retarded(const gsl::not_null<tnsr::ii<DataType, 3>*> result,
                  const tnsr::I<DataType, 3, Frame::Inertial>& x,
                  const PnInspiral& inspiral, const double mass_1,
                  const double mass_2,
                  const std::array<DataType, 2>& retarded_times) {
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      result->get(i, j) = make_with_value<DataType>(x.get(0), 0.0);
    }
  }
  // Adds H^TT[u; t_r] for one body, one u-vector and one point.
  const auto add_retarded = [&result](const size_t point, const double sign,
                                      const std::array<double, 3>& u,
                                      const std::array<double, 3>& normal,
                                      const double inverse_radius) {
    const double u_squared = dot_product(u, u);
    const double u_dot_n = dot_product(u, normal);
    add_wave_structure(result, point, sign * inverse_radius,
                       -2.0 * u_squared + 2.0 * square(u_dot_n), 4.0,
                       2.0 * u_squared + 2.0 * square(u_dot_n), -8.0 * u_dot_n,
                       u, normal);
  };

  const size_t num_points = get_size(x.get(0));
  for (size_t point = 0; point < num_points; ++point) {
    const std::array<double, 3> field_point{
        {x.get(0)[point], x.get(1)[point], x.get(2)[point]}};
    for (size_t body = 0; body < 2; ++body) {
      const double time = gsl::at(retarded_times, body)[point];
      const auto configuration =
          configuration_at(inspiral, time, mass_1, mass_2);
      const auto& position = gsl::at(configuration.positions, body);
      std::array<double, 3> offset{};
      for (size_t i = 0; i < 3; ++i) {
        gsl::at(offset, i) = gsl::at(field_point, i) - gsl::at(position, i);
      }
      const double radius = magnitude(offset);
      const double inverse_radius = 1.0 / radius;
      std::array<double, 3> normal{};
      for (size_t i = 0; i < 3; ++i) {
        gsl::at(normal, i) = gsl::at(offset, i) * inverse_radius;
      }

      // + H[p_a / sqrt(m_a)]
      const double body_mass = (body == 0) ? mass_1 : mass_2;
      const double inverse_sqrt_mass = 1.0 / sqrt(body_mass);
      std::array<double, 3> u_momentum{};
      for (size_t i = 0; i < 3; ++i) {
        gsl::at(u_momentum, i) =
            gsl::at(gsl::at(configuration.momenta, body), i) *
            inverse_sqrt_mass;
      }
      add_retarded(point, 1.0, u_momentum, normal, inverse_radius);

      // - H[w], with w built from the separation at the *same* retarded time.
      std::array<double, 3> binary_offset{};
      for (size_t i = 0; i < 3; ++i) {
        gsl::at(binary_offset, i) = gsl::at(configuration.positions[1], i) -
                                    gsl::at(configuration.positions[0], i);
      }
      const double binary_separation = magnitude(binary_offset);
      const double w_magnitude =
          sqrt(mass_1 * mass_2 / (2.0 * binary_separation));
      std::array<double, 3> u_w{};
      for (size_t i = 0; i < 3; ++i) {
        gsl::at(u_w, i) =
            w_magnitude * gsl::at(binary_offset, i) / binary_separation;
      }
      add_retarded(point, -1.0, u_w, normal, inverse_radius);
    }
  }
}

namespace {
// Target spacing of the interval quadrature. The integrands oscillate on the
// wave period pi/Omega, about 146 M for a binary at D = 12, so 5 M gives ~29
// points per period and a composite-Simpson error of order (5/146)^4 ~ 1e-6.
constexpr double interval_quadrature_step = 5.0;

// Step of the backward difference that gives the wave's time derivative. The
// wave varies on the period pi/Omega ~ 146 M, so 1 M leaves a second-order
// truncation error of order (1/146)^2 ~ 5e-5 relative.
constexpr double wave_time_derivative_step = 1.0;
constexpr size_t minimum_quadrature_intervals = 8;

// Distance from a field point to one body at a given time.
double distance_to_body(const std::array<double, 3>& field_point,
                        const BinaryConfiguration& configuration,
                        const size_t body) {
  const auto& position = gsl::at(configuration.positions, body);
  std::array<double, 3> offset{};
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(offset, i) = gsl::at(field_point, i) - gsl::at(position, i);
  }
  return magnitude(offset);
}

double largest_body_distance(const PnInspiral& inspiral, const double mass_1,
                             const double mass_2) {
  double largest_reduced = 0.0;
  for (const auto& sample : inspiral.state) {
    largest_reduced = std::max(
        largest_reduced,
        sqrt(square(sample[0]) + square(sample[1]) + square(sample[2])));
  }
  return std::max(mass_1, mass_2) * largest_reduced;
}
}  // namespace

double closest_retarded_approach(const std::array<double, 3>& field_point,
                                 const PnInspiral& inspiral,
                                 const double mass_1, const double mass_2) {
  const double max_body_distance =
      largest_body_distance(inspiral, mass_1, mass_2);
  double closest = std::numeric_limits<double>::max();
  for (size_t body = 0; body < 2; ++body) {
    const double start = retarded_time(field_point, inspiral, body, mass_1,
                                       mass_2, max_body_distance, 0.0);
    const size_t num_intervals =
        std::max(minimum_quadrature_intervals,
                 2 * static_cast<size_t>(
                         std::ceil(-start / interval_quadrature_step / 2.0)));
    const double step = -start / static_cast<double>(num_intervals);
    for (size_t k = 0; k <= num_intervals; ++k) {
      const double time = start + step * static_cast<double>(k);
      closest = std::min(
          closest, distance_to_body(
                       field_point,
                       configuration_at(inspiral, time, mass_1, mass_2), body));
    }
  }
  return closest;
}

template <typename DataType>
std::array<DataType, 2> retarded_times(
    const tnsr::I<DataType, 3, Frame::Inertial>& x, const PnInspiral& inspiral,
    const double mass_1, const double mass_2, const double present_time) {
  const double max_body_distance =
      largest_body_distance(inspiral, mass_1, mass_2);
  std::array<DataType, 2> result{};
  const size_t num_points = get_size(x.get(0));
  for (size_t body = 0; body < 2; ++body) {
    gsl::at(result, body) = make_with_value<DataType>(x.get(0), 0.0);
    for (size_t point = 0; point < num_points; ++point) {
      const std::array<double, 3> field_point{
          {x.get(0)[point], x.get(1)[point], x.get(2)[point]}};
      gsl::at(result, body)[point] =
          retarded_time(field_point, inspiral, body, mass_1, mass_2,
                        max_body_distance, present_time);
    }
  }
  return result;
}

template std::array<DataVector, 2> retarded_times(
    const tnsr::I<DataVector, 3, Frame::Inertial>& x,
    const PnInspiral& inspiral, double mass_1, double mass_2,
    double present_time);

template <typename DataType>
void htt_interval(const gsl::not_null<tnsr::ii<DataType, 3>*> result,
                  const tnsr::I<DataType, 3, Frame::Inertial>& x,
                  const PnInspiral& inspiral, const double mass_1,
                  const double mass_2,
                  const std::array<DataType, 2>& retarded_times,
                  const double present_time) {
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      result->get(i, j) = make_with_value<DataType>(x.get(0), 0.0);
    }
  }
  // One integrand sample: both u-vectors, both integrals, one quadrature node.
  const auto accumulate = [&result](const size_t point, const double weight,
                                    const std::array<double, 3>& u,
                                    const std::array<double, 3>& normal,
                                    const double sign, const double elapsed,
                                    const double inverse_radius) {
    const double u_squared = dot_product(u, u);
    const double u_dot_n = dot_product(u, normal);
    const double u_dot_n_squared = square(u_dot_n);
    // -(t - tau) / r_a^3
    const double first = -elapsed * cube(inverse_radius) * weight * sign;
    add_wave_structure(
        result, point, first, -5.0 * u_squared + 9.0 * u_dot_n_squared, 6.0,
        9.0 * u_squared - 15.0 * u_dot_n_squared, -12.0 * u_dot_n, u, normal);
    // -(t - tau)^3 / r_a^5
    const double second = -cube(elapsed) * cube(inverse_radius) *
                          square(inverse_radius) * weight * sign;
    add_wave_structure(result, point, second, u_squared - 5.0 * u_dot_n_squared,
                       2.0, -5.0 * u_squared + 35.0 * u_dot_n_squared,
                       -20.0 * u_dot_n, u, normal);
  };

  const size_t num_points = get_size(x.get(0));
  for (size_t point = 0; point < num_points; ++point) {
    const std::array<double, 3> field_point{
        {x.get(0)[point], x.get(1)[point], x.get(2)[point]}};
    for (size_t body = 0; body < 2; ++body) {
      const double start = gsl::at(retarded_times, body)[point];
      const double span = present_time - start;
      const size_t num_intervals =
          std::max(minimum_quadrature_intervals,
                   2 * static_cast<size_t>(
                           std::ceil(span / interval_quadrature_step / 2.0)));
      const double step = span / static_cast<double>(num_intervals);
      const double body_mass = (body == 0) ? mass_1 : mass_2;
      const double inverse_sqrt_mass = 1.0 / sqrt(body_mass);

      for (size_t k = 0; k <= num_intervals; ++k) {
        // Composite Simpson: endpoints 1, odd nodes 4, even interior nodes 2.
        const double simpson =
            (k == 0 or k == num_intervals) ? 1.0 : ((k % 2 == 1) ? 4.0 : 2.0);
        const double weight = simpson * step / 3.0;
        const double time = start + step * static_cast<double>(k);
        const double elapsed = present_time - time;
        const auto configuration =
            configuration_at(inspiral, time, mass_1, mass_2);
        const auto& position = gsl::at(configuration.positions, body);
        std::array<double, 3> offset{};
        for (size_t i = 0; i < 3; ++i) {
          gsl::at(offset, i) = gsl::at(field_point, i) - gsl::at(position, i);
        }
        const double radius = magnitude(offset);
        const double inverse_radius = 1.0 / radius;
        std::array<double, 3> normal{};
        for (size_t i = 0; i < 3; ++i) {
          gsl::at(normal, i) = gsl::at(offset, i) * inverse_radius;
        }

        std::array<double, 3> u_momentum{};
        for (size_t i = 0; i < 3; ++i) {
          gsl::at(u_momentum, i) =
              gsl::at(gsl::at(configuration.momenta, body), i) *
              inverse_sqrt_mass;
        }
        accumulate(point, weight, u_momentum, normal, 1.0, elapsed,
                   inverse_radius);

        std::array<double, 3> binary_offset{};
        for (size_t i = 0; i < 3; ++i) {
          gsl::at(binary_offset, i) = gsl::at(configuration.positions[1], i) -
                                      gsl::at(configuration.positions[0], i);
        }
        const double binary_separation = magnitude(binary_offset);
        const double w_magnitude =
            sqrt(mass_1 * mass_2 / (2.0 * binary_separation));
        std::array<double, 3> u_w{};
        for (size_t i = 0; i < 3; ++i) {
          gsl::at(u_w, i) =
              w_magnitude * gsl::at(binary_offset, i) / binary_separation;
        }
        accumulate(point, weight, u_w, normal, -1.0, elapsed, inverse_radius);
      }
    }
  }
}

template void htt_interval(gsl::not_null<tnsr::ii<DataVector, 3>*> result,
                           const tnsr::I<DataVector, 3, Frame::Inertial>& x,
                           const PnInspiral& inspiral, double mass_1,
                           double mass_2,
                           const std::array<DataVector, 2>& retarded_times,
                           double present_time);

template void htt_retarded(gsl::not_null<tnsr::ii<DataVector, 3>*> result,
                           const tnsr::I<DataVector, 3, Frame::Inertial>& x,
                           const PnInspiral& inspiral, double mass_1,
                           double mass_2,
                           const std::array<DataVector, 2>& retarded_times);

template void post_newtonian_kernel(
    gsl::not_null<tnsr::ii<DataVector, 3>*> result,
    const tnsr::I<DataVector, 3, Frame::Inertial>& x,
    const std::array<double, 3>& body_position, const std::array<double, 3>& u);
template void htt_instantaneous(
    gsl::not_null<tnsr::ii<DataVector, 3>*> result,
    const tnsr::I<DataVector, 3, Frame::Inertial>& x,
    const std::array<double, 3>& position_1,
    const std::array<double, 3>& position_2, double mass_1, double mass_2);
template void wave_attenuation(gsl::not_null<Scalar<DataVector>*> result,
                               const tnsr::I<DataVector, 3, Frame::Inertial>& x,
                               const std::array<double, 3>& position_1,
                               const std::array<double, 3>& position_2,
                               double width);

template <typename DataType>
tnsr::I<DataType, 3> NumericBinaryWithWavesVariables<DataType>::killing_vector()
    const {
  auto result = make_with_value<tnsr::I<DataType, 3>>(get<0>(x), 0.);
  get<0>(result) = -angular_velocity * get<1>(x) + expansion * get<0>(x);
  get<1>(result) = angular_velocity * get<0>(x) + expansion * get<1>(x);
  get<2>(result) = expansion * get<2>(x);
  return result;
}

template <typename DataType>
tnsr::iJ<DataType, 3>
NumericBinaryWithWavesVariables<DataType>::deriv_killing_vector() const {
  auto result = make_with_value<tnsr::iJ<DataType, 3>>(get<0>(x), 0.);
  get<0, 0>(result) = expansion;
  get<1, 1>(result) = expansion;
  get<2, 2>(result) = expansion;
  get<1, 0>(result) = -angular_velocity;
  get<0, 1>(result) = angular_velocity;
  return result;
}

template <typename DataType>
void NumericBinaryWithWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3>*> spatial_metric,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::SpatialMetric<DataType, 3> /*meta*/) const {
  *spatial_metric = get<gr::Tags::SpatialMetric<DataType, 3>>(loaded_vars);
}

template <typename DataType>
void NumericBinaryWithWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::II<DataType, 3>*> inv_spatial_metric,
    const gsl::not_null<Cache*> cache,
    gr::Tags::InverseSpatialMetric<DataType, 3> /*meta*/) const {
  const auto& spatial_metric =
      cache->get_var(*this, gr::Tags::SpatialMetric<DataType, 3>{});
  *inv_spatial_metric = determinant_and_inverse(spatial_metric).second;
}

template <typename DataType>
void NumericBinaryWithWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> lapse,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::Lapse<DataType> /*meta*/) const {
  *lapse = get<gr::Tags::Lapse<DataType>>(loaded_vars);
}

template <typename DataType>
void NumericBinaryWithWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3>*> extrinsic_curvature,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::ExtrinsicCurvature<DataType, 3> /*meta*/) const {
  *extrinsic_curvature =
      get<gr::Tags::ExtrinsicCurvature<DataType, 3>>(loaded_vars);
}

template <typename DataType>
void NumericBinaryWithWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3>*> conformal_metric,
    const gsl::not_null<Cache*> cache,
    Tags::ConformalMetric<DataType, 3, Frame::Inertial> /*meta*/) const {
  const auto& loaded = cache->get_var(*this, LoadedConformalMetric<DataType>{});
  const auto wave = wave_contribution(0.0, cache);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      conformal_metric->get(i, j) = loaded.get(i, j) + wave.get(i, j);
    }
  }
}

template <typename DataType>
void NumericBinaryWithWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3>*> loaded,
    const gsl::not_null<Cache*> cache,
    LoadedConformalMetric<DataType> /*meta*/) const {
  // Unimodular conformal split: det(conformal metric) = 1, so
  // psi = det(gamma)^(1/12) and conformal metric = det(gamma)^(-1/3) gamma.
  const auto& spatial_metric =
      cache->get_var(*this, gr::Tags::SpatialMetric<DataType, 3>{});
  const auto det_spatial_metric = determinant_and_inverse(spatial_metric).first;
  const DataType conformal_scaling = pow(get(det_spatial_metric), -1. / 3.);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      loaded->get(i, j) = conformal_scaling * spatial_metric.get(i, j);
    }
  }
}

template <typename DataType>
void NumericBinaryWithWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ijj<DataType, 3>*> deriv_loaded,
    const gsl::not_null<Cache*> cache,
    ::Tags::deriv<LoadedConformalMetric<DataType>, tmpl::size_t<3>,
                  Frame::Inertial> /*meta*/) const {
  if constexpr (std::is_same_v<DataType, DataVector>) {
    ASSERT(this->mesh.has_value() and this->inv_jacobian.has_value(),
           "Need a mesh and an inverse Jacobian for numeric differentiation.");
    const auto& loaded =
        cache->get_var(*this, LoadedConformalMetric<DataType>{});
    partial_derivative(deriv_loaded, loaded, this->mesh->get(),
                       this->inv_jacobian->get());
  } else {
    (void)deriv_loaded;
    (void)cache;
    ERROR(
        "Numeric differentiation requires a grid, so it only works with "
        "DataVectors.");
  }
}

template <typename DataType>
tnsr::ii<DataType, 3>
NumericBinaryWithWavesVariables<DataType>::wave_contribution(
    const double present_time, const gsl::not_null<Cache*> cache) const {
  // Body positions at this time, so the attenuation travels with the bodies.
  std::array<double, 3> body_1 = position_1;
  std::array<double, 3> body_2 = position_2;
  tnsr::ii<DataType, 3> wave{};
  if (inspiral.state.empty()) {
    // No past evolution: only the instantaneous term is available, and it
    // decays faster than 1/r, so there is effectively no wave-zone content.
    htt_instantaneous(make_not_null(&wave), x, body_1, body_2, mass_1, mass_2);
  } else {
    const auto configuration =
        configuration_at(inspiral, present_time, mass_1, mass_2);
    body_1 = configuration.positions[0];
    body_2 = configuration.positions[1];
    htt_total(make_not_null(&wave), x, inspiral, mass_1, mass_2, present_time);
  }
  Scalar<DataType> attenuation{};
  wave_attenuation(make_not_null(&attenuation), x, body_1, body_2,
                   attenuation_width);

  // Only the part of h^TT that is trace free with respect to the *loaded*
  // conformal metric is added. h^TT is transverse-traceless with respect to the
  // flat metric, not with respect to gamma-bar, so its gamma-bar trace is
  // non-zero at O(h M/r) and would break the unimodular gauge
  // det(gamma-bar) = 1 that fixes this class's conformal split. A trace piece
  // is in any case pure conformal rescaling, degenerate with psi, so removing
  // it changes no physics.
  //
  // The projection uses the loaded conformal metric at the *present* time even
  // when this is evaluated at an earlier one. That keeps the split
  // gamma-bar = loaded + wave exact by construction, and confines the loaded
  // data's own time dependence to its Lie drag, where it is handled exactly.
  const auto& loaded = cache->get_var(*this, LoadedConformalMetric<DataType>{});
  const auto& spatial_metric =
      cache->get_var(*this, gr::Tags::SpatialMetric<DataType, 3>{});
  const auto& inv_spatial_metric =
      cache->get_var(*this, gr::Tags::InverseSpatialMetric<DataType, 3>{});
  const auto det_spatial_metric = determinant_and_inverse(spatial_metric).first;
  const DataType inverse_conformal_scaling =
      pow(get(det_spatial_metric), 1. / 3.);
  auto wave_trace = make_with_value<DataType>(get<0>(x), 0.);
  for (size_t k = 0; k < 3; ++k) {
    for (size_t l = 0; l < 3; ++l) {
      wave_trace += inverse_conformal_scaling * inv_spatial_metric.get(k, l) *
                    wave.get(k, l);
    }
  }
  tnsr::ii<DataType, 3> result{};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      result.get(i, j) =
          get(attenuation) *
          (wave.get(i, j) - (1. / 3.) * wave_trace * loaded.get(i, j));
    }
  }
  return result;
}

template <typename DataType>
void NumericBinaryWithWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ijj<DataType, 3>*> deriv_conformal_metric,
    const gsl::not_null<Cache*> cache,
    ::Tags::deriv<Tags::ConformalMetric<DataType, 3, Frame::Inertial>,
                  tmpl::size_t<3>, Frame::Inertial> /*meta*/) const {
  if constexpr (std::is_same_v<DataType, DataVector>) {
    ASSERT(this->mesh.has_value() and this->inv_jacobian.has_value(),
           "Need a mesh and an inverse Jacobian for numeric differentiation.");
    const auto& conformal_metric = cache->get_var(
        *this, Tags::ConformalMetric<DataType, 3, Frame::Inertial>{});
    partial_derivative(deriv_conformal_metric, conformal_metric,
                       this->mesh->get(), this->inv_jacobian->get());
  } else {
    (void)deriv_conformal_metric;
    (void)cache;
    ERROR(
        "Numeric differentiation requires a grid, so it only works with "
        "DataVectors.");
  }
}

template <typename DataType>
void NumericBinaryWithWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::ii<DataType, 3>*> dt_conformal_metric,
    const gsl::not_null<Cache*> cache,
    ::Tags::dt<Tags::ConformalMetric<DataType, 3, Frame::Inertial>> /*meta*/)
    const {
  // The conformal metric splits into the loaded data plus the wave, and the two
  // have genuinely different time derivatives.
  //
  // The loaded part is stationary in the frame corotating with the binary, so
  // in the inertial frame it is Lie-dragged along the helical Killing vector:
  // dt = -Lie_xi. That is exact for it, and needs no differencing.
  //
  // The wave is NOT stationary in that frame. It rotates at the post-Newtonian
  // orbital frequency of the trajectory it was built from, which is a different
  // number from the frame's Omega (report 012), so dragging it along xi would
  // mis-state its time derivative by the ratio of the two -- tens of percent,
  // not a small correction (report 017). Its derivative is taken instead by
  // differencing the whole wave construction in the present time.
  const auto& loaded = cache->get_var(*this, LoadedConformalMetric<DataType>{});
  const auto& deriv_loaded =
      cache->get_var(*this, ::Tags::deriv<LoadedConformalMetric<DataType>,
                                          tmpl::size_t<3>, Frame::Inertial>{});
  const auto xi = killing_vector();
  const auto deriv_xi = deriv_killing_vector();
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      dt_conformal_metric->get(i, j) = make_with_value<DataType>(get<0>(x), 0.);
      for (size_t k = 0; k < 3; ++k) {
        dt_conformal_metric->get(i, j) -=
            xi.get(k) * deriv_loaded.get(k, i, j) +
            loaded.get(k, j) * deriv_xi.get(i, k) +
            loaded.get(i, k) * deriv_xi.get(j, k);
      }
    }
  }
  if (inspiral.state.empty()) {
    return;
  }
  // Second-order backward difference of the wave contribution. Backward
  // because the trajectory only exists for t <= 0.
  const auto wave_now = wave_contribution(0.0, cache);
  const auto wave_back = wave_contribution(-wave_time_derivative_step, cache);
  const auto wave_further =
      wave_contribution(-2.0 * wave_time_derivative_step, cache);
  const double inverse_step = 1.0 / (2.0 * wave_time_derivative_step);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      dt_conformal_metric->get(i, j) +=
          (3.0 * wave_now.get(i, j) - 4.0 * wave_back.get(i, j) +
           wave_further.get(i, j)) *
          inverse_step;
    }
  }
}

template <typename DataType>
void NumericBinaryWithWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> extrinsic_curvature_trace,
    const gsl::not_null<Cache*> cache,
    gr::Tags::TraceExtrinsicCurvature<DataType> /*meta*/) const {
  // Taken directly from the loaded extrinsic curvature, K = gamma^ij K_ij.
  // This involves no derivatives and is frame independent, so it is exactly
  // the trace of the previous solve.
  const auto& extrinsic_curvature =
      cache->get_var(*this, gr::Tags::ExtrinsicCurvature<DataType, 3>{});
  const auto& inv_spatial_metric =
      cache->get_var(*this, gr::Tags::InverseSpatialMetric<DataType, 3>{});
  get(*extrinsic_curvature_trace) = make_with_value<DataType>(get<0>(x), 0.);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      get(*extrinsic_curvature_trace) +=
          inv_spatial_metric.get(i, j) * extrinsic_curvature.get(i, j);
    }
  }
}

template <typename DataType>
void NumericBinaryWithWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> dt_extrinsic_curvature_trace,
    const gsl::not_null<Cache*> cache,
    ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataType>> /*meta*/) const {
  // K is a scalar, so the Lie drag reduces to dt(K) = -xi^k d_k K. The spatial
  // derivative is provided spectrally by `CommonVariables`.
  const auto& deriv_extrinsic_curvature_trace = cache->get_var(
      *this, ::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataType>,
                           tmpl::size_t<3>, Frame::Inertial>{});
  const auto xi = killing_vector();
  get(*dt_extrinsic_curvature_trace) = make_with_value<DataType>(get<0>(x), 0.);
  for (size_t k = 0; k < 3; ++k) {
    get(*dt_extrinsic_curvature_trace) -=
        xi.get(k) * deriv_extrinsic_curvature_trace.get(k);
  }
}

template <typename DataType>
void NumericBinaryWithWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*> shift_background,
    const gsl::not_null<Cache*> /*cache*/,
    Tags::ShiftBackground<DataType, 3, Frame::Inertial> /*meta*/) const {
  // Inertial frame: the whole shift is carried by the excess shift, which is
  // the asymptotically small `ShiftExcess` of the previous solve.
  std::fill(shift_background->begin(), shift_background->end(), 0.);
}

template <typename DataType>
void NumericBinaryWithWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::iJ<DataType, 3>*> deriv_shift_background,
    const gsl::not_null<Cache*> /*cache*/,
    ::Tags::deriv<Tags::ShiftBackground<DataType, 3, Frame::Inertial>,
                  tmpl::size_t<3>, Frame::Inertial> /*meta*/) const {
  std::fill(deriv_shift_background->begin(), deriv_shift_background->end(), 0.);
}

template <typename DataType>
void NumericBinaryWithWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::II<DataType, 3, Frame::Inertial>*>
        longitudinal_shift_background,
    const gsl::not_null<Cache*> cache,
    Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
        DataType, 3, Frame::Inertial> /*meta*/) const {
  // With a vanishing background shift this is just -ubar^ij, where ubar is the
  // trace-free part of dt(conformal metric) with indices raised. Because
  // Lie_xi(gbar)_ij = Dbar_i xi_j + Dbar_j xi_i, this equals +(Lbar xi)^ij,
  // which is exactly what the corotating previous solve used as
  // (Lbar beta_background)^ij with ubar = 0.
  const auto& conformal_metric = cache->get_var(
      *this, Tags::ConformalMetric<DataType, 3, Frame::Inertial>{});
  const auto& inv_conformal_metric = cache->get_var(
      *this, Tags::InverseConformalMetric<DataType, 3, Frame::Inertial>{});
  const auto& dt_conformal_metric = cache->get_var(
      *this, ::Tags::dt<Tags::ConformalMetric<DataType, 3, Frame::Inertial>>{});

  auto trace_dt_conformal_metric = make_with_value<DataType>(get<0>(x), 0.);
  for (size_t k = 0; k < 3; ++k) {
    for (size_t l = 0; l < 3; ++l) {
      trace_dt_conformal_metric +=
          inv_conformal_metric.get(k, l) * dt_conformal_metric.get(k, l);
    }
  }

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      longitudinal_shift_background->get(i, j) =
          make_with_value<DataType>(get<0>(x), 0.);
      for (size_t k = 0; k < 3; ++k) {
        for (size_t l = 0; l < 3; ++l) {
          longitudinal_shift_background->get(i, j) -=
              inv_conformal_metric.get(i, k) * inv_conformal_metric.get(j, l) *
              (dt_conformal_metric.get(k, l) - (1. / 3.) *
                                                   trace_dt_conformal_metric *
                                                   conformal_metric.get(k, l));
        }
      }
    }
  }
}

template <typename DataType>
void NumericBinaryWithWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> conformal_energy_density,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::Conformal<gr::Tags::EnergyDensity<DataType>, 0> /*meta*/) const {
  get(*conformal_energy_density) = make_with_value<DataType>(get<0>(x), 0.);
}

template <typename DataType>
void NumericBinaryWithWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> conformal_stress_trace,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::Conformal<gr::Tags::StressTrace<DataType>, 0> /*meta*/) const {
  get(*conformal_stress_trace) = make_with_value<DataType>(get<0>(x), 0.);
}

template <typename DataType>
void NumericBinaryWithWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*> conformal_momentum_density,
    const gsl::not_null<Cache*> /*cache*/,
    gr::Tags::Conformal<gr::Tags::MomentumDensity<DataType, 3>, 0> /*meta*/)
    const {
  std::fill(conformal_momentum_density->begin(),
            conformal_momentum_density->end(), 0.);
}

template <typename DataType>
void NumericBinaryWithWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*> conformal_factor_minus_one,
    const gsl::not_null<Cache*> cache,
    Tags::ConformalFactorMinusOne<DataType> /*meta*/) const {
  const auto& spatial_metric =
      cache->get_var(*this, gr::Tags::SpatialMetric<DataType, 3>{});
  const auto det_spatial_metric = determinant_and_inverse(spatial_metric).first;
  get(*conformal_factor_minus_one) =
      pow(get(det_spatial_metric), 1. / 12.) - 1.;
}

template <typename DataType>
void NumericBinaryWithWavesVariables<DataType>::operator()(
    const gsl::not_null<Scalar<DataType>*>
        lapse_times_conformal_factor_minus_one,
    const gsl::not_null<Cache*> cache,
    Tags::LapseTimesConformalFactorMinusOne<DataType> /*meta*/) const {
  const auto& lapse = cache->get_var(*this, gr::Tags::Lapse<DataType>{});
  const auto& conformal_factor_minus_one =
      cache->get_var(*this, Tags::ConformalFactorMinusOne<DataType>{});
  get(*lapse_times_conformal_factor_minus_one) =
      get(lapse) * (get(conformal_factor_minus_one) + 1.) - 1.;
}

template <typename DataType>
void NumericBinaryWithWavesVariables<DataType>::operator()(
    const gsl::not_null<tnsr::I<DataType, 3>*> shift_excess,
    const gsl::not_null<Cache*> /*cache*/,
    Tags::ShiftExcess<DataType, 3, Frame::Inertial> /*meta*/) const {
  // The background shift vanishes, so the excess shift is the whole shift. It
  // is loaded as `ShiftExcess` from the previous solve, which is asymptotically
  // small; the full `Shift` grows like r and must not be used here.
  *shift_excess =
      get<Tags::ShiftExcess<DataType, 3, Frame::Inertial>>(loaded_vars);
}

template class NumericBinaryWithWavesVariables<DataVector>;

}  // namespace detail

NumericBinaryWithWaves::NumericBinaryWithWaves(
    std::string data_file, std::string subgroup, const int observation_step,
    const bool extrapolate_into_excisions, const double angular_velocity,
    const double expansion, const double separation,
    const double target_eccentricity, const double mass_left,
    const double mass_right, const double attenuation_width,
    const double past_evolution_duration, const double past_evolution_time_step)
    : data_file_(std::move(data_file)),
      subgroup_(std::move(subgroup)),
      observation_step_(observation_step),
      extrapolate_into_excisions_(extrapolate_into_excisions),
      angular_velocity_(angular_velocity),
      expansion_(expansion),
      separation_(separation),
      target_eccentricity_(target_eccentricity),
      attenuation_width_(attenuation_width),
      mass_left_(mass_left),
      mass_right_(mass_right),
      past_evolution_duration_(past_evolution_duration),
      past_evolution_time_step_(past_evolution_time_step) {
  load_interpolator();
  evolve_past();
}

NumericBinaryWithWaves::NumericBinaryWithWaves(
    const NumericBinaryWithWaves& rhs)
    : NumericBinaryWithWaves(
          rhs.data_file_, rhs.subgroup_, rhs.observation_step_,
          rhs.extrapolate_into_excisions_, rhs.angular_velocity_,
          rhs.expansion_, rhs.separation_, rhs.target_eccentricity_,
          rhs.mass_left_, rhs.mass_right_, rhs.attenuation_width_,
          rhs.past_evolution_duration_, rhs.past_evolution_time_step_) {}

NumericBinaryWithWaves& NumericBinaryWithWaves::operator=(
    const NumericBinaryWithWaves& rhs) {
  if (this != &rhs) {
    *this = NumericBinaryWithWaves(rhs);
  }
  return *this;
}

void NumericBinaryWithWaves::load_interpolator() {
  // Reads the volume data into memory once. Constructing the interpolator opens
  // the H5 files and is not thread safe unless HDF5 was built with
  // thread-safety support (it is not, on the machines this runs on), so this
  // must not be called concurrently. It is safe here because the background is
  // a `const_global_cache_tag` and `Parallel::GlobalCache` is a Nodegroup, so
  // exactly one instance exists per node and it is unpacked on a single thread.
  // Interpolating from the loaded data afterwards *is* thread safe, so all
  // elements on the node share this one copy.
  if (data_file_.empty()) {
    return;
  }
  interpolator_ = spectre::Exporter::PointwiseInterpolator<3, Frame::Inertial>{
      data_file_, subgroup_,
      spectre::Exporter::ObservationStep{observation_step_},
      spectre::Exporter::get_tensor_components<
          detail::numeric_load_tags<DataVector>>()};
}

void NumericBinaryWithWaves::evolve_past() {
  // Present positions of the two bodies, in the centre-of-mass frame and along
  // the x axis, matching the convention of `evolve_binary_backwards`: body 1
  // sits at positive x and carries `mass_right_`. This is the t = 0 state of
  // the trajectory below, written directly so that the wave content is
  // available even when the past evolution is switched off.
  const double total_mass_for_positions = mass_left_ + mass_right_;
  present_position_1_ = {
      {mass_left_ / total_mass_for_positions * separation_, 0.0, 0.0}};
  present_position_2_ = {
      {-mass_right_ / total_mass_for_positions * separation_, 0.0, 0.0}};

  if (not(past_evolution_duration_ > 0.0)) {
    return;
  }
  if (target_eccentricity_ != 0.0) {
    ERROR("Only a target eccentricity of zero is supported so far, got "
          << target_eccentricity_
          << ". Constructing an eccentric past evolution also requires "
             "choosing the orbital phase at the present time, which is not "
             "implemented.");
  }
  const double total_mass = mass_left_ + mass_right_;
  const double eta = mass_left_ * mass_right_ / (total_mass * total_mass);
  // The post-Newtonian sector works in units M = 1. The loaded data is already
  // in units where the total mass is close to one, but not exactly, so
  // everything is rescaled rather than assumed.
  const double separation = separation_ / total_mass;
  // The orbit is derived from the separation and the target eccentricity, NOT
  // from `angular_velocity_` and `expansion_`. Those two describe the helical
  // Killing vector of the loaded data and are used for dt(gamma-bar); they are
  // not required to be a consistent orbital state, and driving the past
  // evolution with an inconsistent pair sends it into the strong field.
  const auto orbit = detail::quasi_circular_state(separation, eta);
  pn_inspiral_ =
      detail::evolve_binary_backwards(separation, orbit[0], orbit[1], eta,
                                      past_evolution_duration_ / total_mass,
                                      past_evolution_time_step_ / total_mass);
}

void NumericBinaryWithWaves::pup(PUP::er& p) {
  elliptic::analytic_data::Background::pup(p);
  elliptic::analytic_data::InitialGuess::pup(p);
  p | data_file_;
  p | subgroup_;
  p | observation_step_;
  p | extrapolate_into_excisions_;
  p | angular_velocity_;
  p | expansion_;
  p | separation_;
  p | target_eccentricity_;
  p | attenuation_width_;
  p | mass_left_;
  p | mass_right_;
  p | past_evolution_duration_;
  p | past_evolution_time_step_;
  // Neither the loaded volume data nor the past evolution is serialized. The
  // first is reloaded so that every node reads the file once rather than
  // shipping it around; the second is cheap enough to recompute.
  if (p.isUnpacking()) {
    load_interpolator();
    evolve_past();
  }
}

bool operator==(const NumericBinaryWithWaves& lhs,
                const NumericBinaryWithWaves& rhs) {
  // The past evolution is a deterministic function of these options, so it
  // does not need comparing.
  return lhs.data_file() == rhs.data_file() and
         lhs.subgroup() == rhs.subgroup() and
         lhs.observation_step() == rhs.observation_step() and
         lhs.extrapolate_into_excisions() ==
             rhs.extrapolate_into_excisions() and
         lhs.angular_velocity() == rhs.angular_velocity() and
         lhs.expansion() == rhs.expansion() and
         lhs.separation() == rhs.separation() and
         lhs.target_eccentricity() == rhs.target_eccentricity() and
         lhs.attenuation_width() == rhs.attenuation_width() and
         lhs.mass_left() == rhs.mass_left() and
         lhs.mass_right() == rhs.mass_right() and
         lhs.past_evolution_duration() == rhs.past_evolution_duration() and
         lhs.past_evolution_time_step() == rhs.past_evolution_time_step();
}

bool operator!=(const NumericBinaryWithWaves& lhs,
                const NumericBinaryWithWaves& rhs) {
  return not(lhs == rhs);
}

PUP::able::PUP_ID NumericBinaryWithWaves::my_PUP_ID = 0;  // NOLINT

}  // namespace Xcts::AnalyticData

template class Xcts::AnalyticData::CommonVariables<
    DataVector, typename Xcts::AnalyticData::detail::
                    NumericBinaryWithWavesVariables<DataVector>::Cache>;
