// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/Characteristics.hpp"

#include <cmath>
#include <iterator>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"

namespace NewtonianEuler {
namespace detail {
template <>
Matrix flux_jacobian<1>(const tnsr::I<double, 1>& velocity,
                        const double kappa_over_density,
                        const double b_times_theta,
                        const double specific_enthalpy,
                        const tnsr::i<double, 1>& unit_normal) {
  const double n_x = get<0>(unit_normal);
  const double u = get<0>(velocity);
  const Matrix a_x = blaze::DynamicMatrix<double>{
      {0., 1., 0.},
      {b_times_theta - square(u), u * (2. - kappa_over_density),
       kappa_over_density},
      {u * (b_times_theta - specific_enthalpy),
       specific_enthalpy - kappa_over_density * square(u),
       u * (kappa_over_density + 1.)}};
  return n_x * a_x;
}

template <>
Matrix flux_jacobian<2>(const tnsr::I<double, 2>& velocity,
                        const double kappa_over_density,
                        const double b_times_theta,
                        const double specific_enthalpy,
                        const tnsr::i<double, 2>& unit_normal) {
  const double n_x = get<0>(unit_normal);
  const double n_y = get<1>(unit_normal);
  const double u = get<0>(velocity);
  const double v = get<1>(velocity);
  const Matrix a_x = blaze::DynamicMatrix<double>{
      {0., 1., 0., 0.},
      {b_times_theta - square(u), u * (2. - kappa_over_density),
       -v * kappa_over_density, kappa_over_density},
      {-u * v, v, u, 0.},
      {u * (b_times_theta - specific_enthalpy),
       specific_enthalpy - kappa_over_density * square(u),
       -u * v * kappa_over_density, u * (kappa_over_density + 1.)}};
  const Matrix a_y = blaze::DynamicMatrix<double>{
      {0., 0., 1., 0.},
      {-u * v, v, u, 0.},
      {b_times_theta - square(v), -u * kappa_over_density,
       v * (2. - kappa_over_density), kappa_over_density},
      {v * (b_times_theta - specific_enthalpy), -u * v * kappa_over_density,
       specific_enthalpy - kappa_over_density * square(v),
       v * (kappa_over_density + 1.)}};
  return n_x * a_x + n_y * a_y;
}

template <>
Matrix flux_jacobian<3>(const tnsr::I<double, 3>& velocity,
                        const double kappa_over_density,
                        const double b_times_theta,
                        const double specific_enthalpy,
                        const tnsr::i<double, 3>& unit_normal) {
  const double n_x = get<0>(unit_normal);
  const double n_y = get<1>(unit_normal);
  const double n_z = get<2>(unit_normal);
  const double u = get<0>(velocity);
  const double v = get<1>(velocity);
  const double w = get<2>(velocity);
  const Matrix a_x = blaze::DynamicMatrix<double>{
      {0., 1., 0., 0., 0.},
      {b_times_theta - square(u), u * (2. - kappa_over_density),
       -v * kappa_over_density, -w * kappa_over_density, kappa_over_density},
      {-u * v, v, u, 0., 0.},
      {-u * w, w, 0., u, 0.},
      {u * (b_times_theta - specific_enthalpy),
       specific_enthalpy - kappa_over_density * square(u),
       -u * v * kappa_over_density, -u * w * kappa_over_density,
       u * (kappa_over_density + 1.)}};
  const Matrix a_y = blaze::DynamicMatrix<double>{
      {0., 0., 1., 0., 0.},
      {-u * v, v, u, 0., 0.},
      {b_times_theta - square(v), -u * kappa_over_density,
       v * (2. - kappa_over_density), -w * kappa_over_density,
       kappa_over_density},
      {-v * w, 0., w, v, 0.},
      {v * (b_times_theta - specific_enthalpy), -u * v * kappa_over_density,
       specific_enthalpy - kappa_over_density * square(v),
       -v * w * kappa_over_density, v * (kappa_over_density + 1.)}};
  const Matrix a_z = blaze::DynamicMatrix<double>{
      {0., 0., 0., 1., 0.},
      {-u * w, w, 0., u, 0.},
      {-v * w, 0., w, v, 0.},
      {b_times_theta - square(w), -u * kappa_over_density,
       -v * kappa_over_density, w * (2. - kappa_over_density),
       kappa_over_density},
      {w * (b_times_theta - specific_enthalpy), -u * w * kappa_over_density,
       -v * w * kappa_over_density,
       specific_enthalpy - kappa_over_density * square(w),
       w * (kappa_over_density + 1.)}};
  return n_x * a_x + n_y * a_y + n_z * a_z;
}
}  // namespace detail

template <size_t Dim>
void characteristic_speeds(
    const gsl::not_null<std::array<DataVector, Dim + 2>*> char_speeds,
    const tnsr::I<DataVector, Dim>& velocity,
    const Scalar<DataVector>& sound_speed,
    const tnsr::i<DataVector, Dim>& normal) {
  auto& characteristic_speeds = *char_speeds;
  characteristic_speeds =
      make_array<Dim + 2>(DataVector(get(dot_product(velocity, normal))));

  characteristic_speeds[0] -= get(sound_speed);
  characteristic_speeds[Dim + 1] += get(sound_speed);
}

template <size_t Dim>
std::array<DataVector, Dim + 2> characteristic_speeds(
    const tnsr::I<DataVector, Dim>& velocity,
    const Scalar<DataVector>& sound_speed,
    const tnsr::i<DataVector, Dim>& normal) {
  std::array<DataVector, Dim + 2> char_speeds{};
  characteristic_speeds(make_not_null(&char_speeds), velocity, sound_speed,
                        normal);
  return char_speeds;
}

template <>
Matrix right_eigenvectors<1>(const tnsr::I<double, 1>& velocity,
                             const Scalar<double>& sound_speed_squared,
                             const Scalar<double>& specific_enthalpy,
                             const Scalar<double>& kappa_over_density,
                             const tnsr::i<double, 1>& unit_normal) {
  ASSERT(equal_within_roundoff(get(magnitude(unit_normal)), 1.),
         "Expected unit normal, but got normal with magnitude "
             << get(magnitude(unit_normal)));

  const double u = get<0>(velocity);
  const double n_x = get<0>(unit_normal);
  const double c = sqrt(get(sound_speed_squared));
  const double v_n = get(dot_product(velocity, unit_normal));

  Matrix result(3, 3);
  result(0, 0) = 1.;
  result(0, 1) = get(kappa_over_density);
  result(0, 2) = 1.;
  result(1, 0) = u - n_x * c;
  result(1, 1) = get(kappa_over_density) * u;
  result(1, 2) = u + n_x * c;
  result(2, 0) = get(specific_enthalpy) - c * v_n;
  result(2, 1) = get(kappa_over_density) * get(specific_enthalpy) -
                 get(sound_speed_squared);
  result(2, 2) = get(specific_enthalpy) + c * v_n;
  return result;
}

template <>
Matrix right_eigenvectors<2>(const tnsr::I<double, 2>& velocity,
                             const Scalar<double>& sound_speed_squared,
                             const Scalar<double>& specific_enthalpy,
                             const Scalar<double>& kappa_over_density,
                             const tnsr::i<double, 2>& unit_normal) {
  ASSERT(equal_within_roundoff(get(magnitude(unit_normal)), 1.),
         "Expected unit normal, but got normal with magnitude "
             << get(magnitude(unit_normal)));

  const double u = get<0>(velocity);
  const double v = get<1>(velocity);
  const double n_x = get<0>(unit_normal);
  const double n_y = get<1>(unit_normal);
  const double c = sqrt(get(sound_speed_squared));
  const double v_n = get(dot_product(velocity, unit_normal));

  Matrix result(4, 4);
  result(0, 0) = 1.;
  result(0, 1) = get(kappa_over_density);
  result(0, 2) = 0.;
  result(0, 3) = 1.;
  result(1, 0) = u - n_x * c;
  result(1, 1) = get(kappa_over_density) * u;
  result(1, 2) = -n_y;
  result(1, 3) = u + n_x * c;
  result(2, 0) = v - n_y * c;
  result(2, 1) = get(kappa_over_density) * v;
  result(2, 2) = n_x;
  result(2, 3) = v + n_y * c;
  result(3, 0) = get(specific_enthalpy) - c * v_n;
  result(3, 1) = get(kappa_over_density) * get(specific_enthalpy) -
                 get(sound_speed_squared);
  result(3, 2) = -n_y * u + n_x * v;
  result(3, 3) = get(specific_enthalpy) + c * v_n;
  return result;
}

template <>
Matrix right_eigenvectors<3>(const tnsr::I<double, 3>& velocity,
                             const Scalar<double>& sound_speed_squared,
                             const Scalar<double>& specific_enthalpy,
                             const Scalar<double>& kappa_over_density,
                             const tnsr::i<double, 3>& unit_normal) {
  ASSERT(equal_within_roundoff(get(magnitude(unit_normal)), 1.),
         "Expected unit normal, but got normal with magnitude "
             << get(magnitude(unit_normal)));

  const double u = get<0>(velocity);
  const double v = get<1>(velocity);
  const double w = get<2>(velocity);
  const double n_x = get<0>(unit_normal);
  const double n_y = get<1>(unit_normal);
  const double n_z = get<2>(unit_normal);
  const double c = sqrt(get(sound_speed_squared));
  const double v_n = get(dot_product(velocity, unit_normal));

  Matrix result(5, 5);
  result(0, 0) = 1.;
  result(1, 0) = u - n_x * c;
  result(2, 0) = v - n_y * c;
  result(3, 0) = w - n_z * c;
  result(4, 0) = get(specific_enthalpy) - v_n * c;
  result(0, 1) = get(kappa_over_density);
  result(1, 1) = get(kappa_over_density) * u;
  result(2, 1) = get(kappa_over_density) * v;
  result(3, 1) = get(kappa_over_density) * w;
  result(4, 1) = get(kappa_over_density) * get(specific_enthalpy) -
                 get(sound_speed_squared);
  result(0, 4) = 1.;
  result(1, 4) = u + n_x * c;
  result(2, 4) = v + n_y * c;
  result(3, 4) = w + n_z * c;
  result(4, 4) = get(specific_enthalpy) + v_n * c;

  // There is some degeneracy in the choice of the column eigenvectors. The row
  // eigenvectors are chosen so that the largest of (n_x, n_y, n_z) appears in
  // the denominator, because this avoids division by zero. Here we make the
  // consistent choice of right eigenvectors.
  const auto index_of_largest = std::distance(
      unit_normal.begin(),
      alg::max_element(unit_normal, [](const double n1, const double n2) {
        return fabs(n1) < fabs(n2);
      }));
  if (index_of_largest == 0) {
    // right eigenvectors corresponding to left eigenvectors with 1/n_x terms
    result(0, 2) = 0.;
    result(1, 2) = -n_y;
    result(2, 2) = n_x;
    result(3, 2) = 0.;
    result(4, 2) = -n_y * u + n_x * v;
    result(0, 3) = 0.;
    result(1, 3) = -n_z;
    result(2, 3) = 0.;
    result(3, 3) = n_x;
    result(4, 3) = -n_z * u + n_x * w;
  } else if (index_of_largest == 1) {
    // right eigenvectors corresponding to left eigenvectors with 1/n_y terms
    result(0, 2) = 0.;
    result(1, 2) = -n_y;
    result(2, 2) = n_x;
    result(3, 2) = 0.;
    result(4, 2) = -n_y * u + n_x * v;
    result(0, 3) = 0.;
    result(1, 3) = 0.;
    result(2, 3) = -n_z;
    result(3, 3) = n_y;
    result(4, 3) = -n_z * v + n_y * w;
  } else {
    // right eigenvectors corresponding to left eigenvectors with 1/n_z terms
    result(0, 2) = 0.;
    result(1, 2) = -n_z;
    result(2, 2) = 0.;
    result(3, 2) = n_x;
    result(4, 2) = -n_z * u + n_x * w;
    result(0, 3) = 0.;
    result(1, 3) = 0.;
    result(2, 3) = -n_z;
    result(3, 3) = n_y;
    result(4, 3) = -n_z * v + n_y * w;
  }
  return result;
}

template <>
Matrix left_eigenvectors<1>(const tnsr::I<double, 1>& velocity,
                            const Scalar<double>& sound_speed_squared,
                            const Scalar<double>& specific_enthalpy,
                            const Scalar<double>& kappa_over_density,
                            const tnsr::i<double, 1>& unit_normal) {
  ASSERT(equal_within_roundoff(get(magnitude(unit_normal)), 1.),
         "Expected unit normal, but got normal with magnitude "
             << get(magnitude(unit_normal)));

  const double velocity_squared = get(dot_product(velocity, velocity));
  const double u = get<0>(velocity);
  const double n_x = get<0>(unit_normal);
  const double c = sqrt(get(sound_speed_squared));
  const double v_n = get(dot_product(velocity, unit_normal));

  // Temporary with a useful combination, as per Kulikovskii Ch3
  const double b_times_theta =
      get(kappa_over_density) * (velocity_squared - get(specific_enthalpy)) +
      get(sound_speed_squared);

  Matrix result(3, 3);
  result(0, 0) = 0.5 * (b_times_theta + c * v_n) / get(sound_speed_squared);
  result(0, 1) =
      -0.5 * (get(kappa_over_density) * u + n_x * c) / get(sound_speed_squared);
  result(0, 2) = 0.5 * get(kappa_over_density) / get(sound_speed_squared);
  result(1, 0) =
      (get(specific_enthalpy) - velocity_squared) / get(sound_speed_squared);
  result(1, 1) = u / get(sound_speed_squared);
  result(1, 2) = -1. / get(sound_speed_squared);
  result(2, 0) = 0.5 * (b_times_theta - c * v_n) / get(sound_speed_squared);
  result(2, 1) =
      -0.5 * (get(kappa_over_density) * u - n_x * c) / get(sound_speed_squared);
  result(2, 2) = 0.5 * get(kappa_over_density) / get(sound_speed_squared);
  return result;
}

template <>
Matrix left_eigenvectors<2>(const tnsr::I<double, 2>& velocity,
                            const Scalar<double>& sound_speed_squared,
                            const Scalar<double>& specific_enthalpy,
                            const Scalar<double>& kappa_over_density,
                            const tnsr::i<double, 2>& unit_normal) {
  ASSERT(equal_within_roundoff(get(magnitude(unit_normal)), 1.),
         "Expected unit normal, but got normal with magnitude "
             << get(magnitude(unit_normal)));

  const double velocity_squared = get(dot_product(velocity, velocity));
  const double u = get<0>(velocity);
  const double v = get<1>(velocity);
  const double n_x = get<0>(unit_normal);
  const double n_y = get<1>(unit_normal);
  const double c = sqrt(get(sound_speed_squared));
  const double v_n = get(dot_product(velocity, unit_normal));

  // Temporary with a useful combination, as per Kulikovskii Ch3
  const double b_times_theta =
      get(kappa_over_density) * (velocity_squared - get(specific_enthalpy)) +
      get(sound_speed_squared);

  Matrix result(4, 4);
  result(0, 0) = 0.5 * (b_times_theta + c * v_n) / get(sound_speed_squared);
  result(0, 1) =
      -0.5 * (get(kappa_over_density) * u + n_x * c) / get(sound_speed_squared);
  result(0, 2) =
      -0.5 * (get(kappa_over_density) * v + n_y * c) / get(sound_speed_squared);
  result(0, 3) = 0.5 * get(kappa_over_density) / get(sound_speed_squared);
  result(1, 0) =
      (get(specific_enthalpy) - velocity_squared) / get(sound_speed_squared);
  result(1, 1) = u / get(sound_speed_squared);
  result(1, 2) = v / get(sound_speed_squared);
  result(1, 3) = -1. / get(sound_speed_squared);
  result(2, 0) = n_y * u - n_x * v;
  result(2, 1) = -n_y;
  result(2, 2) = n_x;
  result(2, 3) = 0.;
  result(3, 0) = 0.5 * (b_times_theta - c * v_n) / get(sound_speed_squared);
  result(3, 1) =
      -0.5 * (get(kappa_over_density) * u - n_x * c) / get(sound_speed_squared);
  result(3, 2) =
      -0.5 * (get(kappa_over_density) * v - n_y * c) / get(sound_speed_squared);
  result(3, 3) = 0.5 * get(kappa_over_density) / get(sound_speed_squared);
  return result;
}

template <>
Matrix left_eigenvectors<3>(const tnsr::I<double, 3>& velocity,
                            const Scalar<double>& sound_speed_squared,
                            const Scalar<double>& specific_enthalpy,
                            const Scalar<double>& kappa_over_density,
                            const tnsr::i<double, 3>& unit_normal) {
  ASSERT(equal_within_roundoff(get(magnitude(unit_normal)), 1.),
         "Expected unit normal, but got normal with magnitude "
             << get(magnitude(unit_normal)));

  const double velocity_squared = get(dot_product(velocity, velocity));
  const double u = get<0>(velocity);
  const double v = get<1>(velocity);
  const double w = get<2>(velocity);
  const double n_x = get<0>(unit_normal);
  const double n_y = get<1>(unit_normal);
  const double n_z = get<2>(unit_normal);
  const double c = sqrt(get(sound_speed_squared));
  const double v_n = get(dot_product(velocity, unit_normal));

  // Temporary with a useful combination, as per Kulikovskii Ch3
  const double b_times_theta =
      get(kappa_over_density) * (velocity_squared - get(specific_enthalpy)) +
      get(sound_speed_squared);

  Matrix result(5, 5);
  result(0, 0) = 0.5 * (b_times_theta + c * v_n) / get(sound_speed_squared);
  result(0, 1) =
      -0.5 * (get(kappa_over_density) * u + n_x * c) / get(sound_speed_squared);
  result(0, 2) =
      -0.5 * (get(kappa_over_density) * v + n_y * c) / get(sound_speed_squared);
  result(0, 3) =
      -0.5 * (get(kappa_over_density) * w + n_z * c) / get(sound_speed_squared);
  result(0, 4) = 0.5 * get(kappa_over_density) / get(sound_speed_squared);
  result(1, 0) =
      (get(specific_enthalpy) - velocity_squared) / get(sound_speed_squared);
  result(1, 1) = u / get(sound_speed_squared);
  result(1, 2) = v / get(sound_speed_squared);
  result(1, 3) = w / get(sound_speed_squared);
  result(1, 4) = -1. / get(sound_speed_squared);
  result(4, 0) = 0.5 * (b_times_theta - c * v_n) / get(sound_speed_squared);
  result(4, 1) =
      -0.5 * (get(kappa_over_density) * u - n_x * c) / get(sound_speed_squared);
  result(4, 2) =
      -0.5 * (get(kappa_over_density) * v - n_y * c) / get(sound_speed_squared);
  result(4, 3) =
      -0.5 * (get(kappa_over_density) * w - n_z * c) / get(sound_speed_squared);
  result(4, 4) = 0.5 * get(kappa_over_density) / get(sound_speed_squared);

  // There is some degeneracy in the choice of the row eigenvectors. Here, we
  // use rows where the largest of (n_x, n_y, n_z) appears in the denominator,
  // because this avoids division by zero. A consistent choice of right
  // eigenvectors must be made.
  const auto index_of_largest = std::distance(
      unit_normal.begin(),
      alg::max_element(unit_normal, [](const double n1, const double n2) {
        return fabs(n1) < fabs(n2);
      }));
  if (index_of_largest == 0) {
    // left eigenvectors with 1/n_x terms
    result(2, 0) = (n_y * v_n - v) / n_x;
    result(2, 1) = -n_y;
    result(2, 2) = (1. - square(n_y)) / n_x;
    result(2, 3) = -n_y * n_z / n_x;
    result(2, 4) = 0.;
    result(3, 0) = (n_z * v_n - w) / n_x;
    result(3, 1) = -n_z;
    result(3, 2) = -n_y * n_z / n_x;
    result(3, 3) = n_x + square(n_y) / n_x;
    result(3, 4) = 0.;
  } else if (index_of_largest == 1) {
    // left eigenvectors with 1/n_y terms
    result(2, 0) = (u - n_x * v_n) / n_y;
    result(2, 1) = (-1. + square(n_x)) / n_y;
    result(2, 2) = n_x;
    result(2, 3) = n_x * n_z / n_y;
    result(2, 4) = 0.;
    result(3, 0) = (n_z * v_n - w) / n_y;
    result(3, 1) = -n_x * n_z / n_y;
    result(3, 2) = -n_z;
    result(3, 3) = n_y + square(n_x) / n_y;
    result(3, 4) = 0.;
  } else {
    // left eigenvectors with 1/n_z terms
    result(2, 0) = (u - n_x * v_n) / n_z;
    result(2, 1) = (-1. + square(n_x)) / n_z;
    result(2, 2) = n_x * n_y / n_z;
    result(2, 3) = n_x;
    result(2, 4) = 0.;
    result(3, 0) = (v - n_y * v_n) / n_z;
    result(3, 1) = n_x * n_y / n_z;
    result(3, 2) = (-1. + square(n_y)) / n_z;
    result(3, 3) = n_y;
    result(3, 4) = 0.;
  }
  return result;
}

template <size_t Dim>
std::pair<DataVector, std::pair<Matrix, Matrix>> numerical_eigensystem(
    const tnsr::I<double, Dim>& velocity,
    const Scalar<double>& sound_speed_squared,
    const Scalar<double>& specific_enthalpy,
    const Scalar<double>& kappa_over_density,
    const tnsr::i<double, Dim>& unit_normal) {
  ASSERT(equal_within_roundoff(get(magnitude(unit_normal)), 1.),
         "Expected unit normal, but got normal with magnitude "
             << get(magnitude(unit_normal)));

  const double b_times_theta =
      get(kappa_over_density) *
          (get(dot_product(velocity, velocity)) - get(specific_enthalpy)) +
      get(sound_speed_squared);

  const Matrix a = detail::flux_jacobian<Dim>(
      velocity, get(kappa_over_density), b_times_theta, get(specific_enthalpy),
      unit_normal);

  const double vn = get(dot_product(velocity, unit_normal));
  const double cs = sqrt(get(sound_speed_squared));
  DataVector eigenvalues(Dim + 2, vn);
  eigenvalues[0] -= cs;
  eigenvalues[Dim + 1] += cs;

  Matrix right(Dim + 2, Dim + 2);

  // We'd like to use `blaze::eigen` to get the eigenvalues and eigenvectors
  // of the flux Jacobian matrix `a`... but because `a` is not symmetric,
  // blaze generically produces complex eigenvectors. So instead we find the
  // nullspace of `a - \lambda I` using `blaze::svd`.
  blaze::DynamicMatrix<double, blaze::rowMajor> a_minus_lambda;
  blaze::DynamicMatrix<double, blaze::rowMajor> U;      // left singular vectors
  blaze::DynamicVector<double, blaze::columnVector> s;  // singular values
  blaze::DynamicMatrix<double, blaze::rowMajor> V;  // right singular vectors

  const auto find_group_of_eigenvectors =
      [&a_minus_lambda, &a, &U, &s, &V, &eigenvalues, &right](
          const size_t index, const size_t degeneracy) {
        a_minus_lambda = a;
        for (size_t i = 0; i < Dim + 2; ++i) {
          a_minus_lambda(i, i) -= eigenvalues[index];
        }
        blaze::svd(a_minus_lambda, U, s, V);

    // Check the null space has the expected size: the last degeneracy
    // singular values should vanish
#ifdef SPECTRE_DEBUG
        for (size_t i = 0; i < Dim + 2 - degeneracy; ++i) {
          ASSERT(fabs(s[i]) > 1e-14, "Bad SVD");
        }
        for (size_t i = Dim + 2 - degeneracy; i < Dim + 2; ++i) {
          ASSERT(fabs(s[i]) < 1e-14, "Bad SVD");
        }
#endif  // ifdef SPECTRE_DEBUG

        // Copy the last degeneracy rows of V into the
        // (index, index+degeneracy) columns of right
        for (size_t i = 0; i < Dim + 2; ++i) {
          for (size_t j = 0; j < degeneracy; ++j) {
            right(i, index + j) = V(Dim + 2 - degeneracy + j, i);
          }
        }
      };

  // lambda = vn - cs
  find_group_of_eigenvectors(0, 1);
  // Dim-degenerate eigenvalues, lambda = vn
  find_group_of_eigenvectors(1, Dim);
  // lambda = vn + cs
  find_group_of_eigenvectors(Dim + 1, 1);

  Matrix left = right;
  blaze::invert<blaze::asGeneral>(left);

  return std::make_pair(eigenvalues, std::make_pair(right, left));
}

}  // namespace NewtonianEuler

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template void NewtonianEuler::characteristic_speeds(                         \
      const gsl::not_null<std::array<DataVector, DIM(data) + 2>*> char_speeds, \
      const tnsr::I<DataVector, DIM(data)>& velocity,                          \
      const Scalar<DataVector>& sound_speed,                                   \
      const tnsr::i<DataVector, DIM(data)>& normal);                           \
  template std::array<DataVector, DIM(data) + 2>                               \
  NewtonianEuler::characteristic_speeds(                                       \
      const tnsr::I<DataVector, DIM(data)>& velocity,                          \
      const Scalar<DataVector>& sound_speed,                                   \
      const tnsr::i<DataVector, DIM(data)>& normal);                           \
  template struct NewtonianEuler::Tags::ComputeLargestCharacteristicSpeed<DIM( \
      data)>;                                                                  \
  template std::pair<DataVector, std::pair<Matrix, Matrix>>                    \
  NewtonianEuler::numerical_eigensystem(                                       \
      const tnsr::I<double, DIM(data)>&, const Scalar<double>&,                \
      const Scalar<double>&, const Scalar<double>&,                            \
      const tnsr::i<double, DIM(data)>&);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
