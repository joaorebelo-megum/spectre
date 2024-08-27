// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/Surfaces/Spin.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/EagerMath/Trace.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/LinearAlgebra/FindGeneralizedEigenvalues.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/StrahlkorperFunctions.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"
#include "Utilities/StdArrayHelpers.hpp"

// Functions used by gr::surfaces::dimensionful_spin_magnitude
namespace {
// Find the 2D surface metric by inserting the tangents \f$\partial_\theta\f$
// and \f$\partial_\phi\f$ into the slots of the 3D spatial metric
template <typename Fr>
tnsr::ii<DataVector, 2, Frame::Spherical<Fr>> get_surface_metric(
    const tnsr::ii<DataVector, 3, Fr>& spatial_metric,
    const ylm::Tags::aliases::Jacobian<Fr>& tangents,
    const Scalar<DataVector>& sin_theta) {
  auto surface_metric =
      make_with_value<tnsr::ii<DataVector, 2, Frame::Spherical<Fr>>>(
          get<0, 0>(spatial_metric), 0.0);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      get<0, 1>(surface_metric) += spatial_metric.get(i, j) *
                                   tangents.get(i, 0) * tangents.get(j, 1) *
                                   get(sin_theta);
    }
    // Use symmetry to sum over fewer terms for the 0,0 and 1,1 components
    get<0, 0>(surface_metric) +=
        spatial_metric.get(i, i) * square(tangents.get(i, 0));
    get<1, 1>(surface_metric) += spatial_metric.get(i, i) *
                                 square(tangents.get(i, 1)) *
                                 square(get(sin_theta));
    for (size_t j = i + 1; j < 3; ++j) {
      get<0, 0>(surface_metric) += 2.0 * spatial_metric.get(i, j) *
                                   tangents.get(i, 0) * tangents.get(j, 0);
      get<1, 1>(surface_metric) += 2.0 * spatial_metric.get(i, j) *
                                   tangents.get(i, 1) * tangents.get(j, 1) *
                                   square(get(sin_theta));
    }
  }
  return surface_metric;
}

// Compute the trace of Christoffel 2nd kind on the horizon
template <typename Fr>
tnsr::I<DataVector, 2, Frame::Spherical<Fr>> get_trace_christoffel_second_kind(
    const tnsr::ii<DataVector, 2, Frame::Spherical<Fr>>& surface_metric,
    const tnsr::II<DataVector, 2, Frame::Spherical<Fr>>& inverse_surface_metric,
    const Scalar<DataVector>& sin_theta, const ylm::Spherepack& ylm) {
  const Scalar<DataVector> cos_theta{cos(ylm.theta_phi_points()[0])};

  // Because the surface metric components are not representable in terms
  // of scalar spherical harmonics, you can't naively take first derivatives.
  // To avoid potentially large numerical errors, actually differentiate
  // square(sin_theta) * the metric component, then
  // compute from that the gradient of just the metric component itself.
  //
  // Note: the method implemented here works with ylm::Spherepack but will
  // fail for other expansions that, unlike Spherepack, include a collocation
  // point at theta = 0. Before switching to such an expansion, first
  // reimplement this code to avoid dividing by sin(theta).
  //
  // Note: ylm::Spherepack gradients are flat-space Pfaffian derivatives.
  auto grad_surface_metric_theta_theta =
      ylm.gradient(square(get(sin_theta)) * get<0, 0>(surface_metric));
  get<0>(grad_surface_metric_theta_theta) /= square(get(sin_theta));
  get<1>(grad_surface_metric_theta_theta) /= square(get(sin_theta));
  get<0>(grad_surface_metric_theta_theta) -=
      2.0 * get<0, 0>(surface_metric) * get(cos_theta) / get(sin_theta);

  auto grad_surface_metric_theta_phi =
      ylm.gradient(get(sin_theta) * get<0, 1>(surface_metric));

  get<0>(grad_surface_metric_theta_phi) /= get(sin_theta);
  get<1>(grad_surface_metric_theta_phi) /= get(sin_theta);
  get<0>(grad_surface_metric_theta_phi) -=
      get<0, 1>(surface_metric) * get(cos_theta) / get(sin_theta);

  auto grad_surface_metric_phi_phi = ylm.gradient(get<1, 1>(surface_metric));

  auto deriv_surface_metric =
      make_with_value<tnsr::ijj<DataVector, 2, Frame::Spherical<Fr>>>(
          get<0, 0>(surface_metric), 0.0);
  // Get the partial derivative of the metric from the Pfaffian derivative
  get<0, 0, 0>(deriv_surface_metric) = get<0>(grad_surface_metric_theta_theta);
  get<1, 0, 0>(deriv_surface_metric) =
      get(sin_theta) * get<1>(grad_surface_metric_theta_theta);
  get<0, 0, 1>(deriv_surface_metric) = get<0>(grad_surface_metric_theta_phi);
  get<1, 0, 1>(deriv_surface_metric) =
      get(sin_theta) * get<1>(grad_surface_metric_theta_phi);
  get<0, 1, 1>(deriv_surface_metric) = get<0>(grad_surface_metric_phi_phi);
  get<1, 1, 1>(deriv_surface_metric) =
      get(sin_theta) * get<1>(grad_surface_metric_phi_phi);

  return trace_last_indices(
      raise_or_lower_first_index(
          gr::christoffel_first_kind(deriv_surface_metric),
          inverse_surface_metric),
      inverse_surface_metric);
}

// I'm going to solve a general eigenvalue problem of the form
// A x = lambda B x, where A and B are NxN, where N is the
// number of elements with l > 0 and l < ntheta - 2,
// i.e. l < l_max + 1 - 2 = l_max - 1. This function computes N.
size_t get_matrix_dimension(const ylm::Spherepack& ylm) {
  // If l_max == m_max, there are square(l_max+1) Ylms total
  size_t matrix_dimension = square(ylm.l_max() + 1);
  // If l_max > m_max, there are
  // (l_max - m_max) * (l_max - m_max + 1) fewer Ylms total
  matrix_dimension -=
      (ylm.l_max() - ylm.m_max()) * (ylm.l_max() - ylm.m_max() + 1);
  // The actual matrix dimension is smaller, because we do not count
  // Ylms with l == 0, l == l_max, or l == l_max - 1.
  matrix_dimension -= 4 * ylm.m_max() + 3;
  if (ylm.l_max() == ylm.m_max()) {
    matrix_dimension += 2;
  }
  return matrix_dimension;
}

// Get left matrix A and right matrix B for eigenproblem A x = lambda B x.
template <typename Fr>
void get_left_and_right_eigenproblem_matrices(
    const gsl::not_null<Matrix*> left_matrix,
    const gsl::not_null<Matrix*> right_matrix,
    const tnsr::II<DataVector, 2, Frame::Spherical<Fr>>& inverse_surface_metric,
    const tnsr::I<DataVector, 2, Frame::Spherical<Fr>>&
        trace_christoffel_second_kind,
    const Scalar<DataVector>& sin_theta, const Scalar<DataVector>& ricci_scalar,
    const ylm::Spherepack& ylm) {
  const auto grad_ricci_scalar = ylm.gradient(get(ricci_scalar));
  // loop over all terms with 0<l<l_max-1: each makes a column of
  // the matrices for the eigenvalue problem
  size_t column = 0;  // number which column of the matrix we are filling
  for (auto iter_i = ylm::SpherepackIterator(ylm.l_max(), ylm.m_max()); iter_i;
       ++iter_i) {
    if (iter_i.l() > 0 and iter_i.l() < ylm.l_max() - 1 and
        iter_i.m() <= iter_i.l()) {
      // Make a spectral vector that's all zeros except for one element,
      // which is 1. This corresponds to the ith Ylm, which I call yi.
      DataVector yi_spectral(ylm.spectral_size(), 0.0);
      yi_spectral[iter_i()] = 1.0;

      // Transform column vector corresponding to
      // a specific Y_lm to physical space.
      const DataVector yi_physical = ylm.spec_to_phys(yi_spectral);

      // In physical space, numerically compute the
      // linear differential operators acting on the
      // ith Y_lm.

      // \nabla^2 Y_lm
      const auto derivs_yi = ylm.first_and_second_derivative(yi_physical);
      auto laplacian_yi =
          make_with_value<Scalar<DataVector>>(ricci_scalar, 0.0);
      get(laplacian_yi) +=
          get<0, 0>(derivs_yi.second) * get<0, 0>(inverse_surface_metric);
      get(laplacian_yi) += 2.0 * get<1, 0>(derivs_yi.second) *
                           get<1, 0>(inverse_surface_metric) * get(sin_theta);
      get(laplacian_yi) += get<1, 1>(derivs_yi.second) *
                           get<1, 1>(inverse_surface_metric) *
                           square(get(sin_theta));
      get(laplacian_yi) -=
          get<0>(derivs_yi.first) * get<0>(trace_christoffel_second_kind);
      get(laplacian_yi) -= get<1>(derivs_yi.first) * get(sin_theta) *
                           get<1>(trace_christoffel_second_kind);

      // \nabla^4 Y_lm
      const auto derivs_laplacian_yi =
          ylm.first_and_second_derivative(get(laplacian_yi));
      auto laplacian_squared_yi =
          make_with_value<Scalar<DataVector>>(ricci_scalar, 0.0);
      get(laplacian_squared_yi) += get<0, 0>(derivs_laplacian_yi.second) *
                                   get<0, 0>(inverse_surface_metric);
      get(laplacian_squared_yi) += 2.0 * get<1, 0>(derivs_laplacian_yi.second) *
                                   get<1, 0>(inverse_surface_metric) *
                                   get(sin_theta);
      get(laplacian_squared_yi) += get<1, 1>(derivs_laplacian_yi.second) *
                                   get<1, 1>(inverse_surface_metric) *
                                   square(get(sin_theta));
      get(laplacian_squared_yi) -= get<0>(derivs_laplacian_yi.first) *
                                   get<0>(trace_christoffel_second_kind);
      get(laplacian_squared_yi) -= get<1>(derivs_laplacian_yi.first) *
                                   get(sin_theta) *
                                   get<1>(trace_christoffel_second_kind);

      // \nabla R \cdot \nabla Y_lm
      auto grad_ricci_scalar_dot_grad_yi =
          make_with_value<Scalar<DataVector>>(ricci_scalar, 0.0);
      get(grad_ricci_scalar_dot_grad_yi) += get<0>(derivs_yi.first) *
                                            get<0>(grad_ricci_scalar) *
                                            get<0, 0>(inverse_surface_metric);
      get(grad_ricci_scalar_dot_grad_yi) +=
          get<0>(derivs_yi.first) * get<1>(grad_ricci_scalar) *
          get<1, 0>(inverse_surface_metric) * get(sin_theta);
      get(grad_ricci_scalar_dot_grad_yi) +=
          get<1>(derivs_yi.first) * get<0>(grad_ricci_scalar) *
          get<1, 0>(inverse_surface_metric) * get(sin_theta);
      get(grad_ricci_scalar_dot_grad_yi) +=
          get<1>(derivs_yi.first) * get<1>(grad_ricci_scalar) *
          get<1, 1>(inverse_surface_metric) * square(get(sin_theta));

      // Assemble the operator making up the eigenproblem's left-hand-side
      auto left_matrix_yi_physical =
          make_with_value<Scalar<DataVector>>(ricci_scalar, 0.0);
      get(left_matrix_yi_physical) = get(laplacian_squared_yi) +
                                     get(ricci_scalar) * get(laplacian_yi) +
                                     get(grad_ricci_scalar_dot_grad_yi);

      // Transform back to spectral space, to get one column each for the left
      // and right matrices for the eigenvalue problem.
      const DataVector left_matrix_yi_spectral =
          ylm.phys_to_spec(get(left_matrix_yi_physical));
      const DataVector right_matrix_yi_spectral =
          ylm.phys_to_spec(get(laplacian_yi));

      // Set the current column of the left and right matrices
      // for the eigenproblem.
      size_t row = 0;
      for (auto iter_j = ylm::SpherepackIterator(ylm.l_max(), ylm.m_max());
           iter_j; ++iter_j) {
        if (iter_j.l() > 0 and iter_j.l() < ylm.l_max() - 1) {
          (*left_matrix)(row, column) = left_matrix_yi_spectral[iter_j()];
          (*right_matrix)(row, column) = right_matrix_yi_spectral[iter_j()];
          ++row;
        }
      }  // loop over rows
      ++column;
    }
  }  // loop over columns
}

// Find the eigenvectors corresponding to the three smallest-magnitude
// eigenvalues.
// Note: uses the fact that eigenvalues should be real
std::array<DataVector, 3> get_eigenvectors_for_3_smallest_magnitude_eigenvalues(
    const DataVector& eigenvalues_real_part, const Matrix& eigenvectors,
    const ylm::Spherepack& ylm) {
  size_t index_smallest = 0;
  size_t index_second_smallest = 0;
  size_t index_third_smallest = 0;

  // Simple algorithm that loops over all elements to
  // find indexes of 3 smallest-magnitude eigenvalues
  for (size_t i = 1; i < eigenvalues_real_part.size(); ++i) {
    if (abs(eigenvalues_real_part[i]) <
        abs(eigenvalues_real_part[index_smallest])) {
      index_third_smallest = index_second_smallest;
      index_second_smallest = index_smallest;
      index_smallest = i;
    } else if (i < 2 or abs(eigenvalues_real_part[i]) <
                            abs(eigenvalues_real_part[index_second_smallest])) {
      index_third_smallest = index_second_smallest;
      index_second_smallest = i;
    } else if (i < 3 or abs(eigenvalues_real_part[i]) <
                            abs(eigenvalues_real_part[index_third_smallest])) {
      index_third_smallest = i;
    }
  }

  DataVector smallest_eigenvector(ylm.spectral_size(), 0.0);
  DataVector second_smallest_eigenvector(ylm.spectral_size(), 0.0);
  DataVector third_smallest_eigenvector(ylm.spectral_size(), 0.0);

  size_t row = 0;

  for (auto iter_i = ylm::SpherepackIterator(ylm.l_max(), ylm.m_max()); iter_i;
       ++iter_i) {
    if (iter_i.l() > 0 and iter_i.l() < ylm.l_max() - 1) {
      smallest_eigenvector[iter_i()] = eigenvectors(row, index_smallest);
      second_smallest_eigenvector[iter_i()] =
          eigenvectors(row, index_second_smallest);
      third_smallest_eigenvector[iter_i()] =
          eigenvectors(row, index_third_smallest);
      ++row;
    }
  }

  return {{smallest_eigenvector, second_smallest_eigenvector,
           third_smallest_eigenvector}};
}

// This function converts the three eigenvectors with smallest-magnitude
// eigenvalues to physical space to get the spin potentials corresponding to
// the approximate Killing vectors. The potentials are normalized using the
// "Kerr normalization:" the integral of (potential - the potential average)^2
// is set to (horizon area)^3/(48*pi), as it is for Kerr.
std::array<DataVector, 3> get_normalized_spin_potentials(
    const std::array<DataVector, 3>& eigenvectors_for_potentials,
    const ylm::Spherepack& ylm, const Scalar<DataVector>& area_element) {
  const double area = ylm.definite_integral(get(area_element).data());

  std::array<DataVector, 3> potentials;

  DataVector temp_integrand(get(area_element));
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(potentials, i) =
        ylm.spec_to_phys(gsl::at(eigenvectors_for_potentials, i));

    temp_integrand = gsl::at(potentials, i) * get(area_element);
    const double potential_average =
        ylm.definite_integral(temp_integrand.data()) / area;

    temp_integrand =
        square(gsl::at(potentials, i) - potential_average) * get(area_element);
    const double potential_norm = ylm.definite_integral(temp_integrand.data());
    gsl::at(potentials, i) *=
        sqrt(cube(area) / (48.0 * square(M_PI) * potential_norm));
  }
  return potentials;
}

// Get the spin magnitude. There are three potentials, each corresponding to an
// approximate Killing vector. The spin for each potential is the surface
// integral of the potential times the spin function. The spin magnitude
// is the Euclidean norm of the spin for each potential.
double get_spin_magnitude(const std::array<DataVector, 3>& potentials,
                          const Scalar<DataVector>& spin_function,
                          const Scalar<DataVector>& area_element,
                          const ylm::Spherepack& ylm) {
  double spin_magnitude_squared = 0.0;

  DataVector spin_density(get(area_element));
  for (size_t i = 0; i < 3; ++i) {
    spin_density =
        gsl::at(potentials, i) * get(spin_function) * get(area_element);
    spin_magnitude_squared +=
        square(ylm.definite_integral(spin_density.data()) / (8.0 * M_PI));
  }
  return sqrt(spin_magnitude_squared);
}
}  // namespace

namespace gr::surfaces {
template <typename Frame>
void spin_function(const gsl::not_null<Scalar<DataVector>*> result,
                   const ylm::Tags::aliases::Jacobian<Frame>& tangents,
                   const ylm::Strahlkorper<Frame>& strahlkorper,
                   const tnsr::I<DataVector, 3, Frame>& unit_normal_vector,
                   const Scalar<DataVector>& area_element,
                   const tnsr::ii<DataVector, 3, Frame>& extrinsic_curvature) {
  set_number_of_grid_points(result, area_element);
  for (auto& component : *result) {
    component = 0.0;
  }

  auto extrinsic_curvature_theta_normal_sin_theta =
      make_with_value<Scalar<DataVector>>(area_element, 0.0);
  auto extrinsic_curvature_phi_normal =
      make_with_value<Scalar<DataVector>>(area_element, 0.0);

  // using result as temporary
  DataVector& extrinsic_curvature_dot_normal = get(*result);
  for (size_t i = 0; i < 3; ++i) {
    extrinsic_curvature_dot_normal =
        extrinsic_curvature.get(i, 0) * get<0>(unit_normal_vector);
    for (size_t j = 1; j < 3; ++j) {
      extrinsic_curvature_dot_normal +=
          extrinsic_curvature.get(i, j) * unit_normal_vector.get(j);
    }

    // Note: I must multiply by sin_theta because
    // I take the phi derivative of this term by using
    // the spherepack gradient, which includes a
    // sin_theta in the denominator of the phi derivative.
    // Will do this outside the i,j loops.
    get(extrinsic_curvature_theta_normal_sin_theta) +=
        extrinsic_curvature_dot_normal * tangents.get(i, 0);

    // Note: I must multiply by sin_theta because tangents.get(i,1)
    // actually contains \partial_\phi / sin(theta), but I want just
    //\partial_\phi. Will do this outside the i,j loops.
    get(extrinsic_curvature_phi_normal) +=
        extrinsic_curvature_dot_normal * tangents.get(i, 1);
  }

  // using result as temporary
  DataVector& sin_theta = get(*result);
  sin_theta = sin(strahlkorper.ylm_spherepack().theta_phi_points()[0]);
  get(extrinsic_curvature_theta_normal_sin_theta) *= sin_theta;
  get(extrinsic_curvature_phi_normal) *= sin_theta;

  // now computing actual result
  get(*result) = (get<0>(strahlkorper.ylm_spherepack().gradient(
                      get(extrinsic_curvature_phi_normal))) -
                  get<1>(strahlkorper.ylm_spherepack().gradient(
                      get(extrinsic_curvature_theta_normal_sin_theta)))) /
                 (sin_theta * get(area_element));
}

template <typename Frame>
Scalar<DataVector> spin_function(
    const ylm::Tags::aliases::Jacobian<Frame>& tangents,
    const ylm::Strahlkorper<Frame>& strahlkorper,
    const tnsr::I<DataVector, 3, Frame>& unit_normal_vector,
    const Scalar<DataVector>& area_element,
    const tnsr::ii<DataVector, 3, Frame>& extrinsic_curvature) {
  Scalar<DataVector> result{};
  spin_function(make_not_null(&result), tangents, strahlkorper,
                unit_normal_vector, area_element, extrinsic_curvature);
  return result;
}

template <typename Frame>
void dimensionful_spin_magnitude(
    const gsl::not_null<double*> result, const Scalar<DataVector>& ricci_scalar,
    const Scalar<DataVector>& spin_function,
    const tnsr::ii<DataVector, 3, Frame>& spatial_metric,
    const ylm::Tags::aliases::Jacobian<Frame>& tangents,
    const ylm::Strahlkorper<Frame>& strahlkorper,
    const Scalar<DataVector>& area_element) {
  const Scalar<DataVector> sin_theta{
      sin(strahlkorper.ylm_spherepack().theta_phi_points()[0])};

  const auto surface_metric =
      get_surface_metric(spatial_metric, tangents, sin_theta);
  const auto inverse_surface_metric =
      determinant_and_inverse(surface_metric).second;
  const auto trace_christoffel_second_kind = get_trace_christoffel_second_kind(
      surface_metric, inverse_surface_metric, sin_theta,
      strahlkorper.ylm_spherepack());

  const size_t matrix_dimension =
      get_matrix_dimension(strahlkorper.ylm_spherepack());
  Matrix left_matrix(matrix_dimension, matrix_dimension, 0.0);
  Matrix right_matrix(matrix_dimension, matrix_dimension, 0.0);
  get_left_and_right_eigenproblem_matrices(
      &left_matrix, &right_matrix, inverse_surface_metric,
      trace_christoffel_second_kind, sin_theta, ricci_scalar,
      strahlkorper.ylm_spherepack());

  DataVector eigenvalues_real_part(matrix_dimension, 0.0);
  DataVector eigenvalues_im_part(matrix_dimension, 0.0);
  Matrix eigenvectors(matrix_dimension, matrix_dimension, 0.0);
  find_generalized_eigenvalues(&eigenvalues_real_part, &eigenvalues_im_part,
                               &eigenvectors, left_matrix, right_matrix);

  const std::array<DataVector, 3> smallest_eigenvectors =
      get_eigenvectors_for_3_smallest_magnitude_eigenvalues(
          eigenvalues_real_part, eigenvectors, strahlkorper.ylm_spherepack());

  // Get normalized potentials (Kerr normalization) corresponding to the
  // eigenvectors with three smallest-magnitude eigenvalues.
  const auto potentials = get_normalized_spin_potentials(
      smallest_eigenvectors, strahlkorper.ylm_spherepack(), area_element);

  *result = get_spin_magnitude(potentials, spin_function, area_element,
                               strahlkorper.ylm_spherepack());
}

template <typename Frame>
double dimensionful_spin_magnitude(
    const Scalar<DataVector>& ricci_scalar,
    const Scalar<DataVector>& spin_function,
    const tnsr::ii<DataVector, 3, Frame>& spatial_metric,
    const ylm::Tags::aliases::Jacobian<Frame>& tangents,
    const ylm::Strahlkorper<Frame>& strahlkorper,
    const Scalar<DataVector>& area_element) {
  double result{};
  dimensionful_spin_magnitude(make_not_null(&result), ricci_scalar,
                              spin_function, spatial_metric, tangents,
                              strahlkorper, area_element);
  return result;
}

void dimensionless_spin_magnitude(const gsl::not_null<double*> result,
                                  const double dimensionful_spin_magnitude,
                                  const double christodoulou_mass) {
  *result = dimensionful_spin_magnitude / square(christodoulou_mass);
}

double dimensionless_spin_magnitude(const double dimensionful_spin_magnitude,
                                    const double christodoulou_mass) {
  double result{};
  dimensionless_spin_magnitude(make_not_null(&result),
                               dimensionful_spin_magnitude, christodoulou_mass);
  return result;
}

template <typename MetricDataFrame, typename MeasurementFrame>
void spin_vector(
    const gsl::not_null<std::array<double, 3>*> result, double spin_magnitude,
    const Scalar<DataVector>& area_element,
    const Scalar<DataVector>& ricci_scalar,
    const Scalar<DataVector>& spin_function,
    const ylm::Strahlkorper<MetricDataFrame>& strahlkorper,
    const tnsr::I<DataVector, 3, MeasurementFrame>& measurement_frame_coords) {
  const auto& ylm = strahlkorper.ylm_spherepack();

  // Assert that the DataVectors in area_element, ricci_scalar, and
  // spin_function have the same size as the ylm size.

  // get the ylm's physical size as a variable to reuse
  const size_t ylm_physical_size = ylm.physical_size();
  ASSERT(get(area_element).size() == ylm_physical_size,
         "area_element size doesn't match ylm physical size: "
             << get(area_element).size() << " vs " << ylm_physical_size);
  ASSERT(get(ricci_scalar).size() == ylm_physical_size,
         "ricci_scalar size doesn't match ylm physical size: "
             << get(ricci_scalar).size() << " vs " << ylm_physical_size);
  ASSERT(get(spin_function).size() == ylm_physical_size,
         "spin_function size doesn't match ylm physical size: "
             << get(spin_function).size() << " vs " << ylm_physical_size);

  // Compute very rough center of the measurement frame by simply
  // averaging measurement_frame_coords.  It is ok for this center to
  // be very rough, since it will be corrected below.
  const auto measurement_frame_center =
      [&measurement_frame_coords]() -> std::array<double, 3> {
    std::array<double, 3> center{};
    for (size_t d = 0; d < 3; ++d) {
      gsl::at(center, d) =
          std::accumulate(measurement_frame_coords.get(d).begin(),
                          measurement_frame_coords.get(d).end(), 0.0) /
          measurement_frame_coords.get(d).size();
    }
    return center;
  }();

  std::array<double, 3> spin_vector =
      make_array<3>(std::numeric_limits<double>::signaling_NaN());
  auto integrand = make_with_value<Scalar<DataVector>>(get(area_element), 0.0);

  for (size_t i = 0; i < 3; ++i) {
    get(integrand) = get(area_element) * get(ricci_scalar) *
                     (measurement_frame_coords.get(i) -
                      gsl::at(measurement_frame_center, i));
    get(integrand) =
        ylm.definite_integral(get(integrand).data()) / (-8.0 * M_PI);
    // integrand in the above line is -x^i_R from the paper.

    get(integrand) +=
        measurement_frame_coords.get(i) - gsl::at(measurement_frame_center, i);
    get(integrand) *= get(area_element) * get(spin_function);
    gsl::at(spin_vector, i) = ylm.definite_integral(get(integrand).data());
  }

  // Normalize spin_vector so its magnitude is the magnitude of the spin.
  *result = spin_vector * (spin_magnitude / magnitude(spin_vector));
}

template <typename MetricDataFrame, typename MeasurementFrame>
std::array<double, 3> spin_vector(
    double spin_magnitude, const Scalar<DataVector>& area_element,
    const Scalar<DataVector>& ricci_scalar,
    const Scalar<DataVector>& spin_function,
    const ylm::Strahlkorper<MetricDataFrame>& strahlkorper,
    const tnsr::I<DataVector, 3, MeasurementFrame>& measurement_frame_coords) {
  std::array<double, 3> result{};
  spin_vector(make_not_null(&result), spin_magnitude, area_element,
              ricci_scalar, spin_function, strahlkorper,
              measurement_frame_coords);
  return result;
}
}  // namespace gr::surfaces

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data)                                              \
  template void gr::surfaces::spin_function<FRAME(data)>(                 \
      const gsl::not_null<Scalar<DataVector>*> result,                    \
      const ylm::Tags::aliases::Jacobian<FRAME(data)>& tangents,          \
      const ylm::Strahlkorper<FRAME(data)>& strahlkorper,                 \
      const tnsr::I<DataVector, 3, FRAME(data)>& unit_normal_vector,      \
      const Scalar<DataVector>& area_element,                             \
      const tnsr::ii<DataVector, 3, FRAME(data)>& extrinsic_curvature);   \
  template Scalar<DataVector> gr::surfaces::spin_function<FRAME(data)>(   \
      const ylm::Tags::aliases::Jacobian<FRAME(data)>& tangents,          \
      const ylm::Strahlkorper<FRAME(data)>& strahlkorper,                 \
      const tnsr::I<DataVector, 3, FRAME(data)>& unit_normal_vector,      \
      const Scalar<DataVector>& area_element,                             \
      const tnsr::ii<DataVector, 3, FRAME(data)>& extrinsic_curvature);   \
  template void gr::surfaces::dimensionful_spin_magnitude<FRAME(data)>(   \
      const gsl::not_null<double*> result,                                \
      const Scalar<DataVector>& ricci_scalar,                             \
      const Scalar<DataVector>& spin_function,                            \
      const tnsr::ii<DataVector, 3, FRAME(data)>& spatial_metric,         \
      const ylm::Tags::aliases::Jacobian<FRAME(data)>& tangents,          \
      const ylm::Strahlkorper<FRAME(data)>& strahlkorper,                 \
      const Scalar<DataVector>& area_element);                            \
  template double gr::surfaces::dimensionful_spin_magnitude<FRAME(data)>( \
      const Scalar<DataVector>& ricci_scalar,                             \
      const Scalar<DataVector>& spin_function,                            \
      const tnsr::ii<DataVector, 3, FRAME(data)>& spatial_metric,         \
      const ylm::Tags::aliases::Jacobian<FRAME(data)>& tangents,          \
      const ylm::Strahlkorper<FRAME(data)>& strahlkorper,                 \
      const Scalar<DataVector>& area_element);
GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (Frame::Grid, Frame::Distorted, Frame::Inertial))
#undef INSTANTIATE
#undef FRAME

#define METRICFRAME(data) BOOST_PP_TUPLE_ELEM(0, data)
#define MEASUREMENTFRAME(data) BOOST_PP_TUPLE_ELEM(1, data)
#define INSTANTIATE(_, data)                                               \
  template void                                                            \
  gr::surfaces::spin_vector<METRICFRAME(data), MEASUREMENTFRAME(data)>(    \
      const gsl::not_null<std::array<double, 3>*> result,                  \
      const double spin_magnitude, const Scalar<DataVector>& area_element, \
      const Scalar<DataVector>& ricci_scalar,                              \
      const Scalar<DataVector>& spin_function,                             \
      const ylm::Strahlkorper<METRICFRAME(data)>& strahlkorper,            \
      const tnsr::I<DataVector, 3, MEASUREMENTFRAME(data)>&                \
          measurement_frame_coords);                                       \
  template std::array<double, 3>                                           \
  gr::surfaces::spin_vector<METRICFRAME(data), MEASUREMENTFRAME(data)>(    \
      const double spin_magnitude, const Scalar<DataVector>& area_element, \
      const Scalar<DataVector>& ricci_scalar,                              \
      const Scalar<DataVector>& spin_function,                             \
      const ylm::Strahlkorper<METRICFRAME(data)>& strahlkorper,            \
      const tnsr::I<DataVector, 3, MEASUREMENTFRAME(data)>&                \
          measurement_frame_coords);

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (Frame::Grid, Frame::Distorted, Frame::Inertial),
                        (Frame::Inertial, Frame::Distorted))
#undef INSTANTIATE
#undef MEASUREMENTFRAME
#undef METRICFRAME
