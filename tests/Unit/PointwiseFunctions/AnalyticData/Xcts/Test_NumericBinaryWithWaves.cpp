// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
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

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticData.Xcts.NumericBinaryWithWaves",
    "[PointwiseFunctions][Unit]") {
  write_volume_file();
  const NumericBinaryWithWaves background{
      h5_file_name, "element_data", 0, false, angular_velocity, expansion};

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

  file_system::rm(h5_file_name, true);
}

}  // namespace Xcts::AnalyticData
