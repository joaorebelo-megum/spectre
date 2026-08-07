// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Elliptic/Executables/Xcts/SolveXcts.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "IO/Exporter/PointwiseInterpolator.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/H5/VolumeData.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/PerturbationBackground.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Flatness.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "PointwiseFunctions/InitialDataUtilities/NumericData.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"

namespace Xcts::AnalyticData {

namespace {

using conformal_metric_tag =
    Xcts::Tags::ConformalMetric<DataVector, 3, Frame::Inertial>;
using inverse_conformal_metric_tag =
    Xcts::Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>;
using loaded_tags =
    tmpl::list<conformal_metric_tag, inverse_conformal_metric_tag>;

// `SolveXcts.hpp` defines the executable's metavariables in the global
// namespace. Check the factory registration against those, so this test fails
// if the background is dropped from the executable that actually uses it.
using solve_xcts_backgrounds =
    tmpl::at<typename ::Metavariables::factory_creation::factory_classes,
             elliptic::analytic_data::Background>;
static_assert(
    tmpl::list_contains_v<solve_xcts_backgrounds, PerturbationBackground>,
    "PerturbationBackground must be registered in the SolveXcts background "
    "factory.");
static_assert(tmpl::list_contains_v<solve_xcts_backgrounds,
                                    elliptic::analytic_data::NumericData>,
              "NumericData must be registered in the SolveXcts background "
              "factory.");

// A minimal factory list for option-parsing tests, so we don't have to register
// every class of the executable with Charm.
struct TestMetavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<elliptic::analytic_data::Background,
                             tmpl::list<PerturbationBackground,
                                        elliptic::analytic_data::NumericData>>>;
  };
};

const std::string h5_file_name{
    "Unit.PointwiseFunctions.AnalyticData.Xcts.PerturbationBackground.h5"};
constexpr double amplitude = 0.1;
constexpr double sigma = 2.0;

// The Gaussian that `PerturbationBackground` adds to the conformal metric.
double expected_profile(const double x, const double y, const double z) {
  return amplitude * exp(-(x * x + y * y + z * z) / (sigma * sigma));
}

// Datasets in SpECTRE volume files are named `db::tag_name<Tag>()` plus a
// component suffix. `Xcts::Tags::ConformalMetric` is the prefix tag
// `gr::Tags::Conformal<gr::Tags::SpatialMetric, -4>`, so the names are
// "Conformal(SpatialMetric)_xx" and so on -- not "ConformalMetric_xx". Pin the
// convention down here, so a tag rename upstream produces a readable failure
// rather than a "dataset not found" error deep inside the exporter.
void test_dataset_names() {
  INFO("Dataset names of the tags we load");
  // Symmetric rank-2 tensors are stored in the order xx, yx, zx, yy, zy, zz.
  const std::vector<std::string> expected{"Conformal(SpatialMetric)_xx",
                                          "Conformal(SpatialMetric)_yx",
                                          "Conformal(SpatialMetric)_zx",
                                          "Conformal(SpatialMetric)_yy",
                                          "Conformal(SpatialMetric)_zy",
                                          "Conformal(SpatialMetric)_zz",
                                          "Conformal(InverseSpatialMetric)_xx",
                                          "Conformal(InverseSpatialMetric)_yx",
                                          "Conformal(InverseSpatialMetric)_zx",
                                          "Conformal(InverseSpatialMetric)_yy",
                                          "Conformal(InverseSpatialMetric)_zy",
                                          "Conformal(InverseSpatialMetric)_zz"};
  CHECK(spectre::Exporter::get_tensor_components<loaded_tags>() == expected);
}

// Write a volume-data file that mimics the output of an XCTS solve: a flat
// conformal metric on a single-element brick. The serialized domain is
// essential, because `spectre::Exporter` maps the target points through it and
// dereferences it unconditionally.
void write_volume_file() {
  if (file_system::check_if_file_exists(h5_file_name)) {
    file_system::rm(h5_file_name, true);
  }
  // Serializing the domain pups the block coordinate maps, so they have to be
  // registered with Charm first.
  domain::creators::register_derived_with_charm();
  domain::creators::time_dependence::register_derived_with_charm();
  domain::FunctionsOfTime::register_derived_with_charm();
  const domain::creators::Brick domain_creator{
      {{-1.0, -1.0, -1.0}}, {{1.0, 1.0, 1.0}}, {{0, 0, 0}}, {{3, 3, 3}}};
  const auto domain = domain_creator.create_domain();
  const Mesh<3> mesh{3, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  // Constant data is represented exactly on the grid, so interpolating it back
  // to the target points is exact and the expected values below are analytic.
  const DataVector ones{mesh.number_of_grid_points(), 1.0};
  const DataVector zeros{mesh.number_of_grid_points(), 0.0};
  h5::H5File<h5::AccessType::ReadWrite> h5_file{h5_file_name};
  auto& volume_data = h5_file.insert<h5::VolumeData>("/element_data", 0);
  volume_data.write_volume_data(
      0, 0.0,
      {ElementVolumeData{
          ElementId<3>{0},
          {TensorComponent{"Conformal(SpatialMetric)_xx", ones},
           TensorComponent{"Conformal(SpatialMetric)_yx", zeros},
           TensorComponent{"Conformal(SpatialMetric)_yy", ones},
           TensorComponent{"Conformal(SpatialMetric)_zx", zeros},
           TensorComponent{"Conformal(SpatialMetric)_zy", zeros},
           TensorComponent{"Conformal(SpatialMetric)_zz", ones},
           TensorComponent{"Conformal(InverseSpatialMetric)_xx", ones},
           TensorComponent{"Conformal(InverseSpatialMetric)_yx", zeros},
           TensorComponent{"Conformal(InverseSpatialMetric)_yy", ones},
           TensorComponent{"Conformal(InverseSpatialMetric)_zx", zeros},
           TensorComponent{"Conformal(InverseSpatialMetric)_zy", zeros},
           TensorComponent{"Conformal(InverseSpatialMetric)_zz", ones}},
          mesh}},
      serialize(domain));
}

// Load the volume file directly, without any perturbation, to separate a
// failure of the numeric loading from a failure of the perturbation.
void test_numeric_loading(const tnsr::I<DataVector, 3, Frame::Inertial>& x) {
  INFO("Load the conformal metric from a volume file");
  const elliptic::analytic_data::NumericData numeric_data{
      h5_file_name, "element_data", 0, false};
  const auto vars = numeric_data.variables(x, loaded_tags{});
  const auto& conformal_metric = get<conformal_metric_tag>(vars);
  const auto& inverse_conformal_metric =
      get<inverse_conformal_metric_tag>(vars);
  const DataVector expected_ones{get<0>(x).size(), 1.0};
  const DataVector expected_zeros{get<0>(x).size(), 0.0};
  CHECK_ITERABLE_APPROX((get<0, 0>(conformal_metric)), expected_ones);
  CHECK_ITERABLE_APPROX((get<1, 1>(conformal_metric)), expected_ones);
  CHECK_ITERABLE_APPROX((get<2, 2>(conformal_metric)), expected_ones);
  CHECK_ITERABLE_APPROX((get<1, 0>(conformal_metric)), expected_zeros);
  CHECK_ITERABLE_APPROX((get<2, 0>(conformal_metric)), expected_zeros);
  CHECK_ITERABLE_APPROX((get<2, 1>(conformal_metric)), expected_zeros);
  CHECK_ITERABLE_APPROX((get<0, 0>(inverse_conformal_metric)), expected_ones);
}

// Create the wrapper from options with a numeric background, as an input file
// would, and check that the loaded metric comes back perturbed.
void test_perturbed_numeric_background(
    const tnsr::I<DataVector, 3, Frame::Inertial>& x) {
  INFO("Perturb a conformal metric loaded from a volume file");
  const auto created = TestHelpers::test_creation<
      std::unique_ptr<elliptic::analytic_data::Background>, TestMetavariables>(
      "PerturbationBackground:\n"
      "  Background:\n"
      "    NumericData:\n"
      "      FileGlob: " +
      h5_file_name +
      "\n"
      "      Subgroup: element_data\n"
      "      ObservationStep: 0\n"
      "      ExtrapolateIntoExcisions: False\n"
      "  Amplitude: 0.1\n"
      "  Sigma: 2.0\n"
      "  Components: [xx]\n");
  REQUIRE(dynamic_cast<const PerturbationBackground*>(created.get()) !=
          nullptr);
  const auto& background =
      dynamic_cast<const PerturbationBackground&>(*created);

  // Cloning round-trips the wrapped background through serialization, so this
  // also checks that the numeric background is registered with Charm.
  const auto cloned = background.get_clone();
  REQUIRE(dynamic_cast<const PerturbationBackground*>(cloned.get()) != nullptr);

  const size_t num_points = get<0>(x).size();
  DataVector expected_xx{num_points};
  for (size_t i = 0; i < num_points; ++i) {
    expected_xx[i] =
        1.0 + expected_profile(get<0>(x)[i], get<1>(x)[i], get<2>(x)[i]);
  }
  const DataVector expected_ones{num_points, 1.0};

  for (const auto* const to_check :
       {&background,
        dynamic_cast<const PerturbationBackground*>(cloned.get())}) {
    const auto vars = to_check->variables(x, loaded_tags{});
    const auto& conformal_metric = get<conformal_metric_tag>(vars);
    const auto& inverse_conformal_metric =
        get<inverse_conformal_metric_tag>(vars);
    // Only the xx component was selected for perturbation.
    CHECK_ITERABLE_APPROX((get<0, 0>(conformal_metric)), expected_xx);
    CHECK_ITERABLE_APPROX((get<1, 1>(conformal_metric)), expected_ones);
    CHECK_ITERABLE_APPROX((get<2, 2>(conformal_metric)), expected_ones);
    // The inverse is recomputed from the perturbed metric, so it is no longer
    // the inverse metric that was loaded from the file.
    CHECK_ITERABLE_APPROX((get<0, 0>(inverse_conformal_metric)),
                          DataVector{1.0 / expected_xx});
    CHECK_ITERABLE_APPROX((get<1, 1>(inverse_conformal_metric)), expected_ones);
  }
}

// The same perturbation on an analytic background, where no file I/O is
// involved.
void test_perturbed_analytic_background(
    const tnsr::I<DataVector, 3, Frame::Inertial>& x) {
  INFO("Perturb a flat analytic background");
  const PerturbationBackground background{
      std::make_unique<Xcts::Solutions::Flatness>(), amplitude, sigma,
      std::vector<std::string>{"xx", "yz"}};
  const auto vars = background.variables(x, loaded_tags{});
  const auto& conformal_metric = get<conformal_metric_tag>(vars);
  const size_t num_points = get<0>(x).size();
  DataVector expected_xx{num_points};
  DataVector expected_yz{num_points};
  for (size_t i = 0; i < num_points; ++i) {
    const double profile =
        expected_profile(get<0>(x)[i], get<1>(x)[i], get<2>(x)[i]);
    expected_xx[i] = 1.0 + profile;
    expected_yz[i] = profile;
  }
  CHECK_ITERABLE_APPROX((get<0, 0>(conformal_metric)), expected_xx);
  CHECK_ITERABLE_APPROX((get<1, 2>(conformal_metric)), expected_yz);
  // The perturbation is symmetric.
  CHECK_ITERABLE_APPROX((get<2, 1>(conformal_metric)), expected_yz);
  const DataVector expected_zeros{num_points, 0.0};
  CHECK_ITERABLE_APPROX((get<0, 1>(conformal_metric)), expected_zeros);
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticData.Xcts.BackgroundRegistry",
    "[PointwiseFunctions][Unit]") {
  register_classes_with_charm<PerturbationBackground, Xcts::Solutions::Flatness,
                              elliptic::analytic_data::NumericData>();
  // Target points in the interior of the brick. Points on element boundaries
  // are ambiguous on Gauss-Lobatto grids, so they are avoided here.
  const tnsr::I<DataVector, 3, Frame::Inertial> x{
      {{DataVector{0.0, 0.5, -0.25}, DataVector{0.0, 0.0, 0.5},
        DataVector{0.0, 0.0, 0.75}}}};

  test_dataset_names();
  write_volume_file();
  test_numeric_loading(x);
  test_perturbed_numeric_background(x);
  test_perturbed_analytic_background(x);
  file_system::rm(h5_file_name, true);
}

}  // namespace Xcts::AnalyticData
