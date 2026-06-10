// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Elliptic/Executables/Xcts/SolveXcts.hpp"
#include "Framework/TestCreation.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/H5/VolumeData.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/PerturbationBackground.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Flatness.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Schwarzschild.hpp"
#include "PointwiseFunctions/InitialDataUtilities/NumericData.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

// The compile-time check for factory registration is placed after
// `Metavariables` is defined below.

namespace Xcts::AnalyticData {
namespace {

struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<elliptic::analytic_data::Background,
                   tmpl::list<Xcts::AnalyticData::PerturbationBackground,
                              elliptic::analytic_data::NumericData>>,
        tmpl::pair<elliptic::analytic_data::AnalyticSolution,
                   tmpl::list<Xcts::Solutions::Flatness,
                              Xcts::Solutions::Schwarzschild>>>;
  };
};

namespace {
using background_data_classes =
    tmpl::at<typename Metavariables::factory_creation::factory_classes,
             elliptic::analytic_data::Background>;

static_assert(tmpl::list_contains_v<background_data_classes,
                                    Xcts::AnalyticData::PerturbationBackground>,
              "PerturbationBackground must be registered in the XCTS "
              "background factory.");
}  // namespace

void write_synthetic_volume_file(const std::string& file_name) {
  if (file_system::check_if_file_exists(file_name)) {
    file_system::rm(file_name, true);
  }

  h5::H5File<h5::AccessType::ReadWrite> h5_file{file_name, true};
  auto& volume_data = h5_file.insert<h5::VolumeData>("/element_data", 0);
  const Mesh<3> mesh{2, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const DataVector ones{mesh.number_of_grid_points(), 1.0};
  const DataVector zeros{mesh.number_of_grid_points(), 0.0};
  volume_data.write_volume_data(
      0, 0.0,
      {ElementVolumeData{ElementId<3>{0},
                         {TensorComponent{"ConformalMetric_xx", ones},
                          TensorComponent{"ConformalMetric_xy", zeros},
                          TensorComponent{"ConformalMetric_xz", zeros},
                          TensorComponent{"ConformalMetric_yy", ones},
                          TensorComponent{"ConformalMetric_yz", zeros},
                          TensorComponent{"ConformalMetric_zz", ones},
                          TensorComponent{"InverseConformalMetric_xx", ones},
                          TensorComponent{"InverseConformalMetric_xy", zeros},
                          TensorComponent{"InverseConformalMetric_xz", zeros},
                          TensorComponent{"InverseConformalMetric_yy", ones},
                          TensorComponent{"InverseConformalMetric_yz", zeros},
                          TensorComponent{"InverseConformalMetric_zz", ones}},
                         mesh}});
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticData.Xcts.BackgroundRegistry",
    "[PointwiseFunctions][Unit]") {
  register_classes_with_charm<Xcts::AnalyticData::PerturbationBackground,
                              Xcts::Solutions::Flatness,
                              Xcts::Solutions::Schwarzschild>();
  const auto registered = pretty_type::list_of_names<background_data_classes>();
  std::cout << "Registered background-data implementations: " << registered
            << std::endl;
  CHECK(registered.find("PerturbationBackground") != std::string::npos);

  INFO(
      "Parse PerturbationBackground from options with a NumericData "
      "background");
  const std::string file_name{
      "Unit.PointwiseFunctions.AnalyticData.Xcts.PerturbationBackground.h5"};
  write_synthetic_volume_file(file_name);
  const auto created = TestHelpers::test_creation<
      std::unique_ptr<elliptic::analytic_data::Background>, Metavariables>(
      "PerturbationBackground:\n"
      "  Background:\n"
      "    NumericData:\n"
      "      FileGlob: " +
      file_name +
      "\n"
      "      Subgroup: element_data\n"
      "      ObservationStep: 0\n"
      "      ExtrapolateIntoExcisions: False\n"
      "  Amplitude: 0.1\n"
      "  Sigma: 2.0\n"
      "  Components: [xx]\n");
  REQUIRE(dynamic_cast<const Xcts::AnalyticData::PerturbationBackground*>(
              created.get()) != nullptr);
  const auto& numeric_background =
      dynamic_cast<const Xcts::AnalyticData::PerturbationBackground&>(*created);
  const tnsr::I<DataVector, 3> x{std::array<DataVector, 3>{
      DataVector{0.0}, DataVector{0.0}, DataVector{0.0}}};
  const auto h5_vars = numeric_background.variables(
      x, tmpl::list<Xcts::Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
                    Xcts::Tags::InverseConformalMetric<DataVector, 3,
                                                       Frame::Inertial>>{});
  CHECK(get<0, 0>(
            get<Xcts::Tags::ConformalMetric<DataVector, 3, Frame::Inertial>>(
                h5_vars))[0] == approx(1.1));
  CHECK(get<0, 0>(get<Xcts::Tags::InverseConformalMetric<DataVector, 3,
                                                         Frame::Inertial>>(
            h5_vars))[0] == approx(1.0 / 1.1));
  file_system::rm(file_name, true);

  INFO("Evaluate a Gaussian perturbation on a flat background");
  const Xcts::AnalyticData::PerturbationBackground perturbation_background{
      std::make_unique<Xcts::Solutions::Flatness>(), 0.1, 2.0,
      std::vector<std::string>{"xx"}};
  const auto cloned_background = perturbation_background.get_clone();
  CHECK(dynamic_cast<const Xcts::AnalyticData::PerturbationBackground*>(
            cloned_background.get()) != nullptr);

  const auto vars = perturbation_background.variables(
      x, tmpl::list<Xcts::Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
                    Xcts::Tags::InverseConformalMetric<DataVector, 3,
                                                       Frame::Inertial>>{});
  const auto& conformal_metric =
      get<Xcts::Tags::ConformalMetric<DataVector, 3, Frame::Inertial>>(vars);
  const auto& inverse_conformal_metric =
      get<Xcts::Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>>(
          vars);
  CHECK(get<0, 0>(conformal_metric)[0] == approx(1.1));
  CHECK(get<1, 1>(conformal_metric)[0] == approx(1.0));
  CHECK(get<2, 2>(conformal_metric)[0] == approx(1.0));
  CHECK(get<0, 0>(inverse_conformal_metric)[0] == approx(1.0 / 1.1));
  CHECK(get<1, 1>(inverse_conformal_metric)[0] == approx(1.0));
}

}  // namespace Xcts::AnalyticData
