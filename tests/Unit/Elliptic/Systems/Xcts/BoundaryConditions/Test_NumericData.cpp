// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/BoundaryConditions/ApplyBoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryCondition.hpp"
#include "Elliptic/BoundaryConditions/BoundaryConditionType.hpp"
#include "Elliptic/Systems/Xcts/BoundaryConditions/NumericData.hpp"
#include "Elliptic/Systems/Xcts/FluxesAndSources.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "IO/H5/AccessType.hpp"
#include "IO/H5/File.hpp"
#include "IO/H5/TensorData.hpp"
#include "IO/H5/VolumeData.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/FileSystem.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"

namespace Xcts::BoundaryConditions {

namespace {

const std::string h5_file_name{"Unit.Xcts.BoundaryConditions.NumericData.h5"};

// Constant data, so it is represented exactly on the grid and every expected
// value below is analytic. The metric is anisotropic so the unimodular split
// is exercised non-trivially.
constexpr double gxx = 1.0;
constexpr double gyy = 4.0;
constexpr double gzz = 9.0;
constexpr double det_g = gxx * gyy * gzz;  // 36
constexpr double lapse_value = 2.0;
constexpr double shift_x = 0.1;
constexpr double shift_y = -0.2;
constexpr double shift_z = 0.3;

double expected_conformal_factor() { return std::pow(det_g, 1.0 / 12.0); }

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
           TensorComponent{"ShiftExcess_x", DataVector{num_points, shift_x}},
           TensorComponent{"ShiftExcess_y", DataVector{num_points, shift_y}},
           TensorComponent{"ShiftExcess_z", DataVector{num_points, shift_z}}},
          mesh}},
      serialize(domain));
}

// Interior points of the brick; boundary points are ambiguous on
// Gauss-Lobatto grids.
tnsr::I<DataVector, 3> face_coordinates() {
  return tnsr::I<DataVector, 3>{
      {{DataVector{0.0, 0.5, -0.25}, DataVector{0.0, 0.0, 0.5},
        DataVector{0.0, 0.0, 0.75}}}};
}

template <Xcts::Equations EnabledEquations, bool Linearized>
void test_apply(const NumericData<EnabledEquations>& boundary_condition) {
  const auto x = face_coordinates();
  const size_t num_points = get<0>(x).size();
  const auto direction = Direction<3>::lower_xi();
  const auto box = db::create<domain::make_faces_tags<
      3,
      tmpl::conditional_t<
          Linearized,
          typename NumericData<EnabledEquations>::argument_tags_linearized,
          typename NumericData<EnabledEquations>::argument_tags>,
      tmpl::conditional_t<
          Linearized,
          typename NumericData<EnabledEquations>::volume_tags_linearized,
          typename NumericData<EnabledEquations>::volume_tags>>>(
      DirectionMap<3, tnsr::I<DataVector, 3>>{{direction, x}});

  const double nan = std::numeric_limits<double>::signaling_NaN();
  Scalar<DataVector> conformal_factor_minus_one{num_points, nan};
  Scalar<DataVector> n_dot_conformal_factor_gradient{num_points, nan};
  tnsr::i<DataVector, 3> deriv_conformal_factor{num_points, nan};

  // Linearized Dirichlet conditions impose zero corrections; the non-linearized
  // ones impose the loaded values.
  const double expected_psi_minus_one =
      Linearized ? 0.0 : expected_conformal_factor() - 1.0;
  const double expected_lapse_psi_minus_one =
      Linearized ? 0.0 : lapse_value * expected_conformal_factor() - 1.0;

  if constexpr (EnabledEquations == Xcts::Equations::Hamiltonian) {
    elliptic::apply_boundary_condition<
        Linearized, void, tmpl::list<NumericData<EnabledEquations>>>(
        boundary_condition, box, direction,
        make_not_null(&conformal_factor_minus_one),
        make_not_null(&n_dot_conformal_factor_gradient),
        deriv_conformal_factor);
  } else {
    Scalar<DataVector> lapse_times_conformal_factor_minus_one{num_points, nan};
    Scalar<DataVector> n_dot_lapse_times_conformal_factor_gradient{num_points,
                                                                   nan};
    tnsr::i<DataVector, 3> deriv_lapse_times_conformal_factor{num_points, nan};
    if constexpr (EnabledEquations == Xcts::Equations::HamiltonianAndLapse) {
      elliptic::apply_boundary_condition<
          Linearized, void, tmpl::list<NumericData<EnabledEquations>>>(
          boundary_condition, box, direction,
          make_not_null(&conformal_factor_minus_one),
          make_not_null(&lapse_times_conformal_factor_minus_one),
          make_not_null(&n_dot_conformal_factor_gradient),
          make_not_null(&n_dot_lapse_times_conformal_factor_gradient),
          deriv_conformal_factor, deriv_lapse_times_conformal_factor);
    } else {
      tnsr::I<DataVector, 3> shift_excess{num_points, nan};
      tnsr::I<DataVector, 3> n_dot_longitudinal_shift_excess{num_points, nan};
      tnsr::iJ<DataVector, 3> deriv_shift_excess{num_points, nan};
      elliptic::apply_boundary_condition<
          Linearized, void, tmpl::list<NumericData<EnabledEquations>>>(
          boundary_condition, box, direction,
          make_not_null(&conformal_factor_minus_one),
          make_not_null(&lapse_times_conformal_factor_minus_one),
          make_not_null(&shift_excess),
          make_not_null(&n_dot_conformal_factor_gradient),
          make_not_null(&n_dot_lapse_times_conformal_factor_gradient),
          make_not_null(&n_dot_longitudinal_shift_excess),
          deriv_conformal_factor, deriv_lapse_times_conformal_factor,
          deriv_shift_excess);
      CHECK_ITERABLE_APPROX(
          (get<0>(shift_excess)),
          (DataVector{num_points, Linearized ? 0.0 : shift_x}));
      CHECK_ITERABLE_APPROX(
          (get<1>(shift_excess)),
          (DataVector{num_points, Linearized ? 0.0 : shift_y}));
      CHECK_ITERABLE_APPROX(
          (get<2>(shift_excess)),
          (DataVector{num_points, Linearized ? 0.0 : shift_z}));
    }
    CHECK_ITERABLE_APPROX(
        get(lapse_times_conformal_factor_minus_one),
        (DataVector{num_points, expected_lapse_psi_minus_one}));
  }
  CHECK_ITERABLE_APPROX(get(conformal_factor_minus_one),
                        (DataVector{num_points, expected_psi_minus_one}));
}

template <Xcts::Equations EnabledEquations>
void test_suite() {
  const auto created = TestHelpers::test_factory_creation<
      elliptic::BoundaryConditions::BoundaryCondition<3>,
      NumericData<EnabledEquations>>(
      "NumericData:\n"
      "  DataFile: " +
      h5_file_name +
      "\n"
      "  Subgroup: element_data\n"
      "  ObservationStep: -1\n"
      "  ExtrapolateIntoExcisions: False\n");
  REQUIRE(dynamic_cast<const NumericData<EnabledEquations>*>(created.get()) !=
          nullptr);
  const auto& boundary_condition =
      dynamic_cast<const NumericData<EnabledEquations>&>(*created);
  {
    INFO("Semantics");
    // Copies and deserialization reload the data from file, because the
    // interpolator holds a move-only Domain.
    test_serialization(boundary_condition);
    test_copy_semantics(boundary_condition);
  }
  {
    INFO("Properties");
    const size_t num_conditions =
        EnabledEquations == Xcts::Equations::Hamiltonian
            ? 1
            : (EnabledEquations == Xcts::Equations::HamiltonianAndLapse ? 2
                                                                        : 5);
    CHECK(boundary_condition.boundary_condition_types() ==
          std::vector<elliptic::BoundaryConditionType>{
              num_conditions, elliptic::BoundaryConditionType::Dirichlet});
  }
  test_apply<EnabledEquations, false>(boundary_condition);
  test_apply<EnabledEquations, true>(boundary_condition);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Xcts.BoundaryConditions.NumericData",
                  "[Unit][Elliptic]") {
  write_volume_file();
  test_suite<Xcts::Equations::Hamiltonian>();
  test_suite<Xcts::Equations::HamiltonianAndLapse>();
  test_suite<Xcts::Equations::HamiltonianLapseAndShift>();
  file_system::rm(h5_file_name, true);
}

}  // namespace Xcts::BoundaryConditions
