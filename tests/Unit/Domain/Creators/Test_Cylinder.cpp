// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <unordered_set>
#include <vector>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"  // IWYU pragma: keep
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Equiangular.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/Wedge2D.hpp"
#include "Domain/Creators/Cylinder.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "Domain/OptionTags.hpp"
#include "Domain/Structure/BlockNeighbor.hpp"  // IWYU pragma: keep
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/MakeArray.hpp"

namespace domain {
namespace {
using BoundaryCondVector = std::vector<DirectionMap<
    3, std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>>>;

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
create_boundary_condition() {
  return std::make_unique<
      TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<3>>(
      Direction<3>::upper_zeta(), 50);
}

std::string boundary_conditions_string() {
  return {
      "  BoundaryCondition:\n"
      "    TestBoundaryCondition:\n"
      "      Direction: upper-zeta\n"
      "      BlockId: 50\n"};
}

auto create_boundary_conditions(const bool periodic_in_z) {
  BoundaryCondVector boundary_conditions_all_blocks{5};
  const auto boundary_condition = create_boundary_condition();
  // z-direction
  for (size_t block_id = 0; not periodic_in_z and block_id < 5; ++block_id) {
    boundary_conditions_all_blocks[block_id][Direction<3>::lower_zeta()] =
        boundary_condition->get_clone();
    boundary_conditions_all_blocks[block_id][Direction<3>::upper_zeta()] =
        boundary_condition->get_clone();
  }
  // radial direction
  for (size_t block_id = 1; block_id < 5; ++block_id) {
    boundary_conditions_all_blocks[block_id][Direction<3>::upper_xi()] =
        boundary_condition->get_clone();
  }
  return boundary_conditions_all_blocks;
}

void test_cylinder_construction(
    const creators::Cylinder& cylinder, const double inner_radius,
    const double outer_radius, const double lower_bound,
    const double upper_bound, const bool is_periodic_in_z,
    const std::array<size_t, 3>& expected_wedge_extents,
    const std::vector<std::array<size_t, 3>>& expected_refinement_level,
    const bool use_equiangular_map,
    const BoundaryCondVector& expected_boundary_conditions = {}) {
  const auto domain = cylinder.create_domain();
  const OrientationMap<3> aligned_orientation{};
  const OrientationMap<3> quarter_turn_ccw(std::array<Direction<3>, 3>{
      {Direction<3>::lower_eta(), Direction<3>::upper_xi(),
       Direction<3>::upper_zeta()}});
  const OrientationMap<3> half_turn(std::array<Direction<3>, 3>{
      {Direction<3>::lower_xi(), Direction<3>::lower_eta(),
       Direction<3>::upper_zeta()}});
  const OrientationMap<3> quarter_turn_cw(std::array<Direction<3>, 3>{
      {Direction<3>::upper_eta(), Direction<3>::lower_xi(),
       Direction<3>::upper_zeta()}});
  std::vector<DirectionMap<3, BlockNeighbor<3>>> expected_block_neighbors{};
  std::vector<std::unordered_set<Direction<3>>> expected_external_boundaries{};
  using TargetFrame = Frame::Inertial;
  std::vector<std::unique_ptr<
      domain::CoordinateMapBase<Frame::Logical, TargetFrame, 3>>>
      coord_maps{};
  if (not is_periodic_in_z) {
    expected_block_neighbors = std::vector<DirectionMap<3, BlockNeighbor<3>>>{
        {{Direction<3>::upper_xi(), {1, aligned_orientation}},
         {Direction<3>::upper_eta(), {2, quarter_turn_ccw}},
         {Direction<3>::lower_xi(), {3, half_turn}},
         {Direction<3>::lower_eta(), {4, quarter_turn_cw}}},
        {{Direction<3>::lower_eta(), {4, aligned_orientation}},
         {Direction<3>::upper_eta(), {2, aligned_orientation}},
         {Direction<3>::lower_xi(), {0, aligned_orientation}}},
        {{Direction<3>::lower_eta(), {1, aligned_orientation}},
         {Direction<3>::upper_eta(), {3, aligned_orientation}},
         {Direction<3>::lower_xi(), {0, quarter_turn_cw}}},
        {{Direction<3>::lower_eta(), {2, aligned_orientation}},
         {Direction<3>::upper_eta(), {4, aligned_orientation}},
         {Direction<3>::lower_xi(), {0, half_turn}}},
        {{Direction<3>::lower_eta(), {3, aligned_orientation}},
         {Direction<3>::upper_eta(), {1, aligned_orientation}},
         {Direction<3>::lower_xi(), {0, quarter_turn_ccw}}}};
    expected_external_boundaries =
        std::vector<std::unordered_set<Direction<3>>>{
            {Direction<3>::upper_zeta(), Direction<3>::lower_zeta()},
            {{Direction<3>::upper_xi(), Direction<3>::upper_zeta(),
              Direction<3>::lower_zeta()}},
            {{Direction<3>::upper_xi(), Direction<3>::upper_zeta(),
              Direction<3>::lower_zeta()}},
            {{Direction<3>::upper_xi(), Direction<3>::upper_zeta(),
              Direction<3>::lower_zeta()}},
            {{Direction<3>::upper_xi(), Direction<3>::upper_zeta(),
              Direction<3>::lower_zeta()}}};
  } else {
    expected_block_neighbors = std::vector<DirectionMap<3, BlockNeighbor<3>>>{
        {{Direction<3>::upper_xi(), {1, aligned_orientation}},
         {Direction<3>::upper_eta(), {2, quarter_turn_ccw}},
         {Direction<3>::lower_xi(), {3, half_turn}},
         {Direction<3>::lower_eta(), {4, quarter_turn_cw}},
         {Direction<3>::lower_zeta(), {0, aligned_orientation}},
         {Direction<3>::upper_zeta(), {0, aligned_orientation}}},
        {{Direction<3>::lower_eta(), {4, aligned_orientation}},
         {Direction<3>::upper_eta(), {2, aligned_orientation}},
         {Direction<3>::lower_xi(), {0, aligned_orientation}},
         {Direction<3>::lower_zeta(), {1, aligned_orientation}},
         {Direction<3>::upper_zeta(), {1, aligned_orientation}}},
        {{Direction<3>::lower_eta(), {1, aligned_orientation}},
         {Direction<3>::upper_eta(), {3, aligned_orientation}},
         {Direction<3>::lower_xi(), {0, quarter_turn_cw}},
         {Direction<3>::lower_zeta(), {2, aligned_orientation}},
         {Direction<3>::upper_zeta(), {2, aligned_orientation}}},
        {{Direction<3>::lower_eta(), {2, aligned_orientation}},
         {Direction<3>::upper_eta(), {4, aligned_orientation}},
         {Direction<3>::lower_xi(), {0, half_turn}},
         {Direction<3>::lower_zeta(), {3, aligned_orientation}},
         {Direction<3>::upper_zeta(), {3, aligned_orientation}}},
        {{Direction<3>::lower_eta(), {3, aligned_orientation}},
         {Direction<3>::upper_eta(), {1, aligned_orientation}},
         {Direction<3>::lower_xi(), {0, quarter_turn_ccw}},
         {Direction<3>::lower_zeta(), {4, aligned_orientation}},
         {Direction<3>::upper_zeta(), {4, aligned_orientation}}}};

    expected_external_boundaries =
        std::vector<std::unordered_set<Direction<3>>>{
            {},
            {{Direction<3>::upper_xi()}},
            {{Direction<3>::upper_xi()}},
            {{Direction<3>::upper_xi()}},
            {{Direction<3>::upper_xi()}}};
  }
  const std::vector<std::array<size_t, 3>>& expected_extents{
      {{expected_wedge_extents[1], expected_wedge_extents[1],
        expected_wedge_extents[2]}},
      expected_wedge_extents,
      expected_wedge_extents,
      expected_wedge_extents,
      expected_wedge_extents};

  CHECK(cylinder.initial_extents() == expected_extents);
  CHECK(cylinder.initial_refinement_levels() == expected_refinement_level);
  using TargetFrame = Frame::Inertial;
  using Affine = CoordinateMaps::Affine;
  using Affine3D = CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  using Equiangular = CoordinateMaps::Equiangular;
  using Equiangular3DPrism =
      CoordinateMaps::ProductOf3Maps<Equiangular, Equiangular, Affine>;
  using Wedge2D = CoordinateMaps::Wedge2D;
  using Wedge3DPrism = CoordinateMaps::ProductOf2Maps<Wedge2D, Affine>;

  if (use_equiangular_map) {
    coord_maps.emplace_back(
        make_coordinate_map_base<Frame::Logical, TargetFrame>(
            Equiangular3DPrism{
                Equiangular(-1.0, 1.0, -1.0 * inner_radius / sqrt(2.0),
                            inner_radius / sqrt(2.0)),
                Equiangular(-1.0, 1.0, -1.0 * inner_radius / sqrt(2.0),
                            inner_radius / sqrt(2.0)),
                Affine{-1.0, 1.0, lower_bound, upper_bound}}));
  } else {
    coord_maps.emplace_back(
        make_coordinate_map_base<Frame::Logical, TargetFrame>(
            Affine3D{Affine(-1.0, 1.0, -1.0 * inner_radius / sqrt(2.0),
                            inner_radius / sqrt(2.0)),
                     Affine(-1.0, 1.0, -1.0 * inner_radius / sqrt(2.0),
                            inner_radius / sqrt(2.0)),
                     Affine{-1.0, 1.0, lower_bound, upper_bound}}));
  }
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{inner_radius, outer_radius, 0.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::upper_xi(), Direction<2>::upper_eta()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, lower_bound, upper_bound}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{inner_radius, outer_radius, 0.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, lower_bound, upper_bound}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{inner_radius, outer_radius, 0.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::lower_xi(), Direction<2>::lower_eta()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, lower_bound, upper_bound}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{inner_radius, outer_radius, 0.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::upper_eta(), Direction<2>::lower_xi()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, lower_bound, upper_bound}}));

  test_domain_construction(domain, expected_block_neighbors,
                           expected_external_boundaries, coord_maps,
                           std::numeric_limits<double>::signaling_NaN(), {}, {},
                           expected_boundary_conditions);

  test_initial_domain(domain, cylinder.initial_refinement_levels());

  Parallel::register_classes_in_list<typename creators::Cylinder::maps_list>();
  test_serialization(domain);
}

void test_cylinder_no_refinement() {
  INFO("Cylinder, no refinement");
  for (const bool with_boundary_conditions : {true, false}) {
    CAPTURE(with_boundary_conditions);
    for (const bool equiangular_map : {true, false}) {
      CAPTURE(equiangular_map);
      for (const bool periodic_in_z : {true, false}) {
        CAPTURE(periodic_in_z);
        const double inner_radius = 1.0;
        const double outer_radius = 2.0;
        const double lower_bound = -2.5;
        const double upper_bound = 5.0;
        const size_t refinement_level = 2;
        const std::array<size_t, 3> grid_points{{4, 4, 3}};

        const BoundaryCondVector expected_boundary_conditions =
            with_boundary_conditions ? create_boundary_conditions(periodic_in_z)
                                     : BoundaryCondVector{};
        std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
            boundary_condition =
                with_boundary_conditions ? create_boundary_condition()
                                         : nullptr;

        const creators::Cylinder cylinder{inner_radius,
                                          outer_radius,
                                          lower_bound,
                                          upper_bound,
                                          periodic_in_z,
                                          refinement_level,
                                          grid_points,
                                          equiangular_map,
                                          {},
                                          {},
                                          std::move(boundary_condition)};
        test_physical_separation(cylinder.create_domain().blocks());
        test_cylinder_construction(
            cylinder, inner_radius, outer_radius, lower_bound, upper_bound,
            periodic_in_z, grid_points, {5, make_array<3>(refinement_level)},
            equiangular_map, expected_boundary_conditions);

        const std::string opt_string{
            "Cylinder:\n"
            "  InnerRadius: 1.0\n"
            "  OuterRadius: 2.0\n"
            "  LowerBound: -2.5\n"
            "  UpperBound: 5.0\n"
            "  IsPeriodicInZ: " +
            std::string{periodic_in_z ? "true" : "false"} +
            "\n"
            "  InitialRefinement: 2\n"
            "  InitialGridPoints: [4,4,3]\n"
            "  UseEquiangularMap: " +
            std::string{equiangular_map ? "true" : "false"} +
            "\n"
            "  RadialPartitioning: []\n"
            "  HeightPartitioning: []\n" +
            std::string{with_boundary_conditions ? boundary_conditions_string()
                                                 : ""}};

        const auto cylinder_factory = [&opt_string,
                                       with_boundary_conditions]() {
          if (with_boundary_conditions) {
            return TestHelpers::test_factory_creation<
                DomainCreator<3>, domain::OptionTags::DomainCreator<3>,
                TestHelpers::domain::BoundaryConditions::
                    MetavariablesWithBoundaryConditions<3>>(opt_string);
          } else {
            return TestHelpers::test_factory_creation<
                DomainCreator<3>, domain::OptionTags::DomainCreator<3>,
                TestHelpers::domain::BoundaryConditions::
                    MetavariablesWithoutBoundaryConditions<3>>(opt_string);
          }
        }();
        test_cylinder_construction(
            dynamic_cast<const creators::Cylinder&>(*cylinder_factory),
            inner_radius, outer_radius, lower_bound, upper_bound, periodic_in_z,
            grid_points, {5, make_array<3>(refinement_level)}, equiangular_map,
            expected_boundary_conditions);

        CHECK_THROWS_WITH(
            creators::Cylinder(
                inner_radius, outer_radius, lower_bound, upper_bound,
                periodic_in_z, refinement_level, grid_points, equiangular_map,
                {}, {},
                std::make_unique<TestHelpers::domain::BoundaryConditions::
                                     TestPeriodicBoundaryCondition<3>>(),
                Options::Context{false, {}, 1, 1}),
            Catch::Matchers::Contains(
                "Periodic boundary conditions are not supported in the radial "
                "direction. If you need periodic boundary conditions along the "
                "axis of symmetry, use the is_periodic_in_z option."));
        if (with_boundary_conditions) {
          CHECK_THROWS_WITH(
              creators::Cylinder(inner_radius, outer_radius, lower_bound,
                                 upper_bound, periodic_in_z, refinement_level,
                                 grid_points, equiangular_map, {1.0}, {},
                                 create_boundary_condition(),
                                 Options::Context{false, {}, 1, 1}),
              Catch::Matchers::Contains(
                  "Currently do not support specifying boundary conditions and "
                  "multiple radial partitionings. Support can be added if "
                  "desired."));
          CHECK_THROWS_WITH(
              creators::Cylinder(inner_radius, outer_radius, lower_bound,
                                 upper_bound, periodic_in_z, refinement_level,
                                 grid_points, equiangular_map, {}, {1.4},
                                 create_boundary_condition(),
                                 Options::Context{false, {}, 1, 1}),
              Catch::Matchers::Contains(
                  "Currently do not support specifying boundary conditions and "
                  "multiple height partitionings. The domain creator code to "
                  "support this is written but untested. To enable, please add "
                  "tests."));
          CHECK_THROWS_WITH(
              creators::Cylinder(
                  inner_radius, outer_radius, lower_bound, upper_bound,
                  periodic_in_z, refinement_level, grid_points, equiangular_map,
                  {}, {},
                  std::make_unique<TestHelpers::domain::BoundaryConditions::
                                       TestNoneBoundaryCondition<3>>(),
                  Options::Context{false, {}, 1, 1}),
              Catch::Matchers::Contains(
                  "None boundary condition is not supported. If you would like "
                  "an outflow boundary condition, you must use that."));
        }
      }
    }
  }
}

void test_refined_cylinder_boundaries(const bool use_equiangular_map) {
  INFO("Refined Cylinder boundaries");
  CAPTURE(use_equiangular_map);
  // definition of an arbitrary refined cylinder
  const double inner_radius = 0.3;
  const double outer_radius = 1.0;
  const double lower_bound = -1.5;
  const double upper_bound = 5.0;
  const std::vector<double> radial_partitioning = {0.7};
  const std::vector<double> height_partitioning = {1.5};
  const size_t refinement_level = 2;
  const std::array<size_t, 3> expected_wedge_extents{{5, 4, 3}};
  const bool is_periodic_in_z{false};
  const std::vector<std::array<size_t, 3>> expected_refinement_level{
      (1 + 4 * (1 + radial_partitioning.size())) *
          (1 + height_partitioning.size()),
      make_array<3>(refinement_level)};
  const creators::Cylinder refined_cylinder{inner_radius,
                                            outer_radius,
                                            lower_bound,
                                            upper_bound,
                                            is_periodic_in_z,
                                            refinement_level,
                                            expected_wedge_extents,
                                            use_equiangular_map,
                                            radial_partitioning,
                                            height_partitioning};
  test_physical_separation(refined_cylinder.create_domain().blocks());

  const auto domain = refined_cylinder.create_domain();
  const OrientationMap<3> aligned_orientation{};
  const OrientationMap<3> quarter_turn_ccw(std::array<Direction<3>, 3>{
      {Direction<3>::lower_eta(), Direction<3>::upper_xi(),
       Direction<3>::upper_zeta()}});
  const OrientationMap<3> half_turn(std::array<Direction<3>, 3>{
      {Direction<3>::lower_xi(), Direction<3>::lower_eta(),
       Direction<3>::upper_zeta()}});
  const OrientationMap<3> quarter_turn_cw(std::array<Direction<3>, 3>{
      {Direction<3>::upper_eta(), Direction<3>::lower_xi(),
       Direction<3>::upper_zeta()}});
  std::vector<DirectionMap<3, BlockNeighbor<3>>> expected_block_neighbors{};
  std::vector<std::unordered_set<Direction<3>>> expected_external_boundaries{};
  using TargetFrame = Frame::Inertial;
  std::vector<std::unique_ptr<
      domain::CoordinateMapBase<Frame::Logical, TargetFrame, 3>>>
      coord_maps{};
  // This specific domain consists of two stacked discs with 9 blocks each.
  expected_block_neighbors = std::vector<DirectionMap<3, BlockNeighbor<3>>>{
      // Block 0 - layer 0 center 0
      {{Direction<3>::upper_xi(), {1, aligned_orientation}},
       {Direction<3>::upper_eta(), {2, quarter_turn_ccw}},
       {Direction<3>::lower_xi(), {3, half_turn}},
       {Direction<3>::lower_eta(), {4, quarter_turn_cw}},
       {Direction<3>::upper_zeta(), {0 + 9, aligned_orientation}}},
      // Block 1 - layer 0 east 1
      {{Direction<3>::lower_eta(), {4, aligned_orientation}},
       {Direction<3>::upper_eta(), {2, aligned_orientation}},
       {Direction<3>::lower_xi(), {0, aligned_orientation}},
       {Direction<3>::upper_xi(), {1 + 4, aligned_orientation}},
       {Direction<3>::upper_zeta(), {1 + 9, aligned_orientation}}},
      // Block 2 - layer 0 north 2
      {{Direction<3>::lower_eta(), {1, aligned_orientation}},
       {Direction<3>::upper_eta(), {3, aligned_orientation}},
       {Direction<3>::lower_xi(), {0, quarter_turn_cw}},
       {Direction<3>::upper_xi(), {2 + 4, aligned_orientation}},
       {Direction<3>::upper_zeta(), {2 + 9, aligned_orientation}}},
      // Block 3 - layer 0 west 3
      {{Direction<3>::lower_eta(), {2, aligned_orientation}},
       {Direction<3>::upper_eta(), {4, aligned_orientation}},
       {Direction<3>::lower_xi(), {0, half_turn}},
       {Direction<3>::upper_xi(), {3 + 4, aligned_orientation}},
       {Direction<3>::upper_zeta(), {3 + 9, aligned_orientation}}},
      // Block 4 - layer 0 south 4
      {{Direction<3>::lower_eta(), {3, aligned_orientation}},
       {Direction<3>::upper_eta(), {1, aligned_orientation}},
       {Direction<3>::lower_xi(), {0, quarter_turn_ccw}},
       {Direction<3>::upper_xi(), {4 + 4, aligned_orientation}},
       {Direction<3>::upper_zeta(), {4 + 9, aligned_orientation}}},
      // Block 5 - layer 0 east 5
      {{Direction<3>::lower_eta(), {8, aligned_orientation}},
       {Direction<3>::upper_eta(), {6, aligned_orientation}},
       {Direction<3>::lower_xi(), {5 - 4, aligned_orientation}},
       {Direction<3>::upper_zeta(), {5 + 9, aligned_orientation}}},
      // Block 6 - layer 0 north 6
      {{Direction<3>::lower_eta(), {5, aligned_orientation}},
       {Direction<3>::upper_eta(), {7, aligned_orientation}},
       {Direction<3>::lower_xi(), {6 - 4, aligned_orientation}},
       {Direction<3>::upper_zeta(), {6 + 9, aligned_orientation}}},
      // Block 7 - layer 0 west 7
      {{Direction<3>::lower_eta(), {6, aligned_orientation}},
       {Direction<3>::upper_eta(), {8, aligned_orientation}},
       {Direction<3>::lower_xi(), {7 - 4, aligned_orientation}},
       {Direction<3>::upper_zeta(), {7 + 9, aligned_orientation}}},
      // Block 8 - layer 0 south 8
      {{Direction<3>::lower_eta(), {7, aligned_orientation}},
       {Direction<3>::upper_eta(), {5, aligned_orientation}},
       {Direction<3>::lower_xi(), {8 - 4, aligned_orientation}},
       {Direction<3>::upper_zeta(), {8 + 9, aligned_orientation}}},
      // Block 9 - layer 1 center 0
      {{Direction<3>::upper_xi(), {1 + 9, aligned_orientation}},
       {Direction<3>::upper_eta(), {2 + 9, quarter_turn_ccw}},
       {Direction<3>::lower_xi(), {3 + 9, half_turn}},
       {Direction<3>::lower_eta(), {4 + 9, quarter_turn_cw}},
       {Direction<3>::lower_zeta(), {9 - 9, aligned_orientation}}},
      // Block 10 - layer 1 east 1
      {{Direction<3>::lower_eta(), {4 + 9, aligned_orientation}},
       {Direction<3>::upper_eta(), {2 + 9, aligned_orientation}},
       {Direction<3>::lower_xi(), {0 + 9, aligned_orientation}},
       {Direction<3>::upper_xi(), {1 + 4 + 9, aligned_orientation}},
       {Direction<3>::lower_zeta(), {10 - 9, aligned_orientation}}},
      // Block 11 - layer 1 north 2
      {{Direction<3>::lower_eta(), {1 + 9, aligned_orientation}},
       {Direction<3>::upper_eta(), {3 + 9, aligned_orientation}},
       {Direction<3>::lower_xi(), {0 + 9, quarter_turn_cw}},
       {Direction<3>::upper_xi(), {2 + 4 + 9, aligned_orientation}},
       {Direction<3>::lower_zeta(), {11 - 9, aligned_orientation}}},
      // Block 12 - layer 1 west 3
      {{Direction<3>::lower_eta(), {2 + 9, aligned_orientation}},
       {Direction<3>::upper_eta(), {4 + 9, aligned_orientation}},
       {Direction<3>::lower_xi(), {0 + 9, half_turn}},
       {Direction<3>::upper_xi(), {3 + 4 + 9, aligned_orientation}},
       {Direction<3>::lower_zeta(), {12 - 9, aligned_orientation}}},
      // Block 13 - layer 1 south 4
      {{Direction<3>::lower_eta(), {3 + 9, aligned_orientation}},
       {Direction<3>::upper_eta(), {1 + 9, aligned_orientation}},
       {Direction<3>::lower_xi(), {0 + 9, quarter_turn_ccw}},
       {Direction<3>::upper_xi(), {4 + 4 + 9, aligned_orientation}},
       {Direction<3>::lower_zeta(), {13 - 9, aligned_orientation}}},
      // Block 14 - layer 1 east 5
      {{Direction<3>::lower_eta(), {8 + 9, aligned_orientation}},
       {Direction<3>::upper_eta(), {6 + 9, aligned_orientation}},
       {Direction<3>::lower_xi(), {5 - 4 + 9, aligned_orientation}},
       {Direction<3>::lower_zeta(), {14 - 9, aligned_orientation}}},
      // Block 15 - layer 1 north 6
      {{Direction<3>::lower_eta(), {5 + 9, aligned_orientation}},
       {Direction<3>::upper_eta(), {7 + 9, aligned_orientation}},
       {Direction<3>::lower_xi(), {6 - 4 + 9, aligned_orientation}},
       {Direction<3>::lower_zeta(), {15 - 9, aligned_orientation}}},
      // Block 16 - layer 1 west 7
      {{Direction<3>::lower_eta(), {6 + 9, aligned_orientation}},
       {Direction<3>::upper_eta(), {8 + 9, aligned_orientation}},
       {Direction<3>::lower_xi(), {7 - 4 + 9, aligned_orientation}},
       {Direction<3>::lower_zeta(), {16 - 9, aligned_orientation}}},
      // Block 17 - layer 1 south 8
      {{Direction<3>::lower_eta(), {7 + 9, aligned_orientation}},
       {Direction<3>::upper_eta(), {5 + 9, aligned_orientation}},
       {Direction<3>::lower_xi(), {8 - 4 + 9, aligned_orientation}},
       {Direction<3>::lower_zeta(), {17 - 9, aligned_orientation}}}};

  expected_external_boundaries = std::vector<std::unordered_set<Direction<3>>>{
      {Direction<3>::lower_zeta()},
      {Direction<3>::lower_zeta()},
      {Direction<3>::lower_zeta()},
      {Direction<3>::lower_zeta()},
      {Direction<3>::lower_zeta()},
      {Direction<3>::upper_xi(), Direction<3>::lower_zeta()},
      {Direction<3>::upper_xi(), Direction<3>::lower_zeta()},
      {Direction<3>::upper_xi(), Direction<3>::lower_zeta()},
      {Direction<3>::upper_xi(), Direction<3>::lower_zeta()},
      {Direction<3>::upper_zeta()},
      {Direction<3>::upper_zeta()},
      {Direction<3>::upper_zeta()},
      {Direction<3>::upper_zeta()},
      {Direction<3>::upper_zeta()},
      {Direction<3>::upper_xi(), Direction<3>::upper_zeta()},
      {Direction<3>::upper_xi(), Direction<3>::upper_zeta()},
      {Direction<3>::upper_xi(), Direction<3>::upper_zeta()},
      {Direction<3>::upper_xi(), Direction<3>::upper_zeta()}};

  const std::vector<std::array<size_t, 3>>& expected_extents{
      {{expected_wedge_extents[1], expected_wedge_extents[1],
        expected_wedge_extents[2]}},
      expected_wedge_extents,
      expected_wedge_extents,
      expected_wedge_extents,
      expected_wedge_extents,
      expected_wedge_extents,
      expected_wedge_extents,
      expected_wedge_extents,
      expected_wedge_extents,
      {{expected_wedge_extents[1], expected_wedge_extents[1],
        expected_wedge_extents[2]}},
      expected_wedge_extents,
      expected_wedge_extents,
      expected_wedge_extents,
      expected_wedge_extents,
      expected_wedge_extents,
      expected_wedge_extents,
      expected_wedge_extents,
      expected_wedge_extents};

  CHECK(refined_cylinder.initial_extents() == expected_extents);
  CHECK(refined_cylinder.initial_refinement_levels() ==
        expected_refinement_level);
  using TargetFrame = Frame::Inertial;
  using Affine = CoordinateMaps::Affine;
  using Affine3D = CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  using Equiangular = CoordinateMaps::Equiangular;
  using Equiangular3DPrism =
      CoordinateMaps::ProductOf3Maps<Equiangular, Equiangular, Affine>;
  using Wedge2D = CoordinateMaps::Wedge2D;
  using Wedge3DPrism = CoordinateMaps::ProductOf2Maps<Wedge2D, Affine>;
  // in this section, the coord_map is filled same as with the cylinder;
  // first the first shell with radial boundaries
  // (inner_radius, radial_partitioning.at(0)) and circularity changing from 0
  // to 1 is generated, secondly a further shell with radial boundaries
  // (radial_partitioning.at(0), outer_radius) and uniform circularity is added.
  // this is then repeated for (lower_bound, height_partitioning.at(0)) and
  // (height_partitioning.at(0), upper_bound)
  if (use_equiangular_map) {
    coord_maps.emplace_back(
        make_coordinate_map_base<Frame::Logical, TargetFrame>(
            Equiangular3DPrism{
                Equiangular(-1.0, 1.0, -1.0 * inner_radius / sqrt(2.0),
                            inner_radius / sqrt(2.0)),
                Equiangular(-1.0, 1.0, -1.0 * inner_radius / sqrt(2.0),
                            inner_radius / sqrt(2.0)),
                Affine{-1.0, 1.0, lower_bound, height_partitioning.at(0)}}));
  } else {
    coord_maps.emplace_back(
        make_coordinate_map_base<Frame::Logical, TargetFrame>(Affine3D{
            Affine(-1.0, 1.0, -1.0 * inner_radius / sqrt(2.0),
                   inner_radius / sqrt(2.0)),
            Affine(-1.0, 1.0, -1.0 * inner_radius / sqrt(2.0),
                   inner_radius / sqrt(2.0)),
            Affine{-1.0, 1.0, lower_bound, height_partitioning.at(0)}}));
  }
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{inner_radius, radial_partitioning.at(0), 0.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::upper_xi(), Direction<2>::upper_eta()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, lower_bound, height_partitioning.at(0)}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{inner_radius, radial_partitioning.at(0), 0.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, lower_bound, height_partitioning.at(0)}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{inner_radius, radial_partitioning.at(0), 0.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::lower_xi(), Direction<2>::lower_eta()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, lower_bound, height_partitioning.at(0)}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{inner_radius, radial_partitioning.at(0), 0.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::upper_eta(), Direction<2>::lower_xi()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, lower_bound, height_partitioning.at(0)}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{radial_partitioning.at(0), outer_radius, 1.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::upper_xi(), Direction<2>::upper_eta()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, lower_bound, height_partitioning.at(0)}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{radial_partitioning.at(0), outer_radius, 1.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, lower_bound, height_partitioning.at(0)}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{radial_partitioning.at(0), outer_radius, 1.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::lower_xi(), Direction<2>::lower_eta()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, lower_bound, height_partitioning.at(0)}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{radial_partitioning.at(0), outer_radius, 1.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::upper_eta(), Direction<2>::lower_xi()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, lower_bound, height_partitioning.at(0)}}));
  if (use_equiangular_map) {
    coord_maps.emplace_back(
        make_coordinate_map_base<Frame::Logical, TargetFrame>(
            Equiangular3DPrism{
                Equiangular(-1.0, 1.0, -1.0 * inner_radius / sqrt(2.0),
                            inner_radius / sqrt(2.0)),
                Equiangular(-1.0, 1.0, -1.0 * inner_radius / sqrt(2.0),
                            inner_radius / sqrt(2.0)),
                Affine{-1.0, 1.0, height_partitioning.at(0), upper_bound}}));
  } else {
    coord_maps.emplace_back(
        make_coordinate_map_base<Frame::Logical, TargetFrame>(Affine3D{
            Affine(-1.0, 1.0, -1.0 * inner_radius / sqrt(2.0),
                   inner_radius / sqrt(2.0)),
            Affine(-1.0, 1.0, -1.0 * inner_radius / sqrt(2.0),
                   inner_radius / sqrt(2.0)),
            Affine{-1.0, 1.0, height_partitioning.at(0), upper_bound}}));
  }
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{inner_radius, radial_partitioning.at(0), 0.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::upper_xi(), Direction<2>::upper_eta()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, height_partitioning.at(0), upper_bound}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{inner_radius, radial_partitioning.at(0), 0.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, height_partitioning.at(0), upper_bound}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{inner_radius, radial_partitioning.at(0), 0.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::lower_xi(), Direction<2>::lower_eta()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, height_partitioning.at(0), upper_bound}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{inner_radius, radial_partitioning.at(0), 0.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::upper_eta(), Direction<2>::lower_xi()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, height_partitioning.at(0), upper_bound}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{radial_partitioning.at(0), outer_radius, 1.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::upper_xi(), Direction<2>::upper_eta()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, height_partitioning.at(0), upper_bound}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{radial_partitioning.at(0), outer_radius, 1.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, height_partitioning.at(0), upper_bound}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{radial_partitioning.at(0), outer_radius, 1.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::lower_xi(), Direction<2>::lower_eta()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, height_partitioning.at(0), upper_bound}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{radial_partitioning.at(0), outer_radius, 1.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::upper_eta(), Direction<2>::lower_xi()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, height_partitioning.at(0), upper_bound}}));

  test_domain_construction(domain, expected_block_neighbors,
                           expected_external_boundaries, coord_maps);

  test_initial_domain(domain, refined_cylinder.initial_refinement_levels());

  Parallel::register_classes_in_list<typename creators::Cylinder::maps_list>();
  test_serialization(domain);
}

void test_refined_cylinder_periodic_boundaries(const bool use_equiangular_map) {
  INFO("Refined Cylinder periodic boundaries");
  CAPTURE(use_equiangular_map);
  // definition of an arbitrary refined cylinder
  const double inner_radius = 0.3;
  const double outer_radius = 1.0;
  const double lower_bound = -1.5;
  const double upper_bound = 5.0;
  const std::vector<double> radial_partitioning = {0.7};
  const std::vector<double> height_partitioning = {1.5};
  const size_t refinement_level = 2;
  const std::array<size_t, 3> expected_wedge_extents{{5, 4, 3}};
  const bool is_periodic_in_z{true};
  const std::vector<std::array<size_t, 3>> expected_refinement_level{
      (1 + 4 * (1 + radial_partitioning.size())) *
          (1 + height_partitioning.size()),
      make_array<3>(refinement_level)};
  const creators::Cylinder refined_cylinder{inner_radius,
                                            outer_radius,
                                            lower_bound,
                                            upper_bound,
                                            is_periodic_in_z,
                                            refinement_level,
                                            expected_wedge_extents,
                                            use_equiangular_map,
                                            radial_partitioning,
                                            height_partitioning};

  const auto domain = refined_cylinder.create_domain();
  const OrientationMap<3> aligned_orientation{};
  const OrientationMap<3> quarter_turn_ccw(std::array<Direction<3>, 3>{
      {Direction<3>::lower_eta(), Direction<3>::upper_xi(),
       Direction<3>::upper_zeta()}});
  const OrientationMap<3> half_turn(std::array<Direction<3>, 3>{
      {Direction<3>::lower_xi(), Direction<3>::lower_eta(),
       Direction<3>::upper_zeta()}});
  const OrientationMap<3> quarter_turn_cw(std::array<Direction<3>, 3>{
      {Direction<3>::upper_eta(), Direction<3>::lower_xi(),
       Direction<3>::upper_zeta()}});
  std::vector<DirectionMap<3, BlockNeighbor<3>>> expected_block_neighbors{};
  std::vector<std::unordered_set<Direction<3>>> expected_external_boundaries{};
  using TargetFrame = Frame::Inertial;
  std::vector<std::unique_ptr<
      domain::CoordinateMapBase<Frame::Logical, TargetFrame, 3>>>
      coord_maps{};
  expected_block_neighbors = std::vector<DirectionMap<3, BlockNeighbor<3>>>{
      // Block 0 - layer 0 center 0
      {{Direction<3>::upper_xi(), {1, aligned_orientation}},
       {Direction<3>::upper_eta(), {2, quarter_turn_ccw}},
       {Direction<3>::lower_xi(), {3, half_turn}},
       {Direction<3>::lower_eta(), {4, quarter_turn_cw}},
       {Direction<3>::lower_zeta(), {0 + 9, aligned_orientation}},
       {Direction<3>::upper_zeta(), {0 + 9, aligned_orientation}}},
      // Block 1 - layer 0 east 1
      {{Direction<3>::lower_eta(), {4, aligned_orientation}},
       {Direction<3>::upper_eta(), {2, aligned_orientation}},
       {Direction<3>::lower_xi(), {0, aligned_orientation}},
       {Direction<3>::upper_xi(), {1 + 4, aligned_orientation}},
       {Direction<3>::lower_zeta(), {1 + 9, aligned_orientation}},
       {Direction<3>::upper_zeta(), {1 + 9, aligned_orientation}}},
      // Block 2 - layer 0 north 2
      {{Direction<3>::lower_eta(), {1, aligned_orientation}},
       {Direction<3>::upper_eta(), {3, aligned_orientation}},
       {Direction<3>::lower_xi(), {0, quarter_turn_cw}},
       {Direction<3>::upper_xi(), {2 + 4, aligned_orientation}},
       {Direction<3>::lower_zeta(), {2 + 9, aligned_orientation}},
       {Direction<3>::upper_zeta(), {2 + 9, aligned_orientation}}},
      // Block 3 - layer 0 west 3
      {{Direction<3>::lower_eta(), {2, aligned_orientation}},
       {Direction<3>::upper_eta(), {4, aligned_orientation}},
       {Direction<3>::lower_xi(), {0, half_turn}},
       {Direction<3>::upper_xi(), {3 + 4, aligned_orientation}},
       {Direction<3>::lower_zeta(), {3 + 9, aligned_orientation}},
       {Direction<3>::upper_zeta(), {3 + 9, aligned_orientation}}},
      // Block 4 - layer 0 south 4
      {{Direction<3>::lower_eta(), {3, aligned_orientation}},
       {Direction<3>::upper_eta(), {1, aligned_orientation}},
       {Direction<3>::lower_xi(), {0, quarter_turn_ccw}},
       {Direction<3>::upper_xi(), {4 + 4, aligned_orientation}},
       {Direction<3>::lower_zeta(), {4 + 9, aligned_orientation}},
       {Direction<3>::upper_zeta(), {4 + 9, aligned_orientation}}},
      // Block 5 - layer 0 east 5
      {{Direction<3>::lower_eta(), {8, aligned_orientation}},
       {Direction<3>::upper_eta(), {6, aligned_orientation}},
       {Direction<3>::lower_xi(), {5 - 4, aligned_orientation}},
       {Direction<3>::lower_zeta(), {5 + 9, aligned_orientation}},
       {Direction<3>::upper_zeta(), {5 + 9, aligned_orientation}}},
      // Block 6 - layer 0 north 6
      {{Direction<3>::lower_eta(), {5, aligned_orientation}},
       {Direction<3>::upper_eta(), {7, aligned_orientation}},
       {Direction<3>::lower_xi(), {6 - 4, aligned_orientation}},
       {Direction<3>::lower_zeta(), {6 + 9, aligned_orientation}},
       {Direction<3>::upper_zeta(), {6 + 9, aligned_orientation}}},
      // Block 7 - layer 0 west 7
      {{Direction<3>::lower_eta(), {6, aligned_orientation}},
       {Direction<3>::upper_eta(), {8, aligned_orientation}},
       {Direction<3>::lower_xi(), {7 - 4, aligned_orientation}},
       {Direction<3>::lower_zeta(), {7 + 9, aligned_orientation}},
       {Direction<3>::upper_zeta(), {7 + 9, aligned_orientation}}},
      // Block 8 - layer 0 south 8
      {{Direction<3>::lower_eta(), {7, aligned_orientation}},
       {Direction<3>::upper_eta(), {5, aligned_orientation}},
       {Direction<3>::lower_xi(), {8 - 4, aligned_orientation}},
       {Direction<3>::lower_zeta(), {8 + 9, aligned_orientation}},
       {Direction<3>::upper_zeta(), {8 + 9, aligned_orientation}}},
      // Block 9 - layer 1 center 0
      {{Direction<3>::upper_xi(), {1 + 9, aligned_orientation}},
       {Direction<3>::upper_eta(), {2 + 9, quarter_turn_ccw}},
       {Direction<3>::lower_xi(), {3 + 9, half_turn}},
       {Direction<3>::lower_eta(), {4 + 9, quarter_turn_cw}},
       {Direction<3>::lower_zeta(), {9 - 9, aligned_orientation}},
       {Direction<3>::upper_zeta(), {9 - 9, aligned_orientation}}},
      // Block 10 - layer 1 east 1
      {{Direction<3>::lower_eta(), {4 + 9, aligned_orientation}},
       {Direction<3>::upper_eta(), {2 + 9, aligned_orientation}},
       {Direction<3>::lower_xi(), {0 + 9, aligned_orientation}},
       {Direction<3>::upper_xi(), {1 + 4 + 9, aligned_orientation}},
       {Direction<3>::lower_zeta(), {10 - 9, aligned_orientation}},
       {Direction<3>::upper_zeta(), {10 - 9, aligned_orientation}}},
      // Block 11 - layer 1 north 2
      {{Direction<3>::lower_eta(), {1 + 9, aligned_orientation}},
       {Direction<3>::upper_eta(), {3 + 9, aligned_orientation}},
       {Direction<3>::lower_xi(), {0 + 9, quarter_turn_cw}},
       {Direction<3>::upper_xi(), {2 + 4 + 9, aligned_orientation}},
       {Direction<3>::lower_zeta(), {11 - 9, aligned_orientation}},
       {Direction<3>::upper_zeta(), {11 - 9, aligned_orientation}}},
      // Block 12 - layer 1 west 3
      {{Direction<3>::lower_eta(), {2 + 9, aligned_orientation}},
       {Direction<3>::upper_eta(), {4 + 9, aligned_orientation}},
       {Direction<3>::lower_xi(), {0 + 9, half_turn}},
       {Direction<3>::upper_xi(), {3 + 4 + 9, aligned_orientation}},
       {Direction<3>::lower_zeta(), {12 - 9, aligned_orientation}},
       {Direction<3>::upper_zeta(), {12 - 9, aligned_orientation}}},
      // Block 13 - layer 1 south 4
      {{Direction<3>::lower_eta(), {3 + 9, aligned_orientation}},
       {Direction<3>::upper_eta(), {1 + 9, aligned_orientation}},
       {Direction<3>::lower_xi(), {0 + 9, quarter_turn_ccw}},
       {Direction<3>::upper_xi(), {4 + 4 + 9, aligned_orientation}},
       {Direction<3>::lower_zeta(), {13 - 9, aligned_orientation}},
       {Direction<3>::upper_zeta(), {13 - 9, aligned_orientation}}},
      // Block 14 - layer 1 east 5
      {{Direction<3>::lower_eta(), {8 + 9, aligned_orientation}},
       {Direction<3>::upper_eta(), {6 + 9, aligned_orientation}},
       {Direction<3>::lower_xi(), {5 - 4 + 9, aligned_orientation}},
       {Direction<3>::lower_zeta(), {14 - 9, aligned_orientation}},
       {Direction<3>::upper_zeta(), {14 - 9, aligned_orientation}}},
      // Block 15 - layer 1 north 6
      {{Direction<3>::lower_eta(), {5 + 9, aligned_orientation}},
       {Direction<3>::upper_eta(), {7 + 9, aligned_orientation}},
       {Direction<3>::lower_xi(), {6 - 4 + 9, aligned_orientation}},
       {Direction<3>::lower_zeta(), {15 - 9, aligned_orientation}},
       {Direction<3>::upper_zeta(), {15 - 9, aligned_orientation}}},
      // Block 16 - layer 1 west 7
      {{Direction<3>::lower_eta(), {6 + 9, aligned_orientation}},
       {Direction<3>::upper_eta(), {8 + 9, aligned_orientation}},
       {Direction<3>::lower_xi(), {7 - 4 + 9, aligned_orientation}},
       {Direction<3>::lower_zeta(), {16 - 9, aligned_orientation}},
       {Direction<3>::upper_zeta(), {16 - 9, aligned_orientation}}},
      // Block 17 - layer 1 south 8
      {{Direction<3>::lower_eta(), {7 + 9, aligned_orientation}},
       {Direction<3>::upper_eta(), {5 + 9, aligned_orientation}},
       {Direction<3>::lower_xi(), {8 - 4 + 9, aligned_orientation}},
       {Direction<3>::lower_zeta(), {17 - 9, aligned_orientation}},
       {Direction<3>::upper_zeta(), {17 - 9, aligned_orientation}}}};

  expected_external_boundaries =
      std::vector<std::unordered_set<Direction<3>>>{{},
                                                    {},
                                                    {},
                                                    {},
                                                    {},
                                                    {Direction<3>::upper_xi()},
                                                    {Direction<3>::upper_xi()},
                                                    {Direction<3>::upper_xi()},
                                                    {Direction<3>::upper_xi()},
                                                    {},
                                                    {},
                                                    {},
                                                    {},
                                                    {},
                                                    {Direction<3>::upper_xi()},
                                                    {Direction<3>::upper_xi()},
                                                    {Direction<3>::upper_xi()},
                                                    {Direction<3>::upper_xi()}};

  const std::vector<std::array<size_t, 3>>& expected_extents{
      {{expected_wedge_extents[1], expected_wedge_extents[1],
        expected_wedge_extents[2]}},
      expected_wedge_extents,
      expected_wedge_extents,
      expected_wedge_extents,
      expected_wedge_extents,
      expected_wedge_extents,
      expected_wedge_extents,
      expected_wedge_extents,
      expected_wedge_extents,
      {{expected_wedge_extents[1], expected_wedge_extents[1],
        expected_wedge_extents[2]}},
      expected_wedge_extents,
      expected_wedge_extents,
      expected_wedge_extents,
      expected_wedge_extents,
      expected_wedge_extents,
      expected_wedge_extents,
      expected_wedge_extents,
      expected_wedge_extents};

  CHECK(refined_cylinder.initial_extents() == expected_extents);
  CHECK(refined_cylinder.initial_refinement_levels() ==
        expected_refinement_level);
  using TargetFrame = Frame::Inertial;
  using Affine = CoordinateMaps::Affine;
  using Affine3D = CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  using Equiangular = CoordinateMaps::Equiangular;
  using Equiangular3DPrism =
      CoordinateMaps::ProductOf3Maps<Equiangular, Equiangular, Affine>;
  using Wedge2D = CoordinateMaps::Wedge2D;
  using Wedge3DPrism = CoordinateMaps::ProductOf2Maps<Wedge2D, Affine>;
  // in this section, the coord_map is filled same as with the cylinder;
  // first the first shell with radial boundaries
  // (inner_radius, radial_partitioning.at(0)) and circularity changing from 0
  // to 1 is generated, secondly a further shell with radial boundaries
  // (radial_partitioning.at(0), outer_radius) and uniform circularity is added.
  // this is then repeated for (lower_bound, height_partitioning.at(0)) and
  // (height_partitioning.at(0), upper_bound)
  if (use_equiangular_map) {
    coord_maps.emplace_back(
        make_coordinate_map_base<Frame::Logical, TargetFrame>(
            Equiangular3DPrism{
                Equiangular(-1.0, 1.0, -1.0 * inner_radius / sqrt(2.0),
                            inner_radius / sqrt(2.0)),
                Equiangular(-1.0, 1.0, -1.0 * inner_radius / sqrt(2.0),
                            inner_radius / sqrt(2.0)),
                Affine{-1.0, 1.0, lower_bound, height_partitioning.at(0)}}));
  } else {
    coord_maps.emplace_back(
        make_coordinate_map_base<Frame::Logical, TargetFrame>(Affine3D{
            Affine(-1.0, 1.0, -1.0 * inner_radius / sqrt(2.0),
                   inner_radius / sqrt(2.0)),
            Affine(-1.0, 1.0, -1.0 * inner_radius / sqrt(2.0),
                   inner_radius / sqrt(2.0)),
            Affine{-1.0, 1.0, lower_bound, height_partitioning.at(0)}}));
  }
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{inner_radius, radial_partitioning.at(0), 0.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::upper_xi(), Direction<2>::upper_eta()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, lower_bound, height_partitioning.at(0)}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{inner_radius, radial_partitioning.at(0), 0.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, lower_bound, height_partitioning.at(0)}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{inner_radius, radial_partitioning.at(0), 0.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::lower_xi(), Direction<2>::lower_eta()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, lower_bound, height_partitioning.at(0)}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{inner_radius, radial_partitioning.at(0), 0.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::upper_eta(), Direction<2>::lower_xi()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, lower_bound, height_partitioning.at(0)}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{radial_partitioning.at(0), outer_radius, 1.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::upper_xi(), Direction<2>::upper_eta()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, lower_bound, height_partitioning.at(0)}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{radial_partitioning.at(0), outer_radius, 1.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, lower_bound, height_partitioning.at(0)}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{radial_partitioning.at(0), outer_radius, 1.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::lower_xi(), Direction<2>::lower_eta()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, lower_bound, height_partitioning.at(0)}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{radial_partitioning.at(0), outer_radius, 1.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::upper_eta(), Direction<2>::lower_xi()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, lower_bound, height_partitioning.at(0)}}));
  if (use_equiangular_map) {
    coord_maps.emplace_back(
        make_coordinate_map_base<Frame::Logical, TargetFrame>(
            Equiangular3DPrism{
                Equiangular(-1.0, 1.0, -1.0 * inner_radius / sqrt(2.0),
                            inner_radius / sqrt(2.0)),
                Equiangular(-1.0, 1.0, -1.0 * inner_radius / sqrt(2.0),
                            inner_radius / sqrt(2.0)),
                Affine{-1.0, 1.0, height_partitioning.at(0), upper_bound}}));
  } else {
    coord_maps.emplace_back(
        make_coordinate_map_base<Frame::Logical, TargetFrame>(Affine3D{
            Affine(-1.0, 1.0, -1.0 * inner_radius / sqrt(2.0),
                   inner_radius / sqrt(2.0)),
            Affine(-1.0, 1.0, -1.0 * inner_radius / sqrt(2.0),
                   inner_radius / sqrt(2.0)),
            Affine{-1.0, 1.0, height_partitioning.at(0), upper_bound}}));
  }
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{inner_radius, radial_partitioning.at(0), 0.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::upper_xi(), Direction<2>::upper_eta()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, height_partitioning.at(0), upper_bound}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{inner_radius, radial_partitioning.at(0), 0.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, height_partitioning.at(0), upper_bound}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{inner_radius, radial_partitioning.at(0), 0.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::lower_xi(), Direction<2>::lower_eta()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, height_partitioning.at(0), upper_bound}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{inner_radius, radial_partitioning.at(0), 0.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::upper_eta(), Direction<2>::lower_xi()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, height_partitioning.at(0), upper_bound}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{radial_partitioning.at(0), outer_radius, 1.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::upper_xi(), Direction<2>::upper_eta()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, height_partitioning.at(0), upper_bound}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{radial_partitioning.at(0), outer_radius, 1.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, height_partitioning.at(0), upper_bound}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{radial_partitioning.at(0), outer_radius, 1.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::lower_xi(), Direction<2>::lower_eta()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, height_partitioning.at(0), upper_bound}}));
  coord_maps.emplace_back(
      make_coordinate_map_base<Frame::Logical, TargetFrame>(Wedge3DPrism{
          Wedge2D{radial_partitioning.at(0), outer_radius, 1.0, 1.0,
                  OrientationMap<2>{std::array<Direction<2>, 2>{
                      {Direction<2>::upper_eta(), Direction<2>::lower_xi()}}},
                  use_equiangular_map},
          Affine{-1.0, 1.0, height_partitioning.at(0), upper_bound}}));

  test_domain_construction(domain, expected_block_neighbors,
                           expected_external_boundaries, coord_maps);

  test_initial_domain(domain, refined_cylinder.initial_refinement_levels());

  Parallel::register_classes_in_list<typename creators::Cylinder::maps_list>();
  test_serialization(domain);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.Cylinder", "[Domain][Unit]") {
  test_cylinder_no_refinement();

  {
    INFO("Test for Cylinder with one additional layer and shell");
    test_refined_cylinder_boundaries(true);
    test_refined_cylinder_boundaries(false);
    test_refined_cylinder_periodic_boundaries(true);
    test_refined_cylinder_periodic_boundaries(false);
  }
}
}  // namespace domain
