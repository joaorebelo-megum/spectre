// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Mesh.hpp"
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"

namespace {
template <size_t Dim, typename T>
void test_coordinates_compute_item(const Mesh<Dim>& mesh, T map) noexcept {
  using map_tag = Tags::ElementMap<Dim, Frame::Grid>;
  using coordinates_tag =
      Tags::Coordinates<map_tag, Tags::LogicalCoordinates<Dim>>;
  CHECK(coordinates_tag::name() == "Coordinates");
  const auto box = db::create<
      db::AddSimpleTags<Tags::Mesh<Dim>, map_tag>,
      db::AddComputeTags<
          Tags::LogicalCoordinates<Dim>,
          Tags::Coordinates<map_tag, Tags::LogicalCoordinates<Dim>>>>(
      mesh, ElementMap<Dim, Frame::Grid>(
                ElementId<Dim>(0),
                make_coordinate_map_base<Frame::Logical, Frame::Grid>(map)));
  CHECK_ITERABLE_APPROX(
      (db::get<Tags::Coordinates<map_tag, Tags::LogicalCoordinates<Dim>>>(box)),
      (make_coordinate_map<Frame::Logical, Frame::Grid>(map)(
          db::get<Tags::LogicalCoordinates<Dim>>(box))));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinatesTag", "[Unit][Domain]") {
  using Affine = CoordinateMaps::Affine;
  using Affine2d = CoordinateMaps::ProductOf2Maps<Affine, Affine>;
  using Affine3d = CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

  test_coordinates_compute_item(
      Mesh<1>{5, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto},
      Affine{-1.0, 1.0, -0.3, 0.7});
  test_coordinates_compute_item(
      Mesh<2>{{{5, 7}},
              Spectral::Basis::Legendre,
              Spectral::Quadrature::GaussLobatto},
      Affine2d{Affine{-1.0, 1.0, -0.3, 0.7}, Affine{-1.0, 1.0, 0.3, 0.55}});
  test_coordinates_compute_item(
      Mesh<3>{{{5, 6, 9}},
              Spectral::Basis::Legendre,
              Spectral::Quadrature::GaussLobatto},
      Affine3d{Affine{-1.0, 1.0, -0.3, 0.7}, Affine{-1.0, 1.0, 0.3, 0.55},
               Affine{-1.0, 1.0, 2.3, 2.8}});
}
