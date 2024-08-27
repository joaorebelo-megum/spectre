// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tags.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Utilities/TypeTraits.hpp"

class ComplexDataVector;
class DataVector;
class ModalVector;

namespace {
struct ComplexScalarTag : db::SimpleTag {
  using type = Scalar<ComplexDataVector>;
  static std::string name() { return "ComplexScalar"; }
};

struct ScalarTag : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() { return "Scalar"; }
};

template <size_t Dim>
struct VectorTag : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim>;
  static std::string name() { return "I<" + std::to_string(Dim) + ">"; }
};

void test_mean_tag() {
  static_assert(
      std::is_same_v<typename Tags::Mean<ScalarTag>::type, Scalar<double>>,
      "Failed testing Tags::Mean<ScalarTag>");
  static_assert(std::is_same_v<typename Tags::Mean<VectorTag<1>>::type,
                               tnsr::I<double, 1>>,
                "Failed testing Tags::Mean<ScalarTag>");
  static_assert(std::is_same_v<typename Tags::Mean<VectorTag<3>>::type,
                               tnsr::I<double, 3>>,
                "Failed testing Tags::Mean<ScalarTag>");
  TestHelpers::db::test_prefix_tag<Tags::Mean<ScalarTag>>("Mean(Scalar)");
  TestHelpers::db::test_prefix_tag<Tags::Mean<VectorTag<1>>>("Mean(I<1>)");
  TestHelpers::db::test_prefix_tag<Tags::Mean<VectorTag<3>>>("Mean(I<3>)");
}

void test_modal_tag() {
  static_assert(std::is_same_v<typename Tags::Modal<ScalarTag>::type,
                               Scalar<ModalVector>>,
                "Failed testing Tags::Modal<ScalarTag>");
  static_assert(std::is_same_v<typename Tags::Modal<VectorTag<1>>::type,
                               tnsr::I<ModalVector, 1>>,
                "Failed testing Tags::Modal<VectorTag<1>>");
  static_assert(std::is_same_v<typename Tags::Modal<VectorTag<3>>::type,
                               tnsr::I<ModalVector, 3>>,
                "Failed testing Tags::Modal<VectorTag<3>>");
  TestHelpers::db::test_prefix_tag<Tags::Modal<ScalarTag>>("Modal(Scalar)");
  TestHelpers::db::test_prefix_tag<Tags::Modal<VectorTag<1>>>("Modal(I<1>)");
  TestHelpers::db::test_prefix_tag<Tags::Modal<VectorTag<3>>>("Modal(I<3>)");
}

void test_spin_weighted_tag() {
  static_assert(
      std::is_same_v<
          typename Tags::SpinWeighted<ComplexScalarTag,
                                      std::integral_constant<int, 1>>::type,
          Scalar<SpinWeighted<ComplexDataVector, 1>>>,
      "Failed testing Tags::SpinWeighted<ScalarTag>");
  TestHelpers::db::test_prefix_tag<Tags::SpinWeighted<ComplexScalarTag,
                           std::integral_constant<int, -2>>>(
        "SpinWeighted(ComplexScalar, -2)");
}

void test_multiplies_tag() {
  using test_multiplies_tag = Tags::Multiplies<
      Tags::SpinWeighted<ComplexScalarTag, std::integral_constant<int, 1>>,
      Tags::SpinWeighted<ComplexScalarTag, std::integral_constant<int, -2>>>;
  static_assert(
      std::is_same_v<typename test_multiplies_tag::type,
                     Scalar<SpinWeighted<ComplexDataVector, -1>>>,
      "Failed testing Tags::Multiplies for Tags::SpinWeighted operands");
  TestHelpers::db::test_prefix_tag<test_multiplies_tag>(
        "Multiplies(SpinWeighted(ComplexScalar, 1), "
        "SpinWeighted(ComplexScalar, -2))");
}
}  // namespace


SPECTRE_TEST_CASE("Unit.DataStructures.Tags", "[Unit][DataStructures]") {
  test_mean_tag();
  test_modal_tag();
  test_spin_weighted_tag();
  test_multiplies_tag();
}
