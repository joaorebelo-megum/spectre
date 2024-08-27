// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <random>
#include <string>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshTestHelpers.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/ComplexDataView.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCoefficients.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshTags.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshTransform.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

namespace Spectral::Swsh {
namespace {

// for storing a computed spin-weighted value during the transform test
template <size_t index, int Spin>
struct TestTag : db::SimpleTag {
  using type = Scalar<SpinWeighted<ComplexDataVector, Spin>>;
};

using TestDerivativeTagList =
    tmpl::list<Tags::Derivative<TestTag<0, -1>, Tags::Eth>,
               Tags::Derivative<TestTag<0, -1>, Tags::EthEthbar>,
               Tags::Derivative<TestTag<1, -1>, Tags::EthEthbar>,
               Tags::Derivative<TestTag<0, 2>, Tags::EthbarEthbar>>;

// [make_transform_list]
using ExpectedInverseTransforms = tmpl::list<
    SwshTransform<tmpl::list<Tags::Derivative<TestTag<0, -1>, Tags::EthEthbar>,
                             Tags::Derivative<TestTag<1, -1>, Tags::EthEthbar>>,
                  ComplexRepresentation::RealsThenImags>,
    SwshTransform<
        tmpl::list<Tags::Derivative<TestTag<0, -1>, Tags::Eth>,
                   Tags::Derivative<TestTag<0, 2>, Tags::EthbarEthbar>>,
        ComplexRepresentation::RealsThenImags>>;

static_assert(
    std::is_same_v<make_transform_list<ComplexRepresentation::RealsThenImags,
                                       TestDerivativeTagList>,
                   ExpectedInverseTransforms>,
    "failed testing make_transform_list");
// [make_transform_list]

// [make_transform_from_derivative_tags]
using ExpectedTransforms =
    tmpl::list<SwshTransform<tmpl::list<TestTag<0, -1>, TestTag<1, -1>>,
                             ComplexRepresentation::Interleaved>,
               SwshTransform<tmpl::list<TestTag<0, 2>>,
                             ComplexRepresentation::Interleaved>>;

static_assert(std::is_same_v<make_transform_list_from_derivative_tags<
                                 ComplexRepresentation::Interleaved,
                                 TestDerivativeTagList>,
                             ExpectedTransforms>,
              "failed testing make_transform_list_from_derivative_tags");
// [make_transform_from_derivative_tags]

template <ComplexRepresentation Representation, int S>
void test_transform_and_inverse_transform() {
  // generate parameters for the points to transform
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<size_t> sdist{2, 7};
  const size_t l_max = sdist(gen);
  const size_t number_of_radial_points = 2;
  UniformCustomDistribution<double> coefficient_distribution{-10.0, 10.0};

  // A DataBox of two tags to transform and their spin-weighted transforms, to
  // verify the DataBox-compatible mutate interface
  using collocation_variables_tag =
      ::Tags::Variables<tmpl::list<TestTag<0, S>, TestTag<1, S>>>;
  using coefficients_variables_tag =
      ::Tags::Variables<tmpl::list<Tags::SwshTransform<TestTag<0, S>>,
                                   Tags::SwshTransform<TestTag<1, S>>>>;
  auto box = db::create<
      db::AddSimpleTags<collocation_variables_tag, coefficients_variables_tag,
                        Tags::LMax, Tags::NumberOfRadialPoints>,
      db::AddComputeTags<>>(
      typename collocation_variables_tag::type{
          number_of_radial_points * number_of_swsh_collocation_points(l_max)},
      typename coefficients_variables_tag::type{
          size_of_libsharp_coefficient_vector(l_max) * number_of_radial_points},
      l_max, number_of_radial_points);

  ComplexModalVector expected_modes{number_of_radial_points *
                                    size_of_libsharp_coefficient_vector(l_max)};
  TestHelpers::generate_swsh_modes<S>(
      make_not_null(&expected_modes), make_not_null(&gen),
      make_not_null(&coefficient_distribution), number_of_radial_points, l_max);
  ComplexModalVector two_times_expected_modes = 2.0 * expected_modes;

  // fill the expected collocation point data by evaluating the analytic
  // functions. This is very slow and rough (due to factorial division), but
  // comparatively simple to formulate.
  const auto coefficients_to_analytic_collocation =
      [&l_max](const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, S>>*>
                   computed_collocation,
               ComplexModalVector& modes) {
        TestHelpers::swsh_collocation_from_coefficients_and_basis_func<
            S, Representation>(
            make_not_null(&get(*computed_collocation).data()), modes, l_max,
            number_of_radial_points,
            TestHelpers::spin_weighted_spherical_harmonic);
      };
  db::mutate<TestTag<0, S>>(coefficients_to_analytic_collocation,
                            make_not_null(&box), expected_modes);
  db::mutate<TestTag<1, S>>(coefficients_to_analytic_collocation,
                            make_not_null(&box), two_times_expected_modes);

  const auto source_collocation_copy = get(db::get<TestTag<0, S>>(box));
  // transform using the DataBox mutate interface
  db::mutate_apply<
      SwshTransform<tmpl::list<TestTag<0, S>, TestTag<1, S>>, Representation>>(
      make_not_null(&box));

  // verify that the collocation points haven't been altered by the
  // transformation
  CHECK(source_collocation_copy == get(db::get<TestTag<0, S>>(box)));

  // approximation needs to be a little loose to consistently accommodate
  // the ratios of factorials in the analytic form
  Approx transform_approx =
      Approx::custom()
          .epsilon(std::numeric_limits<double>::epsilon() * 1.0e6)
          .scale(1.0);

  // check transformed modes against the generated ones
  CHECK_ITERABLE_CUSTOM_APPROX(
      get(db::get<Tags::SwshTransform<TestTag<0, S>>>(box)).data(),
      expected_modes, transform_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      get(db::get<Tags::SwshTransform<TestTag<1, S>>>(box)).data(),
      2.0 * expected_modes, transform_approx);

  // check the function interface which transforms a single spin-weighted vector
  // and returns the result
  SpinWeighted<ComplexModalVector, S> transformed_modes_from_function_call =
      swsh_transform<Representation>(l_max, number_of_radial_points,
                                     get(db::get<TestTag<0, S>>(box)));
  CHECK_ITERABLE_CUSTOM_APPROX(transformed_modes_from_function_call.data(),
                               expected_modes, transform_approx);

  // check the parameter-pack return by pointer function interface, which takes
  // an arbitrary number of spin-weighted vectors to transform.
  SpinWeighted<ComplexModalVector, S>
      two_times_transformed_modes_from_function_call;
  swsh_transform<Representation>(
      l_max, number_of_radial_points,
      make_not_null(&transformed_modes_from_function_call),
      make_not_null(&two_times_transformed_modes_from_function_call),
      get(db::get<TestTag<0, S>>(box)), get(db::get<TestTag<1, S>>(box)));

  CHECK_ITERABLE_CUSTOM_APPROX(
      two_times_transformed_modes_from_function_call.data(),
      2.0 * expected_modes, transform_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(transformed_modes_from_function_call.data(),
                               expected_modes, transform_approx);

  ComplexDataVector expected_collocation =
      get(db::get<TestTag<0, S>>(box)).data();

  // clear out the existing collocation data so we know we get the
  // correct inverse transform
  db::mutate<TestTag<0, S>, TestTag<1, S>>(
      [](const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, S>>*>
             collocation,
         const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, S>>*>
             another_collocation) {
        get(*collocation).data() = 0.0;
        get(*another_collocation).data() = 0.0;
      },
      make_not_null(&box));

  // transform using the DataBox mutate interface
  db::mutate_apply<InverseSwshTransform<
      tmpl::list<TestTag<0, S>, TestTag<1, S>>, Representation>>(
      make_not_null(&box));

  CHECK_ITERABLE_CUSTOM_APPROX(get(db::get<TestTag<0, S>>(box)).data(),
                               expected_collocation, transform_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(get(db::get<TestTag<1, S>>(box)).data(),
                               2.0 * expected_collocation, transform_approx);

  // check the function interface which transforms a single spin-weighted vector
  // and returns the result
  SpinWeighted<ComplexDataVector, S>
      transformed_collocation_from_function_call =
          inverse_swsh_transform<Representation>(
              l_max, number_of_radial_points,
              get(db::get<Tags::SwshTransform<TestTag<0, S>>>(box)));
  CHECK_ITERABLE_CUSTOM_APPROX(
      transformed_collocation_from_function_call.data(), expected_collocation,
      transform_approx);

  // check the parameter-pack return by pointer function interface, which takes
  // an arbitrary number of spin-weighted vectors to transform.
  SpinWeighted<ComplexDataVector, S>
      two_times_transformed_collocation_from_function_call;
  inverse_swsh_transform<Representation>(
      l_max, number_of_radial_points,
      make_not_null(&transformed_collocation_from_function_call),
      make_not_null(&two_times_transformed_collocation_from_function_call),
      get(db::get<Tags::SwshTransform<TestTag<0, S>>>(box)),
      get(db::get<Tags::SwshTransform<TestTag<1, S>>>(box)));

  CHECK_ITERABLE_CUSTOM_APPROX(
      two_times_transformed_collocation_from_function_call.data(),
      2.0 * expected_collocation, transform_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(
      transformed_collocation_from_function_call.data(), expected_collocation,
      transform_approx);
}

template <int Spin>
void test_interpolate_to_collocation() {
  // generate parameters for the points to transform
  MAKE_GENERATOR(gen);
  UniformCustomDistribution<size_t> sdist{2, 10};
  const size_t l_max = sdist(gen);
  const size_t number_of_radial_points = 2;
  UniformCustomDistribution<double> coefficient_distribution{0.1, 1.0};

  SpinWeighted<ComplexModalVector, Spin> expected_modes{
      number_of_radial_points * size_of_libsharp_coefficient_vector(l_max)};
  TestHelpers::generate_swsh_modes<Spin>(
      make_not_null(&expected_modes.data()), make_not_null(&gen),
      make_not_null(&coefficient_distribution), number_of_radial_points, l_max);

  const SpinWeighted<ComplexDataVector, Spin> collocation_data =
      inverse_swsh_transform(l_max, number_of_radial_points, expected_modes);
  const size_t target_l_max = sdist(gen);

  SpinWeighted<ComplexDataVector, Spin> resampled_collocation{
      number_of_swsh_collocation_points(target_l_max) *
      number_of_radial_points};
  interpolate_to_collocation(make_not_null(&resampled_collocation),
                             collocation_data, target_l_max, l_max,
                             number_of_radial_points);
  const auto resampled_goldberg_modes = libsharp_to_goldberg_modes(
      swsh_transform(target_l_max, number_of_radial_points,
                     resampled_collocation),
      target_l_max);
  const auto expected_goldberg_modes =
      libsharp_to_goldberg_modes(expected_modes, l_max);
  for (size_t i = 0; i < number_of_radial_points; ++i) {
    for (size_t j = 0; j < square(std::min(l_max, target_l_max) + 1); ++j) {
      CAPTURE(i);
      CAPTURE(j);
      CHECK_ITERABLE_APPROX(
          expected_goldberg_modes.data()[i * square(l_max + 1) + j],
          resampled_goldberg_modes.data()[i * square(target_l_max + 1) + j]);
    }
  }
  if (target_l_max > l_max) {
    for (size_t i = 0; i < number_of_radial_points; ++i) {
      for (size_t j = square(l_max + 1); j < square(target_l_max + 1); ++j) {
        CAPTURE(i);
        CAPTURE(j);
        CHECK(approx(real(resampled_goldberg_modes
                              .data()[i * square(target_l_max + 1) + j])) ==
              0.0);
        CHECK(approx(imag(resampled_goldberg_modes
                              .data()[i * square(target_l_max + 1) + j])) ==
              0.0);
      }
    }
  }
}

SPECTRE_TEST_CASE("Unit.NumericalAlgorithms.Spectral.SwshTransform",
                  "[Unit][NumericalAlgorithms]") {
  {
    INFO("Testing with ComplexRepresentation::Interleaved");
    test_transform_and_inverse_transform<ComplexRepresentation::Interleaved,
                                         -2>();
    test_transform_and_inverse_transform<ComplexRepresentation::Interleaved,
                                         0>();
    test_transform_and_inverse_transform<ComplexRepresentation::Interleaved,
                                         1>();
  }
  {
    INFO("Testing with ComplexRepresentation::RealsThenImags");
    test_transform_and_inverse_transform<ComplexRepresentation::RealsThenImags,
                                         -1>();
    test_transform_and_inverse_transform<ComplexRepresentation::RealsThenImags,
                                         0>();
    test_transform_and_inverse_transform<ComplexRepresentation::RealsThenImags,
                                         2>();
  }
  {
    INFO("Testing interpolate_to_collocation");
    test_interpolate_to_collocation<2>();
    test_interpolate_to_collocation<0>();
    test_interpolate_to_collocation<-1>();
  }
}
}  // namespace
}  // namespace Spectral::Swsh
