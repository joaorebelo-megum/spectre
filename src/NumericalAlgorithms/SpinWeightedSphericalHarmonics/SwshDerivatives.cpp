// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshDerivatives.hpp"

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexDiagonalModalOperator.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/TempBuffer.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/ComplexDataView.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshTransform.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/StaticCache.hpp"

namespace Spectral::Swsh {
namespace detail {
namespace {
template <typename DerivativeKind, int Spin>
ComplexDiagonalModalOperator derivative_factors(const size_t l_max) {
  ComplexDiagonalModalOperator derivative_factors{
      size_of_libsharp_coefficient_vector(l_max)};
  for (const auto mode : cached_coefficients_metadata(l_max)) {
    // note that the libsharp transform data is stored as a pair of complex
    // vectors: one for the transform of the real part and one for the transform
    // of the imaginary part of the collocation data.
    derivative_factors[mode.transform_of_real_part_offset] =
        detail::derivative_factor<DerivativeKind>(mode.l, Spin);
    derivative_factors[mode.transform_of_imag_part_offset] =
        detail::derivative_factor<DerivativeKind>(mode.l, Spin);
  }
  // apply the sign change as appropriate due to the adjusted spin
  // this loop is over the (complex) transform coefficients of first the real
  // nodal data, followed by the imaginary nodal data. They receive the same
  // derivative factors, but potentially different signs.
  for (size_t i = 0; i < 2; i++) {
    ComplexDiagonalModalOperator view_of_derivative_factor{
        derivative_factors.data() +
            i * size_of_libsharp_coefficient_vector(l_max) / 2,
        size_of_libsharp_coefficient_vector(l_max) / 2};
    view_of_derivative_factor *= sharp_swsh_sign_change(
        Spin, Spin + Tags::derivative_spin_weight<DerivativeKind>, i == 0);
  }
  return derivative_factors;
}

template <typename DerivativeKind, int Spin>
const ComplexDiagonalModalOperator& cached_derivative_factors(
    const size_t l_max) {
  const static auto lazy_derivative_operator_cache =
      make_static_cache<CacheRange<0_st, swsh_derivative_maximum_l_max>>(
          [](const size_t generator_l_max) {
            return derivative_factors<DerivativeKind, Spin>(generator_l_max);
          });
  return lazy_derivative_operator_cache(l_max);
}
}  // namespace

template <typename DerivativeKind, int Spin>
void compute_coefficients_of_derivative(
    const gsl::not_null<
        SpinWeighted<ComplexModalVector,
                     Spin + Tags::derivative_spin_weight<DerivativeKind>>*>
        derivative_modes,
    const gsl::not_null<SpinWeighted<ComplexModalVector, Spin>*>
        pre_derivative_modes,
    const size_t l_max, const size_t number_of_radial_points) {
  derivative_modes->data().destructive_resize(pre_derivative_modes->size());
  const ComplexDiagonalModalOperator& modal_derivative_operator =
      cached_derivative_factors<DerivativeKind, Spin>(l_max);
  // multiply each radial chunk by the appropriate derivative factors.
  for (size_t i = 0; i < number_of_radial_points; ++i) {
    ComplexModalVector output_mode_view{
        derivative_modes->data().data() +
            i * size_of_libsharp_coefficient_vector(l_max),
        size_of_libsharp_coefficient_vector(l_max)};
    const ComplexModalVector input_mode_view{
        pre_derivative_modes->data().data() +
            i * size_of_libsharp_coefficient_vector(l_max),
        size_of_libsharp_coefficient_vector(l_max)};
    output_mode_view = modal_derivative_operator * input_mode_view;
  }
}
}  // namespace detail

template <typename DerivKind, ComplexRepresentation Representation, int Spin>
SpinWeighted<ComplexDataVector, Tags::derivative_spin_weight<DerivKind> + Spin>
angular_derivative(
    const size_t l_max, const size_t number_of_radial_points,
    const SpinWeighted<ComplexDataVector, Spin>& to_differentiate) {
  auto result =
      SpinWeighted<ComplexDataVector, Tags::derivative_spin_weight<DerivKind> +
                                          Spin>{to_differentiate.size()};
  angular_derivatives<tmpl::list<DerivKind>, Representation>(
      l_max, number_of_radial_points, make_not_null(&result), to_differentiate);
  return result;
}

#define GET_DERIVKIND(data) BOOST_PP_TUPLE_ELEM(0, data)
#define GET_SPIN(data) BOOST_PP_TUPLE_ELEM(1, data)

#define DERIVKIND_AND_SPIN_INSTANTIATION(r, data)                            \
  namespace detail {                                                         \
  template void                                                              \
  compute_coefficients_of_derivative<GET_DERIVKIND(data), GET_SPIN(data)>(   \
      const gsl::not_null<SpinWeighted<                                      \
          ComplexModalVector, GET_SPIN(data) + Tags::derivative_spin_weight< \
                                                   GET_DERIVKIND(data)>>*>   \
          derivative_modes,                                                  \
      const gsl::not_null<SpinWeighted<ComplexModalVector, GET_SPIN(data)>*> \
          pre_derivative_modes,                                              \
      const size_t l_max, const size_t number_of_radial_points);             \
  }

GENERATE_INSTANTIATIONS(DERIVKIND_AND_SPIN_INSTANTIATION,
                        (Tags::EthEthbar, Tags::EthbarEth), (-2, -1, 0, 1, 2))
GENERATE_INSTANTIATIONS(DERIVKIND_AND_SPIN_INSTANTIATION, (Tags::Eth),
                        (-2, -1, 0, 1))
GENERATE_INSTANTIATIONS(DERIVKIND_AND_SPIN_INSTANTIATION, (Tags::InverseEthbar),
                        (-2, -1, 0, 1))
GENERATE_INSTANTIATIONS(DERIVKIND_AND_SPIN_INSTANTIATION, (Tags::EthEth),
                        (-2, -1, 0))
GENERATE_INSTANTIATIONS(DERIVKIND_AND_SPIN_INSTANTIATION, (Tags::Ethbar),
                        (-1, 0, 1, 2))
GENERATE_INSTANTIATIONS(DERIVKIND_AND_SPIN_INSTANTIATION, (Tags::InverseEth),
                        (-1, 0, 1, 2))
GENERATE_INSTANTIATIONS(DERIVKIND_AND_SPIN_INSTANTIATION, (Tags::EthbarEthbar),
                        (0, 1, 2))

#undef DERIVKIND_AND_SPIN_INSTANTIATION
#undef GET_SPIN
#undef GET_DERIVKIND

#define GET_REPRESENTATION(data) BOOST_PP_TUPLE_ELEM(0, data)
#define GET_DERIVKIND(data) BOOST_PP_TUPLE_ELEM(1, data)
#define GET_SPIN(data) BOOST_PP_TUPLE_ELEM(2, data)

#define FULL_DERIVATIVE_INSTANTIATION(r, data)                              \
  template SpinWeighted<ComplexDataVector,                                  \
                        Tags::derivative_spin_weight<GET_DERIVKIND(data)> + \
                            GET_SPIN(data)>                                 \
  angular_derivative<GET_DERIVKIND(data), GET_REPRESENTATION(data),         \
                     GET_SPIN(data)>(                                       \
      const size_t l_max, const size_t number_of_radial_points,             \
      const SpinWeighted<ComplexDataVector, GET_SPIN(data)>&                \
          to_differentiate);

GENERATE_INSTANTIATIONS(FULL_DERIVATIVE_INSTANTIATION,
                        (ComplexRepresentation::Interleaved,
                         ComplexRepresentation::RealsThenImags),
                        (Tags::EthEthbar, Tags::EthbarEth), (-2, -1, 0, 1, 2))
GENERATE_INSTANTIATIONS(FULL_DERIVATIVE_INSTANTIATION,
                        (ComplexRepresentation::Interleaved,
                         ComplexRepresentation::RealsThenImags),
                        (Tags::Eth), (-2, -1, 0, 1))
GENERATE_INSTANTIATIONS(FULL_DERIVATIVE_INSTANTIATION,
                        (ComplexRepresentation::Interleaved,
                         ComplexRepresentation::RealsThenImags),
                        (Tags::InverseEthbar), (-2, -1, 0, 1))
GENERATE_INSTANTIATIONS(FULL_DERIVATIVE_INSTANTIATION,
                        (ComplexRepresentation::Interleaved,
                         ComplexRepresentation::RealsThenImags),
                        (Tags::EthEth), (-2, -1, 0))
GENERATE_INSTANTIATIONS(FULL_DERIVATIVE_INSTANTIATION,
                        (ComplexRepresentation::Interleaved,
                         ComplexRepresentation::RealsThenImags),
                        (Tags::Ethbar), (-1, 0, 1, 2))
GENERATE_INSTANTIATIONS(FULL_DERIVATIVE_INSTANTIATION,
                        (ComplexRepresentation::Interleaved,
                         ComplexRepresentation::RealsThenImags),
                        (Tags::InverseEth), (-1, 0, 1, 2))
GENERATE_INSTANTIATIONS(FULL_DERIVATIVE_INSTANTIATION,
                        (ComplexRepresentation::Interleaved,
                         ComplexRepresentation::RealsThenImags),
                        (Tags::EthbarEthbar), (0, 1, 2))

#undef GET_SPIN
#undef GET_DERIVKIND
#undef GET_REPRESENTATION
}  // namespace Spectral::Swsh
