// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/SpecialRelativity/LorentzBoostMatrix.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace sr {
template <size_t SpatialDim>
tnsr::Ab<double, SpatialDim, Frame::NoFrame> lorentz_boost_matrix(
    const tnsr::I<double, SpatialDim, Frame::NoFrame>& velocity) {
  auto boost_matrix =
      make_with_value<tnsr::Ab<double, SpatialDim, Frame::NoFrame>>(
          get<0>(velocity), std::numeric_limits<double>::signaling_NaN());
  lorentz_boost_matrix<SpatialDim>(&boost_matrix, velocity);
  return boost_matrix;
}

template <size_t SpatialDim>
void lorentz_boost_matrix(
    gsl::not_null<tnsr::Ab<double, SpatialDim, Frame::NoFrame>*> boost_matrix,
    const tnsr::I<double, SpatialDim, Frame::NoFrame>& velocity) {
  const double velocity_squared{get(dot_product(velocity, velocity))};
  const double lorentz_factor{1.0 / sqrt(1.0 - velocity_squared)};

  // For the spatial-spatial terms of the boost matrix, we need to compute
  // a prefactor, which is essentially kinetic energy per mass per velocity
  // squared. Specifically, the prefactor is
  //
  // kinetic_energy_per_v_squared = (lorentz_factor-1.0)/velocity^2
  //
  // This is algebraically equivalent to
  //
  // kinetic_energy_per_v_squared = lorentz_factor / ((1 + sqrt(1-velocity^2))),
  //
  // a form that avoids division by zero as v->0.

  double kinetic_energy_per_v_squared{square(lorentz_factor) /
                                      (1.0 + lorentz_factor)};

  get<0, 0>(*boost_matrix) = lorentz_factor;
  for (size_t i = 0; i < SpatialDim; ++i) {
    (*boost_matrix).get(0, i + 1) = velocity.get(i) * lorentz_factor;
    (*boost_matrix).get(i + 1, 0) = velocity.get(i) * lorentz_factor;
    for (size_t j = 0; j < SpatialDim; ++j) {
      (*boost_matrix).get(i + 1, j + 1) =
          velocity.get(i) * velocity.get(j) * kinetic_energy_per_v_squared;
    }
    (*boost_matrix).get(i + 1, i + 1) += 1.0;
  }
}

template <size_t SpatialDim>
tnsr::Ab<double, SpatialDim, Frame::NoFrame> lorentz_boost_matrix(
    const std::array<double, SpatialDim>& velocity) {
  tnsr::I<double, SpatialDim, Frame::NoFrame> velocity_as_tensor(velocity);
  return lorentz_boost_matrix(velocity_as_tensor);
}

template <typename DataType, size_t SpatialDim, typename Frame>
void lorentz_boost(
    const gsl::not_null<tnsr::I<DataType, SpatialDim, Frame>*> result,
    const tnsr::I<DataType, SpatialDim, Frame>& vector,
    const double vector_component_0,
    const std::array<double, SpatialDim>& velocity) {
  if (velocity == make_array<SpatialDim>(0.)) {
    *result = vector;
    return;
  }
  // Inverse matrix with respect to the boost applied to one forms
  const auto boost_matrix = lorentz_boost_matrix(-velocity);
  for (size_t i = 0; i < SpatialDim; ++i) {
    result->get(i) = boost_matrix.get(i + 1, 0) * vector_component_0;
    for (size_t j = 0; j < SpatialDim; ++j) {
      result->get(i) += boost_matrix.get(i + 1, j + 1) * vector.get(j);
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
void lorentz_boost(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> result,
    const tnsr::a<DataType, SpatialDim, Frame>& one_form,
    const std::array<double, SpatialDim>& velocity) {
  if (velocity == make_array<SpatialDim>(0.)) {
    *result = one_form;
    return;
  }
  const auto boost_matrix = lorentz_boost_matrix(velocity);
  for (size_t i = 0; i < SpatialDim + 1; ++i) {
    result->get(i) = 0.;
    for (size_t j = 0; j < SpatialDim + 1; ++j) {
      result->get(i) += boost_matrix.get(i, j) * one_form.get(j);
    }
  }
}

template <typename DataType, size_t SpatialDim, typename Frame>
void lorentz_boost(
    const gsl::not_null<tnsr::ab<DataType, SpatialDim, Frame>*> result,
    const tnsr::ab<DataType, SpatialDim, Frame>& tensor,
    const std::array<double, SpatialDim>& velocity) {
  if (velocity == make_array<SpatialDim>(0.)) {
    *result = tensor;
    return;
  }
  const auto boost_matrix = lorentz_boost_matrix(velocity);
  for (size_t i = 0; i < SpatialDim + 1; ++i) {
    for (size_t k = 0; k < SpatialDim + 1; ++k) {
      result->get(i, k) = 0.;
      for (size_t j = 0; j < SpatialDim + 1; ++j) {
        for (size_t l = 0; l < SpatialDim + 1; ++l) {
          result->get(i, k) += boost_matrix.get(i, j) * boost_matrix.get(k, l) *
                               tensor.get(j, l);
        }
      }
    }
  }
}

}  // namespace sr

// Explicit Instantiations
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                       \
  template tnsr::Ab<double, DIM(data), Frame::NoFrame>             \
  sr::lorentz_boost_matrix(                                        \
      const tnsr::I<double, DIM(data), Frame::NoFrame>& velocity); \
  template void sr::lorentz_boost_matrix(                          \
      gsl::not_null<tnsr::Ab<double, DIM(data), Frame::NoFrame>*>  \
          boost_matrix,                                            \
      const tnsr::I<double, DIM(data), Frame::NoFrame>& velocity); \
  template tnsr::Ab<double, DIM(data), Frame::NoFrame>             \
  sr::lorentz_boost_matrix(const std::array<double, DIM(data)>& velocity);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                              \
  template void sr::lorentz_boost(                                        \
      const gsl::not_null<tnsr::I<DTYPE(data), DIM(data), FRAME(data)>*>  \
          result,                                                         \
      const tnsr::I<DTYPE(data), DIM(data), FRAME(data)>& one_form,       \
      const double one_form_component_0,                                  \
      const std::array<double, DIM(data)>& velocity);                     \
  template void sr::lorentz_boost(                                        \
      const gsl::not_null<tnsr::a<DTYPE(data), DIM(data), FRAME(data)>*>  \
          result,                                                         \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>& one_form,       \
      const std::array<double, DIM(data)>& velocity);                     \
  template void sr::lorentz_boost(                                        \
      const gsl::not_null<tnsr::ab<DTYPE(data), DIM(data), FRAME(data)>*> \
          result,                                                         \
      const tnsr::ab<DTYPE(data), DIM(data), FRAME(data)>& tensor,        \
      const std::array<double, DIM(data)>& velocity);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Distorted, Frame::Inertial))

#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
