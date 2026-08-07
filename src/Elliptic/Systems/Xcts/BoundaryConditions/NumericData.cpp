// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Xcts/BoundaryConditions/NumericData.hpp"

#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace Xcts::BoundaryConditions {

template <Xcts::Equations EnabledEquations>
NumericData<EnabledEquations>::NumericData(
    std::string data_file, std::string subgroup, const int observation_step,
    const bool extrapolate_into_excisions)
    : data_file_(std::move(data_file)),
      subgroup_(std::move(subgroup)),
      observation_step_(observation_step),
      extrapolate_into_excisions_(extrapolate_into_excisions) {
  load_interpolator();
}

template <Xcts::Equations EnabledEquations>
NumericData<EnabledEquations>::NumericData(const NumericData& rhs)
    : Base(rhs),
      data_file_(rhs.data_file_),
      subgroup_(rhs.subgroup_),
      observation_step_(rhs.observation_step_),
      extrapolate_into_excisions_(rhs.extrapolate_into_excisions_) {
  load_interpolator();
}

template <Xcts::Equations EnabledEquations>
NumericData<EnabledEquations>& NumericData<EnabledEquations>::operator=(
    const NumericData& rhs) {
  if (this != &rhs) {
    data_file_ = rhs.data_file_;
    subgroup_ = rhs.subgroup_;
    observation_step_ = rhs.observation_step_;
    extrapolate_into_excisions_ = rhs.extrapolate_into_excisions_;
    load_interpolator();
  }
  return *this;
}

template <Xcts::Equations EnabledEquations>
void NumericData<EnabledEquations>::load_interpolator() {
  // Constructing the interpolator opens the H5 files, which is not thread safe
  // unless HDF5 was built with thread-safety support. Boundary conditions live
  // in the domain, which is a `const_global_cache_tag`, and
  // `Parallel::GlobalCache` is a Nodegroup, so this runs once per node on a
  // single thread. Interpolating afterwards is thread safe.
  if (data_file_.empty()) {
    return;
  }
  interpolator_ = spectre::Exporter::PointwiseInterpolator<3, Frame::Inertial>{
      data_file_, subgroup_,
      spectre::Exporter::ObservationStep{observation_step_},
      spectre::Exporter::get_tensor_components<
          detail::numeric_boundary_load_tags>()};
}

template <Xcts::Equations EnabledEquations>
typename NumericData<EnabledEquations>::BoundaryValues
NumericData<EnabledEquations>::boundary_values(
    const tnsr::I<DataVector, 3>& x) const {
  std::vector<DataVector> interpolated_data{};
  interpolator_.interpolate_to_points(make_not_null(&interpolated_data), x,
                                      extrapolate_into_excisions_,
                                      /*error_on_missing_points=*/true,
                                      /*num_threads=*/1);
  auto loaded =
      spectre::Exporter::make_tagged_tuple<detail::numeric_boundary_load_tags>(
          std::move(interpolated_data));
  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<DataVector, 3>>(loaded);
  const auto& lapse = get<gr::Tags::Lapse<DataVector>>(loaded);
  // Unimodular conformal split, matching
  // `Xcts::AnalyticData::NumericBinaryWithWaves`.
  const auto det_spatial_metric = determinant_and_inverse(spatial_metric).first;
  const DataVector conformal_factor = pow(get(det_spatial_metric), 1. / 12.);
  BoundaryValues result{};
  get(result.conformal_factor_minus_one) = conformal_factor - 1.;
  get(result.lapse_times_conformal_factor_minus_one) =
      get(lapse) * conformal_factor - 1.;
  result.shift_excess =
      get<Xcts::Tags::ShiftExcess<DataVector, 3, Frame::Inertial>>(loaded);
  return result;
}

template <>
void NumericData<Xcts::Equations::Hamiltonian>::apply(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_minus_one,
    const gsl::not_null<Scalar<DataVector>*>
    /*n_dot_conformal_factor_gradient*/,
    const tnsr::i<DataVector, 3>& /*deriv_conformal_factor*/,
    const tnsr::I<DataVector, 3>& x) const {
  *conformal_factor_minus_one = boundary_values(x).conformal_factor_minus_one;
}

template <>
void NumericData<Xcts::Equations::HamiltonianAndLapse>::apply(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_minus_one,
    const gsl::not_null<Scalar<DataVector>*>
        lapse_times_conformal_factor_minus_one,
    const gsl::not_null<Scalar<DataVector>*>
    /*n_dot_conformal_factor_gradient*/,
    const gsl::not_null<Scalar<DataVector>*>
    /*n_dot_lapse_times_conformal_factor_gradient*/,
    const tnsr::i<DataVector, 3>& /*deriv_conformal_factor*/,
    const tnsr::i<DataVector, 3>& /*deriv_lapse_times_conformal_factor*/,
    const tnsr::I<DataVector, 3>& x) const {
  auto values = boundary_values(x);
  *conformal_factor_minus_one = std::move(values.conformal_factor_minus_one);
  *lapse_times_conformal_factor_minus_one =
      std::move(values.lapse_times_conformal_factor_minus_one);
}

template <>
void NumericData<Xcts::Equations::HamiltonianLapseAndShift>::apply(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_minus_one,
    const gsl::not_null<Scalar<DataVector>*>
        lapse_times_conformal_factor_minus_one,
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess,
    const gsl::not_null<Scalar<DataVector>*>
    /*n_dot_conformal_factor_gradient*/,
    const gsl::not_null<Scalar<DataVector>*>
    /*n_dot_lapse_times_conformal_factor_gradient*/,
    const gsl::not_null<tnsr::I<DataVector, 3>*>
    /*n_dot_longitudinal_shift_excess*/,
    const tnsr::i<DataVector, 3>& /*deriv_conformal_factor*/,
    const tnsr::i<DataVector, 3>& /*deriv_lapse_times_conformal_factor*/,
    const tnsr::iJ<DataVector, 3>& /*deriv_shift_excess*/,
    const tnsr::I<DataVector, 3>& x) const {
  auto values = boundary_values(x);
  *conformal_factor_minus_one = std::move(values.conformal_factor_minus_one);
  *lapse_times_conformal_factor_minus_one =
      std::move(values.lapse_times_conformal_factor_minus_one);
  *shift_excess = std::move(values.shift_excess);
}

template <>
void NumericData<Xcts::Equations::Hamiltonian>::apply_linearized(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
    const gsl::not_null<Scalar<DataVector>*>
    /*n_dot_conformal_factor_gradient_correction*/,
    const tnsr::i<DataVector, 3>& /*deriv_conformal_factor_correction*/) {
  get(*conformal_factor_correction) = 0.;
}

template <>
void NumericData<Xcts::Equations::HamiltonianAndLapse>::apply_linearized(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
    const gsl::not_null<Scalar<DataVector>*>
        lapse_times_conformal_factor_correction,
    const gsl::not_null<Scalar<DataVector>*>
    /*n_dot_conformal_factor_gradient_correction*/,
    const gsl::not_null<Scalar<DataVector>*>
    /*n_dot_lapse_times_conformal_factor_gradient_correction*/,
    const tnsr::i<DataVector, 3>& /*deriv_conformal_factor_correction*/,
    const tnsr::i<DataVector,
                  3>& /*deriv_lapse_times_conformal_factor_correction*/) {
  get(*conformal_factor_correction) = 0.;
  get(*lapse_times_conformal_factor_correction) = 0.;
}

template <>
void NumericData<Xcts::Equations::HamiltonianLapseAndShift>::apply_linearized(
    const gsl::not_null<Scalar<DataVector>*> conformal_factor_correction,
    const gsl::not_null<Scalar<DataVector>*>
        lapse_times_conformal_factor_correction,
    const gsl::not_null<tnsr::I<DataVector, 3>*> shift_excess_correction,
    const gsl::not_null<Scalar<DataVector>*>
    /*n_dot_conformal_factor_gradient_correction*/,
    const gsl::not_null<Scalar<DataVector>*>
    /*n_dot_lapse_times_conformal_factor_gradient_correction*/,
    const gsl::not_null<tnsr::I<DataVector, 3>*>
    /*n_dot_longitudinal_shift_excess_correction*/,
    const tnsr::i<DataVector, 3>& /*deriv_conformal_factor_correction*/,
    const tnsr::i<DataVector,
                  3>& /*deriv_lapse_times_conformal_factor_correction*/,
    const tnsr::iJ<DataVector, 3>& /*deriv_shift_excess_correction*/) {
  // Dirichlet: the correction to a fixed boundary value vanishes.
  get(*conformal_factor_correction) = 0.;
  get(*lapse_times_conformal_factor_correction) = 0.;
  std::fill(shift_excess_correction->begin(), shift_excess_correction->end(),
            0.);
}

template <Xcts::Equations EnabledEquations>
void NumericData<EnabledEquations>::pup(PUP::er& p) {
  Base::pup(p);
  p | data_file_;
  p | subgroup_;
  p | observation_step_;
  p | extrapolate_into_excisions_;
  // The loaded volume data is not serialized; it is reloaded here so that each
  // node reads the file once rather than shipping it around.
  if (p.isUnpacking()) {
    load_interpolator();
  }
}

template <Xcts::Equations EnabledEquations>
bool operator==(const NumericData<EnabledEquations>& lhs,
                const NumericData<EnabledEquations>& rhs) {
  return lhs.data_file() == rhs.data_file() and
         lhs.subgroup() == rhs.subgroup() and
         lhs.observation_step() == rhs.observation_step() and
         lhs.extrapolate_into_excisions() == rhs.extrapolate_into_excisions();
}

template <Xcts::Equations EnabledEquations>
bool operator!=(const NumericData<EnabledEquations>& lhs,
                const NumericData<EnabledEquations>& rhs) {
  return not(lhs == rhs);
}

template <Xcts::Equations EnabledEquations>
PUP::able::PUP_ID NumericData<EnabledEquations>::my_PUP_ID = 0;  // NOLINT

#define EQNS(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                \
  template class NumericData<EQNS(data)>;                   \
  template bool operator==(const NumericData<EQNS(data)>&,  \
                           const NumericData<EQNS(data)>&); \
  template bool operator!=(const NumericData<EQNS(data)>&,  \
                           const NumericData<EQNS(data)>&);

GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (Xcts::Equations::Hamiltonian,
                         Xcts::Equations::HamiltonianAndLapse,
                         Xcts::Equations::HamiltonianLapseAndShift))
#undef INSTANTIATE
#undef EQNS

}  // namespace Xcts::BoundaryConditions
