// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;

namespace Burgers {
namespace Tags {
struct U;
}  // namespace Tags
}  // namespace Burgers
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl

class DataVector;
/// \endcond

namespace Burgers {
/// Computes the time derivative terms needed for the Burgers system, which are
/// just the fluxes.
struct TimeDerivativeTerms {
  using temporary_tags = tmpl::list<>;
  using argument_tags = tmpl::list<Tags::U>;

  static void apply(
      // Time derivatives returned by reference. No source terms or
      // nonconservative products, so not used. All the tags in the
      // variables_tag in the system struct.
      gsl::not_null<Scalar<DataVector>*> /*non_flux_terms_dt_vars*/,

      // Fluxes returned by reference. Listed in the system struct as
      // flux_variables.
      gsl::not_null<tnsr::I<DataVector, 1, Frame::Inertial>*> flux_u,

      // Arguments listed in argument_tags above
      const Scalar<DataVector>& u);
};
}  // namespace Burgers
