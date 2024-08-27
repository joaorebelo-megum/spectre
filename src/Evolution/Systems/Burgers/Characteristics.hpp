// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/FaceNormal.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;

namespace Burgers {
namespace Tags {
struct U;
}  // namespace Tags
}  // namespace Burgers
/// \endcond

namespace Burgers::Tags {
/// Computes the characteristic speeds
struct CharacteristicSpeedsCompute : CharacteristicSpeeds, db::ComputeTag {
  using base = CharacteristicSpeeds;
  using argument_tags =
      tmpl::list<Tags::U, domain::Tags::UnnormalizedFaceNormal<1>>;
  using return_type = std::array<DataVector, 1>;
  static void function(gsl::not_null<return_type*> result,
                       const Scalar<DataVector>& u,
                       const tnsr::i<DataVector, 1>& normal);
};

struct LargestCharacteristicSpeed : db::SimpleTag {
  using type = double;
};

struct ComputeLargestCharacteristicSpeed : LargestCharacteristicSpeed,
                                           db::ComputeTag {
  using argument_tags = tmpl::list<Tags::U>;
  using return_type = double;
  using base = LargestCharacteristicSpeed;
  static void function(const gsl::not_null<double*> speed,
                       const Scalar<DataVector>& u);
};
}  // namespace Burgers::Tags
