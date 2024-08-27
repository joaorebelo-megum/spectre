// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"

#include <cstddef>
#include <memory>
#include <pup.h>
#include <string>

#include "Domain/FunctionsOfTime/FixedSpeedCubic.hpp"
#include "Domain/FunctionsOfTime/IntegratedFunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/SettleToConstant.hpp"
#include "Domain/FunctionsOfTime/SettleToConstantQuaternion.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace domain::FunctionsOfTime {
void register_derived_with_charm() {
  register_classes_with_charm<FunctionsOfTime::FixedSpeedCubic,
                              FunctionsOfTime::IntegratedFunctionOfTime,
                              FunctionsOfTime::PiecewisePolynomial<0>,
                              FunctionsOfTime::PiecewisePolynomial<1>,
                              FunctionsOfTime::PiecewisePolynomial<2>,
                              FunctionsOfTime::PiecewisePolynomial<3>,
                              FunctionsOfTime::PiecewisePolynomial<4>,
                              FunctionsOfTime::QuaternionFunctionOfTime<2>,
                              FunctionsOfTime::QuaternionFunctionOfTime<3>,
                              FunctionsOfTime::SettleToConstant,
                              FunctionsOfTime::SettleToConstantQuaternion>();
}
}  // namespace domain::FunctionsOfTime
