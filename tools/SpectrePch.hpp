// Distributed under the MIT License.
// See LICENSE.txt for details.

#ifndef SPECTRE_PCH_HPP
#define SPECTRE_PCH_HPP

// Include STL headers
#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <ostream>
#include <utility>
#include <vector>

// For Blaze error handling (see SetupBlaze.cmake)
#include <Utilities/BlazeExceptions.hpp>

#include <Utilities/ErrorHandling/Assert.hpp>
#include <blaze/math/CustomVector.h>
#include <blaze/math/DenseVector.h>
#include <blaze/math/GroupTag.h>
#include <blaze/math/typetraits/IsVector.h>
#include <blaze/system/Optimizations.h>
#include <blaze/system/Version.h>
#include <blaze/util/typetraits/RemoveConst.h>

// Include Brigand related headers
#include <Utilities/TMPL.hpp>

#endif  // SPECTRE_PCH_HPP
