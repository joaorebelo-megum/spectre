// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pybind11/pybind11.h>

namespace domain::creators::py_bindings {
// NOLINTNEXTLINE(google-runtime-references)
// For now this does not support an outer-boundary condition or context.
void bind_binary_compact_object(pybind11::module& m);
}  // namespace domain::creators::py_bindings
