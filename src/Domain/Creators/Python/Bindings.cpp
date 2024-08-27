// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <pybind11/pybind11.h>

#include "Domain/Creators/Python/Cylinder.hpp"
#include "Domain/Creators/Python/DomainCreator.hpp"
#include "Domain/Creators/Python/Rectilinear.hpp"
#include "Domain/Creators/Python/Sphere.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/TimeDependence/RegisterDerivedWithCharm.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"

namespace py = pybind11;

namespace domain::creators {

PYBIND11_MODULE(_Pybindings, m) {  // NOLINT
  enable_segfault_handler();
  py::module_::import("spectre.Domain");
  py::module_::import("spectre.Domain.CoordinateMaps");
  domain::creators::register_derived_with_charm();
  domain::creators::time_dependence::register_derived_with_charm();
  // Order is important: The base class `DomainCreator` needs to have its
  // bindings set up before the derived classes
  py_bindings::bind_domain_creator(m);
  py_bindings::bind_rectilinear(m);
  py_bindings::bind_cylinder(m);
  py_bindings::bind_sphere(m);
}

}  // namespace domain::creators
