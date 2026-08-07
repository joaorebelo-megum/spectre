# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_XctsBoundaryConditions")

set(LIBRARY_SOURCES
  Test_ApparentHorizon.cpp
  Test_Flatness.cpp
  Test_NumericData.cpp
  Test_Robin.cpp
  )

add_test_library(${LIBRARY} "${LIBRARY_SOURCES}")

target_link_libraries(
  ${LIBRARY}
  PRIVATE
  CoordinateMaps
  DataStructures
  Domain
  DomainCreators
  DomainStructure
  Elliptic
  Exporter
  H5
  GeneralRelativity
  GeneralRelativitySolutions
  LinearOperators
  Utilities
  XctsBoundaryConditions
  XctsSolutions
  )
