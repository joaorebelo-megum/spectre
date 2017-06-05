# Distributed under the MIT License.
# See LICENSE.txt for details.

find_package(LIBXSMM REQUIRED)

include_directories(SYSTEM ${LIBXSMM_INCLUDE_DIRS})
set(SPECTRE_LIBRARIES "${SPECTRE_LIBRARIES};${LIBXSMM_LIBRARIES}")

message(STATUS "LIBXSMM libs: " ${LIBXSMM_LIBRARIES})
message(STATUS "LIBXSMM incl: " ${LIBXSMM_INCLUDE_DIRS})
