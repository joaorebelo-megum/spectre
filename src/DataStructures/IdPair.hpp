// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <ostream>
#include <pup.h>
#include <type_traits>
#include <utility>

/*!
 * \ingroup DataStructuresGroup
 * \brief A data structure that contains an ID and data associated with that ID.
 */
template <typename IdType, typename DataType>
struct IdPair {
  IdPair();
  IdPair(IdType id_in, DataType data_in);

  IdType id{};
  DataType data{};
};

template <typename IdType, typename DataType>
IdPair<IdType, DataType>::IdPair() = default;

template <typename IdType, typename DataType>
IdPair<IdType, DataType>::IdPair(IdType id_in, DataType data_in)
    : id(std::move(id_in)), data(std::move(data_in)) {}

template <typename IdType, typename DataType>
IdPair<std::decay_t<IdType>, std::decay_t<DataType>> make_id_pair(
    IdType&& id, DataType&& data) {
  return {std::forward<IdType>(id), std::forward<DataType>(data)};
}

/// \cond
// We write the pup function as a free function to keep IdPair a POD
// clang-tidy: no non-const references
template <typename IdType, typename DataType>
void pup(PUP::er& p, IdPair<IdType, DataType>& t) {  // NOLINT
  p | t.id;
  p | t.data;
}

// clang-tidy: no non-const references
template <typename IdType, typename DataType>
void operator|(PUP::er& p, IdPair<IdType, DataType>& t) {  // NOLINT
  pup(p, t);
}

template <typename IdType, typename DataType>
bool operator==(const IdPair<IdType, DataType>& lhs,
                const IdPair<IdType, DataType>& rhs) {
  return lhs.id == rhs.id and lhs.data == rhs.data;
}

template <typename IdType, typename DataType>
bool operator!=(const IdPair<IdType, DataType>& lhs,
                const IdPair<IdType, DataType>& rhs) {
  return not(lhs == rhs);
}

template <typename IdType, typename DataType>
std::ostream& operator<<(std::ostream& os, const IdPair<IdType, DataType>& t) {
  return os << '(' << t.id << ',' << t.data << ')';
}
/// \endcond
