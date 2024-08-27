// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines actions ApplyLimiter and SendDataForLimiter

#pragma once

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <map>
#include <optional>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Tags.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Parallel/Invoke.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Limiters {
namespace Tags {
/// \ingroup DiscontinuousGalerkinGroup
/// \ingroup LimitersGroup
/// \brief The inbox tag for limiter communication.
template <typename Metavariables>
struct LimiterCommunicationTag : public Parallel::InboxInserters::Map<
                                     LimiterCommunicationTag<Metavariables>> {
  static constexpr size_t volume_dim = Metavariables::system::volume_dim;
  using packaged_data_t = typename Metavariables::limiter::type::PackagedData;
  using temporal_id = typename Metavariables::temporal_id::type;
  using type =
      std::map<temporal_id,
               std::unordered_map<DirectionalId<volume_dim>, packaged_data_t,
                                  boost::hash<DirectionalId<volume_dim>>>>;
};
}  // namespace Tags

namespace Actions {
/// \ingroup ActionsGroup
/// \ingroup DiscontinuousGalerkinGroup
/// \ingroup LimitersGroup
/// \brief Receive limiter data from neighbors, then apply limiter.
///
/// Currently, is not tested for support of:
/// - h-refinement
/// Currently, does not support:
/// - Local time-stepping
///
/// Uses:
/// - GlobalCache:
///   - Metavariables::limiter
/// - DataBox:
///   - Metavariables::limiter::type::limit_argument_tags
///   - Metavariables::temporal_id
///   - Tags::Element<volume_dim>
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies:
///   - Metavariables::limiter::type::limit_tags
///
/// \see SendDataForLimiter
template <typename Metavariables>
struct Limit {
  using const_global_cache_tags = tmpl::list<typename Metavariables::limiter>;

 public:
  using limiter_comm_tag =
      Limiters::Tags::LimiterCommunicationTag<Metavariables>;
  using inbox_tags = tmpl::list<limiter_comm_tag>;

  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    static_assert(
        not Metavariables::local_time_stepping,
        "Limiter communication actions do not yet support local time stepping");

    constexpr size_t volume_dim = Metavariables::system::volume_dim;

    const auto& local_temporal_id =
        db::get<typename Metavariables::temporal_id>(box);
    auto& inbox = tuples::get<limiter_comm_tag>(inboxes);
    const auto& received = inbox.find(local_temporal_id);

    const auto& element = db::get<domain::Tags::Element<volume_dim>>(box);
    const auto num_expected = element.neighbors().size();
    if (num_expected > 0 and
        (received == inbox.end() or received->second.size() != num_expected)) {
      return {Parallel::AlgorithmExecution::Retry, std::nullopt};
    }

    const auto& limiter = get<typename Metavariables::limiter>(cache);
    using mutate_tags = typename Metavariables::limiter::type::limit_tags;
    using argument_tags =
        typename Metavariables::limiter::type::limit_argument_tags;
    db::mutate_apply<mutate_tags, argument_tags>(limiter, make_not_null(&box),
                                                 inbox[local_temporal_id]);

    inbox.erase(local_temporal_id);

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

/// \ingroup ActionsGroup
/// \ingroup DiscontinuousGalerkinGroup
/// \ingroup LimitersGroup
/// \brief Send local data needed for limiting.
///
/// Currently, is not tested for support of:
/// - h-refinement
/// Currently, does not support:
/// - Local time-stepping
///
/// Uses:
/// - GlobalCache:
///   - Metavariables::limiter
/// - DataBox:
///   - Tags::Element<volume_dim>
///   - Metavariables::limiter::type::package_argument_tags
///   - Metavariables::temporal_id
///
/// DataBox changes:
/// - Adds: nothing
/// - Removes: nothing
/// - Modifies: nothing
///
/// \see ApplyLimiter
template <typename Metavariables>
struct SendData {
  using const_global_cache_tags = tmpl::list<typename Metavariables::limiter>;
  using limiter_comm_tag =
      Limiters::Tags::LimiterCommunicationTag<Metavariables>;

  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box, tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    constexpr size_t volume_dim = Metavariables::system::volume_dim;

    auto& receiver_proxy =
        Parallel::get_parallel_component<ParallelComponent>(cache);
    const auto& element = db::get<domain::Tags::Element<volume_dim>>(box);
    const auto& temporal_id = db::get<typename Metavariables::temporal_id>(box);
    const auto& limiter = get<typename Metavariables::limiter>(cache);

    for (const auto& direction_neighbors : element.neighbors()) {
      const auto& direction = direction_neighbors.first;
      const size_t dimension = direction.dimension();
      const auto& neighbors_in_direction = direction_neighbors.second;
      ASSERT(neighbors_in_direction.size() == 1,
             "h-adaptivity is not supported yet.\nDirection: "
                 << direction << "\nDimension: " << dimension
                 << "\nNeighbors:\n"
                 << neighbors_in_direction);
      const auto& orientation = neighbors_in_direction.orientation();
      const auto direction_from_neighbor = orientation(direction.opposite());

      using argument_tags =
          typename Metavariables::limiter::type::package_argument_tags;
      const auto packaged_data = db::apply<argument_tags>(
          [&limiter](const auto&... args) {
            // Note: orientation is received as last element of pack `args`
            typename Metavariables::limiter::type::PackagedData pack{};
            limiter.package_data(make_not_null(&pack), args...);
            return pack;
          },
          box, orientation);

      for (const auto& neighbor : neighbors_in_direction) {
        Parallel::receive_data<limiter_comm_tag>(
            receiver_proxy[neighbor], temporal_id,
            std::make_pair(DirectionalId<volume_dim>{direction_from_neighbor,
                                                     element.id()},
                           packaged_data));

      }  // loop over neighbors_in_direction
    }    // loop over element.neighbors()

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions
}  // namespace Limiters
