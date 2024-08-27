// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "NumericalAlgorithms/Convergence/Tags.hpp"
#include "NumericalAlgorithms/LinearSolver/InnerProduct.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Reduction.hpp"
#include "ParallelAlgorithms/LinearSolver/ConjugateGradient/ResidualMonitorActions.hpp"
#include "ParallelAlgorithms/LinearSolver/ConjugateGradient/Tags/InboxTags.hpp"
#include "ParallelAlgorithms/LinearSolver/Tags.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Convergence {
struct HasConverged;
}  // namespace Convergence
namespace tuples {
template <typename...>
class TaggedTuple;
}  // namespace tuples
namespace LinearSolver::cg::detail {
template <typename Metavariables, typename FieldsTag, typename OptionsGroup>
struct ResidualMonitor;
template <typename FieldsTag, typename OptionsGroup, typename Label>
struct UpdateOperand;
}  // namespace LinearSolver::cg::detail
/// \endcond

namespace LinearSolver::cg::detail {

template <typename FieldsTag, typename OptionsGroup, typename Label,
          typename SourceTag>
struct PrepareSolve {
 private:
  using fields_tag = FieldsTag;
  using source_tag = SourceTag;
  using operator_applied_to_fields_tag =
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, fields_tag>;
  using operand_tag =
      db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;
  using residual_tag =
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>;

 public:
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    db::mutate<Convergence::Tags::IterationId<OptionsGroup>, operand_tag,
               residual_tag>(
        [](const gsl::not_null<size_t*> iteration_id, const auto operand,
           const auto residual, const auto& source,
           const auto& operator_applied_to_fields) {
          *iteration_id = 0;
          *operand = source - operator_applied_to_fields;
          *residual = *operand;
        },
        make_not_null(&box), get<source_tag>(box),
        get<operator_applied_to_fields_tag>(box));

    // Perform global reduction to compute initial residual magnitude square for
    // residual monitor
    const auto& residual = get<residual_tag>(box);
    Parallel::contribute_to_reduction<
        InitializeResidual<FieldsTag, OptionsGroup, ParallelComponent>>(
        Parallel::ReductionData<
            Parallel::ReductionDatum<double, funcl::Plus<>>>{
            magnitude_square(residual)},
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index],
        Parallel::get_parallel_component<
            ResidualMonitor<Metavariables, FieldsTag, OptionsGroup>>(cache));

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

template <typename FieldsTag, typename OptionsGroup, typename Label>
struct InitializeHasConverged {
  using inbox_tags = tmpl::list<Tags::InitialHasConverged<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    auto& inbox = get<Tags::InitialHasConverged<OptionsGroup>>(inboxes);
    const auto& iteration_id =
        db::get<Convergence::Tags::IterationId<OptionsGroup>>(box);
    if (inbox.find(iteration_id) == inbox.end()) {
      return {Parallel::AlgorithmExecution::Retry, std::nullopt};
    }

    auto has_converged = std::move(inbox.extract(iteration_id).mapped());

    db::mutate<Convergence::Tags::HasConverged<OptionsGroup>>(
        [&has_converged](const gsl::not_null<Convergence::HasConverged*>
                             local_has_converged) {
          *local_has_converged = std::move(has_converged);
        },
        make_not_null(&box));

    // Skip steps entirely if the solve has already converged
    constexpr size_t step_end_index =
        tmpl::index_of<ActionList,
                       UpdateOperand<FieldsTag, OptionsGroup, Label>>::value;
    constexpr size_t this_action_index =
        tmpl::index_of<ActionList, InitializeHasConverged>::value;
    return {Parallel::AlgorithmExecution::Continue,
            has_converged ? (step_end_index + 1) : (this_action_index + 1)};
  }
};

template <typename FieldsTag, typename OptionsGroup, typename Label>
struct PerformStep {
  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    using fields_tag = FieldsTag;
    using operand_tag =
        db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;
    using operator_tag =
        db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, operand_tag>;

    // At this point Ap must have been computed in a previous action
    // We compute the inner product <p,p> w.r.t A. This requires a global
    // reduction.
    const double local_conj_grad_inner_product =
        inner_product(get<operand_tag>(box), get<operator_tag>(box));

    Parallel::contribute_to_reduction<
        ComputeAlpha<FieldsTag, OptionsGroup, ParallelComponent>>(
        Parallel::ReductionData<
            Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>,
            Parallel::ReductionDatum<double, funcl::Plus<>>>{
            get<Convergence::Tags::IterationId<OptionsGroup>>(box),
            local_conj_grad_inner_product},
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index],
        Parallel::get_parallel_component<
            ResidualMonitor<Metavariables, FieldsTag, OptionsGroup>>(cache));

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

template <typename FieldsTag, typename OptionsGroup, typename Label>
struct UpdateFieldValues {
 private:
  using fields_tag = FieldsTag;
  using operand_tag =
      db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;
  using operator_tag =
      db::add_tag_prefix<LinearSolver::Tags::OperatorAppliedTo, operand_tag>;
  using residual_tag =
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>;

 public:
  using inbox_tags = tmpl::list<Tags::Alpha<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& array_index, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    auto& inbox = get<Tags::Alpha<OptionsGroup>>(inboxes);
    const auto& iteration_id =
        db::get<Convergence::Tags::IterationId<OptionsGroup>>(box);
    if (inbox.find(iteration_id) == inbox.end()) {
      return {Parallel::AlgorithmExecution::Retry, std::nullopt};
    }

    const double alpha = std::move(inbox.extract(iteration_id).mapped());

    db::mutate<residual_tag, fields_tag>(
        [alpha](const auto residual, const auto fields, const auto& operand,
                const auto& operator_applied_to_operand) {
          *fields += alpha * operand;
          *residual -= alpha * operator_applied_to_operand;
        },
        make_not_null(&box), get<operand_tag>(box), get<operator_tag>(box));

    // Compute new residual norm in a second global reduction
    const auto& residual = get<residual_tag>(box);
    const double local_residual_magnitude_square = magnitude_square(residual);

    Parallel::contribute_to_reduction<
        UpdateResidual<FieldsTag, OptionsGroup, ParallelComponent>>(
        Parallel::ReductionData<
            Parallel::ReductionDatum<size_t, funcl::AssertEqual<>>,
            Parallel::ReductionDatum<double, funcl::Plus<>>>{
            get<Convergence::Tags::IterationId<OptionsGroup>>(box),
            local_residual_magnitude_square},
        Parallel::get_parallel_component<ParallelComponent>(cache)[array_index],
        Parallel::get_parallel_component<
            ResidualMonitor<Metavariables, FieldsTag, OptionsGroup>>(cache));

    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

template <typename FieldsTag, typename OptionsGroup, typename Label>
struct UpdateOperand {
 private:
  using fields_tag = FieldsTag;
  using operand_tag =
      db::add_tag_prefix<LinearSolver::Tags::Operand, fields_tag>;
  using residual_tag =
      db::add_tag_prefix<LinearSolver::Tags::Residual, fields_tag>;

 public:
  using inbox_tags =
      tmpl::list<Tags::ResidualRatioAndHasConverged<OptionsGroup>>;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box, tuples::TaggedTuple<InboxTags...>& inboxes,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    auto& inbox =
        get<Tags::ResidualRatioAndHasConverged<OptionsGroup>>(inboxes);
    const auto& iteration_id =
        db::get<Convergence::Tags::IterationId<OptionsGroup>>(box);
    if (inbox.find(iteration_id) == inbox.end()) {
      return {Parallel::AlgorithmExecution::Retry, std::nullopt};
    }

    auto received_data = std::move(inbox.extract(iteration_id).mapped());
    const double res_ratio = get<0>(received_data);
    auto& has_converged = get<1>(received_data);

    db::mutate<operand_tag, Convergence::Tags::IterationId<OptionsGroup>,
               Convergence::Tags::HasConverged<OptionsGroup>>(
        [res_ratio, &has_converged](
            const auto operand, const gsl::not_null<size_t*> local_iteration_id,
            const gsl::not_null<Convergence::HasConverged*> local_has_converged,
            const auto& residual) {
          *operand = residual + res_ratio * *operand;
          ++(*local_iteration_id);
          *local_has_converged = std::move(has_converged);
        },
        make_not_null(&box), get<residual_tag>(box));

    // Repeat steps until the solve has converged
    constexpr size_t this_action_index =
        tmpl::index_of<ActionList, UpdateOperand>::value;
    constexpr size_t prepare_step_index =
        tmpl::index_of<ActionList, InitializeHasConverged<
                                       FieldsTag, OptionsGroup, Label>>::value +
        1;
    return {Parallel::AlgorithmExecution::Continue,
            get<Convergence::Tags::HasConverged<OptionsGroup>>(box)
                ? (this_action_index + 1)
                : prepare_step_index};
  }
};

}  // namespace LinearSolver::cg::detail
