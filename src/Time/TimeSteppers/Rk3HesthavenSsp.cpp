// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Time/TimeSteppers/Rk3HesthavenSsp.hpp"

#include <cmath>
#include <optional>

#include "Time/EvolutionOrdering.hpp"
#include "Time/History.hpp"
#include "Time/LargestStepperError.hpp"
#include "Time/StepperErrorEstimate.hpp"
#include "Time/StepperErrorTolerances.hpp"
#include "Time/Time.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace TimeSteppers {

size_t Rk3HesthavenSsp::order() const { return 3; }

double Rk3HesthavenSsp::stable_step() const {
  // This is the condition for  y' = -k y  to go to zero.
  return 0.5 * (1. + cbrt(4. + sqrt(17.)) - 1. / cbrt(4. + sqrt(17.)));
}

bool Rk3HesthavenSsp::monotonic() const { return false; }

uint64_t Rk3HesthavenSsp::number_of_substeps() const { return 3; }

uint64_t Rk3HesthavenSsp::number_of_substeps_for_error() const { return 3; }

size_t Rk3HesthavenSsp::number_of_past_steps() const { return 0; }

TimeStepId Rk3HesthavenSsp::next_time_id(const TimeStepId& current_id,
                                         const TimeDelta& time_step) const {
  switch (current_id.substep()) {
    case 0:
      return current_id.next_substep(time_step, 1.0);
    case 1:
      return current_id.next_substep(time_step, 0.5);
    case 2:
      return current_id.next_step(time_step);
    default:
      ERROR("Bad id: " << current_id);
  }
}

TimeStepId Rk3HesthavenSsp::next_time_id_for_error(
    const TimeStepId& current_id, const TimeDelta& time_step) const {
  return next_time_id(current_id, time_step);
}

template <typename T>
void Rk3HesthavenSsp::update_u_impl(const gsl::not_null<T*> u,
                                    const ConstUntypedHistory<T>& history,
                                    const TimeDelta& time_step) const {
  ASSERT(history.integration_order() == order(),
         "Fixed-order stepper cannot run at order "
             << history.integration_order());

  const auto substep = history.at_step_start() ? 0 : history.substeps().size();
  switch (substep) {
    case 0:
      *u =
          *history.back().value + time_step.value() * history.back().derivative;
      return;
    case 1:
      *u = 0.25 * (3.0 * *history.back().value + *history.substeps()[0].value +
                   time_step.value() * history.substeps()[0].derivative);
      return;
    case 2:
      *u = (1.0 / 3.0) *
           (*history.back().value + 2.0 * *history.substeps()[1].value +
            2.0 * time_step.value() * history.substeps()[1].derivative);
      return;
    default:
      ERROR("Bad substep: " << history.substeps().size());
  }
}

template <typename T>
std::optional<StepperErrorEstimate> Rk3HesthavenSsp::update_u_impl(
    gsl::not_null<T*> u, const ConstUntypedHistory<T>& history,
    const TimeDelta& time_step,
    const std::optional<StepperErrorTolerances>& tolerances) const {
  ASSERT(history.integration_order() == order(),
         "Fixed-order stepper cannot run at order "
             << history.integration_order());

  std::optional<StepperErrorEstimate> error{};
  if (not history.at_step_start() and history.substeps().size() == 2 and
      tolerances.has_value()) {
    *u = -(1.0 / 6.0) * *history.back().value +
         (2.0 / 3.0) * *history.substeps()[1].value +
         (2.0 / 3.0) * time_step.value() * history.substeps()[1].derivative -
         0.5 * *history.substeps()[0].value -
         0.5 * time_step.value() * history.substeps()[0].derivative;
    error.emplace(StepperErrorEstimate{
        history.back().time_step_id.step_time(), time_step, 2,
        largest_stepper_error(*history.back().value, *u, *tolerances)});
  }

  update_u_impl(u, history, time_step);
  return error;
}

template <typename T>
void Rk3HesthavenSsp::clean_history_impl(
    const MutableUntypedHistory<T>& history) const {
  if (history.at_step_start()) {
    history.clear_substeps();
  }
  if (history.size() > 1) {
    history.pop_front();
  }
  ASSERT(history.size() == 1, "Too much history supplied.");
}

template <typename T>
bool Rk3HesthavenSsp::dense_update_u_impl(const gsl::not_null<T*> u,
                                          const ConstUntypedHistory<T>& history,
                                          const double time) const {
  if (not history.at_step_start()) {
    return false;
  }
  const double step_start = history.front().time_step_id.step_time().value();
  const double step_end = history.back().time_step_id.step_time().value();
  const evolution_less<double> before{step_end > step_start};
  if (history.size() == 1 or before(step_end, time)) {
    return false;
  }
  const double step_size = step_end - step_start;
  const double output_fraction = (time - step_start) / step_size;
  ASSERT(output_fraction >= 0.0, "Attempting dense output at time "
                                     << time << ", but already progressed past "
                                     << step_start);

  *u += -(2.0 / 3.0) * square(output_fraction) * *history.front().value +
        output_fraction * (1.0 - output_fraction) * step_size *
            history.front().derivative +
        (2.0 / 3.0) * square(output_fraction) *
            (*history.substeps()[1].value +
             step_size * history.substeps()[1].derivative);

  return true;
}

template <typename T>
bool Rk3HesthavenSsp::can_change_step_size_impl(
    const TimeStepId& /*time_id*/,
    const ConstUntypedHistory<T>& /*history*/) const {
  return true;
}

TIME_STEPPER_DEFINE_OVERLOADS(Rk3HesthavenSsp)
}  // namespace TimeSteppers

PUP::able::PUP_ID TimeSteppers::Rk3HesthavenSsp::my_PUP_ID = 0;  // NOLINT
