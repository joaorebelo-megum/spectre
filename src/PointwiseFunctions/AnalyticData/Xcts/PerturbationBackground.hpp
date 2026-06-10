// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/Binary.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Factory.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Flatness.hpp"
#include "PointwiseFunctions/InitialDataUtilities/AnalyticSolution.hpp"
#include "PointwiseFunctions/InitialDataUtilities/NumericData.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"

namespace Xcts::AnalyticData {

namespace detail {

using perturbation_background_classes = tmpl::push_back<
    Xcts::Solutions::all_analytic_solutions,
    Xcts::AnalyticData::Binary<elliptic::analytic_data::AnalyticSolution,
                               Xcts::Solutions::all_analytic_solutions>,
    elliptic::analytic_data::NumericData>;

inline std::array<bool, 6> parse_perturbation_components(
    const std::vector<std::string>& components) {
  std::array<bool, 6> selected{};
  for (const auto& component : components) {
    if (component == "xx") {
      selected[0] = true;
    } else if (component == "xy" or component == "yx") {
      selected[1] = true;
    } else if (component == "xz" or component == "zx") {
      selected[2] = true;
    } else if (component == "yy") {
      selected[3] = true;
    } else if (component == "yz" or component == "zy") {
      selected[4] = true;
    } else if (component == "zz") {
      selected[5] = true;
    } else {
      ERROR("Unknown perturbation component '" << component
                                               << "'. Valid values are xx, xy, "
                                               << "xz, yy, yz, and zz.");
    }
  }
  return selected;
}

template <typename DataType>
inline auto gaussian_profile(const tnsr::I<DataType, 3, Frame::Inertial>& x,
                             const double amplitude, const double sigma) {
  const auto r_squared =
      get<0>(x) * get<0>(x) + get<1>(x) * get<1>(x) + get<2>(x) * get<2>(x);
  return amplitude * exp(-r_squared / (sigma * sigma));
}

}  // namespace detail

class PerturbationBackground
    : public elliptic::analytic_data::AnalyticSolution {
 public:
  struct Background {
    static constexpr Options::String help =
        "The analytic or numeric XCTS background source to perturb.";
    using type = std::unique_ptr<elliptic::analytic_data::Background>;
  };
  struct Amplitude {
    static constexpr Options::String help =
        "Amplitude A of the Gaussian conformal-metric perturbation.";
    using type = double;
  };
  struct Sigma {
    static constexpr Options::String help =
        "Width sigma of the Gaussian conformal-metric perturbation.";
    using type = double;
  };
  struct Components {
    static constexpr Options::String help =
        "Conformal-metric component labels to perturb, e.g. [xx, xy].";
    using type = std::vector<std::string>;
  };
  using options = tmpl::list<Background, Amplitude, Sigma, Components>;
  static constexpr Options::String help{
      "Wrap a background XCTS source and add a configurable Gaussian "
      "perturbation to its conformal metric."};

  PerturbationBackground()
      : background_{std::make_unique<Xcts::Solutions::Flatness>()} {}

  PerturbationBackground(
      std::unique_ptr<elliptic::analytic_data::Background> background,
      double amplitude, double sigma, std::vector<std::string> components)
      : background_{background != nullptr
                        ? std::move(background)
                        : std::make_unique<Xcts::Solutions::Flatness>()},
        amplitude_{amplitude},
        sigma_{sigma},
        selected_components_{
            detail::parse_perturbation_components(components)} {
    if (sigma_ <= 0.0) {
      ERROR("Perturbation sigma must be positive.");
    }
  }

  PerturbationBackground(const PerturbationBackground& rhs)
      : elliptic::analytic_data::AnalyticSolution(rhs),
        background_{serialize_and_deserialize(rhs.background_)},
        amplitude_{rhs.amplitude_},
        sigma_{rhs.sigma_},
        selected_components_{rhs.selected_components_} {}

  PerturbationBackground& operator=(const PerturbationBackground& rhs) {
    background_ = serialize_and_deserialize(rhs.background_);
    amplitude_ = rhs.amplitude_;
    sigma_ = rhs.sigma_;
    selected_components_ = rhs.selected_components_;
    return *this;
  }

  PerturbationBackground(PerturbationBackground&&) = default;
  PerturbationBackground& operator=(PerturbationBackground&&) = default;
  ~PerturbationBackground() override = default;

  std::unique_ptr<elliptic::analytic_data::AnalyticSolution> get_clone()
      const override {
    return std::make_unique<PerturbationBackground>(*this);
  }

  explicit PerturbationBackground(CkMigrateMessage* msg)
      : elliptic::analytic_data::AnalyticSolution(msg) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(PerturbationBackground);

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<RequestedTags...> /*meta*/) const {
    auto result =
        call_with_dynamic_type<tuples::TaggedTuple<RequestedTags...>,
                               detail::perturbation_background_classes>(
            background_.get(), [&x](const auto* const derived) {
              return derived->variables(x, tmpl::list<RequestedTags...>{});
            });
    perturb_metric(make_not_null(&result), x);
    return result;
  }

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x, const Mesh<3>& mesh,
      const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                            Frame::Inertial>& inv_jacobian,
      tmpl::list<RequestedTags...> /*meta*/) const {
    auto result =
        call_with_dynamic_type<tuples::TaggedTuple<RequestedTags...>,
                               detail::perturbation_background_classes>(
            background_.get(),
            [&x, &mesh, &inv_jacobian](const auto* const derived) {
              return derived->variables(x, mesh, inv_jacobian,
                                        tmpl::list<RequestedTags...>{});
            });
    perturb_metric(make_not_null(&result), x);
    return result;
  }

  void pup(PUP::er& p) override {
    elliptic::analytic_data::AnalyticSolution::pup(p);
    p | background_;
    p | amplitude_;
    p | sigma_;
    p | selected_components_;
  }

 private:
  template <typename DataType, typename... RequestedTags>
  void perturb_metric(
      gsl::not_null<tuples::TaggedTuple<RequestedTags...>*> result,
      const tnsr::I<DataType, 3, Frame::Inertial>& x) const {
    using conformal_metric_tag =
        Xcts::Tags::ConformalMetric<DataType, 3, Frame::Inertial>;
    using inverse_conformal_metric_tag =
        Xcts::Tags::InverseConformalMetric<DataType, 3, Frame::Inertial>;
    if constexpr (tmpl::list_contains_v<tmpl::list<RequestedTags...>,
                                        conformal_metric_tag>) {
      auto metric = get<conformal_metric_tag>(*result);
      const auto profile = detail::gaussian_profile(x, amplitude_, sigma_);
      if (selected_components_[0]) {
        get<0, 0>(metric) += profile;
      }
      if (selected_components_[1]) {
        get<0, 1>(metric) += profile;
        get<1, 0>(metric) += profile;
      }
      if (selected_components_[2]) {
        get<0, 2>(metric) += profile;
        get<2, 0>(metric) += profile;
      }
      if (selected_components_[3]) {
        get<1, 1>(metric) += profile;
      }
      if (selected_components_[4]) {
        get<1, 2>(metric) += profile;
        get<2, 1>(metric) += profile;
      }
      if (selected_components_[5]) {
        get<2, 2>(metric) += profile;
      }
      get<conformal_metric_tag>(*result) = metric;
      if constexpr (tmpl::list_contains_v<tmpl::list<RequestedTags...>,
                                          inverse_conformal_metric_tag>) {
        get<inverse_conformal_metric_tag>(*result) =
            determinant_and_inverse(metric).second;
      }
    }
  }

  std::unique_ptr<elliptic::analytic_data::Background> background_{
      std::make_unique<Xcts::Solutions::Flatness>()};
  double amplitude_{0.0};
  double sigma_{1.0};
  std::array<bool, 6> selected_components_{};
};

}  // namespace Xcts::AnalyticData
