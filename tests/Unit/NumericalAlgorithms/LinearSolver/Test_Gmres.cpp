// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#if defined(__GNUC__) and not defined(__clang__) and __GNUC__ == 12
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
#include "Framework/TestCreation.hpp"
#if defined(__GNUC__) and not defined(__clang__) and __GNUC__ == 12
#pragma GCC diagnostic pop
#endif

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/DynamicMatrix.hpp"
#include "DataStructures/DynamicVector.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/NumericalAlgorithms/LinearSolver/TestHelpers.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "NumericalAlgorithms/Convergence/Criteria.hpp"
#include "NumericalAlgorithms/Convergence/Reason.hpp"
#include "NumericalAlgorithms/LinearSolver/Gmres.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

namespace helpers = TestHelpers::LinearSolver;

namespace LinearSolver::Serial {

namespace {
template <typename DataType>
struct ScalarField : db::SimpleTag {
  using type = Scalar<DataType>;
};
template <typename Tag>
struct SomePrefix : db::PrefixTag, db::SimpleTag {
  using type = tmpl::type_from<Tag>;
  using tag = Tag;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.LinearSolver.Serial.Gmres",
                  "[Unit][NumericalAlgorithms][LinearSolver]") {
  {
    // [gmres_example]
    INFO("Solve a symmetric 2x2 matrix");
    blaze::DynamicMatrix<double> matrix{{4., 1.}, {1., 3.}};
    const helpers::ApplyMatrix<double> linear_operator{std::move(matrix)};
    const blaze::DynamicVector<double> source{1., 2.};
    blaze::DynamicVector<double> initial_guess_in_solution_out{2., 1.};
    const blaze::DynamicVector<double> expected_solution{0.0909090909090909,
                                                         0.6363636363636364};
    const Convergence::Criteria convergence_criteria{2, 1.e-14, 0.};
    const Gmres<blaze::DynamicVector<double>> gmres{convergence_criteria,
                                                    ::Verbosity::Verbose};
    CHECK_FALSE(gmres.has_preconditioner());
    std::vector<double> recorded_residuals;
    const auto has_converged =
        gmres.solve(make_not_null(&initial_guess_in_solution_out),
                    linear_operator, source, std::tuple{},
                    [&recorded_residuals](
                        const Convergence::HasConverged& local_has_converged) {
                      recorded_residuals.push_back(
                          local_has_converged.residual_magnitude());
                    });
    REQUIRE(has_converged);
    CHECK(linear_operator.invocations == 3);
    CHECK(has_converged.reason() == Convergence::Reason::AbsoluteResidual);
    CHECK(has_converged.num_iterations() == 2);
    CHECK(has_converged.residual_magnitude() <= 1.e-14);
    CHECK(has_converged.initial_residual_magnitude() ==
          approx(8.54400374531753));
    CHECK_ITERABLE_APPROX(initial_guess_in_solution_out, expected_solution);
    // The residuals should decrease monotonically
    CHECK(recorded_residuals[0] == has_converged.initial_residual_magnitude());
    for (size_t i = 1; i < has_converged.num_iterations(); ++i) {
      CHECK(recorded_residuals[i] <= recorded_residuals[i - 1]);
    }
    // [gmres_example]
    {
      INFO("Check that a solved system terminates early");
      linear_operator.invocations = 0;
      const auto second_has_converged =
          gmres.solve(make_not_null(&initial_guess_in_solution_out),
                      linear_operator, source);
      REQUIRE(second_has_converged);
      CHECK(linear_operator.invocations == 1);
      CHECK(second_has_converged.reason() ==
            Convergence::Reason::AbsoluteResidual);
      CHECK(second_has_converged.num_iterations() == 0);
      CHECK_ITERABLE_APPROX(initial_guess_in_solution_out, expected_solution);
    }
    const auto check_second_solve = [&linear_operator](
                                        const auto& local_gmres) {
      linear_operator.invocations = 0;
      blaze::DynamicVector<double> local_initial_guess_in_solution_out{0., 0.};
      const auto local_has_converged = local_gmres.solve(
          make_not_null(&local_initial_guess_in_solution_out), linear_operator,
          blaze::DynamicVector<double>{2, 1});
      REQUIRE(local_has_converged);
      // The initial guess is zero, so the initial operator should have been
      // skipped, leaving only two operator applications for two iterations
      CHECK(linear_operator.invocations == 2);
      CHECK(local_has_converged.reason() ==
            Convergence::Reason::AbsoluteResidual);
      CHECK(local_has_converged.num_iterations() == 2);
      const blaze::DynamicVector<double> expected_local_solution{
          0.454545454545455, 0.181818181818182};
      CHECK_ITERABLE_APPROX(local_initial_guess_in_solution_out,
                            expected_local_solution);
    };
    {
      INFO("Check two successive solves with different sources");
      check_second_solve(gmres);
    }
    {
      INFO("Check the solver still works after serialization");
      const auto serialized_gmres = serialize_and_deserialize(gmres);
      check_second_solve(serialized_gmres);
    }
    {
      INFO("Check the solver still works after copying");
      // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
      const auto copied_gmres = gmres;
      check_second_solve(copied_gmres);
    }
  }
  {
    INFO("Solve a non-symmetric 2x2 matrix");
    blaze::DynamicMatrix<double> matrix{{4., 1.}, {3., 1.}};
    const helpers::ApplyMatrix<double> linear_operator{std::move(matrix)};
    const blaze::DynamicVector<double> source{1., 2.};
    blaze::DynamicVector<double> initial_guess_in_solution_out{2., 1.};
    const blaze::DynamicVector<double> expected_solution{-1., 5.};
    const Convergence::Criteria convergence_criteria{2, 1.e-14, 0.};
    const Gmres<blaze::DynamicVector<double>> gmres{convergence_criteria,
                                                    ::Verbosity::Verbose};
    const auto has_converged = gmres.solve(
        make_not_null(&initial_guess_in_solution_out), linear_operator, source);
    REQUIRE(has_converged);
    CHECK(has_converged.reason() == Convergence::Reason::AbsoluteResidual);
    CHECK(has_converged.num_iterations() == 2);
    CHECK_ITERABLE_APPROX(initial_guess_in_solution_out, expected_solution);
  }
  {
    INFO("Solve a complex 2x2 matrix");
    blaze::DynamicMatrix<std::complex<double>> matrix{
        {std::complex<double>(1., 2.), std::complex<double>(2., -1.)},
        {std::complex<double>(3., 4.), std::complex<double>(4., 1.)}};
    const helpers::ApplyMatrix<std::complex<double>> linear_operator{
        std::move(matrix)};
    const blaze::DynamicVector<std::complex<double>> source{
        std::complex<double>(1., 1.), std::complex<double>(2., -3.)};
    blaze::DynamicVector<std::complex<double>> initial_guess_in_solution_out{
        0., 0.};
    const blaze::DynamicVector<std::complex<double>> expected_solution{
        std::complex<double>(0.45, -1.4), std::complex<double>(-1.2, 0.15)};
    const Convergence::Criteria convergence_criteria{2, 1.e-14, 0.};
    const Gmres<blaze::DynamicVector<std::complex<double>>> gmres{
        convergence_criteria, ::Verbosity::Verbose};
    const auto has_converged = gmres.solve(
        make_not_null(&initial_guess_in_solution_out), linear_operator, source);
    REQUIRE(has_converged);
    CHECK(has_converged.reason() == Convergence::Reason::AbsoluteResidual);
    CHECK(has_converged.num_iterations() == 2);
    CHECK_ITERABLE_APPROX(initial_guess_in_solution_out, expected_solution);
  }
  {
    INFO("Solve a matrix-free linear operator with Variables");
    using Vars = Variables<tmpl::list<ScalarField<DataVector>>>;
    constexpr size_t num_points = 2;
    // This also tests that the linear operator can be a lambda
    const auto linear_operator = [](const gsl::not_null<Vars*> result,
                                    const Vars& operand) {
      if (result->number_of_grid_points() != num_points) {
        result->initialize(num_points);
      }
      const auto& data = get(get<ScalarField<DataVector>>(operand));
      get(get<ScalarField<DataVector>>(*result)) =
          DataVector{data[0] * 4. + data[1], data[0] * 3. + data[1]};
    };
    // Adding a prefix to make sure prefixed sources work as well
    Variables<tmpl::list<SomePrefix<ScalarField<DataVector>>>> source{
        num_points};
    get(get<SomePrefix<ScalarField<DataVector>>>(source)) = DataVector{1., 2.};
    Vars initial_guess_in_solution_out{num_points};
    get(get<ScalarField<DataVector>>(initial_guess_in_solution_out)) =
        DataVector{2., 1.};
    const DataVector expected_solution{-1., 5.};
    const Convergence::Criteria convergence_criteria{2, 1.e-14, 0.};
    const Gmres<Vars> gmres{convergence_criteria, ::Verbosity::Verbose};
    const auto has_converged = gmres.solve(
        make_not_null(&initial_guess_in_solution_out), linear_operator, source);
    REQUIRE(has_converged);
    CHECK(has_converged.reason() == Convergence::Reason::AbsoluteResidual);
    CHECK(has_converged.num_iterations() == 2);
    CHECK_ITERABLE_APPROX(
        get(get<ScalarField<DataVector>>(initial_guess_in_solution_out)),
        expected_solution);
  }
  {
    INFO("Solve a complex matrix-free linear operator with Variables");
    using Vars = Variables<tmpl::list<ScalarField<ComplexDataVector>>>;
    constexpr size_t num_points = 2;
    const auto linear_operator = [](const gsl::not_null<Vars*> result,
                                    const Vars& operand) {
      if (result->number_of_grid_points() != num_points) {
        result->initialize(num_points);
      }
      const auto& data = get(get<ScalarField<ComplexDataVector>>(operand));
      get(get<ScalarField<ComplexDataVector>>(*result)) =
          ComplexDataVector{data[0] * std::complex<double>(1., 2.) +
                                data[1] * std::complex<double>(2., -1.),
                            data[0] * std::complex<double>(3., 4.) +
                                data[1] * std::complex<double>(4., 1.)};
    };
    Variables<tmpl::list<SomePrefix<ScalarField<ComplexDataVector>>>> source{
        num_points};
    get(get<SomePrefix<ScalarField<ComplexDataVector>>>(source)) =
        ComplexDataVector{std::complex<double>(1., 1.),
                          std::complex<double>(2., -3.)};
    Vars initial_guess_in_solution_out{num_points};
    get(get<ScalarField<ComplexDataVector>>(initial_guess_in_solution_out)) =
        ComplexDataVector{std::complex<double>(0., 0.),
                          std::complex<double>(0., 0.)};
    const ComplexDataVector expected_solution{std::complex<double>(0.45, -1.4),
                                              std::complex<double>(-1.2, 0.15)};
    const Convergence::Criteria convergence_criteria{2, 1.e-14, 0.};
    const Gmres<Vars> gmres{convergence_criteria, ::Verbosity::Verbose};
    const auto has_converged = gmres.solve(
        make_not_null(&initial_guess_in_solution_out), linear_operator, source);
    REQUIRE(has_converged);
    CHECK(has_converged.reason() == Convergence::Reason::AbsoluteResidual);
    CHECK(has_converged.num_iterations() == 2);
    CHECK_ITERABLE_APPROX(
        get(get<ScalarField<ComplexDataVector>>(initial_guess_in_solution_out)),
        expected_solution);
  }
  {
    INFO("Restarting");
    blaze::DynamicMatrix<double> matrix{
        {4., 1., 1.}, {1., 1., 3.}, {0., 2., 0.}};
    const helpers::ApplyMatrix<double> linear_operator{std::move(matrix)};
    const blaze::DynamicVector<double> source{1., 2., 1.};
    blaze::DynamicVector<double> initial_guess_in_solution_out{2., 1., 0.};
    const blaze::DynamicVector<double> expected_solution{0., 0.5, 0.5};
    const Convergence::Criteria convergence_criteria{100, 1.e-14, 0.};
    // Restart every other iteration. The algorithm would converge in 3
    // iterations without restarting, so restarting is of course ridiculously
    // inefficient for this problem size. The number of iterations rises to 59.
    const size_t restart = 2;
    const Gmres<blaze::DynamicVector<double>> gmres{
        convergence_criteria, ::Verbosity::Verbose, restart};
    const auto has_converged = gmres.solve(
        make_not_null(&initial_guess_in_solution_out), linear_operator, source);
    REQUIRE(has_converged);
    CHECK(has_converged.reason() == Convergence::Reason::AbsoluteResidual);
    CHECK(has_converged.num_iterations() == 59);
    CHECK_ITERABLE_APPROX(initial_guess_in_solution_out, expected_solution);
  }
  {
    INFO("Preconditioning");
    blaze::DynamicMatrix<double> matrix{{4., 1.}, {1., 3.}};
    const helpers::ApplyMatrix<double> linear_operator{std::move(matrix)};
    const blaze::DynamicVector<double> source{1., 2.};
    const blaze::DynamicVector<double> expected_solution{0.0909090909090909,
                                                         0.6363636363636364};
    const Convergence::Criteria convergence_criteria{2, 1.e-14, 0.};
    const auto check_solve = [&linear_operator, &source, &expected_solution](
                                 const auto& local_gmres,
                                 const size_t expected_num_iterations) {
      REQUIRE(local_gmres.has_preconditioner());
      blaze::DynamicVector<double> local_initial_guess_in_solution_out{2., 1.};
      std::vector<double> local_recorded_residuals;
      const auto local_has_converged = local_gmres.solve(
          make_not_null(&local_initial_guess_in_solution_out), linear_operator,
          source, std::tuple{},
          [&local_recorded_residuals](
              const Convergence::HasConverged& recorded_has_converged) {
            local_recorded_residuals.push_back(
                recorded_has_converged.residual_magnitude());
          });
      CAPTURE(local_recorded_residuals);
      REQUIRE(local_has_converged);
      CHECK(local_has_converged.reason() ==
            Convergence::Reason::AbsoluteResidual);
      CHECK(local_has_converged.num_iterations() == expected_num_iterations);
      CHECK_ITERABLE_APPROX(local_initial_guess_in_solution_out,
                            expected_solution);
    };
    {
      INFO("Exact inverse preconditioner");
      // Use the exact inverse of the matrix as preconditioner. This
      // should solve the problem in 1 iteration.
      helpers::ExactInversePreconditioner preconditioner{};
      const Gmres<blaze::DynamicVector<double>,
                  helpers::ExactInversePreconditioner>
          preconditioned_gmres{convergence_criteria, ::Verbosity::Verbose,
                               std::nullopt, std::move(preconditioner)};
      check_solve(preconditioned_gmres, 1);
      // Check a second solve with the same solver and preconditioner works
      check_solve(preconditioned_gmres, 1);
      {
        INFO("Check that serialization preserves the preconditioner");
        const auto serialized_gmres =
            serialize_and_deserialize(preconditioned_gmres);
        check_solve(serialized_gmres, 1);
      }
      {
        INFO("Check that copying preserves the preconditioner");
        // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
        const auto copied_gmres = preconditioned_gmres;
        check_solve(copied_gmres, 1);
      }
    }
    {
      INFO("Diagonal (Jacobi) preconditioner");
      // Use the inverse of the diagonal as preconditioner.
      helpers::JacobiPreconditioner preconditioner{};
      const Gmres<blaze::DynamicVector<double>, helpers::JacobiPreconditioner>
          preconditioned_gmres{convergence_criteria, ::Verbosity::Verbose,
                               std::nullopt, std::move(preconditioner)};
      check_solve(preconditioned_gmres, 2);
    }
    {
      INFO("Richardson preconditioner");
      helpers::RichardsonPreconditioner preconditioner{
          // The optimal relaxation parameter for SPD matrices is 2 / (l_max +
          // l_min) where l_max and l_min are the largest and smallest
          // eigenvalues of the linear operator (see
          // `LinearSolver::Richardson::Richardson`).
          0.2857142857142857,
          // Run two Richardson iterations
          2};
      const Gmres<blaze::DynamicVector<double>,
                  helpers::RichardsonPreconditioner>
          preconditioned_gmres{convergence_criteria, ::Verbosity::Verbose,
                               std::nullopt, std::move(preconditioner)};
      check_solve(preconditioned_gmres, 1);
    }
    {
      INFO("Nested linear solver as preconditioner");
      // Running another GMRES solver for 2 iterations as preconditioner. It
      // should already solve the problem, so the preconditioned solve only
      // needs a single iteration.
      const Gmres<blaze::DynamicVector<double>,
                  Gmres<blaze::DynamicVector<double>>>
          preconditioned_gmres{convergence_criteria,
                               ::Verbosity::Verbose,
                               std::nullopt,
                               {{{2, 0., 0.}, ::Verbosity::Verbose}}};
      check_solve(preconditioned_gmres, 1);
    }
    {
      INFO("Nested factory-created linear solver as preconditioner");
      // Also running another GMRES solver as preconditioner, but passing it as
      // a factory-created abstract `LinearSolver` type.
      using LinearSolverRegistrars =
          tmpl::list<Registrars::Gmres<blaze::DynamicVector<double>>>;
      using LinearSolverFactory = LinearSolver<LinearSolverRegistrars>;
      const Gmres<blaze::DynamicVector<double>, LinearSolverFactory,
                  LinearSolverRegistrars>
          preconditioned_gmres{
              convergence_criteria, ::Verbosity::Verbose, std::nullopt,
              std::make_unique<
                  Gmres<blaze::DynamicVector<double>, LinearSolverFactory,
                        LinearSolverRegistrars>>(
                  Convergence::Criteria{2, 0., 0.}, ::Verbosity::Verbose)};
      check_solve(preconditioned_gmres, 1);
      {
        INFO("Check that serialization preserves the preconditioner");
        register_derived_classes_with_charm<LinearSolverFactory>();
        const auto serialized_gmres =
            serialize_and_deserialize(preconditioned_gmres);
        check_solve(serialized_gmres, 1);
      }
      {
        INFO("Check that copying preserves the preconditioner");
        register_derived_classes_with_charm<LinearSolverFactory>();
        // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
        const auto copied_gmres = preconditioned_gmres;
        check_solve(copied_gmres, 1);
      }
    }
  }
  {
    INFO("Option-creation");
    {
      const auto solver =
          TestHelpers::test_creation<Gmres<blaze::DynamicVector<double>>>(
              "ConvergenceCriteria:\n"
              "  MaxIterations: 2\n"
              "  AbsoluteResidual: 0.1\n"
              "  RelativeResidual: 0.5\n"
              "Restart: 50\n"
              "Verbosity: Verbose\n");
      CHECK(solver.convergence_criteria() ==
            Convergence::Criteria{2, 0.1, 0.5});
      CHECK(solver.restart() == 50);
      CHECK(solver.verbosity() == ::Verbosity::Verbose);
      CHECK_FALSE(solver.has_preconditioner());
    }
    {
      const auto solver = TestHelpers::test_creation<Gmres<
          blaze::DynamicVector<double>, helpers::ExactInversePreconditioner>>(
          "ConvergenceCriteria:\n"
          "  MaxIterations: 2\n"
          "  AbsoluteResidual: 0.1\n"
          "  RelativeResidual: 0.5\n"
          "Restart: None\n"
          "Verbosity: Verbose\n"
          "Preconditioner: None\n");
      CHECK(solver.convergence_criteria() ==
            Convergence::Criteria{2, 0.1, 0.5});
      CHECK(solver.restart() == 2);
      CHECK(solver.verbosity() == ::Verbosity::Verbose);
      CHECK_FALSE(solver.has_preconditioner());
    }
    {
      const auto solver = TestHelpers::test_creation<Gmres<
          blaze::DynamicVector<double>, helpers::ExactInversePreconditioner>>(
          "ConvergenceCriteria:\n"
          "  MaxIterations: 2\n"
          "  AbsoluteResidual: 0.1\n"
          "  RelativeResidual: 0.5\n"
          "Restart: None\n"
          "Verbosity: Verbose\n"
          "Preconditioner:\n");
      CHECK(solver.convergence_criteria() ==
            Convergence::Criteria{2, 0.1, 0.5});
      CHECK(solver.restart() == 2);
      CHECK(solver.verbosity() == ::Verbosity::Verbose);
      CHECK(solver.has_preconditioner());
    }
    {
      using LinearSolverRegistrars =
          tmpl::list<Registrars::Gmres<blaze::DynamicVector<double>>>;
      using LinearSolverFactory = LinearSolver<LinearSolverRegistrars>;
      const auto solver =
          TestHelpers::test_creation<std::unique_ptr<LinearSolverFactory>>(
              "Gmres:\n"
              "  ConvergenceCriteria:\n"
              "    MaxIterations: 2\n"
              "    AbsoluteResidual: 0.1\n"
              "    RelativeResidual: 0.5\n"
              "  Restart: 50\n"
              "  Verbosity: Verbose\n"
              "  Preconditioner:\n"
              "    Gmres:\n"
              "      ConvergenceCriteria:\n"
              "        MaxIterations: 1\n"
              "        AbsoluteResidual: 0.5\n"
              "        RelativeResidual: 0.9\n"
              "      Restart: None\n"
              "      Verbosity: Verbose\n"
              "      Preconditioner: None\n");
      REQUIRE(solver);
      using Derived = Gmres<blaze::DynamicVector<double>, LinearSolverFactory,
                            LinearSolverRegistrars>;
      REQUIRE_FALSE(nullptr == dynamic_cast<const Derived*>(solver.get()));
      const auto& derived = dynamic_cast<const Derived&>(*solver);
      CHECK(derived.convergence_criteria() ==
            Convergence::Criteria{2, 0.1, 0.5});
      CHECK(derived.restart() == 50);
      CHECK(derived.verbosity() == ::Verbosity::Verbose);
      REQUIRE(derived.has_preconditioner());
      REQUIRE_FALSE(nullptr ==
                    dynamic_cast<const Derived*>(&derived.preconditioner()));
      const auto& preconditioner =
          dynamic_cast<const Derived&>(derived.preconditioner());
      CHECK(preconditioner.convergence_criteria() ==
            Convergence::Criteria{1, 0.5, 0.9});
      CHECK(preconditioner.restart() == 1);
      CHECK(preconditioner.verbosity() == ::Verbosity::Verbose);
      CHECK_FALSE(preconditioner.has_preconditioner());
      {
        INFO("Copy semantics");
        // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
        const auto copied_solver = derived;
        CHECK(copied_solver.convergence_criteria() ==
              Convergence::Criteria{2, 0.1, 0.5});
        CHECK(copied_solver.restart() == 50);
        CHECK(copied_solver.verbosity() == ::Verbosity::Verbose);
        REQUIRE(copied_solver.has_preconditioner());
        REQUIRE_FALSE(nullptr == dynamic_cast<const Derived*>(
                                     &copied_solver.preconditioner()));
        const auto& copied_preconditioner =
            dynamic_cast<const Derived&>(copied_solver.preconditioner());
        CHECK(copied_preconditioner.convergence_criteria() ==
              Convergence::Criteria{1, 0.5, 0.9});
        CHECK(copied_preconditioner.restart() == 1);
        CHECK(copied_preconditioner.verbosity() == ::Verbosity::Verbose);
        CHECK_FALSE(copied_preconditioner.has_preconditioner());
      }
    }
  }
}

}  // namespace LinearSolver::Serial
