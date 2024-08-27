// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <type_traits>

#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Transpose.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Blas.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"

/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Compute the weak form divergence of fluxes
 *
 * In a discontinuous Galerkin scheme we integrate the equations against the
 * basis functions over the element. For the flux divergence term this gives:
 *
 * \f{align*}{
 *  \int_{\Omega}d^n x \phi_{\breve{\imath}}\partial_i F^i,
 * \f}
 *
 * where the basis functions are denoted by \f$\phi_{\breve{\imath}}\f$.
 *
 * Integrating by parts we get
 *
 * \f{align*}{
 *  \int_{\Omega}d^n x\, \phi_{\breve{\imath}}\partial_i F^i
 *  = -\int_{\Omega}d^n x\,  F^i \partial_i \phi_{\breve{\imath}}
 *   + \int_{\partial\Omega} d^{(n-1)}\Sigma\, n_i F^i \phi_{\breve{\imath}}
 * \f}
 *
 * Next we expand the flux \f$F^i\f$ in terms of the basis functions, yielding
 *
 * \f{align*}{
 *  - \int_{\Omega}d^n x\,F^i_{\breve{\jmath}} \phi_{\breve{\jmath}} \partial_i
 *    \phi_{\breve{\imath}}
 *  + \int_{\partial\Omega} d^{(n-1)}\Sigma\, n_i F^i_{\breve{\jmath}}
 *    \phi_{\breve{\jmath}} \phi_{\breve{\imath}}
 * \f}
 *
 * This function computes the volume term:
 *
 * \f{align*}{
 * \frac{1}{w_\breve{\imath}} \int_{\Omega}d^n x \,
 *   F^i_{\breve{\jmath}} \phi_{\breve{\jmath}} \partial_i \phi_{\breve{\imath}}
 * \f}
 *
 * \note When using Gauss-Lobatto points the numerical values of the
 * divergence are the same for the strong and weak divergence at the interior
 * points. When using Gauss points they are only the same at the central grid
 * point and only when an odd number of grid points is used.
 */
template <typename... ResultTags, typename... FluxTags, size_t Dim>
void weak_divergence(
    const gsl::not_null<Variables<tmpl::list<ResultTags...>>*>
        divergence_of_fluxes,
    const Variables<tmpl::list<FluxTags...>>& fluxes, const Mesh<Dim>& mesh,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          Frame::Inertial>& det_jac_times_inverse_jacobian) {
  if (UNLIKELY(divergence_of_fluxes->number_of_grid_points() !=
               fluxes.number_of_grid_points())) {
    divergence_of_fluxes->initialize(fluxes.number_of_grid_points());
  }

  using ValueType = typename Variables<tmpl::list<FluxTags...>>::value_type;

  if constexpr (Dim == 1) {
    (void)det_jac_times_inverse_jacobian;  // is identically 1.0 in 1d

    partial_derivatives_detail::apply_matrix_in_first_dim(
        divergence_of_fluxes->data(), fluxes.data(),
        Spectral::weak_flux_differentiation_matrix(mesh), fluxes.size(), false);
  } else {
    // Multiplies the flux by det_jac_time_inverse_jacobian.
    const auto transform_to_logical_frame =
        [&det_jac_times_inverse_jacobian, &fluxes](
            auto flux_tag_v, auto div_tag_v,
            const gsl::not_null<Variables<tmpl::list<ResultTags...>>*>
                result_buffer,
            auto logical_index_of_jacobian) {
          using flux_tag = tmpl::type_from<decltype(flux_tag_v)>;
          using div_tag = tmpl::type_from<decltype(div_tag_v)>;
          using TensorIndexType =
              std::array<size_t, div_tag::type::num_tensor_indices>;
          using FluxIndexType =
              std::array<size_t, flux_tag::type::num_tensor_indices>;

          auto& result = get<div_tag>(*result_buffer);
          const auto& flux = get<flux_tag>(fluxes);

          for (size_t result_storage_index = 0;
               result_storage_index < result.size(); ++result_storage_index) {
            const TensorIndexType result_tensor_index =
                result.get_tensor_index(result_storage_index);
            const FluxIndexType flux_x_tensor_index =
                prepend(result_tensor_index, 0_st);
            const FluxIndexType flux_y_tensor_index =
                prepend(result_tensor_index, 1_st);
            if constexpr (Dim == 2) {
              result[result_storage_index] =
                  get<std::decay_t<decltype(logical_index_of_jacobian)>::value,
                      0>(det_jac_times_inverse_jacobian) *
                      flux.get(flux_x_tensor_index) +
                  get<std::decay_t<decltype(logical_index_of_jacobian)>::value,
                      1>(det_jac_times_inverse_jacobian) *
                      flux.get(flux_y_tensor_index);
            } else {
              const FluxIndexType flux_z_tensor_index =
                  prepend(result_tensor_index, 2_st);
              result[result_storage_index] =
                  get<std::decay_t<decltype(logical_index_of_jacobian)>::value,
                      0>(det_jac_times_inverse_jacobian) *
                      flux.get(flux_x_tensor_index) +
                  get<std::decay_t<decltype(logical_index_of_jacobian)>::value,
                      1>(det_jac_times_inverse_jacobian) *
                      flux.get(flux_y_tensor_index) +
                  get<std::decay_t<decltype(logical_index_of_jacobian)>::value,
                      2>(det_jac_times_inverse_jacobian) *
                      flux.get(flux_z_tensor_index);
            }
          }
        };

    if constexpr (Dim == 2) {
      Variables<tmpl::list<ResultTags...>> data_buffer{
          divergence_of_fluxes->number_of_grid_points()};

      const Matrix& eta_weak_div_matrix =
          Spectral::weak_flux_differentiation_matrix(mesh.slice_through(1));
      const Matrix& xi_weak_div_matrix =
          Spectral::weak_flux_differentiation_matrix(mesh.slice_through(0));

      // Compute the eta divergence term. Since that also needs a transpose,
      // copy into result, then transpose into `data_buffer`
      EXPAND_PACK_LEFT_TO_RIGHT(transform_to_logical_frame(
          tmpl::type_<FluxTags>{}, tmpl::type_<ResultTags>{},
          make_not_null(&data_buffer), std::integral_constant<size_t, 1>{}));
      ValueType* div_ptr = divergence_of_fluxes->data();
      raw_transpose(make_not_null(div_ptr), data_buffer.data(),
                    xi_weak_div_matrix.rows(),
                    divergence_of_fluxes->size() / xi_weak_div_matrix.rows());
      partial_derivatives_detail::apply_matrix_in_first_dim(
          data_buffer.data(), divergence_of_fluxes->data(), eta_weak_div_matrix,
          data_buffer.size(), false);

      const size_t chunk_size = Variables<tmpl::list<Tags::div<FluxTags>...>>::
                                    number_of_independent_components *
                                eta_weak_div_matrix.rows();
      raw_transpose(make_not_null(div_ptr), data_buffer.data(), chunk_size,
                    data_buffer.size() / chunk_size);

      // Now compute xi divergence and *add* to eta divergence
      EXPAND_PACK_LEFT_TO_RIGHT(transform_to_logical_frame(
          tmpl::type_<FluxTags>{}, tmpl::type_<ResultTags>{},
          make_not_null(&data_buffer), std::integral_constant<size_t, 0>{}));
      partial_derivatives_detail::apply_matrix_in_first_dim(
          divergence_of_fluxes->data(), data_buffer.data(), xi_weak_div_matrix,
          data_buffer.size(), true);
    } else if constexpr (Dim == 3) {
      Variables<tmpl::list<ResultTags...>> data_buffer0{
          divergence_of_fluxes->number_of_grid_points()};
      Variables<tmpl::list<ResultTags...>> data_buffer1{
          divergence_of_fluxes->number_of_grid_points()};
      constexpr size_t number_of_independent_components =
          decltype(data_buffer1)::number_of_independent_components;

      const Matrix& zeta_weak_div_matrix =
          Spectral::weak_flux_differentiation_matrix(mesh.slice_through(2));
      const Matrix& eta_weak_div_matrix =
          Spectral::weak_flux_differentiation_matrix(mesh.slice_through(1));
      const Matrix& xi_weak_div_matrix =
          Spectral::weak_flux_differentiation_matrix(mesh.slice_through(0));

      // Compute the zeta divergence term. Since that also needs a transpose,
      // copy into data_buffer0, then transpose into `data_buffer1`.
      EXPAND_PACK_LEFT_TO_RIGHT(transform_to_logical_frame(
          tmpl::type_<FluxTags>{}, tmpl::type_<ResultTags>{},
          make_not_null(&data_buffer0), std::integral_constant<size_t, 2>{}));
      size_t chunk_size =
          xi_weak_div_matrix.rows() * eta_weak_div_matrix.rows();
      ValueType* result_ptr = data_buffer1.data();
      raw_transpose(make_not_null(result_ptr), data_buffer0.data(), chunk_size,
                    data_buffer0.size() / chunk_size);
      partial_derivatives_detail::apply_matrix_in_first_dim(
          data_buffer0.data(), data_buffer1.data(), zeta_weak_div_matrix,
          data_buffer0.size(), false);
      chunk_size =
          number_of_independent_components * zeta_weak_div_matrix.rows();
      result_ptr = divergence_of_fluxes->data();
      raw_transpose(make_not_null(result_ptr), data_buffer0.data(), chunk_size,
                    data_buffer1.size() / chunk_size);

      // Compute the eta divergence term. Since that also needs a transpose,
      // copy into data_buffer0, then transpose into `data_buffer1`.
      EXPAND_PACK_LEFT_TO_RIGHT(transform_to_logical_frame(
          tmpl::type_<FluxTags>{}, tmpl::type_<ResultTags>{},
          make_not_null(&data_buffer0), std::integral_constant<size_t, 1>{}));
      chunk_size = xi_weak_div_matrix.rows();
      result_ptr = data_buffer1.data();
      raw_transpose(make_not_null(result_ptr), data_buffer0.data(), chunk_size,
                    data_buffer1.size() / chunk_size);
      partial_derivatives_detail::apply_matrix_in_first_dim(
          data_buffer0.data(), data_buffer1.data(), eta_weak_div_matrix,
          data_buffer0.size(), false);
      chunk_size = number_of_independent_components *
                   eta_weak_div_matrix.rows() * zeta_weak_div_matrix.rows();
      result_ptr = data_buffer1.data();
      raw_transpose(make_not_null(result_ptr), data_buffer0.data(), chunk_size,
                    data_buffer0.size() / chunk_size);
      *divergence_of_fluxes += data_buffer1;

      // Now compute xi divergence and *add* to eta divergence
      EXPAND_PACK_LEFT_TO_RIGHT(transform_to_logical_frame(
          tmpl::type_<FluxTags>{}, tmpl::type_<ResultTags>{},
          make_not_null(&data_buffer0), std::integral_constant<size_t, 0>{}));
      partial_derivatives_detail::apply_matrix_in_first_dim(
          divergence_of_fluxes->data(), data_buffer0.data(), xi_weak_div_matrix,
          data_buffer0.size(), true);
    } else {
      static_assert(Dim == 1 or Dim == 2 or Dim == 3,
                    "Weak divergence only implemented in 1d, 2d, and 3d.");
    }
  }
}

/// @{
/*!
 * \brief Compute the weak divergence of fluxes in logical coordinates
 *
 * Applies the transpose logical differentiation matrix to the fluxes in each
 * dimension and sums over dimensions.
 *
 * \see weak_divergence
 */
template <typename ResultTensor, typename FluxTensor, size_t Dim>
void logical_weak_divergence(gsl::not_null<ResultTensor*> div_flux,
                             const FluxTensor& flux, const Mesh<Dim>& mesh);

/*!
 * \param divergence_of_fluxes Result buffer. Will be resized if necessary.
 * \param fluxes The fluxes to take the divergence of. Their first index must be
 * an upper spatial index in element-logical coordinates.
 * \param mesh The element mesh.
 * \param add_to_buffer If `true`, add the divergence to the existing values in
 * `divergence_of_fluxes`. Otherwise, overwrite the existing values.
 */
template <typename... ResultTags, typename... FluxTags, size_t Dim>
void logical_weak_divergence(
    const gsl::not_null<Variables<tmpl::list<ResultTags...>>*>
        divergence_of_fluxes,
    const Variables<tmpl::list<FluxTags...>>& fluxes, const Mesh<Dim>& mesh,
    const bool add_to_buffer = false) {
  static_assert(
      (... and
       std::is_same_v<tmpl::front<typename FluxTags::type::index_list>,
                      SpatialIndex<Dim, UpLo::Up, Frame::ElementLogical>>),
      "The first index of each flux must be an upper spatial index in "
      "element-logical coordinates.");
  static_assert(
      (... and
       std::is_same_v<
           typename ResultTags::type,
           TensorMetafunctions::remove_first_index<typename FluxTags::type>>),
      "The result tensors must have the same type as the flux tensors with "
      "their first index removed.");
  if (not add_to_buffer) {
    divergence_of_fluxes->initialize(mesh.number_of_grid_points(), 0.);
  }
  EXPAND_PACK_LEFT_TO_RIGHT(logical_weak_divergence(
      make_not_null(&get<ResultTags>(*divergence_of_fluxes)),
      get<FluxTags>(fluxes), mesh));
}
/// @}
