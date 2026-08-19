// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "DataStructures/CachedTempBuffer.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "IO/Exporter/PointwiseInterpolator.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/CommonVariables.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags/Conformal.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Background.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialGuess.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Xcts::AnalyticData {

namespace detail {

/// \name Post-Newtonian past evolution
///
/// The wave content added to the conformal metric depends on the state of the
/// binary at *retarded* times, so the inspiral has to be evolved backwards far
/// enough that the retarded time of the outermost grid point is covered. The
/// binary is evolved with the 3PN ADM Hamiltonian plus a 3.5PN
/// radiation-reaction force, following \cite Buonanno2006 (`gr-qc/0508067`).
///
/// The design goal is that the *only* physics transcribed from the papers is
/// the Hamiltonian itself and the energy flux. All derivatives are taken
/// numerically to machine precision, so the 3PN polynomial is never
/// differentiated by hand. See report 010 for the derivation.
/// @{

/*!
 * \brief The reduced 3PN ADM Hamiltonian \f$\hat{H} = H/\mu\f$
 *
 * The Hamiltonian depends on \f$\vec{q} = \vec{X}/M\f$ and
 * \f$\vec{p} = \vec{P}/\mu\f$ only through the three scalars
 *
 * \f{equation}
 * q = |\vec{q}|, \qquad s = \vec{p}\cdot\vec{p}, \qquad
 * z = \left(\vec{n}\cdot\vec{p}\right)^2, \qquad \vec{n} = \vec{q}/q,
 * \f}
 *
 * because \f$\vec{n}\cdot\vec{p}\f$ appears only in even powers — a consequence
 * of the time-reversal invariance of the conservative dynamics,
 * \f$\hat{H}(\vec{q},-\vec{p}) = \hat{H}(\vec{q},\vec{p})\f$. Written in terms
 * of \f$u = 1/q\f$ the result is a polynomial in \f$(u, s, z)\f$ with no
 * logarithms, so it is an entire function of each argument.
 *
 * That last property is what makes `hamiltonian_derivatives` exact: it
 * evaluates this function at a complex argument. `DataType` must therefore be
 * `double` or `std::complex<double>`, and the implementation must stay free of
 * comparisons, `abs` and branches.
 *
 * \see `Buonanno2006` equations (2)–(5), transcribed as printed.
 */
template <typename DataType>
DataType reduced_hamiltonian(const DataType& q, const DataType& s,
                             const DataType& z, double eta);

/*!
 * \brief The three partial derivatives
 * \f$\left(\hat{H}_q, \hat{H}_s, \hat{H}_z\right)\f$, by complex step
 *
 * For an entire function, \f$\hat{H}(q + ih) = \hat{H}(q) + ih\hat{H}_q +
 * O(h^2)\f$, so
 *
 * \f{equation}
 * \hat{H}_q = \frac{\mathrm{Im}\,\hat{H}(q + ih, s, z)}{h} + O(h^2).
 * \f}
 *
 * Unlike a finite difference this involves no subtractive cancellation, so
 * \f$h\f$ can be taken far below \f$\sqrt{\epsilon_\mathrm{mach}}\f$ and the
 * truncation error is irrelevant. The derivatives are exact to round-off and
 * there is no step size to tune.
 */
std::array<double, 3> hamiltonian_derivatives(double q, double s, double z,
                                              double eta);

/*!
 * \brief Derivative with respect to the momentum,
 * \f$\partial\hat{H}/\partial p_i\f$
 *
 * Obtained from the scalar derivatives by the chain rule,
 *
 * \f{equation}
 * \frac{\partial\hat{H}}{\partial p_i} = 2\hat{H}_s\,p^i
 *     + 2\hat{H}_z\left(\vec{n}\cdot\vec{p}\right)n^i.
 * \f}
 *
 * This and `hamiltonian_deriv_separation` are the only hand-derived algebra in
 * the scheme. They are validated indirectly but sharply by the conservation of
 * \f$\hat{H}\f$ and \f$\vec{L}\f$ along the conservative flow: any error here
 * produces a *linear* energy drift.
 */
std::array<double, 3> hamiltonian_deriv_momentum(
    const std::array<double, 3>& separation,
    const std::array<double, 3>& momentum, double eta);

/*!
 * \brief Derivative with respect to the separation,
 * \f$\partial\hat{H}/\partial q^i\f$
 *
 * The counterpart of `hamiltonian_deriv_momentum`,
 *
 * \f{equation}
 * \frac{\partial\hat{H}}{\partial q^i} = \hat{H}_q\,n^i
 *     + \frac{2\hat{H}_z\left(\vec{n}\cdot\vec{p}\right)}{q}
 *       \left[p^i - \left(\vec{n}\cdot\vec{p}\right)n^i\right].
 * \f}
 */
std::array<double, 3> hamiltonian_deriv_separation(
    const std::array<double, 3>& separation,
    const std::array<double, 3>& momentum, double eta);

/*!
 * \brief The orbital angular frequency \f$M\omega\f$
 *
 * \f$\omega = |\vec{q}\times\dot{\vec{q}}|/q^2\f$ with
 * \f$\dot{\vec{q}} = \partial\hat{H}/\partial\vec{p}\f$, so it comes from the
 * same evaluation that drives the equations of motion. The flux uses
 * \f$v_\omega = (M\omega)^{1/3}\f$.
 */
double orbital_frequency(const std::array<double, 3>& separation,
                         const std::array<double, 3>& momentum, double eta);

/*!
 * \brief The 3.5PN energy flux \f$dE/dt\f$, negative for an inspiral
 *
 * \f{equation}
 * \frac{dE}{dt} = -\frac{32}{5}\eta^2 v_\omega^{10}\left\{
 *   1 + f_2 v_\omega^2 + f_3 v_\omega^3 + f_4 v_\omega^4 + f_5 v_\omega^5
 *   + \left(f_6 + f_{\ell 6}\ln 4v_\omega\right)v_\omega^6
 *   + f_7 v_\omega^7\right\}
 * \f}
 *
 * \see `Buonanno2006`, transcribed as printed.
 */
double energy_flux(double v_omega, double eta);

/*!
 * \brief The right-hand side of the equations of motion
 *
 * In units \f$M = 1\f$, with the state \f$(\vec{q}, \vec{p})\f$,
 *
 * \f{align}
 * \frac{dq^i}{dt} &= \frac{\partial\hat{H}}{\partial p_i}, \\
 * \frac{dp_i}{dt} &= -\frac{\partial\hat{H}}{\partial q^i}
 *   + \frac{1}{\eta}\frac{\dot{E}}{\omega\,\ell}\,p_i,
 *   \qquad \ell = |\vec{q}\times\vec{p}|.
 * \f}
 *
 * The normalisation of the flux term is fixed by requiring
 * \f$d\hat{H}/dt = \dot{E}/\eta\f$ for circular orbits, which is the unit test
 * for the radiation-reaction sector.
 */
std::array<double, 6> inspiral_rhs(const std::array<double, 6>& state,
                                   double eta,
                                   bool with_radiation_reaction = true);

/*!
 * \brief The circular orbit at a given separation
 *
 * Returns \f$(s, M\omega)\f$ with \f$s = \vec{p}\cdot\vec{p}\f$. At
 * \f$\vec{n}\cdot\vec{p} = 0\f$ the radial balance
 * \f$d(\vec{n}\cdot\vec{p})/dt = 0\f$ reads
 *
 * \f{equation}
 * \hat{H}_q = \frac{2s\hat{H}_s}{q},
 * \f}
 *
 * which is *not* \f$\hat{H}_q = 0\f$: at fixed canonical \f$p_i\f$,
 * \f$\partial\hat{H}/\partial q\f$ is the gravitational attraction and never
 * vanishes. In the Newtonian limit this gives \f$s = 1/q\f$ and
 * \f$\omega = q^{-3/2}\f$.
 *
 * Used by the tests, and available for choosing quasi-circular parameters for a
 * future solve.
 */
std::array<double, 2> circular_orbit(double separation, double eta);

/*!
 * \brief The quasi-circular orbital state at a given separation
 *
 * Returns \f$(M\Omega, \dot{a})\f$ with \f$\dot{a} = \dot{r}/r\f$: the orbital
 * parameters an inspiral of zero eccentricity has at this separation.
 *
 * \f$\Omega\f$ comes from `circular_orbit`. The radial drift follows from
 * energy balance along the sequence,
 *
 * \f{equation}
 * \dot{r} = \frac{\dot{E}/\eta}{d\hat{E}/dq},
 * \qquad
 * \hat{E}(q) = \hat{H}\left(q,\,s_\mathrm{circ}(q),\,0\right),
 * \f}
 *
 * which is negative, so the binary shrinks.
 *
 * The radial drift is what actually makes the start non-eccentric, and it is
 * worth being explicit about why. `circular_orbit` returns the circular orbit
 * of the *conservative* Hamiltonian, which has \f$\vec{n}\cdot\vec{p} = 0\f$,
 * whereas the adiabatic inspiral of the dissipative system wants a small radial
 * momentum. Starting from \f$\dot{r} = 0\f$ therefore injects a residual
 * eccentricity of order \f$\dot{r}/(\Omega r) \sim 6\times10^{-3}\f$.
 */
std::array<double, 2> quasi_circular_state(double separation, double eta);

/*!
 * \brief The past trajectory of the binary, sampled on a uniform time grid
 *
 * Built once when the background is constructed and read-only afterwards, so
 * evaluating it from several threads is safe. Sample \f$n\f$ is at
 * \f$t_n = -n\,\Delta t\f$, running backwards from the present, and both the
 * state and its time derivative are stored so that `state_at` can use cubic
 * Hermite interpolation.
 *
 * \note Deliberately *not* an `intrp::CubicSpline`. That class holds a
 * `gsl_interp_accel*` that its `const operator()` mutates, which is a data race
 * when the background is shared across the threads of a node. A uniform grid
 * needs no accelerator: the enclosing interval is an O(1) index computation.
 */
struct PnInspiral {
  /// Spacing of the samples. Positive; sample `n` is at `-n * time_step`.
  double time_step{std::numeric_limits<double>::signaling_NaN()};
  /// Symmetric mass ratio \f$\eta = \mu/M\f$
  double eta{std::numeric_limits<double>::signaling_NaN()};
  /// Reduced state \f$(\vec{q}, \vec{p})\f$ at each sample
  std::vector<std::array<double, 6>> state{};
  /// \f$(\dot{\vec{q}}, \dot{\vec{p}})\f$ at each sample
  std::vector<std::array<double, 6>> dt_state{};

  /// The earliest time covered, \f$-(N-1)\Delta t\f$
  double earliest_time() const;

  /*!
   * \brief The reduced state \f$(\vec{q}, \vec{p})\f$ at `time`
   *
   * `time` must lie in `[earliest_time(), 0]`. Cubic Hermite interpolation,
   * which is \f$O(\Delta t^4)\f$ and needs no state, so this is thread safe.
   */
  std::array<double, 6> state_at(double time) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);
};

bool operator==(const PnInspiral& lhs, const PnInspiral& rhs);
bool operator!=(const PnInspiral& lhs, const PnInspiral& rhs);

/*!
 * \brief Integrate the binary backwards from the given orbital state
 *
 * The present state is specified the way a previous XCTS solve specifies it: a
 * coordinate separation \f$D\f$ along the \f$x\f$ axis, an orbital angular
 * velocity \f$\Omega\f$ and a radial expansion \f$\dot{a} = \dot{r}/r\f$, i.e.
 *
 * \f{equation}
 * \vec{q}(0) = (D, 0, 0), \qquad
 * \dot{\vec{q}}(0) = \left(\dot{a}D,\; \Omega D,\; 0\right),
 * \f}
 *
 * with the canonical momentum obtained by inverting
 * \f$\dot{\vec{q}} = \partial\hat{H}/\partial\vec{p}\f$. No quasi-circularity
 * is assumed, so whatever orbit the previous solve actually describes —
 * including an eccentric one — is reproduced.
 *
 * Setting `with_radiation_reaction` to `false` gives the conservative flow, for
 * which \f$\hat{H}\f$ and \f$\vec{L}\f$ are exactly conserved. That is the
 * sharpest available test of the derivatives and is used as such.
 */
PnInspiral evolve_binary_backwards(double separation, double angular_velocity,
                                   double expansion, double eta,
                                   double duration, double time_step,
                                   bool with_radiation_reaction = true);
/// @}

/// \name Post-Newtonian wave content
/// @{

/*!
 * \brief The kernel \f$\Phi^{ij}(\vec{u}; A)\f$ shared by the near-zone and
 * remainder pieces of \f$h^{TT}_{ij}\f$
 *
 * \f{equation}
 * \Phi^{ij}(\vec{u}; A) = \frac{1}{r_A}\Big\{
 *   \left[u^2 - 5(\vec{u}\cdot\hat{n}_A)^2\right]\delta^{ij}
 *   + 2u^i u^j
 *   + \left[3(\vec{u}\cdot\hat{n}_A)^2 - 5u^2\right] n_A^i n_A^j
 *   + 12(\vec{u}\cdot\hat{n}_A)\, u^{(i} n_A^{j)} \Big\},
 * \f}
 *
 * with \f$\vec{r}_A = \vec{x} - \vec{x}_A\f$, \f$\hat{n}_A = \vec{r}_A/r_A\f$,
 * and the symmetrisation convention
 * \f$X^{(i}Y^{j)} = \tfrac{1}{2}(X^iY^j + X^jY^i)\f$.
 *
 * Kelly's present-time remainder is \f$H^{TT\,A}_{ij}[\vec{u}; t] =
 * -\tfrac{1}{4}\Phi^{ij}(\vec{u}; A)\f$, and the momentum sum of the near-zone
 * term is \f$+\tfrac{1}{4}\sum_A \Phi^{ij}(\vec{p}_A/\sqrt{m_A}; A)\f$, so the
 * two cancel identically — see report 003 and
 * `htt_instantaneous`. The same kernel evaluated at retarded times supplies the
 * surviving pieces.
 *
 * \see \cite Buonanno2006 for the dynamics; the wave expressions are from
 * Kelly et al. `0704.0628`, verified in report 002.
 */
template <typename DataType>
void post_newtonian_kernel(gsl::not_null<tnsr::ii<DataType, 3>*> result,
                           const tnsr::I<DataType, 3, Frame::Inertial>& x,
                           const std::array<double, 3>& body_position,
                           const std::array<double, 3>& u);

/*!
 * \brief The instantaneous part of \f$h^{TT}_{ij}\f$
 *
 * The near-zone expression of Jaranowski & Schäfer combined with the
 * present-time part of the Kelly et al. remainder. Their momentum sectors
 * cancel *exactly*, and part of the mass–mass sector cancels too, leaving
 *
 * \f{align}
 * h^{TT,\mathrm{inst}}_{ij} = \frac{m_1 m_2}{8}\sum_{A=1,2}\Bigg\{
 * &\left[\frac{3c_A^2}{r_A r_{12}}
 *    - \frac{1}{r_{12}^3}\left(\frac{r_B^2}{r_A} + 3r_A\right)
 *    - \frac{8}{s}\left(\frac{1}{r_A} + \frac{1}{s}\right)\right] n_A^i n_A^j
 *  \nonumber\\
 * &+ \left[\frac{5 r_A}{r_{12}^3}\left(\frac{r_A}{r_B} - 1\right)
 *    - \frac{16 + 5c_A^2}{r_A r_{12}}
 *    + \frac{4}{r_A r_B}
 *    + \frac{8}{s}\left(\frac{1}{r_A} + \frac{4}{r_{12}}\right)\right]
 *    \delta^{ij} \nonumber\\
 * &+ \left[\frac{2}{r_A r_{12}}
 *    - \frac{32}{s}\left(\frac{1}{r_{12}} + \frac{1}{s}\right)\right]
 *    n_{12}^i n_{12}^j \nonumber\\
 * &+ 2\left[\frac{r_A + r_B}{r_{12}^3} + \frac{12}{s^2}\right] n_A^i n_B^j
 *  \nonumber\\
 * &+ \left[32\,\sigma_A\left(\frac{2}{s^2} - \frac{1}{r_{12}^2}\right)
 *    + \frac{12 c_A}{r_A r_{12}}\right] n_A^{(i} n_{12}^{j)} \Bigg\},
 * \f}
 *
 * where \f$s = r_A + r_B + r_{12}\f$, \f$c_A = \hat{n}_{12}\cdot\hat{n}_A\f$,
 * \f$\hat{n}_{12}\f$ points from body 1 to body 2, and \f$\sigma_A = \pm 1\f$
 * for \f$A = 1, 2\f$. Derived in report 003.
 *
 * \warning This is only the instantaneous part. The full \f$h^{TT}_{ij}\f$ adds
 * the retarded and interval pieces of the remainder, which are not implemented
 * yet. On its own the \f$n_A^i n_A^j\f$ coefficient grows like
 * \f$-4r/r_{12}^3\f$ at large \f$r\f$ — an artefact of the near-zone expansion
 * that the retarded terms are supposed to cancel. Do not use this alone as wave
 * content far from the binary.
 *
 * \note Diverges at the punctures, like the near-zone expansion itself. The
 * attenuation function is applied afterwards.
 */
template <typename DataType>
void htt_instantaneous(gsl::not_null<tnsr::ii<DataType, 3>*> result,
                       const tnsr::I<DataType, 3, Frame::Inertial>& x,
                       const std::array<double, 3>& position_1,
                       const std::array<double, 3>& position_2, double mass_1,
                       double mass_2);

/// Positions and momenta of the two bodies at one instant, in physical units.
/// Body 0 is the one at positive \f$x\f$ at the present time and carries
/// `mass_1`, matching `evolve_binary_backwards`.
struct BinaryConfiguration {
  std::array<std::array<double, 3>, 2> positions{};
  std::array<std::array<double, 3>, 2> momenta{};
};

/*!
 * \brief The two bodies' positions and momenta at `time`
 *
 * Converts the reduced trajectory of `PnInspiral` back to physical units and
 * splits the relative motion into the two bodies about the centre of mass,
 *
 * \f{equation}
 * \vec{X} = M\vec{q},\qquad
 * \vec{x}_1 = \frac{m_2}{M}\vec{X},\quad \vec{x}_2 = -\frac{m_1}{M}\vec{X},
 * \qquad
 * \vec{P}_1 = \mu\vec{p} = -\vec{P}_2 .
 * \f}
 *
 * `time` is physical and must be non-positive.
 */
BinaryConfiguration configuration_at(const PnInspiral& inspiral, double time,
                                     double mass_1, double mass_2);

/*!
 * \brief The retarded time of one body as seen from a field point
 *
 * Solves \f$t - t^r_a - r_a(t^r_a) = 0\f$ with the present time \f$t = 0\f$,
 * i.e. \f$-t^r_a = r_a(t^r_a)\f$.
 *
 * The root is unique: \f$g(\tau) = -\tau - r_a(\tau)\f$ has
 * \f$g' = -1 - \dot{r}_a < 0\f$ because the bodies move slower than light, so
 * \f$g\f$ decreases monotonically, and it is negative at \f$\tau = 0\f$.
 * Bracketing uses \f$r_a \le |\vec{x}| + d_\mathrm{max}\f$, which makes
 * \f$\tau = -(|\vec{x}| + d_\mathrm{max})\f$ a guaranteed lower bound;
 * `max_body_distance` is that \f$d_\mathrm{max}\f$, the largest distance of
 * either body from the origin over the stored trajectory.
 */
double retarded_time(const std::array<double, 3>& field_point,
                     const PnInspiral& inspiral, size_t body, double mass_1,
                     double mass_2, double max_body_distance,
                     double present_time);

/*!
 * \brief Retarded times for both bodies at every field point
 *
 * The root find of `retarded_time` vectorised over the grid, returning one
 * `DataType` per body. Both the retarded and the interval parts of the
 * remainder need exactly these, so they are computed once and passed to both:
 * they depend only on the grid and the trajectory, never on the solved fields.
 */
template <typename DataType>
std::array<DataType, 2> retarded_times(
    const tnsr::I<DataType, 3, Frame::Inertial>& x, const PnInspiral& inspiral,
    double mass_1, double mass_2, double present_time);

/*!
 * \brief The retarded part of the Kelly et al. remainder
 *
 * \f{equation}
 * \sum_a\left\{
 *   H^{TT\,a}_{ij}\!\left[\frac{\vec{p}_a}{\sqrt{m_a}}; t^r_a\right]
 *   - H^{TT\,a}_{ij}\!\left[\vec{w}; t^r_a\right]\right\},
 * \f}
 *
 * with
 *
 * \f{equation}
 * H^{TT\,a}_{ij}[\vec{u}; t^r_a] = \frac{1}{r_a(t^r_a)}\Big\{
 *   \left[-2u^2 + 2(\vec{u}\cdot\hat{n}_a)^2\right]\delta^{ij}
 *   + 4u^i u^j
 *   + \left[2u^2 + 2(\vec{u}\cdot\hat{n}_a)^2\right]n_a^i n_a^j
 *   - 8(\vec{u}\cdot\hat{n}_a)\,u^{(i}n_a^{j)}\Big\}_{t^r_a}.
 * \f}
 *
 * Everything inside the braces — \f$r_a\f$, \f$\hat{n}_a\f$ *and* \f$\vec{u}\f$
 * — is evaluated at the retarded time, so both the momenta and the separation
 * entering \f$\vec{w} = \sqrt{m_1m_2/2r_{12}}\,\hat{n}_{12}\f$ are the ones the
 * binary had then. One root find per body per field point serves both
 * \f$\vec{u}\f$ vectors.
 *
 * \note Unlike the instantaneous term this falls off like \f$1/r\f$, so it is
 * what carries radiation into the wave zone.
 */
template <typename DataType>
void htt_retarded(gsl::not_null<tnsr::ii<DataType, 3>*> result,
                  const tnsr::I<DataType, 3, Frame::Inertial>& x,
                  const PnInspiral& inspiral, double mass_1, double mass_2,
                  const std::array<DataType, 2>& retarded_times);

/*!
 * \brief The interval part of the Kelly et al. remainder
 *
 * \f{align}
 * H^{TT\,a}_{ij}[\vec{u}; t^r_a \to t] =
 * &-\int_{t^r_a}^{t}\!\! d\tau\, \frac{t-\tau}{r_a(\tau)^3}\Big\{
 *   \left[-5u^2 + 9(\vec{u}\cdot\hat{n}_a)^2\right]\delta^{ij}
 *   + 6u^iu^j
 *   - 12(\vec{u}\cdot\hat{n}_a)u^{(i}n_a^{j)}
 *   + \left[9u^2 - 15(\vec{u}\cdot\hat{n}_a)^2\right]n_a^in_a^j\Big\}
 *   \nonumber\\
 * &-\int_{t^r_a}^{t}\!\! d\tau\, \frac{(t-\tau)^3}{r_a(\tau)^5}\Big\{
 *   \left[u^2 - 5(\vec{u}\cdot\hat{n}_a)^2\right]\delta^{ij}
 *   + 2u^iu^j
 *   - 20(\vec{u}\cdot\hat{n}_a)u^{(i}n_a^{j)}
 *   + \left[-5u^2 + 35(\vec{u}\cdot\hat{n}_a)^2\right]n_a^in_a^j\Big\},
 * \f}
 *
 * summed as \f$\sum_a\{H^{TT\,a}[\vec{p}_a/\sqrt{m_a}] -
 * H^{TT\,a}[\vec{w}]\}\f$ like the retarded part. Everything in the integrands
 * is evaluated at
 * \f$\tau\f$.
 *
 * Both integrands are finite at either end: at \f$\tau\to t\f$ the factors
 * \f$(t-\tau)\f$ vanish, and at \f$\tau = t^r_a\f$ one has \f$t - \tau =
 * r_a\f$, so both reduce to \f$O(1/r_a^2)\f$. Composite Simpson is used, with
 * the number of intervals scaled to the span so that the orbital oscillation is
 * resolved at a roughly fixed number of points per wave period.
 *
 * \note The \f$1/r_a(\tau)^5\f$ might look dangerous, since it is evaluated
 * along the past trajectory and a body could in principle sweep close to a
 * field point that is far from it now. It cannot happen for a bound, slowly
 * moving binary: the integration span is \f$|t^r_a| = r_a\f$, so a nearby
 * field point gets a short window in which the binary barely rotates, whereas
 * a window long enough for a body to swing round — of order a quarter orbital
 * period — belongs to a field point far outside the orbit. Measured in report
 * 016: the closest approach never falls below \f$6M\f$ anywhere, including
 * for points sitting on the orbital track.
 */
template <typename DataType>
void htt_interval(gsl::not_null<tnsr::ii<DataType, 3>*> result,
                  const tnsr::I<DataType, 3, Frame::Inertial>& x,
                  const PnInspiral& inspiral, double mass_1, double mass_2,
                  const std::array<DataType, 2>& retarded_times,
                  double present_time);

/*!
 * \brief Smallest distance either body reaches from a field point over its
 * retarded history
 *
 * Diagnostic for the close-approach problem described in `htt_interval`: it is
 * the impact parameter that controls how large the interval integrals get.
 */
double closest_retarded_approach(const std::array<double, 3>& field_point,
                                 const PnInspiral& inspiral, double mass_1,
                                 double mass_2);

/*!
 * \brief The complete \f$h^{TT}_{ij}\f$ at a given present time
 *
 * Instantaneous, retarded and interval parts summed, with the body positions
 * for the instantaneous part taken from the trajectory at `present_time`
 * rather than assumed to be the ones at \f$t = 0\f$.
 *
 * Making the present time an argument is what allows `dt_htt` to differentiate
 * the whole construction — retarded-time root find, quadrature and all —
 * without any of it being differentiated by hand.
 */
template <typename DataType>
void htt_total(gsl::not_null<tnsr::ii<DataType, 3>*> result,
               const tnsr::I<DataType, 3, Frame::Inertial>& x,
               const PnInspiral& inspiral, double mass_1, double mass_2,
               double present_time);

/*!
 * \brief Time derivative of the wave, by backward differences
 *
 * \f{equation}
 * \partial_t h^{TT}_{ij} \simeq \frac{3h(0) - 4h(-\Delta) + h(-2\Delta)}
 *                                    {2\Delta}
 * \f}
 *
 * Backward rather than centred because the trajectory only exists for
 * \f$t \le 0\f$: there is no future to centre a stencil on.
 *
 * This replaces the Lie drag that was previously applied to the wave along the
 * *frame's* Killing vector. That was not merely inaccurate at the level of the
 * adiabatic drift: the frame rotates at the previous solve's
 * \f$\Omega = 0.016\f$ while the wave rotates at the post-Newtonian
 * \f$\Omega \approx 0.0215\f$, so the drag mis-stated
 * \f$\partial_t h^{TT}\f$ by tens of percent. See report 017.
 *
 * \note Costs three full evaluations of `htt_total`, each of which carries a
 * retarded-time root find and a quadrature per point. This is the most
 * expensive thing the class does.
 */
template <typename DataType>
void dt_htt(gsl::not_null<tnsr::ii<DataType, 3>*> result,
            const tnsr::I<DataType, 3, Frame::Inertial>& x,
            const PnInspiral& inspiral, double mass_1, double mass_2,
            double step);

/*!
 * \brief Attenuation that switches the wave content off near the punctures
 *
 * \f{equation}
 * f_\mathrm{att}(\vec{x}) = \prod_{A=1,2}
 *   \left[1 - e^{-r_A^2/w^2}\right]
 * \f}
 *
 * The post-Newtonian wave expressions diverge like \f$1/r_A\f$ at the point
 * masses, which is unphysical there — the numerical data already contains real
 * black holes. Each factor vanishes like \f$r_A^2/w^2\f$, so
 * \f$f_\mathrm{att}h^{TT}_{ij}\f$ vanishes linearly at the punctures and the
 * divergence is removed rather than merely damped.
 *
 * A Gaussian is used rather than something with compact support because the
 * result is represented on a spectral grid, where a \f$C^\infty\f$-but-not-
 * analytic transition would ring.
 *
 * \note This deliberately mirrors the `FalloffWidths` of
 * `Xcts::AnalyticData::Binary`, which attenuates the superposed-Kerr-Schild
 * conformal metric the same way and with the same default width, \f$0.3D\f$.
 * A consequence worth knowing when plotting: beyond a few widths the loaded
 * off-diagonal conformal metric is *identically* zero, so any off-diagonal
 * signal out there is wave content and nothing else.
 */
template <typename DataType>
void wave_attenuation(gsl::not_null<Scalar<DataType>*> result,
                      const tnsr::I<DataType, 3, Frame::Inertial>& x,
                      const std::array<double, 3>& position_1,
                      const std::array<double, 3>& position_2, double width);
/// @}

/*!
 * \brief The conformal metric of the loaded data alone, before the wave
 *
 * \f$\bar\gamma^{(0)}_{ij} = (\det\gamma)^{-1/3}\gamma_{ij}\f$, the
 * unimodular split of the previous solve.
 *
 * Kept as a separate cached quantity because the previous solve is stationary
 * along the helical Killing vector but the wave is not: the two need different
 * time derivatives, and separating them needs the spatial derivative of this
 * piece on its own.
 */
template <typename DataType>
struct LoadedConformalMetric : db::SimpleTag {
  using type = tnsr::ii<DataType, 3, Frame::Inertial>;
};

/*!
 * \brief Fields loaded from the volume data of a previous XCTS solve.
 *
 * \note `Xcts::Tags::ShiftExcess` is loaded rather than `gr::Tags::Shift`. The
 * observed `Shift` is the *full* corotating shift
 * \f$\beta^i = \beta^i_\mathrm{background} + \beta^i_\mathrm{excess}\f$
 * (see `Xcts::SpacetimeQuantities`), which grows like \f$\propto r\f$ and is
 * therefore badly resolved in the outer shell. The excess shift is
 * asymptotically small, which is exactly why the previous solve writes it.
 */
template <typename DataType>
using numeric_load_tags =
    tmpl::list<gr::Tags::SpatialMetric<DataType, 3>, gr::Tags::Lapse<DataType>,
               Tags::ShiftExcess<DataType, 3, Frame::Inertial>,
               gr::Tags::ExtrinsicCurvature<DataType, 3>>;

template <typename DataType>
using NumericBinaryWithWavesVariablesCache =
    cached_temp_buffer_from_typelist<tmpl::append<
        common_tags<DataType>,
        tmpl::list<
            // Physical fields loaded from the previous solve
            gr::Tags::SpatialMetric<DataType, 3>,
            gr::Tags::InverseSpatialMetric<DataType, 3>,
            gr::Tags::Lapse<DataType>,
            gr::Tags::ExtrinsicCurvature<DataType, 3>,
            // The loaded conformal metric and its spatial derivative, needed
            // separately from the wave so the Lie drag applies only to it
            LoadedConformalMetric<DataType>,
            ::Tags::deriv<LoadedConformalMetric<DataType>, tmpl::size_t<3>,
                          Frame::Inertial>,
            // Time derivative of the conformal metric, from the helical
            // Killing vector (see report 004)
            ::Tags::dt<Tags::ConformalMetric<DataType, 3, Frame::Inertial>>,
            // Vacuum matter sources
            gr::Tags::Conformal<gr::Tags::EnergyDensity<DataType>, 0>,
            gr::Tags::Conformal<gr::Tags::StressTrace<DataType>, 0>,
            gr::Tags::Conformal<gr::Tags::MomentumDensity<DataType, 3>, 0>,
            // Initial guesses
            Tags::ConformalFactorMinusOne<DataType>,
            Tags::LapseTimesConformalFactorMinusOne<DataType>,
            Tags::ShiftExcess<DataType, 3, Frame::Inertial>>,
        hydro_tags<DataType>>>;

template <typename DataType>
struct NumericBinaryWithWavesVariables
    : CommonVariables<DataType,
                      NumericBinaryWithWavesVariablesCache<DataType>> {
  static constexpr size_t Dim = 3;
  using Cache = NumericBinaryWithWavesVariablesCache<DataType>;
  using Base = CommonVariables<DataType, Cache>;
  using Base::operator();

  NumericBinaryWithWavesVariables(
      std::optional<std::reference_wrapper<const Mesh<Dim>>> local_mesh,
      std::optional<std::reference_wrapper<const InverseJacobian<
          DataType, Dim, Frame::ElementLogical, Frame::Inertial>>>
          local_inv_jacobian,
      const tnsr::I<DataType, Dim, Frame::Inertial>& local_x,
      double local_angular_velocity, double local_expansion,
      const std::array<double, 3>& local_position_1,
      const std::array<double, 3>& local_position_2, double local_mass_1,
      double local_mass_2, double local_attenuation_width,
      const PnInspiral& local_inspiral,
      tuples::tagged_tuple_from_typelist<numeric_load_tags<DataType>>
          local_loaded_vars)
      : Base(std::move(local_mesh), std::move(local_inv_jacobian)),
        x(local_x),
        angular_velocity(local_angular_velocity),
        expansion(local_expansion),
        position_1(local_position_1),
        position_2(local_position_2),
        mass_1(local_mass_1),
        mass_2(local_mass_2),
        attenuation_width(local_attenuation_width),
        inspiral(local_inspiral),
        loaded_vars(std::move(local_loaded_vars)) {}

  const tnsr::I<DataType, Dim, Frame::Inertial>& x;
  double angular_velocity;
  double expansion;
  // Present positions of the two bodies on the post-Newtonian trajectory.
  // Body 1 sits at positive x, matching the sign convention of
  // `evolve_binary_backwards`.
  std::array<double, 3> position_1;
  std::array<double, 3> position_2;
  double mass_1;
  double mass_2;
  double attenuation_width;
  // Outlives this computer: it is a member of the enclosing background.
  const PnInspiral& inspiral;
  tuples::tagged_tuple_from_typelist<numeric_load_tags<DataType>> loaded_vars;

  // Fields taken straight from the loaded data
  void operator()(gsl::not_null<tnsr::ii<DataType, Dim>*> spatial_metric,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::SpatialMetric<DataType, Dim> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::II<DataType, Dim>*> inv_spatial_metric,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::InverseSpatialMetric<DataType, Dim> /*meta*/) const;
  void operator()(gsl::not_null<Scalar<DataType>*> lapse,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::Lapse<DataType> /*meta*/) const;
  void operator()(gsl::not_null<tnsr::ii<DataType, Dim>*> extrinsic_curvature,
                  gsl::not_null<Cache*> cache,
                  gr::Tags::ExtrinsicCurvature<DataType, Dim> /*meta*/) const;

  void operator()(gsl::not_null<tnsr::ii<DataType, Dim>*> loaded,
                  gsl::not_null<Cache*> cache,
                  LoadedConformalMetric<DataType> /*meta*/) const;
  void operator()(
      gsl::not_null<tnsr::ijj<DataType, Dim>*> deriv_loaded,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<LoadedConformalMetric<DataType>, tmpl::size_t<Dim>,
                    Frame::Inertial> /*meta*/) const;

  /// The attenuated, trace-projected wave added to the conformal metric, at an
  /// arbitrary present time. Differencing this in `present_time` is what gives
  /// the wave's time derivative (report 017).
  tnsr::ii<DataType, Dim> wave_contribution(double present_time,
                                            gsl::not_null<Cache*> cache) const;

  // Time derivative of the conformal metric from the helical Killing vector
  void operator()(gsl::not_null<tnsr::ii<DataType, Dim>*> dt_conformal_metric,
                  gsl::not_null<Cache*> cache,
                  ::Tags::dt<Tags::ConformalMetric<
                      DataType, Dim, Frame::Inertial>> /*meta*/) const;

  // The six quantities CommonVariables leaves pure virtual
  void operator()(
      gsl::not_null<tnsr::ii<DataType, Dim>*> conformal_metric,
      gsl::not_null<Cache*> cache,
      Tags::ConformalMetric<DataType, Dim, Frame::Inertial> /*meta*/)
      const override;
  void operator()(
      gsl::not_null<tnsr::ijj<DataType, Dim>*> deriv_conformal_metric,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<Tags::ConformalMetric<DataType, Dim, Frame::Inertial>,
                    tmpl::size_t<Dim>, Frame::Inertial> /*meta*/)
      const override;
  void operator()(
      gsl::not_null<Scalar<DataType>*> extrinsic_curvature_trace,
      gsl::not_null<Cache*> cache,
      gr::Tags::TraceExtrinsicCurvature<DataType> /*meta*/) const override;
  void operator()(
      gsl::not_null<Scalar<DataType>*> dt_extrinsic_curvature_trace,
      gsl::not_null<Cache*> cache,
      ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataType>> /*meta*/)
      const override;
  void operator()(
      gsl::not_null<tnsr::I<DataType, Dim>*> shift_background,
      gsl::not_null<Cache*> cache,
      Tags::ShiftBackground<DataType, Dim, Frame::Inertial> /*meta*/)
      const override;
  void operator()(
      gsl::not_null<tnsr::iJ<DataType, Dim>*> deriv_shift_background,
      gsl::not_null<Cache*> cache,
      ::Tags::deriv<Tags::ShiftBackground<DataType, Dim, Frame::Inertial>,
                    tmpl::size_t<Dim>, Frame::Inertial> /*meta*/)
      const override;
  void operator()(gsl::not_null<tnsr::II<DataType, Dim, Frame::Inertial>*>
                      longitudinal_shift_background,
                  gsl::not_null<Cache*> cache,
                  Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
                      DataType, Dim, Frame::Inertial> /*meta*/) const override;

  // Vacuum matter sources
  void operator()(
      gsl::not_null<Scalar<DataType>*> conformal_energy_density,
      gsl::not_null<Cache*> cache,
      gr::Tags::Conformal<gr::Tags::EnergyDensity<DataType>, 0> /*meta*/) const;
  void operator()(
      gsl::not_null<Scalar<DataType>*> conformal_stress_trace,
      gsl::not_null<Cache*> cache,
      gr::Tags::Conformal<gr::Tags::StressTrace<DataType>, 0> /*meta*/) const;
  void operator()(
      gsl::not_null<tnsr::I<DataType, Dim>*> conformal_momentum_density,
      gsl::not_null<Cache*> cache,
      gr::Tags::Conformal<gr::Tags::MomentumDensity<DataType, Dim>, 0> /*meta*/)
      const;

  // Initial guesses, taken from the loaded solution
  void operator()(gsl::not_null<Scalar<DataType>*> conformal_factor_minus_one,
                  gsl::not_null<Cache*> cache,
                  Tags::ConformalFactorMinusOne<DataType> /*meta*/) const;
  void operator()(
      gsl::not_null<Scalar<DataType>*> lapse_times_conformal_factor_minus_one,
      gsl::not_null<Cache*> cache,
      Tags::LapseTimesConformalFactorMinusOne<DataType> /*meta*/) const;
  void operator()(
      gsl::not_null<tnsr::I<DataType, Dim>*> shift_excess,
      gsl::not_null<Cache*> cache,
      Tags::ShiftExcess<DataType, Dim, Frame::Inertial> /*meta*/) const;

  // Vacuum hydro quantities. `Xcts::Tags::HydroQuantitiesCompute` requests
  // these from every background, so they must be provided even though this is
  // a vacuum system.
  void operator()(const gsl::not_null<Scalar<DataType>*> rest_mass_density,
                  const gsl::not_null<Cache*> /*cache*/,
                  hydro::Tags::RestMassDensity<DataType> /*meta*/) const {
    get(*rest_mass_density) = make_with_value<DataType>(get<0>(x), 0.);
  }
  void operator()(const gsl::not_null<Scalar<DataType>*> specific_enthalpy,
                  const gsl::not_null<Cache*> /*cache*/,
                  hydro::Tags::SpecificEnthalpy<DataType> /*meta*/) const {
    get(*specific_enthalpy) = make_with_value<DataType>(get<0>(x), 1.);
  }
  void operator()(const gsl::not_null<Scalar<DataType>*> pressure,
                  const gsl::not_null<Cache*> /*cache*/,
                  hydro::Tags::Pressure<DataType> /*meta*/) const {
    get(*pressure) = make_with_value<DataType>(get<0>(x), 0.);
  }
  void operator()(const gsl::not_null<tnsr::I<DataType, Dim>*> spatial_velocity,
                  const gsl::not_null<Cache*> /*cache*/,
                  hydro::Tags::SpatialVelocity<DataType, Dim> /*meta*/) const {
    std::fill(spatial_velocity->begin(), spatial_velocity->end(), 0.);
  }
  void operator()(const gsl::not_null<Scalar<DataType>*> lorentz_factor,
                  const gsl::not_null<Cache*> /*cache*/,
                  hydro::Tags::LorentzFactor<DataType> /*meta*/) const {
    get(*lorentz_factor) = make_with_value<DataType>(get<0>(x), 1.);
  }
  void operator()(const gsl::not_null<tnsr::I<DataType, Dim>*> magnetic_field,
                  const gsl::not_null<Cache*> /*cache*/,
                  hydro::Tags::MagneticField<DataType, Dim> /*meta*/) const {
    std::fill(magnetic_field->begin(), magnetic_field->end(), 0.);
  }

  /// The helical Killing vector \f$\xi^i\f$ of the previous solve, evaluated at
  /// the grid points. See `deriv_killing_vector` for its (constant) gradient.
  tnsr::I<DataType, Dim> killing_vector() const;
  /// \f$\partial_i \xi^j\f$, a constant matrix
  tnsr::iJ<DataType, Dim> deriv_killing_vector() const;
};

}  // namespace detail

/*!
 * \brief XCTS background built from the volume data of a previous solve, with
 * post-Newtonian gravitational waves added
 *
 * This class implements the second stage of an incremental construction of
 * binary-black-hole initial data with realistic wave content:
 *
 * 1. A standard BBH XCTS solve produces a solution in corotating coordinates.
 * 2. This class loads that solution and uses it as both background and initial
 *    guess, adding post-Newtonian wave content on top of the conformal metric,
 *    and the XCTS equations are solved again.
 *
 * ## Frame
 *
 * The previous solve is performed in coordinates that corotate with the binary,
 * so that \f$\partial_t\bar{\gamma}_{ij} = 0\f$ and \f$\partial_t K = 0\f$,
 * with a background shift
 *
 * \f{equation}
 * \xi^i = \left(-\Omega y + \dot{a} x,\; \Omega x + \dot{a} y,\; \dot{a}
 * z\right)
 * \f}
 *
 * This class instead works in the **inertial** frame, where
 * \f$\beta^i_\mathrm{background} = 0\f$ and the solved-for shift is the excess
 * shift of the previous solve. The two are equivalent for the XCTS equations,
 * which see the shift only through
 * \f$\left(\bar{L}\beta\right)^{ij} +
 * \left(\bar{L}\beta_\mathrm{bg}\right)^{ij}
 * - \bar{u}^{ij}\f$: since \f$\mathcal{L}_\xi\bar{\gamma}_{ij} =
 * \bar{D}_i\xi_j + \bar{D}_j\xi_i\f$, the trace-free raised part of
 * \f$-\partial_t\bar{\gamma}\f$ is exactly \f$-\left(\bar{L}\xi\right)^{ij}\f$,
 * so the inertial-frame combination reproduces the corotating one.
 *
 * \warning Load `ShiftExcess`, not `Shift`. The observed `Shift` is the full
 * corotating shift \f$\beta^i_\mathrm{bg} + \beta^i_\mathrm{excess}\f$, which
 * grows \f$\propto r\f$ and is badly resolved in the outer shell; using it as
 * the excess shift also double-counts \f$\left(\bar{L}\xi\right)^{ij}\f$.
 *
 * ## Conformal decomposition
 *
 * The physical fields \f$\gamma_{ij}\f$, \f$\alpha\f$,
 * \f$\beta^i_\mathrm{excess}\f$ and \f$K_{ij}\f$ are loaded from the volume
 * data. The conformal decomposition is fixed by the unimodular gauge
 * \f$\det\bar{\gamma}_{ij} = 1\f$, which determines
 *
 * \f{align}
 * \psi = \left(\det\gamma_{ij}\right)^{1/12}, \quad
 * \bar{\gamma}_{ij} = \left(\det\gamma_{ij}\right)^{-1/3}\gamma_{ij}
 * \f}
 *
 * uniquely from the loaded data. The wave content is then added as
 * \f$\bar{\gamma}_{ij} \to \bar{\gamma}_{ij} + f_\mathrm{att} h^{TT}_{ij}\f$.
 *
 * The trace of the extrinsic curvature is taken directly from the loaded
 * \f$K_{ij}\f$ as \f$K = \gamma^{ij}K_{ij}\f$, which involves no derivatives
 * and is frame independent.
 *
 * ## Post-Newtonian past evolution
 *
 * \f$h^{TT}_{ij}\f$ depends on the state of the binary at retarded times, so
 * the inspiral is evolved backwards with the 3PN ADM Hamiltonian and a 3.5PN
 * radiation-reaction force before any point is evaluated. See `PnInspiral` and
 * `evolve_binary_backwards`; the trajectory is built once in the constructor
 * and is read-only afterwards.
 *
 * The present orbital state is built from `Separation` and
 * `TargetEccentricity`: given the separation of the previous solve and the
 * eccentricity it was aiming for, the angular velocity and the radial drift
 * are read off the post-Newtonian sequence.
 *
 * \warning `AngularVelocity` and `Expansion` are *not* used for this. They
 * describe the helical Killing vector of the loaded data — the frame — and are
 * needed for \f$\partial_t\bar{\gamma}_{ij}\f$ whatever orbit that data
 * happens to represent. They are not required to be a mutually consistent
 * orbital state, and in practice need not be: a previous solve may be given a
 * separation and an angular velocity that do not lie on the quasi-circular
 * sequence, in which case its true orbit is eccentric. Driving the past
 * evolution from such a pair sends it into the strong field, where the
 * post-Newtonian expansion fails (see `evolve_binary_backwards`). Deriving the
 * orbit from `TargetEccentricity` instead keeps the wave-generating history
 * well defined.
 *
 * \note The wave \f$h^{TT}_{ij}\f$ itself is not added yet; this class
 * currently reproduces the loaded solution. That makes it testable on its own:
 * solving the XCTS equations with this background should return the data that
 * was loaded.
 */
class NumericBinaryWithWaves : public elliptic::analytic_data::Background,
                               public elliptic::analytic_data::InitialGuess {
 public:
  struct DataFile {
    static constexpr Options::String help =
        "Path or glob pattern to the volume data of the previous solve";
    using type = std::string;
  };
  struct Subgroup {
    static constexpr Options::String help =
        "The subgroup within the volume data file, excluding extensions";
    using type = std::string;
  };
  struct ObservationStep {
    static constexpr Options::String help =
        "The observation step at which to read the data. Use -1 for the last.";
    using type = int;
    static int suggested_value() { return -1; }
  };
  struct ExtrapolateIntoExcisions {
    static constexpr Options::String help =
        "Whether to extrapolate the loaded data into excised regions";
    using type = bool;
    static bool suggested_value() { return false; }
  };
  struct AngularVelocity {
    static constexpr Options::String help =
        "Orbital angular velocity of the previous solve. Together with "
        "'Expansion' this defines the helical Killing vector along which the "
        "previous solution is stationary, and hence the time derivative of "
        "the conformal metric in the inertial frame. This is a property of the "
        "loaded data and must match the previous solve exactly. It does NOT "
        "set the orbit of the post-Newtonian past evolution, which is derived "
        "from 'Separation' and 'TargetEccentricity'.";
    using type = double;
  };
  struct Expansion {
    static constexpr Options::String help =
        "Radial expansion velocity of the previous solve. Like "
        "'AngularVelocity', this defines the helical Killing vector and must "
        "match the previous solve exactly.";
    using type = double;
  };
  struct Separation {
    static constexpr Options::String help =
        "Coordinate separation of the two objects in the previous solve. With "
        "'TargetEccentricity' this fixes the present orbital state from which "
        "the post-Newtonian inspiral is evolved backwards.";
    using type = double;
    static double lower_bound() { return 0.0; }
  };
  struct TargetEccentricity {
    static constexpr Options::String help =
        "Orbital eccentricity of the post-Newtonian past evolution. Use the "
        "target eccentricity of the previous solve. Given this and "
        "'Separation', the angular velocity and radial drift of the past "
        "evolution are derived from the post-Newtonian sequence rather than "
        "copied from 'AngularVelocity' and 'Expansion' -- those describe the "
        "frame of the loaded data, which need not be a consistent orbital "
        "state. Only zero is supported so far.";
    using type = double;
    static double suggested_value() { return 0.0; }
    static double lower_bound() { return 0.0; }
  };
  struct MassLeft {
    static constexpr Options::String help =
        "Mass of the left object. Use the target ADM (Christodoulou) mass, "
        "not the bare Kerr mass parameter of the conformal superposition.";
    using type = double;
    static double lower_bound() { return 0.0; }
  };
  struct MassRight {
    static constexpr Options::String help = "Mass of the right object";
    using type = double;
    static double lower_bound() { return 0.0; }
  };
  struct PastEvolutionDuration {
    static constexpr Options::String help =
        "How far back in time to evolve the binary. The retarded time of the "
        "outermost grid point must be covered, so this should exceed the "
        "outer radius of the domain plus the separation.";
    using type = double;
    static double lower_bound() { return 0.0; }
  };
  struct AttenuationWidth {
    static constexpr Options::String help =
        "Width of the Gaussian that switches the wave content off near each "
        "puncture, where the post-Newtonian expressions diverge. Mirrors the "
        "'FalloffWidths' of the Binary background; 0.3 times the separation is "
        "a reasonable default.";
    using type = double;
    static double lower_bound() { return 0.0; }
  };
  struct PastEvolutionTimeStep {
    static constexpr Options::String help =
        "Spacing of the stored trajectory samples. The interpolation error is "
        "fourth order in this, against an orbital timescale of 2 pi / Omega, "
        "so it is not a demanding parameter.";
    using type = double;
    static double suggested_value() { return 0.5; }
    static double lower_bound() { return 0.0; }
  };
  using options =
      tmpl::list<DataFile, Subgroup, ObservationStep, ExtrapolateIntoExcisions,
                 AngularVelocity, Expansion, Separation, TargetEccentricity,
                 MassLeft, MassRight, AttenuationWidth, PastEvolutionDuration,
                 PastEvolutionTimeStep>;
  static constexpr Options::String help =
      "Binary black hole initial data built from a previous XCTS solve, with "
      "post-Newtonian gravitational waves added.";

  NumericBinaryWithWaves() = default;
  // The interpolator holds a `Domain`, which is move-only, so it cannot be
  // copied. Copies reload the data from file instead, following
  // `ray_tracing::NumericData`.
  NumericBinaryWithWaves(const NumericBinaryWithWaves& rhs);
  NumericBinaryWithWaves& operator=(const NumericBinaryWithWaves& rhs);
  NumericBinaryWithWaves(NumericBinaryWithWaves&&) = default;
  NumericBinaryWithWaves& operator=(NumericBinaryWithWaves&&) = default;
  ~NumericBinaryWithWaves() override = default;

  NumericBinaryWithWaves(std::string data_file, std::string subgroup,
                         int observation_step, bool extrapolate_into_excisions,
                         double angular_velocity, double expansion,
                         double separation, double target_eccentricity,
                         double mass_left, double mass_right,
                         double attenuation_width,
                         double past_evolution_duration,
                         double past_evolution_time_step);

  explicit NumericBinaryWithWaves(CkMigrateMessage* m)
      : elliptic::analytic_data::Background(m),
        elliptic::analytic_data::InitialGuess(m) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(NumericBinaryWithWaves);

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      tmpl::list<RequestedTags...> /*meta*/) const {
    return variables_impl<DataType>(x, std::nullopt, std::nullopt,
                                    tmpl::list<RequestedTags...>{});
  }

  template <typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x, const Mesh<3>& mesh,
      const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                            Frame::Inertial>& inv_jacobian,
      tmpl::list<RequestedTags...> /*meta*/) const {
    return variables_impl<DataVector>(x, mesh, inv_jacobian,
                                      tmpl::list<RequestedTags...>{});
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

  const std::string& data_file() const { return data_file_; }
  const std::string& subgroup() const { return subgroup_; }
  int observation_step() const { return observation_step_; }
  bool extrapolate_into_excisions() const {
    return extrapolate_into_excisions_;
  }
  double angular_velocity() const { return angular_velocity_; }
  double expansion() const { return expansion_; }
  double separation() const { return separation_; }
  double target_eccentricity() const { return target_eccentricity_; }
  double attenuation_width() const { return attenuation_width_; }
  double mass_left() const { return mass_left_; }
  double mass_right() const { return mass_right_; }
  double past_evolution_duration() const { return past_evolution_duration_; }
  double past_evolution_time_step() const { return past_evolution_time_step_; }

  /// The past trajectory of the binary, from which the wave content at
  /// retarded times is built.
  const detail::PnInspiral& pn_inspiral() const { return pn_inspiral_; }

 private:
  /// Read the volume data into memory. Expensive, and not thread safe unless
  /// HDF5 was built with thread-safety support, so it must happen exactly once
  /// per node and on a single thread. See `load_interpolator` in the .cpp.
  void load_interpolator();

  /// Integrate the binary backwards and store the trajectory. Cheap (a single
  /// ODE integration of a six-component system), so it is redone on unpacking
  /// rather than serialized.
  void evolve_past();

  std::string data_file_{};
  std::string subgroup_{};
  int observation_step_{-1};
  bool extrapolate_into_excisions_{false};
  double angular_velocity_ = std::numeric_limits<double>::signaling_NaN();
  double expansion_ = std::numeric_limits<double>::signaling_NaN();
  double separation_ = std::numeric_limits<double>::signaling_NaN();
  double target_eccentricity_ = 0.0;
  double attenuation_width_ = std::numeric_limits<double>::signaling_NaN();
  // Present positions of the two bodies, from the trajectory. Body 1 is the
  // one at positive x, matching `evolve_binary_backwards`, so it pairs with
  // `mass_right_`.
  std::array<double, 3> present_position_1_{};
  std::array<double, 3> present_position_2_{};
  double mass_left_ = std::numeric_limits<double>::signaling_NaN();
  double mass_right_ = std::numeric_limits<double>::signaling_NaN();
  double past_evolution_duration_ =
      std::numeric_limits<double>::signaling_NaN();
  double past_evolution_time_step_ =
      std::numeric_limits<double>::signaling_NaN();
  // Computed once in the constructor and on unpacking, then read only. Cheap
  // enough (one ODE integration) that serializing it would be no faster.
  detail::PnInspiral pn_inspiral_{};
  // Holds the loaded volume data in memory. Deliberately not serialized: it is
  // reloaded on unpacking, so that each node loads it once.
  spectre::Exporter::PointwiseInterpolator<3, Frame::Inertial> interpolator_{};

  template <typename DataType, typename... RequestedTags>
  tuples::TaggedTuple<RequestedTags...> variables_impl(
      const tnsr::I<DataType, 3, Frame::Inertial>& x,
      std::optional<std::reference_wrapper<const Mesh<3>>> mesh,
      std::optional<std::reference_wrapper<const InverseJacobian<
          DataType, 3, Frame::ElementLogical, Frame::Inertial>>>
          inv_jacobian,
      tmpl::list<RequestedTags...> /*meta*/) const {
    static_assert(
        std::is_same_v<DataType, DataVector>,
        "NumericBinaryWithWaves interpolates volume data onto a grid, so it "
        "only works with DataVectors.");
    using VarsComputer = detail::NumericBinaryWithWavesVariables<DataType>;
    // Interpolate all loaded fields from the already-resident volume data. This
    // is thread safe, unlike constructing the interpolator. `num_threads` is
    // pinned to 1 because this is already called from one Charm++ thread per
    // element; letting it spawn its own OpenMP team would oversubscribe.
    std::vector<DataVector> interpolated_data{};
    interpolator_.interpolate_to_points(
        make_not_null(&interpolated_data), x, extrapolate_into_excisions_,
        /*error_on_missing_points=*/true, /*num_threads=*/1);
    auto loaded_vars = spectre::Exporter::make_tagged_tuple<
        detail::numeric_load_tags<DataType>>(std::move(interpolated_data));
    typename VarsComputer::Cache cache{get_size(*x.begin())};
    const VarsComputer computer{std::move(mesh),
                                std::move(inv_jacobian),
                                x,
                                angular_velocity_,
                                expansion_,
                                present_position_1_,
                                present_position_2_,
                                mass_right_,
                                mass_left_,
                                attenuation_width_,
                                pn_inspiral_,
                                std::move(loaded_vars)};
    return {cache.get_var(computer, RequestedTags{})...};
  }
};

bool operator==(const NumericBinaryWithWaves& lhs,
                const NumericBinaryWithWaves& rhs);
bool operator!=(const NumericBinaryWithWaves& lhs,
                const NumericBinaryWithWaves& rhs);

}  // namespace Xcts::AnalyticData
