// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Wraps the template metaprogramming library used (brigand)

#pragma once

// Since this header only wraps brigand and several additions to it we mark
// it as a system header file so that clang-tidy ignores it.
#ifdef __GNUC__
#pragma GCC system_header
#endif

/// \cond NEVER
#define BRIGAND_NO_BOOST_SUPPORT
/// \endcond
#include <brigand/brigand.hpp>

#include <cstddef>
#include <initializer_list>
#include <type_traits>
#include <variant>

#include "Utilities/TmplDigraph.hpp"

namespace brigand {
/// \cond
namespace detail {

// specializations to catch attempts to transform a non-sequence
template <typename NotSeq, class Func>
struct transform<0, NotSeq, Func> {
  static_assert(
      std::is_same_v<list<>, NotSeq>,
      "Cannot transform a non-sequence (the second argument of is_same_v).");
  using type = std::void_t<>;
};

template <typename NotSeq, template <class...> class Seq, class... T,
          class Func>
struct transform<1, NotSeq, Seq<T...>, Func> {
  static_assert(
      std::is_same_v<list<>, NotSeq>,
      "Cannot transform a non-sequence (the second argument of is_same_v).");
  using type = std::void_t<>;
};

template <template <class...> class Seq, class... T, typename NotSeq,
          class Func>
struct transform<1, Seq<T...>, NotSeq, Func> {
  static_assert(
      std::is_same_v<list<>, NotSeq>,
      "Cannot transform a non-sequence (the second argument of is_same_v).");
  using type = std::void_t<>;
};

template <typename NotSeq1, typename NotSeq2, class Func>
struct transform<1, NotSeq1, NotSeq2, Func> {
  static_assert(
      std::is_same_v<list<>, NotSeq1>,
      "Cannot transform a non-sequence (the second argument of is_same_v).");
  using type = std::void_t<>;
};

template <bool b, typename O, typename L, std::size_t I, typename R,
          typename U = void>
struct replace_at_impl;

template <template <typename...> class S, typename... Os, typename... Ts,
          typename R>
struct replace_at_impl<false, S<Os...>, S<Ts...>, 0, R> {
  using type = S<Os..., Ts...>;
};

template <template <typename...> class S, typename... Os, typename... Ts,
          typename T, typename R>
struct replace_at_impl<false, S<Os...>, S<T, Ts...>, 1, R>
    : replace_at_impl<false, S<Os..., R>, S<Ts...>, 0, R> {};

template <template <typename...> class S, typename... Os, typename T,
          typename... Ts, std::size_t I, typename R>
struct replace_at_impl<false, S<Os...>, S<T, Ts...>, I, R,
                       typename std::enable_if<(I > 1)>::type>
    : replace_at_impl<false, S<Os..., T>, S<Ts...>, (I - 1), R> {};

template <template <typename...> class S, typename... Os, typename T1,
          typename T2, typename T3, typename T4, typename T5, typename T6,
          typename T7, typename T8, typename T9, typename T10, typename T11,
          typename T12, typename T13, typename T14, typename T15, typename T16,
          typename... Ts, std::size_t I, typename R>
struct replace_at_impl<true, S<Os...>,
                       S<T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13,
                         T14, T15, T16, Ts...>,
                       I, R>
    : replace_at_impl<((I - 16) > 16),
                      S<Os..., T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11,
                        T12, T13, T14, T15, T16>,
                      S<Ts...>, (I - 16), R> {};

template <typename L, typename I, typename R>
struct call_replace_at_impl
    : replace_at_impl<(I::value > 15), brigand::clear<L>, L, I::value + 1, R> {
};
}  // namespace detail
/// \endcond

namespace lazy {
template <typename L, typename I, typename R>
using replace_at = ::brigand::detail::call_replace_at_impl<L, I, R>;
}  // namespace lazy
template <typename L, typename I, typename R>
using replace_at = typename ::brigand::lazy::replace_at<L, I, R>::type;
}  // namespace brigand

namespace brigand {
namespace detail {
template <typename List, typename Ind1, typename Ind2>
struct swap_at_impl {
  using type = ::brigand::replace_at<
      ::brigand::replace_at<List, Ind1, ::brigand::at<List, Ind2>>, Ind2,
      ::brigand::at<List, Ind1>>;
};
}  // namespace detail

template <typename List, typename Ind1, typename Ind2>
using swap_at =
    typename ::brigand::detail::swap_at_impl<List, Ind1, Ind2>::type;
}  // namespace brigand

namespace brigand {
template <typename V>
using abs = std::integral_constant<typename V::value_type,
                                   (V::value < 0 ? -V::value : V::value)>;

template <typename V>
using sign =
    std::integral_constant<typename V::value_type, (V::value < 0 ? -1 : 1)>;

template <int T>
using int_ = std::integral_constant<int, T>;

template <typename V, typename N>
struct power
    : std::integral_constant<
          typename V::value_type,
          V::value * power<V, std::integral_constant<typename N::value_type,
                                                     N::value - 1>>::value> {};

template <typename V, typename T>
struct power<V, std::integral_constant<T, 0>>
    : std::integral_constant<typename V::value_type, 1> {};

template <typename T>
struct factorial : times<T, factorial<uint64_t<T::value - 1>>> {};
template <>
struct factorial<uint64_t<1>> : uint64_t<1> {};
}  // namespace brigand

namespace brigand {
namespace detail {
template <typename List, std::size_t Size = size<List>::value>
struct permutations_impl {
  template <typename T, typename List1>
  struct helper {
    using type = ::brigand::transform<
        typename permutations_impl<::brigand::remove<List1, T>>::type,
        ::brigand::lazy::push_front<_state, T>>;
  };

  using type = ::brigand::fold<
      List, list<>,
      ::brigand::lazy::append<::brigand::_state, helper<::brigand::_element,
                                                        ::brigand::pin<List>>>>;
};

template <typename List>
struct permutations_impl<List, 1> {
  using type = list<List>;
};
}  // namespace detail

namespace lazy {
template <typename List>
using permutations = detail::permutations_impl<List>;
}  // namespace lazy

template <typename List>
using permutations = typename lazy::permutations<List>::type;
}  // namespace brigand

namespace brigand {
namespace detail {
template <typename List, std::size_t Size = ::brigand::size<List>::value>
struct generic_permutations_impl {
  template <typename Lc, typename List1>
  struct helper {
    using type = ::brigand::transform<
        typename generic_permutations_impl<::brigand::erase<List1, Lc>>::type,
        ::brigand::lazy::push_front<::brigand::_state,
                                    ::brigand::at<List1, Lc>>>;
  };
  using type = ::brigand::fold<
      ::brigand::make_sequence<brigand::uint32_t<0>, Size>, ::brigand::list<>,
      ::brigand::lazy::append<::brigand::_state, helper<::brigand::_element,
                                                        ::brigand::pin<List>>>>;
};

template <typename List>
struct generic_permutations_impl<List, 1> {
  using type = ::brigand::list<List>;
};
}  // namespace detail

namespace lazy {
template <typename List>
using generic_permutations = detail::generic_permutations_impl<List>;
}  // namespace lazy

template <typename List>
using generic_permutations = typename lazy::generic_permutations<List>::type;
}  // namespace brigand

namespace brigand {
namespace detail {
template <typename List, typename Number = uint32_t<1>>
struct combinations_impl_helper {
  using split_list = split_at<List, Number>;
  using type =
      fold<back<split_list>, list<>,
           lazy::append<
               _state,
               bind<list, bind<push_back, pin<front<split_list>>, _element>>>>;
};

template <typename List, typename OutSize, typename = void>
struct combinations_impl {
  using type =
      append<list<>,
             typename combinations_impl_helper<List, prev<OutSize>>::type,
             typename combinations_impl<pop_front<List>, OutSize>::type>;
};
template <typename List, typename OutSize>
struct combinations_impl<
    List, OutSize,
    typename std::enable_if<OutSize::value == size<List>::value>::type> {
  using type = typename combinations_impl_helper<List, prev<OutSize>>::type;
};
}  // namespace detail

namespace lazy {
template <typename List, typename OutSize = uint32_t<2>>
using combinations = detail::combinations_impl<List, OutSize>;
}  // namespace lazy

template <typename List, typename OutSize = uint32_t<2>>
using combinations = typename lazy::combinations<List, OutSize>::type;
}  // namespace brigand

namespace brigand {
namespace detail {
template <typename Seq, typename T>
struct equal_members_helper
    : std::is_same<count_if<Seq, std::is_same<T, _1>>, size_t<1>> {};
}  // namespace detail

template <typename List1, typename List2>
using equal_members = and_<
    fold<List1, bool_<true>,
         and_<_state, detail::equal_members_helper<pin<List2>, _element>>>,
    fold<List2, bool_<true>,
         and_<_state, detail::equal_members_helper<pin<List1>, _element>>>>;
}  // namespace brigand

namespace brigand {
namespace detail {
template <typename Functor, typename State, typename I, typename Sequence>
struct enumerated_fold_impl {
  using type = State;
};

template <typename Functor, typename State, typename I,
          template <typename...> class Sequence, typename T0>
struct enumerated_fold_impl<Functor, State, I, Sequence<T0>> {
  using type = brigand::apply<Functor, State, T0, I>;
};

template <typename Functor, typename State, typename I,
          template <typename...> class Sequence, typename T0, typename T1>
struct enumerated_fold_impl<Functor, State, I, Sequence<T0, T1>> {
  using type = brigand::apply<Functor, brigand::apply<Functor, State, T0, I>,
                              T1, brigand::next<I>>;
};

template <typename Functor, typename State, typename I,
          template <typename...> class Sequence, typename T0, typename T1,
          typename T2>
struct enumerated_fold_impl<Functor, State, I, Sequence<T0, T1, T2>> {
  using type = brigand::apply<
      Functor,
      brigand::apply<Functor, brigand::apply<Functor, State, T0, I>, T1,
                     brigand::next<I>>,
      T2, brigand::plus<I, brigand::int32_t<2>>>;
};

template <typename Functor, typename State, typename I,
          template <typename...> class Sequence, typename T0, typename T1,
          typename T2, typename T3>
struct enumerated_fold_impl<Functor, State, I, Sequence<T0, T1, T2, T3>> {
  using type = brigand::apply<
      Functor,
      brigand::apply<
          Functor,
          brigand::apply<Functor, brigand::apply<Functor, State, T0, I>, T1,
                         brigand::next<I>>,
          T2, brigand::plus<I, brigand::int32_t<2>>>,
      T3, brigand::plus<I, brigand::int32_t<3>>>;
};

template <typename Functor, typename State, typename I,
          template <typename...> class Sequence, typename T0, typename T1,
          typename T2, typename T3, typename T4>
struct enumerated_fold_impl<Functor, State, I, Sequence<T0, T1, T2, T3, T4>> {
  using type = brigand::apply<
      Functor,
      brigand::apply<
          Functor,
          brigand::apply<
              Functor,
              brigand::apply<Functor, brigand::apply<Functor, State, T0, I>, T1,
                             brigand::next<I>>,
              T2, brigand::plus<I, brigand::int32_t<2>>>,
          T3, brigand::plus<I, brigand::int32_t<3>>>,
      T4, brigand::plus<I, brigand::int32_t<4>>>;
};

template <typename Functor, typename State, typename I,
          template <typename...> class Sequence, typename T0, typename T1,
          typename T2, typename T3, typename T4, typename T5>
struct enumerated_fold_impl<Functor, State, I,
                            Sequence<T0, T1, T2, T3, T4, T5>> {
  using type = brigand::apply<
      Functor,
      brigand::apply<
          Functor,
          brigand::apply<
              Functor,
              brigand::apply<
                  Functor,
                  brigand::apply<Functor, brigand::apply<Functor, State, T0, I>,
                                 T1, brigand::next<I>>,
                  T2, brigand::plus<I, brigand::int32_t<2>>>,
              T3, brigand::plus<I, brigand::int32_t<3>>>,
          T4, brigand::plus<I, brigand::int32_t<4>>>,
      T5, brigand::plus<I, brigand::int32_t<5>>>;
};

template <typename Functor, typename State, typename I,
          template <typename...> class Sequence, typename T0, typename T1,
          typename T2, typename T3, typename T4, typename T5, typename T6>
struct enumerated_fold_impl<Functor, State, I,
                            Sequence<T0, T1, T2, T3, T4, T5, T6>> {
  using type = brigand::apply<
      Functor,
      brigand::apply<
          Functor,
          brigand::apply<
              Functor,
              brigand::apply<
                  Functor,
                  brigand::apply<
                      Functor,
                      brigand::apply<Functor,
                                     brigand::apply<Functor, State, T0, I>, T1,
                                     brigand::next<I>>,
                      T2, brigand::plus<I, brigand::int32_t<2>>>,
                  T3, brigand::plus<I, brigand::int32_t<3>>>,
              T4, brigand::plus<I, brigand::int32_t<4>>>,
          T5, brigand::plus<I, brigand::int32_t<5>>>,
      T6, brigand::plus<I, brigand::int32_t<6>>>;
};

template <typename Functor, typename State, typename I,
          template <typename...> class Sequence, typename T0, typename T1,
          typename T2, typename T3, typename T4, typename T5, typename T6,
          typename T7>
struct enumerated_fold_impl<Functor, State, I,
                            Sequence<T0, T1, T2, T3, T4, T5, T6, T7>> {
  using type = brigand::apply<
      Functor,
      brigand::apply<
          Functor,
          brigand::apply<
              Functor,
              brigand::apply<
                  Functor,
                  brigand::apply<
                      Functor,
                      brigand::apply<
                          Functor,
                          brigand::apply<Functor,
                                         brigand::apply<Functor, State, T0, I>,
                                         T1, brigand::next<I>>,
                          T2, brigand::plus<I, brigand::int32_t<2>>>,
                      T3, brigand::plus<I, brigand::int32_t<3>>>,
                  T4, brigand::plus<I, brigand::int32_t<4>>>,
              T5, brigand::plus<I, brigand::int32_t<5>>>,
          T6, brigand::plus<I, brigand::int32_t<6>>>,
      T7, brigand::plus<I, brigand::int32_t<7>>>;
};

template <typename Functor, typename State, typename I,
          template <typename...> class Sequence, typename T0, typename T1,
          typename T2, typename T3, typename T4, typename T5, typename T6,
          typename T7, typename... T>
struct enumerated_fold_impl<Functor, State, I,
                            Sequence<T0, T1, T2, T3, T4, T5, T6, T7, T...>>
    : enumerated_fold_impl<
          Functor,
          brigand::apply<
              Functor,
              brigand::apply<
                  Functor,
                  brigand::apply<
                      Functor,
                      brigand::apply<
                          Functor,
                          brigand::apply<
                              Functor,
                              brigand::apply<
                                  Functor,
                                  brigand::apply<
                                      Functor,
                                      brigand::apply<Functor, State, T0, I>, T1,
                                      brigand::next<I>>,
                                  T2, brigand::plus<I, brigand::int32_t<2>>>,
                              T3, brigand::plus<I, brigand::int32_t<3>>>,
                          T4, brigand::plus<I, brigand::int32_t<4>>>,
                      T5, brigand::plus<I, brigand::int32_t<5>>>,
                  T6, brigand::plus<I, brigand::int32_t<6>>>,
              T7, brigand::plus<I, brigand::int32_t<7>>>,
          brigand::plus<I, brigand::int32_t<8>>, Sequence<T...>> {};
}  // namespace detail

namespace lazy {
template <typename Sequence, typename State, typename Functor,
          typename I = brigand::int32_t<0>>
using enumerated_fold =
    typename detail::enumerated_fold_impl<Functor, State, I, Sequence>;
}  // namespace lazy

template <typename Sequence, typename State, typename Functor,
          typename I = brigand::int32_t<0>>
using enumerated_fold =
    typename lazy::enumerated_fold<Sequence, State, Functor, I>::type;
}  // namespace brigand

namespace brigand {
template <bool>
struct conditional;

template <>
struct conditional<true> {
  template <typename T, typename F>
  using type = T;
};

template <>
struct conditional<false> {
  template <typename T, typename F>
  using type = F;
};

template <bool B, typename T, typename F>
using conditional_t = typename conditional<B>::template type<T, F>;
}  // namespace brigand

namespace brigand {
template <typename List>
using remove_duplicates =
    fold<List, clear<List>,
         if_<bind<none, _state, defer<std::is_same<_1, parent<_element>>>>,
             bind<push_back, _state, _element>, _state>>;

template <bool>
struct branch_if;

template <>
struct branch_if<true> {
  template <typename T, typename F>
  using type = typename detail::apply<T>::type;
};

template <>
struct branch_if<false> {
  template <typename T, typename F>
  using type = typename detail::apply<F>::type;
};

template <bool B, typename T, typename F>
using branch_if_t = typename branch_if<B>::template type<T, F>;
}  // namespace brigand

namespace tmpl = brigand;

/*!
 * \ingroup UtilitiesGroup
 * \brief Metaprogramming things that are not planned to be submitted to Brigand
 */
namespace tmpl2 {
/*!
 * \ingroup UtilitiesGroup
 * \brief A compile-time list of values of the same type
 */
template <class T, T...>
struct value_list {};

/*!
 * \ingroup UtilitiesGroup
 * \brief A non-short-circuiting logical AND between bools 'B""
 *
 * Useful when arbitrarily large parameter packs need to be evaluated, since
 * std::conjunction and std::disjunction use recursion
 */
template <bool... Bs>
using flat_all =
    std::is_same<value_list<bool, Bs...>,
                 value_list<bool, (static_cast<void>(Bs), true)...>>;

/*!
 * \ingroup UtilitiesGroup
 * \brief A non-short-circuiting logical AND between bools 'B""
 *
 * Useful when arbitrarily large parameter packs need to be evaluated, since
 * std::conjunction and std::disjunction use recursion
 */
template <bool... Bs>
constexpr bool flat_all_v = flat_all<Bs...>::value;

/*!
 * \ingroup UtilitiesGroup
 * \brief A non-short-circuiting logical OR between bools 'B""
 *
 * Useful when arbitrarily large parameter packs need to be evaluated, since
 * std::conjunction and std::disjunction use recursion
 */
template <bool... Bs>
using flat_any = std::integral_constant<
    bool, not std::is_same<
              value_list<bool, Bs...>,
              value_list<bool, (static_cast<void>(Bs), false)...>>::value>;

/*!
 * \ingroup UtilitiesGroup
 * \brief A non-short-circuiting logical OR between bools 'B""
 *
 * Useful when arbitrarily large parameter packs need to be evaluated, since
 * std::conjunction and std::disjunction use recursion
 */
template <bool... Bs>
constexpr bool flat_any_v = flat_any<Bs...>::value;
}  // namespace tmpl2

/*!
 * \ingroup UtilitiesGroup
 * \brief Allows zero-cost unordered expansion of a parameter
 *
 * \details
 * Expands a parameter pack, typically useful for runtime evaluation via a
 * Callable such as a lambda, function, or function object. For example,
 * an unordered transform of a std::tuple can be implemented as:
 * \snippet Utilities/Test_TMPL.cpp expand_pack_example
 *
 * \see tuple_fold tuple_counted_fold tuple_transform std::tuple
 * EXPAND_PACK_LEFT_TO_RIGHT
 */
template <typename... Ts>
constexpr void expand_pack(Ts&&... /*unused*/) {}

/*!
 * \ingroup UtilitiesGroup
 * \brief Expand a parameter pack evaluating the terms from left to right.
 *
 * The parameter pack inside the argument to the macro must not be expanded
 * since the macro will do the expansion correctly for you. In the below example
 * a parameter pack of `std::integral_constant<size_t, I>` is passed to the
 * function. The closure `lambda` is used to sum up the values of all the `Ts`.
 * Note that the `Ts` passed to `EXPAND_PACK_LEFT_TO_RIGHT` is not expanded.
 *
 * \snippet Utilities/Test_TMPL.cpp expand_pack_left_to_right
 *
 * \see tuple_fold tuple_counted_fold tuple_transform std::tuple expand_pack
 */
#define EXPAND_PACK_LEFT_TO_RIGHT(...) \
  (void)std::initializer_list<char> { ((void)(__VA_ARGS__), '0')... }

/*!
 * \ingroup UtilitiesGroup
 * \brief Returns the first argument of a parameter pack
 */
template <typename T, typename... Ts>
constexpr decltype(auto) get_first_argument(T&& t, Ts&&... /*rest*/) {
  return t;
}

namespace brigand {
namespace lazy {
/// Check if a typelist contains an item.
template <typename Sequence, typename Item>
struct list_contains;

/// \cond
template <template <typename...> typename L, typename... Items, typename Item>
struct list_contains<L<Items...>, Item>
    : bool_<(... or std::is_same_v<Items, Item>)> {};
/// \endcond
}  // namespace lazy

/// Check if a typelist contains an item.
/// @{
template <typename Sequence, typename Item>
using list_contains = typename lazy::list_contains<Sequence, Item>::type;

template <typename Sequence, typename Item>
constexpr bool list_contains_v = lazy::list_contains<Sequence, Item>::value;
/// @}

/// Obtain the elements of `Sequence1` that are not in `Sequence2`.
template <typename Sequence1, typename Sequence2>
using list_difference =
    fold<Sequence2, Sequence1, lazy::remove<_state, _element>>;

namespace detail {
template <typename List>
struct as_pack_impl;

template <template <typename...> typename L, typename... Args>
struct as_pack_impl<L<Args...>> {
  template <typename F>
  static constexpr decltype(auto) apply(F&& f) {
    return std::forward<F>(f)(type_<Args>{}...);
  }
};
}  // namespace detail

/// Call a functor with the types from a list.
///
/// Given a typelist `List = tmpl::list<A, B, ...>` (not necessarily
/// with head `tmpl::list`), calls \p f as `f(tmpl::type_<A>{},
/// tmpl::type_<B>{}, ...)` and returns the result.
///
/// This is useful for converting a typelist into a parameter pack.
///
/// \snippet Utilities/Test_TMPL.cpp as_pack
template <typename List, typename F>
constexpr decltype(auto) as_pack(F&& f) {
  return detail::as_pack_impl<List>::apply(std::forward<F>(f));
}

namespace detail {
template <typename Sequence>
struct make_std_variant_over_impl;

template <template <typename...> class Sequence, typename... Ts>
struct make_std_variant_over_impl<Sequence<Ts...>> {
  static_assert(((not std::is_same_v<std::decay_t<std::remove_pointer_t<Ts>>,
                                     void>)&&...),
                "Cannot create a std::variant with a 'void' type.");
  using type = std::variant<Ts...>;
};
}  // namespace detail

/*!
 * \ingroup UtilitiesGroup
 * \brief Create a std::variant with all all the types inside the typelist
 * Sequence
 *
 * \metareturns std::variant of all types inside `Sequence`
 */
template <typename Sequence>
using make_std_variant_over = typename detail::make_std_variant_over_impl<
    tmpl::remove_duplicates<Sequence>>::type;
}  // namespace brigand
