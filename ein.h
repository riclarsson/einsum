#include <algorithm>
#include <array>
#include <experimental/mdspan>
#include <limits>
#include <ranges>
#include <tuple>
#include <utility>

namespace ein {
namespace stdx = std::experimental;
namespace stdr = std::ranges;

namespace {
template <std::size_t N>
struct str_ {
  consteval str_(const char (&in)[N]) { stdr::copy_n(in, N, str); }
  char str[N];

  consteval std::array<char, N - 1> to_array() const {
    std::array<char, N - 1> out{};
    stdr::copy_n(str, N - 1, out.begin());
    return out;
  }
};

template <std::array cs>
consteval bool empty_() {
  return cs.size() == 0;
}

template <std::array lh, std::array... rh>
consteval char char_() {
  constexpr std::size_t N = lh.size();
  constexpr std::size_t M = sizeof...(rh);

  if constexpr (N == 0 and M == 0) {
    return '\0';
  } else if constexpr (N != 0) {
    //! NOTE: Here is a good place to add optimizations
    //        for specific cases, such as empty sparse arrays
    //        returning '\0' short-circuits the reduction

    return lh.front();
  } else if constexpr (M != 0) {
    return char_<rh...>();
  } else {
    return '\0';
  }
}

template <char c, std::array cs>
consteval std::size_t find_() {
  constexpr std::size_t N = cs.size();

  for (std::size_t i = 0; i < N; ++i) {
    if (cs[i] == c) return i;
  }

  return std::numeric_limits<std::size_t>::max();
}

template <char c, std::array cf, std::array... cs>
constexpr std::size_t size_(const auto& xf, const auto&... xs) {
  constexpr std::size_t N = cf.size();

  if constexpr (N > 0 and char_<cf>() == c) {
    constexpr std::size_t dim = find_<c, cf>();

    if constexpr (requires {
                    { xf.extent(dim) } -> std::integral;
                  }) {
      return static_cast<std::size_t>(xf.extent(dim));
    } else if constexpr (requires {
                           { stdr::size(xf) } -> std::integral;
                         }) {
      return stdr::size(xf);
    } else {
      throw "Cannot determine size";
    }
  } else if constexpr (sizeof...(cs) == 0) {
    return std::numeric_limits<std::size_t>::max();
  } else {
    return size_<c, cs...>(xs...);
  }
}

template <char c, std::array cs>
consteval auto drop_() {
  constexpr std::size_t N = cs.size();

  std::array<char, N - 1> result{};
  bool done = false;
  for (std::size_t i = 0; i < N; ++i) {
    if (cs[i] == c and not done) {
      done = true;
    } else {
      result[i - done] = cs[i];
    }
  }

  return result;
}

template <char c, std::array cs>
consteval std::size_t count_() {
  constexpr std::size_t N = cs.size();
  std::size_t n           = 0;

  for (std::size_t i = 0; i < N; ++i) n += cs[i] == c;

  return n;
}

template <char c, std::array cs>
consteval auto redrank_() {
  if constexpr (count_<c, cs>() == 0) {
    return cs;
  } else {
    return redrank_<c, drop_<c, cs>()>();
  }
}

template <std::size_t N>
consteval std::array<stdx::full_extent_t, N> jokers_() {
  std::array<stdx::full_extent_t, N> out;
  out.fill(stdx::full_extent);
  return out;
}

template <std::size_t M, std::size_t N, typename T>
[[nodiscard]] constexpr auto tup_(T&& i)
  requires(M < N)
{
  constexpr std::size_t l = M;
  constexpr std::size_t r = N - M - 1;

  static_assert(l < N, "Left must be less than N");
  static_assert(r < N, "Right must be less than N");

  constexpr bool has_left  = l > 0;
  constexpr bool has_right = r > 0;

  if constexpr (has_left and has_right) {
    return std::tuple_cat(
        jokers_<l>(), std::forward_as_tuple(std::forward<T>(i)), jokers_<r>());
  } else if constexpr (has_left) {
    return std::tuple_cat(jokers_<l>(),
                          std::forward_as_tuple(std::forward<T>(i)));
  } else if constexpr (has_right) {
    return std::tuple_cat(std::forward_as_tuple(std::forward<T>(i)),
                          jokers_<r>());
  } else {
    static_assert(M == 0, "M must be 0");
    static_assert(N == 1, "N must be 1");
    return std::forward_as_tuple(std::forward<T>(i));
  }
}

template <std::size_t M, std::size_t N, typename Self>
constexpr decltype(auto) sub_(Self&& s, std::size_t i)
  requires(M < N)
{
  return std::apply(
      [&s]<typename... AccT>(AccT&&... x) {
        return stdx::submdspan(s, std::forward<AccT>(x)...);
      },
      tup_<M, N>(i));
}

template <char c, std::array cs, typename T>
constexpr decltype(auto) reddim_(T&& arr, std::size_t i [[maybe_unused]]) {
  constexpr std::size_t N = cs.size();
  constexpr std::size_t n = count_<c, cs>();

  if constexpr (n == 0 or N == 0) {
    return std::forward<T>(arr);
  } else if constexpr (n == 1) {
    if constexpr (N == 1) {
      return arr[i];
    } else {
      return sub_<find_<c, cs>(), N>(arr, i);
    }
  } else {
    return reddim_<c, drop_<c, cs>()>(sub_<find_<c, cs>(), N>(arr, i), i);
  }
}

template <typename T, std::array... cs>
T redsum_(const auto&... xs) {
  if constexpr ((empty_<cs>() and ...)) {
    return (... * static_cast<T>(xs));
  } else {
    T sum{};

    if constexpr (constexpr char first_char = char_<cs...>();
                  first_char != '\0') {
      const std::size_t n = size_<first_char, cs...>(xs...);

      for (std::size_t i = 0; i < n; ++i) {
        sum += redsum_<T, redrank_<first_char, cs>()...>(
            reddim_<first_char, cs>(xs, i)...);
      }
    }

    return sum;
  }
}

template <std::array rs, std::array... cs>
constexpr void sum_(auto&& xr, const auto&... xs) {
  if constexpr (empty_<rs>()) {
    using T = std::remove_cvref_t<decltype(xr)>;
    xr      = redsum_<T, cs...>(xs...);
  }

  /*! NOTE: 
   * Here is a good place to add optimizations
   * for specific cases, such as copy, matmul, etc.
   * For instance, any "i", "ij", "j" pattern is a
   * matrix-vector multiplication.
   *
   * So if an optimization is implemented, patterns
   * like "mi", "mij", "mj" would end up making use of it.
   */

  else if constexpr (constexpr char first_char = char_<rs>();
                     first_char != '\0') {
    const std::size_t n = size_<first_char, rs>(xr);

    for (std::size_t i = 0; i < n; i++) {
      sum_<redrank_<first_char, rs>(), redrank_<first_char, cs>()...>(
          reddim_<first_char, rs>(xr, i), reddim_<first_char, cs>(xs, i)...);
    }
  }
}

template <std::array rs, std::array... cs, class Transform>
constexpr void tra_(auto&& xr, Transform tra, const auto&... xs) {
  if constexpr (empty_<rs>()) {
    xr = tra(xs...);
  } else if constexpr (constexpr char first_char = char_<rs>();
                       first_char != '\0') {
    const std::size_t n = size_<first_char, rs>(xr);

    for (std::size_t i = 0; i < n; i++) {
      sum_<redrank_<first_char, rs>(), redrank_<first_char, cs>()...>(
          reddim_<first_char, rs>(xr, i), reddim_<first_char, cs>(xs, i)...);
    }
  }
}
}  // namespace

template <str_ sr, str_... si>
constexpr void sum(auto&& xr, const auto&... xi)
  requires(sizeof...(si) == sizeof...(xi))
{
  sum_<sr.to_array(), si.to_array()...>(std::forward<decltype(xr)>(xr), xi...);
}

template <str_ sr, str_... si, class Transform>
constexpr void tra(auto&& xr, Transform tra, const auto&... xi)
  requires(sizeof...(si) == sizeof...(xi))
{
  tra_<sr.to_array(), si.to_array()...>(
      std::forward<decltype(xr)>(xr), std::move(tra), xi...);
}
}  // namespace ein
