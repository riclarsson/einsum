#include "ein.h"

#include <print>
#include <ranges>
#include <vector>

namespace stdv = std::ranges::views;
namespace stdx = std::experimental;

namespace {
void test_sum() {
  const std::vector<double> a{1, 2, 3, 4};
  std::println("a              = {}", a);

  std::println("Method to sum elements of a vector:");
  double v{};
  ein::sum<"", "i">(v, a);
  std::println("sum(a)         = {}", v);

  std::println("Method to sum the square of elements of a vector:");
  ein::sum<"", "i", "i">(v, a, a);
  std::println("sum(dot(a, a)) = {}", v);

  std::println("Method to store the square of elements of a vector:");
  std::vector<double> b(4);
  ein::sum<"i", "i", "i">(b, a, a);
  std::println("a .^ 2         = {}", b);

  std::println("Method to store the square of square of elements of a vector:");
  ein::sum<"i", "i", "i">(b, a, b);
  std::println("a .^ 4         = {}", b);

  std::println("You can use ranges to transform results partially:");
  ein::sum<"i", "i", "i">(
      b, a, a | stdv::transform([](double x) { return std::exp(-x); }));
  std::println("a .* exp(-a)   = {}", b);

  std::println("Importantly, you can deal with 'tensors':");
  stdx::mdspan<const double, stdx::dextents<std::size_t, 2>> A(
      a.data(), std::array{2, 2});
  std::println(R"(A              = [[{}, {}],
                  [{}, {}]])",
               A[0, 0],
               A[0, 1],
               A[1, 0],
               A[1, 1]);
  const std::vector<double> B{3, -2};
  std::println("B              = {}^T", B);

  std::println("So matmul works:");
  std::vector<double> c(2);
  ein::sum<"i", "ij", "j">(c, A, B);
  std::println("A * B          = {}", c);

  std::println("As does weird transformations of the matrix multiplication:");
  ein::sum<"i", "ij", "j", "j">(c, A, B, B);
  std::println("A * (B .* B)   = {}", c);

  std::println("And transposes:");
  ein::sum<"i", "ji", "j", "j">(c, A, B, B);
  std::println("A^T * (B .* B) = {}", c);

  std::println(
      "Note that ein::sum is not optimized in this example, so you may want to modify it to use appropriate optimizations for specific cases.");
}

void test_tra() {
  const std::vector<double> a{1, 2, 3, 4};
  std::println("a              = {}", a);

  double v{};
  ein::tra<"", "i">(
      v,
      [](const auto& x) {
        double v{};
        for (const auto& xi : x) v += xi;
        return v;
      },
      a);
  std::println("sum(a)         = {}", v);

  v = 1.0;
  ein::tra<"", "i">(
      v,
      [](const auto& x) {
        double v{1.0};
        for (const auto& xi : x) v *= xi;
        return v;
      },
      a);
  std::println("gam(a)         = {}", v);

  v = 0.0;
  ein::tra<"", "i", "i">(
      v,
      [](const auto& x, const auto& y) {
        double v{};
        for (const auto& [xi, yi] : stdv::zip(x, y)) v += xi * yi;
        return v;
      },
      a,
      a);
  std::println("sum(dot(a, a)) = {}", v);

  v = 1.0;
  ein::tra<"", "i", "i">(
      v,
      [](const auto& x, const auto& y) {
        double v{1.0};
        for (const auto& [xi, yi] : stdv::zip(x, y)) v *= xi * yi;
        return v;
      },
      a,
      a);
  std::println("gam(dot(a, a)) = {}", v);

  v = 0.0;
  ein::tra<"", "i">(
      v,
      [](const auto& x) {
        double v{};
        for (const auto& xi : x) v += xi * xi;
        return std::sqrt(v);
      },
      a);
  std::println("hypot(a)       = {}", v);

  stdx::mdspan<const double, stdx::dextents<std::size_t, 2>> A(
      a.data(), std::array{2, 2});
  std::println(R"(A              = [[{}, {}],
                  [{}, {}]])",
               A[0, 0],
               A[0, 1],
               A[1, 0],
               A[1, 1]);
  const std::vector<double> B{3, -2};
  std::println("B              = {}^T", B);

  std::vector<double> C(2);
  ein::tra<"i", "ij", "j">(
      C,
      [](const auto& x, const auto& y) {
        double v{};
        for (const auto& [xi, yi] : stdv::zip(x, y)) v += xi * yi;
        return v;
      },
      A,
      B);
  std::println("A * B          = {}", C);
}
}  // namespace

int main() {
  std::println("EIN::SUM tests");
  test_sum();

  std::println("EIN::TRA tests");
  test_tra();
}