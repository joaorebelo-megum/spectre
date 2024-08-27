// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/Rational.hpp"

#include <boost/functional/hash.hpp>
#include <boost/integer/common_factor_rt.hpp>
#include <ostream>
#include <pup.h>
#include <tuple>

#include "Utilities/ErrorHandling/Assert.hpp"

namespace {
// This is needed a lot, so define a shorter name.
std::int64_t to64(const std::int32_t n) { return static_cast<std::int64_t>(n); }

template <typename IntType>
std::tuple<std::int32_t, std::int32_t> reduce(IntType numerator,
                                              IntType denominator) {
  const IntType common_factor = boost::integer::gcd(numerator, denominator);
  numerator /= common_factor;
  denominator /= common_factor;
  if (denominator < 0) {
    numerator = -numerator;
    denominator = -denominator;
  }
  ASSERT(static_cast<std::int32_t>(numerator) == numerator and
         static_cast<std::int32_t>(denominator) == denominator,
         "Rational overflow: " << numerator << "/" << denominator);
  return std::make_tuple(numerator, denominator);
}
}  // namespace

Rational::Rational(const std::int32_t numerator,
                   const std::int32_t denominator) {
  ASSERT(denominator != 0, "Division by zero");
  std::tie(numerator_, denominator_) = reduce(numerator, denominator);
}

double Rational::value() const {
  return static_cast<double>(numerator_) / static_cast<double>(denominator_);
}

Rational Rational::inverse() const {
  // Default construct to avoid the reduce() call.
  ASSERT(*this != 0, "Division by zero");
  Rational ret;
  ret.numerator_ = denominator_;
  ret.denominator_ = numerator_;
  if (ret.denominator_ < 0) {
    ret.numerator_ = -ret.numerator_;
    ret.denominator_ = -ret.denominator_;
  }
  return ret;
}

Rational& Rational::operator+=(const Rational& other) {
  std::tie(numerator_, denominator_) =
      reduce(to64(numerator_) * to64(other.denominator_) +
             to64(denominator_) * to64(other.numerator_),
             to64(denominator_) * to64(other.denominator_));
  return *this;
}
Rational& Rational::operator-=(const Rational& other) {
  return *this += -other;
}
Rational& Rational::operator*=(const Rational& other) {
  std::tie(numerator_, denominator_) =
      reduce(to64(numerator_) * to64(other.numerator()),
             to64(denominator_) * to64(other.denominator()));
  return *this;
}
Rational& Rational::operator/=(const Rational& other) {
  return *this *= other.inverse();
}

void Rational::pup(PUP::er& p) {
  p | numerator_;
  p | denominator_;
}

Rational operator-(Rational r) {
  // No reduced-form check needed
  r.numerator_ = -r.numerator_;
  return r;
}

Rational operator+(const Rational& a, const Rational& b) {
  Rational ret = a;
  ret += b;
  return ret;
}
Rational operator-(const Rational& a, const Rational& b) {
  Rational ret = a;
  ret -= b;
  return ret;
}
Rational operator*(const Rational& a, const Rational& b) {
  Rational ret = a;
  ret *= b;
  return ret;
}
Rational operator/(const Rational& a, const Rational& b) {
  Rational ret = a;
  ret /= b;
  return ret;
}

bool operator==(const Rational& a, const Rational& b) {
  return a.numerator() == b.numerator() and a.denominator() == b.denominator();
}
bool operator!=(const Rational& a, const Rational& b) { return not(a == b); }
bool operator<(const Rational& a, const Rational& b) {
  return to64(a.numerator()) * to64(b.denominator()) <
         to64(b.numerator()) * to64(a.denominator());
}
bool operator>(const Rational& a, const Rational& b) { return b < a; }
bool operator<=(const Rational& a, const Rational& b) { return not(b < a); }
bool operator>=(const Rational& a, const Rational& b) { return not(a < b); }

Rational abs(const Rational& r) { return r.numerator() >= 0 ? r : -r; }

std::ostream& operator<<(std::ostream& os, const Rational& r) {
  return os << r.numerator() << '/' << r.denominator();
}

size_t hash_value(const Rational& r) {
  size_t h = 0;
  boost::hash_combine(h, r.numerator());
  boost::hash_combine(h, r.denominator());
  return h;
}

// clang-tidy: do not modify std namespace (okay for hash)
namespace std {  // NOLINT
size_t hash<Rational>::operator()(const Rational& r) const {
  return boost::hash<Rational>{}(r);
}
}  // namespace std
