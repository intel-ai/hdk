/*
 * Copyright (c) 2020 OmniSci, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "IR/Type.h"
#include "sqltypes.h"

#include <optional>
#include <ostream>
#include <string_view>

template <hdk::ir::Type::Id TYPE>
std::optional<int64_t> dateTimeParseOptional(std::string_view, hdk::ir::TimeUnit unit);

template <>
std::optional<int64_t> dateTimeParseOptional<hdk::ir::Type::kDate>(
    std::string_view,
    hdk::ir::TimeUnit unit);

template <>
std::optional<int64_t> dateTimeParseOptional<hdk::ir::Type::kTime>(
    std::string_view,
    hdk::ir::TimeUnit unit);

template <>
std::optional<int64_t> dateTimeParseOptional<hdk::ir::Type::kTimestamp>(
    std::string_view,
    hdk::ir::TimeUnit unit);

template <hdk::ir::Type::Id TYPE>
int64_t dateTimeParse(std::string_view const s, hdk::ir::TimeUnit unit) {
  if (auto const time = dateTimeParseOptional<TYPE>(s, unit)) {
    return *time;
  } else {
    throw std::runtime_error(
        cat("Invalid date/time (", std::to_string(TYPE), ") string (", s, ')'));
  }
}

namespace {

template <
    class To,
    class From,
    std::enable_if_t<std::is_arithmetic_v<To> && std::is_arithmetic_v<From>, bool> = true>
To numeric_cast(From v) {
  auto r = static_cast<To>(v);
  if (static_cast<From>(r) != v || std::signbit(r) != std::signbit(v))
    throw std::runtime_error("numeric_cast<>() failed");
  return r;
}
}  // namespace

template <typename R, hdk::ir::Type::Id TYPE>
R dateTimeParse(std::string_view const s, hdk::ir::TimeUnit unit) {
  if (auto const time = dateTimeParseOptional<TYPE>(s, unit)) {
    return numeric_cast<R>(*time);
  } else {
    throw std::runtime_error(
        cat("Invalid date/time (", std::to_string(TYPE), ") string (", s, ')'));
  }
}

template <hdk::ir::Type::Id TYPE>
int64_t dateTimeParse(std::string_view const s, int dim) {
  hdk::ir::TimeUnit unit;
  switch (dim) {
    case 0:
      unit = hdk::ir::TimeUnit::kSecond;
      break;
    case 3:
      unit = hdk::ir::TimeUnit::kMilli;
      break;
    case 6:
      unit = hdk::ir::TimeUnit::kMicro;
      break;
    case 9:
      unit = hdk::ir::TimeUnit::kNano;
      break;
    default:
      abort();
  }
  if (auto const time = dateTimeParseOptional<TYPE>(s, unit)) {
    return *time;
  } else {
    throw std::runtime_error(
        cat("Invalid date/time (", std::to_string(TYPE), ") string (", s, ')'));
  }
}

/**
 * Set format_type_ and parse date/time/timestamp strings into (s,ms,us,ns) since the
 * epoch based on given dim in (0,3,6,9) respectively.  Basic idea is to parse given
 * string by matching to formats ("%Y-%m-%d", "%m/%d/%Y", ...) until a valid parse is
 * found.  Save parsed values into a DateTime dt_ struct from which the final
 * epoch-based int64_t value is calculated.
 */
class DateTimeParser {
 public:
  enum class FormatType { Date, Time, Timezone };
  std::optional<int64_t> parse(std::string_view const, hdk::ir::TimeUnit unit);
  void setFormatType(FormatType);
  std::string_view unparsed() const;

  struct DateTime {
    int64_t Y{1970};        // Full year
    unsigned m{1};          // month (1-12)
    unsigned d{1};          // day of month (1-31)
    unsigned H{0};          // hour (0-23)
    unsigned M{0};          // minute (0-59)
    unsigned S{0};          // second (0-61)
    unsigned n{0};          // fraction of a second in nanoseconds (0-999999999)
    int z{0};               // timezone offset in seconds
    std::optional<bool> p;  // true if pm, false if am, nullopt if unspecified

    int64_t getTime(hdk::ir::TimeUnit unit) const;
    friend std::ostream& operator<<(std::ostream&, DateTime const&);
  };

 private:
  DateTime dt_;
  FormatType format_type_;
  std::string_view unparsed_;

  bool parseWithFormat(std::string_view format, std::string_view& str);
  void resetDateTime();
  bool updateDateTimeAndStr(char const field, std::string_view&);
};
