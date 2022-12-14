/**
 * Copyright 2017 MapD Technologies, Inc.
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "DateTimeEnums.h"

#include <iostream>
#include <string>

inline std::string toString(hdk::ir::TimeUnit unit) {
  switch (unit) {
    case hdk::ir::TimeUnit::kMonth:
      return "Month";
    case hdk::ir::TimeUnit::kDay:
      return "Day";
    case hdk::ir::TimeUnit::kSecond:
      return "Second";
    case hdk::ir::TimeUnit::kMilli:
      return "Milli";
    case hdk::ir::TimeUnit::kMicro:
      return "Micro";
    case hdk::ir::TimeUnit::kNano:
      return "Nano";
    default:
      return "InvalidTimeUnit";
  }
}

inline std::string toString(hdk::ir::DateAddField field) {
  switch (field) {
    case hdk::ir::DateAddField::kYear:
      return "Year";
    case hdk::ir::DateAddField::kQuarter:
      return "Quarter";
    case hdk::ir::DateAddField::kMonth:
      return "Month";
    case hdk::ir::DateAddField::kDay:
      return "Day";
    case hdk::ir::DateAddField::kHour:
      return "Hour";
    case hdk::ir::DateAddField::kMinute:
      return "Minute";
    case hdk::ir::DateAddField::kSecond:
      return "Second";
    case hdk::ir::DateAddField::kMillennium:
      return "Millennium";
    case hdk::ir::DateAddField::kCentury:
      return "Century";
    case hdk::ir::DateAddField::kDecade:
      return "Decade";
    case hdk::ir::DateAddField::kMilli:
      return "Milli";
    case hdk::ir::DateAddField::kMicro:
      return "Micro";
    case hdk::ir::DateAddField::kNano:
      return "Nano";
    case hdk::ir::DateAddField::kWeek:
      return "Week";
    case hdk::ir::DateAddField::kQuarterDay:
      return "QuarterDay";
    case hdk::ir::DateAddField::kWeekDay:
      return "WeekDay";
    case hdk::ir::DateAddField::kDayOfYear:
      return "DayOfYear";
    case hdk::ir::DateAddField::kInvalid:
      return "Invalid";
    default:
      return "InvalidDateAddField";
  }
}

inline std::string toString(hdk::ir::DateTruncField field) {
  switch (field) {
    case hdk::ir::DateTruncField::kYear:
      return "Year";
    case hdk::ir::DateTruncField::kQuarter:
      return "Quarter";
    case hdk::ir::DateTruncField::kMonth:
      return "Month";
    case hdk::ir::DateTruncField::kDay:
      return "Day";
    case hdk::ir::DateTruncField::kHour:
      return "Hour";
    case hdk::ir::DateTruncField::kMinute:
      return "Minute";
    case hdk::ir::DateTruncField::kSecond:
      return "Second";
    case hdk::ir::DateTruncField::kMilli:
      return "Milli";
    case hdk::ir::DateTruncField::kMicro:
      return "Micro";
    case hdk::ir::DateTruncField::kNano:
      return "Nano";
    case hdk::ir::DateTruncField::kMillennium:
      return "Millennium";
    case hdk::ir::DateTruncField::kCentury:
      return "Century";
    case hdk::ir::DateTruncField::kDecade:
      return "Decade";
    case hdk::ir::DateTruncField::kWeek:
      return "Week";
    case hdk::ir::DateTruncField::kWeekSunday:
      return "WeekSunday";
    case hdk::ir::DateTruncField::kWeekSaturday:
      return "WeekSaturday";
    case hdk::ir::DateTruncField::kQuarterDay:
      return "QuarterDay";
    case hdk::ir::DateTruncField::kInvalid:
      return "Invalid";
    default:
      return "InvalidDateTruncField";
  }
}

inline std::string toString(hdk::ir::DateExtractField field) {
  switch (field) {
    case hdk::ir::DateExtractField::kYear:
      return "Year";
    case hdk::ir::DateExtractField::kQuarter:
      return "Quarter";
    case hdk::ir::DateExtractField::kMonth:
      return "Month";
    case hdk::ir::DateExtractField::kDay:
      return "Day";
    case hdk::ir::DateExtractField::kHour:
      return "Hour";
    case hdk::ir::DateExtractField::kMinute:
      return "Minute";
    case hdk::ir::DateExtractField::kSecond:
      return "Second";
    case hdk::ir::DateExtractField::kMilli:
      return "Milli";
    case hdk::ir::DateExtractField::kMicro:
      return "Micro";
    case hdk::ir::DateExtractField::kNano:
      return "Nano";
    case hdk::ir::DateExtractField::kDayOfWeek:
      return "DayOfWeek";
    case hdk::ir::DateExtractField::kIsoDayOfWeek:
      return "IsoDayOfWeek";
    case hdk::ir::DateExtractField::kDayOfYear:
      return "DayOfYear";
    case hdk::ir::DateExtractField::kEpoch:
      return "Epoch";
    case hdk::ir::DateExtractField::kQuarterDay:
      return "QuarterDay";
    case hdk::ir::DateExtractField::kWeek:
      return "Week";
    case hdk::ir::DateExtractField::kWeekSunday:
      return "WeekSunday";
    case hdk::ir::DateExtractField::kWeekSaturday:
      return "WeekSaturday";
    case hdk::ir::DateExtractField::kDateEpoch:
      return "DateEpoch";
    default:
      return "InvalidDateExtractField";
  }
}

namespace hdk::ir {

inline std::ostream& operator<<(std::ostream& os, TimeUnit unit) {
  os << toString(unit);
  return os;
}

inline std::ostream& operator<<(std::ostream& os, DateAddField field) {
  os << toString(field);
  return os;
}

inline std::ostream& operator<<(std::ostream& os, DateTruncField field) {
  os << toString(field);
  return os;
}

inline int64_t unitsPerSecond(TimeUnit unit) {
  switch (unit) {
    case hdk::ir::TimeUnit::kSecond:
      return 1;
    case hdk::ir::TimeUnit::kMilli:
      return 1'000;
    case hdk::ir::TimeUnit::kMicro:
      return 1'000'000;
    case hdk::ir::TimeUnit::kNano:
      return 1'000'000'000;
    default:
      throw std::runtime_error("Enexpected unit in unitsPerSecond: " + ::toString(unit));
  }
}

}  // namespace hdk::ir
