/*
 * Copyright 2017 MapD Technologies, Inc.
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

#include "ExtensionFunctionsBinding.h"
#include <algorithm>
#include "ExternalExecutor.h"

// A rather crude function binding logic based on the types of the arguments.
// We want it to be possible to write specialized versions of functions to be
// exposed as SQL extensions. This is important especially for performance
// reasons, since double operations can be significantly slower than float. We
// compute a score for each candidate signature based on conversions required to
// from the function arguments as specified in the SQL query to the versions in
// ExtensionFunctions.hpp.

/*
  New implementation for binding a SQL function operator to the
  optimal candidate within in all available extension functions.
 */

namespace {

static int get_numeric_scalar_scale(const hdk::ir::Type* arg_type) {
  switch (arg_type->id()) {
    case hdk::ir::Type::kBoolean:
    case hdk::ir::Type::kInteger:
    case hdk::ir::Type::kFloatingPoint:
      return arg_type->size();
    case hdk::ir::Type::kDecimal:
      return arg_type->as<hdk::ir::DecimalType>()->precision() > 7 ? 8 : 4;
    default:
      CHECK(false);
  }
  return -1;
}

/**
 * @brief returns true if the lhs_type can be cast to the rhs_type with no loss of
 * precision.
 */
static bool is_numeric_scalar_auto_castable(const hdk::ir::Type* lhs_type,
                                            const hdk::ir::Type* rhs_type) {
  switch (lhs_type->id()) {
    case hdk::ir::Type::kBoolean:
      return rhs_type->isBoolean();
    case hdk::ir::Type::kInteger:
      if (!rhs_type->isNumber()) {
        return false;
      }
      if (rhs_type->isFloatingPoint()) {
        // We can lose precision here, but preserving existing behavior
        return true;
      }
      return rhs_type->size() >= lhs_type->size();
    case hdk::ir::Type::kFloatingPoint:
      if (!rhs_type->isFloatingPoint()) {
        return false;
      }
      return rhs_type->size() >= lhs_type->size();
    case hdk::ir::Type::kDecimal:
      if (rhs_type->isDecimal()) {
        return rhs_type->as<hdk::ir::DecimalType>()->precision() >=
               lhs_type->as<hdk::ir::DecimalType>()->precision();
      } else if (rhs_type->isFp64()) {
        return true;
      } else if (rhs_type->isFp32()) {
        return lhs_type->as<hdk::ir::DecimalType>()->precision() <= 7;
      } else {
        return false;
      }
    case hdk::ir::Type::kTimestamp:
      if (!rhs_type->isTimestamp()) {
        return false;
      }
      return rhs_type->as<hdk::ir::TimestampType>()->unit() >=
             lhs_type->as<hdk::ir::TimestampType>()->unit();
    case hdk::ir::Type::kDate:
      return rhs_type->isDate();
    case hdk::ir::Type::kTime:
      return rhs_type->isTime();
    default:
      UNREACHABLE();
      return false;
  }
}

static int match_numeric_argument(const hdk::ir::Type* arg_type,
                                  const bool is_arg_literal,
                                  const ExtArgumentType& sig_ext_arg_type,
                                  int32_t& penalty_score) {
  CHECK(arg_type->isBoolean() || arg_type->isNumber());
  // Todo (todd): Add support for timestamp, date, and time types
  const auto sig_type = ext_arg_type_to_type(arg_type->ctx(), sig_ext_arg_type);

  // If we can't legally auto-cast to sig_type, abort
  if (!is_numeric_scalar_auto_castable(arg_type, sig_type)) {
    return -1;
  }

  // We now compare a measure of the scale of the sig_type with the
  // arg_type, which provides a basis for scoring the match between
  // the two.  Note that get_numeric_scalar_scale for the most part
  // returns the logical byte width of the type, with a few caveats
  // for decimals and timestamps described in more depth in comments
  // in the function itself.  Also even though for example float and
  // int types return 4 (as in 4 bytes), and double and bigint types
  // return 8, a fp32 type cannot express every 32-bit integer (even
  // if it can cover a larger absolute range), and an fp64 type
  // likewise cannot express every 64-bit integer.  With the aim to
  // minimize the precision loss from casting (always precise) integer
  // value to (imprecise) floating point value, in the case of integer
  // inputs, we'll penalize wider floating point argument types least
  // by a specific scale transformation (see the implementation
  // below). For instance, casting tinyint to fp64 is prefered over
  // casting it to fp32 to minimize precision loss.
  const bool is_integer_to_fp_cast = arg_type->isInteger() && sig_type->isFloatingPoint();

  const auto arg_type_relative_scale = get_numeric_scalar_scale(arg_type);
  CHECK_GE(arg_type_relative_scale, 1);
  CHECK_LE(arg_type_relative_scale, 8);
  auto sig_type_relative_scale = get_numeric_scalar_scale(sig_type);
  CHECK_GE(sig_type_relative_scale, 1);
  CHECK_LE(sig_type_relative_scale, 8);

  if (is_integer_to_fp_cast) {
    // transform fp scale: 4 becomes 16, 8 remains 8
    sig_type_relative_scale = (3 - (sig_type_relative_scale >> 2)) << 3;
  }

  // We do not allow auto-casting to types with less scale/precision
  // within the same type family.
  CHECK_GE(sig_type_relative_scale, arg_type_relative_scale);

  // Calculate the ratio of the sig_type by the arg_type, per the above check will be >= 1
  const auto sig_type_scale_gain_ratio =
      sig_type_relative_scale / arg_type_relative_scale;
  CHECK_GE(sig_type_scale_gain_ratio, 1);

  // Following the old bespoke scoring logic this function replaces, we heavily penalize
  // any casts that move ints to floats/doubles for the precision-loss reasons above
  // Arguably all integers in the tinyint and smallint can be fully specified with both
  // float and double types, but we treat them the same as int and bigint types here.
  const int32_t type_family_cast_penalty_score = is_integer_to_fp_cast ? 1001000 : 1000;

  int32_t scale_cast_penalty_score;

  // The following logic is new. Basically there are strong reasons to
  // prefer the promotion of constant literals to the most precise type possible, as
  // rather than the type being inherent in the data - that is a column or columns where
  // a user specified a type (and with any expressions on those columns following our
  // standard sql casting logic), literal types are given to us by Calcite and do not
  // necessarily convey any semantic intent (i.e. 10 will be an int, but 10.0 a decimal)
  // Hence it is better to promote these types to the most precise sig_type available,
  // while at the same time keeping column expressions as close as possible to the input
  // types (mainly for performance, we have many float versions of various functions
  // to allow for greater performance when the underlying data is not of double precision,
  // and hence there is little benefit of the extra cost of computing double precision
  // operators on this data)
  if (is_arg_literal) {
    scale_cast_penalty_score =
        (8000 / arg_type_relative_scale) - (1000 * sig_type_scale_gain_ratio);
  } else {
    scale_cast_penalty_score = (1000 * sig_type_scale_gain_ratio);
  }

  const auto cast_penalty_score =
      type_family_cast_penalty_score + scale_cast_penalty_score;
  CHECK_GT(cast_penalty_score, 0);
  penalty_score += cast_penalty_score;
  return 1;
}

namespace {

ExtArgumentType get_array_arg_elem_type(const ExtArgumentType ext_arg_array_type) {
  switch (ext_arg_array_type) {
    case ExtArgumentType::ArrayInt8:
      return ExtArgumentType::Int8;
    case ExtArgumentType::ArrayInt16:
      return ExtArgumentType::Int16;
    case ExtArgumentType::ArrayInt32:
      return ExtArgumentType::Int32;
    case ExtArgumentType::ArrayInt64:
      return ExtArgumentType::Int64;
    case ExtArgumentType::ArrayFloat:
      return ExtArgumentType::Float;
    case ExtArgumentType::ArrayDouble:
      return ExtArgumentType::Double;
    case ExtArgumentType::ArrayBool:
      return ExtArgumentType::Bool;
    default:
      UNREACHABLE();
  }
  return ExtArgumentType{};
}

}  // namespace

static int match_arguments(const hdk::ir::Type* arg_type,
                           const bool is_arg_literal,
                           int sig_pos,
                           const std::vector<ExtArgumentType>& sig_types,
                           int& penalty_score) {
  /*
    Returns non-negative integer `offset` if `arg_type` and
    `sig_types[sig_pos:sig_pos + offset]` match.

    The `offset` value can be interpreted as the number of extension
    function arguments that is consumed by the given `arg_type`. For
    instance, for scalar types the offset is always 1, for array
    types the offset is 2: one argument for array pointer value and
    one argument for the array size value, etc.

    Returns -1 when the types of an argument and the corresponding
    extension function argument(s) mismatch, or when downcasting would
    be effective.

    In case of non-negative `offset` result, the function updates
    penalty_score argument as follows:

      add 1000 if arg_type is non-scalar, otherwise:
      add 1000 * sizeof(sig_type) / sizeof(arg_type)
      add 1000000 if type kinds differ (integer vs double, for instance)

   */
  int max_pos = sig_types.size() - 1;
  if (sig_pos > max_pos) {
    return -1;
  }
  auto sig_type = sig_types[sig_pos];
  switch (arg_type->id()) {
    case hdk::ir::Type::kBoolean:
    case hdk::ir::Type::kInteger:
    case hdk::ir::Type::kFloatingPoint:
    case hdk::ir::Type::kDecimal:
      return match_numeric_argument(arg_type, is_arg_literal, sig_type, penalty_score);
    case hdk::ir::Type::kFixedLenArray:
    case hdk::ir::Type::kVarLenArray:
      if ((sig_type == ExtArgumentType::PInt8 || sig_type == ExtArgumentType::PInt16 ||
           sig_type == ExtArgumentType::PInt32 || sig_type == ExtArgumentType::PInt64 ||
           sig_type == ExtArgumentType::PFloat || sig_type == ExtArgumentType::PDouble ||
           sig_type == ExtArgumentType::PBool) &&
          sig_pos < max_pos && sig_types[sig_pos + 1] == ExtArgumentType::Int64) {
        penalty_score += 1000;
        return 2;
      } else if (is_ext_arg_type_array(sig_type)) {
        // array arguments must match exactly
        CHECK(arg_type->isArray());
        const auto sig_type_type =
            ext_arg_type_to_type(arg_type->ctx(), get_array_arg_elem_type(sig_type));
        auto elem_type = arg_type->as<hdk::ir::ArrayBaseType>()->elemType();
        if (elem_type->isBoolean() && sig_type_type->isInt8()) {
          /* Boolean array has the same low-level structure as Int8 array. */
          penalty_score += 1000;
          return 1;
        } else if (elem_type->id() == sig_type_type->id() &&
                   elem_type->size() == sig_type_type->size()) {
          penalty_score += 1000;
          return 1;
        } else {
          return -1;
        }
      }
      break;
    case hdk::ir::Type::kNull:  // NULL maps to a pointer and size argument
      if ((sig_type == ExtArgumentType::PInt8 || sig_type == ExtArgumentType::PInt16 ||
           sig_type == ExtArgumentType::PInt32 || sig_type == ExtArgumentType::PInt64 ||
           sig_type == ExtArgumentType::PFloat || sig_type == ExtArgumentType::PDouble ||
           sig_type == ExtArgumentType::PBool) &&
          sig_pos < max_pos && sig_types[sig_pos + 1] == ExtArgumentType::Int64) {
        penalty_score += 1000;
        return 2;
      }
      break;
    case hdk::ir::Type::kVarChar:
    case hdk::ir::Type::kText:
      if (sig_type != ExtArgumentType::TextEncodingNone) {
        return -1;
      }
      penalty_score += 1000;
      return 1;
      /* Not implemented types:
        kCHAR
        kTIME
        kTIMESTAMP
        kDATE
        kINTERVAL_DAY_TIME
        kINTERVAL_YEAR_MONTH
        kEVAL_CONTEXT_TYPE
        kVOID
        kCURSOR
     */
    default:
      throw std::runtime_error(std::string(__FILE__) + "#" + std::to_string(__LINE__) +
                               ": support for " + arg_type->toString() +
                               " not implemented: \n  pos=" + std::to_string(sig_pos) +
                               " max_pos=" + std::to_string(max_pos) + "\n  sig_types=(" +
                               ExtensionFunctionsWhitelist::toString(sig_types) + ")");
  }
  return -1;
}

bool is_valid_identifier(std::string str) {
  if (!str.size()) {
    return false;
  }

  if (!(std::isalpha(str[0]) || str[0] == '_')) {
    return false;
  }

  for (size_t i = 1; i < str.size(); i++) {
    if (!(std::isalnum(str[i]) || str[i] == '_')) {
      return false;
    }
  }

  return true;
}

}  // namespace

template <typename T>
std::tuple<T, std::vector<const hdk::ir::Type*>> bind_function(
    std::string name,
    hdk::ir::ExprPtrVector func_args,  // function args from sql query
    const std::vector<T>& ext_funcs,   // list of functions registered
    const std::string processor) {
  /* worker function

     Template type T must implement the following methods:

       std::vector<ExtArgumentType> getInputArgs()
   */
  /*
    Return extension function/table function that has the following
    properties

    1. each argument type in `arg_types` matches with extension
       function argument types.

       For scalar types, the matching means that the types are either
       equal or the argument type is smaller than the corresponding
       the extension function argument type. This ensures that no
       information is lost when casting of argument values is
       required.

       For array, the matching means that the argument
       type matches exactly with a group of extension function
       argument types. See `match_arguments`.

    2. has minimal penalty score among all implementations of the
       extension function with given `name`, see `get_penalty_score`
       for the definition of penalty score.

    It is assumed that function_oper and extension functions in
    ext_funcs have the same name.
   */
  if (!is_valid_identifier(name)) {
    throw NativeExecutionError("Cannot bind function with invalid UDF function name: " +
                               name);
  }

  int minimal_score = std::numeric_limits<int>::max();
  int index = -1;
  int optimal = -1;
  int optimal_variant = -1;

  std::vector<const hdk::ir::Type*> types_input;
  std::vector<bool> args_are_constants;
  for (auto atype : func_args) {
    types_input.push_back(atype->type());
    if (dynamic_cast<const hdk::ir::Constant*>(atype.get())) {
      args_are_constants.push_back(true);
    } else {
      args_are_constants.push_back(false);
    }
  }
  CHECK_EQ(types_input.size(), args_are_constants.size());

  if (types_input.size() == 0 && ext_funcs.size() > 0) {
    CHECK_EQ(ext_funcs.size(), static_cast<size_t>(1));
    CHECK_EQ(ext_funcs[0].getInputArgs().size(), static_cast<size_t>(0));
    std::vector<const hdk::ir::Type*> empty_type_info_variant(0);
    return {ext_funcs[0], empty_type_info_variant};
  }

  std::vector<std::vector<const hdk::ir::Type*>> types_variants;
  for (auto type : types_input) {
    if (types_variants.begin() == types_variants.end()) {
      types_variants.push_back({type});
      continue;
    }
    std::vector<std::vector<const hdk::ir::Type*>> new_types_variants;
    for (auto& types : types_variants) {
      types.push_back(type);
    }
    types_variants.insert(
        types_variants.end(), new_types_variants.begin(), new_types_variants.end());
  }

  // Find extension function that gives the best match on the set of
  // argument type variants:
  for (auto ext_func : ext_funcs) {
    index++;

    auto ext_func_args = ext_func.getInputArgs();
    int index_variant = -1;
    for (const auto& types : types_variants) {
      index_variant++;
      int penalty_score = 0;
      int pos = 0;
      int original_input_idx = 0;
      CHECK_LE(types.size(), args_are_constants.size());
      // for (size_t ti_idx = 0; ti_idx != type_infos.size(); ++ti_idx) {
      for (auto type : types) {
        int offset = match_arguments(type,
                                     args_are_constants[original_input_idx],
                                     pos,
                                     ext_func_args,
                                     penalty_score);
        if (offset < 0) {
          // atype does not match with ext_func argument
          pos = -1;
          break;
        }
        if (type->isColumnList()) {
          original_input_idx += type->as<hdk::ir::ColumnListType>()->length();
        } else {
          original_input_idx++;
        }
        pos += offset;
      }

      if ((size_t)pos == ext_func_args.size()) {
        CHECK_EQ(args_are_constants.size(), original_input_idx);
        // prefer smaller return types
        penalty_score +=
            ext_arg_type_to_type(hdk::ir::Context::defaultCtx(), ext_func.getRet())
                ->canonicalize()
                ->size();
        if (penalty_score < minimal_score) {
          optimal = index;
          minimal_score = penalty_score;
          optimal_variant = index_variant;
        }
      }
    }
  }

  if (optimal == -1) {
    /* no extension function found that argument types would match
       with types in `arg_types` */
    auto sarg_types = ExtensionFunctionsWhitelist::toString(types_input);
    std::string message;
    if (!ext_funcs.size()) {
      message = "Function " + name + "(" + sarg_types + ") not supported.";
      throw ExtensionFunctionBindingError(message);
    } else {
      if constexpr (std::is_same_v<T, ExtensionFunction>) {
        message = "Could not bind " + name + "(" + sarg_types + ") to any " + processor +
                  " UDF implementation.";
      } else {
        LOG(FATAL) << "bind_function: unknown extension function type "
                   << typeid(T).name();
      }
      message += "\n  Existing extension function implementations:";
    }
    throw ExtensionFunctionBindingError(message);
  }

  return {ext_funcs[optimal], types_variants[optimal_variant]};
}

ExtensionFunction bind_function(std::string name, hdk::ir::ExprPtrVector func_args) {
  // used in RelAlgTranslator.cpp, first try GPU UDFs, then fall back
  // to CPU UDFs.
  bool is_gpu = true;
  std::string processor = "GPU";
  auto ext_funcs = ExtensionFunctionsWhitelist::get_ext_funcs(name, is_gpu);
  if (!ext_funcs.size()) {
    is_gpu = false;
    processor = "CPU";
    ext_funcs = ExtensionFunctionsWhitelist::get_ext_funcs(name, is_gpu);
  }
  try {
    return std::get<0>(
        bind_function<ExtensionFunction>(name, func_args, ext_funcs, processor));
  } catch (ExtensionFunctionBindingError& e) {
    if (is_gpu) {
      is_gpu = false;
      processor = "GPU|CPU";
      ext_funcs = ExtensionFunctionsWhitelist::get_ext_funcs(name, is_gpu);
      return std::get<0>(
          bind_function<ExtensionFunction>(name, func_args, ext_funcs, processor));
    } else {
      throw;
    }
  }
}

ExtensionFunction bind_function(std::string name,
                                hdk::ir::ExprPtrVector func_args,
                                const bool is_gpu) {
  // used below
  std::vector<ExtensionFunction> ext_funcs =
      ExtensionFunctionsWhitelist::get_ext_funcs(name, is_gpu);
  std::string processor = (is_gpu ? "GPU" : "CPU");
  return std::get<0>(
      bind_function<ExtensionFunction>(name, func_args, ext_funcs, processor));
}

ExtensionFunction bind_function(const hdk::ir::FunctionOper* function_oper,
                                const bool is_gpu) {
  // used in ExtensionsIR.cpp
  auto name = function_oper->name();
  hdk::ir::ExprPtrVector func_args = {};
  for (size_t i = 0; i < function_oper->arity(); ++i) {
    func_args.push_back(function_oper->argShared(i));
  }
  return bind_function(name, func_args, is_gpu);
}
