#include "ExpressionRange.h"
#include "ExtractFromTime.h"
#include "DateTruncate.h"
#include "GroupByAndAggregate.h"
#include "Execute.h"

#include <cfenv>

#define DEF_OPERATOR(fname, op)                                                                                    \
  ExpressionRange fname(const ExpressionRange& other) const {                                                      \
    return (type_ == ExpressionRangeType::Integer && other.type_ == ExpressionRangeType::Integer)                  \
               ? binOp<int64_t>(other,                                                                             \
                                [](const int64_t x, const int64_t y) { return int64_t(checked_int64_t(x) op y); }) \
               : binOp<double>(other, [](const double x, const double y) {                                         \
      std::feclearexcept(FE_OVERFLOW);                                                                             \
      std::feclearexcept(FE_UNDERFLOW);                                                                            \
      auto result = x op y;                                                                                        \
      if (std::fetestexcept(FE_OVERFLOW) || std::fetestexcept(FE_UNDERFLOW)) {                                     \
        throw std::runtime_error("overflow / underflow");                                                          \
      }                                                                                                            \
      return result;                                                                                               \
                 });                                                                                               \
  }

DEF_OPERATOR(ExpressionRange::operator+, +)
DEF_OPERATOR(ExpressionRange::operator-, -)
DEF_OPERATOR(ExpressionRange::operator*, *)

ExpressionRange ExpressionRange::operator/(const ExpressionRange& other) const {
  if (type_ != ExpressionRangeType::Integer || other.type_ != ExpressionRangeType::Integer) {
    return ExpressionRange::makeInvalidRange();
  }
  if (other.int_min_ * other.int_max_ <= 0) {
    // if the other interval contains 0, the rule is more complicated;
    // punt for now, we can revisit by splitting the other interval and
    // taking the convex hull of the resulting two intervals
    return ExpressionRange::makeInvalidRange();
  }
  return binOp<int64_t>(other, [](const int64_t x, const int64_t y) { return int64_t(checked_int64_t(x) / y); });
}

ExpressionRange ExpressionRange::operator||(const ExpressionRange& other) const {
  if (type_ != other.type_) {
    return ExpressionRange::makeInvalidRange();
  }
  ExpressionRange result;
  switch (type_) {
    case ExpressionRangeType::Invalid:
      return ExpressionRange::makeInvalidRange();
    case ExpressionRangeType::Integer: {
      result.type_ = ExpressionRangeType::Integer;
      result.has_nulls_ = has_nulls_ || other.has_nulls_;
      result.int_min_ = std::min(int_min_, other.int_min_);
      result.int_max_ = std::max(int_max_, other.int_max_);
      break;
    }
    case ExpressionRangeType::FloatingPoint: {
      result.type_ = ExpressionRangeType::FloatingPoint;
      result.has_nulls_ = has_nulls_ || other.has_nulls_;
      result.fp_min_ = std::min(fp_min_, other.fp_min_);
      result.fp_max_ = std::max(fp_max_, other.fp_max_);
      break;
    }
    default:
      CHECK(false);
  }
  return result;
}

bool ExpressionRange::operator==(const ExpressionRange& other) const {
  if (type_ != other.type_) {
    return false;
  }
  switch (type_) {
    case ExpressionRangeType::Invalid:
      return true;
    case ExpressionRangeType::Integer: {
      return has_nulls_ == other.has_nulls_ && int_min_ == other.int_min_ && int_max_ == other.int_max_;
    }
    case ExpressionRangeType::FloatingPoint: {
      return has_nulls_ == other.has_nulls_ && fp_min_ == other.fp_min_ && fp_max_ == other.fp_max_;
    }
    default:
      CHECK(false);
  }
  return false;
}

ExpressionRange getExpressionRange(const Analyzer::BinOper* expr,
                                   const std::vector<InputTableInfo>& query_infos,
                                   const Executor*);

ExpressionRange getExpressionRange(const Analyzer::Constant* expr);

ExpressionRange getExpressionRange(const Analyzer::ColumnVar* col_expr,
                                   const std::vector<InputTableInfo>& query_infos,
                                   const Executor* executor);

ExpressionRange getExpressionRange(const Analyzer::LikeExpr* like_expr);

ExpressionRange getExpressionRange(const Analyzer::CaseExpr* case_expr,
                                   const std::vector<InputTableInfo>& query_infos,
                                   const Executor*);

ExpressionRange getExpressionRange(const Analyzer::UOper* u_expr,
                                   const std::vector<InputTableInfo>& query_infos,
                                   const Executor*);

ExpressionRange getExpressionRange(const Analyzer::ExtractExpr* extract_expr,
                                   const std::vector<InputTableInfo>& query_infos,
                                   const Executor*);

ExpressionRange getExpressionRange(const Analyzer::DatetruncExpr* datetrunc_expr,
                                   const std::vector<InputTableInfo>& query_infos,
                                   const Executor* executor);

ExpressionRange getExpressionRange(const Analyzer::Expr* expr,
                                   const std::vector<InputTableInfo>& query_infos,
                                   const Executor* executor) {
  auto bin_oper_expr = dynamic_cast<const Analyzer::BinOper*>(expr);
  if (bin_oper_expr) {
    return getExpressionRange(bin_oper_expr, query_infos, executor);
  }
  auto constant_expr = dynamic_cast<const Analyzer::Constant*>(expr);
  if (constant_expr) {
    return getExpressionRange(constant_expr);
  }
  auto column_var_expr = dynamic_cast<const Analyzer::ColumnVar*>(expr);
  if (column_var_expr) {
    return getExpressionRange(column_var_expr, query_infos, executor);
  }
  auto like_expr = dynamic_cast<const Analyzer::LikeExpr*>(expr);
  if (like_expr) {
    return getExpressionRange(like_expr);
  }
  auto case_expr = dynamic_cast<const Analyzer::CaseExpr*>(expr);
  if (case_expr) {
    return getExpressionRange(case_expr, query_infos, executor);
  }
  auto u_expr = dynamic_cast<const Analyzer::UOper*>(expr);
  if (u_expr) {
    return getExpressionRange(u_expr, query_infos, executor);
  }
  auto extract_expr = dynamic_cast<const Analyzer::ExtractExpr*>(expr);
  if (extract_expr) {
    return getExpressionRange(extract_expr, query_infos, executor);
  }
  auto datetrunc_expr = dynamic_cast<const Analyzer::DatetruncExpr*>(expr);
  if (datetrunc_expr) {
    return getExpressionRange(datetrunc_expr, query_infos, executor);
  }
  return ExpressionRange::makeInvalidRange();
}

namespace {

int64_t scale_down_interval_endpoint(const int64_t endpoint, const SQLTypeInfo& ti) {
  return endpoint / static_cast<int64_t>(exp_to_scale(ti.get_scale()));
}

int64_t scale_up_interval_endpoint(const int64_t endpoint, const SQLTypeInfo& ti) {
  return endpoint * static_cast<int64_t>(exp_to_scale(ti.get_scale()));
}

}  // namespace

ExpressionRange getExpressionRange(const Analyzer::BinOper* expr,
                                   const std::vector<InputTableInfo>& query_infos,
                                   const Executor* executor) {
  const auto& lhs = getExpressionRange(expr->get_left_operand(), query_infos, executor);
  const auto& rhs = getExpressionRange(expr->get_right_operand(), query_infos, executor);
  switch (expr->get_optype()) {
    case kPLUS:
      return lhs + rhs;
    case kMINUS:
      return lhs - rhs;
    case kMULTIPLY: {
      const auto& lhs_type = expr->get_left_operand()->get_type_info();
      auto er = lhs * rhs;
      if (lhs_type.is_decimal() && er.getType() != ExpressionRangeType::Invalid) {
        CHECK(er.getType() == ExpressionRangeType::Integer);
        CHECK_EQ(int64_t(0), er.getBucket());
        return ExpressionRange::makeIntRange(scale_down_interval_endpoint(er.getIntMin(), lhs_type),
                                             scale_down_interval_endpoint(er.getIntMax(), lhs_type),
                                             0,
                                             er.hasNulls());
      }
      return er;
    }
    case kDIVIDE: {
      const auto& lhs_type = expr->get_left_operand()->get_type_info();
      if (lhs_type.is_decimal() && lhs.getType() != ExpressionRangeType::Invalid) {
        CHECK(lhs.getType() == ExpressionRangeType::Integer);
        const auto adjusted_lhs = ExpressionRange::makeIntRange(scale_up_interval_endpoint(lhs.getIntMin(), lhs_type),
                                                                scale_up_interval_endpoint(lhs.getIntMax(), lhs_type),
                                                                0,
                                                                lhs.hasNulls());
        return adjusted_lhs / rhs;
      }
      return lhs / rhs;
    }
    default:
      break;
  }
  return ExpressionRange::makeInvalidRange();
}

ExpressionRange getExpressionRange(const Analyzer::Constant* constant_expr) {
  if (constant_expr->get_is_null()) {
    return ExpressionRange::makeInvalidRange();
  }
  const auto constant_type = constant_expr->get_type_info().get_type();
  const auto datum = constant_expr->get_constval();
  switch (constant_type) {
    case kSMALLINT: {
      const int64_t v = datum.smallintval;
      return ExpressionRange::makeIntRange(v, v, 0, false);
    }
    case kINT: {
      const int64_t v = datum.intval;
      return ExpressionRange::makeIntRange(v, v, 0, false);
    }
    case kBIGINT:
    case kNUMERIC:
    case kDECIMAL: {
      const int64_t v = datum.bigintval;
      return ExpressionRange::makeIntRange(v, v, 0, false);
    }
    case kTIME:
    case kTIMESTAMP:
    case kDATE: {
      const int64_t v = datum.timeval;
      return ExpressionRange::makeIntRange(v, v, 0, false);
    }
    case kFLOAT:
    case kDOUBLE: {
      const double v = constant_type == kDOUBLE ? datum.doubleval : datum.floatval;
      return ExpressionRange::makeFpRange(v, v, false);
    }
    default:
      break;
  }
  return ExpressionRange::makeInvalidRange();
}

#define FIND_STAT_FRAG(stat_name)                                                                            \
  const auto stat_name##_frag =                                                                              \
      std::stat_name##_element(fragments.begin(),                                                            \
                               fragments.end(),                                                              \
                               [&has_nulls, col_id, col_ti](const Fragmenter_Namespace::FragmentInfo& lhs,   \
                                                            const Fragmenter_Namespace::FragmentInfo& rhs) { \
    auto lhs_meta_it = lhs.chunkMetadataMap.find(col_id);                                                    \
    if (lhs_meta_it == lhs.chunkMetadataMap.end()) {                                                         \
      return false;                                                                                          \
    }                                                                                                        \
    auto rhs_meta_it = rhs.chunkMetadataMap.find(col_id);                                                    \
    CHECK(rhs_meta_it != rhs.chunkMetadataMap.end());                                                        \
    if (lhs_meta_it->second.chunkStats.has_nulls || rhs_meta_it->second.chunkStats.has_nulls) {              \
      has_nulls = true;                                                                                      \
    }                                                                                                        \
    if (col_ti.is_fp()) {                                                                                    \
      return extract_##stat_name##_stat_double(lhs_meta_it->second.chunkStats, col_ti) <                     \
             extract_##stat_name##_stat_double(rhs_meta_it->second.chunkStats, col_ti);                      \
    }                                                                                                        \
    return extract_##stat_name##_stat(lhs_meta_it->second.chunkStats, col_ti) <                              \
           extract_##stat_name##_stat(rhs_meta_it->second.chunkStats, col_ti);                               \
      });                                                                                                    \
  if (stat_name##_frag == fragments.end()) {                                                                 \
    return ExpressionRange::makeInvalidRange();                                                              \
  }

namespace {

double extract_min_stat_double(const ChunkStats& stats, const SQLTypeInfo& col_ti) {
  return col_ti.get_type() == kDOUBLE ? stats.min.doubleval : stats.min.floatval;
}

double extract_max_stat_double(const ChunkStats& stats, const SQLTypeInfo& col_ti) {
  return col_ti.get_type() == kDOUBLE ? stats.max.doubleval : stats.max.floatval;
}

int64_t get_conservative_datetrunc_bucket(const DatetruncField datetrunc_field) {
  const int64_t day_seconds{24 * 3600};
  const int64_t year_days{365};
  switch (datetrunc_field) {
    case dtYEAR:
      return year_days * day_seconds;
    case dtQUARTER:
      return 90 * day_seconds;  // 90 is least number of days in any quater
    case dtMONTH:
      return 28 * day_seconds;
    case dtDAY:
      return day_seconds;
    case dtHOUR:
      return 3600;
    case dtMINUTE:
      return 60;
    case dtMILLENNIUM:
      return 1000 * year_days * day_seconds;
    case dtCENTURY:
      return 100 * year_days * day_seconds;
    case dtDECADE:
      return 10 * year_days * day_seconds;
    case dtWEEK:
      return 7 * day_seconds;
    case dtQUARTERDAY:
      return 4 * 60 * 50;
    default:
      return 0;
  }
}

}  // namespace

ExpressionRange getExpressionRange(const Analyzer::ColumnVar* col_expr,
                                   const std::vector<InputTableInfo>& query_infos,
                                   const Executor* executor) {
  int col_id = col_expr->get_column_id();
  const auto& col_ti =
      col_expr->get_type_info().is_array() ? col_expr->get_type_info().get_elem_type() : col_expr->get_type_info();
  switch (col_ti.get_type()) {
    case kTEXT:
    case kCHAR:
    case kVARCHAR:
      CHECK_EQ(kENCODING_DICT, col_ti.get_compression());
    case kBOOLEAN:
    case kSMALLINT:
    case kINT:
    case kBIGINT:
    case kDECIMAL:
    case kNUMERIC:
    case kDATE:
    case kTIMESTAMP:
    case kTIME:
    case kFLOAT:
    case kDOUBLE: {
      const int rte_idx = col_expr->get_rte_idx();
      CHECK_GE(rte_idx, 0);
      CHECK_LT(static_cast<size_t>(rte_idx), query_infos.size());
      ssize_t ti_idx = -1;
      for (size_t i = 0; i < query_infos.size(); ++i) {
        if (col_expr->get_table_id() == query_infos[i].table_id) {
          ti_idx = i;
          break;
        }
      }
      CHECK_NE(ssize_t(-1), ti_idx);
      const auto& query_info = query_infos[ti_idx].info;
      const auto& fragments = query_info.fragments;
      bool has_nulls = rte_idx > 0 && executor->isOuterJoin();
      const auto cd = executor->getColumnDescriptor(col_expr);
      if (cd && cd->isVirtualCol) {
        CHECK(cd->columnName == "rowid");
        CHECK_EQ(kBIGINT, col_ti.get_type());
        const int64_t num_tuples = query_info.numTuples;
        return ExpressionRange::makeIntRange(0, std::max(num_tuples - 1, int64_t(0)), 0, has_nulls);
      }
      if (query_info.numTuples == 0 && !col_ti.is_fp()) {
        // The column doesn't contain any values, synthesize an empty range.
        return ExpressionRange::makeIntRange(0, -1, 0, false);
      }
      FIND_STAT_FRAG(min);
      FIND_STAT_FRAG(max);
      const auto min_it = min_frag->chunkMetadataMap.find(col_id);
      if (min_it == min_frag->chunkMetadataMap.end()) {
        return ExpressionRange::makeInvalidRange();
      }
      const auto max_it = max_frag->chunkMetadataMap.find(col_id);
      CHECK(max_it != max_frag->chunkMetadataMap.end());
      for (const auto& fragment : fragments) {
        const auto it = fragment.chunkMetadataMap.find(col_id);
        if (it != fragment.chunkMetadataMap.end()) {
          if (it->second.chunkStats.has_nulls) {
            has_nulls = true;
          }
        }
      }
      if (col_ti.is_fp()) {
        return ExpressionRange::makeFpRange(extract_min_stat_double(min_it->second.chunkStats, col_ti),
                                            extract_max_stat_double(max_it->second.chunkStats, col_ti),
                                            has_nulls);
      }
      const auto min_val = extract_min_stat(min_it->second.chunkStats, col_ti);
      const auto max_val = extract_max_stat(max_it->second.chunkStats, col_ti);
      if (max_val < min_val) {
        // The column doesn't contain any non-null values, synthesize an empty range.
        CHECK_LT(max_val, 0);
        CHECK_GT(min_val, 0);
        CHECK_EQ(-(min_val + 1), max_val);
        return ExpressionRange::makeIntRange(0, -1, 0, has_nulls);
      }
      const int64_t bucket = col_ti.get_type() == kDATE ? get_conservative_datetrunc_bucket(dtDAY) : 0;
      return ExpressionRange::makeIntRange(min_val, max_val, bucket, has_nulls);
    }
    default:
      break;
  }
  return ExpressionRange::makeInvalidRange();
}

#undef FIND_STAT_FRAG

ExpressionRange getExpressionRange(const Analyzer::LikeExpr* like_expr) {
  const auto& ti = like_expr->get_type_info();
  CHECK(ti.is_boolean());
  const auto& arg_ti = like_expr->get_arg()->get_type_info();
  return ExpressionRange::makeIntRange(arg_ti.get_notnull() ? 0 : inline_int_null_val(ti), 1, 0, false);
}

ExpressionRange getExpressionRange(const Analyzer::CaseExpr* case_expr,
                                   const std::vector<InputTableInfo>& query_infos,
                                   const Executor* executor) {
  const auto& expr_pair_list = case_expr->get_expr_pair_list();
  auto expr_range = ExpressionRange::makeInvalidRange();
  for (const auto& expr_pair : expr_pair_list) {
    CHECK_EQ(expr_pair.first->get_type_info().get_type(), kBOOLEAN);
    const auto crt_range = getExpressionRange(expr_pair.second.get(), query_infos, executor);
    if (crt_range.getType() == ExpressionRangeType::Invalid) {
      return ExpressionRange::makeInvalidRange();
    }
    expr_range = (expr_range.getType() != ExpressionRangeType::Invalid) ? expr_range || crt_range : crt_range;
  }
  const auto else_expr = case_expr->get_else_expr();
  CHECK(else_expr);
  const auto else_null_expr = dynamic_cast<const Analyzer::Constant*>(else_expr);
  if (else_null_expr && else_null_expr->get_is_null()) {
    expr_range.setHasNulls();
    return expr_range;
  }
  return expr_range || getExpressionRange(else_expr, query_infos, executor);
}

ExpressionRange getExpressionRange(const Analyzer::UOper* u_expr,
                                   const std::vector<InputTableInfo>& query_infos,
                                   const Executor* executor) {
  if (u_expr->get_optype() == kUNNEST) {
    return getExpressionRange(u_expr->get_operand(), query_infos, executor);
  }
  if (u_expr->get_optype() != kCAST) {
    return ExpressionRange::makeInvalidRange();
  }
  const auto& ti = u_expr->get_type_info();
  if (ti.is_string() && ti.get_compression() == kENCODING_DICT) {
    const auto sd = executor->getStringDictionary(ti.get_comp_param(), nullptr);
    CHECK(sd);
    const auto const_operand = dynamic_cast<const Analyzer::Constant*>(u_expr->get_operand());
    CHECK(const_operand);
    CHECK(const_operand->get_constval().stringval);
    const int64_t v = sd->get(*const_operand->get_constval().stringval);
    return ExpressionRange::makeIntRange(v, v, 0, false);
  }
  const auto arg_range = getExpressionRange(u_expr->get_operand(), query_infos, executor);
  const auto& arg_ti = u_expr->get_operand()->get_type_info();
  // Timestamp to date cast is generated like a date trunc on day in the executor.
  if (arg_ti.get_type() == kTIMESTAMP && ti.get_type() == kDATE) {
    const auto dt_expr = makeExpr<Analyzer::DatetruncExpr>(arg_ti, false, dtDAY, u_expr->get_own_operand());
    return getExpressionRange(dt_expr.get(), query_infos, executor);
  }
  switch (arg_range.getType()) {
    case ExpressionRangeType::FloatingPoint: {
      if (ti.is_fp() && ti.get_logical_size() >= arg_ti.get_logical_size()) {
        return arg_range;
      }
      if (ti.is_integer()) {
        return ExpressionRange::makeIntRange(arg_range.getFpMin(), arg_range.getFpMax(), 0, arg_range.hasNulls());
      }
      break;
    }
    case ExpressionRangeType::Integer: {
      if (ti.is_decimal()) {
        CHECK_EQ(int64_t(0), arg_range.getBucket());
        const int64_t scale = exp_to_scale(ti.get_scale() - arg_ti.get_scale());
        return ExpressionRange::makeIntRange(
            arg_range.getIntMin() * scale, arg_range.getIntMax() * scale, 0, arg_range.hasNulls());
      }
      if (arg_ti.is_decimal()) {
        CHECK_EQ(int64_t(0), arg_range.getBucket());
        const int64_t scale = exp_to_scale(arg_ti.get_scale());
        return ExpressionRange::makeIntRange(
            arg_range.getIntMin() / scale, arg_range.getIntMax() / scale, 0, arg_range.hasNulls());
      }
      if (ti.is_integer() || ti.is_time()) {
        return arg_range;
      }
      if (ti.get_type() == kDOUBLE) {
        return ExpressionRange::makeFpRange(arg_range.getIntMin(), arg_range.getIntMax(), arg_range.hasNulls());
      }
      break;
    }
    case ExpressionRangeType::Invalid:
      break;
    default:
      CHECK(false);
  }
  return ExpressionRange::makeInvalidRange();
}

ExpressionRange getExpressionRange(const Analyzer::ExtractExpr* extract_expr,
                                   const std::vector<InputTableInfo>& query_infos,
                                   const Executor* executor) {
  const int32_t extract_field{extract_expr->get_field()};
  const auto arg_range = getExpressionRange(extract_expr->get_from_expr(), query_infos, executor);
  const bool has_nulls = arg_range.getType() == ExpressionRangeType::Invalid || arg_range.hasNulls();
  switch (extract_field) {
    case kYEAR: {
      if (arg_range.getType() == ExpressionRangeType::Invalid) {
        return ExpressionRange::makeInvalidRange();
      }
      CHECK(arg_range.getType() == ExpressionRangeType::Integer);
      const int64_t year_range_min = ExtractFromTime(kYEAR, arg_range.getIntMin());
      const int64_t year_range_max = ExtractFromTime(kYEAR, arg_range.getIntMax());
      return ExpressionRange::makeIntRange(year_range_min, year_range_max, 0, arg_range.hasNulls());
    }
    case kEPOCH:
      return arg_range;
    case kQUARTERDAY:
    case kQUARTER:
      return ExpressionRange::makeIntRange(1, 4, 0, has_nulls);
    case kMONTH:
      return ExpressionRange::makeIntRange(1, 12, 0, has_nulls);
    case kDAY:
      return ExpressionRange::makeIntRange(1, 31, 0, has_nulls);
    case kHOUR:
      return ExpressionRange::makeIntRange(0, 23, 0, has_nulls);
    case kMINUTE:
      return ExpressionRange::makeIntRange(0, 59, 0, has_nulls);
    case kSECOND:
      return ExpressionRange::makeIntRange(0, 60, 0, has_nulls);
    case kDOW:
      return ExpressionRange::makeIntRange(0, 6, 0, has_nulls);
    case kISODOW:
      return ExpressionRange::makeIntRange(1, 7, 0, has_nulls);
    case kDOY:
      return ExpressionRange::makeIntRange(1, 366, 0, has_nulls);
    case kWEEK:
      return ExpressionRange::makeIntRange(1, 53, 0, has_nulls);
    default:
      CHECK(false);
  }
  return ExpressionRange::makeInvalidRange();
}

ExpressionRange getExpressionRange(const Analyzer::DatetruncExpr* datetrunc_expr,
                                   const std::vector<InputTableInfo>& query_infos,
                                   const Executor* executor) {
  const auto arg_range = getExpressionRange(datetrunc_expr->get_from_expr(), query_infos, executor);
  if (arg_range.getType() == ExpressionRangeType::Invalid) {
    return ExpressionRange::makeInvalidRange();
  }
  const int64_t min_ts = DateTruncate(datetrunc_expr->get_field(), arg_range.getIntMin());
  const int64_t max_ts = DateTruncate(datetrunc_expr->get_field(), arg_range.getIntMax());
  const int64_t bucket = get_conservative_datetrunc_bucket(datetrunc_expr->get_field());
  return ExpressionRange::makeIntRange(min_ts, max_ts, bucket, arg_range.hasNulls());
}
