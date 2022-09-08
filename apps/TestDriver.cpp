/**
 * Copyright (C) 2022 Intel Corporation
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "HDK.h"

#include <iostream>
#include <memory>

int main(void) {
  int table_id = 0;
  SQLTypeInfo sql_type(SQLTypes::kINT);

  const auto col_expr =
      hdk::ir::makeExpr<hdk::ir::ColumnVar>(sql_type, table_id, /*column_id=*/0, 0);
  std::cout << "Test program worked: " << col_expr->toString() << std::endl;
}
