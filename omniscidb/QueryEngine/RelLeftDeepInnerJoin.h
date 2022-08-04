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

#pragma once

#include <memory>
#include <vector>

#include "IR/Expr.h"

class RelAlgNode;
class RelLeftDeepInnerJoin;
class RexScalar;

// Gets the start node of a left-deep pattern starting at the node itself or its child.
std::shared_ptr<const RelAlgNode> get_left_deep_join_root(
    const std::shared_ptr<RelAlgNode>& node);

void create_left_deep_join(std::vector<std::shared_ptr<RelAlgNode>>& nodes);

void rebind_inputs_from_left_deep_join(const RexScalar* rex,
                                       const RelLeftDeepInnerJoin* left_deep_join);
hdk::ir::ExprPtr rebind_inputs_from_left_deep_join(
    const hdk::ir::Expr* expr,
    const RelLeftDeepInnerJoin* left_deep_join);
