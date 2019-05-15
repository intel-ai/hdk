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

package org.apache.calcite.rel.externalize;

import com.google.common.collect.ImmutableList;

import org.apache.calcite.adapter.enumerable.EnumerableTableScan;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelWriter;
import org.apache.calcite.rel.logical.LogicalAggregate;
import org.apache.calcite.rel.logical.LogicalTableModify;
import org.apache.calcite.rel.logical.LogicalTableScan;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.sql.SqlExplainLevel;
import org.apache.calcite.util.JsonBuilder;
import org.apache.calcite.util.Pair;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/**
 * Callback for a relational expression to dump itself as JSON.
 *
 * @see RelJsonReader
 */
public class MapDRelJsonWriter implements RelWriter {
  //~ Instance fields ----------------------------------------------------------

  private final JsonBuilder jsonBuilder;
  private final MapDRelJson relJson;
  private final Map<RelNode, String> relIdMap = new IdentityHashMap<RelNode, String>();
  private final List<Object> relList;
  private final List<Pair<String, Object>> values = new ArrayList<Pair<String, Object>>();
  private String previousId;

  //~ Constructors -------------------------------------------------------------

  public MapDRelJsonWriter() {
    jsonBuilder = new JsonBuilder();
    relList = jsonBuilder.list();
    relJson = new MapDRelJson(jsonBuilder);
  }

  //~ Methods ------------------------------------------------------------------

  protected void explain_(RelNode rel, List<Pair<String, Object>> values) {
    final Map<String, Object> map = jsonBuilder.map();

    map.put("id", null); // ensure that id is the first attribute
    map.put("relOp", relJson.classToTypeName(rel.getClass()));
    if (rel instanceof EnumerableTableScan) {
      RelDataType row_type = ((EnumerableTableScan) rel).getTable().getRowType();
      List<String> field_names = row_type.getFieldNames();
      map.put("fieldNames", field_names);
    }
    if (rel instanceof LogicalAggregate) {
      map.put("fields", rel.getRowType().getFieldNames());
    }
    if (rel instanceof LogicalTableModify) {
      // FIX-ME:  What goes here?
    }

    for (Pair<String, Object> value : values) {
      if (value.right instanceof RelNode) {
        continue;
      }
      put(map, value.left, value.right);
    }
    // omit 'inputs: ["3"]' if "3" is the preceding rel
    final List<Object> list = explainInputs(rel.getInputs());
    if (list.size() != 1 || !list.get(0).equals(previousId)) {
      map.put("inputs", list);
    }

    final String id = Integer.toString(relIdMap.size());
    relIdMap.put(rel, id);
    map.put("id", id);

    relList.add(map);
    previousId = id;
  }

  private void put(Map<String, Object> map, String name, Object value) {
    map.put(name, relJson.toJson(value));
  }

  private List<Object> explainInputs(List<RelNode> inputs) {
    final List<Object> list = jsonBuilder.list();
    for (RelNode input : inputs) {
      String id = relIdMap.get(input);
      if (id == null) {
        input.explain(this);
        id = previousId;
      }
      list.add(id);
    }
    return list;
  }

  public final void explain(RelNode rel, List<Pair<String, Object>> valueList) {
    explain_(rel, valueList);
  }

  public SqlExplainLevel getDetailLevel() {
    return SqlExplainLevel.ALL_ATTRIBUTES;
  }

  public RelWriter input(String term, RelNode input) {
    return this;
  }

  public RelWriter item(String term, Object value) {
    values.add(Pair.of(term, value));
    return this;
  }

  private List<Object> getList(List<Pair<String, Object>> values, String tag) {
    for (Pair<String, Object> value : values) {
      if (value.left.equals(tag)) {
        // noinspection unchecked
        return (List<Object>) value.right;
      }
    }
    final List<Object> list = new ArrayList<Object>();
    values.add(Pair.of(tag, (Object) list));
    return list;
  }

  public RelWriter itemIf(String term, Object value, boolean condition) {
    if (condition) {
      item(term, value);
    }
    return this;
  }

  public RelWriter done(RelNode node) {
    final List<Pair<String, Object>> valuesCopy = ImmutableList.copyOf(values);
    values.clear();
    explain_(node, valuesCopy);
    return this;
  }

  public boolean nest() {
    return true;
  }

  /**
   * Returns a JSON string describing the relational expressions that were just
   * explained.
   */
  public String asString() {
    return jsonBuilder.toJsonString(asJsonMap());
  }

  public Map<String, Object> asJsonMap() {
    final Map<String, Object> map = jsonBuilder.map();
    map.put("rels", relList);
    return map;
  }
}

// End RelJsonWriter.java
