/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.mapd.calcite.parser;

import com.mapd.metadata.MetaConnect;

import org.apache.calcite.linq4j.tree.Expression;
import org.apache.calcite.rel.type.RelProtoDataType;
import org.apache.calcite.schema.Function;
import org.apache.calcite.schema.Schema;
import org.apache.calcite.schema.SchemaPlus;
import org.apache.calcite.schema.SchemaVersion;
import org.apache.calcite.schema.Table;
import org.apache.calcite.util.ConversionUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

/**
 *
 * @author michael
 */
public class MapDSchema implements Schema {
  final static Logger MAPDLOGGER = LoggerFactory.getLogger(MapDSchema.class);

  final private MetaConnect metaConnect;
  public MapDSchema(MapDParser mp, MapDUser mapdUser, String db, String schemaJson) {
    System.setProperty(
            "saffron.default.charset", ConversionUtil.NATIVE_UTF16_CHARSET_NAME);
    System.setProperty(
            "saffron.default.nationalcharset", ConversionUtil.NATIVE_UTF16_CHARSET_NAME);
    System.setProperty("saffron.default.collation.name",
            ConversionUtil.NATIVE_UTF16_CHARSET_NAME + "$en_US");
    metaConnect = new MetaConnect(mapdUser, mp, db, schemaJson);
  }

  public MapDSchema(MapDParser mp, MapDUser mapdUser, String db) {
    this(mp, mapdUser, db, null);
  }

  public MapDSchema(MapDParser mp, MapDUser mapdUser) {
    this(mp, mapdUser, null, null);
  }

  @Override
  public Table getTable(String string) {
    Table table = metaConnect.getTable(string);
    return table;
  }

  @Override
  public Set<String> getTableNames() {
    Set<String> tableSet = metaConnect.getTables();
    return tableSet;
  }

  @Override
  public Collection<Function> getFunctions(String string) {
    Collection<Function> functionCollection = new HashSet<Function>();
    return functionCollection;
  }

  @Override
  public Set<String> getFunctionNames() {
    Set<String> functionSet = new HashSet<String>();
    return functionSet;
  }

  @Override
  public Schema getSubSchema(String string) {
    return null;
  }

  @Override
  public Set<String> getSubSchemaNames() {
    Set<String> hs = new HashSet<String>();
    return hs;
  }

  @Override
  public Expression getExpression(SchemaPlus sp, String string) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public boolean isMutable() {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public Schema snapshot(SchemaVersion sv) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public RelProtoDataType getType(String arg0) {
    throw new UnsupportedOperationException("Not supported yet.");
  }

  @Override
  public Set<String> getTypeNames() {
    throw new UnsupportedOperationException("Not supported yet.");
  }
}
