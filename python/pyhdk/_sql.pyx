#
# Copyright 2022 Intel Corporation.
#
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport make_shared, make_unique
from libcpp.utility cimport move
from cython.operator cimport dereference, preincrement

from pyarrow.lib cimport pyarrow_wrap_table
from pyarrow.lib cimport CTable as CArrowTable

from pyhdk._common cimport CConfig, Config
from pyhdk._storage cimport SchemaProvider, CDataMgr, DataMgr
from pyhdk._execute cimport Executor, CExecutorDeviceType, CArrowResultSetConverter, CResultSet

cdef class Calcite:
  cdef shared_ptr[CalciteJNI] calcite

  def __cinit__(self, SchemaProvider schema_provider, Config config, **kwargs):
    cdef string udf_filename = kwargs.get("udf_filename", "")
    cdef size_t calcite_max_mem_mb = kwargs.get("calcite_max_mem_mb", 1024)

    self.calcite = make_shared[CalciteJNI](schema_provider.c_schema_provider, config.c_config, udf_filename, calcite_max_mem_mb)

    CExtensionFunctionsWhitelist.add(self.calcite.get().getExtensionFunctionWhitelist())
    if not udf_filename.empty():
      CExtensionFunctionsWhitelist.addUdfs(self.calcite.get().getUserDefinedFunctionWhitelist())

    CTableFunctionsFactory.init();
    cdef vector[CTableFunction] udtfs = move(CTableFunctionsFactory.get_table_funcs(False))
    cdef vector[CExtensionFunction] udfs = move(vector[CExtensionFunction]())
    self.calcite.get().setRuntimeExtensionFunctions(udfs, udtfs, False)

  def process(self, string sql, **kwargs):
    cdef string db_name = kwargs.get("db_name", "test-db")
    cdef vector[FilterPushDownInfo] filter_push_down_info = vector[FilterPushDownInfo]()
    cdef bool legacy_syntax = kwargs.get("legacy_syntax", False)
    cdef bool is_explain = kwargs.get("is_explain", False)
    cdef bool is_view_optimize = kwargs.get("is_view_optimize", False)
    return self.calcite.get().process(db_name, sql, filter_push_down_info, legacy_syntax, is_explain, is_view_optimize)

cdef class ExecutionResult:
  cdef CExecutionResult c_result
  # DataMgr has to outlive ResultSet objects to avoid use-after-free errors.
  # Currently, C++ library doesn't enforce this and user is responsible for
  # obects lifetime control. In Python we achieve it by holding DataMgr in
  # each ExecutionResult object.
  cdef shared_ptr[CDataMgr] c_data_mgr

  def row_count(self):
    cdef shared_ptr[CResultSet] c_res
    c_res = self.c_result.getRows()
    return int(c_res.get().rowCount())

  def to_arrow(self):
    cdef vector[string] col_names
    cdef vector[CTargetMetaInfo].const_iterator it = self.c_result.getTargetsMeta().const_begin()

    while it != self.c_result.getTargetsMeta().const_end():
      col_names.push_back(dereference(it).get_resname())
      preincrement(it)

    cdef unique_ptr[CArrowResultSetConverter] converter = make_unique[CArrowResultSetConverter](self.c_result.getRows(), col_names, -1)
    cdef shared_ptr[CArrowTable] at = converter.get().convertToArrowTable()
    return pyarrow_wrap_table(at)

  def to_explain_str(self):
    return self.c_result.getExplanation()

cdef class RelAlgExecutor:
  cdef shared_ptr[CRelAlgExecutor] c_rel_alg_executor
  # DataMgr is used only to pass it to each produced ExecutionResult
  cdef shared_ptr[CDataMgr] c_data_mgr

  def __cinit__(self, Executor executor, SchemaProvider schema_provider, DataMgr data_mgr, string ra_json):
    cdef CExecutor* c_executor = executor.c_executor.get()
    cdef CSchemaProviderPtr c_schema_provider = schema_provider.c_schema_provider
    cdef CDataProvider* c_data_provider = data_mgr.c_data_mgr.get().getDataProvider()
    cdef unique_ptr[CRelAlgDag] c_dag
    cdef int db_id = 0

    db_ids = schema_provider.listDatabases()
    assert len(db_ids) <= 1
    if len(db_ids) == 1:
      db_id = db_ids[0]

    c_dag.reset(new CRelAlgDagBuilder(ra_json, db_id, c_schema_provider, c_executor.getConfigPtr()))

    self.c_rel_alg_executor = make_shared[CRelAlgExecutor](c_executor, c_schema_provider, c_data_provider, move(c_dag))
    self.c_data_mgr = data_mgr.c_data_mgr

  def execute(self, **kwargs):
    cdef const CConfig *config = self.c_rel_alg_executor.get().getExecutor().getConfigPtr().get()
    cdef CCompilationOptions c_co
    if config.exec.enable_gpu_offloading:
      c_co = CCompilationOptions.defaults(CExecutorDeviceType.GPU)
    else:
      c_co = CCompilationOptions.defaults(CExecutorDeviceType.CPU)
    c_co.allow_lazy_fetch = kwargs.get("enable_lazy_fetch", config.rs.enable_lazy_fetch)
    c_co.with_dynamic_watchdog = kwargs.get("enable_dynamic_watchdog", config.exec.watchdog.enable_dynamic)
    cdef CExecutionOptions c_eo = CExecutionOptions.defaults()
    c_eo.output_columnar_hint = kwargs.get("enable_columnar_output", config.rs.enable_columnar_output)
    c_eo.with_watchdog = kwargs.get("enable_watchdog", config.exec.watchdog.enable)
    c_eo.with_dynamic_watchdog = kwargs.get("enable_dynamic_watchdog", config.exec.watchdog.enable_dynamic)
    c_eo.just_explain = kwargs.get("just_explain", False)
    c_eo.gpu_input_mem_limit_percent = 0.9 
    cdef CExecutionResult c_res = self.c_rel_alg_executor.get().executeRelAlgQuery(c_co, c_eo, False)
    cdef ExecutionResult res = ExecutionResult()
    res.c_result = move(c_res)
    res.c_data_mgr = self.c_data_mgr
    return res
