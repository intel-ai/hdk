/*
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

#include "Calcite/CalciteJNI.h"
#include "DataMgr/DataMgrBufferProvider.h"
#include "DataMgr/DataMgrDataProvider.h"
#include "QueryEngine/RelAlgExecutor.h"
#include "ResultSet/ArrowResultSet.h"
#include "SchemaMgr/SimpleSchemaProvider.h"
#include "Shared/scope.h"

#include "ArrowTestHelpers.h"
#include "TestDataProvider.h"
#include "TestHelpers.h"
#include "TestRelAlgDagBuilder.h"

#include <gtest/gtest.h>
#include <boost/program_options.hpp>

constexpr int TEST_SCHEMA_ID = 1;
constexpr int TEST_DB_ID = (TEST_SCHEMA_ID << 24) + 1;
constexpr int TEST1_TABLE_ID = 1;
constexpr int TEST2_TABLE_ID = 2;
constexpr int TEST_AGG_TABLE_ID = 3;

static bool use_groupby_buffer_desc = false;

using ArrowTestHelpers::compare_res_data;

class TestSchemaProvider : public SimpleSchemaProvider {
 public:
  TestSchemaProvider()
      : SimpleSchemaProvider(hdk::ir::Context::defaultCtx(), TEST_SCHEMA_ID, "test") {
    // Table test1
    addTableInfo(TEST_DB_ID, TEST1_TABLE_ID, "test1", false, 1, 5);
    addColumnInfo(TEST_DB_ID, TEST1_TABLE_ID, 1, "col_bi", ctx_.int64(), false);
    addColumnInfo(TEST_DB_ID, TEST1_TABLE_ID, 2, "col_i", ctx_.int32(), false);
    addColumnInfo(TEST_DB_ID, TEST1_TABLE_ID, 3, "col_f", ctx_.fp32(), false);
    addColumnInfo(TEST_DB_ID, TEST1_TABLE_ID, 4, "col_d", ctx_.fp64(), false);
    addRowidColumn(TEST_DB_ID, TEST1_TABLE_ID, 5);

    // Table test2
    addTableInfo(TEST_DB_ID, TEST2_TABLE_ID, "test2", false, 3, 9);
    addColumnInfo(TEST_DB_ID, TEST2_TABLE_ID, 1, "col_bi", ctx_.int64(), false);
    addColumnInfo(TEST_DB_ID, TEST2_TABLE_ID, 2, "col_i", ctx_.int32(), false);
    addColumnInfo(TEST_DB_ID, TEST2_TABLE_ID, 3, "col_f", ctx_.fp32(), false);
    addColumnInfo(TEST_DB_ID, TEST2_TABLE_ID, 4, "col_d", ctx_.fp64(), false);
    addRowidColumn(TEST_DB_ID, TEST2_TABLE_ID, 5);

    // Table test_agg
    addTableInfo(TEST_DB_ID, TEST_AGG_TABLE_ID, "test_agg", false, 2, 10);
    addColumnInfo(TEST_DB_ID, TEST_AGG_TABLE_ID, 1, "id", ctx_.int32(), false);
    addColumnInfo(TEST_DB_ID, TEST_AGG_TABLE_ID, 2, "val", ctx_.int32(), false);
    addRowidColumn(TEST_DB_ID, TEST_AGG_TABLE_ID, 3);
  }

  ~TestSchemaProvider() override = default;
};

class TestDataProvider : public TestHelpers::TestDataProvider {
 public:
  TestDataProvider(SchemaProviderPtr schema_provider)
      : TestHelpers::TestDataProvider(TEST_DB_ID, schema_provider) {
    TestHelpers::TestTableData test1(TEST_DB_ID, TEST1_TABLE_ID, 4, schema_provider_);
    test1.addColFragment<int64_t>(1, {1, 2, 3, 4, 5});
    test1.addColFragment<int32_t>(2, {10, 20, 30, 40, 50});
    test1.addColFragment<float>(3, {1.1, 2.2, 3.3, 4.4, 5.5});
    test1.addColFragment<double>(4, {10.1, 20.2, 30.3, 40.4, 50.5});
    tables_.emplace(std::make_pair(TEST1_TABLE_ID, test1));

    TestHelpers::TestTableData test2(TEST_DB_ID, TEST2_TABLE_ID, 4, schema_provider_);
    test2.addColFragment<int64_t>(1, {1, 2, 3});
    test2.addColFragment<int64_t>(1, {4, 5, 6});
    test2.addColFragment<int64_t>(1, {7, 8, 9});
    test2.addColFragment<int32_t>(2, {110, 120, 130});
    test2.addColFragment<int32_t>(2, {140, 150, 160});
    test2.addColFragment<int32_t>(2, {170, 180, 190});
    test2.addColFragment<float>(3, {101.1, 102.2, 103.3});
    test2.addColFragment<float>(3, {104.4, 105.5, 106.6});
    test2.addColFragment<float>(3, {107.7, 108.8, 109.9});
    test2.addColFragment<double>(4, {110.1, 120.2, 130.3});
    test2.addColFragment<double>(4, {140.4, 150.5, 160.6});
    test2.addColFragment<double>(4, {170.7, 180.8, 190.9});
    tables_.emplace(std::make_pair(TEST2_TABLE_ID, test2));

    TestHelpers::TestTableData test_agg(
        TEST_DB_ID, TEST_AGG_TABLE_ID, 2, schema_provider_);
    test_agg.addColFragment<int32_t>(1, {1, 2, 1, 2, 1});
    test_agg.addColFragment<int32_t>(1, {2, 1, 3, 1, 3});
    test_agg.addColFragment<int32_t>(2, {10, 20, 30, 40, 50});
    test_agg.addColFragment<int32_t>(
        2, {inline_null_value<int32_t>(), 70, inline_null_value<int32_t>(), 90, 100});
    tables_.emplace(std::make_pair(TEST_AGG_TABLE_ID, test_agg));
  }

  ~TestDataProvider() override = default;
};

class NoCatalogSqlTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    config_ = std::make_shared<Config>();
    config_->exec.cpu_only = true;

    schema_provider_ = std::make_shared<TestSchemaProvider>();

    data_mgr_ = std::make_shared<DataMgr>(*config_, false);

    auto* ps_mgr = data_mgr_->getPersistentStorageMgr();
    ps_mgr->registerDataProvider(TEST_SCHEMA_ID,
                                 std::make_shared<TestDataProvider>(schema_provider_));

    executor_ = Executor::getExecutor(data_mgr_.get(), config_, "", "");

    init_calcite("");
  }

  static void init_calcite(const std::string& udf_filename) {
    calcite_ = CalciteMgr::get(udf_filename, config_->debug.log_dir);
  }

  static void TearDownTestSuite() {
    data_mgr_.reset();
    schema_provider_.reset();
    executor_.reset();
  }

  ExecutionResult runRAQuery(const std::string& query_ra, Executor* executor) {
    auto dag = std::make_unique<RelAlgDagBuilder>(
        query_ra, TEST_DB_ID, schema_provider_, config_);
    auto ra_executor = RelAlgExecutor(executor, schema_provider_, std::move(dag));

    auto co = CompilationOptions::defaults(ExecutorDeviceType::CPU);
    co.use_groupby_buffer_desc = use_groupby_buffer_desc;
    return ra_executor.executeRelAlgQuery(
        co, ExecutionOptions::fromConfig(config()), false);
  }

  ExecutionResult runSqlQuery(const std::string& sql, Executor* executor) {
    const auto query_ra =
        calcite_->process("test_db", sql, schema_provider_.get(), config_.get());
    return runRAQuery(query_ra, executor);
  }

  RelAlgExecutor getExecutor(const std::string& sql) {
    const auto query_ra =
        calcite_->process("test_db", sql, schema_provider_.get(), config_.get());
    auto dag = std::make_unique<RelAlgDagBuilder>(
        query_ra, TEST_DB_ID, schema_provider_, config_);
    return RelAlgExecutor(executor_.get(), schema_provider_, std::move(dag));
  }

  TestDataProvider& getDataProvider() {
    auto* ps_mgr = data_mgr_->getPersistentStorageMgr();
    auto data_provider_ptr = ps_mgr->getDataProvider(TEST_SCHEMA_ID);
    return dynamic_cast<TestDataProvider&>(*data_provider_ptr);
  }

  const Config& config() const { return *config_; }

 protected:
  static ConfigPtr config_;
  static std::shared_ptr<DataMgr> data_mgr_;
  static SchemaProviderPtr schema_provider_;
  static std::shared_ptr<Executor> executor_;
  static CalciteMgr* calcite_;
};

ConfigPtr NoCatalogSqlTest::config_;
std::shared_ptr<DataMgr> NoCatalogSqlTest::data_mgr_;
SchemaProviderPtr NoCatalogSqlTest::schema_provider_;
std::shared_ptr<Executor> NoCatalogSqlTest::executor_;
CalciteMgr* NoCatalogSqlTest::calcite_;

TEST_F(NoCatalogSqlTest, SelectSingleColumn) {
  auto res = runSqlQuery("SELECT col_i FROM test1;", executor_.get());
  compare_res_data(res, std::vector<int>({10, 20, 30, 40, 50}));
}

TEST_F(NoCatalogSqlTest, SelectAllColumns) {
  auto res = runSqlQuery("SELECT * FROM test1;", executor_.get());
  compare_res_data(res,
                   std::vector<int64_t>({1, 2, 3, 4, 5}),
                   std::vector<int>({10, 20, 30, 40, 50}),
                   std::vector<float>({1.1, 2.2, 3.3, 4.4, 5.5}),
                   std::vector<double>({10.1, 20.2, 30.3, 40.4, 50.5}));
}

TEST_F(NoCatalogSqlTest, SelectAllColumnsMultiFrag) {
  auto res = runSqlQuery("SELECT * FROM test2;", executor_.get());
  compare_res_data(
      res,
      std::vector<int64_t>({1, 2, 3, 4, 5, 6, 7, 8, 9}),
      std::vector<int>({110, 120, 130, 140, 150, 160, 170, 180, 190}),
      std::vector<float>({101.1, 102.2, 103.3, 104.4, 105.5, 106.6, 107.7, 108.8, 109.9}),
      std::vector<double>(
          {110.1, 120.2, 130.3, 140.4, 150.5, 160.6, 170.7, 180.8, 190.9}));
}

TEST_F(NoCatalogSqlTest, GroupBySingleColumn) {
  auto res = runSqlQuery(
      "SELECT id, COUNT(*), COUNT(val), SUM(val), AVG(val) FROM test_agg GROUP BY id "
      "ORDER BY id;",
      executor_.get());
  compare_res_data(res,
                   std::vector<int32_t>({1, 2, 3}),
                   std::vector<int32_t>({5, 3, 2}),
                   std::vector<int32_t>({5, 2, 1}),
                   std::vector<int64_t>({250, 60, 100}),
                   std::vector<double>({50, 30, 100}));
}

TEST_F(NoCatalogSqlTest, MultipleCalciteMultipleThreads) {
  constexpr size_t TEST_NTHREADS = 100;
  std::vector<ExecutionResult> res;
  std::vector<std::future<void>> threads;
  res.resize(TEST_NTHREADS);
  threads.resize(TEST_NTHREADS);
  std::vector<std::shared_ptr<Executor>> executors;
  executors.resize(TEST_NTHREADS);

  auto calcite = calcite_;

  for (size_t i = 0; i < TEST_NTHREADS; ++i) {
    executors[i] = Executor::getExecutor(data_mgr_.get(), config_);
    threads[i] = std::async(std::launch::async,
                            [this,
                             i,
                             &res,
                             &executors,
                             calcite,
                             schema_provider = schema_provider_.get(),
                             config = config_.get()]() {
                              auto query_ra = calcite->process(
                                  "test_db",
                                  "SELECT col_bi + " + std::to_string(i) + " FROM test1;",
                                  schema_provider,
                                  config);
                              CHECK(i < executors.size() && executors[i]);
                              res[i] = runRAQuery(query_ra, executors[i].get());
                            });
  }
  for (size_t i = 0; i < TEST_NTHREADS; ++i) {
    threads[i].wait();
  }
  for (size_t i = 0; i < TEST_NTHREADS; ++i) {
    const auto i_int = static_cast<int64_t>(i);
    compare_res_data(
        res[i],
        std::vector<int64_t>({1 + i_int, 2 + i_int, 3 + i_int, 4 + i_int, 5 + i_int}));
  }
}

TEST(CalciteReinitTest, SingleThread) {
  auto schema_provider = std::make_shared<TestSchemaProvider>();
  auto config = std::make_shared<Config>();
  config->exec.cpu_only = true;
  for (int i = 0; i < 10; ++i) {
    auto calcite = CalciteMgr::get();
    auto query_ra =
        calcite->process("test_db", "SELECT 1;", schema_provider.get(), config.get());
    CHECK(query_ra.find("LogicalValues") != std::string::npos) << query_ra;
  }
}

TEST(CalciteReinitTest, MultipleThreads) {
  auto schema_provider = std::make_shared<TestSchemaProvider>();
  auto config = std::make_shared<Config>();
  config->exec.cpu_only = true;
  for (int i = 0; i < 10; ++i) {
    auto f = std::async(std::launch::async, [schema_provider, config]() {
      auto calcite = CalciteMgr::get();
      auto query_ra =
          calcite->process("test_db", "SELECT 1;", schema_provider.get(), config.get());
      CHECK(query_ra.find("LogicalValues") != std::string::npos) << query_ra;
    });
    f.wait();
  }
}

void parse_cli_args_to_globals(int argc, char* argv[]) {
  namespace po = boost::program_options;

  po::options_description desc("Options");

  desc.add_options()("help,h", "Print help messages ");

  desc.add_options()("use-groupby-buffer-desc",
                     po::bool_switch()->default_value(false),
                     "Use GroupBy Buffer Descriptor for hash tables.");
  desc.add_options()("cpu-only", "ignored option");

  po::variables_map vm;

  try {
    po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);

    if (vm.count("help")) {
      std::cout << desc;
      std::exit(EXIT_SUCCESS);
    }
    po::notify(vm);
    use_groupby_buffer_desc = vm["use-groupby-buffer-desc"].as<bool>();

  } catch (boost::program_options::error& e) {
    std::cerr << "Usage Error: " << e.what() << std::endl;
    std::cout << desc;
    std::exit(EXIT_FAILURE);
  }
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  int err{0};
  parse_cli_args_to_globals(argc, argv);

  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  return err;
}
