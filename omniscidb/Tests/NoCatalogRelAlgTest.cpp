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

#include "QueryEngine/RelAlgExecutor.h"
#include "SchemaMgr/SimpleSchemaProvider.h"

#include "ArrowTestHelpers.h"
#include "DataMgr/DataMgrBufferProvider.h"
#include "DataMgr/DataMgrDataProvider.h"
#include "TestDataProvider.h"
#include "TestHelpers.h"
#include "TestRelAlgDagBuilder.h"

#include <gtest/gtest.h>

constexpr int TEST_SCHEMA_ID = 1;
constexpr int TEST_DB_ID = (TEST_SCHEMA_ID << 24) + 1;
constexpr int TEST1_TABLE_ID = 1;
constexpr int TEST2_TABLE_ID = 2;
constexpr int TEST_AGG_TABLE_ID = 3;

constexpr int TEST_SCHEMA_ID2 = 2;
constexpr int TEST_DB2_ID = (TEST_SCHEMA_ID2 << 24) + 1;

using ArrowTestHelpers::compare_res_data;

class TestSchemaProvider : public SimpleSchemaProvider {
 public:
  TestSchemaProvider()
      : SimpleSchemaProvider(hdk::ir::Context::defaultCtx(), TEST_SCHEMA_ID, "test") {
    // Table test1
    addTableInfo(TEST_DB_ID, TEST1_TABLE_ID, "test1", false, 1);
    addColumnInfo(TEST_DB_ID, TEST1_TABLE_ID, 1, "col_bi", ctx_.int64(), false);
    addColumnInfo(TEST_DB_ID, TEST1_TABLE_ID, 2, "col_i", ctx_.int32(), false);
    addColumnInfo(TEST_DB_ID, TEST1_TABLE_ID, 3, "col_f", ctx_.fp32(), false);
    addColumnInfo(TEST_DB_ID, TEST1_TABLE_ID, 4, "col_d", ctx_.fp64(), false);
    addRowidColumn(TEST_DB_ID, TEST1_TABLE_ID, 5);

    // Table test2
    addTableInfo(TEST_DB_ID, TEST2_TABLE_ID, "test2", false, 1);
    addColumnInfo(TEST_DB_ID, TEST2_TABLE_ID, 1, "col_bi", ctx_.int64(), false);
    addColumnInfo(TEST_DB_ID, TEST2_TABLE_ID, 2, "col_i", ctx_.int32(), false);
    addColumnInfo(TEST_DB_ID, TEST2_TABLE_ID, 3, "col_f", ctx_.fp32(), false);
    addColumnInfo(TEST_DB_ID, TEST2_TABLE_ID, 4, "col_d", ctx_.fp64(), false);
    addRowidColumn(TEST_DB_ID, TEST2_TABLE_ID, 5);

    // Table test2 in db2
    addTableInfo(TEST_DB2_ID, TEST2_TABLE_ID, "db2.test2", false, 1);
    addColumnInfo(TEST_DB2_ID, TEST2_TABLE_ID, 1, "col_bi", ctx_.int64(), false);
    addColumnInfo(TEST_DB2_ID, TEST2_TABLE_ID, 2, "col_i", ctx_.int32(), false);
    addRowidColumn(TEST_DB2_ID, TEST2_TABLE_ID, 3);

    // Table test_agg
    addTableInfo(TEST_DB_ID, TEST_AGG_TABLE_ID, "test_agg", false, 1);
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

class TestDataProvider2 : public TestHelpers::TestDataProvider {
 public:
  TestDataProvider2(SchemaProviderPtr schema_provider)
      : TestHelpers::TestDataProvider(TEST_DB2_ID, schema_provider) {
    TestHelpers::TestTableData test2(TEST_DB2_ID, TEST2_TABLE_ID, 2, schema_provider_);
    test2.addColFragment<int64_t>(1, {1, 2, 3});
    test2.addColFragment<int64_t>(1, {4, 5, 6});
    test2.addColFragment<int32_t>(2, {111, 122, 133});
    test2.addColFragment<int32_t>(2, {144, 155, 166});
    tables_.emplace(std::make_pair(TEST2_TABLE_ID, test2));
  }

  ~TestDataProvider2() override = default;
};

class NoCatalogRelAlgTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    config_ = std::make_shared<Config>();
    schema_provider_ = std::make_shared<TestSchemaProvider>();

    data_mgr_ = std::make_shared<DataMgr>(*config_, false);
    auto* ps_mgr = data_mgr_->getPersistentStorageMgr();
    ps_mgr->registerDataProvider(TEST_SCHEMA_ID,
                                 std::make_shared<TestDataProvider>(schema_provider_));
    ps_mgr->registerDataProvider(TEST_SCHEMA_ID2,
                                 std::make_shared<TestDataProvider2>(schema_provider_));

    executor_ = Executor::getExecutor(data_mgr_.get(), config_, "", "");
  }

  static void TearDownTestSuite() {}

  ExecutionResult runRelAlgQuery(const std::string& ra) {
    return runRelAlgQuery(
        std::make_unique<RelAlgDagBuilder>(ra, TEST_DB_ID, schema_provider_, config_));
  }

  ExecutionResult runRelAlgQuery(std::unique_ptr<hdk::ir::QueryDag> dag) {
    auto ra_executor = RelAlgExecutor(executor_.get(), schema_provider_, std::move(dag));
    return ra_executor.executeRelAlgQuery(
        CompilationOptions::defaults(ExecutorDeviceType::CPU),
        ExecutionOptions::fromConfig(executor_->getConfig()),
        false);
  }

  TestDataProvider& getDataProvider() {
    auto* ps_mgr = data_mgr_->getPersistentStorageMgr();
    auto data_provider_ptr = ps_mgr->getDataProvider(TEST_SCHEMA_ID);
    return dynamic_cast<TestDataProvider&>(*data_provider_ptr);
  }

 protected:
  static ConfigPtr config_;
  static std::shared_ptr<DataMgr> data_mgr_;
  static SchemaProviderPtr schema_provider_;
  static std::shared_ptr<Executor> executor_;
};

ConfigPtr NoCatalogRelAlgTest::config_;
std::shared_ptr<DataMgr> NoCatalogRelAlgTest::data_mgr_;
SchemaProviderPtr NoCatalogRelAlgTest::schema_provider_;
std::shared_ptr<Executor> NoCatalogRelAlgTest::executor_;

TEST_F(NoCatalogRelAlgTest, SelectSingleColumn) {
  auto dag = std::make_unique<TestRelAlgDagBuilder>(schema_provider_, config_);
  dag->addProject(dag->addScan(TEST_DB_ID, "test1"), std::vector<int>({1}));
  dag->finalize();
  auto res = runRelAlgQuery(std::move(dag));
  compare_res_data(res, std::vector<int>({10, 20, 30, 40, 50}));
}

TEST_F(NoCatalogRelAlgTest, SelectAllColumns) {
  auto dag = std::make_unique<TestRelAlgDagBuilder>(schema_provider_, config_);
  dag->addProject(dag->addScan(TEST_DB_ID, "test1"), std::vector<int>({0, 1, 2, 3}));
  dag->finalize();
  auto res = runRelAlgQuery(std::move(dag));
  compare_res_data(res,
                   std::vector<int64_t>({1, 2, 3, 4, 5}),
                   std::vector<int>({10, 20, 30, 40, 50}),
                   std::vector<float>({1.1, 2.2, 3.3, 4.4, 5.5}),
                   std::vector<double>({10.1, 20.2, 30.3, 40.4, 50.5}));
}

TEST_F(NoCatalogRelAlgTest, SelectAllColumnsMultiFrag) {
  auto dag = std::make_unique<TestRelAlgDagBuilder>(schema_provider_, config_);
  dag->addProject(dag->addScan(TEST_DB_ID, "test2"), std::vector<int>({0, 1, 2, 3}));
  dag->finalize();
  auto res = runRelAlgQuery(std::move(dag));
  compare_res_data(
      res,
      std::vector<int64_t>({1, 2, 3, 4, 5, 6, 7, 8, 9}),
      std::vector<int>({110, 120, 130, 140, 150, 160, 170, 180, 190}),
      std::vector<float>({101.1, 102.2, 103.3, 104.4, 105.5, 106.6, 107.7, 108.8, 109.9}),
      std::vector<double>(
          {110.1, 120.2, 130.3, 140.4, 150.5, 160.6, 170.7, 180.8, 190.9}));
}

TEST_F(NoCatalogRelAlgTest, GroupBySingleColumn) {
  auto& ctx = hdk::ir::Context::defaultCtx();
  auto dag = std::make_unique<TestRelAlgDagBuilder>(schema_provider_, config_);
  auto proj =
      dag->addProject(dag->addScan(TEST_DB_ID, "test_agg"), std::vector<int>({0, 1}));
  auto agg = dag->addAgg(proj,
                         1,
                         {{hdk::ir::AggType::kCount},
                          {hdk::ir::AggType::kCount, ctx.int32(), 1},
                          {hdk::ir::AggType::kSum, ctx.int64(), 1},
                          {hdk::ir::AggType::kAvg, ctx.int32(), 1}});
  dag->addSort(
      agg, {{0, hdk::ir::SortDirection::Ascending, hdk::ir::NullSortedPosition::Last}});
  dag->finalize();
  auto res = runRelAlgQuery(std::move(dag));
  compare_res_data(res,
                   std::vector<int32_t>({1, 2, 3}),
                   std::vector<int32_t>({5, 3, 2}),
                   std::vector<int32_t>({5, 2, 1}),
                   std::vector<int64_t>({250, 60, 100}),
                   std::vector<double>({50, 30, 100}));
}

TEST_F(NoCatalogRelAlgTest, InnerJoin) {
  auto dag = std::make_unique<TestRelAlgDagBuilder>(schema_provider_, config_);
  auto join = dag->addEquiJoin(dag->addScan(TEST_DB_ID, "test1"),
                               dag->addScan(TEST_DB_ID, "test2"),
                               JoinType::INNER,
                               0,
                               0);
  dag->addProject(join, std::vector<int>({0, 1, 2, 3, 6, 7, 8}));
  dag->finalize();
  auto res = runRelAlgQuery(std::move(dag));
  compare_res_data(res,
                   std::vector<int64_t>({1, 2, 3, 4, 5}),
                   std::vector<int32_t>({10, 20, 30, 40, 50}),
                   std::vector<float>({1.1, 2.2, 3.3, 4.4, 5.5}),
                   std::vector<double>({10.1, 20.2, 30.3, 40.4, 50.5}),
                   std::vector<int32_t>({110, 120, 130, 140, 150}),
                   std::vector<float>({101.1, 102.2, 103.3, 104.4, 105.5}),
                   std::vector<double>({110.1, 120.2, 130.3, 140.4, 150.5}));
}

TEST_F(NoCatalogRelAlgTest, InterDatabaseJoin) {
  auto dag = std::make_unique<TestRelAlgDagBuilder>(schema_provider_, config_);
  auto join = dag->addEquiJoin(dag->addScan(TEST_DB_ID, "test1"),
                               dag->addScan(TEST_DB2_ID, "db2.test2"),
                               JoinType::INNER,
                               0,
                               0);
  dag->addProject(join, std::vector<int>({0, 1, 6}));
  dag->finalize();
  auto res = runRelAlgQuery(std::move(dag));
  compare_res_data(res,
                   std::vector<int64_t>({1, 2, 3, 4, 5}),
                   std::vector<int32_t>({10, 20, 30, 40, 50}),
                   std::vector<int32_t>({111, 122, 133, 144, 155}));
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  return err;
}
