/*
 * Copyright 2022 Intel Corporation.
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

#include "CalciteJNI.h"
#include "CalciteAdapter.h"
#include "SchemaJson.h"

#include "Logger/Logger.h"
#include "OSDependent/omnisci_path.h"

#include <jni.h>
#include <filesystem>

using namespace std::string_literals;

namespace {

class JVM {
 public:
  class JNIEnvWrapper {
   public:
    JNIEnvWrapper(JVM* jvm, JNIEnv* env, bool detach_thread_on_destruction)
        : jvm_(jvm)
        , env_(env)
        , detach_thread_on_destruction_(detach_thread_on_destruction) {
      if (env_) {
        auto res = env_->PushLocalFrame(100);
        if (!res) {
          std::runtime_error("Cannot create JNI Local Frame");
        }
      }
    }

    JNIEnvWrapper(const JNIEnvWrapper& other) = delete;

    JNIEnvWrapper(JNIEnvWrapper&& other) { *this = std::move(other); }

    JNIEnvWrapper& operator=(const JNIEnvWrapper& other) = delete;

    JNIEnvWrapper& operator=(JNIEnvWrapper&& other) {
      jvm_ = other.jvm_;
      env_ = other.env_;
      detach_thread_on_destruction_ = other.detach_thread_on_destruction_;

      other.detach_thread_on_destruction_ = false;
      other.env_ = nullptr;
      other.jvm_ = nullptr;

      return *this;
    }

    ~JNIEnvWrapper() {
      if (env_) {
        env_->PopLocalFrame(nullptr);
      }
      if (detach_thread_on_destruction_) {
        jvm_->detachThread();
      }
    }

    JNIEnv* get() const { return env_; }

    JNIEnv* operator->() const { return env_; }

   private:
    JVM* jvm_;
    JNIEnv* env_;
    bool detach_thread_on_destruction_;
  };

  static std::shared_ptr<JVM> getInstance(size_t max_mem_mb) {
    std::call_once(instance_init_flag_, [=] { instance_ = createJVM(max_mem_mb); });
    return instance_;
  }

  static void destroyInstance() { instance_ = nullptr; }

  // Get JNI environment for the current thread.
  // You souldn't pass this obect between threads. It should be deallocated
  // in the same thread it was requrested in. It shouldn't outlive JVM object.
  JNIEnvWrapper getEnv() {
    JNIEnv* env;
    bool need_detach = false;

    auto res = jvm_->GetEnv((void**)&env, JNI_VERSION_1_8);
    if (res != JNI_OK) {
      if (res != JNI_EDETACHED) {
        LOG(FATAL) << "Cannot get Java Env: error code " << res;
      }

      JavaVMAttachArgs args;
      args.version = JNI_VERSION_1_8;
      args.group = nullptr;
      args.name = nullptr;
      res = jvm_->AttachCurrentThread((void**)&env, &args);
      if (res != JNI_OK) {
        LOG(FATAL) << "Cannot attach thread to JavaVM: error code " << res;
      }
      need_detach = true;
    }

    return {this, env, need_detach};
  }

  ~JVM() { jvm_->DestroyJavaVM(); }

 private:
  JVM(JavaVM* jvm) : jvm_(jvm) {}

  static std::shared_ptr<JVM> createJVM(size_t max_mem_mb) {
    auto root_abs_path = omnisci::get_root_abs_path();
    std::string class_path_arg = "-Djava.class.path=";
    if (std::filesystem::exists(root_abs_path +
                                "/bin/calcite-1.0-SNAPSHOT-jar-with-dependencies.jar")) {
      class_path_arg +=
          root_abs_path + "/bin/calcite-1.0-SNAPSHOT-jar-with-dependencies.jar";
    } else if (std::filesystem::exists(
                   root_abs_path +
                   "/../bin/calcite-1.0-SNAPSHOT-jar-with-dependencies.jar")) {
      class_path_arg +=
          root_abs_path + "/../bin/calcite-1.0-SNAPSHOT-jar-with-dependencies.jar";
    } else {
      LOG(FATAL) << "Cannot find calcite jar library.";
    }
    std::string max_mem_arg = "-Xmx" + std::to_string(max_mem_mb) + "m";
    std::string log_dir_arg = "-DlogDir=hdk_log";
    JavaVMInitArgs vm_args;
    auto options = std::make_unique<JavaVMOption[]>(3);
    options[0].optionString = const_cast<char*>(class_path_arg.c_str());
    options[1].optionString = const_cast<char*>(max_mem_arg.c_str());
    options[2].optionString = const_cast<char*>(log_dir_arg.c_str());
    vm_args.version = JNI_VERSION_1_8;
    vm_args.nOptions = 3;
    vm_args.options = options.get();
    vm_args.ignoreUnrecognized = false;

    // Java machine and environment.
    JavaVM* jvm;
    JNIEnv* env;
    if (JNI_CreateJavaVM(&jvm, (void**)&env, &vm_args) != JNI_OK) {
      LOG(FATAL) << "Couldn't initialize JVM.";
    }

    return std::shared_ptr<JVM>(new JVM(jvm));
  }

  // Detach current thread from JVM.
  void detachThread() { jvm_->DetachCurrentThread(); }

  JavaVM* jvm_;

  static std::once_flag instance_init_flag_;
  static std::shared_ptr<JVM> instance_;
};

std::once_flag JVM::instance_init_flag_;
std::shared_ptr<JVM> JVM::instance_;

}  // namespace

class CalciteJNI {
 public:
  CalciteJNI(const std::string& udf_filename, size_t calcite_max_mem_mb) {
    // Initialize JVM.
    jvm_ = JVM::getInstance(calcite_max_mem_mb);
    auto env = jvm_->getEnv();

    // Create CalciteServerHandler object.
    createCalciteServerHandler(env.get(), udf_filename);

    // Prepare references to some Java classes and methods we will use for processing.
    findQueryParsingOption(env.get());
    findOptimizationOption(env.get());
    findPlanResult(env.get());
    findExtArgumentType(env.get());
    findExtensionFunction(env.get());
    findInvalidParseRequest(env.get());
    findArrayList(env.get());
    findHashMap(env.get());
  }

  ~CalciteJNI() {
    auto env = jvm_->getEnv();
    for (auto obj : global_refs_) {
      env->DeleteGlobalRef(obj);
    }

    JVM::destroyInstance();
  }

  std::string process(const std::string& db_name,
                      const std::string& sql_string,
                      SchemaProvider* schema_provider,
                      Config* config,
                      const std::vector<FilterPushDownInfo>& filter_push_down_info,
                      const bool legacy_syntax,
                      const bool is_explain,
                      const bool is_view_optimize) {
    auto env = jvm_->getEnv();
    jstring arg_catalog = env->NewStringUTF(db_name.c_str());
    std::string modified_sql = pg_shim(sql_string);
    jstring arg_query = env->NewStringUTF(modified_sql.c_str());
    jobject arg_parsing_options = env->NewObject(parsing_opts_cls_,
                                                 parsing_opts_ctor_,
                                                 (jboolean)legacy_syntax,
                                                 (jboolean)is_explain,
                                                 /*check_privileges=*/(jboolean)(false));
    if (!arg_parsing_options) {
      throw std::runtime_error("cannot create QueryParsingOption object");
    }
    jobject arg_filter_push_down_info = env->NewObject(array_list_cls_, array_list_ctor_);
    if (!filter_push_down_info.empty()) {
      throw std::runtime_error(
          "Filter pushdown info is not yet implemented in Calcite JNI client.");
    }
    jobject arg_optimization_options =
        env->NewObject(optimization_opts_cls_,
                       optimization_opts_ctor_,
                       (jboolean)is_view_optimize,
                       (jboolean)config->exec.watchdog.enable,
                       arg_filter_push_down_info);
    jobject arg_restriction = nullptr;
    auto schema_json = schema_to_json(schema_provider);
    jstring arg_schema = env->NewStringUTF(schema_json.c_str());

    jobject java_res = env->CallObjectMethod(handler_obj_,
                                             handler_process_,
                                             arg_catalog,
                                             arg_query,
                                             arg_parsing_options,
                                             arg_optimization_options,
                                             arg_restriction,
                                             arg_schema);
    if (!java_res) {
      if (env->ExceptionCheck() == JNI_FALSE) {
        throw std::runtime_error(
            "CalciteServerHandler::process call failed for unknown reason\n  Query: " +
            sql_string + "\n  Schema: " + schema_json);
      } else {
        jthrowable e = env->ExceptionOccurred();
        CHECK(e);
        throw std::invalid_argument(
            readStringField(env.get(), e, invalid_parse_req_msg_));
      }
    }

    return readStringField(env.get(), java_res, plan_result_plan_result_);
  }

  std::string getExtensionFunctionWhitelist() {
    auto env = jvm_->getEnv();
    jstring java_res =
        (jstring)env->CallObjectMethod(handler_obj_, handler_get_ext_fn_list_);
    return convertJavaString(env.get(), java_res);
  }

  std::string getUserDefinedFunctionWhitelist() {
    auto env = jvm_->getEnv();
    jstring java_res =
        (jstring)env->CallObjectMethod(handler_obj_, handler_get_udf_list_);
    return convertJavaString(env.get(), java_res);
  }

  std::string getRuntimeExtensionFunctionWhitelist() {
    auto env = jvm_->getEnv();
    jstring java_res =
        (jstring)env->CallObjectMethod(handler_obj_, handlhandler_get_rt_fn_list_);
    return convertJavaString(env.get(), java_res);
  }

  void setRuntimeExtensionFunctions(const std::vector<ExtensionFunction>& udfs,
                                    bool is_runtime) {
    auto env = jvm_->getEnv();
    jobject udfs_list = env->NewObject(array_list_cls_, array_list_ctor_);
    for (auto& udf : udfs) {
      env->CallVoidMethod(
          udfs_list, array_list_add_, convertExtensionFunction(env.get(), udf));
    }

    env->CallVoidMethod(
        handler_obj_, handler_set_rt_fns_, udfs_list, (jboolean)is_runtime);
    if (env->ExceptionCheck() != JNI_FALSE) {
      env->ExceptionDescribe();
      throw std::runtime_error("Failed Java call to setRuntimeExtensionFunctions");
    }
  }

 private:
  jobject addGlobalRef(JNIEnv* env, jobject obj) {
    auto res = env->NewGlobalRef(obj);
    global_refs_.push_back(res);
    return res;
  }

  jclass findClass(JNIEnv* env, std::string_view class_name) {
    jclass cls = env->FindClass(class_name.data());
    if (!cls) {
      throw std::runtime_error("cannot find Java class: "s.append(class_name));
    }
    return (jclass)addGlobalRef(env, cls);
  }

  void createCalciteServerHandler(JNIEnv* env, const std::string& udf_filename) {
    jclass handler_cls = findClass(env, "com/mapd/parser/server/CalciteServerHandler");
    jmethodID handler_ctor = env->GetMethodID(
        handler_cls, "<init>", "(Ljava/lang/String;Ljava/lang/String;)V");
    if (!handler_ctor) {
      throw std::runtime_error("cannot find CalciteServerHandler ctor");
    }

    auto root_abs_path = omnisci::get_root_abs_path();
    std::string ext_ast_path = root_abs_path + "/QueryEngine/ExtensionFunctions.ast";
    jstring ext_ast_file = env->NewStringUTF(ext_ast_path.c_str());
    jstring udf_ast_file = env->NewStringUTF(udf_filename.c_str());
    handler_obj_ = env->NewObject(handler_cls, handler_ctor, ext_ast_file, udf_ast_file);
    if (!handler_obj_) {
      throw std::runtime_error("cannot create CalciteServerHandler object");
    }
    handler_obj_ = addGlobalRef(env, handler_obj_);

    // Find 'CalciteServerHandler::process' method.
    handler_process_ = env->GetMethodID(
        handler_cls,
        "process",
        "(Ljava/lang/String;Ljava/lang/String;Lcom/"
        "mapd/parser/server/QueryParsingOption;Lcom/mapd/parser/server/"
        "OptimizationOption;Lorg/apache/calcite/rel/rules/Restriction;Ljava/lang/"
        "String;)Lcom/mapd/parser/server/PlanResult;");
    if (!handler_process_) {
      throw std::runtime_error("cannot find CalciteServerHandler::process method");
    }

    // Find 'CalciteServerHandler::getExtensionFunctionWhitelist' method.
    handler_get_ext_fn_list_ = env->GetMethodID(
        handler_cls, "getExtensionFunctionWhitelist", "()Ljava/lang/String;");
    if (!handler_get_ext_fn_list_) {
      throw std::runtime_error(
          "cannot find CalciteServerHandler::getExtensionFunctionWhitelist method");
    }

    // Find 'CalciteServerHandler::getUserDefinedFunctionWhitelist' method.
    handler_get_udf_list_ = env->GetMethodID(
        handler_cls, "getUserDefinedFunctionWhitelist", "()Ljava/lang/String;");
    if (!handler_get_udf_list_) {
      throw std::runtime_error(
          "cannot find CalciteServerHandler::getUserDefinedFunctionWhitelist method");
    }

    // Find 'CalciteServerHandler::getRuntimeExtensionFunctionWhitelist' method.
    handlhandler_get_rt_fn_list_ = env->GetMethodID(
        handler_cls, "getRuntimeExtensionFunctionWhitelist", "()Ljava/lang/String;");
    if (!handlhandler_get_rt_fn_list_) {
      throw std::runtime_error(
          "cannot find CalciteServerHandler::getRuntimeExtensionFunctionWhitelist "
          "method");
    }

    // Find 'CalciteServerHandler::setRuntimeExtensionFunctions' method.
    handler_set_rt_fns_ = env->GetMethodID(
        handler_cls, "setRuntimeExtensionFunctions", "(Ljava/util/List;Z)V");
    if (!handler_set_rt_fns_) {
      throw std::runtime_error(
          "cannot find CalciteServerHandler::setRuntimeExtensionFunctions "
          "method");
    }
  }

  void findQueryParsingOption(JNIEnv* env) {
    parsing_opts_cls_ = findClass(env, "com/mapd/parser/server/QueryParsingOption");
    parsing_opts_ctor_ = env->GetMethodID(parsing_opts_cls_, "<init>", "(ZZZ)V");
    if (!parsing_opts_ctor_) {
      throw std::runtime_error("cannot find QueryParsingOption ctor");
    }
  }

  void findOptimizationOption(JNIEnv* env) {
    optimization_opts_cls_ = findClass(env, "com/mapd/parser/server/OptimizationOption");
    optimization_opts_ctor_ =
        env->GetMethodID(optimization_opts_cls_, "<init>", "(ZZLjava/util/List;)V");
    if (!optimization_opts_ctor_) {
      throw std::runtime_error("cannot find OptimizationOption ctor");
    }
  }

  void findPlanResult(JNIEnv* env) {
    plan_result_cls_ = findClass(env, "com/mapd/parser/server/PlanResult");
    plan_result_plan_result_ =
        env->GetFieldID(plan_result_cls_, "planResult", "Ljava/lang/String;");
    if (!plan_result_plan_result_) {
      throw std::runtime_error("cannot find PlanResult::planResult field");
    }
  }

  void findExtArgumentType(JNIEnv* env) {
    jclass cls =
        findClass(env, "com/mapd/parser/server/ExtensionFunction$ExtArgumentType");
    jmethodID values_method = env->GetStaticMethodID(
        cls, "values", "()[Lcom/mapd/parser/server/ExtensionFunction$ExtArgumentType;");
    if (!values_method) {
      throw std::runtime_error("cannot find ExtArgumentType::values method");
    }
    jobjectArray values = (jobjectArray)env->CallStaticObjectMethod(cls, values_method);
    for (jsize i = 0; i < env->GetArrayLength(values); i++) {
      ext_arg_type_vals_.push_back(
          addGlobalRef(env, env->GetObjectArrayElement(values, i)));
      CHECK(ext_arg_type_vals_.back());
    }
  }

  void findExtensionFunction(JNIEnv* env) {
    extension_fn_cls_ = findClass(env, "com/mapd/parser/server/ExtensionFunction");
    extension_fn_udf_ctor_ =
        env->GetMethodID(extension_fn_cls_,
                         "<init>",
                         "(Ljava/lang/String;Ljava/util/List;Lcom/mapd/parser/server/"
                         "ExtensionFunction$ExtArgumentType;)V");
    if (!extension_fn_udf_ctor_) {
      throw std::runtime_error("cannot find ExtensionFunction (UDF) ctor");
    }
  }

  void findInvalidParseRequest(JNIEnv* env) {
    invalid_parse_req_cls_ = findClass(env, "com/mapd/parser/server/InvalidParseRequest");
    invalid_parse_req_msg_ =
        env->GetFieldID(invalid_parse_req_cls_, "msg", "Ljava/lang/String;");
    if (!invalid_parse_req_msg_) {
      throw std::runtime_error("cannot find InvalidParseRequest::msg field");
    }
  }

  void findArrayList(JNIEnv* env) {
    array_list_cls_ = findClass(env, "java/util/ArrayList");
    array_list_ctor_ = env->GetMethodID(array_list_cls_, "<init>", "()V");
    if (!array_list_ctor_) {
      throw std::runtime_error("cannot find ArrayList ctor");
    }
    array_list_add_ = env->GetMethodID(array_list_cls_, "add", "(Ljava/lang/Object;)Z");
    if (!array_list_add_) {
      throw std::runtime_error("cannot find ArrayList::add method");
    }
  }

  void findHashMap(JNIEnv* env) {
    hash_map_cls_ = findClass(env, "java/util/HashMap");
    hash_map_ctor_ = env->GetMethodID(hash_map_cls_, "<init>", "()V");
    if (!hash_map_ctor_) {
      throw std::runtime_error("cannot find HashMap ctor");
    }
    hash_map_put_ = env->GetMethodID(
        hash_map_cls_, "put", "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");
    if (!hash_map_put_) {
      throw std::runtime_error("cannot find HashMap::put method");
    }
  }

  std::string convertJavaString(JNIEnv* env, jstring str_obj) {
    const char* res_str = env->GetStringUTFChars(str_obj, 0);
    std::string res = res_str;
    env->ReleaseStringUTFChars(str_obj, res_str);
    return res;
  }

  std::string readStringField(JNIEnv* env, jobject obj, jfieldID field) {
    auto field_obj = (jstring)env->GetObjectField(obj, field);
    return convertJavaString(env, field_obj);
  }

  jobject convertExtArgumentType(ExtArgumentType val) {
    size_t index = static_cast<size_t>(val);
    if (index < ext_arg_type_vals_.size()) {
      return ext_arg_type_vals_[index];
    }
    throw std::runtime_error("ExtArgumentType enum value is out of range (" +
                             to_string(index) + ")");
  }

  void addArgTypesAndNames(
      JNIEnv* env,
      jobject types,
      jobject names,
      const std::vector<ExtArgumentType>& args,
      const std::vector<std::map<std::string, std::string>>& annotations,
      const std::string& prefix,
      size_t ann_offset) {
    for (size_t index = 0; index < args.size(); ++index) {
      env->CallVoidMethod(types, array_list_add_, convertExtArgumentType(args[index]));
      auto& ann = annotations[index + ann_offset];
      if (ann.count("name")) {
        env->CallVoidMethod(
            names, array_list_add_, env->NewStringUTF(ann.at("name").c_str()));
      } else {
        env->CallVoidMethod(names,
                            array_list_add_,
                            env->NewStringUTF((prefix + std::to_string(index)).c_str()));
      }
    }
  }

  jobject convertExtensionFunction(JNIEnv* env, const ExtensionFunction& udf) {
    jstring name_arg = env->NewStringUTF(udf.getName().c_str());
    jobject args_arg = env->NewObject(array_list_cls_, array_list_ctor_);
    for (auto& arg : udf.getArgs()) {
      env->CallVoidMethod(args_arg, array_list_add_, convertExtArgumentType(arg));
    }
    jobject ret_arg = convertExtArgumentType(udf.getRet());
    return env->NewObject(
        extension_fn_cls_, extension_fn_udf_ctor_, name_arg, args_arg, ret_arg);
  }

  // com.mapd.parser.server.CalciteServerHandler instance and methods.
  jobject handler_obj_;
  jmethodID handler_process_;
  jmethodID handler_get_ext_fn_list_;
  jmethodID handler_get_udf_list_;
  jmethodID handlhandler_get_rt_fn_list_;
  jmethodID handler_set_rt_fns_;

  // com.mapd.parser.server.QueryParsingOption class and methods
  jclass parsing_opts_cls_;
  jmethodID parsing_opts_ctor_;

  // com.mapd.parser.server.OptimizationOption class and methods
  jclass optimization_opts_cls_;
  jmethodID optimization_opts_ctor_;

  // com.mapd.parser.server.PlanResult class and fields
  jclass plan_result_cls_;
  jfieldID plan_result_plan_result_;

  // com.mapd.parser.server.ExtensionFunction$ExtArgumentType enum values
  std::vector<jobject> ext_arg_type_vals_;

  // com.mapd.parser.server.ExtensionFunction class and methods
  jclass extension_fn_cls_;
  jmethodID extension_fn_udf_ctor_;

  // com.mapd.parser.server.InvalidParseRequest class and fields
  jclass invalid_parse_req_cls_;
  jfieldID invalid_parse_req_msg_;

  // java.util.ArrayList class and methods
  jclass array_list_cls_;
  jmethodID array_list_ctor_;
  jmethodID array_list_add_;

  // java.util.HashMap class and methods
  jclass hash_map_cls_;
  jmethodID hash_map_ctor_;
  jmethodID hash_map_put_;

  std::vector<jobject> global_refs_;
  std::shared_ptr<JVM> jvm_;
};

CalciteMgr::~CalciteMgr() {
  {
    std::lock_guard<decltype(queue_mutex_)> lock(queue_mutex_);
    should_exit_ = true;
  }
  worker_cv_.notify_all();
  worker_.join();
}

CalciteMgr* CalciteMgr::get(const std::string& udf_filename, size_t calcite_max_mem_mb) {
  std::call_once(instance_init_flag_, [=] {
    instance_ =
        std::unique_ptr<CalciteMgr>(new CalciteMgr(udf_filename, calcite_max_mem_mb));
  });
  return instance_.get();
}

std::string CalciteMgr::process(
    const std::string& db_name,
    const std::string& sql_string,
    SchemaProvider* schema_provider,
    Config* config,
    const std::vector<FilterPushDownInfo>& filter_push_down_info,
    const bool legacy_syntax,
    const bool is_explain,
    const bool is_view_optimize) {
  auto task = Task([&db_name,
                    &sql_string,
                    &filter_push_down_info,
                    schema_provider,
                    config,
                    legacy_syntax,
                    is_explain,
                    is_view_optimize](CalciteJNI* calcite_jni) {
    CHECK(calcite_jni);
    return calcite_jni->process(db_name,
                                sql_string,
                                schema_provider,
                                config,
                                filter_push_down_info,
                                legacy_syntax,
                                is_explain,
                                is_view_optimize);
  });
  auto result = task.get_future();
  submitTaskToQueue(std::move(task));

  result.wait();
  return result.get();
}

std::string CalciteMgr::getExtensionFunctionWhitelist() {
  auto task = Task([](CalciteJNI* calcite_jni) {
    CHECK(calcite_jni);
    return calcite_jni->getExtensionFunctionWhitelist();
  });

  auto result = task.get_future();
  submitTaskToQueue(std::move(task));

  result.wait();
  return result.get();
}

std::string CalciteMgr::getUserDefinedFunctionWhitelist() {
  auto task = Task([](CalciteJNI* calcite_jni) {
    CHECK(calcite_jni);
    return calcite_jni->getUserDefinedFunctionWhitelist();
  });

  auto result = task.get_future();
  submitTaskToQueue(std::move(task));

  result.wait();
  return result.get();
}

std::string CalciteMgr::getRuntimeExtensionFunctionWhitelist() {
  auto task = Task([](CalciteJNI* calcite_jni) {
    CHECK(calcite_jni);
    return calcite_jni->getRuntimeExtensionFunctionWhitelist();
  });

  auto result = task.get_future();
  submitTaskToQueue(std::move(task));

  result.wait();
  return result.get();
}

void CalciteMgr::setRuntimeExtensionFunctions(const std::vector<ExtensionFunction>& udfs,
                                              bool is_runtime) {
  auto task = Task([&udfs, is_runtime](CalciteJNI* calcite_jni) {
    CHECK(calcite_jni);
    calcite_jni->setRuntimeExtensionFunctions(udfs, is_runtime);
    return "";  // all tasks return strings
  });

  auto result = task.get_future();
  submitTaskToQueue(std::move(task));

  result.wait();
}

CalciteMgr::CalciteMgr(const std::string& udf_filename, size_t calcite_max_mem_mb) {
  // todo: should register an exit handler for ctrl + c
  worker_ = std::thread(&CalciteMgr::worker, this, udf_filename, calcite_max_mem_mb);
}

void CalciteMgr::worker(const std::string& udf_filename, size_t calcite_max_mem_mb) {
  auto calcite_jni = std::make_unique<CalciteJNI>(udf_filename, calcite_max_mem_mb);

  std::unique_lock<std::mutex> lock(queue_mutex_);
  while (true) {
    worker_cv_.wait(lock, [this] { return !queue_.empty() || should_exit_; });
    if (should_exit_) {
      return;
    }

    if (!queue_.empty()) {
      auto task = std::move(queue_.front());
      queue_.pop();

      lock.unlock();
      task(calcite_jni.get());

      lock.lock();
    }
  }
}

void CalciteMgr::submitTaskToQueue(Task&& task) {
  std::unique_lock<decltype(queue_mutex_)> lock(queue_mutex_);

  queue_.push(std::move(task));

  lock.unlock();
  worker_cv_.notify_all();
}

std::once_flag CalciteMgr::instance_init_flag_;
std::unique_ptr<CalciteMgr> CalciteMgr::instance_;
