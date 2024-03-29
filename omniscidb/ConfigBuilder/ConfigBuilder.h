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

#pragma once

#include "Shared/Config.h"

class ConfigBuilder {
 public:
  ConfigBuilder();
  ConfigBuilder(ConfigPtr config);

  ConfigBuilder(const ConfigBuilder& other) = delete;
  ConfigBuilder(ConfigBuilder&& other) = delete;

  bool parseCommandLineArgs(int argc,
                            char const* const* argv,
                            bool allow_gtest_flags = false);
  bool parseCommandLineArgs(const std::string& app_name,
                            const std::string& cmd_line,
                            bool allow_gtest_flags = false);

  ConfigPtr config();

 private:
  ConfigPtr config_;
};
