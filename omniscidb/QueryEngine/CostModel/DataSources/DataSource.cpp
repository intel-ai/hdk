/*
    Copyright (c) 2022 Intel Corporation
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
        http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#include "DataSource.h"

namespace costmodel {

DataSource::DataSource(const DataSourceConfig& config) : config_(config) {}

const std::string& DataSource::getName() {
  return config_.dataSourceName;
}

bool DataSource::isDeviceSupported(ExecutorDeviceType device) {
  return config_.supportedDevices.find(device) != config_.supportedDevices.end();
}

bool DataSource::isTemplateSupported(AnalyticalTemplate templ) {
  return config_.supportedTemplates.find(templ) != config_.supportedTemplates.end();
}

}  // namespace costmodel
