/*
    Copyright (c) 2023 Intel Corporation
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

#include "ExtrapolationModelProvider.h"

namespace costmodel {
std::unique_ptr<ExtrapolationModel> ExtrapolationModelProvider::provide(
    const std::vector<Detail::Measurement>& measurement) {
#ifdef HAVE_ARMADILLO
  return std::make_unique<LinearRegression>(measurement);
#else
  return std::make_unique<LinearExtrapolation>(measurement);
#endif
}
std::unique_ptr<ExtrapolationModel> ExtrapolationModelProvider::provide(
    std::vector<Detail::Measurement>&& measurement) {
#ifdef HAVE_ARMADILLO
  return std::make_unique<LinearRegression>(std::move(measurement));
#else
  return std::make_unique<LinearExtrapolation>(std::move(measurement));
#endif
}
}  // namespace costmodel
