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

#pragma once

#include "ExtrapolationModel.h"

#include <memory>

namespace costmodel {

class LinearRegression : public ExtrapolationModel {
 public:
  LinearRegression(const std::vector<Detail::Measurement>& measurement);

  LinearRegression(std::vector<Detail::Measurement>&& measurement);

  virtual ~LinearRegression();

  size_t getExtrapolatedData(size_t bytes) const override;

 protected:
  void buildRegressionCoefficients();

  struct PrivateImpl;
  std::unique_ptr<PrivateImpl> pimpl_;
};

}  // namespace costmodel
