#pragma once

#include "model.h"
#include <string>

namespace ml {

void save_weights(const Model& m, const std::string& path);
void load_weights(Model& m, const std::string& path);

}  // namespace ml
