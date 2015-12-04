//
// Created by kristoffer on 2015-12-04.
//

#include <assert.h>
#include "JuliaMaterial.h"

namespace JuliaMaterial {
void JuliaMaterial::check_exception() {
  if (jl_exception_occurred()) {
    jl_show(jl_stderr_obj(), jl_exception_occurred());
    jl_printf(jl_stderr_stream(), "\n");
    throw;
  }
}

JuliaMaterial::JuliaMaterial(const std::string &input_file) {
  // Make julia include the file that contains our function
  auto julia_string =
      std::string("include(\"") + input_file + std::string("\")");
  jl_eval_string(julia_string.c_str());
  // Load the function
  julia_func = jl_get_function(jl_current_module, "JLMAT");
  check_exception();
}

void JuliaMaterial::evaluate(std::vector<double> strain,
                             std::vector<double> &stress) {
  assert(strain.size() == 6);
  assert(stress.size() == 6);
  // Extract data
  auto strain_data = strain.data();
  auto stress_data = stress.data();
  // Make type
  jl_value_t *array_type = jl_apply_array_type(jl_float64_type, 1);
  // Wrap arrays
  jl_array_t *strain_arr = jl_ptr_to_array_1d(array_type, strain_data, 6, 0);
  jl_array_t *stress_arr = jl_ptr_to_array_1d(array_type, stress_data, 6, 0);

  jl_call2(julia_func, (jl_value_t *)strain_arr, (jl_value_t *)stress_arr);
  check_exception();
}
}
