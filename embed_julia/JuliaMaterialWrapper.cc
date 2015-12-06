//
// Created by kristoffer on 2015-12-04.
//

#include "JuliaMaterialWrapper.h"
#include <assert.h>
#include <vector>

namespace JuliaMaterialWrapper {
int JuliaMaterialWrapper::check_exception() {
  if (jl_exception_occurred()) {
    jl_show(jl_stderr_obj(), jl_exception_occurred());
    jl_printf(jl_stderr_stream(), "\n");
    return -1;
  }
  return 0;
}

JuliaMaterialWrapper::JuliaMaterialWrapper(const std::string &input_file,
                                           const int nparas, const int nstates)
    : nparas(nparas), nstates(nstates) {
  // Make julia include the file that contains our function
  auto julia_string =
      std::string("include(\"") + input_file + std::string("\")");
  jl_eval_string(julia_string.c_str());
  // Load the function
  julia_func = jl_get_function(jl_current_module, "JUMAT");
  assert(check_exception() == 0);
}

void JuliaMaterialWrapper::evaluate(std::array<double, 6> &strain,
                                    std::array<double, 10> &parameters,
                                    std::array<double, 21> &prev_matstat,
                                    std::array<double, 21> &matstat, double t,
                                    double dt) const {
#ifdef DEBUG
  assert(parameters.size() == nparas);
  assert(prev_matstat.size() == nstates);
  assert(matstat.size() == nstates);
#endif

  jl_value_t *array_type = jl_apply_array_type(jl_float64_type, 1);

  // Wrap the vectors into julia arrays
  jl_array_t *parameters_jl_arr =
      jl_ptr_to_array_1d(array_type, parameters.data(), nparas, 0);
  jl_array_t *prev_matstat_jl_arr =
      jl_ptr_to_array_1d(array_type, prev_matstat.data(), nstates, 0);
  jl_array_t *matstat_jl_arr =
      jl_ptr_to_array_1d(array_type, matstat.data(), nstates, 0);
  jl_array_t *strain_jl_arr =
      jl_ptr_to_array_1d(array_type, strain.data(), 6, 0);

  // Box the values
  auto *t_jl = jl_box_float64(t);
  auto *dt_jl = jl_box_float64(dt);

  std::vector<jl_value_t *> args{(jl_value_t *)strain_jl_arr,
                                 (jl_value_t *)parameters_jl_arr,
                                 (jl_value_t *)prev_matstat_jl_arr,
                                 (jl_value_t *)matstat_jl_arr, t_jl, dt_jl};
  jl_call(julia_func, args.data(), args.size());
  assert(check_exception() == 0);
}
} // end namespace JuliaMaterialWrapper
