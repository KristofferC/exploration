//
// Created by kristoffer on 2015-12-04.
//

#ifndef EMBED_JULIA_JULIAMATERIAL_H
#define EMBED_JULIA_JULIAMATERIAL_H

#include <string>
#include <vector>
#include <array>

#include <julia.h>

namespace JuliaMaterialWrapper {
template <int nstates, int nparas> class JuliaMaterialWrapper {

public:
  JuliaMaterialWrapper(const std::string &input_file);

  void evaluate(std::array<double, 6> &strain,
                std::array<double, nparas> &parameters,
                std::array<double, nstates> &prev_matstat,
                std::array<double, nstates> &matstat, double t,
                double dt) const;

private:
  jl_function_t *julia_func;

  static int check_exception();
};

template <int nstates, int nparas>
int JuliaMaterialWrapper<nstates, nparas>::check_exception() {
  if (jl_exception_occurred()) {
    jl_show(jl_stderr_obj(), jl_exception_occurred());
    jl_printf(jl_stderr_stream(), "\n");
    return -1;
  }
  return 0;
}

template <int nstates, int nparas>
JuliaMaterialWrapper<nstates, nparas>::JuliaMaterialWrapper(
    const std::string &input_file) {
  // Make julia include the file that contains our function
  auto julia_string =
      std::string("include(\"") + input_file + std::string("\")");
  jl_eval_string(julia_string.c_str());
  // Load the function
  julia_func = jl_get_function(jl_current_module, "JUMAT");
  assert(check_exception() == 0);
}
template <int nstates, int nparas>
void JuliaMaterialWrapper<nstates, nparas>::evaluate(
    std::array<double, 6> &strain, std::array<double, nparas> &parameters,
    std::array<double, nstates> &prev_matstat,
    std::array<double, nstates> &matstat, double t, double dt) const {
#ifdef DEBUG
  assert(parameters.size() == nparas);
  assert(prev_matstat.size() == nstates);
  assert(matstat.size() == nstates);
#endif

  jl_value_t *array_type = jl_apply_array_type(jl_float64_type, 1);

  // Wrap the vectors into julia arrays
  jl_array_t *strain_jl_arr =
      jl_ptr_to_array_1d(array_type, strain.data(), 6, 0);
  jl_array_t *parameters_jl_arr =
      jl_ptr_to_array_1d(array_type, parameters.data(), nparas, 0);
  jl_array_t *prev_matstat_jl_arr =
      jl_ptr_to_array_1d(array_type, prev_matstat.data(), nstates, 0);
  jl_array_t *matstat_jl_arr =
      jl_ptr_to_array_1d(array_type, matstat.data(), nstates, 0);

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
}
#endif // EMBED_JULIA_JULIAMATERIAL_H
