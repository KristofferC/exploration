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
class JuliaMaterialWrapper {
public:
  JuliaMaterialWrapper(const std::string &input_file, const int nparas,
                       const int nstates);

  void evaluate(std::array<double, 6> &strain,
                std::array<double, 10> &parameters,
                std::array<double, 21> &prev_matstat,
                std::array<double, 21> &matstat, double t, double dt) const;

private:
  jl_function_t *julia_func;
  int nparas;
  int nstates;

  static int check_exception();
};
}
#endif // EMBED_JULIA_JULIAMATERIAL_H
