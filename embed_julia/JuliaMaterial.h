//
// Created by kristoffer on 2015-12-04.
//

#ifndef EMBED_JULIA_JULIAMATERIAL_H
#define EMBED_JULIA_JULIAMATERIAL_H

#include <string>
#include <vector>
#include <julia.h>

namespace JuliaMaterial {
class JuliaMaterial {
public:
  explicit JuliaMaterial(const std::string &input_file);

  void evaluate(std::vector<double> strain, std::vector<double> &stress);

private:
  jl_function_t *julia_func;

  void check_exception();
};
}
#endif // EMBED_JULIA_JULIAMATERIAL_H
