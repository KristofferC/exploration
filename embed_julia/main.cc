#include "./JuliaMaterial.h"

#include <iostream>
#include <vector>
#include <julia.h>

int main(int argc, char *argv[]) {
  /* required: setup the julia context */
  jl_init(JULIA_INIT_DIR);

  // Assuming main.jl is two levels over the build path...
  auto file_path = std::string("../../main.jl");
  auto jlmat = JuliaMaterial::JuliaMaterial(file_path);

  std::vector<double> strains = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  std::vector<double> stress(6);
  jlmat.evaluate(strains, stress);

  for (auto i: stress)
    std::cout << i << ",";

  int ret = 0;
  jl_atexit_hook(ret);
  return 0;
}
