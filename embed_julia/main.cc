#include "vonMisesNonLinVisc.h"

#include <iostream>
#include <array>
#include <julia.h>

int main(int argc, char *argv[]) {
  /* required: setup the julia context */
  jl_init(JULIA_INIT_DIR);
  /* include the .juliarc.jl file to load the stuff in there */
  jl_eval_string("include(homedir() * \"/.juliarc.jl\")");

  /*
   * Material parameters
   */
  double E = 200.0e+09;
  double nu = 0.3;
  double sigy = 200.0e+06;
  double H = 0.1 * E;
  double kappa_inf = 0.2 * sigy;
  double alpha_inf = 0.1 * sigy;
  double r = 0.5;
  double sigc = 0.1 * E;
  double tstar = 100.0;
  double n = 1.5;

  vonMisesNonLinVisc::vonMisesNonLinViscMatPar mp = {
      E, nu, sigy, H, r, kappa_inf, alpha_inf, tstar, sigc, n};

  /*
   * Create the material
   */
  /* Assuming main.jl is two levels over the build path... */
  auto file_path = std::string("../../mises_mixed.jl");
  auto vmnonlinmat = vonMisesNonLinVisc::vonMisesNonLinVisc(file_path, mp);

  /*
   * Test it with some reasonable data
   */
  /* Previous converged material status */
  std::array<double, 6> strain_p = {0.001942341756956, -0.000971170878478,
                                    -0.000971170878478, 0, 0, 0};

  std::array<double, 6> stress = {1.047332037391335e+09, 0.726333981304334e+09,
                                  0.726333981304334e+09, 0, 0, 0};

  std::array<double, 6> alpha = {1.024808728562482e+07, -0.512404364281240e+07,
                                 -0.512404364281240e+07, 0, 0, 0};
  double kap = 1.240950920857154e+07;
  double mu = 1.060650627081043e-05;
  vonMisesNonLinVisc::vonMisesNonLinViscMatStat prev_ms = {strain_p, stress,
                                                           alpha, kap, mu};
  vonMisesNonLinVisc::vonMisesNonLinViscMatStat ms = {};

  std::array<double, 6> strain = {0.005016666666667, 0.0, 0.0, 0.0, 0.0, 0.0};

  double t = 100.0;
  double dt = 3.3333333333333;

  std::cout << "old plastic strain:" << std::endl;
  for (auto item : prev_ms.strain_p)
    std::cout << item << ", ";
  std::cout << std::endl;

  // Solve for this time step
  vmnonlinmat.solve_tstep(strain, prev_ms, ms, t, dt);

  std::cout << "new plastic strain:" << std::endl;
  for (auto item : ms.strain_p)
    std::cout << item << ", ";
  std::cout << std::endl;

  int ret = 0;
  jl_atexit_hook(ret);
  return 0;
}
