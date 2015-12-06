//
// Created by kristoffer on 2015-12-05.
//

#include "vonMisesNonLinVisc.h"

namespace vonMisesNonLinVisc {
/**
 * A material is created from the path to the file
 * and a struct of the material parameters.
 * The states are not stored inside the class.
 */
vonMisesNonLinVisc::vonMisesNonLinVisc(const std::string filename,
                                       vonMisesNonLinViscMatPar matparameters)
    : matparameters(matparameters), julia_mat_wrap(filename) {
  // TODO: Maybe make this into a map..
  // Since the parameters are static we store them directly into
  // packed array.
  parameter_arr[0] = matparameters.E;
  parameter_arr[1] = matparameters.nu;
  parameter_arr[2] = matparameters.sig_y;
  parameter_arr[3] = matparameters.H;
  parameter_arr[4] = matparameters.r;
  parameter_arr[5] = matparameters.kap_inf;
  parameter_arr[6] = matparameters.alpha_inf;
  parameter_arr[7] = matparameters.tstar;
  parameter_arr[8] = matparameters.sig_c;
  parameter_arr[9] = matparameters.n;
}

/**
 * TODO: The following function would be better of
 * in the wrapper and instead we have some data structure
 * (some map) that gives (matstat_component -> start of full array)
*/

/**
 * Packs the state from the different vectors into one contiguous
 * vector that will be sent to the julia function
 */
void vonMisesNonLinVisc::pack_states(const vonMisesNonLinViscMatStat &matstat) {
  unsigned int idx = 0;
  for (auto strain : matstat.strain_p)
    prev_state_arr[idx++] = strain;
  for (auto stress : matstat.stress)
    prev_state_arr[idx++] = stress;
  for (auto alpha : matstat.alpha)
    prev_state_arr[idx++] = alpha;
  prev_state_arr[idx++] = matstat.kappa;
  prev_state_arr[idx++] = matstat.mu;
  prev_state_arr[idx++] = matstat.loading;
  assert(idx == prev_state_arr.size());
}

/**
 * Unpack the states from the big state vector back into the struct
 */
void vonMisesNonLinVisc::unpack_states(vonMisesNonLinViscMatStat &matstat) {
  int strain_p_idx = 0;
  int stress_p_idx = 0;
  int alpha_idx = 0;
  for (unsigned int i = 0; i < state_arr.size(); ++i) {
    if (i < VOIGT_SIZE)
      matstat.strain_p[strain_p_idx++] = state_arr[i];
    else if (i < 2 * VOIGT_SIZE)
      matstat.stress[stress_p_idx++] = state_arr[i];
    else if (i < 3 * VOIGT_SIZE)
      matstat.alpha[alpha_idx++] = state_arr[i];
    else if (i < 3 * VOIGT_SIZE + 1)
      matstat.kappa = state_arr[i];
    else if (i < 3 * VOIGT_SIZE + 2)
      matstat.mu = state_arr[i];
    else if (i < 3 * VOIGT_SIZE + 3)
      matstat.loading = state_arr[i];
  }
}

void vonMisesNonLinVisc::solve_tstep(std::array<double, 6> &strain,
                                     const vonMisesNonLinViscMatStat &prev_ms,
                                     vonMisesNonLinViscMatStat &ms,
                                     const double &t, const double &dt) {

  pack_states(prev_ms);
  julia_mat_wrap.evaluate(strain, parameter_arr, prev_state_arr, state_arr, t,
                          dt);
  unpack_states(ms);
}
} // end namespace