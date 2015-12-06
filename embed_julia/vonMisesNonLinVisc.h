//
// Created by kristoffer on 2015-12-05.
//

#ifndef EMBED_JULIA_VONMISESNONLINVISC_H
#define EMBED_JULIA_VONMISESNONLINVISC_H

#include "JuliaMaterialWrapper.h"

#include <string>
#include <array>

struct vonMisesNonLinViscMatPar {
  const double E;
  const double nu;
  const double sig_y;
  const double H;
  const double r;
  const double kap_inf;
  const double alpha_inf;
  const double tstar;
  const double sig_c;
  const double n;
};

struct vonMisesNonLinViscMatStat {
  std::array<double, 6> strain_p;
  std::array<double, 6> stress;
  std::array<double, 6> alpha;
  double kappa;
  double mu;
  bool loading;
};

class vonMisesNonLinVisc {
public:
  explicit vonMisesNonLinVisc(const std::string filename,
                              const vonMisesNonLinViscMatPar matparameters);

  const vonMisesNonLinViscMatPar matparameters;

  void solve_tstep(std::array<double, 6> &strain,
                   const vonMisesNonLinViscMatStat &oldms,
                   vonMisesNonLinViscMatStat &newms, const double &t,
                   const double &dt);

private:
  void pack_states(const vonMisesNonLinViscMatStat &matstat);
  void unpack_states(vonMisesNonLinViscMatStat &matstat);
  const JuliaMaterialWrapper::JuliaMaterialWrapper julia_mat_wrap;
  std::array<double, 10> parameter_arr;
  std::array<double, 21> state_arr;
  std::array<double, 21> prev_state_arr;
};

#endif // EMBED_JULIA_VONMISESNONLINVISC_H
