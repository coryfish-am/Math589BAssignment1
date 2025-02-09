#ifndef COMPUTE_ENERGY_H
#define COMPUTE_ENERGY_H

#ifdef __cplusplus
extern "C" {
#endif

/*
  compute_energy_and_gradient:

  positions: length = 3*n_beads (x0, y0, z0, x1, y1, z1, ...)
  n_beads:
  epsilon, sigma (LJ parameters)
  bond_len, k_bond (bond potential parameters)
  grad_out: length = 3*n_beads, will be filled in
  energy_out: a single double to be written
*/
void compute_energy_and_gradient(const double *positions,
                                 int n_beads,
                                 double epsilon,
                                 double sigma,
                                 double bond_len,
                                 double k_bond,
                                 double *grad_out,
                                 double *energy_out);

#ifdef __cplusplus
}
#endif

#endif
