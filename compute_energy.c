#include <math.h>
#include <stdio.h>
#include "compute_energy.h"

/*
  Lennard-Jones: U_LJ(r) = 4*epsilon * [ (sigma/r)^12 - (sigma/r)^6 ]
  Bond potential: U_bond(r) = k_b * (r - b)^2

  We also compute partial derivatives (gradient) for each dimension (x,y,z).

  For example, for Lennard-Jones between beads i and j:

    U_LJ'(r) = derivative wrt r = 4*epsilon [12*sigma^12/r^13 - 6*sigma^6/r^7]
    Then chain rule for partial wrt x_i (or y_i, z_i) given:
       partial/partial x_i = U'(r) * (x_i - x_j)/r
*/

static double bond_potential(double r, double b, double k_b) {
    double diff = (r - b);
    return k_b * diff * diff;
}

static double lennard_jones_potential(double r, double epsilon, double sigma) {
    if (r < 1.0e-12)
        r = 1.0e-12; // avoid blow-up or /0
    double sr = sigma / r;
    double sr2 = sr * sr;      // (sigma/r)^2
    double sr6 = sr2 * sr2 * sr2;
    double sr12 = sr6 * sr6;
    return 4.0 * epsilon * (sr12 - sr6);
}

// Derivative of Lennard-Jones w.r.t r
static double dUdr_lj(double r, double epsilon, double sigma) {
    // U'(r) = derivative wrt r of 4*epsilon[(sigma/r)^12 - (sigma/r)^6]
    //        = 4*epsilon[ -12*sigma^12/r^13 + 6*sigma^6/r^7 ]
    // for numerical stability, we clamp r as well:
    if (r < 1.0e-12)
        r = 1.0e-12;
    double sr = sigma / r;
    double sr2 = sr * sr;
    double sr6 = sr2 * sr2 * sr2;
    double sr12 = sr6 * sr6;
    // dU/dr
    return 4.0 * epsilon * ( -12.0*sr12 + 6.0*sr6 ) / r;
}

// Derivative of bond potential w.r.t. r
static double dUdr_bond(double r, double b, double k_b) {
    // derivative of k_b*(r - b)^2 wrt r is 2*k_b*(r - b)
    return 2.0*k_b*(r - b);
}

void compute_energy_and_gradient(const double *positions,
                                 int n_beads,
                                 double epsilon,
                                 double sigma,
                                 double bond_len,
                                 double k_bond,
                                 double *grad_out,
                                 double *energy_out)
{
    // positions: array of length 3*n_beads
    // grad_out:  array of length 3*n_beads
    // We'll zero out the gradient first
    for(int i=0; i<3*n_beads; i++){
        grad_out[i] = 0.0;
    }

    double U = 0.0;

    // Bonded interactions (i.e. i and i+1)
    for(int i=0; i<(n_beads - 1); i++) {
        int i3 = 3*i;
        int i3_next = 3*(i+1);

        double dx = positions[i3+0] - positions[i3_next+0];
        double dy = positions[i3+1] - positions[i3_next+1];
        double dz = positions[i3+2] - positions[i3_next+2];
        double r = sqrt(dx*dx + dy*dy + dz*dz);

        // bond potential
        double Ubond = bond_potential(r, bond_len, k_bond);
        U += Ubond;

        // derivative wrt r
        double dUdr = dUdr_bond(r, bond_len, k_bond);

        // apply chain rule to x,y,z
        double inv_r = (r < 1.0e-12) ? 0.0 : 1.0 / r;
        double fx = dUdr * dx * inv_r;  // force in x direction
        double fy = dUdr * dy * inv_r;
        double fz = dUdr * dz * inv_r;

        // The gradient is negative of the force, but we can be consistent as long as we
        // do the partial derivative of the potential wrt x_i (which is + dUdr*(dx/r)).
        // Typically grad_i = dU/dx_i. Actually, from the perspective of positions i vs j:
        //
        // For the i-th bead:
        //   partial U/partial x_i = dUdr * (x_i - x_j)/r
        // For the j-th bead:
        //   partial U/partial x_j = dUdr * (x_j - x_i)/r = - partial U/partial x_i
        //
        // We'll do i +=, j -=
        grad_out[i3+0]   += fx;
        grad_out[i3+1]   += fy;
        grad_out[i3+2]   += fz;

        grad_out[i3_next+0] -= fx;
        grad_out[i3_next+1] -= fy;
        grad_out[i3_next+2] -= fz;
    }

    // Lennard-Jones for i < j
    // O(n^2) approach, fine for up to a few thousands of beads
    for(int i=0; i<n_beads; i++){
        int i3 = 3*i;
        for(int j=i+1; j<n_beads; j++){
            int j3 = 3*j;
            double dx = positions[i3+0] - positions[j3+0];
            double dy = positions[i3+1] - positions[j3+1];
            double dz = positions[i3+2] - positions[j3+2];
            double r2 = dx*dx + dy*dy + dz*dz;
            if(r2 < 1.0e-12) {
                // very close, to avoid blow up, ignore or set minimal distance
                r2 = 1.0e-12;
            }
            double r = sqrt(r2);

            // accumulate Lennard-Jones energy
            double Ulj = lennard_jones_potential(r, epsilon, sigma);
            U += Ulj;

            // derivative wrt r
            double dUdr = dUdr_lj(r, epsilon, sigma);

            // chain rule for partial wrt x_i, x_j
            double inv_r = 1.0 / r;
            double fx = dUdr * dx * inv_r;
            double fy = dUdr * dy * inv_r;
            double fz = dUdr * dz * inv_r;

            // add to i, subtract from j
            grad_out[i3+0] += fx;
            grad_out[i3+1] += fy;
            grad_out[i3+2] += fz;

            grad_out[j3+0] -= fx;
            grad_out[j3+1] -= fy;
            grad_out[j3+2] -= fz;
        }
    }

    // Write final energy
    *energy_out = U;
}
