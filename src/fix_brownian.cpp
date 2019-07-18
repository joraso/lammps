/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Joseph Raso $ Joel Eaves (CU Boulder)

   This fix will implement a Huen algorythm for overdamped brownian
   dynamics. (This is unlike the langevin fix, which does not handle
   the overdamped case.)

   Currently, this fix works with the verlet run_styl, in place of
   fix_nve or similar.
------------------------------------------------------------------------- */

#include <cstdio>
#include <cstring>
#include <cmath>
#include "fix_brownian.h"
#include "atom.h"
#include "atom_vec.h"
#include "neighbor.h"
#include "force.h"
#include "angle.h"
#include "dihedral.h"
#include "improper.h"
#include "kspace.h"
#include "update.h"
#include "error.h"
#include "random_mars.h"
#include "force.h"
#include "pair.h"
#include "bond.h"
#include "comm.h"
#include "timer.h"


using namespace LAMMPS_NS;
using namespace FixConst;

/* ----------------------------------------------------------------------
 Canonical functions
------------------------------------------------------------------------- */

FixBrownian::FixBrownian(LAMMPS * lmp, int narg, char **arg):
    Fix(lmp, narg, arg)
{
    // include error messages.
    if(narg < 6) {error->all(FLERR,"Illegal Fix Brownian command");};

    // parse arguments: Temp, drag, seed
    if(narg < 6) {error->all(FLERR,"Illegal Fix Brownian command");};
    Temp = atof(arg[3]);
    if(Temp <= 0.0) {error->all(FLERR,"Illegal Fix Brownian command");};
    drag = atof(arg[4]);
    if(drag < 0.0) {error->all(FLERR,"Illegal Fix Brownian command");};
    seed = atoi(arg[5]);
    if(seed <= 0) {error->all(FLERR,"Illegal Fix Brownian command");};

    // Create and seed the random number generator
    random = new RanMars(lmp,seed + comm->me);

    // Allocate save arrays

}

FixBrownian::~FixBrownian()
{
    // Destroy RNG
    delete random;

    // Destroy Save Arrays
}

int FixBrownian::setmask()
{
    int mask = 0;
    mask |= INITIAL_INTEGRATE;
    mask |= FINAL_INTEGRATE;
    return mask;
}

void FixBrownian::init()
{
    // is there more than one kind of
    dt_eff = update->dt / drag;

    // We have to retrieve a few flags for the force calculations
    if (force->pair && force->pair->compute_flag) pair_compute_flag = 1;
    else pair_compute_flag = 0;
    if (force->kspace && force->kspace->compute_flag) kspace_compute_flag = 1;
    else kspace_compute_flag = 0;

    // And for force-clearing
    // (currently does not support external force clearing via omp)
    torqueflag = extraflag = 0;
    if (atom->torque_flag) torqueflag = 1;
    if (atom->avec->forceclearflag) extraflag = 1;
}

void FixBrownian::initial_integrate(int /*vflag*/)
{

    double **x = atom->x;
    double **v = atom->v;
    double **f = atom->f;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    double (*eta)[3] = new double[nlocal][3];
    double v_f[3];
    if (igroup == atom->firstgroup) nlocal = atom->nfirst;

    for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
            // Draw random forces here
            eta[i][0] = random_force();
            eta[i][1] = random_force();
            eta[i][2] = random_force();
            // Set velocities
            v[i][0] = f[i][0] + eta[i][0];
            v[i][1] = f[i][1] + eta[i][0];
            v[i][2] = f[i][2] + eta[i][0];
            // update to virtual position
            x[i][0] += dt_eff * v[i][0];
            x[i][1] += dt_eff * v[i][1];
            x[i][2] += dt_eff * v[i][2];
        }
    }

    // recalculate the forces.
    force_recalculate();

    // update the velocities
    // take the real step
    for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
            // retrieve random forces?
            // calculate final velocities
            v_f[0] = 0.5 * (v[i][0] + f[i][0] + eta[i][0]);
            v_f[1] = 0.5 * (v[i][1] + f[i][1] + eta[i][1]);
            v_f[2] = 0.5 * (v[i][2] + f[i][2] + eta[i][2]);
            // Update to the real final position
            x[i][0] += dt_eff * (v_f[0] - v[i][0]);
            x[i][1] += dt_eff * (v_f[1] - v[i][1]);
            x[i][2] += dt_eff * (v_f[2] - v[i][2]);
            // properly store final velocities
            v[i][0] = v_f[0];
            v[i][1] = v_f[1];
            v[i][2] = v_f[2];
        }
    }

    // gotta clean up, don't want a memory leak...
    delete[] eta;
}

/* ----------------------------------------------------------------------
 Non-canonical functions
------------------------------------------------------------------------- */

inline double FixBrownian::random_force()
{
    return sqrt(2*Temp*dt_eff) * (random->uniform() - 0.5);
}

void FixBrownian::force_clear()
{
    // Clears the forces - this version is from copied Verlet->force_clear
    // there is a similar version in min->force_clear

    // if either newton flag is set, also include ghosts
    // when using threads always clear all forces.
    int nlocal = atom->nlocal;
    size_t nbytes;

    if (neighbor->includegroup == 0) {
        nbytes = sizeof(double) * nlocal;
        if (force->newton) nbytes += sizeof(double) * atom->nghost;

        if (nbytes) {
            memset(&atom->f[0][0],0,3*nbytes);
            if (torqueflag) memset(&atom->torque[0][0],0,3*nbytes);
            if (extraflag) atom->avec->force_clear(0,nbytes);
        }
    } else {
        // neighbor includegroup flag is set
        // clear force only on initial nfirst particles
        // if either newton flag is set, also include ghosts
        nbytes = sizeof(double) * atom->nfirst;
        if (nbytes) {
            memset(&atom->f[0][0],0,3*nbytes);
            if (torqueflag) memset(&atom->torque[0][0],0,3*nbytes);
            if (extraflag) atom->avec->force_clear(0,nbytes);
        }
        if (force->newton) {
            nbytes = sizeof(double) * atom->nghost;
            if (nbytes) {
                memset(&atom->f[nlocal][0],0,3*nbytes);
                if (torqueflag) memset(&atom->torque[nlocal][0],0,3*nbytes);
                if (extraflag) atom->avec->force_clear(nlocal,nbytes);
            }
        }
    }


}

void FixBrownian::force_recalculate()
{
    // recalculates the forces
    // mostly copied from the verlet integrator itself in Verlet->run()
    // but also compare to min->energy_force()

    // notably, this WILL NOT re-partition the atoms to different processors

    // Note: for the time being, both these flags are set to zero, meaning
    // neither the energy nor the pressure (virial) will be computed as
    // part of this recalculation.
    int eflag = 0;
    int vflag = 0;

    force_clear();

    timer->stamp();

    if (pair_compute_flag) {
        force->pair->compute(eflag,vflag);
        timer->stamp(Timer::PAIR);
    }

    if (atom->molecular) {
        if (force->bond) force->bond->compute(eflag,vflag);
        if (force->angle) force->angle->compute(eflag,vflag);
        if (force->dihedral) force->dihedral->compute(eflag,vflag);
        if (force->improper) force->improper->compute(eflag,vflag);
        timer->stamp(Timer::BOND);
    }

    if (kspace_compute_flag) {
        force->kspace->compute(eflag,vflag);
        timer->stamp(Timer::KSPACE);
    }

    if (force->newton) {
        comm->reverse_comm();
        timer->stamp(Timer::COMM);
    }
}

/* ---------------------------------------------------------------------- */

