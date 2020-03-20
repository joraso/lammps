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
   Contributing authors: Joseph Raso & Joel Eaves (CU Boulder)

   This fix will implement a Huen algorythm for overdamped brownian
   dynamics. (This is unlike the langevin fix, which does not handle
   the overdamped case.)

   Currently, this fix works in conjuction with fix nve (which must be invoked
   after it.)
------------------------------------------------------------------------- */

#include <iostream> // yeah yeah, remove later etc.

#include <cstdio>
#include <cstring>
#include <cmath>
#include <memory>
#include "memory.h"
#include "fix_brownian.h"
#include "atom.h"
#include "atom_vec.h"
/*
#include "neighbor.h"
#include "force.h"
#include "angle.h"
#include "dihedral.h"
#include "improper.h"
#include "kspace.h"
*/
#include "update.h"
#include "error.h"
#include "random_mars.h"
#include "force.h"
/*
#include "pair.h"
#include "bond.h"
*/
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
    fp_ind = new int[3];

    for (int k=0; k<3; k++){
        fp_ind[k] = atom->add_custom("fprev",1);
    }
    
    nmax_old = 0; //setting initial array sizes to 0.
    if (!lmp->kokkos) grow_arrays(atom->nmax); //init arrays
    atom->add_callback(0);
    
    
}

FixBrownian::~FixBrownian()
{
    // Delete Arrays
    for (int k=0; k<3; k++){
        atom->remove_custom(1,fp_ind[k]);
    }
    atom->delete_callback(id,0);
    delete [] fp_ind;
    
    // Destroy RNG
    //if (random == nullptr) return; // Doesn't work on enemy, requires C++11
    delete random;
}

int FixBrownian::setmask()
{
    int mask = 0;
    mask |= INITIAL_INTEGRATE;
    return mask;
}

void FixBrownian::init()
{

    // Calculate prefactor on the random force
    rand_prefactor = sqrt((2*Temp)/(update->dt * drag));
    
    // Calculate adjustment to LJ forces
    force_adjust = 1 / (update->dt * drag * force->ftm2v);
    
    // Another prefactor used in the correction step
    v_prefactor = 0.5 / drag;
    
    // Store initial values into f_previous.
    //Should not be neccessary, but just in case.
    double **f = atom->f;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    if (igroup == atom->firstgroup) nlocal = atom->nfirst;
    for (int i = 0; i < nlocal; i++) {
        if (mask[i] & groupbit) {
            for (int k=0; k<3; k++){
                atom->dvector[fp_ind[k]][i] = f[i][k];
            }
        }
    }
    
    // Finally (redundantly), ensure the intigrator begins with a
    // predictor step.
    halfstepflag = 0;
}

void FixBrownian::initial_integrate(int /*vflag*/)
{

    switch (halfstepflag) {
        case 0:
            predictor();
            halfstepflag = 1;
            break;
        case 1:
            corrector();
            halfstepflag = 0;
            break;
    }
    
    
}

/* ----------------------------------------------------------------------
 Non-canonical functions
------------------------------------------------------------------------- */

inline double FixBrownian::random_force()
{
    return rand_prefactor * random->gaussian();
}

void FixBrownian::predictor(){
    double **v = atom->v;
    double **f = atom->f;
    double *rmass = atom->rmass;
    double *mass = atom->mass;
    int *type = atom->type;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    double f_mass_adj;

    if (igroup == atom->firstgroup) nlocal = atom->nfirst;
    
    if (rmass) {
        for (int i = 0; i < nlocal; i++) {
            if (mask[i] & groupbit) {
                f_mass_adj = force_adjust * rmass[i];
                for (int k=0; k<3; k++){
                    atom->dvector[fp_ind[k]][i] = f[i][k];
                    v[i][k] = random_force();
                    f[i][k] *= 2 * f_mass_adj;
                    
                }
            }
        }
    } else {
        for (int i = 0; i < nlocal; i++) {
            if (mask[i] & groupbit) {
                f_mass_adj = force_adjust * mass[type[i]];
                for (int k=0; k<3; k++){
                    atom->dvector[fp_ind[k]][i] = f[i][k];
                    v[i][k] = random_force();
                    f[i][k] *= 2 * f_mass_adj;
                    
                }
            }
        }
    }
    
}

void FixBrownian::corrector(){
    double **v = atom->v;
    double **f = atom->f;
    double *rmass = atom->rmass;
    double *mass = atom->mass;
    int *type = atom->type;
    int *mask = atom->mask;
    int nlocal = atom->nlocal;
    double f_mass_adj;
    
    if (igroup == atom->firstgroup) nlocal = atom->nfirst;
    
    if (rmass) {
        for (int i = 0; i < nlocal; i++) {
            if (mask[i] & groupbit) {
                f_mass_adj = force_adjust * rmass[i];
                for (int k=0; k<3; k++){
                    v[i][k] = -(v_prefactor * atom->dvector[fp_ind[k]][i]);
                    f[i][k] *= force_adjust;
                }
            }
        }
    } else {
        for (int i = 0; i < nlocal; i++) {
            if (mask[i] & groupbit) {
            f_mass_adj = force_adjust * mass[type[i]];
                for (int k=0; k<3; k++){
                    v[i][k] = -(v_prefactor * atom->dvector[fp_ind[k]][i]);
                    f[i][k] *= force_adjust;
                }
            }
        }
    }
}

/* ----------------------------------------------------------------------
memory manipulation functions
------------------------------------------------------------------------- */

void FixBrownian::grow_arrays(int nmax)
{
    //memory->grow(this->x_previous,nmax,3,"fix_brownian:x_previous");
    //memory->grow(this->f_random,nmax,3,"fix_brownian:f_random");
    size_t nbytes = (nmax-nmax_old) * sizeof(double);
    for (int k=0; k<3; k++){
        memory->grow(atom->dvector[fp_ind[k]],nmax,"atom:dvector");
        memset(&atom->dvector[fp_ind[k]][nmax_old],0,nbytes);
    }
    nmax_old = nmax;
}

void FixBrownian::copy_arrays(int i, int j, int /*delflag*/)
{
    for (int k=0; k<3; k++){
        atom->dvector[fp_ind[k]][j] = atom->dvector[fp_ind[k]][i];
    }
}

int FixBrownian::pack_exchange(int i, double *buf)
{
    int m = 0;
    for (int k=0; k<3; k++){
        buf[m++] = atom->dvector[fp_ind[k]][i];
    }
    return m;
}

int FixBrownian::unpack_exchange(int nlocal, double *buf)
{
    int m = 0;
    for (int k=0; k<3; k++){
        atom->dvector[fp_ind[k]][nlocal] = buf[m++];
    }
    return m;
}

double FixBrownian::memory_usage() {
    int nmax = atom->nmax;
    double bytes = 0.0;
    bytes = 3*nmax*sizeof(double);
    return bytes;
}

/* ---------------------------------------------------------------------- */

