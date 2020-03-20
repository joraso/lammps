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

#ifdef FIX_CLASS

FixStyle(brownian,FixBrownian)

#else

#ifndef LMPS_FIX_BROWNIAN_H
#define LMPS_FIX_BROWNIAN_H

#include "fix.h"

namespace LAMMPS_NS {

class FixBrownian : public Fix {
public:
    // Canonical functions
    FixBrownian(class LAMMPS *, int, char **);
    virtual ~FixBrownian();
    int setmask();
    virtual void init();
    virtual void initial_integrate(int);
    // Memory functions
    void grow_arrays(int);
    void copy_arrays(int,int,int);
    int pack_exchange(int, double*);
    int unpack_exchange(int, double*);
    double memory_usage();
    //void set_arrays(int);
protected:
    // Non-Canonical functions
    inline double random_force();
    void predictor();
    void corrector();
    int halfstepflag; // track whether to perform predictor or corrector.
    // Save Arrays
    int *fp_ind; // the f_previous atom property index
    int nmax_old; // length of peratom arrays the last time they grew
    // Fix variables
    double Temp, drag;
    int seed;
    double rand_prefactor, force_adjust, v_prefactor;
    class RanMars *random;
};

}

#endif
#endif
