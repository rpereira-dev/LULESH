/*
  This is a Version 2.0 MPI + OpenMP implementation of LULESH

                 Copyright (c) 2010-2013.
      Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory.
                  LLNL-CODE-461231
                All rights reserved.

This file is part of LULESH, Version 2.0.
Please also read this link -- http://www.opensource.org/licenses/index.php

//////////////
DIFFERENCES BETWEEN THIS VERSION (2.x) AND EARLIER VERSIONS:
* Addition of regions to make work more representative of multi-material codes
* Default size of each domain is 30^3 (27000 elem) instead of 45^3. This is
  more representative of our actual working set sizes
* Single source distribution supports pure serial, pure OpenMP, MPI-only,
  and MPI+OpenMP
* Addition of ability to visualize the mesh using VisIt
  https://wci.llnl.gov/codes/visit/download.html
* Various command line options (see ./lulesh2.0 -h)
 -q              : quiet mode - suppress stdout
 -i <iterations> : number of cycles to run
 -s <size>       : length of cube mesh along side
 -r <numregions> : Number of distinct regions (def: 11)
 -b <balance>    : Load balance between regions of a domain (def: 1)
 -c <cost>       : Extra cost of more expensive regions (def: 1)
 -f <filepieces> : Number of file parts for viz output (def: np/9)
 -p              : Print out progress
 -v              : Output viz file (requires compiling with -DVIZ_MESH
 -h              : This message

 printf("Usage: %s [opts]\n", execname);
      printf(" where [opts] is one or more of:\n");
      printf(" -q              : quiet mode - suppress all stdout\n");
      printf(" -i <iterations> : number of cycles to run\n");
      printf(" -s <size>       : length of cube mesh along side\n");
      printf(" -r <numregions> : Number of distinct regions (def: 11)\n");
      printf(" -b <balance>    : Load balance between regions of a domain (def: 1)\n");
      printf(" -c <cost>       : Extra cost of more expensive regions (def: 1)\n");
      printf(" -f <numfiles>   : Number of files to split viz dump into (def: (np+10)/9)\n");
      printf(" -p              : Print out progress\n");
      printf(" -v              : Output viz file (requires compiling with -DVIZ_MESH\n");
      printf(" -h              : This message\n");
      printf("\n\n");

*Notable changes in LULESH 2.0

* Split functionality into different files
lulesh.cc - where most (all?) of the timed functionality lies
lulesh-comm.cc - MPI functionality
lulesh-init.cc - Setup code
lulesh-viz.cc  - Support for visualization option
lulesh-util.cc - Non-timed functions
*
* The concept of "regions" was added, although every region is the same ideal
*    gas material, and the same sedov blast wave problem is still the only
*    problem its hardcoded to solve.
* Regions allow two things important to making this proxy app more representative:
*   Four of the LULESH routines are now performed on a region-by-region basis,
*     making the memory access patterns non-unit stride
*   Artificial load imbalances can be easily introduced that could impact
*     parallelization strategies.
* The load balance flag changes region assignment.  Region number is raised to
*   the power entered for assignment probability.  Most likely regions changes
*   with MPI process id.
* The cost flag raises the cost of ~45% of the regions to evaluate EOS by the
*   entered multiple. The cost of 5% is 10x the entered multiple.
* MPI and OpenMP were added, and coalesced into a single version of the source
*   that can support serial builds, MPI-only, OpenMP-only, and MPI+OpenMP
* Added support to write plot files using "poor mans parallel I/O" when linked
*   with the silo library, which in turn can be read by VisIt.
* Enabled variable timestep calculation by default (courant condition), which
*   results in an additional reduction.
* Default domain (mesh) size reduced from 45^3 to 30^3
* Command line options to allow numerous test cases without needing to recompile
* Performance optimizations and code cleanup beyond LULESH 1.0
* Added a "Figure of Merit" calculation (elements solved per microsecond) and
*   output in support of using LULESH 2.0 for the 2017 CORAL procurement
*
* Possible Differences in Final Release (other changes possible)
*
* High Level mesh structure to allow data structure transformations
* Different default parameters
* Minor code performance changes and cleanup

TODO in future versions
* Add reader for (truly) unstructured meshes, probably serial only
* CMake based build system

//////////////

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

   * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the disclaimer below.

   * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the disclaimer (as noted below)
     in the documentation and/or other materials provided with the
     distribution.

   * Neither the name of the LLNS/LLNL nor the names of its contributors
     may be used to endorse or promote products derived from this software
     without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC,
THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Additional BSD Notice

1. This notice is required to be provided under our contract with the U.S.
   Department of Energy (DOE). This work was produced at Lawrence Livermore
   National Laboratory under Contract No. DE-AC52-07NA27344 with the DOE.

2. Neither the United States Government nor Lawrence Livermore National
   Security, LLC nor any of their employees, makes any warranty, express
   or implied, or assumes any liability or responsibility for the accuracy,
   completeness, or usefulness of any information, apparatus, product, or
   process disclosed, or represents that its use would not infringe
   privately-owned rights.

3. Also, reference herein to any specific commercial products, process, or
   services by trade name, trademark, manufacturer or otherwise does not
   necessarily constitute or imply its endorsement, recommendation, or
   favoring by the United States Government or Lawrence Livermore National
   Security, LLC. The views and opinions of authors expressed herein do not
   necessarily state or reflect those of the United States Government or
   Lawrence Livermore National Security, LLC, and shall not be used for
   advertising or product endorsement purposes.

*/

#include <climits>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <unistd.h>

#if _OPENMP
# include <omp.h>
# include <mpc_omp.h>
# include <map>
#endif

#if USE_MPI
# include <mpi.h>
#endif

# if USE_MPI && USE_OMPI
#  define TASK_SHARED stdout, ompi_mpi_comm_world, ompi_mpi_op_min, ompi_mpi_float, ompi_mpi_double
# else
#  define TASK_SHARED stdout
# endif /* USE_OMPI */

#include "lulesh.h"
#include "lulesh-trace.h"

/* BLOCK SIZES */
static Index_t EBS = -1;
static Index_t NBS = -1;

/* set to 1 if measuring runtime dependency hash map maximum occupation */
# define MEASURE_HASH_OCCUPATION 1

//##############################################################################
// CalcMonotonicQGradientsForElems
static const Real_t ptiny = Real_t(1.e-36) ;

// MPI ranks
Int_t numRanks ;
Int_t myRank ;
struct cmdLineOpts opts;

// IntregrateStressTerms and CalcFBHourglassForceForElems
Real_t *fx_elem, *fy_elem, *fz_elem;
Real_t *fx_elem_FBH, *fy_elem_FBH, *fz_elem_FBH;
Real_t gamma_v[4][8];
Real_t * dvdx, * dvdy, * dvdz;
Real_t * x8n, * y8n, * z8n;

// CalcVolumeForceForElems
Real_t *sigxx, *sigyy, *sigzz, *determ;

// LagrangeElements
Real_t *vnew;

// CalcEnergyForElems
Real_t **pHalfStepRegs;

// EvalEOSForElems
Real_t **e_oldRegs;
Real_t **delvcRegs;
Real_t **p_oldRegs;
Real_t **q_oldRegs;
Real_t **compressionRegs;
Real_t **compHalfStepRegs;
Real_t **qq_oldRegs;
Real_t **ql_oldRegs;
Real_t **workRegs;
Real_t **p_newRegs;
Real_t **e_newRegs;
Real_t **q_newRegs;
Real_t **bvcRegs;
Real_t **pbvcRegs;
Real_t **regRep;

// current iteration during graph generation
int iter = -1;

// CalcTimeConstraintsForElems
typedef struct
{
    Real_t value;
    Index_t index;
} dt_reduction_t;

dt_reduction_t ** dt_reduction_courant;
dt_reduction_t ** dt_reduction_hydro;

/******************************************/

/* task dependencies */
/**
 * NOTE
 *
 * This implementation performances is intentionnally poor.
 * This code follow the guidelines from LULESH 2.0 report,
 * which stipulates not to modify the loop structures.
 *
 * This implementation also may generate redundant arcs between tasks.
 * The graph generation could be lightened by removing these arcs.
 * For instance in 'CalcMonotonicQRegionForElems_deps', tasks consumes 'delx' and 'delv',
 * but actually, only 1 task produce both 'delx' and 'delv' for a given block.
 * So 'CalcMonotonicQRegionForElems_deps' tasks could only consume 'delx' OR 'delv',
 * and the right order of execution would be preserved.
 * Though, we intentionally kept those extra arcs, since we assume this is how
 * it would be implemented in a real-world application.
 *
 * From our experiments of fusing loops and optimizing the graph,
 * we saw up to 50% speed gain on graph generation, and ~40% of overall performance gain.
 */

static mpc_omp_task_dependency_t * dependencies_domain_x_y_z;
static mpc_omp_task_dependency_t * dependencies_fx_fy_fz_elem;

static mpc_omp_task_dependency_t * dependencies_domain_xd_yd_zd;
static mpc_omp_task_dependency_t * dependencies_fx_fy_fz_elem_FBH;

static mpc_omp_task_dependency_t * dependencies_bc_xdd;
static mpc_omp_task_dependency_t * dependencies_bc_ydd;
static mpc_omp_task_dependency_t * dependencies_bc_zdd;

static mpc_omp_task_dependency_t ** dt_courant_deps;
static mpc_omp_task_dependency_t ** dt_hydro_deps;

static mpc_omp_task_dependency_t ** CalcMonotonicQRegionForElems_deps;

static mpc_omp_task_dependency_t ** EvalEOSForElems_deps_1;
static mpc_omp_task_dependency_t ** EvalEOSForElems_deps_2;
static mpc_omp_task_dependency_t ** vnew_in_deps;

static mpc_omp_task_dependency_t ** CalcSoundSpeedForElems_deps;

# define MPC_PUSH_DEPENDENCIES(...) do {                                            \
                                        if (!opts.persistentTasks || iter == 0)     \
                                            mpc_omp_task_dependencies(__VA_ARGS__); \
                                    } while (0)

# define PRUNE_INOUTSET_EXPLICITLY 0
# define TASK_COMPUTE(...)
# define TASK_COMM(...)

/**********************************************/
# include <atomic>
static std::atomic<bool> cancelled;

/**********************************************/
/* Work Routines */
static
void TimeDump(Domain * domain)
{
    TASK_SET_COLOR(iter);
    TASK_SET_LABEL("TimeDump");
    # pragma omp task default(none)                             \
        firstprivate(domain)                                    \
        shared(numRanks, opts, cancelled, myRank, TASK_SHARED)  \
        depend(in: domain->m_deltatime)
    {
        if ((opts.showProg != 0) && (opts.quiet == 0) && (myRank == 0))
        {
            printf("cycle = %d, time = %e, dt=%e\n", domain->cycle(), domain->time(), domain->deltatime());
            fflush(stdout);
        }

        if (domain->time() >= domain->stoptime())
        {
            // TODO : check that every tasks of the next iterations depends on this one,
            // to avoid corrupting memory by pre-computing the next iteration
            puts("cancelling!");
            cancelled = 1;
            assert(numRanks == 1); // MPI cancelling isnt trivial, not working
            # pragma omp cancel taskgroup
        }
    }
}

static
void TimeIncrement(Domain * domain)
{
    TASK_SET_COLOR(iter - 1);
    TASK_SET_LABEL("TimeIncrement");
#if USE_MPI
    mpc_omp_task_fiber();
    # pragma omp task default(none)                             \
        shared(TASK_SHARED)                                     \
        firstprivate(domain)                                    \
        depend(in:      domain->m_dtcourant, domain->m_dthydro) \
        depend(inout:   domain->m_deltatime)                    \
        untied                                                  \
        priority(PRIORITY_REDUCE)
#else
    # pragma omp task default(none)                             \
        firstprivate(domain)                                    \
        depend(in:      domain->m_dtcourant, domain->m_dthydro) \
        depend(inout:   domain->m_deltatime)                    \
        priority(PRIORITY_REDUCE)
#endif
    {
        Real_t targetdt = domain->stoptime() - domain->time();

        if ((domain->dtfixed() <= Real_t(0.0)) && (domain->cycle() != Int_t(0))) {
            Real_t ratio ;
            Real_t olddt = domain->deltatime() ;

            /* This will require a reduction in parallel */
            Real_t gnewdt = Real_t(1.0e+20) ;
            Real_t newdt ;
            if (domain->dtcourant() < gnewdt) {
                gnewdt = domain->dtcourant() / Real_t(2.0) ;
            }
            if (domain->dthydro() < gnewdt) {
                gnewdt = domain->dthydro() * Real_t(2.0) / Real_t(3.0) ;
            }

#if USE_MPI
            CommReduceDt(&gnewdt, &newdt);
#else
            newdt = gnewdt;
#endif
            ratio = newdt / olddt ;
            if (ratio >= Real_t(1.0)) {
                if (ratio < domain->deltatimemultlb()) {
                    newdt = olddt ;
                }
                else if (ratio > domain->deltatimemultub()) {
                    newdt = olddt* domain->deltatimemultub() ;
                }
            }

            if (newdt > domain->dtmax()) {
                newdt = domain->dtmax() ;
            }
            domain->deltatime() = newdt ;
        }

        /* TRY TO PREVENT VERY SMALL SCALING ON THE NEXT CYCLE */
        if ((targetdt > domain->deltatime()) &&
                (targetdt < (Real_t(4.0) * domain->deltatime() / Real_t(3.0))) ) {
            targetdt = Real_t(2.0) * domain->deltatime() / Real_t(3.0) ;
        }

        if (targetdt < domain->deltatime()) {
            domain->deltatime() = targetdt ;
        }

        domain->time() += domain->deltatime() ;

        ++domain->cycle() ;
    }
}

/******************************************/

static
void CollectDomainNodesToElemNodes(Domain * domain,
        const Index_t* elemToNode,
        Real_t elemX[8],
        Real_t elemY[8],
        Real_t elemZ[8])
{
    Index_t nd0i = elemToNode[0] ;
    Index_t nd1i = elemToNode[1] ;
    Index_t nd2i = elemToNode[2] ;
    Index_t nd3i = elemToNode[3] ;
    Index_t nd4i = elemToNode[4] ;
    Index_t nd5i = elemToNode[5] ;
    Index_t nd6i = elemToNode[6] ;
    Index_t nd7i = elemToNode[7] ;

    elemX[0] = domain->x(nd0i);
    elemX[1] = domain->x(nd1i);
    elemX[2] = domain->x(nd2i);
    elemX[3] = domain->x(nd3i);
    elemX[4] = domain->x(nd4i);
    elemX[5] = domain->x(nd5i);
    elemX[6] = domain->x(nd6i);
    elemX[7] = domain->x(nd7i);

    elemY[0] = domain->y(nd0i);
    elemY[1] = domain->y(nd1i);
    elemY[2] = domain->y(nd2i);
    elemY[3] = domain->y(nd3i);
    elemY[4] = domain->y(nd4i);
    elemY[5] = domain->y(nd5i);
    elemY[6] = domain->y(nd6i);
    elemY[7] = domain->y(nd7i);

    elemZ[0] = domain->z(nd0i);
    elemZ[1] = domain->z(nd1i);
    elemZ[2] = domain->z(nd2i);
    elemZ[3] = domain->z(nd3i);
    elemZ[4] = domain->z(nd4i);
    elemZ[5] = domain->z(nd5i);
    elemZ[6] = domain->z(nd6i);
    elemZ[7] = domain->z(nd7i);
}

/******************************************/

static inline
void InitStressTermsForElems(Domain * domain,
        Real_t *sigxx, Real_t *sigyy, Real_t *sigzz,
        Index_t numElem)
{
    const Real_t * domain_e = domain->m_e.data();   (void) domain_e;
    //
    // pull in the stresses appropriate to the hydro integration
    //
    for (Index_t b = 0; b < numElem ; b += EBS)
    {
        TASK_SET_COLOR(iter);
        TASK_SET_LABEL("InitStressTermsForElems");
        // if 'domain_e' is fulfilled, then 'domain_p' and 'domain_q' are too
        # pragma omp task default(none)                             \
            firstprivate(domain, b, sigxx, sigyy, sigzz, numElem)   \
            shared(EBS)                                             \
            depend(in: domain_e[b])                                 \
            depend(out: sigxx[b])
        {
            Index_t start = b;
            Index_t end = MIN(start + EBS, numElem);
            for (Index_t i = start ; i < end ; ++i)
            {
                sigxx[i] = sigyy[i] = sigzz[i] =  - domain->p(i) - domain->q(i) ;
            }
        }
    }
}

/******************************************/

static
void CalcElemShapeFunctionDerivatives( Real_t const x[],
        Real_t const y[],
        Real_t const z[],
        Real_t b[][8],
        Real_t* const volume )
{
    const Real_t x0 = x[0] ;   const Real_t x1 = x[1] ;
    const Real_t x2 = x[2] ;   const Real_t x3 = x[3] ;
    const Real_t x4 = x[4] ;   const Real_t x5 = x[5] ;
    const Real_t x6 = x[6] ;   const Real_t x7 = x[7] ;

    const Real_t y0 = y[0] ;   const Real_t y1 = y[1] ;
    const Real_t y2 = y[2] ;   const Real_t y3 = y[3] ;
    const Real_t y4 = y[4] ;   const Real_t y5 = y[5] ;
    const Real_t y6 = y[6] ;   const Real_t y7 = y[7] ;

    const Real_t z0 = z[0] ;   const Real_t z1 = z[1] ;
    const Real_t z2 = z[2] ;   const Real_t z3 = z[3] ;
    const Real_t z4 = z[4] ;   const Real_t z5 = z[5] ;
    const Real_t z6 = z[6] ;   const Real_t z7 = z[7] ;

    Real_t fjxxi, fjxet, fjxze;
    Real_t fjyxi, fjyet, fjyze;
    Real_t fjzxi, fjzet, fjzze;
    Real_t cjxxi, cjxet, cjxze;
    Real_t cjyxi, cjyet, cjyze;
    Real_t cjzxi, cjzet, cjzze;

    fjxxi = Real_t(.125) * ( (x6-x0) + (x5-x3) - (x7-x1) - (x4-x2) );
    fjxet = Real_t(.125) * ( (x6-x0) - (x5-x3) + (x7-x1) - (x4-x2) );
    fjxze = Real_t(.125) * ( (x6-x0) + (x5-x3) + (x7-x1) + (x4-x2) );

    fjyxi = Real_t(.125) * ( (y6-y0) + (y5-y3) - (y7-y1) - (y4-y2) );
    fjyet = Real_t(.125) * ( (y6-y0) - (y5-y3) + (y7-y1) - (y4-y2) );
    fjyze = Real_t(.125) * ( (y6-y0) + (y5-y3) + (y7-y1) + (y4-y2) );

    fjzxi = Real_t(.125) * ( (z6-z0) + (z5-z3) - (z7-z1) - (z4-z2) );
    fjzet = Real_t(.125) * ( (z6-z0) - (z5-z3) + (z7-z1) - (z4-z2) );
    fjzze = Real_t(.125) * ( (z6-z0) + (z5-z3) + (z7-z1) + (z4-z2) );

    /* compute cofactors */
    cjxxi =    (fjyet * fjzze) - (fjzet * fjyze);
    cjxet =  - (fjyxi * fjzze) + (fjzxi * fjyze);
    cjxze =    (fjyxi * fjzet) - (fjzxi * fjyet);

    cjyxi =  - (fjxet * fjzze) + (fjzet * fjxze);
    cjyet =    (fjxxi * fjzze) - (fjzxi * fjxze);
    cjyze =  - (fjxxi * fjzet) + (fjzxi * fjxet);

    cjzxi =    (fjxet * fjyze) - (fjyet * fjxze);
    cjzet =  - (fjxxi * fjyze) + (fjyxi * fjxze);
    cjzze =    (fjxxi * fjyet) - (fjyxi * fjxet);

    /* calculate partials :
       this need only be done for l = 0,1,2,3   since , by symmetry ,
       (6,7,4,5) = - (0,1,2,3) .
     */
    b[0][0] =   -  cjxxi  -  cjxet  -  cjxze;
    b[0][1] =      cjxxi  -  cjxet  -  cjxze;
    b[0][2] =      cjxxi  +  cjxet  -  cjxze;
    b[0][3] =   -  cjxxi  +  cjxet  -  cjxze;
    b[0][4] = -b[0][2];
    b[0][5] = -b[0][3];
    b[0][6] = -b[0][0];
    b[0][7] = -b[0][1];

    b[1][0] =   -  cjyxi  -  cjyet  -  cjyze;
    b[1][1] =      cjyxi  -  cjyet  -  cjyze;
    b[1][2] =      cjyxi  +  cjyet  -  cjyze;
    b[1][3] =   -  cjyxi  +  cjyet  -  cjyze;
    b[1][4] = -b[1][2];
    b[1][5] = -b[1][3];
    b[1][6] = -b[1][0];
    b[1][7] = -b[1][1];

    b[2][0] =   -  cjzxi  -  cjzet  -  cjzze;
    b[2][1] =      cjzxi  -  cjzet  -  cjzze;
    b[2][2] =      cjzxi  +  cjzet  -  cjzze;
    b[2][3] =   -  cjzxi  +  cjzet  -  cjzze;
    b[2][4] = -b[2][2];
    b[2][5] = -b[2][3];
    b[2][6] = -b[2][0];
    b[2][7] = -b[2][1];

    /* calculate jacobian determinant (volume) */
    *volume = Real_t(8.) * ( fjxet * cjxet + fjyet * cjyet + fjzet * cjzet);
}

/******************************************/

static
void SumElemFaceNormal(Real_t *normalX0, Real_t *normalY0, Real_t *normalZ0,
        Real_t *normalX1, Real_t *normalY1, Real_t *normalZ1,
        Real_t *normalX2, Real_t *normalY2, Real_t *normalZ2,
        Real_t *normalX3, Real_t *normalY3, Real_t *normalZ3,
        const Real_t x0, const Real_t y0, const Real_t z0,
        const Real_t x1, const Real_t y1, const Real_t z1,
        const Real_t x2, const Real_t y2, const Real_t z2,
        const Real_t x3, const Real_t y3, const Real_t z3)
{
    Real_t bisectX0 = Real_t(0.5) * (x3 + x2 - x1 - x0);
    Real_t bisectY0 = Real_t(0.5) * (y3 + y2 - y1 - y0);
    Real_t bisectZ0 = Real_t(0.5) * (z3 + z2 - z1 - z0);
    Real_t bisectX1 = Real_t(0.5) * (x2 + x1 - x3 - x0);
    Real_t bisectY1 = Real_t(0.5) * (y2 + y1 - y3 - y0);
    Real_t bisectZ1 = Real_t(0.5) * (z2 + z1 - z3 - z0);
    Real_t areaX = Real_t(0.25) * (bisectY0 * bisectZ1 - bisectZ0 * bisectY1);
    Real_t areaY = Real_t(0.25) * (bisectZ0 * bisectX1 - bisectX0 * bisectZ1);
    Real_t areaZ = Real_t(0.25) * (bisectX0 * bisectY1 - bisectY0 * bisectX1);

    *normalX0 += areaX;
    *normalX1 += areaX;
    *normalX2 += areaX;
    *normalX3 += areaX;

    *normalY0 += areaY;
    *normalY1 += areaY;
    *normalY2 += areaY;
    *normalY3 += areaY;

    *normalZ0 += areaZ;
    *normalZ1 += areaZ;
    *normalZ2 += areaZ;
    *normalZ3 += areaZ;
}

/******************************************/

static
void CalcElemNodeNormals(Real_t pfx[8],
        Real_t pfy[8],
        Real_t pfz[8],
        const Real_t x[8],
        const Real_t y[8],
        const Real_t z[8])
{
    for (Index_t i = 0 ; i < 8 ; ++i) {
        pfx[i] = Real_t(0.0);
        pfy[i] = Real_t(0.0);
        pfz[i] = Real_t(0.0);
    }
    /* evaluate face one: nodes 0, 1, 2, 3 */
    SumElemFaceNormal(&pfx[0], &pfy[0], &pfz[0],
            &pfx[1], &pfy[1], &pfz[1],
            &pfx[2], &pfy[2], &pfz[2],
            &pfx[3], &pfy[3], &pfz[3],
            x[0], y[0], z[0], x[1], y[1], z[1],
            x[2], y[2], z[2], x[3], y[3], z[3]);
    /* evaluate face two: nodes 0, 4, 5, 1 */
    SumElemFaceNormal(&pfx[0], &pfy[0], &pfz[0],
            &pfx[4], &pfy[4], &pfz[4],
            &pfx[5], &pfy[5], &pfz[5],
            &pfx[1], &pfy[1], &pfz[1],
            x[0], y[0], z[0], x[4], y[4], z[4],
            x[5], y[5], z[5], x[1], y[1], z[1]);
    /* evaluate face three: nodes 1, 5, 6, 2 */
    SumElemFaceNormal(&pfx[1], &pfy[1], &pfz[1],
            &pfx[5], &pfy[5], &pfz[5],
            &pfx[6], &pfy[6], &pfz[6],
            &pfx[2], &pfy[2], &pfz[2],
            x[1], y[1], z[1], x[5], y[5], z[5],
            x[6], y[6], z[6], x[2], y[2], z[2]);
    /* evaluate face four: nodes 2, 6, 7, 3 */
    SumElemFaceNormal(&pfx[2], &pfy[2], &pfz[2],
            &pfx[6], &pfy[6], &pfz[6],
            &pfx[7], &pfy[7], &pfz[7],
            &pfx[3], &pfy[3], &pfz[3],
            x[2], y[2], z[2], x[6], y[6], z[6],
            x[7], y[7], z[7], x[3], y[3], z[3]);
    /* evaluate face five: nodes 3, 7, 4, 0 */
    SumElemFaceNormal(&pfx[3], &pfy[3], &pfz[3],
            &pfx[7], &pfy[7], &pfz[7],
            &pfx[4], &pfy[4], &pfz[4],
            &pfx[0], &pfy[0], &pfz[0],
            x[3], y[3], z[3], x[7], y[7], z[7],
            x[4], y[4], z[4], x[0], y[0], z[0]);
    /* evaluate face six: nodes 4, 7, 6, 5 */
    SumElemFaceNormal(&pfx[4], &pfy[4], &pfz[4],
            &pfx[7], &pfy[7], &pfz[7],
            &pfx[6], &pfy[6], &pfz[6],
            &pfx[5], &pfy[5], &pfz[5],
            x[4], y[4], z[4], x[7], y[7], z[7],
            x[6], y[6], z[6], x[5], y[5], z[5]);
}

/******************************************/

static
void SumElemStressesToNodeForces( const Real_t B[][8],
        const Real_t stress_xx,
        const Real_t stress_yy,
        const Real_t stress_zz,
        Real_t fx[], Real_t fy[], Real_t fz[] )
{
    for(Index_t i = 0; i < 8; i++) {
        fx[i] = -( stress_xx * B[0][i] );
        fy[i] = -( stress_yy * B[1][i]  );
        fz[i] = -( stress_zz * B[2][i] );
    }
}

/******************************************/

static
void IntegrateStressForElems(Domain * domain)
{
    Index_t numElem = domain->numElem();

    const Real_t * domain_e = domain->m_e.data();   (void) domain_e;

    // loop over all elements
    for (Index_t b = 0; b < numElem ; b += EBS)
    {
        MPC_PUSH_DEPENDENCIES(dependencies_domain_x_y_z + (b/EBS), 1);
        TASK_SET_COLOR(iter);
        TASK_SET_LABEL("IntegrateStressForElems1");
        # pragma omp task default(none)                             \
            firstprivate(domain, b, numElem)                        \
            shared( EBS, determ, sigxx, sigyy, sigzz,               \
                    fx_elem, fy_elem, fz_elem)                      \
            depend(in: sigxx[b])                                    \
            depend(out: determ[b], fx_elem[8*b])
        {
            Index_t start = b;
            Index_t end = MIN(start + EBS, numElem);
            for (Index_t k = start ; k < end ; ++k)
            {
                const Index_t* const elemToNode = domain->nodelist(k);
                Real_t B[3][8] ;// shape function derivatives
                Real_t x_local[8] ;
                Real_t y_local[8] ;
                Real_t z_local[8] ;

                // get nodal coordinates from global arrays and copy into local arrays.
                CollectDomainNodesToElemNodes(domain, elemToNode, x_local, y_local, z_local);

                // Volume calculation involves extra work for numerical consistency
                CalcElemShapeFunctionDerivatives(x_local, y_local, z_local,
                        B, &determ[k]);

                // check for negative element volume
                if (determ[k] <= Real_t(0.0)) lulesh_abort();

                CalcElemNodeNormals( B[0] , B[1], B[2],
                        x_local, y_local, z_local );

                // Eliminate thread writing conflicts at the nodes by giving
                // each element its own copy to write to
                SumElemStressesToNodeForces( B, sigxx[k], sigyy[k], sigzz[k],
                        &fx_elem[k*8],
                        &fy_elem[k*8],
                        &fz_elem[k*8] ) ;
            }
        }
    }

    // we need to copy the data out of the temporary
    // arrays used above into the final forces field
    const Real_t * domain_fx = domain->m_fx.data(); (void) domain_fx;
    Index_t numNode = domain->numNode();
    for (Index_t b = 0; b < numNode ; b += NBS)
    {
        MPC_PUSH_DEPENDENCIES(dependencies_fx_fy_fz_elem + (b/NBS), 1);
        TASK_SET_COLOR(iter);
        TASK_SET_LABEL("IntegrateStressForElems2");
        # pragma omp task default(none)             \
            firstprivate(domain, b, numNode)        \
            shared(NBS, fx_elem, fy_elem, fz_elem)  \
            depend(out: domain_fx[b])
        {
            Index_t start = b;
            Index_t end = MIN(start + NBS, numNode);
            for (Index_t gnode = start ; gnode < end ; ++gnode)
            {
                Index_t count = domain->nodeElemCount(gnode) ;
                Index_t *cornerList = domain->nodeElemCornerList(gnode) ;
                Real_t fx_tmp = Real_t(0.0) ;
                Real_t fy_tmp = Real_t(0.0) ;
                Real_t fz_tmp = Real_t(0.0) ;
                for (Index_t i=0 ; i < count ; ++i) {
                    Index_t elem = cornerList[i] ;
                    fx_tmp += fx_elem[elem] ;
                    fy_tmp += fy_elem[elem] ;
                    fz_tmp += fz_elem[elem] ;
                }
                domain->fx(gnode) = fx_tmp ;
                domain->fy(gnode) = fy_tmp ;
                domain->fz(gnode) = fz_tmp ;
            }
        }
    }
}

/******************************************/

static
void VoluDer(const Real_t x0, const Real_t x1, const Real_t x2,
        const Real_t x3, const Real_t x4, const Real_t x5,
        const Real_t y0, const Real_t y1, const Real_t y2,
        const Real_t y3, const Real_t y4, const Real_t y5,
        const Real_t z0, const Real_t z1, const Real_t z2,
        const Real_t z3, const Real_t z4, const Real_t z5,
        Real_t* dvdx, Real_t* dvdy, Real_t* dvdz)
{
    const Real_t twelfth = Real_t(1.0) / Real_t(12.0) ;

    *dvdx =
        (y1 + y2) * (z0 + z1) - (y0 + y1) * (z1 + z2) +
        (y0 + y4) * (z3 + z4) - (y3 + y4) * (z0 + z4) -
        (y2 + y5) * (z3 + z5) + (y3 + y5) * (z2 + z5);
    *dvdy =
        - (x1 + x2) * (z0 + z1) + (x0 + x1) * (z1 + z2) -
        (x0 + x4) * (z3 + z4) + (x3 + x4) * (z0 + z4) +
        (x2 + x5) * (z3 + z5) - (x3 + x5) * (z2 + z5);

    *dvdz =
        - (y1 + y2) * (x0 + x1) + (y0 + y1) * (x1 + x2) -
        (y0 + y4) * (x3 + x4) + (y3 + y4) * (x0 + x4) +
        (y2 + y5) * (x3 + x5) - (y3 + y5) * (x2 + x5);

    *dvdx *= twelfth;
    *dvdy *= twelfth;
    *dvdz *= twelfth;
}

/******************************************/

static
void CalcElemVolumeDerivative(Real_t dvdx[8],
        Real_t dvdy[8],
        Real_t dvdz[8],
        const Real_t x[8],
        const Real_t y[8],
        const Real_t z[8])
{
    VoluDer(x[1], x[2], x[3], x[4], x[5], x[7],
            y[1], y[2], y[3], y[4], y[5], y[7],
            z[1], z[2], z[3], z[4], z[5], z[7],
            &dvdx[0], &dvdy[0], &dvdz[0]);
    VoluDer(x[0], x[1], x[2], x[7], x[4], x[6],
            y[0], y[1], y[2], y[7], y[4], y[6],
            z[0], z[1], z[2], z[7], z[4], z[6],
            &dvdx[3], &dvdy[3], &dvdz[3]);
    VoluDer(x[3], x[0], x[1], x[6], x[7], x[5],
            y[3], y[0], y[1], y[6], y[7], y[5],
            z[3], z[0], z[1], z[6], z[7], z[5],
            &dvdx[2], &dvdy[2], &dvdz[2]);
    VoluDer(x[2], x[3], x[0], x[5], x[6], x[4],
            y[2], y[3], y[0], y[5], y[6], y[4],
            z[2], z[3], z[0], z[5], z[6], z[4],
            &dvdx[1], &dvdy[1], &dvdz[1]);
    VoluDer(x[7], x[6], x[5], x[0], x[3], x[1],
            y[7], y[6], y[5], y[0], y[3], y[1],
            z[7], z[6], z[5], z[0], z[3], z[1],
            &dvdx[4], &dvdy[4], &dvdz[4]);
    VoluDer(x[4], x[7], x[6], x[1], x[0], x[2],
            y[4], y[7], y[6], y[1], y[0], y[2],
            z[4], z[7], z[6], z[1], z[0], z[2],
            &dvdx[5], &dvdy[5], &dvdz[5]);
    VoluDer(x[5], x[4], x[7], x[2], x[1], x[3],
            y[5], y[4], y[7], y[2], y[1], y[3],
            z[5], z[4], z[7], z[2], z[1], z[3],
            &dvdx[6], &dvdy[6], &dvdz[6]);
    VoluDer(x[6], x[5], x[4], x[3], x[2], x[0],
            y[6], y[5], y[4], y[3], y[2], y[0],
            z[6], z[5], z[4], z[3], z[2], z[0],
            &dvdx[7], &dvdy[7], &dvdz[7]);
}

/******************************************/

static
void CalcElemFBHourglassForce(Real_t *xd, Real_t *yd, Real_t *zd,  Real_t hourgam[][4],
        Real_t coefficient,
        Real_t *hgfx, Real_t *hgfy, Real_t *hgfz )
{
    Real_t hxx[4];
    for(Index_t i = 0; i < 4; i++) {
        hxx[i] = hourgam[0][i] * xd[0] + hourgam[1][i] * xd[1] +
            hourgam[2][i] * xd[2] + hourgam[3][i] * xd[3] +
            hourgam[4][i] * xd[4] + hourgam[5][i] * xd[5] +
            hourgam[6][i] * xd[6] + hourgam[7][i] * xd[7];
    }
    for(Index_t i = 0; i < 8; i++) {
        hgfx[i] = coefficient *
            (hourgam[i][0] * hxx[0] + hourgam[i][1] * hxx[1] +
             hourgam[i][2] * hxx[2] + hourgam[i][3] * hxx[3]);
    }
    for(Index_t i = 0; i < 4; i++) {
        hxx[i] = hourgam[0][i] * yd[0] + hourgam[1][i] * yd[1] +
            hourgam[2][i] * yd[2] + hourgam[3][i] * yd[3] +
            hourgam[4][i] * yd[4] + hourgam[5][i] * yd[5] +
            hourgam[6][i] * yd[6] + hourgam[7][i] * yd[7];
    }
    for(Index_t i = 0; i < 8; i++) {
        hgfy[i] = coefficient *
            (hourgam[i][0] * hxx[0] + hourgam[i][1] * hxx[1] +
             hourgam[i][2] * hxx[2] + hourgam[i][3] * hxx[3]);
    }
    for(Index_t i = 0; i < 4; i++) {
        hxx[i] = hourgam[0][i] * zd[0] + hourgam[1][i] * zd[1] +
            hourgam[2][i] * zd[2] + hourgam[3][i] * zd[3] +
            hourgam[4][i] * zd[4] + hourgam[5][i] * zd[5] +
            hourgam[6][i] * zd[6] + hourgam[7][i] * zd[7];
    }
    for(Index_t i = 0; i < 8; i++) {
        hgfz[i] = coefficient *
            (hourgam[i][0] * hxx[0] + hourgam[i][1] * hxx[1] +
             hourgam[i][2] * hxx[2] + hourgam[i][3] * hxx[3]);
    }
}

/*************************************************
 *
 *     FUNCTION: Calculates the Flanagan-Belytschko anti-hourglass
 *               force.
 *
 *************************************************/
static inline
void CalcFBHourglassForceForElems( Domain * domain,
                                   Real_t *determ,
                                   Real_t *x8n, Real_t *y8n, Real_t *z8n,
                                   Real_t *dvdx, Real_t *dvdy, Real_t *dvdz,
                                   Real_t hourg, Index_t numElem,
                                   Index_t numNode)
{
    const Real_t * domain_v = domain->m_v.data();   (void) domain_v;
    const Real_t * domain_ss = domain->m_ss.data(); (void) domain_ss;

    for (Index_t b = 0; b < numElem ; b += EBS)
    {
        MPC_PUSH_DEPENDENCIES(dependencies_domain_xd_yd_zd + b/EBS, 1);
        TASK_SET_COLOR(iter);
        TASK_SET_LABEL("CalcFBHourglassForceForElems1");
        # pragma omp task default(none)                                         \
            firstprivate(   domain, b, numElem, determ, hourg,                  \
                            x8n, y8n, z8n, dvdx, dvdy, dvdz)                    \
            shared( EBS, gamma_v,                                               \
                    fx_elem_FBH, fy_elem_FBH, fz_elem_FBH)                      \
            depend(in:  determ[b], domain_v[b], domain_ss[b])                   \
            depend(out: fx_elem_FBH[8*b])
        {
            Index_t start = b;
            Index_t end = MIN(start + EBS, numElem);
            for (Index_t i2 = start ; i2 < end ; ++i2)
            {
                Real_t *fx_local, *fy_local, *fz_local ;
                Real_t hgfx[8], hgfy[8], hgfz[8] ;

                Real_t coefficient;

                Real_t hourgam[8][4];
                Real_t xd1[8], yd1[8], zd1[8] ;

                const Index_t *elemToNode = domain->nodelist(i2);
                Index_t i3=8*i2;
                Real_t volinv=Real_t(1.0)/determ[i2];
                Real_t ss1, mass1, volume13 ;
                for(Index_t i1=0;i1<4;++i1){

                    Real_t hourmodx =
                        x8n[i3] * gamma_v[i1][0] + x8n[i3+1] * gamma_v[i1][1] +
                        x8n[i3+2] * gamma_v[i1][2] + x8n[i3+3] * gamma_v[i1][3] +
                        x8n[i3+4] * gamma_v[i1][4] + x8n[i3+5] * gamma_v[i1][5] +
                        x8n[i3+6] * gamma_v[i1][6] + x8n[i3+7] * gamma_v[i1][7];

                    Real_t hourmody =
                        y8n[i3] * gamma_v[i1][0] + y8n[i3+1] * gamma_v[i1][1] +
                        y8n[i3+2] * gamma_v[i1][2] + y8n[i3+3] * gamma_v[i1][3] +
                        y8n[i3+4] * gamma_v[i1][4] + y8n[i3+5] * gamma_v[i1][5] +
                        y8n[i3+6] * gamma_v[i1][6] + y8n[i3+7] * gamma_v[i1][7];

                    Real_t hourmodz =
                        z8n[i3] * gamma_v[i1][0] + z8n[i3+1] * gamma_v[i1][1] +
                        z8n[i3+2] * gamma_v[i1][2] + z8n[i3+3] * gamma_v[i1][3] +
                        z8n[i3+4] * gamma_v[i1][4] + z8n[i3+5] * gamma_v[i1][5] +
                        z8n[i3+6] * gamma_v[i1][6] + z8n[i3+7] * gamma_v[i1][7];

                    hourgam[0][i1] = gamma_v[i1][0] -  volinv*(dvdx[i3  ] * hourmodx +
                            dvdy[i3  ] * hourmody +
                            dvdz[i3  ] * hourmodz );

                    hourgam[1][i1] = gamma_v[i1][1] -  volinv*(dvdx[i3+1] * hourmodx +
                            dvdy[i3+1] * hourmody +
                            dvdz[i3+1] * hourmodz );

                    hourgam[2][i1] = gamma_v[i1][2] -  volinv*(dvdx[i3+2] * hourmodx +
                            dvdy[i3+2] * hourmody +
                            dvdz[i3+2] * hourmodz );

                    hourgam[3][i1] = gamma_v[i1][3] -  volinv*(dvdx[i3+3] * hourmodx +
                            dvdy[i3+3] * hourmody +
                            dvdz[i3+3] * hourmodz );

                    hourgam[4][i1] = gamma_v[i1][4] -  volinv*(dvdx[i3+4] * hourmodx +
                            dvdy[i3+4] * hourmody +
                            dvdz[i3+4] * hourmodz );

                    hourgam[5][i1] = gamma_v[i1][5] -  volinv*(dvdx[i3+5] * hourmodx +
                            dvdy[i3+5] * hourmody +
                            dvdz[i3+5] * hourmodz );
                    hourgam[6][i1] = gamma_v[i1][6] -  volinv*(dvdx[i3+6] * hourmodx +
                            dvdy[i3+6] * hourmody +
                            dvdz[i3+6] * hourmodz );

                    hourgam[7][i1] = gamma_v[i1][7] -  volinv*(dvdx[i3+7] * hourmodx +
                            dvdy[i3+7] * hourmody +
                            dvdz[i3+7] * hourmodz );

                }
                /* compute forces */
                /* store forces into h arrays (force arrays) */

                ss1=domain->ss(i2);
                mass1=domain->elemMass(i2);
                volume13=CBRT(determ[i2]);

                Index_t n0si2 = elemToNode[0];
                Index_t n1si2 = elemToNode[1];
                Index_t n2si2 = elemToNode[2];
                Index_t n3si2 = elemToNode[3];
                Index_t n4si2 = elemToNode[4];
                Index_t n5si2 = elemToNode[5];
                Index_t n6si2 = elemToNode[6];
                Index_t n7si2 = elemToNode[7];

                xd1[0] = domain->xd(n0si2);
                xd1[1] = domain->xd(n1si2);
                xd1[2] = domain->xd(n2si2);
                xd1[3] = domain->xd(n3si2);
                xd1[4] = domain->xd(n4si2);
                xd1[5] = domain->xd(n5si2);
                xd1[6] = domain->xd(n6si2);
                xd1[7] = domain->xd(n7si2);

                yd1[0] = domain->yd(n0si2);
                yd1[1] = domain->yd(n1si2);
                yd1[2] = domain->yd(n2si2);
                yd1[3] = domain->yd(n3si2);
                yd1[4] = domain->yd(n4si2);
                yd1[5] = domain->yd(n5si2);
                yd1[6] = domain->yd(n6si2);
                yd1[7] = domain->yd(n7si2);

                zd1[0] = domain->zd(n0si2);
                zd1[1] = domain->zd(n1si2);
                zd1[2] = domain->zd(n2si2);
                zd1[3] = domain->zd(n3si2);
                zd1[4] = domain->zd(n4si2);
                zd1[5] = domain->zd(n5si2);
                zd1[6] = domain->zd(n6si2);
                zd1[7] = domain->zd(n7si2);

                coefficient = - hourg * Real_t(0.01) * ss1 * mass1 / volume13;
                CalcElemFBHourglassForce(xd1,yd1,zd1,
                        hourgam,
                        coefficient, hgfx, hgfy, hgfz);

                fx_local = &fx_elem_FBH[i3] ;
                fx_local[0] = hgfx[0];
                fx_local[1] = hgfx[1];
                fx_local[2] = hgfx[2];
                fx_local[3] = hgfx[3];
                fx_local[4] = hgfx[4];
                fx_local[5] = hgfx[5];
                fx_local[6] = hgfx[6];
                fx_local[7] = hgfx[7];

                fy_local = &fy_elem_FBH[i3] ;
                fy_local[0] = hgfy[0];
                fy_local[1] = hgfy[1];
                fy_local[2] = hgfy[2];
                fy_local[3] = hgfy[3];
                fy_local[4] = hgfy[4];
                fy_local[5] = hgfy[5];
                fy_local[6] = hgfy[6];
                fy_local[7] = hgfy[7];

                fz_local = &fz_elem_FBH[i3] ;
                fz_local[0] = hgfz[0];
                fz_local[1] = hgfz[1];
                fz_local[2] = hgfz[2];
                fz_local[3] = hgfz[3];
                fz_local[4] = hgfz[4];
                fz_local[5] = hgfz[5];
                fz_local[6] = hgfz[6];
                fz_local[7] = hgfz[7];

            } /* for i2 */
        } /* task */
    } /* for b */

    // Collect the data from the local arrays into the final force array
    for (Index_t b = 0; b < numNode ; b += NBS)
    {
        const Real_t * domain_fx = domain->m_fx.data(); (void) domain_fx;
        MPC_PUSH_DEPENDENCIES(dependencies_fx_fy_fz_elem_FBH + (b/NBS), 1);
        TASK_SET_COLOR(iter);
        TASK_SET_LABEL("CalcFBHourglassForceForElems2");
        # pragma omp task default(none)                         \
            firstprivate(domain, b, numNode)                    \
            shared(NBS, fx_elem_FBH, fy_elem_FBH, fz_elem_FBH)  \
            depend(out: domain_fx[b])
        {
            Index_t start = b;
            Index_t end = MIN(start + NBS, numNode);
            for (Index_t gnode = start ; gnode < end ; ++gnode)
            {
                Index_t count = domain->nodeElemCount(gnode) ;
                Index_t *cornerList = domain->nodeElemCornerList(gnode) ;
                Real_t fx_tmp = Real_t(0.0) ;
                Real_t fy_tmp = Real_t(0.0) ;
                Real_t fz_tmp = Real_t(0.0) ;
                for (Index_t i=0 ; i < count ; ++i) {
                    Index_t ielem = cornerList[i] ;
                    fx_tmp += fx_elem_FBH[ielem] ;
                    fy_tmp += fy_elem_FBH[ielem] ;
                    fz_tmp += fz_elem_FBH[ielem] ;
                }
                domain->fx(gnode) += fx_tmp ;
                domain->fy(gnode) += fy_tmp ;
                domain->fz(gnode) += fz_tmp ;
            }
        }
    }
}

/******************************************/

static
void CalcHourglassControlForElems(Domain * domain, Real_t determ[], Real_t hgcoef)
{
    Index_t numElem = domain->numElem() ;
    const Real_t * domain_v = domain->m_v.data();   (void) domain_v;

    for (Index_t b = 0; b < numElem ; b += EBS)
    {
        MPC_PUSH_DEPENDENCIES(dependencies_domain_x_y_z + b/EBS, 1);
        TASK_SET_COLOR(iter);
        TASK_SET_LABEL("CalcHourglassControlForElems");
        # pragma omp task default(none)                 \
            firstprivate(domain, b, numElem, determ)    \
            shared( gamma_v, EBS,                       \
                    dvdx, dvdy, dvdz,                   \
                    x8n, y8n, z8n)                      \
            depend(in:  domain_v[b])                    \
            depend(out: determ[b])
        {
            Index_t start = b;
            Index_t end = MIN(start + EBS, numElem);
            for (Index_t i = start ; i < end ; ++i)
            {
                Real_t  x1[8],  y1[8],  z1[8] ;
                Real_t pfx[8], pfy[8], pfz[8] ;

                Index_t* elemToNode = domain->nodelist(i);
                CollectDomainNodesToElemNodes(domain, elemToNode, x1, y1, z1);

                CalcElemVolumeDerivative(pfx, pfy, pfz, x1, y1, z1);

                /* load into temporary storage for FB Hour Glass control */
                for(Index_t ii=0;ii<8;++ii){
                    Index_t jj=8*i+ii;

                    dvdx[jj] = pfx[ii];
                    dvdy[jj] = pfy[ii];
                    dvdz[jj] = pfz[ii];
                    x8n[jj]  = x1[ii];
                    y8n[jj]  = y1[ii];
                    z8n[jj]  = z1[ii];
                }

                determ[i] = domain->volo(i) * domain->v(i);

                /* Do a check for negative volumes */
                if (domain->v(i) <= Real_t(0.0))   lulesh_abort();
            }
        }
    }

    if ( hgcoef > Real_t(0.) ) {
        CalcFBHourglassForceForElems( domain,
                determ, x8n, y8n, z8n, dvdx, dvdy, dvdz,
                hgcoef, numElem, domain->numNode()) ;
    }
}

/******************************************/

static
void CalcVolumeForceForElems(Domain * domain)
{
    Index_t numElem = domain->numElem() ;
    if (numElem != 0)
    {
        Real_t hgcoef = domain->hgcoef() ;

        /* Sum contributions to total stress tensor */
        InitStressTermsForElems(domain, sigxx, sigyy, sigzz, numElem);

        // call elemlib stress integration loop to produce nodal forces from
        // material stresses.
        IntegrateStressForElems(domain);

        CalcHourglassControlForElems(domain, determ, hgcoef);
    }
}

/******************************************/

static void CalcForceForNodes(Domain * domain)
{
#if USE_MPI
    CommRecv(domain, MSG_FX_FY_FZ);
#endif

    const Real_t * domain_fx = domain->m_fx.data(); (void) domain_fx;
    Index_t numNode = domain->numNode();
    for (Index_t b = 0; b < numNode ; b += NBS)
    {
        TASK_SET_COLOR(iter);
        TASK_SET_LABEL("CalcForceForNodes");
        # pragma omp task default(none)         \
            firstprivate(domain, b, numNode)    \
            shared(NBS)                         \
            depend(out: domain_fx[b])
        {
            Index_t start = b;
            Index_t end = MIN(start + NBS, numNode);
            for (Index_t i = start ; i < end ; ++i)
            {
                domain->fx(i) = Real_t(0.0) ;
                domain->fy(i) = Real_t(0.0) ;
                domain->fz(i) = Real_t(0.0) ;
            }
        }
    }

    /* Calcforce calls partial, force, hourq */
    CalcVolumeForceForElems(domain);

#if USE_MPI
    CommPack(domain,   MSG_FX_FY_FZ);
    CommSend(domain,   MSG_FX_FY_FZ);
    CommUnpack(domain, MSG_FX_FY_FZ);
#endif  /* USE_MPI */
}

/******************************************/

static
void CalcAccelerationForNodes(Domain * domain)
{
    const Real_t * domain_fx    = domain->m_fx.data();  (void) domain_fx;
    const Real_t * domain_xdd   = domain->m_xdd.data(); (void) domain_xdd;
    const Real_t * domain_ydd   = domain->m_ydd.data(); (void) domain_ydd;
    const Real_t * domain_zdd   = domain->m_zdd.data(); (void) domain_zdd;
    Index_t numNode = domain->numNode();
    for (Index_t b = 0; b < numNode ; b += NBS)
    {
        TASK_SET_COLOR(iter);
        TASK_SET_LABEL("CalcAccelerationForNodes");
        # pragma omp task default(none)                             \
            firstprivate(domain, b, numNode)                        \
            shared(NBS)                                             \
            depend(in: domain_fx[b])                                \
            depend(out: domain_xdd[b], domain_ydd[b], domain_zdd[b])
        {
            Index_t start = b;
            Index_t end = MIN(start + NBS, numNode);
            for (Index_t i = start ; i < end ; ++i)
            {
                domain->xdd(i) = domain->fx(i) / domain->nodalMass(i);
                domain->ydd(i) = domain->fy(i) / domain->nodalMass(i);
                domain->zdd(i) = domain->fz(i) / domain->nodalMass(i);
            }
        }
    }
}

/******************************************/

static
void ApplyAccelerationBoundaryConditionsForNodes(Domain * domain)
{
    Index_t size = domain->sizeX();
    Index_t numNodeBC = (size+1)*(size+1) ;

    // 0 0 0
    if (domain->symmXempty() && domain->symmYempty() && domain->symmZempty())
    {
        // nothing to do
    }

    // 1 0 0
    else if (!domain->symmXempty() && domain->symmYempty() && domain->symmZempty())
    {
        for (Index_t b = 0; b < numNodeBC ; b += NBS)
        {
            mpc_omp_task_dependency_t dependencies[1];
            dependencies[0] = dependencies_bc_xdd[b/NBS];
            MPC_PUSH_DEPENDENCIES(dependencies, 1);
            TASK_SET_COLOR(iter);
            TASK_SET_LABEL("AABCForNodes");
            # pragma omp task default(none) firstprivate(domain, b, numNodeBC) shared(NBS)
            {
                Index_t start = b;
                Index_t end = MIN(start + NBS, numNodeBC);
                for (Index_t i = start ; i < end ; ++i)
                {
                    domain->xdd(domain->symmX(i)) = Real_t(0.0);
                }
            }
        }
    }
    // 0 1 0
    else if (domain->symmXempty() && !domain->symmYempty() && domain->symmZempty())
    {
        for (Index_t b = 0; b < numNodeBC ; b += NBS)
        {
            mpc_omp_task_dependency_t dependencies[1];
            dependencies[0] = dependencies_bc_ydd[b/NBS];
            MPC_PUSH_DEPENDENCIES(dependencies, 1);
            TASK_SET_COLOR(iter);
            TASK_SET_LABEL("AABCForNodes");
            # pragma omp task default(none) firstprivate(domain, b, numNodeBC) shared(NBS)
            {
                Index_t start = b;
                Index_t end = MIN(start + NBS, numNodeBC);
                for (Index_t i = start ; i < end ; ++i)
                {
                    domain->ydd(domain->symmY(i)) = Real_t(0.0);
                }
            }
        }
    }
    // 1 1 0
    else if (!domain->symmXempty() && !domain->symmYempty() && domain->symmZempty())
    {
        for (Index_t b = 0; b < numNodeBC ; b += NBS)
        {
            mpc_omp_task_dependency_t dependencies[2];
            dependencies[0] = dependencies_bc_xdd[b/NBS];
            dependencies[1] = dependencies_bc_ydd[b/NBS];
            MPC_PUSH_DEPENDENCIES(dependencies, 2);
            TASK_SET_COLOR(iter);
            TASK_SET_LABEL("AABCForNodes");
            # pragma omp task default(none) firstprivate(domain, b, numNodeBC) shared(NBS)
            {
                Index_t start = b;
                Index_t end = MIN(start + NBS, numNodeBC);
                for (Index_t i = start ; i < end ; ++i)
                {
                    domain->xdd(domain->symmX(i)) = Real_t(0.0);
                    domain->ydd(domain->symmY(i)) = Real_t(0.0);
                }
            }
        }
    }
    // 0 0 1
    else if (domain->symmXempty() && domain->symmYempty() && !domain->symmZempty())
    {
        for (Index_t b = 0; b < numNodeBC ; b += NBS)
        {
            mpc_omp_task_dependency_t dependencies[1];
            dependencies[0] = dependencies_bc_zdd[b/NBS];
            MPC_PUSH_DEPENDENCIES(dependencies, 1);
            TASK_SET_COLOR(iter);
            TASK_SET_LABEL("AABCForNodes");
            # pragma omp task default(none) firstprivate(domain, b, numNodeBC) shared(NBS)
            {
                Index_t start = b;
                Index_t end = MIN(start + NBS, numNodeBC);
                for (Index_t i = start ; i < end ; ++i)
                {
                    domain->zdd(domain->symmZ(i)) = Real_t(0.0);
                }
            }
        }
    }
    // 1 0 1
    else if (!domain->symmXempty() && domain->symmYempty() && !domain->symmZempty())
    {
        for (Index_t b = 0; b < numNodeBC ; b += NBS)
        {
            mpc_omp_task_dependency_t dependencies[2];
            dependencies[0] = dependencies_bc_xdd[b/NBS];
            dependencies[1] = dependencies_bc_zdd[b/NBS];
            MPC_PUSH_DEPENDENCIES(dependencies, 2);
            TASK_SET_COLOR(iter);
            TASK_SET_LABEL("AABCForNodes");
            # pragma omp task default(none) firstprivate(domain, b, numNodeBC) shared(NBS)
            {
                Index_t start = b;
                Index_t end = MIN(start + NBS, numNodeBC);
                for (Index_t i = start ; i < end ; ++i)
                {
                    domain->xdd(domain->symmX(i)) = Real_t(0.0);
                    domain->zdd(domain->symmZ(i)) = Real_t(0.0);
                }
            }
        }
    }
    // 0 1 1
    else if (domain->symmXempty() && !domain->symmYempty() && !domain->symmZempty())
    {
        for (Index_t b = 0; b < numNodeBC ; b += NBS)
        {
            mpc_omp_task_dependency_t dependencies[2];
            dependencies[0] = dependencies_bc_ydd[b/NBS];
            dependencies[1] = dependencies_bc_zdd[b/NBS];
            MPC_PUSH_DEPENDENCIES(dependencies, 2);
            TASK_SET_COLOR(iter);
            TASK_SET_LABEL("AABCForNodes");
            # pragma omp task default(none) firstprivate(domain, b, numNodeBC) shared(NBS)
            {
                Index_t start = b;
                Index_t end = MIN(start + NBS, numNodeBC);
                for (Index_t i = start ; i < end ; ++i)
                {
                    domain->ydd(domain->symmY(i)) = Real_t(0.0);
                    domain->zdd(domain->symmZ(i)) = Real_t(0.0);
                }
            }
        }
    }
    // 1 1 1
    else if (!domain->symmXempty() && !domain->symmYempty() && !domain->symmZempty())
    {
        for (Index_t b = 0; b < numNodeBC ; b += NBS)
        {
            mpc_omp_task_dependency_t dependencies[3];
            dependencies[0] = dependencies_bc_xdd[b/NBS];
            dependencies[1] = dependencies_bc_ydd[b/NBS];
            dependencies[2] = dependencies_bc_zdd[b/NBS];
            MPC_PUSH_DEPENDENCIES(dependencies, 3);
            TASK_SET_COLOR(iter);
            TASK_SET_LABEL("AABCForNodes");
            # pragma omp task default(none) firstprivate(domain, b, numNodeBC) shared(NBS)
            {
                Index_t start = b;
                Index_t end = MIN(start + NBS, numNodeBC);
                for (Index_t i = start ; i < end ; ++i)
                {
                    domain->xdd(domain->symmX(i)) = Real_t(0.0);
                    domain->ydd(domain->symmY(i)) = Real_t(0.0);
                    domain->zdd(domain->symmZ(i)) = Real_t(0.0);
                }
            }
        }
    }
    else
    {
        /* should never reach this code */
        assert(0);
    }
}

/******************************************/

static
void CalcVelocityForNodes(Domain * domain)
{
    const Real_t * domain_xd    = domain->m_xd.data();  (void) domain_xd;
    const Real_t * domain_yd    = domain->m_yd.data();  (void) domain_yd;
    const Real_t * domain_zd    = domain->m_zd.data();  (void) domain_zd;
    const Real_t * domain_xdd   = domain->m_xdd.data(); (void) domain_xdd;
    const Real_t * domain_ydd   = domain->m_ydd.data(); (void) domain_ydd;
    const Real_t * domain_zdd   = domain->m_zdd.data(); (void) domain_zdd;

    Index_t numNode = domain->numNode();
    for (Index_t b = 0; b < numNode ; b += NBS)
    {
        TASK_SET_COLOR(iter);
        TASK_SET_LABEL("CalcVelocityForNodes");
        # pragma omp task default(none)                                     \
            firstprivate(domain, b, numNode)                                \
            shared(NBS)                                                     \
            depend(in:      domain->m_deltatime,                            \
                            domain_xdd[b], domain_ydd[b], domain_zdd[b])    \
            depend(inout:   domain_xd[b],  domain_yd[b],  domain_zd[b])
        {
            const Real_t dt = domain->deltatime();
            const Real_t u_cut = domain->u_cut() ;

            Index_t start = b;
            Index_t end = MIN(start + NBS, numNode);
            for (Index_t i = start ; i < end ; ++i)
            {
                Real_t xdtmp, ydtmp, zdtmp ;

                xdtmp = domain->xd(i) + domain->xdd(i) * dt ;
                if( FABS(xdtmp) < u_cut ) xdtmp = Real_t(0.0);
                domain->xd(i) = xdtmp ;

                ydtmp = domain->yd(i) + domain->ydd(i) * dt ;
                if( FABS(ydtmp) < u_cut ) ydtmp = Real_t(0.0);
                domain->yd(i) = ydtmp ;

                zdtmp = domain->zd(i) + domain->zdd(i) * dt ;
                if( FABS(zdtmp) < u_cut ) zdtmp = Real_t(0.0);
                domain->zd(i) = zdtmp ;
            }
        }
    }
}

/******************************************/

static
void CalcPositionForNodes(Domain * domain)
{
    const Real_t * domain_x     = domain->m_x.data();   (void) domain_x;
    const Real_t * domain_y     = domain->m_y.data();   (void) domain_y;
    const Real_t * domain_z     = domain->m_z.data();   (void) domain_z;
    const Real_t * domain_xd    = domain->m_xd.data();  (void) domain_xd;
    const Real_t * domain_yd    = domain->m_yd.data();  (void) domain_yd;
    const Real_t * domain_zd    = domain->m_zd.data();  (void) domain_zd;

    Index_t numNode = domain->numNode();
    for (Index_t b = 0; b < numNode ; b += NBS)
    {
        TASK_SET_COLOR(iter);
        TASK_SET_LABEL("CalcPositionForNodes");
        # pragma omp task default(none)                                                 \
            firstprivate(domain, b, numNode)                                            \
            shared(NBS)                                                                 \
            depend(in: domain->m_deltatime, domain_xd[b], domain_yd[b], domain_zd[b])   \
            depend(out: domain_x[b], domain_y[b], domain_z[b])
        {
            const Real_t dt = domain->deltatime();
            Index_t start = b;
            Index_t end = MIN(start + NBS, numNode);
            for (Index_t i = start ; i < end ; ++i)
            {
                domain->x(i) += domain->xd(i) * dt ;
                domain->y(i) += domain->yd(i) * dt ;
                domain->z(i) += domain->zd(i) * dt ;
            }
        }
    }
}

/******************************************/

static
void LagrangeNodal(Domain * domain)
{
    /* time of boundary condition evaluation is beginning of step for force and
     * acceleration boundary conditions. */
    CalcForceForNodes(domain);
    CalcAccelerationForNodes(domain);
    ApplyAccelerationBoundaryConditionsForNodes(domain);
    CalcVelocityForNodes(domain);
    CalcPositionForNodes(domain);
}

/******************************************/

static
Real_t CalcElemVolume(const Real_t x0, const Real_t x1,
        const Real_t x2, const Real_t x3,
        const Real_t x4, const Real_t x5,
        const Real_t x6, const Real_t x7,
        const Real_t y0, const Real_t y1,
        const Real_t y2, const Real_t y3,
        const Real_t y4, const Real_t y5,
        const Real_t y6, const Real_t y7,
        const Real_t z0, const Real_t z1,
        const Real_t z2, const Real_t z3,
        const Real_t z4, const Real_t z5,
        const Real_t z6, const Real_t z7)
{
    Real_t twelveth = Real_t(1.0)/Real_t(12.0);

    Real_t dx61 = x6 - x1;
    Real_t dy61 = y6 - y1;
    Real_t dz61 = z6 - z1;

    Real_t dx70 = x7 - x0;
    Real_t dy70 = y7 - y0;
    Real_t dz70 = z7 - z0;

    Real_t dx63 = x6 - x3;
    Real_t dy63 = y6 - y3;
    Real_t dz63 = z6 - z3;

    Real_t dx20 = x2 - x0;
    Real_t dy20 = y2 - y0;
    Real_t dz20 = z2 - z0;

    Real_t dx50 = x5 - x0;
    Real_t dy50 = y5 - y0;
    Real_t dz50 = z5 - z0;

    Real_t dx64 = x6 - x4;
    Real_t dy64 = y6 - y4;
    Real_t dz64 = z6 - z4;

    Real_t dx31 = x3 - x1;
    Real_t dy31 = y3 - y1;
    Real_t dz31 = z3 - z1;

    Real_t dx72 = x7 - x2;
    Real_t dy72 = y7 - y2;
    Real_t dz72 = z7 - z2;

    Real_t dx43 = x4 - x3;
    Real_t dy43 = y4 - y3;
    Real_t dz43 = z4 - z3;

    Real_t dx57 = x5 - x7;
    Real_t dy57 = y5 - y7;
    Real_t dz57 = z5 - z7;

    Real_t dx14 = x1 - x4;
    Real_t dy14 = y1 - y4;
    Real_t dz14 = z1 - z4;

    Real_t dx25 = x2 - x5;
    Real_t dy25 = y2 - y5;
    Real_t dz25 = z2 - z5;

#define TRIPLE_PRODUCT(x1, y1, z1, x2, y2, z2, x3, y3, z3) \
    ((x1)*((y2)*(z3) - (z2)*(y3)) + (x2)*((z1)*(y3) - (y1)*(z3)) + (x3)*((y1)*(z2) - (z1)*(y2)))

    Real_t volume =
        TRIPLE_PRODUCT(dx31 + dx72, dx63, dx20,
                dy31 + dy72, dy63, dy20,
                dz31 + dz72, dz63, dz20) +
        TRIPLE_PRODUCT(dx43 + dx57, dx64, dx70,
                dy43 + dy57, dy64, dy70,
                dz43 + dz57, dz64, dz70) +
        TRIPLE_PRODUCT(dx14 + dx25, dx61, dx50,
                dy14 + dy25, dy61, dy50,
                dz14 + dz25, dz61, dz50);

#undef TRIPLE_PRODUCT

    volume *= twelveth;

    return volume ;
}

/******************************************/

//inline
Real_t CalcElemVolume( const Real_t x[8], const Real_t y[8], const Real_t z[8] )
{
    return CalcElemVolume( x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7],
            y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7],
            z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7]);
}

/******************************************/

static
Real_t AreaFace( const Real_t x0, const Real_t x1,
        const Real_t x2, const Real_t x3,
        const Real_t y0, const Real_t y1,
        const Real_t y2, const Real_t y3,
        const Real_t z0, const Real_t z1,
        const Real_t z2, const Real_t z3)
{
    Real_t fx = (x2 - x0) - (x3 - x1);
    Real_t fy = (y2 - y0) - (y3 - y1);
    Real_t fz = (z2 - z0) - (z3 - z1);
    Real_t gx = (x2 - x0) + (x3 - x1);
    Real_t gy = (y2 - y0) + (y3 - y1);
    Real_t gz = (z2 - z0) + (z3 - z1);
    Real_t area =
        (fx * fx + fy * fy + fz * fz) *
        (gx * gx + gy * gy + gz * gz) -
        (fx * gx + fy * gy + fz * gz) *
        (fx * gx + fy * gy + fz * gz);
    return area ;
}

/******************************************/

static
Real_t CalcElemCharacteristicLength( const Real_t x[8],
        const Real_t y[8],
        const Real_t z[8],
        const Real_t volume)
{
    Real_t a, charLength = Real_t(0.0);

    a = AreaFace(x[0],x[1],x[2],x[3],
            y[0],y[1],y[2],y[3],
            z[0],z[1],z[2],z[3]) ;
    charLength = std::max(a,charLength) ;

    a = AreaFace(x[4],x[5],x[6],x[7],
            y[4],y[5],y[6],y[7],
            z[4],z[5],z[6],z[7]) ;
    charLength = std::max(a,charLength) ;

    a = AreaFace(x[0],x[1],x[5],x[4],
            y[0],y[1],y[5],y[4],
            z[0],z[1],z[5],z[4]) ;
    charLength = std::max(a,charLength) ;

    a = AreaFace(x[1],x[2],x[6],x[5],
            y[1],y[2],y[6],y[5],
            z[1],z[2],z[6],z[5]) ;
    charLength = std::max(a,charLength) ;

    a = AreaFace(x[2],x[3],x[7],x[6],
            y[2],y[3],y[7],y[6],
            z[2],z[3],z[7],z[6]) ;
    charLength = std::max(a,charLength) ;

    a = AreaFace(x[3],x[0],x[4],x[7],
            y[3],y[0],y[4],y[7],
            z[3],z[0],z[4],z[7]) ;
    charLength = std::max(a,charLength) ;

    charLength = Real_t(4.0) * volume / SQRT(charLength);

    return charLength;
}

/******************************************/

static
void CalcElemVelocityGradient( const Real_t* const xvel,
        const Real_t* const yvel,
        const Real_t* const zvel,
        const Real_t b[][8],
        const Real_t detJ,
        Real_t* const d )
{
    const Real_t inv_detJ = Real_t(1.0) / detJ ;
    Real_t dyddx, dxddy, dzddx, dxddz, dzddy, dyddz;
    const Real_t* const pfx = b[0];
    const Real_t* const pfy = b[1];
    const Real_t* const pfz = b[2];

    d[0] = inv_detJ * ( pfx[0] * (xvel[0]-xvel[6])
            + pfx[1] * (xvel[1]-xvel[7])
            + pfx[2] * (xvel[2]-xvel[4])
            + pfx[3] * (xvel[3]-xvel[5]) );

    d[1] = inv_detJ * ( pfy[0] * (yvel[0]-yvel[6])
            + pfy[1] * (yvel[1]-yvel[7])
            + pfy[2] * (yvel[2]-yvel[4])
            + pfy[3] * (yvel[3]-yvel[5]) );

    d[2] = inv_detJ * ( pfz[0] * (zvel[0]-zvel[6])
            + pfz[1] * (zvel[1]-zvel[7])
            + pfz[2] * (zvel[2]-zvel[4])
            + pfz[3] * (zvel[3]-zvel[5]) );

    dyddx  = inv_detJ * ( pfx[0] * (yvel[0]-yvel[6])
            + pfx[1] * (yvel[1]-yvel[7])
            + pfx[2] * (yvel[2]-yvel[4])
            + pfx[3] * (yvel[3]-yvel[5]) );

    dxddy  = inv_detJ * ( pfy[0] * (xvel[0]-xvel[6])
            + pfy[1] * (xvel[1]-xvel[7])
            + pfy[2] * (xvel[2]-xvel[4])
            + pfy[3] * (xvel[3]-xvel[5]) );

    dzddx  = inv_detJ * ( pfx[0] * (zvel[0]-zvel[6])
            + pfx[1] * (zvel[1]-zvel[7])
            + pfx[2] * (zvel[2]-zvel[4])
            + pfx[3] * (zvel[3]-zvel[5]) );

    dxddz  = inv_detJ * ( pfz[0] * (xvel[0]-xvel[6])
            + pfz[1] * (xvel[1]-xvel[7])
            + pfz[2] * (xvel[2]-xvel[4])
            + pfz[3] * (xvel[3]-xvel[5]) );

    dzddy  = inv_detJ * ( pfy[0] * (zvel[0]-zvel[6])
            + pfy[1] * (zvel[1]-zvel[7])
            + pfy[2] * (zvel[2]-zvel[4])
            + pfy[3] * (zvel[3]-zvel[5]) );

    dyddz  = inv_detJ * ( pfz[0] * (yvel[0]-yvel[6])
            + pfz[1] * (yvel[1]-yvel[7])
            + pfz[2] * (yvel[2]-yvel[4])
            + pfz[3] * (yvel[3]-yvel[5]) );
    d[5]  = Real_t( .5) * ( dxddy + dyddx );
    d[4]  = Real_t( .5) * ( dxddz + dzddx );
    d[3]  = Real_t( .5) * ( dzddy + dyddz );
}

/******************************************/

static
void CalcKinematicsForElems(Domain * domain)
{
    /* OpenMP dependencies */
    const Real_t * domain_v         = domain->m_v.data();       (void) domain_v;
    const Real_t * domain_delv      = domain->m_delv.data();    (void) domain_delv;
    const Real_t * domain_arealg    = domain->m_arealg.data();  (void) domain_arealg;

    Index_t numElem = domain->numElem();
    for (Index_t b = 0; b < numElem ; b += EBS)
    {
        mpc_omp_task_dependency_t dependencies[2];
        dependencies[0] = dependencies_domain_x_y_z[b/EBS];
        dependencies[1] = dependencies_domain_xd_yd_zd[b/EBS];
        MPC_PUSH_DEPENDENCIES(dependencies, 2);
        TASK_SET_COLOR(iter);
        TASK_SET_LABEL("CalcKinematicsForElems");
        #pragma omp task default(none)                                              \
            firstprivate(domain, b, numElem, iter)                                  \
            shared(EBS, vnew)                                                       \
            depend(in:  domain->m_deltatime, domain_v[b])                           \
            depend(out: vnew[b],            domain_delv[b],     domain_arealg[b],   \
                        domain->m_dxx[b])
        {
            double dt = domain->deltatime();
            Index_t start = b;
            Index_t end = MIN(start + EBS, numElem);
            for (Index_t i = start ; i < end ; ++i)
            {
                // get nodal coordinates from global arrays and copy into local arrays.
                Real_t x_local[8] ;
                Real_t y_local[8] ;
                Real_t z_local[8] ;
                const Index_t * const elemToNode = domain->nodelist(i);
                CollectDomainNodesToElemNodes(domain, elemToNode, x_local, y_local, z_local);

                // volume calculations
                Real_t volume = CalcElemVolume(x_local, y_local, z_local);

                Real_t relativeVolume = volume / domain->volo(i);
                vnew[i] = MIN(MAX(relativeVolume, domain->eosvmin()), domain->eosvmax());

                // See if any volumes are negative, and take appropriate action.
                if (vnew[i] <= Real_t(0.0)) lulesh_abort();

                domain->delv(i) = relativeVolume - domain->v(i) ;

                // set characteristic length
                domain->arealg(i) = CalcElemCharacteristicLength(x_local, y_local, z_local, volume);

                // get nodal velocities from global array and copy into local arrays.
                Real_t xd_local[8] ;
                Real_t yd_local[8] ;
                Real_t zd_local[8] ;
                for (Index_t lnode = 0 ; lnode < 8 ; ++lnode)
                {
                    Index_t gnode = elemToNode[lnode];
                    xd_local[lnode] = domain->xd(gnode);
                    yd_local[lnode] = domain->yd(gnode);
                    zd_local[lnode] = domain->zd(gnode);
                }

                Real_t dt2 = Real_t(0.5) * dt;
                for (Index_t j = 0 ; j < 8 ; ++j)
                {
                    x_local[j] -= dt2 * xd_local[j];
                    y_local[j] -= dt2 * yd_local[j];
                    z_local[j] -= dt2 * zd_local[j];
                }

                Real_t B[3][8] ; /** shape function derivatives */
                Real_t detJ = Real_t(0.0) ;
                CalcElemShapeFunctionDerivatives(x_local, y_local, z_local, B, &detJ);

                Real_t D[6] ;
                CalcElemVelocityGradient(xd_local, yd_local, zd_local, B, detJ, D);

                // put velocity gradient quantities into their global arrays.
                domain->dxx(i) = D[0];
                domain->dyy(i) = D[1];
                domain->dzz(i) = D[2];
            }
        }
    }
}

/******************************************/

static
void CalcLagrangeElements(Domain * domain)
{
    Index_t numElem = domain->numElem();
    assert(numElem > 0);

    CalcKinematicsForElems(domain);

    const Real_t * domain_vdov  = domain->m_vdov.data();  (void) domain_vdov;

    // element loop to do some stuff not included in the elemlib function.
    for (Index_t b = 0; b < numElem ; b += EBS)
    {
        TASK_SET_COLOR(iter);
        TASK_SET_LABEL("CalcLagrangeElements");
        #pragma omp task default(none)          \
            firstprivate(domain, b, numElem)    \
            shared(EBS)                         \
            depend(inout:   domain->m_dxx[b])   \
            depend(out:     domain_vdov[b])
        {
            Index_t start = b;
            Index_t end = MIN(start + EBS, numElem);
            for (Index_t k = start ; k < end ; ++k)
            {
                // calc strain rate and apply as constraint (only done in FB element)
                Real_t vdov = domain->dxx(k) + domain->dyy(k) + domain->dzz(k) ;
                Real_t vdovthird = vdov/Real_t(3.0) ;

                // make the rate of deformation tensor deviatoric
                domain->vdov(k) = vdov ;
                domain->dxx(k) -= vdovthird ;
                domain->dyy(k) -= vdovthird ;
                domain->dzz(k) -= vdovthird ;
            }
        }
    }
}

/******************************************/

static
void CalcMonotonicQGradientsForElems(Domain * domain)
{
    Index_t numElem = domain->numElem();

    const Real_t * domain_delx_zeta = domain->m_delx_zeta;   (void) domain_delx_zeta;
    const Real_t * domain_delx_xi   = domain->m_delx_xi;     (void) domain_delx_xi;
    const Real_t * domain_delx_eta  = domain->m_delx_eta;    (void) domain_delx_eta;
    const Real_t * domain_delv_zeta = domain->m_delv_zeta;   (void) domain_delv_zeta;
    const Real_t * domain_delv_xi   = domain->m_delv_xi;     (void) domain_delv_xi;
    const Real_t * domain_delv_eta  = domain->m_delv_eta;    (void) domain_delv_eta;

    // element loop to do some stuff not included in the elemlib function.
    for (Index_t b = 0; b < numElem ; b += EBS)
    {
        mpc_omp_task_dependency_t dependencies[2];
        dependencies[0] = dependencies_domain_x_y_z[b/EBS];
        dependencies[1] = dependencies_domain_xd_yd_zd[b/EBS];
        MPC_PUSH_DEPENDENCIES(dependencies, 2);
        TASK_SET_COLOR(iter);
        TASK_SET_LABEL("CalcMonotonicQGradientsForElems");
        #pragma omp task default(none)                                  \
            firstprivate(domain, b, numElem)                            \
            shared(EBS, vnew, ptiny)                                    \
            depend(out: domain_delx_xi[b],      domain_delv_xi[b],      \
                        domain_delx_eta[b],     domain_delv_eta[b],     \
                        domain_delx_zeta[b],    domain_delv_zeta[b])    \
            depend(in: vnew[b])

        {
            Index_t start = b;
            Index_t end = MIN(start + EBS, numElem);
            for (Index_t i = start ; i < end ; ++i)
            {
                Real_t ax,ay,az ;
                Real_t dxv,dyv,dzv ;

                const Index_t * elemToNode = domain->nodelist(i);
                Index_t n0 = elemToNode[0] ;
                Index_t n1 = elemToNode[1] ;
                Index_t n2 = elemToNode[2] ;
                Index_t n3 = elemToNode[3] ;
                Index_t n4 = elemToNode[4] ;
                Index_t n5 = elemToNode[5] ;
                Index_t n6 = elemToNode[6] ;
                Index_t n7 = elemToNode[7] ;

                Real_t x0 = domain->x(n0) ;
                Real_t x1 = domain->x(n1) ;
                Real_t x2 = domain->x(n2) ;
                Real_t x3 = domain->x(n3) ;
                Real_t x4 = domain->x(n4) ;
                Real_t x5 = domain->x(n5) ;
                Real_t x6 = domain->x(n6) ;
                Real_t x7 = domain->x(n7) ;

                Real_t y0 = domain->y(n0) ;
                Real_t y1 = domain->y(n1) ;
                Real_t y2 = domain->y(n2) ;
                Real_t y3 = domain->y(n3) ;
                Real_t y4 = domain->y(n4) ;
                Real_t y5 = domain->y(n5) ;
                Real_t y6 = domain->y(n6) ;
                Real_t y7 = domain->y(n7) ;

                Real_t z0 = domain->z(n0) ;
                Real_t z1 = domain->z(n1) ;
                Real_t z2 = domain->z(n2) ;
                Real_t z3 = domain->z(n3) ;
                Real_t z4 = domain->z(n4) ;
                Real_t z5 = domain->z(n5) ;
                Real_t z6 = domain->z(n6) ;
                Real_t z7 = domain->z(n7) ;

                Real_t xv0 = domain->xd(n0) ;
                Real_t xv1 = domain->xd(n1) ;
                Real_t xv2 = domain->xd(n2) ;
                Real_t xv3 = domain->xd(n3) ;
                Real_t xv4 = domain->xd(n4) ;
                Real_t xv5 = domain->xd(n5) ;
                Real_t xv6 = domain->xd(n6) ;
                Real_t xv7 = domain->xd(n7) ;

                Real_t yv0 = domain->yd(n0) ;
                Real_t yv1 = domain->yd(n1) ;
                Real_t yv2 = domain->yd(n2) ;
                Real_t yv3 = domain->yd(n3) ;
                Real_t yv4 = domain->yd(n4) ;
                Real_t yv5 = domain->yd(n5) ;
                Real_t yv6 = domain->yd(n6) ;
                Real_t yv7 = domain->yd(n7) ;

                Real_t zv0 = domain->zd(n0) ;
                Real_t zv1 = domain->zd(n1) ;
                Real_t zv2 = domain->zd(n2) ;
                Real_t zv3 = domain->zd(n3) ;
                Real_t zv4 = domain->zd(n4) ;
                Real_t zv5 = domain->zd(n5) ;
                Real_t zv6 = domain->zd(n6) ;
                Real_t zv7 = domain->zd(n7) ;

                Real_t vol = domain->volo(i)*vnew[i] ;
                Real_t norm = Real_t(1.0) / ( vol + ptiny ) ;

                Real_t dxj = Real_t(-0.25)*((x0+x1+x5+x4) - (x3+x2+x6+x7)) ;
                Real_t dyj = Real_t(-0.25)*((y0+y1+y5+y4) - (y3+y2+y6+y7)) ;
                Real_t dzj = Real_t(-0.25)*((z0+z1+z5+z4) - (z3+z2+z6+z7)) ;

                Real_t dxi = Real_t( 0.25)*((x1+x2+x6+x5) - (x0+x3+x7+x4)) ;
                Real_t dyi = Real_t( 0.25)*((y1+y2+y6+y5) - (y0+y3+y7+y4)) ;
                Real_t dzi = Real_t( 0.25)*((z1+z2+z6+z5) - (z0+z3+z7+z4)) ;

                Real_t dxk = Real_t( 0.25)*((x4+x5+x6+x7) - (x0+x1+x2+x3)) ;
                Real_t dyk = Real_t( 0.25)*((y4+y5+y6+y7) - (y0+y1+y2+y3)) ;
                Real_t dzk = Real_t( 0.25)*((z4+z5+z6+z7) - (z0+z1+z2+z3)) ;

                /* find delvk and delxk ( i cross j ) */

                ax = dyi*dzj - dzi*dyj ;
                ay = dzi*dxj - dxi*dzj ;
                az = dxi*dyj - dyi*dxj ;

                domain->delx_zeta(i) = vol / SQRT(ax*ax + ay*ay + az*az + ptiny) ;

                ax *= norm ;
                ay *= norm ;
                az *= norm ;

                dxv = Real_t(0.25)*((xv4+xv5+xv6+xv7) - (xv0+xv1+xv2+xv3)) ;
                dyv = Real_t(0.25)*((yv4+yv5+yv6+yv7) - (yv0+yv1+yv2+yv3)) ;
                dzv = Real_t(0.25)*((zv4+zv5+zv6+zv7) - (zv0+zv1+zv2+zv3)) ;

                domain->delv_zeta(i) = ax*dxv + ay*dyv + az*dzv ;

                /* find delxi and delvi ( j cross k ) */

                ax = dyj*dzk - dzj*dyk ;
                ay = dzj*dxk - dxj*dzk ;
                az = dxj*dyk - dyj*dxk ;

                domain->delx_xi(i) = vol / SQRT(ax*ax + ay*ay + az*az + ptiny) ;

                ax *= norm ;
                ay *= norm ;
                az *= norm ;

                dxv = Real_t(0.25)*((xv1+xv2+xv6+xv5) - (xv0+xv3+xv7+xv4)) ;
                dyv = Real_t(0.25)*((yv1+yv2+yv6+yv5) - (yv0+yv3+yv7+yv4)) ;
                dzv = Real_t(0.25)*((zv1+zv2+zv6+zv5) - (zv0+zv3+zv7+zv4)) ;

                domain->delv_xi(i) = ax*dxv + ay*dyv + az*dzv ;

                /* find delxj and delvj ( k cross i ) */

                ax = dyk*dzi - dzk*dyi ;
                ay = dzk*dxi - dxk*dzi ;
                az = dxk*dyi - dyk*dxi ;

                domain->delx_eta(i) = vol / SQRT(ax*ax + ay*ay + az*az + ptiny) ;

                ax *= norm ;
                ay *= norm ;
                az *= norm ;

                dxv = Real_t(-0.25)*((xv0+xv1+xv5+xv4) - (xv3+xv2+xv6+xv7)) ;
                dyv = Real_t(-0.25)*((yv0+yv1+yv5+yv4) - (yv3+yv2+yv6+yv7)) ;
                dzv = Real_t(-0.25)*((zv0+zv1+zv5+zv4) - (zv3+zv2+zv6+zv7)) ;

                domain->delv_eta(i) = ax*dxv + ay*dyv + az*dzv ;
            }
        }
    }
}

/******************************************/

static
void CalcMonotonicQRegionForElems(Domain * domain, Int_t r)
{
    // element loop to do some stuff not included in the elemlib function.
    Index_t regElemSize = domain->regElemSize(r);
    for (Index_t b = 0; b < regElemSize ; b += EBS)
    {
        MPC_PUSH_DEPENDENCIES(CalcMonotonicQRegionForElems_deps[r] + 2*(b/EBS), 2);
        TASK_SET_COLOR(iter);
        TASK_SET_LABEL("CalcMonotonicQRegionForElems");
        #pragma omp task default(none)                      \
            firstprivate(domain, b, regElemSize, r, iter)   \
            shared(stderr, vnew, EBS, ptiny)
        {
            Real_t monoq_limiter_mult = domain->monoq_limiter_mult();
            Real_t monoq_max_slope = domain->monoq_max_slope();
            Real_t qlc_monoq = domain->qlc_monoq();
            Real_t qqc_monoq = domain->qqc_monoq();

            Index_t start = b;
            Index_t end = MIN(start + EBS, regElemSize);
            for (Index_t ielem = start ; ielem < end ; ++ielem)
            {
                Index_t i = domain->regElemlist(r,ielem);
                Real_t qlin, qquad ;
                Real_t phixi, phieta, phizeta ;
                Int_t bcMask = domain->elemBC(i) ;
                Real_t delvm = 0.0, delvp =0.0;

                /*  phixi     */
                Real_t norm = Real_t(1.) / (domain->delv_xi(i) + ptiny);

                switch (bcMask & XI_M)
                {
                    case XI_M_COMM: /* needs comm data */
                    case 0:         delvm = domain->delv_xi(domain->lxim(i));   break ;
                    case XI_M_SYMM: delvm = domain->delv_xi(i) ;                break ;
                    case XI_M_FREE: delvm = Real_t(0.0) ;                       break ;
                    default:          fprintf(stderr, "Error in switch at %s line %d\n",
                                              __FILE__, __LINE__);
                                      delvm = 0; /* ERROR - but quiets the compiler */
                                      break;
                }
                switch (bcMask & XI_P)
                {
                    case XI_P_COMM: /* needs comm data */
                    case 0:         delvp = domain->delv_xi(domain->lxip(i)) ; break ;
                    case XI_P_SYMM: delvp = domain->delv_xi(i) ;       break ;
                    case XI_P_FREE: delvp = Real_t(0.0) ;      break ;
                    default:          fprintf(stderr, "Error in switch at %s line %d\n",
                                              __FILE__, __LINE__);
                                      delvp = 0; /* ERROR - but quiets the compiler */
                                      break;
                }

                delvm = delvm * norm ;
                delvp = delvp * norm ;

                phixi = Real_t(.5) * ( delvm + delvp ) ;

                delvm *= monoq_limiter_mult ;
                delvp *= monoq_limiter_mult ;

                if ( delvm < phixi ) phixi = delvm ;
                if ( delvp < phixi ) phixi = delvp ;
                if ( phixi < Real_t(0.)) phixi = Real_t(0.) ;
                if ( phixi > monoq_max_slope) phixi = monoq_max_slope;


                /*  phieta     */
                norm = Real_t(1.) / ( domain->delv_eta(i) + ptiny ) ;

                switch (bcMask & ETA_M) {
                    case ETA_M_COMM: /* needs comm data */
                    case 0:          delvm = domain->delv_eta(domain->letam(i)) ; break ;
                    case ETA_M_SYMM: delvm = domain->delv_eta(i) ;        break ;
                    case ETA_M_FREE: delvm = Real_t(0.0) ;        break ;
                    default:          fprintf(stderr, "Error in switch at %s line %d\n",
                                              __FILE__, __LINE__);
                                      delvm = 0; /* ERROR - but quiets the compiler */
                                      break;
                }
                switch (bcMask & ETA_P) {
                    case ETA_P_COMM: /* needs comm data */
                    case 0:          delvp = domain->delv_eta(domain->letap(i)) ; break ;
                    case ETA_P_SYMM: delvp = domain->delv_eta(i) ;        break ;
                    case ETA_P_FREE: delvp = Real_t(0.0) ;        break ;
                    default:          fprintf(stderr, "Error in switch at %s line %d\n",
                                              __FILE__, __LINE__);
                                      delvp = 0; /* ERROR - but quiets the compiler */
                                      break;
                }

                delvm = delvm * norm ;
                delvp = delvp * norm ;

                phieta = Real_t(.5) * ( delvm + delvp ) ;

                delvm *= monoq_limiter_mult ;
                delvp *= monoq_limiter_mult ;

                if ( delvm  < phieta ) phieta = delvm ;
                if ( delvp  < phieta ) phieta = delvp ;
                if ( phieta < Real_t(0.)) phieta = Real_t(0.) ;
                if ( phieta > monoq_max_slope)  phieta = monoq_max_slope;

                /*  phizeta     */
                norm = Real_t(1.) / ( domain->delv_zeta(i) + ptiny ) ;

                switch (bcMask & ZETA_M) {
                    case ZETA_M_COMM: /* needs comm data */
                    case 0:           delvm = domain->delv_zeta(domain->lzetam(i)) ; break ;
                    case ZETA_M_SYMM: delvm = domain->delv_zeta(i) ;         break ;
                    case ZETA_M_FREE: delvm = Real_t(0.0) ;          break ;
                    default:          fprintf(stderr, "Error in switch at %s line %d\n",
                                              __FILE__, __LINE__);
                                      delvm = 0; /* ERROR - but quiets the compiler */
                                      break;
                }
                switch (bcMask & ZETA_P) {
                    case ZETA_P_COMM: /* needs comm data */
                    case 0:           delvp = domain->delv_zeta(domain->lzetap(i)) ; break ;
                    case ZETA_P_SYMM: delvp = domain->delv_zeta(i) ;         break ;
                    case ZETA_P_FREE: delvp = Real_t(0.0) ;          break ;
                    default:          fprintf(stderr, "Error in switch at %s line %d\n",
                                              __FILE__, __LINE__);
                                      delvp = 0; /* ERROR - but quiets the compiler */
                                      break;
                }

                delvm = delvm * norm ;
                delvp = delvp * norm ;

                phizeta = Real_t(.5) * ( delvm + delvp ) ;

                delvm *= monoq_limiter_mult ;
                delvp *= monoq_limiter_mult ;

                if ( delvm   < phizeta ) phizeta = delvm ;
                if ( delvp   < phizeta ) phizeta = delvp ;
                if ( phizeta < Real_t(0.)) phizeta = Real_t(0.);
                if ( phizeta > monoq_max_slope  ) phizeta = monoq_max_slope;

                /* Remove length scale */

                if ( domain->vdov(i) > Real_t(0.) )  {
                    qlin  = Real_t(0.) ;
                    qquad = Real_t(0.) ;
                }
                else {
                    Real_t delvxxi   = domain->delv_xi(i)   * domain->delx_xi(i)   ;
                    Real_t delvxeta  = domain->delv_eta(i)  * domain->delx_eta(i)  ;
                    Real_t delvxzeta = domain->delv_zeta(i) * domain->delx_zeta(i) ;

                    if ( delvxxi   > Real_t(0.) ) delvxxi   = Real_t(0.) ;
                    if ( delvxeta  > Real_t(0.) ) delvxeta  = Real_t(0.) ;
                    if ( delvxzeta > Real_t(0.) ) delvxzeta = Real_t(0.) ;

                    Real_t rho = domain->elemMass(i) / (domain->volo(i) * vnew[i]) ;

                    qlin = -qlc_monoq * rho *
                        (  delvxxi   * (Real_t(1.) - phixi) +
                           delvxeta  * (Real_t(1.) - phieta) +
                           delvxzeta * (Real_t(1.) - phizeta)  ) ;

                    qquad = qqc_monoq * rho *
                        (  delvxxi*delvxxi     * (Real_t(1.) - phixi*phixi) +
                           delvxeta*delvxeta   * (Real_t(1.) - phieta*phieta) +
                           delvxzeta*delvxzeta * (Real_t(1.) - phizeta*phizeta)  ) ;
                }

                domain->qq(i) = qquad ;
                domain->ql(i) = qlin  ;
            } /* for ielem */
        } /* task */
    } /* for b */
}

static
void CalcMonotonicQForElems(Domain * domain)
{
    // calculate the monotonic q for all regions
    //
    const Index_t numReg = domain->numReg();
    for (Index_t r = 0 ; r < numReg ; ++r)
    {
        if (domain->regElemSize(r) > 0)
        {
            CalcMonotonicQRegionForElems(domain, r);
        }
    }

# if PRUNE_INOUTSET_EXPLICITLY
    // insert inoutset reduce tasks on 'domain_qq'
    const Real_t * domain_qq = domain->m_qq.data(); (void) domain_qq;
    const Index_t numElem = domain->numElem();
    for (Index_t b = 0; b < numElem ; b += EBS)
    {
        TASK_SET_COLOR(iter);
        TASK_SET_LABEL("EMPTY(domain_qq)");
        # pragma omp task depend(inout: domain_qq[b])
        {}
    }
# endif /* PRUNE_INOUTSET_EXPLICITLY */
}

/******************************************/

/**
 * MPI communication graph
 *
 *      Recv                Compute
 *       |                     |
 *       |                     |
 *       |                     v
 *       |                    Pack
 *       |                    /|
 *       v                   / |
 *     Unpack <--------------  |
 *                             |
 *                             v
 */

static
void CalcQForElems(Domain * domain)
{
    //
    // MONOTONIC Q option
    //

    Index_t numElem = domain->numElem();
    if (numElem != 0)
    {
#if USE_MPI
        CommRecv(domain, MSG_DELV);
#endif
        /* Calculate velocity gradients */
        CalcMonotonicQGradientsForElems(domain);

#if USE_MPI
        CommPack  (domain, MSG_DELV);
        CommSend  (domain, MSG_DELV);
        CommUnpack(domain, MSG_DELV);
#endif /* USE_MPI */

        CalcMonotonicQForElems(domain);
    }
}

/******************************************/

static inline
void CalcPressureForElems(Real_t* p_new, Real_t* bvc,
        Real_t* pbvc, Real_t* e_old,
        Real_t* compression, Real_t *vnew,
        Real_t pmin,
        Real_t p_cut, Real_t eosvmax,
        Index_t regElemSize, Index_t *regElemList)
{
    for (Index_t b = 0; b < regElemSize ; b += EBS)
    {
        TASK_SET_COLOR(iter);
        TASK_SET_LABEL("CalcPressureForElems1");
        # pragma omp task default(none)                                         \
            firstprivate(b, regElemSize, regElemList, compression, bvc, pbvc)   \
            shared(EBS)                                                         \
            depend(in: compression[b])                                          \
            depend(out: bvc[b])
        {
            Real_t c1s = Real_t(2.0)/Real_t(3.0) ;
            Index_t start = b;
            Index_t end = MIN(start + EBS, regElemSize);
            for (Index_t i = start ; i < end ; ++i)
            {
                bvc[i] = c1s * (compression[i] + Real_t(1.));
                pbvc[i] = c1s;
            }
        }
    }

    for (Index_t b = 0; b < regElemSize ; b += EBS)
    {
        TASK_SET_COLOR(iter);
        TASK_SET_LABEL("CalcPressureForElems2");
        # pragma omp task default(none)                             \
            firstprivate(   b, regElemSize, regElemList, p_new,     \
                            bvc, e_old, p_cut, pmin, eosvmax, vnew) \
            shared(EBS)                                             \
            depend(in: bvc[b], e_old[b])                            \
            depend(out: p_new[b])
        {
            Index_t start = b;
            Index_t end = MIN(start + EBS, regElemSize);
            for (Index_t i = start ; i < end ; ++i)
            {
                p_new[i] = bvc[i] * e_old[i] ;

                if    (FABS(p_new[i]) <  p_cut   )
                    p_new[i] = Real_t(0.0) ;

                Index_t elem = regElemList[i];
                if    ( vnew[elem] >= eosvmax ) /* impossible condition here? */
                    p_new[i] = Real_t(0.0) ;

                if    (p_new[i]       <  pmin)
                    p_new[i]   = pmin ;
            }
        }
    }
}

/******************************************/

static inline
void CalcEnergyForElems(Real_t* p_new, Real_t* e_new, Real_t* q_new,
        Real_t* bvc, Real_t* pbvc,
        Real_t* p_old, Real_t* e_old, Real_t* q_old,
        Real_t* compression, Real_t* compHalfStep,
        Real_t* vnew, Real_t* work, Real_t* delvc, Real_t pmin,
        Real_t p_cut, Real_t  e_cut, Real_t q_cut, Real_t emin,
        Real_t* qq_old, Real_t* ql_old, Real_t * pHalfStep,
        Real_t rho0,
        Real_t eosvmax,
        Index_t regElemSize, Index_t *regElemList)
{

    for (Index_t b = 0; b < regElemSize ; b += EBS)
    {
        TASK_SET_COLOR(iter);
        TASK_SET_LABEL("CalcEnergyForElems1");
        # pragma omp task default(none)                                                 \
            firstprivate(b, regElemSize, delvc, e_old, p_old, q_old, work, e_new, emin) \
            shared(EBS)                                                                 \
            depend(in: e_old[b], delvc[b], p_old[b], q_old[b], work[b])                 \
            depend(out: e_new[b])
        {
            Index_t start = b;
            Index_t end = MIN(start + EBS, regElemSize);
            for (Index_t i = start ; i < end ; ++i)
            {
                e_new[i] = e_old[i] - Real_t(0.5) * delvc[i] * (p_old[i] + q_old[i])
                    + Real_t(0.5) * work[i];

                if (e_new[i]  < emin ) {
                    e_new[i] = emin ;
                }
            }
        }
    }

    CalcPressureForElems(pHalfStep, bvc, pbvc, e_new, compHalfStep, vnew,
            pmin, p_cut, eosvmax, regElemSize, regElemList);

    for (Index_t b = 0; b < regElemSize ; b += EBS)
    {
        TASK_SET_COLOR(iter);
        TASK_SET_LABEL("CalcEnergyForElems2");
        # pragma omp task default(none)                                                     \
            firstprivate(   b, regElemSize, compHalfStep, q_new, pbvc, e_new,               \
                            delvc, p_old, q_old, pHalfStep, bvc, rho0, ql_old, qq_old)      \
            shared(EBS)                                                                     \
            depend(in: compHalfStep[b], delvc[b], e_new[b], bvc[b], pHalfStep[b], delvc[b]) \
            depend(out: q_new[b], e_new[b])
        {
            Index_t start = b;
            Index_t end = MIN(start + EBS, regElemSize);
            for (Index_t i = start ; i < end ; ++i)
            {
                Real_t vhalf = Real_t(1.) / (Real_t(1.) + compHalfStep[i]) ;

                if ( delvc[i] > Real_t(0.) ) {
                    q_new[i] /* = qq_old[i] = ql_old[i] */ = Real_t(0.) ;
                }
                else {
                    Real_t ssc = ( pbvc[i] * e_new[i]
                            + vhalf * vhalf * bvc[i] * pHalfStep[i] ) / rho0 ;

                    if ( ssc <= Real_t(.1111111e-36) ) {
                        ssc = Real_t(.3333333e-18) ;
                    } else {
                        ssc = SQRT(ssc) ;
                    }

                    q_new[i] = (ssc*ql_old[i] + qq_old[i]) ;
                }

                e_new[i] = e_new[i] + Real_t(0.5) * delvc[i]
                    * (  Real_t(3.0)*(p_old[i]     + q_old[i])
                            - Real_t(4.0)*(pHalfStep[i] + q_new[i])) ;
            }
        }
    }

    for (Index_t b = 0; b < regElemSize ; b += EBS)
    {
        TASK_SET_COLOR(iter);
        TASK_SET_LABEL("CalcEnergyForElems3");
        # pragma omp task default(none)                             \
            firstprivate(b, regElemSize, e_new, work, e_cut, emin)  \
            shared(EBS)                                             \
            depend(in: work[b])                                     \
            depend(out: e_new[b])
        {
            Index_t start = b;
            Index_t end = MIN(start + EBS, regElemSize);
            for (Index_t i = start ; i < end ; ++i)
            {
                e_new[i] += Real_t(0.5) * work[i];

                if (FABS(e_new[i]) < e_cut) {
                    e_new[i] = Real_t(0.)  ;
                }
                if (     e_new[i]  < emin ) {
                    e_new[i] = emin ;
                }
            }
        }
    }

    CalcPressureForElems(p_new, bvc, pbvc, e_new, compression, vnew,
            pmin, p_cut, eosvmax, regElemSize, regElemList);

    for (Index_t b = 0; b < regElemSize ; b += EBS)
    {
        TASK_SET_COLOR(iter);
        TASK_SET_LABEL("CalcEnergyForElems4");
        # pragma omp task default(none)                                                         \
            firstprivate(   b, regElemSize, regElemList, delvc, pbvc, e_new, bvc, p_new, rho0,  \
                            ql_old, qq_old, p_old, pHalfStep, e_cut, emin, q_old, q_new, vnew)  \
            shared(EBS)                                                                         \
            depend(in: delvc[b], bvc[b], p_new[b], qq_old[b])                                   \
            depend(inout: e_new[b])
        {
            Index_t start = b;
            Index_t end = MIN(start + EBS, regElemSize);
            for (Index_t i = start ; i < end ; ++i)
            {
                const Real_t sixth = Real_t(1.0) / Real_t(6.0) ;
                Index_t elem = regElemList[i];
                Real_t q_tilde ;

                if (delvc[i] > Real_t(0.)) {
                    q_tilde = Real_t(0.) ;
                }
                else {
                    Real_t ssc = ( pbvc[i] * e_new[i]
                            + vnew[elem] * vnew[elem] * bvc[i] * p_new[i] ) / rho0 ;

                    if ( ssc <= Real_t(.1111111e-36) ) {
                        ssc = Real_t(.3333333e-18) ;
                    } else {
                        ssc = SQRT(ssc) ;
                    }

                    q_tilde = (ssc*ql_old[i] + qq_old[i]) ;
                }

                e_new[i] = e_new[i] - (  Real_t(7.0)*(p_old[i]     + q_old[i])
                        - Real_t(8.0)*(pHalfStep[i] + q_new[i])
                        + (p_new[i] + q_tilde)) * delvc[i]*sixth ;

                if (FABS(e_new[i]) < e_cut) {
                    e_new[i] = Real_t(0.)  ;
                }
                if (     e_new[i]  < emin ) {
                    e_new[i] = emin ;
                }
            }
        }
    }

    CalcPressureForElems(p_new, bvc, pbvc, e_new, compression, vnew,
            pmin, p_cut, eosvmax, regElemSize, regElemList);

    for (Index_t b = 0; b < regElemSize ; b += EBS)
    {
        TASK_SET_COLOR(iter);
        TASK_SET_LABEL("CalcEnergyForElems5");
        # pragma omp task default(none)                                             \
            firstprivate(   b, regElemSize, regElemList, pbvc, e_new, bvc, p_new,   \
                            rho0, ql_old, qq_old, q_new, delvc, q_cut, vnew)        \
            shared(EBS)                                                             \
            depend(in: delvc[b], e_new[b], bvc[b], p_new[b])                        \
            depend(inout: q_new[b])
        {
            Index_t start = b;
            Index_t end = MIN(start + EBS, regElemSize);
            for (Index_t i = start ; i < end ; ++i)
            {
                Index_t elem = regElemList[i];

                if ( delvc[i] <= Real_t(0.) ) {
                    Real_t ssc = ( pbvc[i] * e_new[i]
                            + vnew[elem] * vnew[elem] * bvc[i] * p_new[i] ) / rho0 ;

                    if ( ssc <= Real_t(.1111111e-36) ) {
                        ssc = Real_t(.3333333e-18) ;
                    } else {
                        ssc = SQRT(ssc) ;
                    }

                    q_new[i] = (ssc*ql_old[i] + qq_old[i]) ;

                    if (FABS(q_new[i]) < q_cut) q_new[i] = Real_t(0.) ;
                }
            }
        }
    }
}

/******************************************/

static inline
void CalcSoundSpeedForElems(Domain * domain,
        Real_t *vnew, Real_t rho0, Real_t *enewc,
        Real_t *pnewc, Real_t *pbvc,
        Real_t *bvc,
        Index_t regElemSize, Index_t *regElemList,
        Index_t r)
{
    for (Index_t b = 0; b < regElemSize ; b += EBS)
    {
        MPC_PUSH_DEPENDENCIES(CalcSoundSpeedForElems_deps[r] + b/EBS, 1);
        TASK_SET_COLOR(iter);
        TASK_SET_LABEL("CalcSoundSpeedForElems");
        # pragma omp task default(none)                                                             \
            firstprivate(b, regElemSize, regElemList, pbvc, enewc, vnew, bvc, pnewc, rho0, domain)  \
            shared(EBS)                                                                             \
            depend(in: enewc[b], bvc[b], pnewc[b])
        {
            Index_t start = b;
            Index_t end = MIN(start + EBS, regElemSize);
            for (Index_t i = start ; i < end ; ++i)
            {
                Index_t elem = regElemList[i];
                Real_t ssTmp = (pbvc[i] * enewc[i] + vnew[elem] * vnew[elem] *
                        bvc[i] * pnewc[i]) / rho0;
                if (ssTmp <= Real_t(.1111111e-36)) {
                    ssTmp = Real_t(.3333333e-18);
                }
                else {
                    ssTmp = SQRT(ssTmp);
                }
                domain->ss(elem) = ssTmp ;
            }
        }
    }
}

/******************************************/

static inline
void EvalEOSForElems(Domain * domain, Int_t r, Int_t rep)
{
    Index_t regElemSize = domain->regElemSize(r);
    Index_t * regElemList = domain->regElemlist(r);

    Real_t  e_cut = domain->e_cut() ;
    Real_t  p_cut = domain->p_cut() ;
    Real_t  q_cut = domain->q_cut() ;

    Real_t eosvmax = domain->eosvmax() ;
    Real_t eosvmin = domain->eosvmin() ;
    Real_t pmin    = domain->pmin() ;
    Real_t emin    = domain->emin() ;
    Real_t rho0    = domain->refdens() ;

    // These temporaries will be of different size for
    // each call (due to different sized region element
    // lists)
    Real_t * e_old          = e_oldRegs[r];
    Real_t * delvc          = delvcRegs[r];
    Real_t * p_old          = p_oldRegs[r];
    Real_t * q_old          = q_oldRegs[r];
    Real_t * compression    = compressionRegs[r];
    Real_t * compHalfStep   = compHalfStepRegs[r];
    Real_t * qq_old         = qq_oldRegs[r];
    Real_t * ql_old         = ql_oldRegs[r];
    Real_t * work           = workRegs[r];
    Real_t * p_new          = p_newRegs[r];
    Real_t * e_new          = e_newRegs[r];
    Real_t * q_new          = q_newRegs[r];
    Real_t * bvc            = bvcRegs[r];
    Real_t * pbvc           = pbvcRegs[r];
    Real_t * pHalfStep      = pHalfStepRegs[r];

    //loop to add load imbalance based on region number
    for (Int_t j = 0; j < rep; j++)
    {
        for (Index_t b = 0; b < regElemSize ; b += EBS)
        {
            MPC_PUSH_DEPENDENCIES(EvalEOSForElems_deps_1[r] + b/EBS, 1);
            TASK_SET_COLOR(iter);
            TASK_SET_LABEL("EvalEOSForElems1");
            # pragma omp task default(none)                                                 \
                firstprivate(   b, regElemSize, regElemList, domain,                        \
                                e_old, delvc, p_old, q_old, qq_old, ql_old)                 \
                shared(EBS)                                                                 \
                depend(out: e_old[b], delvc[b], p_old[b], q_old[b], qq_old[b])
            {
                Index_t start = b;
                Index_t end = MIN(start + EBS, regElemSize);
                for (Index_t i = start ; i < end ; ++i)
                {
                    Index_t elem = regElemList[i];
                    e_old[i] = domain->e(elem) ;
                    delvc[i] = domain->delv(elem) ;
                    p_old[i] = domain->p(elem) ;
                    q_old[i] = domain->q(elem) ;
                    qq_old[i] = domain->qq(elem) ;
                    ql_old[i] = domain->ql(elem) ;
                }
            }
        }

        for (Index_t b = 0; b < regElemSize ; b += EBS)
        {
            MPC_PUSH_DEPENDENCIES(vnew_in_deps[r] + b/EBS, 1);
            TASK_SET_COLOR(iter);
            TASK_SET_LABEL("EvalEOSForElems2");
            # pragma omp task default(none)                         \
                firstprivate(   b, regElemSize, regElemList,        \
                                compression, delvc, compHalfStep)   \
                shared(EBS, vnew)                                   \
                depend(in: delvc[b])                                \
                depend(out: compression[b], compHalfStep[b])
            {
                Index_t start = b;
                Index_t end = MIN(start + EBS, regElemSize);

                for (Index_t i = start ; i < end ; ++i)
                {
                    Index_t elem = regElemList[i];
                    Real_t vchalf ;
                    compression[i] = Real_t(1.) / vnew[elem] - Real_t(1.);
                    vchalf = vnew[elem] - delvc[i] * Real_t(.5);
                    compHalfStep[i] = Real_t(1.) / vchalf - Real_t(1.);
                }
            }
        }

        if (eosvmin != Real_t(0.))
        {
            for (Index_t b = 0; b < regElemSize ; b += EBS)
            {
                TASK_SET_COLOR(iter);
                TASK_SET_LABEL("EvalEOSForElems3");
                # pragma omp task default(none)                                 \
                    firstprivate(   b, regElemSize, regElemList, compHalfStep,  \
                                    compression, eosvmin)                       \
                    shared(EBS, vnew)                                           \
                    depend(in: compression[b])                                  \
                    depend(out: compHalfStep[b])
                {
                    Index_t start = b;
                    Index_t end = MIN(start + EBS, regElemSize);

                    for (Index_t i = start ; i < end ; ++i)
                    {
                        Index_t elem = regElemList[i];
                        if (vnew[elem] <= eosvmin) { /* impossible due to calling func? */
                            compHalfStep[i] = compression[i] ;
                        }
                    }
                }
            }
        }

        if (eosvmax != Real_t(0.))
        {
            for (Index_t b = 0; b < regElemSize ; b += EBS)
            {
                TASK_SET_COLOR(iter);
                TASK_SET_LABEL("EvalEOSForElems4");
                # pragma omp task default(none)                             \
                    firstprivate(   b, regElemSize, regElemList, p_old,     \
                                    compression, compHalfStep, eosvmax)     \
                    shared(EBS, vnew)                                       \
                    depend(out: p_old[b], compression[b], compHalfStep[b])
                {
                    Index_t start = b;
                    Index_t end = MIN(start + EBS, regElemSize);

                    for (Index_t i = start ; i < end ; ++i)
                    {
                        Index_t elem = regElemList[i];
                        if (vnew[elem] >= eosvmax) { /* impossible due to calling func? */
                            p_old[i]        = Real_t(0.) ;
                            compression[i]  = Real_t(0.) ;
                            compHalfStep[i] = Real_t(0.) ;
                        }
                    }
                }
            }
        }

        for (Index_t b = 0; b < regElemSize ; b += EBS)
        {
            TASK_SET_COLOR(iter);
            TASK_SET_LABEL("EvalEOSForElems5");
            # pragma omp task default(none)         \
                firstprivate(b, regElemSize, work)  \
                shared(EBS)                         \
                depend(out: work[b])
            {
                Index_t start = b;
                Index_t end = MIN(start + EBS, regElemSize);

                for (Index_t i = start ; i < end ; ++i)
                {
                    work[i] = Real_t(0.) ;
                }
            }
        }

        CalcEnergyForElems(p_new, e_new, q_new, bvc, pbvc,
                p_old, e_old,  q_old, compression, compHalfStep,
                vnew, work,  delvc, pmin,
                p_cut, e_cut, q_cut, emin,
                qq_old, ql_old, pHalfStep, rho0, eosvmax,
                regElemSize, regElemList);
    }

    for (Index_t b = 0; b < regElemSize ; b += EBS)
    {
        MPC_PUSH_DEPENDENCIES(EvalEOSForElems_deps_2[r] + b/EBS, 1);
        TASK_SET_COLOR(iter);
        TASK_SET_LABEL("EvalEOSForElems6");
        # pragma omp task default(none)                                             \
            firstprivate(b, regElemSize, regElemList, p_new, e_new, q_new, domain)  \
            shared(EBS)                                                             \
            depend(in: p_new[b], e_new[b], q_new[b])
        {
            Index_t start = b;
            Index_t end = MIN(start + EBS, regElemSize);
            for (Index_t i = start ; i < end ; ++i)
            {
                Index_t elem = regElemList[i];
                domain->p(elem) = p_new[i] ;
                domain->e(elem) = e_new[i] ;
                domain->q(elem) = q_new[i] ;

                /* Don't allow excessive artificial viscosity */
                if (domain->q(elem) > domain->qstop()) lulesh_abort();
            }
        }
    }

    CalcSoundSpeedForElems(domain,
            vnew, rho0, e_new, p_new,
            pbvc, bvc,
            regElemSize, regElemList, r);
}


/******************************************/

static
void ApplyMaterialPropertiesForElems(Domain * domain)
{
    Index_t numElem = domain->numElem();

    if (numElem)
    {
        // element loop to do some stuff not included in the elemlib function.
        for (Index_t b = 0; b < numElem ; b += EBS)
        {
            // This check may not make perfect sense in LULESH, but
            // it's representative of something in the full code -
            // just leave it in, please
            TASK_SET_COLOR(iter);
            TASK_SET_LABEL("ApplyMaterialPropertiesForElems");
            const Real_t * domain_v = domain->m_v.data(); (void)domain_v;
            #pragma omp task default(none)          \
                firstprivate(domain, b, numElem)    \
                shared(EBS)                         \
                depend(in: domain_v[b])
            {
                Index_t start = b;
                Index_t end = MIN(start + EBS, numElem);
                for (Index_t i = start ; i < end ; ++i)
                {
                    Real_t vc = MIN(MAX(domain->v(i), domain->eosvmin()), domain->eosvmax());
                    if (vc <= 0.)   lulesh_abort();
                }
            }
        }

        for (Int_t r = 0 ; r < domain->numReg() ; ++r)
        {
            Int_t rep;
            //Determine load imbalance for this region
            //round down the number with lowest cost
            if(r < domain->numReg()/2)
                rep = 1;
            //you don't get an expensive region unless you at least have 5 regions
            else if(r < (domain->numReg() - (domain->numReg()+15)/20))
                rep = 1 + domain->cost();
            //very expensive regions
            else
                rep = 10 * (1 + domain->cost());
            EvalEOSForElems(domain, r, rep);
        } // for regions

#if PRUNE_INOUTSET_EXPLICITLY
        // clean previous iteration INOUTSET on 'domain_e' and 'domain_ss'
        const Real_t * domain_e  = domain->m_e.data();  (void) domain_e;
        const Real_t * domain_ss = domain->m_ss.data(); (void) domain_ss;
        for (Index_t b = 0; b < numElem ; b += EBS)
        {
            TASK_SET_COLOR(iter);
            TASK_SET_LABEL("EMPTY(domain_e)");
            # pragma omp task depend(inout: domain_e[b])
            {}

            TASK_SET_COLOR(iter);
            TASK_SET_LABEL("EMPTY(domain_ss)");
            # pragma omp task depend(inout: domain_ss[b])
            {}
        }
#endif /* PRUNE_INOUTSET_EXPLICITLY */
    }
}

/******************************************/

    static
void UpdateVolumesForElems(Domain * domain)
{
    Index_t numElem = domain->numElem();
    if (numElem)
    {
        const Real_t * domain_v = domain->m_v.data(); (void)domain_v;
        for (Index_t b = 0; b < numElem ; b += EBS)
        {
            // Bound the updated relative volumes with eosvmin/max
            TASK_SET_COLOR(iter);
            TASK_SET_LABEL("UpdateVolumesForElems");
            #pragma omp task default(none)              \
                firstprivate(domain, b, numElem, vnew)  \
                shared(EBS)                             \
                depend(in: vnew[b])                     \
                depend(out: domain_v[b])
            {
                Real_t v_cut = domain->v_cut();
                Index_t start = b;
                Index_t end = MIN(start + EBS, numElem);
                for (Index_t i = start ; i < end ; ++i)
                {
                    Real_t tmpV = vnew[i] ;
                    if (FABS(tmpV - Real_t(1.0)) < v_cut) tmpV = Real_t(1.0) ;
                    domain->v(i) = tmpV ;

                    /* Do a check for negative volumes */
                    if (domain->v(i) <= Real_t(0.0))   lulesh_abort();
                }
            }
        }
    }
}

/******************************************/

static
void LagrangeElements(Domain * domain)
{
    /* new relative vol -- temp */
    CalcLagrangeElements(domain);

    /* Calculate Q.  (Monotonic q option requires communication) */
    CalcQForElems(domain);
    ApplyMaterialPropertiesForElems(domain) ;
    UpdateVolumesForElems(domain);
}

/******************************************/

static inline
void CalcCourantConstraintForElems(Domain * domain, Index_t r)
{
    const Index_t regElemSize = domain->regElemSize(r);
    for (Index_t b = 0; b < regElemSize ; b += EBS)
    {
        // compute minimum for this block of this region
        MPC_PUSH_DEPENDENCIES(dt_courant_deps[r] + 2*(b/EBS), 2);
        TASK_SET_COLOR(iter);
        TASK_SET_LABEL("CalcCourantConstraintForElems");
        #pragma omp task default(none)              \
            firstprivate(domain, b, regElemSize, r) \
            shared(dt_reduction_courant, EBS)       \
            depend(in: domain->m_dtcourant)
        {
            const Index_t * regElemlist = domain->regElemlist(r);
            const Real_t qqc            = domain->qqc();
            const Real_t qqc2           = Real_t(64.0) * qqc * qqc ;

            dt_reduction_t * dt = dt_reduction_courant[r] + (b/EBS);
            dt->value   = domain->dtcourant();
            dt->index   = -1;

            Index_t start = b;
            Index_t end = MIN(start + EBS, regElemSize);
            for (Index_t i = start ; i < end ; ++i)
            {
                Index_t indx = regElemlist[i] ;

                Real_t dtf = domain->ss(indx) * domain->ss(indx) ;

                if ( domain->vdov(indx) < Real_t(0.) ) {
                    dtf = dtf
                        + qqc2 * domain->arealg(indx) * domain->arealg(indx)
                        * domain->vdov(indx) * domain->vdov(indx) ;
                }

                dtf = SQRT(dtf) ;
                dtf = domain->arealg(indx) / dtf ;

                if (domain->vdov(indx) != Real_t(0.)) {
                    if ( dtf < dt->value ) {
                        dt->value = dtf;
                        dt->index = indx;
                    }
                }
            }
        }
    } /* for each block */

# if PRUNE_INOUTSET_EXPLICITLY
    TASK_SET_COLOR(iter);
    TASK_SET_LABEL("EMPTY(dt_reduction_courant[r])");
    # pragma omp task depend(inout: dt_reduction_courant[r])
    {}
# endif /* PRUNE_INOUTSET_EXPLICITLY */

    // reduce minimum of each block of this region
    // Should be replaced by a 'depend(inoutset: domain->m_dtcourant)'
    // once supported by gcc and mpc-openmp
    void * addrs[1];
    mpc_omp_task_dependency_t inoutset;
    inoutset.type       = MPC_OMP_TASK_DEP_INOUTSET;
    inoutset.addrs_size = 1;
    inoutset.addrs      = addrs;
    inoutset.addrs[0]   = (void *) (&dt_reduction_courant);

    MPC_PUSH_DEPENDENCIES(&inoutset, 1);
    TASK_SET_COLOR(iter);
    TASK_SET_LABEL("CalcCourantConstraintForElems_reduce");
    # pragma omp task default(none)             \
        firstprivate(domain, r, regElemSize)    \
        shared(dt_reduction_courant, EBS)       \
        depend(in: dt_reduction_courant[r])
    {
        dt_reduction_t * dt0 = dt_reduction_courant[r] + 0;
        for (Index_t b = EBS; b < regElemSize ; b += EBS)
        {
            dt_reduction_t * dt = dt_reduction_courant[r] + (b/EBS);

            if (dt->index != -1 && dt->value < dt0->value)
            {
                dt0->value = dt->value;
                dt0->index = dt->index;
            }
        }
    }
}

/******************************************/

static inline
void CalcHydroConstraintForElems(Domain * domain, Index_t r)
{
    const Index_t regElemSize = domain->regElemSize(r);
    for (Index_t b = 0; b < regElemSize ; b += EBS)
    {
        MPC_PUSH_DEPENDENCIES(dt_hydro_deps[r] + 2*(b/EBS), 2);
        TASK_SET_COLOR(iter);
        TASK_SET_LABEL("CalcHydroConstraintForElems");
        #pragma omp task default(none)              \
            firstprivate(domain, b, regElemSize, r) \
            shared(dt_reduction_hydro, EBS)         \
            depend(in: domain->m_dthydro)
        {
            const Index_t * regElemlist = domain->regElemlist(r);
            const Real_t dvovmax        = domain->dvovmax();

            dt_reduction_t * dt = dt_reduction_hydro[r] + (b/EBS);
            dt->value   = domain->dthydro();
            dt->index   = -1;

            Index_t start = b;
            Index_t end = MIN(start + EBS, regElemSize);
            for (Index_t i = start ; i < end ; ++i)
            {
                Index_t indx = regElemlist[i] ;

                if (domain->vdov(indx) != Real_t(0.)) {
                    Real_t dtdvov = dvovmax / (FABS(domain->vdov(indx))+Real_t(1.e-20)) ;

                    if ( dt->value > dtdvov ) {
                        dt->value = dtdvov;
                        dt->index = indx;
                    }
                }
            }
        }
    } /* for each block */

# if PRUNE_INOUTSET_EXPLICITLY
    TASK_SET_COLOR(iter);
    TASK_SET_LABEL("EMPTY(dt_reduction_hydro[r])");
    # pragma omp task depend(inout: dt_reduction_hydro[r])
    {}
# endif /* PRUNE_INOUTSET_EXPLICITLY */

    void * addrs[1];
    mpc_omp_task_dependency_t inoutset;
    inoutset.type       = MPC_OMP_TASK_DEP_INOUTSET;
    inoutset.addrs_size = 1;
    inoutset.addrs      = addrs;
    inoutset.addrs[0]   = (void *) (&dt_reduction_hydro);

    MPC_PUSH_DEPENDENCIES(&inoutset, 1);
    TASK_SET_COLOR(iter);
    TASK_SET_LABEL("CalcHydroConstraintForElems_reduce");
    # pragma omp task default(none)             \
        firstprivate(r, regElemSize, domain)    \
        shared(dt_reduction_hydro, EBS)         \
        depend(in: dt_reduction_hydro[r])
    {
        dt_reduction_t * dt0 = dt_reduction_hydro[r] + 0;
        for (Index_t b = EBS; b < regElemSize ; b += EBS)
        {
            dt_reduction_t * dt = dt_reduction_hydro[r] + (b/EBS);

            if (dt->index != -1 && dt->value < dt0->value)
            {
                dt0->value = dt->value;
                dt0->index = dt->index;
            }
        }
    }
}

static
void CalcTimeConstraintsForElems(Domain * domain)
{
    // Initialize conditions to a very large value
    TASK_SET_COLOR(iter);
    TASK_SET_LABEL("CalcTimeConstraintsForElems_init");
    # pragma omp task default(none)                             \
        firstprivate(domain)                                    \
        depend(inout: domain->m_dtcourant, domain->m_dthydro)
    {
        domain->dtcourant() = 1.0e+20;
        domain->dthydro()   = 1.0e+20;
    }

    const Index_t numReg = domain->numReg();
    for (Index_t r = 0 ; r < numReg ; ++r)
    {
        /* evaluate time constraint */
        CalcCourantConstraintForElems(domain, r);
        /* check hydro constraint */
        CalcHydroConstraintForElems(domain, r);
    }

     // reduce minimum dt courant and hydro of each region
     TASK_SET_COLOR(iter);
     TASK_SET_LABEL("CalcTimeConstraintsForElems_reduce_courant");
     # pragma omp task default(none)             \
         firstprivate(domain, numReg)            \
         shared(dt_reduction_courant)            \
         depend(inout: dt_reduction_courant)     \
         depend(out: domain->m_dtcourant)
     {
         Real_t & dtcourant = domain->dtcourant();
         for (Index_t r = 0 ; r < numReg ; ++r)
         {
             if (domain->regElemSize(r) == 0) continue ;
             dt_reduction_t * dt = dt_reduction_courant[r];
             if (dt->index != -1 && dt->value < dtcourant)
             {
                 dtcourant = dt->value;
             }
         }
     }

     TASK_SET_COLOR(iter);
     TASK_SET_LABEL("CalcTimeConstraintsForElems_reduce_hydro");
     # pragma omp task default(none)         \
         firstprivate(domain, numReg)        \
         shared(dt_reduction_hydro)          \
         depend(inout: dt_reduction_hydro)   \
         depend(out: domain->m_dthydro)
     {
         Real_t & dthydro = domain->dthydro();
         for (Index_t r = 0 ; r < numReg ; ++r)
         {
             if (domain->regElemSize(r) == 0) continue ;
             dt_reduction_t * dt = dt_reduction_hydro[r];
             if (dt->index != -1 && dt->value < dthydro)
             {
                 dthydro = dt->value;
             }
         }
     }
}

/******************************************/

static
void LagrangeLeapFrog(Domain * domain)
{
    /* calculate nodal forces, accelerations, velocities, positions, with
     * applied boundary conditions and slide surface considerations */
    LagrangeNodal(domain);

    /* calculate element quantities (i.e. velocity gradient & q), and update
     * material states */
    LagrangeElements(domain);
    CalcTimeConstraintsForElems(domain);
}

static void deallocate(Domain * domain)
{
    Index_t numElem = domain->numElem();

    if (numElem > 0) {

        Release(&sigxx);
        Release(&sigyy);
        Release(&sigzz);
        Release(&determ);
        Release(&vnew);

        Release(&dvdx);
        Release(&dvdy);
        Release(&dvdz);
        Release(&x8n);
        Release(&y8n);
        Release(&z8n);
    }

    Release(&fz_elem) ;
    Release(&fy_elem) ;
    Release(&fx_elem) ;

    Release(&fz_elem_FBH) ;
    Release(&fy_elem_FBH) ;
    Release(&fx_elem_FBH) ;

    Index_t numReg = domain->numReg();
    if (numReg > 0) {
        for (Int_t i = 0; i < numReg; i++) {
            free(pHalfStepRegs[i]);
            free(e_oldRegs[i]);
            free(delvcRegs[i]);
            free(p_oldRegs[i]);
            free(q_oldRegs[i]);
            free(compressionRegs[i]);
            free(compHalfStepRegs[i]);
            free(qq_oldRegs[i]);
            free(ql_oldRegs[i]);
            free(workRegs[i]);
            free(p_newRegs[i]);
            free(e_newRegs[i]);
            free(q_newRegs[i]);
            free(bvcRegs[i]);
            free(pbvcRegs[i]);
            free(dt_reduction_courant[i]);
            free(dt_reduction_hydro[i]);
        }
        free(pHalfStepRegs);
        free(e_oldRegs);
        free(delvcRegs);
        free(p_oldRegs);
        free(q_oldRegs);
        free(compressionRegs);
        free(compHalfStepRegs);
        free(qq_oldRegs);
        free(ql_oldRegs);
        free(workRegs);
        free(p_newRegs);
        free(e_newRegs);
        free(q_newRegs);
        free(bvcRegs);
        free(pbvcRegs);
        free(dt_reduction_courant);
        free(dt_reduction_hydro);
    }
}

static void
allocate(Domain * domain)
{
    // Allocate domain strains and gradients
    Int_t numElem = domain->numElem();
    if (numElem > 0)
    {
        // Allocate pointers used in CalcVolumeForceForElems
        sigxx  = Allocate<Real_t>(numElem) ;
        sigyy  = Allocate<Real_t>(numElem) ;
        sigzz  = Allocate<Real_t>(numElem) ;
        determ = Allocate<Real_t>(numElem) ;

        dvdx = Allocate<Real_t>(8 * numElem);
        dvdy = Allocate<Real_t>(8 * numElem);
        dvdz = Allocate<Real_t>(8 * numElem);
        x8n  = Allocate<Real_t>(8 * numElem);
        y8n  = Allocate<Real_t>(8 * numElem);
        z8n  = Allocate<Real_t>(8 * numElem);

        // Allocate pointer used in LagrangeElements
        vnew = Allocate<Real_t>(numElem) ;  /* new relative vol -- temp */
    }

    Int_t numReg = domain->numReg();
    if (numReg > 0)
    {
        // CalcEnergyForElems
        pHalfStepRegs = Allocate<Real_t *>(numReg);

        // EvalEOSForElems
        e_oldRegs           = Allocate<Real_t *>(numReg);
        delvcRegs           = Allocate<Real_t *>(numReg);
        p_oldRegs           = Allocate<Real_t *>(numReg);
        q_oldRegs           = Allocate<Real_t *>(numReg);
        compressionRegs     = Allocate<Real_t *>(numReg);
        compHalfStepRegs    = Allocate<Real_t *>(numReg);
        qq_oldRegs          = Allocate<Real_t *>(numReg);
        ql_oldRegs          = Allocate<Real_t *>(numReg);
        workRegs            = Allocate<Real_t *>(numReg);
        p_newRegs           = Allocate<Real_t *>(numReg);
        e_newRegs           = Allocate<Real_t *>(numReg);
        q_newRegs           = Allocate<Real_t *>(numReg);
        bvcRegs             = Allocate<Real_t *>(numReg);
        pbvcRegs            = Allocate<Real_t *>(numReg);

        dt_reduction_courant    = Allocate<dt_reduction_t *>(numReg);
        dt_reduction_hydro      = Allocate<dt_reduction_t *>(numReg);

        for (Int_t i = 0; i < numReg; i++)
        {
            Int_t length = domain->regElemSize(i);

            pHalfStepRegs[i]    = Allocate<Real_t>(length);
            e_oldRegs[i]        = Allocate<Real_t>(length);
            delvcRegs[i]        = Allocate<Real_t>(length);
            p_oldRegs[i]        = Allocate<Real_t>(length);
            q_oldRegs[i]        = Allocate<Real_t>(length);
            compressionRegs[i]  = Allocate<Real_t>(length);
            compHalfStepRegs[i] = Allocate<Real_t>(length);
            qq_oldRegs[i]       = Allocate<Real_t>(length);
            ql_oldRegs[i]       = Allocate<Real_t>(length);
            workRegs[i]         = Allocate<Real_t>(length);
            p_newRegs[i]        = Allocate<Real_t>(length);
            e_newRegs[i]        = Allocate<Real_t>(length);
            q_newRegs[i]        = Allocate<Real_t>(length);
            bvcRegs[i]          = Allocate<Real_t>(length);
            pbvcRegs[i]         = Allocate<Real_t>(length);

            unsigned int nblocks = length / EBS + (length % EBS != 0);
            dt_reduction_courant[i] = Allocate<dt_reduction_t>(nblocks);
            dt_reduction_hydro[i]   = Allocate<dt_reduction_t>(nblocks);

            //regRep[i]           = Allocate<Real_t>(NUM_TASKS_BS);
        }
    }

    Int_t numElem8 = numElem*8;
    // Allocate pointers used in IntegrateStressForElems and CalcFBHourglassForceForElems
    fx_elem = Allocate<Real_t>(numElem8) ;
    fy_elem = Allocate<Real_t>(numElem8) ;
    fz_elem = Allocate<Real_t>(numElem8) ;

    fx_elem_FBH = Allocate<Real_t>(numElem8) ;
    fy_elem_FBH = Allocate<Real_t>(numElem8) ;
    fz_elem_FBH = Allocate<Real_t>(numElem8) ;

    // Init array used in CalcFBHourglassForceForElems
    gamma_v[0][0] = Real_t( 1.);
    gamma_v[0][1] = Real_t( 1.);
    gamma_v[0][2] = Real_t(-1.);
    gamma_v[0][3] = Real_t(-1.);
    gamma_v[0][4] = Real_t(-1.);
    gamma_v[0][5] = Real_t(-1.);
    gamma_v[0][6] = Real_t( 1.);
    gamma_v[0][7] = Real_t( 1.);
    gamma_v[1][0] = Real_t( 1.);
    gamma_v[1][1] = Real_t(-1.);
    gamma_v[1][2] = Real_t(-1.);
    gamma_v[1][3] = Real_t( 1.);
    gamma_v[1][4] = Real_t(-1.);
    gamma_v[1][5] = Real_t( 1.);
    gamma_v[1][6] = Real_t( 1.);
    gamma_v[1][7] = Real_t(-1.);
    gamma_v[2][0] = Real_t( 1.);
    gamma_v[2][1] = Real_t(-1.);
    gamma_v[2][2] = Real_t( 1.);
    gamma_v[2][3] = Real_t(-1.);
    gamma_v[2][4] = Real_t( 1.);
    gamma_v[2][5] = Real_t(-1.);
    gamma_v[2][6] = Real_t( 1.);
    gamma_v[2][7] = Real_t(-1.);
    gamma_v[3][0] = Real_t(-1.);
    gamma_v[3][1] = Real_t( 1.);
    gamma_v[3][2] = Real_t(-1.);
    gamma_v[3][3] = Real_t( 1.);
    gamma_v[3][4] = Real_t( 1.);
    gamma_v[3][5] = Real_t(-1.);
    gamma_v[3][6] = Real_t( 1.);
    gamma_v[3][7] = Real_t(-1.);
}

static void deinit_deps(Domain * domain)
{
    // free element dependencies
    const Index_t numElem = domain->numElem();
    for (Index_t b = 0 ; b < numElem ; b += EBS)
    {
        free(dependencies_domain_x_y_z[b/EBS].addrs);
        free(dependencies_domain_xd_yd_zd[b/EBS].addrs);
    }
    free(dependencies_domain_x_y_z);
    free(dependencies_domain_xd_yd_zd);

    // free nodes dependencies
    const Index_t numNode = domain->numNode();
    for (Index_t b = 0; b < numNode ; b += NBS)
    {
        free(dependencies_fx_fy_fz_elem[b/NBS].addrs);
        free(dependencies_fx_fy_fz_elem_FBH[b/NBS].addrs);
    }
    free(dependencies_fx_fy_fz_elem);
    free(dependencies_fx_fy_fz_elem_FBH);

    // free boundary nodes dependencies
    const Index_t size = domain->sizeX();
    const Index_t numNodeBC = (size+1)*(size+1);
    for (Index_t b = 0; b < numNodeBC ; b += NBS)
    {
        free(dependencies_bc_xdd[b/NBS].addrs);
        free(dependencies_bc_ydd[b/NBS].addrs);
        free(dependencies_bc_zdd[b/NBS].addrs);
    }
    free(dependencies_bc_xdd);
    free(dependencies_bc_ydd);
    free(dependencies_bc_zdd);

    // free reg. dependencies
    const Index_t numReg = domain->numReg();
    for (Index_t r = 0 ; r < numReg ; ++r)
    {
        const Index_t regElemSize = domain->regElemSize(r);
        for (Index_t b = 0 ; b < regElemSize ; b += EBS)
        {
            free(dt_courant_deps[r][2*(b/EBS)+0].addrs);
            free(dt_courant_deps[r][2*(b/EBS)+1].addrs);
            free(dt_hydro_deps[r][2*(b/EBS)+0].addrs);
            free(dt_hydro_deps[r][2*(b/EBS)+1].addrs);
        }
        free(dt_courant_deps[r]);
        free(dt_hydro_deps[r]);
    }
    free(dt_courant_deps);
    free(dt_hydro_deps);

    for (Index_t r = 0 ; r < numReg ; ++r)
    {
        const Index_t regElemSize = domain->regElemSize(r);
        for (Index_t b = 0; b < regElemSize ; b += EBS)
        {
            free(EvalEOSForElems_deps_1[r][b/EBS].addrs);
            free(vnew_in_deps[r][b/EBS].addrs);
            free(EvalEOSForElems_deps_2[r][b/EBS].addrs);
            free(CalcSoundSpeedForElems_deps[r][b/EBS].addrs);
        }
        free(EvalEOSForElems_deps_1[r]);
        free(vnew_in_deps[r]);
        free(EvalEOSForElems_deps_2[r]);
        free(CalcSoundSpeedForElems_deps[r]);
    }
    free(EvalEOSForElems_deps_1);
    free(vnew_in_deps);
    free(EvalEOSForElems_deps_2);
    free(CalcSoundSpeedForElems_deps);
}

static void init_deps(Domain * domain)
{
    // data pointers
    const Real_t * x = domain->m_x.data();
    const Real_t * y = domain->m_y.data();
    const Real_t * z = domain->m_z.data();
    const Real_t * xd = domain->m_xd.data();
    const Real_t * yd = domain->m_yd.data();
    const Real_t * zd = domain->m_zd.data();

    const Real_t * xdd = domain->m_xdd.data();
    const Real_t * ydd = domain->m_ydd.data();
    const Real_t * zdd = domain->m_zdd.data();

    const Real_t * ss       = domain->m_ss.data();
    const Real_t * vdov     = domain->m_vdov.data();
    const Real_t * arealg   = domain->m_arealg.data();

    // elem -> node loop
    const Index_t numElem = domain->numElem();
    const Index_t n_elem_blocks = numElem/EBS + (numElem % EBS != 0);

    dependencies_domain_x_y_z       = (mpc_omp_task_dependency_t *) malloc(sizeof(mpc_omp_task_dependency_t) * n_elem_blocks);
    dependencies_domain_xd_yd_zd    = (mpc_omp_task_dependency_t *) malloc(sizeof(mpc_omp_task_dependency_t) * n_elem_blocks);

    for (Index_t b = 0; b < numElem ; b += EBS)
    {
        TASK_SET_COLOR(iter);
        TASK_SET_LABEL("init");
        # pragma omp task
        {
            std::map<Index_t, bool> blocks;

            int redundancy = 0;
            Index_t start = b;
            Index_t end = MIN(start + EBS, numElem);
            for (Index_t k = start ; k < end ; ++k)
            {
                const Index_t * const elemtonode = domain->nodelist(k);
                int i;
                for (i = 0 ; i < 8 ; ++i)
                {
                    Index_t index = elemtonode[i] / NBS * NBS;
                    if (blocks.count(index) == 0) blocks[index] = true;
                    else ++redundancy;
                }
            }
//            printf("uniq=%ld, redundant=%d\n", blocks.size(), redundancy);

            // allocate dependency array
            mpc_omp_task_dependency_t * dependency_domain_x_y_z = dependencies_domain_x_y_z + (b/EBS);
            dependency_domain_x_y_z->type             = MPC_OMP_TASK_DEP_IN;
            dependency_domain_x_y_z->addrs_size       = 3 * blocks.size();
            dependency_domain_x_y_z->addrs            = (void **)malloc(sizeof(void *) * dependency_domain_x_y_z->addrs_size);

            mpc_omp_task_dependency_t * dependency_domain_xd_yd_zd = dependencies_domain_xd_yd_zd + (b/EBS);
            dependency_domain_xd_yd_zd->type        = MPC_OMP_TASK_DEP_IN;
            dependency_domain_xd_yd_zd->addrs_size  = 3 * blocks.size();
            dependency_domain_xd_yd_zd->addrs       = (void **)malloc(sizeof(void *) * dependency_domain_xd_yd_zd->addrs_size);

            // copy unique blocks to the dependency array
            unsigned int j = 0;
            for (std::map<Index_t, bool>::iterator it = blocks.begin(); it != blocks.end(); ++it)
            {
                const Index_t index = it->first;
                dependency_domain_x_y_z->addrs[3 * j + 0]       = (void *)(x + index);
                dependency_domain_x_y_z->addrs[3 * j + 1]       = (void *)(y + index);
                dependency_domain_x_y_z->addrs[3 * j + 2]       = (void *)(z + index);
                dependency_domain_xd_yd_zd->addrs[3 * j + 0]    = (void *)(xd + index);
                dependency_domain_xd_yd_zd->addrs[3 * j + 1]    = (void *)(yd + index);
                dependency_domain_xd_yd_zd->addrs[3 * j + 2]    = (void *)(zd + index);
                j += 1;
            }
            assert(3 * j == dependency_domain_x_y_z->addrs_size);
            assert(3 * j == dependency_domain_xd_yd_zd->addrs_size);
        }
    }

    // node -> elem loop
    const Index_t numNode       = domain->numNode();
    const Index_t n_node_blocks = numNode/NBS + (numNode % NBS != 0);

    dependencies_fx_fy_fz_elem     = (mpc_omp_task_dependency_t *) malloc(sizeof(mpc_omp_task_dependency_t) * n_node_blocks);
    dependencies_fx_fy_fz_elem_FBH = (mpc_omp_task_dependency_t *) malloc(sizeof(mpc_omp_task_dependency_t) * n_node_blocks);

    for (Index_t b = 0; b < numNode ; b += NBS)
    {
        TASK_SET_COLOR(iter);
        TASK_SET_LABEL("init");
        # pragma omp task
        {
            std::map<Index_t, bool> blocks;
            int redundancy = 0;
            Index_t start = b;
            Index_t end = MIN(start + NBS, numNode);
            for (Index_t gnode = start ; gnode < end ; ++gnode)
            {
                Index_t count = domain->nodeElemCount(gnode) ;
                Index_t * cornerList = domain->nodeElemCornerList(gnode);
                for (Index_t i = 0 ; i < count ; ++i)
                {
                    Index_t index = cornerList[i] / (8 * EBS) * (8 * EBS);
                    if (blocks.count(index) == 0) blocks[index] = true;
                    else ++redundancy;
                }
            }
//            printf("uniq=%ld, redundant=%d\n", blocks.size(), redundancy);

            mpc_omp_task_dependency_t * dependency_fx_fy_fz_elem = dependencies_fx_fy_fz_elem + (b/NBS);
            dependency_fx_fy_fz_elem->type         = MPC_OMP_TASK_DEP_IN;
            dependency_fx_fy_fz_elem->addrs_size   = 1 * blocks.size();
            dependency_fx_fy_fz_elem->addrs        = (void **)malloc(sizeof(void *) * dependency_fx_fy_fz_elem->addrs_size);

            mpc_omp_task_dependency_t * dependency_fx_fy_fz_elem_FBH = dependencies_fx_fy_fz_elem_FBH + (b/NBS);
            dependency_fx_fy_fz_elem_FBH->type        = MPC_OMP_TASK_DEP_IN;
            dependency_fx_fy_fz_elem_FBH->addrs_size  = 1 * blocks.size();
            dependency_fx_fy_fz_elem_FBH->addrs       = (void **)malloc(sizeof(void *) * dependency_fx_fy_fz_elem_FBH->addrs_size);

            unsigned int j = 0;
            for (std::map<Index_t, bool>::iterator it = blocks.begin(); it != blocks.end(); ++it)
            {
                const Index_t index = it->first;

                dependency_fx_fy_fz_elem->addrs[j+0] = (void *)(fx_elem + index);
                dependency_fx_fy_fz_elem_FBH->addrs[j+0] = (void *)(fx_elem_FBH + index);

                j += 1;
            }
            assert(j == dependency_fx_fy_fz_elem->addrs_size);
            assert(j == dependency_fx_fy_fz_elem_FBH->addrs_size);
        }
    }

    // Boundary nodes dependencies
    const Index_t sizeX = domain->sizeX();
    const Index_t numNodeBC = (sizeX+1)*(sizeX+1) ;
    const Index_t n_blocks_node_bc = numNodeBC / NBS + (numNodeBC % NBS != 0);

    dependencies_bc_xdd  = (mpc_omp_task_dependency_t *) malloc(sizeof(mpc_omp_task_dependency_t) * n_blocks_node_bc);
    dependencies_bc_ydd  = (mpc_omp_task_dependency_t *) malloc(sizeof(mpc_omp_task_dependency_t) * n_blocks_node_bc);
    dependencies_bc_zdd  = (mpc_omp_task_dependency_t *) malloc(sizeof(mpc_omp_task_dependency_t) * n_blocks_node_bc);


    for (Index_t b = 0; b < numNodeBC ; b += NBS)
    {
        TASK_SET_COLOR(iter);
        TASK_SET_LABEL("init");
        # pragma omp task
        {
            std::map<Index_t, bool> blocks_x, blocks_y, blocks_z;
            Index_t start = b;
            Index_t end = MIN(start + NBS, numNodeBC);
            int redundancy = 0;
            for (Index_t i = start ; i < end ; ++i)
            {
                if (!domain->symmXempty())
                {
                    const Index_t index_x = domain->symmX(i) / NBS * NBS;
                    if (blocks_x.count(index_x) == 0) blocks_x[index_x] = true;
                    else ++redundancy;
                }

                if (!domain->symmYempty())
                {
                    const Index_t index_y = domain->symmY(i) / NBS * NBS;
                    if (blocks_y.count(index_y) == 0) blocks_y[index_y] = true;
                    else ++redundancy;
                }

                if (!domain->symmZempty())
                {
                    const Index_t index_z = domain->symmZ(i) / NBS * NBS;
                    if (blocks_z.count(index_z) == 0) blocks_z[index_z] = true;
                    else ++redundancy;
                }
            }
//            printf("uniq=%ld, redundant=%d\n", blocks_x.size() + blocks_y.size() + blocks_z.size(), redundancy);

            mpc_omp_task_dependency_t * dependency_bc_xdd = dependencies_bc_xdd + (b/NBS);
            dependency_bc_xdd->type         = MPC_OMP_TASK_DEP_INOUTSET;
            dependency_bc_xdd->addrs_size   = blocks_x.size();
            dependency_bc_xdd->addrs        = (void **) malloc(sizeof(void *) * blocks_x.size());

            mpc_omp_task_dependency_t * dependency_bc_ydd = dependencies_bc_ydd + (b/NBS);
            dependency_bc_ydd->type         = MPC_OMP_TASK_DEP_INOUTSET;
            dependency_bc_ydd->addrs_size   = blocks_y.size();
            dependency_bc_ydd->addrs        = (void **) malloc(sizeof(void *) * blocks_y.size());

            mpc_omp_task_dependency_t * dependency_bc_zdd = dependencies_bc_zdd + (b/NBS);
            dependency_bc_zdd->type         = MPC_OMP_TASK_DEP_INOUTSET;
            dependency_bc_zdd->addrs_size   = blocks_z.size();
            dependency_bc_zdd->addrs        = (void **) malloc(sizeof(void *) * blocks_z.size());

            unsigned int j;

            j = 0;
            for (std::map<Index_t, bool>::iterator it = blocks_x.begin(); it != blocks_x.end(); ++it)
            {
                dependency_bc_xdd->addrs[j++] = (void *) (xdd + it->first);
            }
            assert(j == dependency_bc_xdd->addrs_size);

            j = 0;
            for (std::map<Index_t, bool>::iterator it = blocks_y.begin(); it != blocks_y.end(); ++it)
            {
                dependency_bc_ydd->addrs[j++] = (void *) (ydd + it->first);
            }
            assert(j == dependency_bc_ydd->addrs_size);

            j = 0;
            for (std::map<Index_t, bool>::iterator it = blocks_z.begin(); it != blocks_z.end(); ++it)
            {
                dependency_bc_zdd->addrs[j++] = (void *) (zdd + it->first);
            }
            assert(j == dependency_bc_zdd->addrs_size);
        }
    }

    // dt time courant and hydro deps
    const Index_t numReg = domain->numReg();
    const size_t size = sizeof(mpc_omp_task_dependency_t *) * numReg;
    dt_courant_deps = (mpc_omp_task_dependency_t **) malloc(size);
    dt_hydro_deps = (mpc_omp_task_dependency_t **) malloc(size);
    for (Index_t r = 0 ; r < numReg ; ++r)
    {
        const Index_t regElemSize   = domain->regElemSize(r);
        const Index_t nblocks       = regElemSize / EBS + (regElemSize % EBS != 0);
        const size_t size = sizeof(mpc_omp_task_dependency_t) * 2*nblocks;
        dt_courant_deps[r] = (mpc_omp_task_dependency_t *) malloc(size);
        dt_hydro_deps[r] = (mpc_omp_task_dependency_t *) malloc(size);

        for (Index_t b = 0 ; b < regElemSize ; b += EBS)
        {
            TASK_SET_COLOR(iter);
            TASK_SET_LABEL("init");
            # pragma omp task
            {
                std::map<Index_t, bool> blocks;

                mpc_omp_task_dependency_t * inoutset_courant = dt_courant_deps[r] + 2*(b/EBS)+0;
                mpc_omp_task_dependency_t * in_courant       = dt_courant_deps[r] + 2*(b/EBS)+1;

                mpc_omp_task_dependency_t * inoutset_hydro = dt_hydro_deps[r] + 2*(b/EBS)+0;
                mpc_omp_task_dependency_t * in_hydro       = dt_hydro_deps[r] + 2*(b/EBS)+1;

                // reduction on 'dtcourant' and 'dthydro' per block
                inoutset_courant->type          = MPC_OMP_TASK_DEP_INOUTSET;
                inoutset_courant->addrs_size    = 1;
                inoutset_courant->addrs         = (void **) malloc(sizeof(void *) * 1);
                inoutset_courant->addrs[0]      = (void *) &(dt_reduction_courant[r]);

                inoutset_hydro->type            = MPC_OMP_TASK_DEP_INOUTSET;
                inoutset_hydro->addrs_size      = 1;
                inoutset_hydro->addrs           = (void **) malloc(sizeof(void *) * 1);
                inoutset_hydro->addrs[0]        = (void *) &(dt_reduction_hydro[r]);

                // in dependencies
                // IN : domain->ss(index)
                // IN : domain->vdov(index)
                // IN : domain->arealg(index)
                Index_t start = b;
                Index_t end = MIN(start + EBS, regElemSize);
                int redundancy = 0;
                for (Index_t i = start ; i < end ; ++i)
                {
                    const Index_t * regElemlist = domain->regElemlist(r);
                    const Index_t index = regElemlist[i] / EBS * EBS;
                    if (blocks.count(index) == 0)   blocks[index] = true;
                    else ++redundancy;
                }
//                printf("uniq=%ld, redundant=%d\n", blocks.size(), redundancy);

                in_courant->type        = MPC_OMP_TASK_DEP_IN;
                in_courant->addrs_size  = 3 * blocks.size();
                in_courant->addrs       = (void **) malloc(sizeof(void *) * in_courant->addrs_size);

                in_hydro->type          = MPC_OMP_TASK_DEP_IN;
                in_hydro->addrs_size    = 1 * blocks.size();
                in_hydro->addrs         = (void **) malloc(sizeof(void *) * in_hydro->addrs_size);

                unsigned int j = 0;
                for (std::map<Index_t, bool>::iterator it = blocks.begin(); it != blocks.end(); ++it)
                {
                    const Index_t index = it->first;
                    in_courant->addrs[3 * j + 0] = (void *) (ss       + index);
                    in_courant->addrs[3 * j + 1] = (void *) (vdov     + index);
                    in_courant->addrs[3 * j + 2] = (void *) (arealg   + index);

                    in_hydro->addrs[j] = (void *) (vdov + index);

                    ++j;
                }
                assert(3 * j == in_courant->addrs_size);
                assert(    j ==   in_hydro->addrs_size);
            }
        }
    }

    // CalcMonotonicQRegionForElems
    // IN : delv_xi(i),   delv_xi(lxim(i)),     delv_xi(lxip(i))
    // IN : delv_eta(i),  delv_eta(letam(i)),   delv_eta(letap(i))
    // IN : delv_zeta(i), delv_zeta(lzetam(i)), delv_zeta(lzetap(i))
    // IN : delx_xi(i), delx_eta(i), delx_zeta(i)
    // IN : vdov(i), volo(i), vnew[i]
    // OUT : qq[i], ql[i]
    CalcMonotonicQRegionForElems_deps = (mpc_omp_task_dependency_t **) malloc(sizeof(mpc_omp_task_dependency_t *) * numReg);

    const Real_t * domain_vdov      = domain->m_vdov.data();        (void) domain_vdov;
    const Real_t * domain_qq        = domain->m_qq.data();          (void) domain_qq;

    for (Index_t r = 0 ; r < numReg ; ++r)
    {
        Index_t regElemSize   = domain->regElemSize(r);
        const Index_t nblocks = regElemSize / EBS + (regElemSize % EBS != 0);
        CalcMonotonicQRegionForElems_deps[r] = (mpc_omp_task_dependency_t *) malloc(sizeof(mpc_omp_task_dependency_t) * 2*nblocks);
        for (Index_t b = 0; b < regElemSize ; b += EBS)
        {
            TASK_SET_COLOR(iter);
            TASK_SET_LABEL("init");
            # pragma omp task
            {
                std::map<Index_t, bool> blocks;
                std::map<Index_t, bool> blocks_delv_xi;
                std::map<Index_t, bool> blocks_delv_eta;
                std::map<Index_t, bool> blocks_delv_zeta;

                Index_t start = b;
                Index_t end = MIN(start + EBS, regElemSize);
                for (Index_t ielem = start ; ielem < end ; ++ielem)
                {
                    Index_t i = domain->regElemlist(r, ielem);
                    Index_t i_block = i / EBS * EBS;

                    // IN : delx_xi(i), delx_eta(i), delx_zeta(i)
                    // IN : domain->vdov(i), vnew[i]
                    // IN : domain->delv_xi(i), domain->delv_eta(i), domain->delv_zeta(i)
                    // IN : domain->delx_xi(i), domain->delx_eta(i), domain->delx_zeta(i)
                    // OUT : domain->qq(i), domain->ql(i)
                    if (blocks.count(i_block) == 0) blocks[i_block] = true;

                    // IN : maybe domain->delv_xi(domain->lxim(i))
                    // IN : maybe domain->delv_xi(domain->lxip(i))
                    // IN : maybe domain->delv_eta(domain->letam(i))
                    // IN : maybe domain->delv_eta(domain->letap(i))
                    // IN : maybe domain->delv_zeta(domain->lzetam(i))
                    // IN : maybe domain->delv_zeta(domain->lzetap(i))
                    Int_t bcMask = domain->elemBC(i) ;
                    switch (bcMask & XI_M)
                    {
                        case XI_M_COMM:
                        case 0:
                        {
                            Index_t index = domain->lxim(i) / EBS * EBS;
                            if (blocks_delv_xi.count(index) == 0) blocks_delv_xi[index] = true;
                            break ;
                        }
                        default: break;
                    }

                    switch (bcMask & XI_P)
                    {
                        case XI_P_COMM:
                        case 0:
                        {
                            Index_t index = domain->lxip(i) / EBS * EBS;
                            if (blocks_delv_xi.count(index) == 0) blocks_delv_xi[index] = true;
                            break ;
                        }
                        default: break;
                    }

                    switch (bcMask & ETA_M)
                    {
                        case ETA_M_COMM:
                        case 0:
                        {
                            Index_t index = domain->letam(i) / EBS * EBS;
                            if (blocks_delv_eta.count(index) == 0) blocks_delv_eta[index] = true;
                            break ;
                        }
                        default: break;
                    }

                    switch (bcMask & ETA_P)
                    {
                        case ETA_P_COMM:
                        case 0:
                        {
                            Index_t index = domain->letap(i) / EBS * EBS;
                            if (blocks_delv_eta.count(index) == 0) blocks_delv_eta[index] = true;
                            break ;
                        }
                        default: break;
                    }

                    switch (bcMask & ZETA_M)
                    {
                        case ZETA_M_COMM:
                        case 0:
                        {
                            Index_t index = domain->lzetam(i) / EBS * EBS;
                            if (blocks_delv_zeta.count(index) == 0) blocks_delv_zeta[index] = true;
                            break ;
                        }
                        default: break;
                    }

                    switch (bcMask & ZETA_P)
                    {
                        case ZETA_P_COMM:
                        case 0:
                        {
                            Index_t index = domain->lzetap(i) / EBS * EBS;
                            if (blocks_delv_zeta.count(index) == 0) blocks_delv_zeta[index] = true;
                            break ;
                        }
                        default: break;
                    }
                } /* for ielem */

                mpc_omp_task_dependency_t * inoutset = CalcMonotonicQRegionForElems_deps[r] + 2*(b/EBS)+0;
                inoutset->type       = MPC_OMP_TASK_DEP_INOUTSET;
                inoutset->addrs_size = 1 * blocks.size();
                inoutset->addrs      = (void **) malloc(sizeof(void *) * inoutset->addrs_size);

                mpc_omp_task_dependency_t * in  = CalcMonotonicQRegionForElems_deps[r] + 2*(b/EBS)+1;
                in->type        = MPC_OMP_TASK_DEP_IN;
                in->addrs_size  = 8 * blocks.size()
                                    + blocks_delv_xi.size()
                                    + blocks_delv_eta.size()
                                    + blocks_delv_zeta.size();
                in->addrs       = (void **) malloc(sizeof(void *) * in->addrs_size);

                unsigned int j1 = 0;
                unsigned int j2 = 0;
                for (std::map<Index_t, bool>::iterator it = blocks.begin(); it != blocks.end(); ++it)
                {
                    const Index_t index = it->first;

                    in->addrs[j1++] = (void *) (domain->m_delv_xi    + index);
                    in->addrs[j1++] = (void *) (domain->m_delv_eta   + index);
                    in->addrs[j1++] = (void *) (domain->m_delv_zeta  + index);

                    in->addrs[j1++] = (void *) (domain->m_delx_xi    + index);
                    in->addrs[j1++] = (void *) (domain->m_delx_eta   + index);
                    in->addrs[j1++] = (void *) (domain->m_delx_zeta  + index);

                    in->addrs[j1++] = (void *) (domain_vdov       + index);
                    in->addrs[j1++] = (void *) (vnew              + index);

                    inoutset->addrs[j2++] = (void *) (domain_qq + index);
                }

                for (std::map<Index_t, bool>::iterator it = blocks_delv_xi.begin(); it != blocks_delv_xi.end(); ++it)
                {
                    in->addrs[j1++] = (void *) (domain->m_delv_xi + it->first);
                }

                for (std::map<Index_t, bool>::iterator it = blocks_delv_eta.begin(); it != blocks_delv_eta.end(); ++it)
                {
                    in->addrs[j1++] = (void *) (domain->m_delv_eta + it->first);
                }

                for (std::map<Index_t, bool>::iterator it = blocks_delv_zeta.begin(); it != blocks_delv_zeta.end(); ++it)
                {
                    in->addrs[j1++] = (void *) (domain->m_delv_zeta + it->first);
                }
                assert(j1   ==       in->addrs_size);
                assert(j2   == inoutset->addrs_size);
            }
        } /* for b */
    }

    // EvalEOSForElems(domain, vnew, r, rep);
    EvalEOSForElems_deps_1 = (mpc_omp_task_dependency_t **) malloc(sizeof(mpc_omp_task_dependency_t *) * numReg);
    vnew_in_deps = (mpc_omp_task_dependency_t **) malloc(sizeof(mpc_omp_task_dependency_t *) * numReg);
    EvalEOSForElems_deps_2 = (mpc_omp_task_dependency_t **) malloc(sizeof(mpc_omp_task_dependency_t *) * numReg);
    CalcSoundSpeedForElems_deps = (mpc_omp_task_dependency_t **) malloc(sizeof(mpc_omp_task_dependency_t *) * numReg);
    for (Int_t r = 0 ; r < numReg ; ++r)
    {
        Real_t * domain_e       = domain->m_e.data();
        Real_t * domain_qq      = domain->m_qq.data();
        Real_t * domain_delv    = domain->m_delv.data();
        Real_t * domain_ss      = domain->m_ss.data();

        const Index_t regElemSize   = domain->regElemSize(r);
        const Index_t * regElemList = domain->regElemlist(r);
        const Index_t nblocks = regElemSize / EBS + (regElemSize % EBS != 0);

        EvalEOSForElems_deps_1[r] = (mpc_omp_task_dependency_t *) malloc(sizeof(mpc_omp_task_dependency_t) * nblocks);
        vnew_in_deps[r] = (mpc_omp_task_dependency_t *) malloc(sizeof(mpc_omp_task_dependency_t) * nblocks);
        EvalEOSForElems_deps_2[r] = (mpc_omp_task_dependency_t *) malloc(sizeof(mpc_omp_task_dependency_t) * nblocks);
        CalcSoundSpeedForElems_deps[r] = (mpc_omp_task_dependency_t *) malloc(sizeof(mpc_omp_task_dependency_t) * nblocks);
        for (Index_t b = 0; b < regElemSize ; b += EBS)
        {
            TASK_SET_COLOR(iter);
            TASK_SET_LABEL("init");
            # pragma omp task
            {
                std::map<Index_t, bool> blocks;

                Index_t start = b;
                Index_t end = MIN(start + EBS, regElemSize);

                int redundancy = 0;
                for (Index_t i = start ; i < end ; ++i)
                {
                    const Index_t elem = regElemList[i];
                    const Index_t index = elem / EBS * EBS;
                    if (blocks.count(index) == 0) blocks[index] = true;
                    else ++redundancy;
                }
//                printf("uniq=%ld, redundant=%d\n", blocks.size(), redundancy);

                mpc_omp_task_dependency_t * in_e_qq_delv = EvalEOSForElems_deps_1[r] + b/EBS;
                in_e_qq_delv->type       = MPC_OMP_TASK_DEP_IN;
                in_e_qq_delv->addrs_size = 3 * blocks.size();
                in_e_qq_delv->addrs      = (void **) malloc(sizeof(void *) * in_e_qq_delv->addrs_size);

                mpc_omp_task_dependency_t * vnew_in_dep = vnew_in_deps[r] + b/EBS;
                vnew_in_dep->type       = MPC_OMP_TASK_DEP_IN;
                vnew_in_dep->addrs_size = blocks.size();
                vnew_in_dep->addrs      = (void **) malloc(sizeof(void *) * vnew_in_dep->addrs_size);

                mpc_omp_task_dependency_t * inoutset_e = EvalEOSForElems_deps_2[r] + b/EBS;
                inoutset_e->type         = MPC_OMP_TASK_DEP_INOUTSET;
                inoutset_e->addrs_size   = blocks.size();
                inoutset_e->addrs        = (void **) malloc(sizeof(void *) * inoutset_e->addrs_size);

                mpc_omp_task_dependency_t * inoutset_ss = CalcSoundSpeedForElems_deps[r] + b/EBS;
                inoutset_ss->type         = MPC_OMP_TASK_DEP_INOUTSET;
                inoutset_ss->addrs_size   = blocks.size();
                inoutset_ss->addrs        = (void **) malloc(sizeof(void *) * inoutset_ss->addrs_size);

                unsigned int j = 0;
                for (std::map<Index_t, bool>::iterator it = blocks.begin(); it != blocks.end(); ++it)
                {
                    const Index_t index = it->first;

                    in_e_qq_delv->addrs[3 * j + 0] = (void *) (domain_e     + index);
                    in_e_qq_delv->addrs[3 * j + 1] = (void *) (domain_qq    + index);
                    in_e_qq_delv->addrs[3 * j + 2] = (void *) (domain_delv  + index);

                    vnew_in_dep->addrs[j] = (void *) (vnew + index);

                    inoutset_e->addrs[j] = (void *) (domain_e + index);

                    inoutset_ss->addrs[j] = (void *) (domain_ss + index);

                    ++j;
                }
                assert(3 * j == in_e_qq_delv->addrs_size);
                assert(    j == vnew_in_dep->addrs_size);
            }

        } /* b */
    }

    if (myRank == 0) printf("numElem=%d, elemBlockSize=%d, tel=%d\n", numElem, EBS, opts.tel);
    if (myRank == 0) printf("numNode=%d, nodeBlockSize=%d, tnl=%d\n", numNode, NBS, opts.tnl);
    if (myRank == 0) printf("nodesPerFaceRequest=%d, requestsPerFace=%d\n", domain->m_npfr, domain->m_rpf);
    if (myRank == 0) printf("nodesPerEdgeRequest=%d, requestsPerEdge=%d\n", domain->m_nper, domain->m_rpe);
}

/******************************************/
int main(int argc, char *argv[])
{
#if USE_MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided) ;
    assert(provided == MPI_THREAD_MULTIPLE);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks) ;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank) ;
#else
    numRanks = 1;
    myRank = 0;
#endif

    /* Set defaults that can be overridden by command line opts */
    opts.its = 9999999;
    opts.nx  = 30;
    opts.numReg = 11;
    opts.numFiles = (int)(numRanks+10)/9;
    opts.showProg = 0;
    opts.quiet = 0;
    opts.viz = 0;
    opts.balance = 1;
    opts.cost = 1;
    opts.tel = omp_get_max_threads();
    opts.tnl = omp_get_max_threads();
    opts.rpf = 1;
    opts.rpe = 1;

    ParseCommandLineOptions(argc, argv, myRank, &opts);

    if ((myRank == 0) && (opts.quiet == 0)) {
        printf("Running problem size %d^3 per domain until completion\n", opts.nx);
        printf("Num processors: %d\n", numRanks);
        printf("Num threads: %d\n", omp_get_max_threads());
        printf("Total number of elements: %lld\n\n", (long long int)(numRanks*opts.nx*opts.nx*opts.nx));
        printf("To run other sizes, use -s <integer>.\n");
        printf("To run a fixed number of iterations, use -i <integer>.\n");
        printf("To run a more or less balanced region set, use -b <integer>.\n");
        printf("To change the relative costs of regions, use -c <integer>.\n");
        printf("To print out progress, use -p\n");
        printf("To write an output file for VisIt, use -v\n");
        printf("See help (-h) for more options\n\n");
        fflush(stdout);
    }

    Domain * domain;
    double start = 0.0, tgraph = 0.0, tgraph0 = 0.0;

#if MEASURE_HASH_OCCUPATION
    double max_occupation = 0;
#endif /* MEASURE_HASH_OCCUPATION */

    # pragma omp parallel
    {
        # pragma omp single
        {
# if USE_MPI && TRACE
            MPC_OMP_TASK_TRACE_RANK(0, myRank);
            if (TRACE_CONDITION)
            {
                char name[128];
                gethostname(name, sizeof(name) - 1);
                printf("Tracing rank %d on %s\n", myRank, name);
            }
# endif /* USE_MPI && TRACE */

            if (myRank == 0) puts("Initializing mesh...");

            // Set up the mesh and decompose. Assumes regular cubes for now
            TASK_SET_COLOR(iter);
            TASK_SET_LABEL("InitMeshDecomp");
            Int_t col, row, plane, side;
            # pragma omp task if(0) default(shared)
            {
                InitMeshDecomp(numRanks, myRank, &col, &row, &plane, &side);
            }

            if (myRank == 0) puts("Instancing domain...");

            // Build the main data structure and initialize it
            TASK_SET_COLOR(iter);
            TASK_SET_LABEL("Domain instanciation");
            # pragma omp task if(0) default(shared)
            {
                domain = new Domain(numRanks, col, row, plane, opts.nx,
                    side, opts.numReg, opts.balance, opts.cost,
                    opts.tel, opts.tnl, opts.rpf, opts.rpe);
                EBS = domain->m_ebs;
                NBS = domain->m_nbs;
            }

            // compute block sizes
            if (myRank == 0) puts("Initializing...");

            TASK_SET_COLOR(iter);
            TASK_SET_LABEL("allocate");
            # pragma omp task default(none) firstprivate(domain) if(0)
            {
                // preallocate memory used in various routines
                allocate(domain);
            }

            TASK_SET_COLOR(iter);
            TASK_SET_LABEL("init_deps");
            # pragma omp task default(none) firstprivate(domain) if(0)
            {
                init_deps(domain);
            }

#if USE_MPI
            // Initial domain boundary communicatio
            CommRecv   (domain,   MSG_MASS);
            CommPack   (domain,   MSG_MASS);
            CommSend   (domain,   MSG_MASS);
            CommUnpack (domain,   MSG_MASS);
#endif /* USE_MPI */

            if (myRank == 0) puts("Initialized");

        } /* single */

#ifdef TRACE
        if (TRACE_CONDITION) mpc_omp_task_trace_begin();
#endif

        # pragma omp single
        {
#ifdef DRYRUN
            mpc_omp_task_dry_run(1);
#endif /* DRYRUN */

            // BEGIN timestep to solution */
            start = lulesh_timer();

            # pragma omp taskgroup
            {
                //debug to see region sizes
                //for(Int_t i = 0; i < domain->numReg(); i++)
                //    std::cout << "region" << i + 1<< "size" << domain->regElemSize(i) <<std::endl;
                if (myRank == 0) printf("[%d] Initial Origin Energy : %12.6e \n", myRank, domain->e(0));
                if (opts.persistentTasks) mpc_omp_persistent_region_begin();
                for (iter = 0 ; iter < opts.its && !cancelled ; ++iter)
                {
                    double t0 = lulesh_timer();
                    TimeIncrement(domain);
                    LagrangeLeapFrog(domain);
                    TimeDump(domain);
# if MEASURE_HASH_OCCUPATION
                    double occupation = mpc_omp_task_dependencies_buckets_occupation();
                    max_occupation = max_occupation < occupation ? occupation : max_occupation;
# endif /* MEASURE_HASH_OCCUPATION */
                    if (opts.persistentTasks)
                    {
                        double dt = lulesh_timer() - t0;
                        if (iter == 0)  tgraph0 = dt;
                        if (iter >= 0)  tgraph += dt;
                        mpc_omp_persistent_region_iterate();
                    }
                }
                if (opts.persistentTasks)
                {
                    mpc_omp_persistent_region_end();
                    if (myRank == 0) printf("Graph 1st iteration generated in %lf s.\n", tgraph0);
                }
                else
                {
                    tgraph = lulesh_timer() - start;
                }
                if (myRank == 0) printf("Graph generated in %lf s.\n", tgraph);
            } /* taskgroup (implicit barrier) */

        } /* single */

#ifdef TRACE
        if (TRACE_CONDITION && omp_get_thread_num() == 0) { printf("[%d] Flushing...\n", myRank); fflush(stdout); }
        if (TRACE_CONDITION) mpc_omp_task_trace_end();
        if (TRACE_CONDITION && omp_get_thread_num() == 0) { printf("[%d] Flushed.\n", myRank); fflush(stdout); }
#endif

        # pragma omp single
        {
            double elapsed_time = lulesh_timer() - start;
            double elapsed_timeG;
            double tgraphG;

#ifdef DRYRUN
            mpc_omp_task_dry_run(0);
#endif /* DRYRUN */

            TASK_SET_COLOR(iter);
            TASK_SET_LABEL("Compute Elapsed Time");
            # pragma omp task default(shared) if(0)
            {
                // Use reduced max elapsed time
#if USE_MPI
                MPI_Reduce(&elapsed_time, &elapsed_timeG, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
                MPI_Reduce(&tgraph,       &tgraphG,       1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
#else
                elapsed_timeG = elapsed_time;
                tgraphG = tgraph;
#endif
            }

            // Write out final viz file */
            if (opts.viz)
            {
                TASK_SET_COLOR(iter);
                TASK_SET_LABEL("Dump to viz");
                # pragma omp task default(shared) if(0)
                {
                    DumpToVisit(domain, opts.numFiles, myRank, numRanks) ;
                }
            }

            if ((myRank == 0) && (opts.quiet == 0))
            {
                TASK_SET_COLOR(iter);
                TASK_SET_LABEL("Verify result");
                # pragma omp task default(shared) if(0)
                {
                    int numThreads = omp_get_num_threads();
                    VerifyAndWriteFinalOutput(elapsed_timeG, domain, opts.nx, numRanks, numThreads);
                    printf("[%d] Global Graph generation time: %lf\n", myRank, tgraphG);
# if MEASURE_HASH_OCCUPATION
                    printf("[%d] Hash map max occupation = %lf\n", myRank, max_occupation);
                    printf("[%d] Hash map time = %lf\n", myRank, mpc_omp_task_dependencies_hash_time());
# endif /* MEASURE_HASH_OCCUPATION */
                }
            }

            TASK_SET_COLOR(iter);
            TASK_SET_LABEL("Deallocate");
            # pragma omp task default(shared) if(0)
            {
                deallocate(domain);
                deinit_deps(domain);
                delete domain;
            }
        } /* single */
    } /* parallel */

#if USE_MPI
    /* Barrier to avoid 'Finalize' timeout */
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
#endif
    return 0;
}
