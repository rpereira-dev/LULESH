#include "lulesh.h"

// If no MPI, then this whole file is stubbed out
#if USE_MPI

#include <mpi.h>
#include <string.h>
#include <iostream>

# ifdef TRACE
#  include <mpc_omp_task_trace.h>
# endif /* TRACE */

# if USE_MPIX
#  include <mpc_omp_interop_mpi.h>
# endif /* USE_MPIX */

# include "lulesh-trace.h"

extern int iter;

/***************/
/** PUBLIC API */
/***************/
void
CommGetMsgSpecifics(
    Domain * domain, int msgType,
    Domain_member fields[MAX_FIELDS_PER_MPI_COMM], int & nfields,
    bool & planeOnly,
    Index_t & dx, Index_t & dy, Index_t & dz)
{
    switch (msgType)
    {
        case (MSG_MASS):
        {
            fields[0]   = &Domain::nodalMass;
            nfields     = 1;
            planeOnly   = false;
            dx          = domain->sizeX() + 1;
            dy          = domain->sizeY() + 1;
            dz          = domain->sizeZ() + 1;
            break ;
        }

        case (MSG_FX_FY_FZ):
        {
            fields[0]   = &Domain::fx;
            fields[1]   = &Domain::fy;
            fields[2]   = &Domain::fz;
            nfields     = 3;
            planeOnly   = false;
            dx          = domain->sizeX() + 1;
            dy          = domain->sizeY() + 1;
            dz          = domain->sizeZ() + 1;
            break ;
        }

        case (MSG_DELV):
        {
            fields[0]   = &Domain::delv_xi;
            fields[1]   = &Domain::delv_eta;
            fields[2]   = &Domain::delv_zeta;
            nfields     = 3;
            planeOnly   = true;
            dx          = domain->sizeX();
            dy          = domain->sizeY();
            dz          = domain->sizeZ();
            break ;
        }

        default:
        {
            assert("Invalid msgType" && 0);
            break ;
        }
    }
}

void
CommUnpack(Domain * domain, int msgType)
{
    if (domain->numRanks() == 1) return ;

    /** Retrieve message specific informations */
    Domain_member fields[MAX_FIELDS_PER_MPI_COMM];
    int nfields;
    bool planeOnly;
    Index_t dx, dy, dz;
    CommGetMsgSpecifics(domain, msgType, fields, nfields, planeOnly, dx, dy, dz);

    /* for each requests */
    for (auto & c : domain->m_comm[msgType])
    {
        TASK_SET_COLOR(iter);
        TASK_SET_LABEL("CommUnpack(id=%d, tag=%d, to=%d, type=%d)", c.caseID, c.tag, c.otherRank, c.msgType);
        c.deps.type = MPC_OMP_TASK_DEP_INOUTSET;
        mpc_omp_task_dependencies(&(c.deps), 1);
        # pragma omp task default(none)                         \
            firstprivate(domain, fields, nfields, c, msgType)   \
            depend(in: c.sbuffer, c.rbuffer)
        {
            for (Index_t i = 0 ; i < c.ninodes ; ++i)
            {
                Index_t inode = c.inodes[i];
                for (int fi = 0 ; fi < nfields ; ++fi)
                {
                    // TODO : use a C++ operator function pointer attached
                    // to a communication type, instead of this
                    if (msgType == MSG_DELV)
                    {
                        (domain->*fields[fi])(inode) = c.sbuffer[i * nfields + fi];
                    }
                    else
                    {
                        // TODO : original code apply '+=' on 'fx', 'fy', 'fz' and 'nodalMass'.
                        // On 'nodalMass' seems suspicious to me
                        (domain->*fields[fi])(inode) += c.sbuffer[i * nfields + fi];
                    }
                }
            }
        }
    }
}

void
CommRecv(Domain * domain, int msgType)
{
    if (domain->numRanks() == 1) return ;

    for (auto & c : domain->m_comm[msgType])
    {
        TASK_SET_COLOR(iter);
        TASK_SET_LABEL("CommRecv(id=%d, tag=%d, from=%d, type=%d)", c.caseID, c.tag, c.otherRank, c.msgType);
        mpc_omp_task_fiber();
        # pragma omp task default(none) \
            firstprivate(c)             \
            depend(out: c.rbuffer)      \
            priority(PRIORITY_RECV)     \
            untied                      \
            detach(cc.rhandle)
        {
# if   MPI_MODE == MPI_MODE_NON_BLOCKING
            MPI_Irecv(c.rbuffer, c.count, c.datatype, c.otherRank, c.tag, c.comm, &(c.rreq));
# elif MPI_MODE == MPI_MODE_PERSISTENT
            MPI_Start(&(c.rreq));
# endif
            MPIX_Detach(&(c.rreq), omp_fulfill_event, c.rhandle);
        }
    }
}

void
CommPack(Domain * domain, int msgType)
{
    if (domain->numRanks() == 1) return ;

    /** Retrieve message specific informations */
    Domain_member fields[MAX_FIELDS_PER_MPI_COMM];
    int nfields;
    bool planeOnly;
    Index_t dx, dy, dz;
    CommGetMsgSpecifics(domain, msgType, fields, nfields, planeOnly, dx, dy, dz);

    /* for each requests */
    for (auto & c : domain->m_comm[msgType])
    {
        TASK_SET_COLOR(iter);
        TASK_SET_LABEL("CommPack(id=%d, tag=%d, to=%d, type=%d)", c.caseID, c.tag, c.otherRank, c.msgType);
        c.deps.type = MPC_OMP_TASK_DEP_IN;
        mpc_omp_task_dependencies(&(c.deps), 1);
        # pragma omp task default(none)                 \
            firstprivate(domain, fields, nfields, c)    \
            depend(out: c.sbuffer)
        {
            for (Index_t i = 0 ; i < c.ninodes ; ++i)
            {
                Index_t inode = c.inodes[i];
                for (int fi = 0 ; fi < nfields ; ++fi)
                {
                    c.sbuffer[i * nfields + fi] = (domain->*fields[fi])(inode);
                }
            }
        }
    }
}

void
CommSend(Domain * domain, int msgType)
{
    if (domain->numRanks() == 1) return ;

    std::vector<communication_partite_t> & comms = domain->m_comm[msgType];
    for (auto & c : comms)
    {
        TASK_SET_COLOR(iter);
        TASK_SET_LABEL("CommSend(id=%d, tag=%d, to=%d, type=%d)", c.caseID, c.tag, c.otherRank, c.msgType);
        mpc_omp_task_fiber();
        # pragma omp task default(none) \
            firstprivate(domain, c)     \
            depend(in: c.sbuffer)       \
            untied                      \
            priority(PRIORITY_SEND)     \
            detach(c.shandle)
        {
# if   MPI_MODE == MPI_MODE_NON_BLOCKING
            MPI_Isend(c.sbuffer, c.count, c.datatype, c.otherRank, c.tag, c.comm, &(c.sreq));
# elif MPI_MODE == MPI_MODE_PERSISTENT
            MPI_Start(&(c.sreq));
# endif
            MPIX_Detach(&(c.sreq), omp_fulfill_event, c.shandle);
        }
    }
}

void
CommReduceDt(Real_t * gnewdt, Real_t * newdt)
{
    omp_event_handle_t event;
    # pragma omp task if(0) detach(event)
    {
        MPI_Datatype size = ((sizeof(Real_t) == 4) ? MPI_FLOAT : MPI_DOUBLE);
        MPI_Request req;
        MPI_Status status;
        MPI_Iallreduce(gnewdt, newdt, 1, size, MPI_MIN, MPI_COMM_WORLD, &req);
        MPIX_Detach(&req, omp_fulfill_event, event);
    }
    # pragma omp taskwait
}

#endif
