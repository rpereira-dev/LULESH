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

// TODO : implement partitionnde communication version

static inline void
wait_req(MPI_Request * req, MPI_Status * status)
{
# if USE_MPIX
    int err = MPIX_Wait(req, status);
# else /* USE_MPIX */
    int err = MPI_Wait(req, status);
# endif /* USE_MPIX */

    if (err == MPI_ERR_IN_STATUS)
    {
        assert(status->MPI_ERROR == MPI_SUCCESS);
    }
    else
    {
        assert(err == MPI_SUCCESS);
    }
}

static inline void
wait_comm(communication_partite_t & c, int type)
{
# if (MPI_MODE == MPI_MODE_PERSISTENT) || (MPI_MODE == MPI_MODE_NON_BLOCKING)
    MPI_Request * req    = (type == COMM_SEND) ? &(c.sreq)    : &(c.rreq);
    MPI_Status  * status = (type == COMM_SEND) ? &(c.sstatus) : &(c.rstatus);
    wait_req(req, status);
# endif /* (MPI_MODE == MPI_MODE_PERSISTENT) || (MPI_MODE == MPI_MODE_NON_BLOCKING) */
}

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
            untied
        {
#ifdef TRACE
            MPC_OMP_TASK_TRACE_RECV(0, 0, c.otherRank, c.tag, 0, 0); // TODO : remove this, and use PMPI
#endif
# if   MPI_MODE == MPI_MODE_NON_BLOCKING
            MPI_Irecv(c.rbuffer, c.count, c.datatype, c.otherRank, c.tag, c.comm, &(c.rreq));
# elif MPI_MODE == MPI_MODE_PERSISTENT
            MPI_Start(&(c.rreq));
# endif
            wait_comm(c, COMM_RECV);
#ifdef TRACE
            MPC_OMP_TASK_TRACE_RECV(0, 0, c.otherRank, c.tag, 0, 1); // TODO : remove this, and use PMPI
#endif
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
            priority(PRIORITY_SEND)
        {
#ifdef TRACE
            MPC_OMP_TASK_TRACE_SEND(0, 0, c.otherRank, c.tag, 0, 0); // TODO : remove this, and use PMPI
#endif  /* TRACE */
# if   MPI_MODE == MPI_MODE_NON_BLOCKING
            MPI_Isend(c.sbuffer, c.count, c.datatype, c.otherRank, c.tag, c.comm, &(c.sreq));
# elif MPI_MODE == MPI_MODE_PERSISTENT
            MPI_Start(&(c.sreq));
# endif
            wait_comm(c, COMM_SEND);
#ifdef TRACE
            MPC_OMP_TASK_TRACE_SEND(0, 0, c.otherRank, c.tag, 0, 1); // TODO : remove this, and use PMPI
#endif  /* TRACE */
        }
    }
}

void
CommReduceDt(Real_t * gnewdt, Real_t * newdt)
{
    MPI_Datatype size = ((sizeof(Real_t) == 4) ? MPI_FLOAT : MPI_DOUBLE);
    MPI_Request req;
    MPI_Status status;
# ifdef TRACE
    MPC_OMP_TASK_TRACE_ALLREDUCE(1, 0, 0, 0, 0);   // TODO : remove this, and use PMPI
# endif /* TRACE */
    MPI_Iallreduce(gnewdt, newdt, 1, size, MPI_MIN, MPI_COMM_WORLD, &req);
    wait_req(&req, &status);
# ifdef TRACE
    MPC_OMP_TASK_TRACE_ALLREDUCE(1, 0, 0, 0, 1);   // TODO : remove this, and use PMPI
# endif /* TRACE */
}

#endif
