#include "lulesh.h"

// If no MPI, then this whole file is stubbed out
#if USE_MPI

static double t_send = 0.0;
static double t_allreduce = 0.0;

#include <omp.h>
#include <mpi.h>
#include <string.h>
#include <iostream>

static inline void
wait_req(MPI_Request * req, MPI_Status * status)
{
    int err = MPI_Wait(req, status);
    if (err == MPI_ERR_IN_STATUS)
    {
        assert(status->MPI_ERROR == MPI_SUCCESS);
    }
    else
    {
        assert(err == MPI_SUCCESS);
    }
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
CommRecv(Domain * domain, int msgType)
{
    if (domain->numRanks() == 1) return ;

    std::vector<communication_partite_t> & comms = domain->m_comm[msgType];
    for (auto & comm : comms)
    {
        MPI_Start(&(comm.rreq));
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
    std::vector<communication_partite_t> & comms = domain->m_comm[msgType];
    for (auto & comm : comms)
    {
        for (Index_t i = 0 ; i < comm.ninodes ; ++i)
        {
            Index_t inode = comm.inodes[i];
            for (int fi = 0 ; fi < nfields ; ++fi)
            {
                comm.sbuffer[i * nfields + fi] = (domain->*fields[fi])(inode);
            }
        }
    }
}

void
CommSend(Domain * domain, int msgType)
{
    if (domain->numRanks() == 1) return ;

    std::vector<communication_partite_t> & comms = domain->m_comm[msgType];
    double t0[comms.size()];
    int i = 0;
    for (auto & comm : comms)
    {
        t0[i++] = omp_get_wtime();
        MPI_Start(&(comm.sreq));
    }

    unsigned int done = 0;
    int flags[comms.size()];
    memset(flags, 0, sizeof(flags));
    do {
        i = 0;
        for (auto & comm : comms)
        {
            if (!flags[i])
            {
                MPI_Test(&(comm.sreq), &(flags[i]), &(comm.sstatus));
                if (flags[i])
                {
                    ++done;
                    # pragma omp atomic
                    t_send += omp_get_wtime() - t0[i];
                }
            }
            ++i;
        }
    } while (done != comms.size());
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
    std::vector<communication_partite_t> & comms = domain->m_comm[msgType];
    for (auto & comm : comms)
    {
        wait_req(&(comm.rreq), &(comm.rstatus));
        for (Index_t i = 0 ; i < comm.ninodes ; ++i)
        {
            Index_t inode = comm.inodes[i];
            for (int fi = 0 ; fi < nfields ; ++fi)
            {
                if (msgType == MSG_DELV)
                {
                    (domain->*fields[fi])(inode) = comm.sbuffer[i * nfields + fi];
                }
                else
                {
                    // TODO : original code apply '+=' on 'fx', 'fy', 'fz' and 'nodalMass'.
                    // On 'nodalMass' seems suspicious to me
                    (domain->*fields[fi])(inode) += comm.sbuffer[i * nfields + fi];
                }
            }
        }
    }
}

void
CommReduceDt(Real_t * gnewdt, Real_t * newdt)
{
    MPI_Request req;
    MPI_Status status;
    double t0 = omp_get_wtime();
    MPI_Iallreduce(gnewdt, newdt, 1, ((sizeof(Real_t) == 4) ? MPI_FLOAT : MPI_DOUBLE), MPI_MIN, MPI_COMM_WORLD, &req);
    wait_req(&req, &status);
    # pragma omp atomic
    t_allreduce += omp_get_wtime() - t0;
}

void
CommDumpTime(void)
{
    printf("t_send=%lf, t_allreduce=%lf, t_overlap=0\n", t_send, t_allreduce);
}

#endif
