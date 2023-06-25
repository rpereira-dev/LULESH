#include <math.h>
#if USE_MPI
# include <mpi.h>
#endif
#if _OPENMP
#include <omp.h>
#endif
#if _OMPSS_2
#include <nanos6.h>
#include <nanos6/debug.h>
#endif
#if USE_MPC
# include "../mpc/lulesh-trace.h"
#endif /* USE_MPC */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <cstdlib>
#include "lulesh.h"

#include <map>

/////////////////////////////////////////////////////////////////////
Domain::Domain(Int_t numRanks, Index_t colLoc,
               Index_t rowLoc, Index_t planeLoc,
               Index_t nx, int tp, int nr, int balance, Int_t cost,
               Int_t tel, Int_t tnl, Int_t rpf, Int_t rpe)
   :
   m_e_cut(Real_t(1.0e-7)),
   m_p_cut(Real_t(1.0e-7)),
   m_q_cut(Real_t(1.0e-7)),
   m_v_cut(Real_t(1.0e-10)),
   m_u_cut(Real_t(1.0e-7)),
   m_hgcoef(Real_t(3.0)),
   m_ss4o3(Real_t(4.0)/Real_t(3.0)),
   m_qstop(Real_t(1.0e+12)),
   m_monoq_max_slope(Real_t(1.0)),
   m_monoq_limiter_mult(Real_t(2.0)),
   m_qlc_monoq(Real_t(0.5)),
   m_qqc_monoq(Real_t(2.0)/Real_t(3.0)),
   m_qqc(Real_t(2.0)),
   m_eosvmax(Real_t(1.0e+9)),
   m_eosvmin(Real_t(1.0e-9)),
   m_pmin(Real_t(0.)),
   m_emin(Real_t(-1.0e+15)),
   m_dvovmax(Real_t(0.1)),
   m_refdens(Real_t(1.0)),
   m_tel(tel),
   m_tnl(tnl)
{
    Index_t edgeElems = nx ;
    Index_t edgeNodes = edgeElems+1 ;
    this->cost() = cost;

    m_tp       = tp ;
    m_numRanks = numRanks ;

    ///////////////////////////////
    //   Initialize Sedov Mesh
    ///////////////////////////////

    // construct a uniform box for this processor

    /* find out connections with neighbor ranks */
    m_colLoc    = colLoc;
    m_rowLoc    = rowLoc;
    m_planeLoc  = planeLoc;
    m_hasFront  = (m_rowLoc == 0)          ? 0 : 1;
    m_hasBack   = (m_rowLoc == m_tp - 1)   ? 0 : 1;
    m_hasLeft   = (m_colLoc == 0)          ? 0 : 1;
    m_hasRight  = (m_colLoc == m_tp - 1)   ? 0 : 1;
    m_hasBottom = (m_planeLoc == 0)        ? 0 : 1;
    m_hasTop    = (m_planeLoc == m_tp - 1) ? 0 : 1;

    m_sizeX = edgeElems;
    m_sizeY = edgeElems;
    m_sizeZ = edgeElems;

    m_numElem = edgeElems*edgeElems*edgeElems;
    m_numNode = edgeNodes*edgeNodes*edgeNodes;

    Index_t size = MAX(MAX(sizeX(), sizeY()), sizeZ()) + 1;

    Index_t npf = size * size;
    m_rpf = MIN(npf, rpf);
    m_npfr = MIN(MAX(1, npf / m_rpf), npf);

    Index_t npe = size;
    m_rpe  = MIN(npe, rpe);
    m_nper = MIN(MAX(1, npe / m_rpe), npe);

    m_ebs = m_numElem / m_tel + (m_numElem % m_tel != 0);
    m_nbs = m_numNode / m_tnl + (m_numNode % m_tnl != 0);

    m_regNumList = new Index_t[numElem()] ;  // material indexset

    // Elem-centered
    AllocateElemPersistent(numElem()) ;

    // Node-centered
    AllocateNodePersistent(numNode()) ;

    // Allocate memory
    AllocateStrains(numElem());
    Int_t allElem = numElem() +         /* local elem */
        2 * sizeX() * sizeY() +         /* plane ghosts */
        2 * sizeX() * sizeZ() +         /* row ghosts */
        2 * sizeY() * sizeZ() ;         /* col ghosts */
    AllocateGradients(numElem(), allElem);

#if USE_MPI
    SetupCommBuffers();
# if USE_MPC && TRACE
    if (TRACE_CONDITION)
    {
        int ncomm = 0;
        for (int caseID = 0 ; caseID < CASE_MAX ; ++caseID)
        {
            communication_case_t * cas = COMMUNICATION_CASES + caseID;
            ncomm += cas->exists(m_hasBottom, m_hasTop, m_hasFront, m_hasBack, m_hasLeft, m_hasRight);
        }
        printf("Rank %d has %d/%d communication cases to perform\n", myRank, ncomm, CASE_MAX);
    }
# endif /* USE_MPC && TRACE */
#endif /* USE_MPI */

    // Boundary nodesets
    if (m_colLoc == 0)      m_symmX.resize(edgeNodes*edgeNodes);
    if (m_rowLoc == 0)      m_symmY.resize(edgeNodes*edgeNodes);
    if (m_planeLoc == 0)    m_symmZ.resize(edgeNodes*edgeNodes);

    // Basic Field Initialization
    for (Index_t i=0; i<numElem(); ++i) {
        e(i) =  Real_t(0.0) ;
        p(i) =  Real_t(0.0) ;
        q(i) =  Real_t(0.0) ;
        ss(i) = Real_t(0.0) ;
    }

    // Note - v initializes to 1.0, not 0.0!
    for (Index_t i=0; i<numElem(); ++i) {
        v(i) = Real_t(1.0) ;
    }

    for (Index_t i=0; i<numNode(); ++i) {
        xd(i) = Real_t(0.0) ;
        yd(i) = Real_t(0.0) ;
        zd(i) = Real_t(0.0) ;
    }

    for (Index_t i=0; i<numNode(); ++i) {
        xdd(i) = Real_t(0.0) ;
        ydd(i) = Real_t(0.0) ;
        zdd(i) = Real_t(0.0) ;
    }

    for (Index_t i=0; i<numNode(); ++i) {
        nodalMass(i) = Real_t(0.0) ;
    }

    BuildMesh(nx, edgeNodes, edgeElems);

    SetupThreadSupportStructures();

    // Setup region index sets. For now, these are constant sized
    // throughout the run, but could be changed every cycle to
    // simulate effects of ALE on the lagrange solver
    CreateRegionIndexSets(nr, balance);

    // Setup symmetry nodesets
    SetupSymmetryPlanes(edgeNodes);

    // Setup element connectivities
    SetupElementConnectivities(edgeElems);

    // Setup symmetry planes and free surface boundary arrays
    SetupBoundaryConditions(edgeElems);


    // Setup defaults

    // These can be changed (requires recompile) if you want to run
    // with a fixed timestep, or to a different end time, but it's
    // probably easier/better to just run a fixed number of timesteps
    // using the -i flag in 2.x

    dtfixed() = Real_t(-1.0e-6) ; // Negative means use courant condition
    stoptime()  = Real_t(1.0e-2); // *Real_t(edgeElems*tp/45.0) ;

    // Initial conditions
    deltatimemultlb() = Real_t(1.1) ;
    deltatimemultub() = Real_t(1.2) ;
    dtcourant() = Real_t(1.0e+20) ;
    dthydro()   = Real_t(1.0e+20) ;
    dtmax()     = Real_t(1.0e-2) ;
    time()    = Real_t(0.) ;
    cycle()   = Int_t(0) ;

    // initialize field data
    for (Index_t i=0; i<numElem(); ++i) {
        Real_t x_local[8], y_local[8], z_local[8] ;
        Index_t *elemToNode = nodelist(i) ;
        for( Index_t lnode=0 ; lnode<8 ; ++lnode )
        {
            Index_t gnode = elemToNode[lnode];
            x_local[lnode] = x(gnode);
            y_local[lnode] = y(gnode);
            z_local[lnode] = z(gnode);
        }

        // volume calculations
        Real_t volume = CalcElemVolume(x_local, y_local, z_local );
        volo(i) = volume ;
        elemMass(i) = volume ;
        for (Index_t j=0; j<8; ++j) {
            Index_t idx = elemToNode[j] ;
            nodalMass(idx) += volume / Real_t(8.0) ;
        }
    }

    // deposit initial energy
    // An energy of 3.948746e+7 is correct for a problem with
    // 45 zones along a side - we need to scale it
    const Real_t ebase = Real_t(3.948746e+7);
    Real_t scale = (nx*m_tp)/Real_t(45.0);
    Real_t einit = ebase*scale*scale*scale;
    if (m_rowLoc + m_colLoc + m_planeLoc == 0) {
        // Dump into the first zone (which we know is in the corner)
        // of the domain that sits at the origin
        e(0) = einit;
    }
    //set initial deltatime base on analytic CFL calculation
    deltatime() = (Real_t(.5)*cbrt(volo(0)))/sqrt(Real_t(2.0)*einit);

} // End constructor


////////////////////////////////////////////////////////////////////////////////
Domain::~Domain()
{
    delete [] m_regNumList;
    delete [] m_nodeElemStart;
    delete [] m_nodeElemCornerList;
    delete [] m_regElemSize;
    for (Index_t i=0 ; i<numReg() ; ++i) {
        delete [] m_regElemlist[i];
    }
    delete [] m_regElemlist;

    DeallocateStrains();
    DeallocateGradients();

#if USE_MPI && USE_MPC
    for (int msgType = 0 ; msgType < MSG_MAX ; ++msgType)
    {
        std::vector<communication_partite_t> & comms = m_comm[msgType];
        for (auto & comm : comms)
        {
            free(comm.deps.addrs);
            free(comm.inodes);
# if MPI_MODE == MPI_MODE_PERSISTENT
            MPI_Request_free(&(comm.rreq));
            MPI_Request_free(&(comm.sreq));
# endif /* MPI_MODE == MPI_MODE_PERSISTENT */
        }
# if MPI_MODE == MPI_MODE_PARTITIONNED
    // TODO : free partitionned requests
    // MPI_Request_free(...);
    // MPI_Request_free(...);
# endif /* MPI_MODE == MPI_MODE_PARTITIONNED */
    }
    free(m_rbuffer);
    free(m_sbuffer);
#endif
} // End destructor

////////////////////////////////////////////////////////////////////////////////
void
Domain::BuildMesh(Int_t nx, Int_t edgeNodes, Int_t edgeElems)
{
    Index_t meshEdgeElems = m_tp*nx ;

    // initialize nodal coordinates
    Index_t nidx = 0 ;
    Real_t tz = Real_t(1.125)*Real_t(m_planeLoc*nx)/Real_t(meshEdgeElems) ;
    for (Index_t plane=0; plane<edgeNodes; ++plane) {
        Real_t ty = Real_t(1.125)*Real_t(m_rowLoc*nx)/Real_t(meshEdgeElems) ;
        for (Index_t row=0; row<edgeNodes; ++row) {
            Real_t tx = Real_t(1.125)*Real_t(m_colLoc*nx)/Real_t(meshEdgeElems) ;
            for (Index_t col=0; col<edgeNodes; ++col) {
                x(nidx) = tx ;
                y(nidx) = ty ;
                z(nidx) = tz ;
                ++nidx ;
                // tx += ds ; // may accumulate roundoff...
                tx = Real_t(1.125)*Real_t(m_colLoc*nx+col+1)/Real_t(meshEdgeElems) ;
            }
            // ty += ds ;  // may accumulate roundoff...
            ty = Real_t(1.125)*Real_t(m_rowLoc*nx+row+1)/Real_t(meshEdgeElems) ;
        }
        // tz += ds ;  // may accumulate roundoff...
        tz = Real_t(1.125)*Real_t(m_planeLoc*nx+plane+1)/Real_t(meshEdgeElems) ;
    }


    // embed hexehedral elements in nodal point lattice
    Index_t zidx = 0 ;
    nidx = 0 ;
    for (Index_t plane=0; plane<edgeElems; ++plane) {
        for (Index_t row=0; row<edgeElems; ++row) {
            for (Index_t col=0; col<edgeElems; ++col) {
                Index_t *localNode = nodelist(zidx) ;
                localNode[0] = nidx                                       ;
                localNode[1] = nidx                                   + 1 ;
                localNode[2] = nidx                       + edgeNodes + 1 ;
                localNode[3] = nidx                       + edgeNodes     ;
                localNode[4] = nidx + edgeNodes*edgeNodes                 ;
                localNode[5] = nidx + edgeNodes*edgeNodes             + 1 ;
                localNode[6] = nidx + edgeNodes*edgeNodes + edgeNodes + 1 ;
                localNode[7] = nidx + edgeNodes*edgeNodes + edgeNodes     ;
                ++zidx ;
                ++nidx ;
            }
            ++nidx ;
        }
        nidx += edgeNodes ;
    }
}


////////////////////////////////////////////////////////////////////////////////
void
Domain::SetupThreadSupportStructures()
{
    // set up node-centered indexing of elements
    Index_t *nodeElemCount = new Index_t[numNode()] ;

    for (Index_t i=0; i<numNode(); ++i) {
        nodeElemCount[i] = 0 ;
    }

    for (Index_t i=0; i<numElem(); ++i) {
        Index_t *nl = nodelist(i) ;
        for (Index_t j=0; j < 8; ++j) {
            ++(nodeElemCount[nl[j]] );
        }
    }

    m_nodeElemStart = new Index_t[numNode()+1] ;

    m_nodeElemStart[0] = 0;

    for (Index_t i=1; i <= numNode(); ++i) {
        m_nodeElemStart[i] =
            m_nodeElemStart[i-1] + nodeElemCount[i-1] ;
    }

    m_nodeElemCornerList = new Index_t[m_nodeElemStart[numNode()]];

    for (Index_t i=0; i < numNode(); ++i) {
        nodeElemCount[i] = 0;
    }

    for (Index_t i=0; i < numElem(); ++i) {
        Index_t *nl = nodelist(i) ;
        for (Index_t j=0; j < 8; ++j) {
            Index_t m = nl[j];
            Index_t k = i*8 + j ;
            Index_t offset = m_nodeElemStart[m] + nodeElemCount[m] ;
            m_nodeElemCornerList[offset] = k;
            ++(nodeElemCount[m]) ;
        }
    }

    Index_t clSize = m_nodeElemStart[numNode()] ;
    for (Index_t i=0; i < clSize; ++i) {
        Index_t clv = m_nodeElemCornerList[i] ;
        if ((clv < 0) || (clv > numElem()*8)) {
            fprintf(stderr, "AllocateNodeElemIndexes(): nodeElemCornerList entry out of range!\n");
#if USE_MPI
            MPI_Abort(MPI_COMM_WORLD, -1);
#else
            exit(-1);
#endif
        }
    }

    delete [] nodeElemCount ;
}

////////////////////////////////////////////////////////////////////////////////
/********************************/
/*  MPI REQUESTS INITIALIZATION */
/********************************/
#if USE_MPI
void
Domain::SetupCommBuffersForMsg(int msgType)
{
    /* Dupplicate world communicator per message type */
    MPI_Comm & comm = m_mpi_comm[msgType];
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);

    /** Retrieve message specific informations */
    Domain_member fields[MAX_FIELDS_PER_MPI_COMM];
    int nfields;
    bool planeOnly;
    Index_t dx, dy, dz;
    CommGetMsgSpecifics(this, msgType, fields, nfields, planeOnly, dx, dy, dz);

    /* communication lists for this message */
    std::vector<communication_partite_t> & communications = m_comm[msgType];

    /* requests constants */
    MPI_Datatype datatype = ((sizeof(Real_t) == 4) ? MPI_FLOAT : MPI_DOUBLE);
    int myRank;
    MPI_Comm_rank(comm, &myRank);

    /* compute the total size for the communications buffer */
    size_t buffer_size = 0;
    for (int caseID = 0 ; caseID < CASE_MAX ; ++caseID)
    {
        communication_case_t * cas = COMMUNICATION_CASES + caseID;
        if (planeOnly && !CASE_IS_PLAN(caseID)) continue ;

        if (cas->exists(m_hasBottom, m_hasTop, m_hasFront, m_hasBack, m_hasLeft, m_hasRight))
        {
            // communication dimension and number of nodes involved
            Index_t di, dj, dk;
            cas->dimensions(dx, dy, dz, &di, &dj, &dk);

            // nodes implied in the communication remaining
            Index_t remains = di * dj * dk;
            buffer_size += remains;
        }
    }
    buffer_size *= nfields;

    m_rbuffer = (Real_t *) malloc(sizeof(Real_t) * buffer_size);
    assert(m_rbuffer);

    m_sbuffer = (Real_t *) malloc(sizeof(Real_t) * buffer_size);
    assert(m_sbuffer);

    size_t offset = 0;

    /* Process all cases */
    for (int caseID = 0 ; caseID < CASE_MAX ; ++caseID)
    {
        communication_case_t * cas = COMMUNICATION_CASES + caseID;
        if (planeOnly && !CASE_IS_PLAN(caseID)) continue ;

        // maximum number of nodes per request
        Index_t npr = CASE_IS_PLAN(caseID) ? m_npfr : CASE_IS_EDGE(caseID) ? m_nper : 1;

        if (cas->exists(m_hasBottom, m_hasTop, m_hasFront, m_hasBack, m_hasLeft, m_hasRight))
        {
# if MPI_MODE == MPI_MODE_PARTITIONNED
            Real_t * sbuffer = m_sbuffer + offset;          // TODO : check this
            Real_t * rbuffer = m_rbuffer + offset;          // TODO : check this

            int partitions  = npr;
            int count       = di * dj * dk * nfields / npr; // TODO : check this
            int dest        = cas->otherRank(myRank, m_tp);
            int tag         = 0;

            MPI_Request rrecv;                      // TODO : move this to object attributes
            MPI_Request rsend;                      // TODO : move this to object attributes
            MPI_Precv_init(rbuffer, partitions, count, datatype, dest, tag, comm, MPI_INFO_NULL, &rrecv);
            MPI_Psend_init(sbuffer, partitions, count, datatype, dest, tag, comm, MPI_INFO_NULL, &rsend);
            assert(0);

            // TODO : an issue is the OpenMP dependency management

            // TODO : increment offset

            // TODO : we don't need the upcoming loops in case of partitionned communications
            // TODO : remove this loops and refactor 'mpc/lulesh-comm.cc'

# else /* MPI_MODE == MPI_MODE_PARTITIONNED */

            // communication dimension and number of nodes involved
            Index_t di, dj, dk;
            cas->dimensions(dx, dy, dz, &di, &dj, &dk);

            // nodes implied in the communication remaining
            Index_t remains = di * dj * dk;

            // unique request ID for this communication
            Index_t partiteID = 0;

            // the nodes involved this request
            Index_t * inodes = (Index_t *) malloc(sizeof(Index_t) * MIN(remains, npr));

            // filter redundant node blocks
            std::map<Index_t, bool> blocks;

            // number of nodes processed for this request
            Index_t ninodes = 0;

            // for each nodes involved in the communications
            for (Index_t i = 0 ; i < di ; ++i)
            {
                for (Index_t j = 0 ; j < dj ; ++j)
                {
                    for (Index_t k = 0 ; k < dk ; ++k)
                    {
                        // find out the node index in the nodes list when packing data to send
                        Index_t inode = cas->node_index(dx, dy, dz, i, j, k);

                        // find matching node block for dependencies resolution */
                        Index_t blockID = inode / m_nbs * m_nbs;
                        if (blocks.count(blockID) == 0) blocks[blockID] = true;

                        // save the node
                        inodes[ninodes++] = inode;

                        // if we processed every nodes for a partite,
                        //  - generate the request
                        //  - process next nodes to generate the next partite
                        if (ninodes == npr || ((i == di - 1) && (j == dj - 1) && (k == dk - 1)))
                        {
                            // initialize communication
                            communication_partite_t c;

                            // partite communication infos
                            c.partiteID = partiteID;
                            c.count     = nfields * ninodes;
                            c.inodes    = inodes;
                            c.ninodes   = ninodes;

# if USE_MPC
                            // save nodes block as openmp dependencies array
                            c.deps.addrs_size  = nfields * blocks.size();
                            c.deps.addrs       = (void **) malloc(sizeof(void *) * c.deps.addrs_size);

                            // register the addresses of each block of each field to the dependencies array
                            unsigned int l = 0;
                            for (std::map<Index_t, bool>::iterator it = blocks.begin(); it != blocks.end(); ++it)
                            {
                                for (int fi = 0 ; fi < nfields ; fi++)
                                {
                                    c.deps.addrs[l++] = &(this->*fields[fi])(it->first);
                                }
                            }
                            assert(l == c.deps.addrs_size);
                            blocks.clear();
# endif /* USE_MPC */
                            c.rbuffer = m_rbuffer + offset;
                            c.sbuffer = m_sbuffer + offset;
                            offset += c.count;

                            // save fields (for debug purposes)
                            c.caseID    = caseID;
                            c.tag       = partiteID;
                            c.otherRank = cas->otherRank(myRank, m_tp);
                            c.msgType   = msgType;
                            c.datatype  = datatype;
                            c.comm      = comm;

# if (MPI_MODE == MPI_MODE_PERSISTENT) || (MPI_MODE == MPI_MODE_NON_BLOCKING)
                            // set MPI request
                            c.rreq = MPI_REQUEST_NULL;
                            c.sreq = MPI_REQUEST_NULL;
# endif /* (MPI_MODE == MPI_MODE_PERSISTENT) || (MPI_MODE == MPI_MODE_NON_BLOCKING) */

# if   MPI_MODE == MPI_MODE_PERSISTENT
                            MPI_Recv_init(c.rbuffer, c.count, c.datatype, c.otherRank, c.tag, c.comm, &(c.rreq));
                            MPI_Send_init(c.sbuffer, c.count, c.datatype, c.otherRank, c.tag, c.comm, &(c.sreq));
# endif /* MPI_MODE == MPI_MODE_PERSISTENT */

                            communications.push_back(c);

                            ///////////////////////////////////////////////
                            // RESET VARIABLES FOR NEXT BLOCK GENERATION //
                            ///////////////////////////////////////////////

                            // number of nodes for the next communication block
                            remains = remains - ninodes;
                            if (remains > 0)
                            {
                                // it is a new block of 'ninodes' nodes
                                // reset the number of nodes for the next request
                                ++partiteID;
                                inodes = (Index_t *) malloc(sizeof(Index_t) * MIN(remains, npr));
                                blocks.clear();
                                ninodes = 0;
                            }
                        }
                    } /* for k */
                } /* for j */
            } /* for i */
# endif /* MPI_MODE == MPI_MODE_PARTITIONNED */
        }
    }
}

void
Domain::SetupCommBuffers(void)
{
    // allocate a buffer large enough for nodal ghost data
    Index_t maxEdgeSize = MAX(this->sizeX(), MAX(this->sizeY(), this->sizeZ())) + 1;
    m_maxPlaneSize  = CACHE_ALIGN_REAL(maxEdgeSize*maxEdgeSize) ;
    m_maxEdgeSize   = CACHE_ALIGN_REAL(maxEdgeSize) ;

    if (m_numRanks > 1)
    {
        SetupCommBuffersForMsg(MSG_MASS);
        SetupCommBuffersForMsg(MSG_FX_FY_FZ);
        SetupCommBuffersForMsg(MSG_DELV);
    }
}
#endif /* USE_MPI */


////////////////////////////////////////////////////////////////////////////////
void
Domain::CreateRegionIndexSets(Int_t nr, Int_t balance)
{
#if USE_MPI
    int myRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    srand(myRank);
#else
    srand(0);
    int myRank = 0;
#endif
    this->numReg() = nr;
    m_regElemSize = new Index_t[numReg()];
    m_regElemlist = new Index_t*[numReg()];
    Index_t nextIndex = 0;
    //if we only have one region just fill it
    // Fill out the regNumList with material numbers, which are always
    // the region index plus one
    if(numReg() == 1) {
        while (nextIndex < numElem()) {
            this->regNumList(nextIndex) = 1;
            nextIndex++;
        }
        regElemSize(0) = 0;
    }
    //If we have more than one region distribute the elements.
    else {
        Int_t regionNum;
        Int_t regionVar;
        Int_t lastReg = -1;
        Int_t binSize;
        Index_t elements;
        Index_t runto = 0;
        Int_t costDenominator = 0;
        Int_t* regBinEnd = new Int_t[numReg()];
        //Determine the relative weights of all the regions.  This is based off the -b flag.  Balance is the value passed into b.
        for (Index_t i=0 ; i<numReg() ; ++i) {
            regElemSize(i) = 0;
            costDenominator += pow((i+1), balance);  //Total sum of all regions weights
            regBinEnd[i] = costDenominator;  //Chance of hitting a given region is (regBinEnd[i] - regBinEdn[i-1])/costDenominator
        }
        //Until all elements are assigned
        while (nextIndex < numElem()) {
            //pick the region
            regionVar = rand() % costDenominator;
            Index_t i = 0;
            while(regionVar >= regBinEnd[i])
                i++;
            //rotate the regions based on MPI rank.  Rotation is Rank % NumRegions this makes each domain have a different region with
            //the highest representation
            regionNum = ((i + myRank) % numReg()) + 1;
            // make sure we don't pick the same region twice in a row
            while(regionNum == lastReg) {
                regionVar = rand() % costDenominator;
                i = 0;
                while(regionVar >= regBinEnd[i])
                    i++;
                regionNum = ((i + myRank) % numReg()) + 1;
            }
            //Pick the bin size of the region and determine the number of elements.
            binSize = rand() % 1000;
            if(binSize < 773) {
                elements = rand() % 15 + 1;
            }
            else if(binSize < 937) {
                elements = rand() % 16 + 16;
            }
            else if(binSize < 970) {
                elements = rand() % 32 + 32;
            }
            else if(binSize < 974) {
                elements = rand() % 64 + 64;
            }
            else if(binSize < 978) {
                elements = rand() % 128 + 128;
            }
            else if(binSize < 981) {
                elements = rand() % 256 + 256;
            }
            else
                elements = rand() % 1537 + 512;
            runto = elements + nextIndex;
            //Store the elements.  If we hit the end before we run out of elements then just stop.
            while (nextIndex < runto && nextIndex < numElem()) {
                this->regNumList(nextIndex) = regionNum;
                nextIndex++;
            }
            lastReg = regionNum;
        }
    }
    // Convert regNumList to region index sets
    // First, count size of each region
    for (Index_t i=0 ; i<numElem() ; ++i) {
        int r = this->regNumList(i)-1; // region index == regnum-1
        regElemSize(r)++;
    }
    // Second, allocate each region index set
    for (Index_t i=0 ; i<numReg() ; ++i) {
        m_regElemlist[i] = new Index_t[regElemSize(i)];
        regElemSize(i) = 0;
    }
    // Third, fill index sets
    for (Index_t i=0 ; i<numElem() ; ++i) {
        Index_t r = regNumList(i)-1;       // region index == regnum-1
        Index_t regndx = regElemSize(r)++; // Note increment
        regElemlist(r,regndx) = i;
    }
}

/////////////////////////////////////////////////////////////
void
Domain::SetupSymmetryPlanes(Int_t edgeNodes)
{
    Index_t nidx = 0 ;
    for (Index_t i=0; i<edgeNodes; ++i) {
        Index_t planeInc = i*edgeNodes*edgeNodes ;
        Index_t rowInc   = i*edgeNodes ;
        for (Index_t j=0; j<edgeNodes; ++j) {
            if (m_planeLoc == 0) {
                m_symmZ[nidx] = rowInc   + j ;
            }
            if (m_rowLoc == 0) {
                m_symmY[nidx] = planeInc + j ;
            }
            if (m_colLoc == 0) {
                m_symmX[nidx] = planeInc + j*edgeNodes ;
            }
            ++nidx ;
        }
    }
}

/////////////////////////////////////////////////////////////
void
Domain::SetupElementConnectivities(Int_t edgeElems)
{
    lxim(0) = 0 ;
    for (Index_t i=1; i<numElem(); ++i) {
        lxim(i)   = i-1 ;
        lxip(i-1) = i ;
    }
    lxip(numElem()-1) = numElem()-1 ;

    for (Index_t i=0; i<edgeElems; ++i) {
        letam(i) = i ;
        letap(numElem()-edgeElems+i) = numElem()-edgeElems+i ;
    }
    for (Index_t i=edgeElems; i<numElem(); ++i) {
        letam(i) = i-edgeElems ;
        letap(i-edgeElems) = i ;
    }

    for (Index_t i=0; i<edgeElems*edgeElems; ++i) {
        lzetam(i) = i ;
        lzetap(numElem()-edgeElems*edgeElems+i) = numElem()-edgeElems*edgeElems+i ;
    }
    for (Index_t i=edgeElems*edgeElems; i<numElem(); ++i) {
        lzetam(i) = i - edgeElems*edgeElems ;
        lzetap(i-edgeElems*edgeElems) = i ;
    }
}

/////////////////////////////////////////////////////////////
void
Domain::SetupBoundaryConditions(Int_t edgeElems)
{
    Index_t ghostIdx[6] ;  // offsets to ghost locations

    // set up boundary condition information
    for (Index_t i=0; i<numElem(); ++i) {
        elemBC(i) = Int_t(0) ;
    }

    for (Index_t i=0; i<6; ++i) {
        ghostIdx[i] = INT_MIN ;
    }

    Int_t pidx = numElem() ;
    if (m_hasBottom != 0) {
        ghostIdx[0] = pidx ;
        pidx += sizeX()*sizeY() ;
    }

    if (m_hasTop != 0) {
        ghostIdx[1] = pidx ;
        pidx += sizeX()*sizeY() ;
    }

    if (m_hasFront != 0) {
        ghostIdx[2] = pidx ;
        pidx += sizeX()*sizeZ() ;
    }

    if (m_hasBack != 0) {
        ghostIdx[3] = pidx ;
        pidx += sizeX()*sizeZ() ;
    }

    if (m_hasLeft != 0) {
        ghostIdx[4] = pidx ;
        pidx += sizeY()*sizeZ() ;
    }

    if (m_hasRight != 0) {
        ghostIdx[5] = pidx ;
    }

    // symmetry plane or free surface BCs
    for (Index_t i=0; i<edgeElems; ++i) {
        Index_t planeInc = i*edgeElems*edgeElems ;
        Index_t rowInc   = i*edgeElems ;
        for (Index_t j=0; j<edgeElems; ++j) {
            if (m_planeLoc == 0) {
                elemBC(rowInc+j) |= ZETA_M_SYMM ;
            }
            else {
                elemBC(rowInc+j) |= ZETA_M_COMM ;
                lzetam(rowInc+j) = ghostIdx[0] + rowInc + j ;
            }

            if (m_planeLoc == m_tp-1) {
                elemBC(rowInc+j+numElem()-edgeElems*edgeElems) |=
                    ZETA_P_FREE;
            }
            else {
                elemBC(rowInc+j+numElem()-edgeElems*edgeElems) |=
                    ZETA_P_COMM ;
                lzetap(rowInc+j+numElem()-edgeElems*edgeElems) =
                    ghostIdx[1] + rowInc + j ;
            }

            if (m_rowLoc == 0) {
                elemBC(planeInc+j) |= ETA_M_SYMM ;
            }
            else {
                elemBC(planeInc+j) |= ETA_M_COMM ;
                letam(planeInc+j) = ghostIdx[2] + rowInc + j ;
            }

            if (m_rowLoc == m_tp-1) {
                elemBC(planeInc+j+edgeElems*edgeElems-edgeElems) |=
                    ETA_P_FREE ;
            }
            else {
                elemBC(planeInc+j+edgeElems*edgeElems-edgeElems) |=
                    ETA_P_COMM ;
                letap(planeInc+j+edgeElems*edgeElems-edgeElems) =
                    ghostIdx[3] +  rowInc + j ;
            }

            if (m_colLoc == 0) {
                elemBC(planeInc+j*edgeElems) |= XI_M_SYMM ;
            }
            else {
                elemBC(planeInc+j*edgeElems) |= XI_M_COMM ;
                lxim(planeInc+j*edgeElems) = ghostIdx[4] + rowInc + j ;
            }

            if (m_colLoc == m_tp-1) {
                elemBC(planeInc+j*edgeElems+edgeElems-1) |= XI_P_FREE ;
            }
            else {
                elemBC(planeInc+j*edgeElems+edgeElems-1) |= XI_P_COMM ;
                lxip(planeInc+j*edgeElems+edgeElems-1) =
                    ghostIdx[5] + rowInc + j ;
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////
void InitMeshDecomp(Int_t numRanks, Int_t myRank,
        Int_t *col, Int_t *row, Int_t *plane, Int_t *side)
{
    Int_t testProcs;
    Int_t dx, dy, dz;
    Int_t myDom;

    // Assume cube processor layout for now
    testProcs = Int_t(cbrt(Real_t(numRanks))+0.5) ;
    if (testProcs*testProcs*testProcs != numRanks) {
        printf("Num processors must be a cube of an integer (1, 8, 27, ...)\n") ;
#if USE_MPI
        MPI_Abort(MPI_COMM_WORLD, -1) ;
#else
        exit(-1);
#endif
    }
    if (sizeof(Real_t) != 4 && sizeof(Real_t) != 8) {
        printf("MPI operations only support float and double right now...\n");
#if USE_MPI
        MPI_Abort(MPI_COMM_WORLD, -1) ;
#else
        exit(-1);
#endif
    }
    if (MAX_FIELDS_PER_MPI_COMM > CACHE_COHERENCE_PAD_REAL) {
        printf("corner element comm buffers too small.  Fix code.\n") ;
#if USE_MPI
        MPI_Abort(MPI_COMM_WORLD, -1) ;
#else
        exit(-1);
#endif
    }

    dx = testProcs ;
    dy = testProcs ;
    dz = testProcs ;

    // temporary test
    if (dx*dy*dz != numRanks) {
        printf("error -- must have as many domains as procs\n") ;
#if USE_MPI
        MPI_Abort(MPI_COMM_WORLD, -1) ;
#else
        exit(-1);
#endif
    }
    Int_t remainder = dx*dy*dz % numRanks ;
    if (myRank < remainder) {
        myDom = myRank*( 1+ (dx*dy*dz / numRanks)) ;
    }
    else {
        myDom = remainder*( 1+ (dx*dy*dz / numRanks)) +
            (myRank - remainder)*(dx*dy*dz/numRanks) ;
    }

    *col = myDom % dx ;
    *row = (myDom / dx) % dy ;
    *plane = myDom / (dx*dy) ;
    *side = testProcs;
}

