/**
 * LULESH implementation is based on a uniform cubical mesh.
 * While we do not rely on this anywhere in the code, we still to generate
 * indirection array to represent a truly unstructed mesh.
 */

#if !defined(__LULESH_COMM_CASES_H__) && USE_MPI
# define __LULESH_COMM_CASES_H__

# include <mpi.h>
# if USE_MPC
#  include <mpc_omp.h>
# endif

# include "lulesh-arithmetic.h"

/*****************************************************************/
/*                      MPI COMMUNICATION MODEL                  */
/*****************************************************************/
/* Available MPI_MODE are "non-blocking", "persistent" or "partitionned" */
# define MPI_MODE_NON_BLOCKING  0
# define MPI_MODE_PERSISTENT    1
# define MPI_MODE_PARTITIONNED  2
# ifndef MPI_MODE
//#  define MPI_MODE MPI_MODE_NON_BLOCKING
#  define MPI_MODE MPI_MODE_PERSISTENT
//#  define MPI_MODE MPI_MODE_PARTITIONNED
# endif

# define COMM_RECV 0
# define COMM_SEND 1

/* AXIS DIRECTIONS */
/*
            ^
    		|     ^
            |    /
    	  z |   /
    		|  / y
    		| /
    		|/
    		------------>
                  x
*/

/*****************************************************************/
/*                       FACES NUMBERING                         */
/*****************************************************************/

/** ORIGINAL NUMBERING THAT WAS CONSERVED
 *
 *              .-------------.
 *			   /|            /|
 *			  / |    1      / |
 *			 /  |          /  |
 *			.----------3--.---|
 *			|   |         |   |
 *			|   |         | 5 |
 *			| 4 |         |   |
 *			|   .---2-----|---.
 *			|  /          |  /
 *			| /      0    | /
 *			|/            |/
 *			.-------------.
 */

# define CASE_PLANE_BOTTOM  0
# define CASE_PLANE_TOP     1
# define CASE_PLANE_LEFT    2
# define CASE_PLANE_RIGHT   3
# define CASE_PLANE_BACK    4
# define CASE_PLANE_FRONT   5

# define CASE_IS_PLAN(ID) ((0 <= ID) && (ID <= 5))

/*****************************************************************/
/*                       EDGES NUMBERING                         */
/*****************************************************************/

/** ORIGINAL NUMBERING
 *
 *				.-----10------.
 *			   /|            /|
 *			 14 |          11 |
 *			 /  |          /  |
 *			.-------13----.---|
 *			|   12        |   9
 *			|   |         |   |
 *			|   |         |   |
 *			6   .-----16--15--.
 *			|  /          |  /
 *			| 8           | 17
 *			|/            |/
 *		    .------7------.
 */

/** NEW IMPLEMENTATION NUMBERING
 *				.-----16------.
 *			   /|            /|
 *			 15 |          17 |
 *			 /  |          /  |
 *			.-------14----.---|
 *			|   11        |   12
 *			|   |         |   |
 *			|   |         |   |
 *		   10   .-----8---13--.
 *			|  /          |  /
 *			| 7           | 9
 *			|/            |/
 *		    .------6------.
 */

# define CASE_EDGE_BOTTOM_FRONT 6
# define CASE_EDGE_BOTTOM_LEFT  7
# define CASE_EDGE_BOTTOM_BACK  8
# define CASE_EDGE_BOTTOM_RIGHT 9

# define CASE_EDGE_LEFT_FRONT   10
# define CASE_EDGE_LEFT_BACK    11
# define CASE_EDGE_RIGHT_BACK   12
# define CASE_EDGE_RIGHT_FRONT  13

# define CASE_EDGE_TOP_FRONT    14
# define CASE_EDGE_TOP_LEFT     15
# define CASE_EDGE_TOP_BACK     16
# define CASE_EDGE_TOP_RIGHT    17

# define CASE_IS_EDGE(ID) ((6 <= ID) && (ID <= 17))

/*****************************************************************/
/*                       NODES NUMBERING                         */
/*****************************************************************/

/** ORIGINAL NUMBERING
 *
 *				23------------25
 *			   /|            /|
 *			  / |           / |
 *			 /  |          /  |
 *			19------------21--|
 *			|   |         |   |
 *			|   |         |   |
 *			|   |         |   |
 *			|   22--------|---24
 *			|  /          |  /
 *			| /           | /
 *			|/            |/
 *		    18------------20
 */

/** NEW IMPLEMENTATION NUMBERING
 *
 *				23------------24
 *			   /|            /|
 *			  / |           / |
 *			 /  |          /  |
 *			22------------25--|
 *			|   |         |   |
 *			|   |         |   |
 *			|   |         |   |
 *			|   19--------|---20
 *			|  /          |  /
 *			| /           | /
 *			|/            |/
 *		    18------------21
 */

# define CASE_NODE_BOTTOM_LEFT_FRONT    18
# define CASE_NODE_BOTTOM_LEFT_BACK     19
# define CASE_NODE_BOTTOM_RIGHT_BACK    20
# define CASE_NODE_BOTTOM_RIGHT_FRONT   21

# define CASE_NODE_TOP_LEFT_FRONT       22
# define CASE_NODE_TOP_LEFT_BACK        23
# define CASE_NODE_TOP_RIGHT_BACK       24
# define CASE_NODE_TOP_RIGHT_FRONT      25

# define CASE_IS_NODE(ID) ((18 <= ID) && (ID <= 25))

/* Number of cases (26) */
# define CASE_MAX   26

/******************************************************************************/
/* A frontier communication : can be whether a face, edge or node connection. */
/******************************************************************************/
typedef struct  communication_partite_s
{
    /* partite of the communication */
    int partiteID;

    /* number of elements in the buffers */
    int count;

    /* The node indices involved in the communication */
    Index_t * inodes;
    Index_t ninodes;

# if USE_MPC
    /* The dependencies for (un)packing */
    mpc_omp_task_dependency_t deps;
# endif /* USE_MPC */

    /* this communication details (debug purposes) */
    int caseID;
    int tag;
    int otherRank;
    int msgType;
    MPI_Datatype datatype;
    MPI_Comm comm;

# if (MPI_MODE == MPI_MODE_PERSISTENT) || (MPI_MODE == MPI_MODE_NON_BLOCKING)
    /* The MPI requests */
    MPI_Request rreq;
    MPI_Request sreq;

    /* The MPI statuses */
    MPI_Status rstatus;
    MPI_Status sstatus;
# endif /* (MPI_MODE == MPI_MODE_PERSISTENT) || (MPI_MODE == MPI_MODE_NON_BLOCKING) */

    /* the subbuffer used in the requests */
    Real_t * rbuffer;
    Real_t * sbuffer;

    /* omp event handle */
    omp_event_handle_t rhandle;
    omp_event_handle_t shandle;
}               communication_partite_t;

/**********************/
/* CREATE CASES ARRAY */
/**********************/
typedef struct  communication_case_s
{
    /* the caseID (debug purposes) */
    int caseID;

    /* if this communication existance makes sense (borders...) */
    int (*exists)(int, int, int, int, int, int);

    /* get dimensions of the case (face, edge or node) */
    void (*dimensions)(Index_t, Index_t, Index_t, Index_t *, Index_t *, Index_t *);

    /* function to node_index indices for sequential loop to the indirection array
     *  - dx, dy, dz are the cube dimensions
     *  - i, j, k are the indices relative to the dimensoins given to 'di', 'dj' and 'dk'
     *  through the 'dimensions' function
     */
    Index_t (*node_index)(Index_t, Index_t, Index_t, Index_t, Index_t, Index_t);

    /* get the other MPI rank associated with this communication */
    int (*otherRank)(int, Index_t);

    /* opposite case, to match recv/send */
    int opposite;
}               communication_case_t;

extern communication_case_t COMMUNICATION_CASES[26];

#endif /* __LULESH_COMM_CASES_H__ */
