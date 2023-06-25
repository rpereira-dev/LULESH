/**
 * LULESH implementation is based on a uniform cubical mesh.
 * While we do not rely on this anywhere in the code, we still have
 * to generate indirection array to represent a truly unstructed mesh.
 *
 *  -x+ to -y+ to -z+ partitionning implementation down here
 */

# if USE_MPI
# include "lulesh-comm-cases.h"

/******************************/
/* EACH CASES SPECIFIC VALUES */
/******************************/

/****************************** CASE 0 ************************************/

static inline Index_t node_index0(Index_t dx, Index_t dy, Index_t dz, Index_t i, Index_t j, Index_t k)
{
    return j*dx + i;
}

static inline int exists0(int bottom, int top, int front, int back, int left, int right)
{
    return bottom;
}

static inline void dimensions0(Index_t dx, Index_t dy, Index_t dz, Index_t * di, Index_t * dj, Index_t * dk)
{
    *di = dx;
    *dj = dy;
    *dk = 1;
}

static inline int otherRank0(int rank, Index_t tp)
{
    return rank - tp * tp;
}

/****************************** CASE 1 ************************************/

static inline Index_t node_index1(Index_t dx, Index_t dy, Index_t dz, Index_t i, Index_t j, Index_t k)
{
    return dx*dy*(dz-1) + node_index0(dx, dy, dz, i, j, k);
}

static inline int exists1(int bottom, int top, int front, int back, int left, int right)
{
    return top;
}

static inline void dimensions1(Index_t dx, Index_t dy, Index_t dz, Index_t * di, Index_t * dj, Index_t * dk)
{
    return dimensions0(dx, dy, dz, di, dj, dk);
}

static inline int otherRank1(int rank, Index_t tp)
{
    return rank + tp * tp;
}

/****************************** CASE 2 ************************************/

static inline Index_t node_index2(Index_t dx, Index_t dy, Index_t dz, Index_t i, Index_t j, Index_t k)
{
    return k*dx*dy + i;
}

static inline int exists2(int bottom, int top, int front, int back, int left, int right)
{
    return front;
}

static inline void dimensions2(Index_t dx, Index_t dy, Index_t dz, Index_t * di, Index_t * dj, Index_t * dk)
{
    *di = dx;
    *dj = 1;
    *dk = dz;
}

static inline int otherRank2(int rank, Index_t tp)
{
    return rank - tp;
}

/****************************** CASE 3 ************************************/

static inline Index_t node_index3(Index_t dx, Index_t dy, Index_t dz, Index_t i, Index_t j, Index_t k)
{
    return dx*(dy-1) + node_index2(dx, dy, dz, i, j, k);
}

static inline int exists3(int bottom, int top, int front, int back, int left, int right)
{
    return back;
}

static inline void dimensions3(Index_t dx, Index_t dy, Index_t dz, Index_t * di, Index_t * dj, Index_t * dk)
{
    return dimensions2(dx, dy, dz, di, dj, dk);
}

static inline int otherRank3(int rank, Index_t tp)
{
    return rank + tp;
}

/****************************** CASE 4 ************************************/

static inline Index_t node_index4(Index_t dx, Index_t dy, Index_t dz, Index_t i, Index_t j, Index_t k)
{
    return j*dx + k*dx*dy;
}

static inline int exists4(int bottom, int top, int front, int back, int left, int right)
{
    return left;
}

static inline void dimensions4(Index_t dx, Index_t dy, Index_t dz, Index_t * di, Index_t * dj, Index_t * dk)
{
    *di = 1;
    *dj = dy;
    *dk = dz;
}

static inline int otherRank4(int rank, Index_t tp)
{
    return rank - 1;
}

/****************************** CASE 5 ************************************/

static inline Index_t node_index5(Index_t dx, Index_t dy, Index_t dz, Index_t i, Index_t j, Index_t k)
{
    return (dx-1) + node_index4(dx, dy, dz, i, j, k);
}

static inline int exists5(int bottom, int top, int front, int back, int left, int right)
{
    return right;
}

static inline void dimensions5(Index_t dx, Index_t dy, Index_t dz, Index_t * di, Index_t * dj, Index_t * dk)
{
    dimensions4(dx, dy, dz, di, dj, dk);
}

static inline int otherRank5(int rank, Index_t tp)
{
    return rank + 1;
}

/****************************** CASE 6 ************************************/

static inline Index_t node_index6(Index_t dx, Index_t dy, Index_t dz, Index_t i, Index_t j, Index_t k)
{
    return i;
}

static inline int exists6(int bottom, int top, int front, int back, int left, int right)
{
    return bottom && front;
}

static inline void dimensions6(Index_t dx, Index_t dy, Index_t dz, Index_t * di, Index_t * dj, Index_t * dk)
{
    *di = dx;
    *dj = 1;
    *dk = 1;
}

static inline int otherRank6(int rank, Index_t tp)
{
    return rank - tp * tp - tp;
}

/****************************** CASE 7 ************************************/

static inline Index_t node_index7(Index_t dx, Index_t dy, Index_t dz, Index_t i, Index_t j, Index_t k)
{
    return j*dx;
}

static inline int exists7(int bottom, int top, int front, int back, int left, int right)
{
    return bottom && left;
}

static inline void dimensions7(Index_t dx, Index_t dy, Index_t dz, Index_t * di, Index_t * dj, Index_t * dk)
{
    *di = 1;
    *dj = dy;
    *dk = 1;
}

static inline int otherRank7(int rank, Index_t tp)
{
    return rank - tp * tp - 1;
}

/****************************** CASE 8 ************************************/

static inline Index_t node_index8(Index_t dx, Index_t dy, Index_t dz, Index_t i, Index_t j, Index_t k)
{
    return dx*(dy-1) + node_index6(dx, dy, dz, i, j, k);
}

static inline int exists8(int bottom, int top, int front, int back, int left, int right)
{
    return bottom && back;
}

static inline void dimensions8(Index_t dx, Index_t dy, Index_t dz, Index_t * di, Index_t * dj, Index_t * dk)
{
    dimensions6(dx, dy, dz, di, dj, dk);
}

static inline int otherRank8(int rank, Index_t tp)
{
    return rank - tp * tp + tp;
}

/****************************** CASE 9 ************************************/

static inline Index_t node_index9(Index_t dx, Index_t dy, Index_t dz, Index_t i, Index_t j, Index_t k)
{
    return (dx-1) + node_index7(dx, dy, dz, i, j, k);
}

static inline int exists9(int bottom, int top, int front, int back, int left, int right)
{
    return bottom && right;
}

static inline void dimensions9(Index_t dx, Index_t dy, Index_t dz, Index_t * di, Index_t * dj, Index_t * dk)
{
    dimensions7(dx, dy, dz, di, dj, dk);
}

static inline int otherRank9(int rank, Index_t tp)
{
    return rank - tp * tp + 1;
}

/****************************** CASE 10 ************************************/

static inline Index_t node_index10(Index_t dx, Index_t dy, Index_t dz, Index_t i, Index_t j, Index_t k)
{
    return k*dx*dy;
}

static inline int exists10(int bottom, int top, int front, int back, int left, int right)
{
    return left && front;
}

static inline void dimensions10(Index_t dx, Index_t dy, Index_t dz, Index_t * di, Index_t * dj, Index_t * dk)
{
    *di = 1;
    *dj = 1;
    *dk = dz;
}

static inline int otherRank10(int rank, Index_t tp)
{
    return rank - tp - 1;
}

/****************************** CASE 11 ************************************/

static inline Index_t node_index11(Index_t dx, Index_t dy, Index_t dz, Index_t i, Index_t j, Index_t k)
{
    return dx*(dy-1) + node_index10(dx, dy, dz, i, j, k);
}

static inline int exists11(int bottom, int top, int front, int back, int left, int right)
{
    return left && back;
}

static inline void dimensions11(Index_t dx, Index_t dy, Index_t dz, Index_t * di, Index_t * dj, Index_t * dk)
{
    dimensions10(dx, dy, dz, di, dj, dk);
}

static inline int otherRank11(int rank, Index_t tp)
{
    return rank + tp - 1;
}

/****************************** CASE 12 ************************************/

static inline Index_t node_index12(Index_t dx, Index_t dy, Index_t dz, Index_t i, Index_t j, Index_t k)
{
    return (dx-1) + node_index11(dx, dy, dz, i, j, k);
}

static inline int exists12(int bottom, int top, int front, int back, int left, int right)
{
    return right && back;
}

static inline void dimensions12(Index_t dx, Index_t dy, Index_t dz, Index_t * di, Index_t * dj, Index_t * dk)
{
    dimensions10(dx, dy, dz, di, dj, dk);
}

static inline int otherRank12(int rank, Index_t tp)
{
    return rank + tp + 1;
}

/****************************** CASE 13 ************************************/

static inline Index_t node_index13(Index_t dx, Index_t dy, Index_t dz, Index_t i, Index_t j, Index_t k)
{
    return (dx-1) + node_index10(dx, dy, dz, i, j, k);
}

static inline int exists13(int bottom, int top, int front, int back, int left, int right)
{
    return right && front;
}

static inline void dimensions13(Index_t dx, Index_t dy, Index_t dz, Index_t * di, Index_t * dj, Index_t * dk)
{
    dimensions10(dx, dy, dz, di, dj, dk);
}

static inline int otherRank13(int rank, Index_t tp)
{
    return rank - tp + 1;
}

/****************************** CASE 14 ************************************/

static inline Index_t node_index14(Index_t dx, Index_t dy, Index_t dz, Index_t i, Index_t j, Index_t k)
{
    return dx*dy*(dz-1) + i;
}

static inline int exists14(int bottom, int top, int front, int back, int left, int right)
{
    return top && front;
}

static inline void dimensions14(Index_t dx, Index_t dy, Index_t dz, Index_t * di, Index_t * dj, Index_t * dk)
{
    *di = dx;
    *dj = 1;
    *dk = 1;
}

static inline int otherRank14(int rank, Index_t tp)
{
    return rank + tp * tp - tp;
}

/****************************** CASE 15 ************************************/

static inline Index_t node_index15(Index_t dx, Index_t dy, Index_t dz, Index_t i, Index_t j, Index_t k)
{
    return dx*dy*(dz-1) + j*dx;
}

static inline int exists15(int bottom, int top, int front, int back, int left, int right)
{
    return top && left;
}

static inline void dimensions15(Index_t dx, Index_t dy, Index_t dz, Index_t * di, Index_t * dj, Index_t * dk)
{
    *di = 1;
    *dj = dy;
    *dk = 1;
}

static inline int otherRank15(int rank, Index_t tp)
{
    return rank + tp * tp - 1;
}

/****************************** CASE 16 ************************************/

static inline Index_t node_index16(Index_t dx, Index_t dy, Index_t dz, Index_t i, Index_t j, Index_t k)
{
    return dx*(dy-1) + node_index14(dx, dy, dz, i, j, k);
}

static inline int exists16(int bottom, int top, int front, int back, int left, int right)
{
    return top && back;
}

static inline void dimensions16(Index_t dx, Index_t dy, Index_t dz, Index_t * di, Index_t * dj, Index_t * dk)
{
    dimensions14(dx, dy, dz, di, dj, dk);
}

static inline int otherRank16(int rank, Index_t tp)
{
    return rank + tp * tp + tp;
}

/****************************** CASE 17 ************************************/

static inline Index_t node_index17(Index_t dx, Index_t dy, Index_t dz, Index_t i, Index_t j, Index_t k)
{
    return (dx-1) + node_index15(dx, dy, dz, i, j, k);
}

static inline int exists17(int bottom, int top, int front, int back, int left, int right)
{
    return top && right;
}

static inline void dimensions17(Index_t dx, Index_t dy, Index_t dz, Index_t * di, Index_t * dj, Index_t * dk)
{
    dimensions15(dx, dy, dz, di, dj, dk);
}

static inline int otherRank17(int rank, Index_t tp)
{
    return rank + tp * tp + 1;
}

/****************************** CASE 18 ************************************/

static inline Index_t node_index18(Index_t dx, Index_t dy, Index_t dz, Index_t i, Index_t j, Index_t k)
{
    return 0;
}

static inline int exists18(int bottom, int top, int front, int back, int left, int right)
{
    return bottom && left && front;
}

static inline void dimensions18(Index_t dx, Index_t dy, Index_t dz, Index_t * di, Index_t * dj, Index_t * dk)
{
    *di = 1;
    *dj = 1;
    *dk = 1;
}

static inline int otherRank18(int rank, Index_t tp)
{
    return rank - tp * tp - tp - 1;
}

/****************************** CASE 19 ************************************/

static inline Index_t node_index19(Index_t dx, Index_t dy, Index_t dz, Index_t i, Index_t j, Index_t k)
{
    return dx*(dy-1);
}

static inline int exists19(int bottom, int top, int front, int back, int left, int right)
{
    return bottom && left && back;
}

static inline void dimensions19(Index_t dx, Index_t dy, Index_t dz, Index_t * di, Index_t * dj, Index_t * dk)
{
    *di = 1;
    *dj = 1;
    *dk = 1;
}

static inline int otherRank19(int rank, Index_t tp)
{
    return rank - tp * tp + tp - 1;
}

/****************************** CASE 21 ************************************/

static inline Index_t node_index21(Index_t dx, Index_t dy, Index_t dz, Index_t i, Index_t j, Index_t k)
{
    return dx-1;
}

static inline int exists21(int bottom, int top, int front, int back, int left, int right)
{
    return bottom && right && front;
}

static inline void dimensions21(Index_t dx, Index_t dy, Index_t dz, Index_t * di, Index_t * dj, Index_t * dk)
{
    *di = 1;
    *dj = 1;
    *dk = 1;
}

static inline int otherRank21(int rank, Index_t tp)
{
    return rank - tp * tp - tp + 1;
}

/****************************** CASE 20 ************************************/

static inline Index_t node_index20(Index_t dx, Index_t dy, Index_t dz, Index_t i, Index_t j, Index_t k)
{
    return node_index19(dx, dy, dz, i, j, k) + node_index21(dx, dy, dz, i, j, k);
}

static inline int exists20(int bottom, int top, int front, int back, int left, int right)
{
    return bottom && right && back;
}

static inline void dimensions20(Index_t dx, Index_t dy, Index_t dz, Index_t * di, Index_t * dj, Index_t * dk)
{
    *di = 1;
    *dj = 1;
    *dk = 1;
}

static inline int otherRank20(int rank, Index_t tp)
{
    return rank - tp * tp + tp + 1;
}


/****************************** CASE 22 ************************************/

static inline Index_t node_index22(Index_t dx, Index_t dy, Index_t dz, Index_t i, Index_t j, Index_t k)
{
    return dx*dy*(dz-1);
}

static inline int exists22(int bottom, int top, int front, int back, int left, int right)
{
    return top && left && front;
}

static inline void dimensions22(Index_t dx, Index_t dy, Index_t dz, Index_t * di, Index_t * dj, Index_t * dk)
{
    *di = 1;
    *dj = 1;
    *dk = 1;
}

static inline int otherRank22(int rank, Index_t tp)
{
    return rank + tp * tp - tp - 1;
}

/****************************** CASE 23 ************************************/

static inline Index_t node_index23(Index_t dx, Index_t dy, Index_t dz, Index_t i, Index_t j, Index_t k)
{
    return node_index19(dx, dy, dz, i, j, k) + node_index22(dx, dy, dz, i, j, k);
}

static inline int exists23(int bottom, int top, int front, int back, int left, int right)
{
    return top && left && back;
}

static inline void dimensions23(Index_t dx, Index_t dy, Index_t dz, Index_t * di, Index_t * dj, Index_t * dk)
{
    *di = 1;
    *dj = 1;
    *dk = 1;
}

static inline int otherRank23(int rank, Index_t tp)
{
    return rank + tp * tp + tp - 1;
}

/****************************** CASE 24 ************************************/

static inline Index_t node_index24(Index_t dx, Index_t dy, Index_t dz, Index_t i, Index_t j, Index_t k)
{
    return node_index20(dx, dy, dz, i, j, k) + node_index22(dx, dy, dz, i, j, k);
}

static inline int exists24(int bottom, int top, int front, int back, int left, int right)
{
    return top && right && back;
}

static inline void dimensions24(Index_t dx, Index_t dy, Index_t dz, Index_t * di, Index_t * dj, Index_t * dk)
{
    *di = 1;
    *dj = 1;
    *dk = 1;
}

static inline int otherRank24(int rank, Index_t tp)
{
    return rank + tp * tp + tp + 1;
}

/****************************** CASE 25 ************************************/

static inline Index_t node_index25(Index_t dx, Index_t dy, Index_t dz, Index_t i, Index_t j, Index_t k)
{
    return node_index21(dx, dy, dz, i, j, k) + node_index22(dx, dy, dz, i, j, k);
}

static inline int exists25(int bottom, int top, int front, int back, int left, int right)
{
    return top && right && front;
}

static inline void dimensions25(Index_t dx, Index_t dy, Index_t dz, Index_t * di, Index_t * dj, Index_t * dk)
{
    *di = 1;
    *dj = 1;
    *dk = 1;
}

static inline int otherRank25(int rank, Index_t tp)
{
    return rank + tp * tp - tp + 1;
}

/**********************/
/* CREATE CASES ARRAY */
/**********************/
# define DEFINE_CASE(ID, OID) {ID, exists##ID, dimensions##ID, node_index##ID, otherRank##ID, OID}

communication_case_t COMMUNICATION_CASES[26] =
{
    DEFINE_CASE( 0,  1),
    DEFINE_CASE( 1,  0),
    DEFINE_CASE( 2,  3),
    DEFINE_CASE( 3,  2),
    DEFINE_CASE( 4,  5),
    DEFINE_CASE( 5,  4),

    DEFINE_CASE( 6, 16),
    DEFINE_CASE( 7, 17),
    DEFINE_CASE( 8, 14),
    DEFINE_CASE( 9, 15),
    DEFINE_CASE(10, 12),
    DEFINE_CASE(11, 13),
    DEFINE_CASE(12, 10),
    DEFINE_CASE(13, 11),
    DEFINE_CASE(14,  8),
    DEFINE_CASE(15,  9),
    DEFINE_CASE(16,  6),
    DEFINE_CASE(17,  7),

    DEFINE_CASE(18, 24),
    DEFINE_CASE(19, 25),
    DEFINE_CASE(20, 22),
    DEFINE_CASE(21, 23),
    DEFINE_CASE(22, 20),
    DEFINE_CASE(23, 21),
    DEFINE_CASE(24, 18),
    DEFINE_CASE(25, 19),
};

#endif
