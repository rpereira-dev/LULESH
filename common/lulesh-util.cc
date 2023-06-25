#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdio.h>
#if USE_MPI
#include <mpi.h>
#endif
#include "lulesh.h"
#if _OPENMP
# include <omp.h>
#else
# include <sys/time.h>
#endif

/* Helper function for converting strings to ints, with error checking */
int StrToInt(const char *token, int *retVal)
{
   const char *c ;
   char *endptr ;
   const int decimal_base = 10 ;

   if (token == NULL)
      return 0 ;

   c = token ;
   *retVal = (int)strtol(c, &endptr, decimal_base) ;
   if((endptr != c) && ((*endptr == ' ') || (*endptr == '\0')))
      return 1 ;
   else
      return 0 ;
}

static void PrintCommandLineOptions(char *execname, int myRank)
{
   if (myRank == 0) {

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
      printf(" -tel            : Number of tasks for element-wise loops (MPC).\n");
      printf(" -tnl            : Number of tasks for node-wise loops (MPC).\n");
      printf(" -rpf            : Number of MPI requests for face communications.\n");
      printf(" -rpe            : Number of MPI requests for edge communications.\n");
      printf(" -persistent     : Enable persistent taskgraph between iterations.\n");
      printf(" -h              : This message\n");
      printf("\n\n");
      fflush(stdout);
   }
}

static void ParseError(const char *message, int myRank)
{
   if (myRank == 0) {
      printf("%s\n", message);
#if USE_MPI
      MPI_Abort(MPI_COMM_WORLD, -1);
#else
      exit(-1);
#endif
   }
}

void ParseCommandLineOptions(int argc, char *argv[],
                             int myRank, struct cmdLineOpts *opts)
{
   if(argc > 1) {
      int i = 1;

      while(i < argc) {
         int ok;
         /* -i <iterations> */
         if(strcmp(argv[i], "-i") == 0) {
            if (i+1 >= argc) {
               ParseError("Missing integer argument to -i", myRank);
            }
            ok = StrToInt(argv[i+1], &(opts->its));
            if(!ok) {
               ParseError("Parse Error on option -i integer value required after argument\n", myRank);
            }
            i+=2;
         }
         /* -s <size, sidelength> */
         else if(strcmp(argv[i], "-s") == 0) {
            if (i+1 >= argc) {
               ParseError("Missing integer argument to -s\n", myRank);
            }
            ok = StrToInt(argv[i+1], &(opts->nx));
            if(!ok) {
               ParseError("Parse Error on option -s integer value required after argument\n", myRank);
            }
            i+=2;
         }
	 /* -r <numregions> */
         else if (strcmp(argv[i], "-r") == 0) {
            if (i+1 >= argc) {
               ParseError("Missing integer argument to -r\n", myRank);
            }
            ok = StrToInt(argv[i+1], &(opts->numReg));
            if (!ok) {
               ParseError("Parse Error on option -r integer value required after argument\n", myRank);
            }
            i+=2;
         }
	 /* -f <numfilepieces> */
         else if (strcmp(argv[i], "-f") == 0) {
            if (i+1 >= argc) {
               ParseError("Missing integer argument to -f\n", myRank);
            }
            ok = StrToInt(argv[i+1], &(opts->numFiles));
            if (!ok) {
               ParseError("Parse Error on option -f integer value required after argument\n", myRank);
            }
            i+=2;
         }
         /* -p */
         else if (strcmp(argv[i], "-p") == 0) {
            opts->showProg = 1;
            i++;
         }
         /* -q */
         else if (strcmp(argv[i], "-q") == 0) {
            opts->quiet = 1;
            i++;
         }
         else if (strcmp(argv[i], "-b") == 0) {
            if (i+1 >= argc) {
               ParseError("Missing integer argument to -b\n", myRank);
            }
            ok = StrToInt(argv[i+1], &(opts->balance));
            if (!ok) {
               ParseError("Parse Error on option -b integer value required after argument\n", myRank);
            }
            i+=2;
         }
         else if (strcmp(argv[i], "-c") == 0) {
            if (i+1 >= argc) {
               ParseError("Missing integer argument to -c\n", myRank);
            }
            ok = StrToInt(argv[i+1], &(opts->cost));
            if (!ok) {
               ParseError("Parse Error on option -c integer value required after argument\n", myRank);
            }
            i+=2;
         }
         else if (strcmp(argv[i], "-tel") == 0) {
            if (i+1 >= argc) {
               ParseError("Missing integer argument to -tel\n", myRank);
            }
            ok = StrToInt(argv[i+1], &(opts->tel));
            if (!ok) {
               ParseError("Parse Error on option -tel integer value required after argument\n", myRank);
            }
            i+=2;
         }
         else if (strcmp(argv[i], "-tnl") == 0) {
            if (i+1 >= argc) {
               ParseError("Missing integer argument to -tnl\n", myRank);
            }
            ok = StrToInt(argv[i+1], &(opts->tnl));
            if (!ok) {
               ParseError("Parse Error on option -tn integer value required after argument\n", myRank);
            }
            i+=2;
         }
         else if (strcmp(argv[i], "-rpf") == 0) {
            if (i+1 >= argc) {
               ParseError("Missing integer argument to -rpf\n", myRank);
            }
            ok = StrToInt(argv[i+1], &(opts->rpf));
            if (!ok) {
               ParseError("Parse Error on option -rpf integer value required after argument\n", myRank);
            }
            i+=2;
         }
         else if (strcmp(argv[i], "-rpe") == 0) {
            if (i+1 >= argc) {
               ParseError("Missing integer argument to -rpe\n", myRank);
            }
            ok = StrToInt(argv[i+1], &(opts->rpe));
            if (!ok) {
               ParseError("Parse Error on option -rpe integer value required after argument\n", myRank);
            }
            i+=2;
         }

         /* -persistent */
         else if (strcmp(argv[i], "-persistent") == 0) {
            opts->persistentTasks = 1;
            i++;
         }
         /* -v */
         else if (strcmp(argv[i], "-v") == 0) {
#if VIZ_MESH
            opts->viz = 1;
#else
            ParseError("Use of -v requires compiling with -DVIZ_MESH\n", myRank);
#endif
            i++;
         }
         /* -h */
         else if (strcmp(argv[i], "-h") == 0) {
            PrintCommandLineOptions(argv[0], myRank);
#if USE_MPI
            MPI_Abort(MPI_COMM_WORLD, 0);
#else
            exit(0);
#endif
         }
         else {
            char msg[80];
            PrintCommandLineOptions(argv[0], myRank);
            sprintf(msg, "ERROR: Unknown command line argument: %s\n", argv[i]);
            ParseError(msg, myRank);
         }
      }
   }
}

/////////////////////////////////////////////////////////////////////

void VerifyAndWriteFinalOutput(Real_t elapsed_time,
                               Domain * domain,
                               Int_t nx,
                               Int_t numRanks,
                               Int_t numThreads)
{
   // GrindTime1 only takes a single domain into account, and is thus a good way to measure
   // processor speed indepdendent of MPI parallelism.
   // GrindTime2 takes into account speedups from MPI parallelism
   Real_t grindTime1 = ((elapsed_time*1e6)/domain->cycle())/(nx*nx*nx);
   // Real_t grindTime2 = ((elapsed_time*1e6)/domain->cycle())/(nx*nx*nx*numRanks);
   Real_t grindTime2 = grindTime1 / numRanks;

   Index_t ElemId = 0;
   printf("Run completed:  \n");
   printf(":   Problem size             :  %i \n",    nx);
   printf(":   MPI tasks                :  %i \n",    numRanks);
   printf(":   OpenMP threads           :  %i \n",    numThreads);
   printf(":   Iteration count          :  %i \n",    domain->cycle());
   printf(":   Final Origin Energy      : %12.6e \n", domain->e(ElemId));

   Real_t   MaxAbsDiff = Real_t(0.0);
   Real_t TotalAbsDiff = Real_t(0.0);
   Real_t   MaxRelDiff = Real_t(0.0);
   Real_t TotalRelDiff = Real_t(0.0);

   for (Index_t j=0; j<nx; ++j) {
      for (Index_t k=j+1; k<nx; ++k) {
         Real_t AbsDiff = FABS(domain->e(j*nx+k)-domain->e(k*nx+j));
         TotalAbsDiff  += AbsDiff;

         if (MaxAbsDiff <AbsDiff) MaxAbsDiff = AbsDiff;

         Real_t RelDiff = AbsDiff / domain->e(k*nx+j);
         if ( domain->e(k*nx+j) != .0 ) TotalRelDiff  += FABS(RelDiff);
         else TotalRelDiff  += FABS(AbsDiff);

         if (MaxRelDiff <RelDiff)  MaxRelDiff = RelDiff;
      }
   }

   // Quick symmetry check
   printf("Testing Plane 0 of Energy Array on rank 0:\n");
   printf(":      MaxAbsDiff            : %12.6e\n", MaxAbsDiff   );
   printf(":      TotalAbsDiff          : %12.6e\n", TotalAbsDiff );
   printf(":      MaxRelDiff            : %12.6e\n", MaxRelDiff   );
   printf(":      TotalRelDiff          : %12.6e\n", TotalRelDiff );
   printf(":   Verification (TrD<1e-09) : %s    \n", (TotalRelDiff < 1e-09)? "pass":"fail");

   // Timing information
   printf(":Elapsed time (s)            : %10.2f \n", elapsed_time);
   printf(":Grind time (us/z/c) per dom : %10.8g \n", grindTime1);
   printf(":Grind time (us/z/c) overall : %10.8g \n", grindTime2);
   printf(":FOM (z/s)                   : %10.8g \t\n", 1000.0/grindTime2); // zones per second
}

double lulesh_timer(void)
{
#if USE_MPI
    return MPI_Wtime();
#elif _OPENMP
    return omp_get_wtime();
#else
    timeval t;
    gettimeofday(&t, NULL) ;
    return (double)(1000000 * t.tv_sec + t.tv_usec) / 1000000.0;
#endif
}
