# Common code
COMMON_CFLAGS=-Wall -Wextra -I common/ -O3 -g
COMMON_CFLAGS+=-Wno-unused-parameter -Wno-reserved-user-defined-literal
#COMMON_CFLAGS+=-fsanitize=thread
#COMMON_CFLAGS+=-DDRYRUN
#COMMON_CFLAGS+=-DTRACE=1
#COMMON_CFLAGS+=-DTRACE_COLOR=1
#COMMON_CFLAGS+=-DTRACE_LABEL=1
#COMMON_CFLAGS+=-Wno-unused-variable
#COMMON_CFLAGS+=-Wno-comment -Wno-pointer-arith
#COMMON_CFLAGS+=-Wno-cast-function-type
#COMMON_CFLAGS+=-Wno-literal-suffix
#COMMON_CFLAGS+=-Wno-reserved-user-defined-literal -Wno-bad-function-cast

COMMON_LDFLAGS=
#COMMON_LDFLAGS+=-L/ccc/work/cont001/ocre/pereirar/these/tools/mpc/task/mpi/ -lmpcmpitrace
#COMMON_LDFLAGS+=-lstdc++
#COMMON_LDFLAGS+=-L $(POT)/lib -lpot_ompt

COMMON_SRC=common/lulesh-init.cc common/lulesh-util.cc common/lulesh-viz.cc common/lulesh-comm-cases.cc

help:
	@echo "usage: make [seq|omp-for|omp-for-mpi|omp-task|mpc|mpc-mpi|mpc-mpi-ext"

##################################
# -------- Sequential  --------- #
##################################
SEQ_CXX=g++
SEQ_CFLAGS=-DUSE_MPI=0 -DUSE_MPIX=0  -DUSE_MPC=0
SEQ_LDFLAGS=
SEQ_SRC=$(COMMON_SRC) omp/lulesh-for.cc
SEQ_OBJ=$(SEQ_SRC:.cc=.seq.o)
SEQ_TARGET=lulesh-seq

%.seq.o: %.cc
	$(SEQ_CXX) $(COMMON_CFLAGS) $(SEQ_CFLAGS) -o $@ -c $<

$(SEQ_TARGET): $(SEQ_OBJ)
	$(SEQ_CXX) $(COMMON_CFLAGS) $(SEQ_CFLAGS) $(SEQ_OBJ) $(COMMON_LDFLAGS) $(SEQ_LDFLAGS) -o $(SEQ_TARGET)

seq: $(SEQ_TARGET)

##################################
# -------- OMP FOR ------------- #
##################################
#OMP_CXX=mpc_cxx
OMP_CXX=clang++
OMP_FOR_CFLAGS=-DUSE_MPI=0 -DUSE_MPIX=0 -DUSE_MPC=0
OMP_FOR_CFLAGS+=-fopenmp
#OMP_FOR_CFLAGS+=-cc=clang++
#OMP_FOR_CFLAGS+=-fno-mpc-privatize
OMP_FOR_LDFLAGS=
OMP_FOR_SRC=$(COMMON_SRC) omp/lulesh-for.cc
OMP_FOR_OBJ=$(OMP_FOR_SRC:.cc=.omp.for.$(OMP_CXX).o)
OMP_FOR_TARGET=lulesh-omp-for-$(OMP_CXX)

#OMP_FOR_CFLAGS+=-DUSE_CALI=1
#OMP_FOR_CFLAGS+=-I$(CALI_ROOT)/include
#OMP_FOR_LDFLAGS+=-L$(CALI_ROOT)/lib64 -lcaliper

%.omp.for.$(OMP_CXX).o: %.cc
	$(OMP_CXX) $(COMMON_CFLAGS) $(OMP_FOR_CFLAGS) -o $@ -c $<

$(OMP_FOR_TARGET): $(OMP_FOR_OBJ)
	$(OMP_CXX) $(COMMON_CFLAGS) $(OMP_FOR_CFLAGS) $(OMP_FOR_OBJ) $(COMMON_LDFLAGS) $(OMP_FOR_LDFLAGS) -o $(OMP_FOR_TARGET)

omp-for: $(OMP_FOR_TARGET)

##################################
# --------- OMP + MPI ---------- #
##################################
OMP_FOR_MPI_MPICXX=I_MPI_CXX="$(OMP_CXX)" MPICH_CXX="$(OMP_CXX)" OMPI_CXX="$(OMP_CXX)" mpicxx
OMP_FOR_MPI_CFLAGS=-DUSE_MPI=1 -DUSE_MPIX=0 -DUSE_MPC=0
OMP_FOR_MPI_CFLAGS+=-fopenmp
#OMP_FOR_MPI_CFLAGS+=-fno-mpc-privatize
OMP_FOR_MPI_LDFLAGS=

OMP_FOR_MPI_CFLAGS+=-DUSE_CALI=1
OMP_FOR_MPI_CFLAGS+=-I$(CALI_ROOT)/include
OMP_FOR_MPI_LDFLAGS+=-L$(CALI_ROOT)/lib64 -lcaliper

# parallel for
OMP_FOR_MPI_FOR_TARGET=lulesh-omp-mpi-for-$(OMP_CXX)
OMP_FOR_MPI_FOR_SRC=$(OMP_FOR_SRC) omp/lulesh-comm.cc
OMP_FOR_MPI_FOR_OBJ=$(OMP_FOR_MPI_FOR_SRC:.cc=.omp.mpi.$(OMP_CXX).o)

%.omp.mpi.$(OMP_CXX).o: %.cc
	$(OMP_FOR_MPI_MPICXX) $(COMMON_CFLAGS) $(OMP_FOR_MPI_CFLAGS) -o $@ -c $<

$(OMP_FOR_MPI_FOR_TARGET): $(OMP_FOR_MPI_FOR_OBJ)
	$(OMP_FOR_MPI_MPICXX) $(COMMON_CFLAGS) $(OMP_FOR_MPI_CFLAGS) $(OMP_FOR_MPI_FOR_OBJ) $(COMMON_LDFLAGS) $(OMP_FOR_MPI_LDFLAGS) -o $(OMP_FOR_MPI_FOR_TARGET)

# task
OMP_FOR_MPI_TASK_TARGET=lulesh-omp-mpi-task
OMP_FOR_MPI_TASK_SRC=$(COMMON_SRC) omp/lulesh-t.cc common/lulesh-comm-task.cc
OMP_FOR_MPI_TASK_OBJ=$(OMP_FOR_MPI_TASK_SRC:.cc=.omp.mpi.o)

$(OMP_FOR_MPI_TASK_TARGET): $(OMP_FOR_MPI_TASK_OBJ)
	$(OMP_FOR_MPI_MPICXX) $(COMMON_CFLAGS) $(OMP_FOR_MPI_CFLAGS) $(OMP_FOR_MPI_TASK_OBJ) $(COMMON_LDFLAGS) $(OMP_FOR_MPI_LDFLAGS) -o $(OMP_FOR_MPI_TASK_TARGET)

# targets
omp-for-mpi: $(OMP_FOR_MPI_FOR_TARGET)
#$(OMP_FOR_MPI_TASK_TARGET)

##################################
# ---------- OMP TASK ---------- #
##################################
#OMP_T_CXX=mpc_cxx -cc=g++
#OMP_T_CXX=g++
OMP_T_CXX=clang++
#OMP_T_CXX=mpc_cxx -cc=clang++

OMP_T_CFLAGS=-DUSE_MPI=0 -DUSE_MPIX=0 -DUSE_MPC=0 -fopenmp
OMP_T_LDFLAGS=

%.omp-task.o: %.cc
	$(OMP_T_CXX) $(COMMON_CFLAGS) $(OMP_T_CFLAGS) -o $@ -c $<

# tasks
OMP_T_TARGET=lulesh-omp-t-$(OMP_T_CXX)
OMP_T_SRC=$(COMMON_SRC) omp-task/lulesh.cc
OMP_T_OBJ=$(OMP_T_SRC:.cc=.omp-task.o)

$(OMP_T_TARGET): $(OMP_T_OBJ)
	$(OMP_T_CXX) $(COMMON_CFLAGS) $(OMP_T_CFLAGS) $(OMP_T_OBJ) $(COMMON_LDFLAGS) $(OMP_T_LDFLAGS) -o $(OMP_T_TARGET)

omp-task: $(OMP_T_TARGET)

##################################
# ------------ MPC ------------- #
##################################
MPC_CXX=mpc_cxx -cc=clang++

MPC_CFLAGS=-DUSE_MPI=0 -DUSE_MPIX=0 -DUSE_MPC=1 -fopenmp
#MPC_CFLAGS+=-stdlib=libc++ -Wl,rpath,/ccc/work/cont001/sanl_ipc/sanl_ipc/install/llvm/13.x/lib
MPC_LDFLAGS=
MPC_SRC=$(COMMON_SRC)

%.mpc.o: %.cc
	$(MPC_CXX) $(COMMON_CFLAGS) $(MPC_CFLAGS) -o $@ -c $<

# tasks
MPC_T_TARGET=lulesh-mpc-t
MPC_T_SRC=$(MPC_SRC) mpc/lulesh.cc
MPC_T_OBJ=$(MPC_T_SRC:.cc=.mpc.o)

$(MPC_T_TARGET): $(MPC_T_OBJ)
	$(MPC_CXX) $(COMMON_CFLAGS) $(MPC_CFLAGS) $(MPC_T_OBJ) $(COMMON_LDFLAGS) $(MPC_LDFLAGS) -o $(MPC_T_TARGET)

mpc: $(MPC_T_TARGET)

##################################
# ---------- MPC-MPI ----------- #
##################################
MPC_MPI_MPICXX=mpc_cxx -cc=g++

MPC_MPI_CFLAGS=-DUSE_MPI=1 -DUSE_MPIX=1 -DUSE_MPC=1 -DUSE_OMPI=0
MPC_MPI_CFLAGS+=-fopenmp
MPC_MPI_LDFLAGS=
MPC_MPI_SRC=$(COMMON_SRC) mpc/lulesh-comm.cc

%.mpc.mpi.o: %.cc
	$(MPC_MPI_MPICXX) $(COMMON_CFLAGS) $(MPC_MPI_CFLAGS) -o $@ -c $<

# tasks
MPC_MPI_T_TARGET=lulesh-mpc-mpi-t-trace
MPC_MPI_T_SRC=$(MPC_MPI_SRC) mpc/lulesh.cc
MPC_MPI_T_OBJ=$(MPC_MPI_T_SRC:.cc=.mpc.mpi.o)

$(MPC_MPI_T_TARGET): $(MPC_MPI_T_OBJ)
	$(MPC_MPI_MPICXX) $(COMMON_CFLAGS) $(MPC_MPI_CFLAGS) $(MPC_MPI_T_OBJ) $(COMMON_LDFLAGS) $(MPC_MPI_LDFLAGS) -o $(MPC_MPI_T_TARGET)

MPC_MPI_TARGET=$(MPC_MPI_T_TARGET)
MPC_MPI_OBJ=$(MPC_MPI_T_OBJ)

mpc-mpi: $(MPC_MPI_TARGET)

##################################
# --------- MPC-OMPI ----------- #
##################################
MPC_MPI_EXT_CXX="mpc_cxx -cc=g++"
MPC_MPI_EXT_MPICXX=I_MPI_CXX=$(MPC_MPI_EXT_CXX) MPICH_CXX=$(MPC_MPI_EXT_CXX) OMPI_CXX=$(MPC_MPI_EXT_CXX) mpicxx

MPC_MPI_EXT_CFLAGS=-DUSE_MPI=1 -DUSE_MPIX=1 -DUSE_MPC=1 -DUSE_OMPI=1
MPC_MPI_EXT_CFLAGS+=-fopenmp
MPC_MPI_EXT_LDFLAGS=
MPC_MPI_EXT_SRC=$(MPC_MPI_SRC)

%.mpc.mpi-ext.o: %.cc
	$(MPC_MPI_EXT_MPICXX) $(COMMON_CFLAGS) $(MPC_MPI_EXT_CFLAGS) -o $@ -c $<

# tasks
#MPC_MPI_EXT_T_TARGET=lulesh-mpc-mpi-ext-t-persistent#-trace
MPC_MPI_EXT_T_TARGET=lulesh-mpc-mpi-ext-t-v3
MPC_MPI_EXT_T_SRC=$(MPC_MPI_SRC) mpc/lulesh.cc
MPC_MPI_EXT_T_OBJ=$(MPC_MPI_EXT_T_SRC:.cc=.mpc.mpi-ext.o)

$(MPC_MPI_EXT_T_TARGET): $(MPC_MPI_EXT_T_OBJ)
	$(MPC_MPI_EXT_MPICXX) $(COMMON_CFLAGS) $(MPC_MPI_EXT_CFLAGS) $(MPC_MPI_EXT_T_OBJ) $(COMMON_LDFLAGS) $(MPC_MPI_EXT_LDFLAGS) -o $(MPC_MPI_EXT_T_TARGET)

mpc-mpi-ext: $(MPC_MPI_EXT_T_TARGET)

###################################
# ------------ utils ------------ #
###################################
all: $(OMP_T_TARGET) $(OMP_FOR_TARGET) $(OMP_FOR_MPI_FOR_TARGET) $(OMP_FOR_MPI_TASK_TARGET) $(MPC_T_TARGET) $(MPC_MPI_EXT_T_TARGET)

clean:
	rm -rf $(SEQ_OBJ) $(OMP_T_OBJ) $(OMP_FOR_OBJ) $(OMP_FOR_MPI_FOR_OBJ) $(OMP_FOR_MPI_TASK_TARGET) $(MPC_T_OBJ) $(MPC_TR_OBJ) $(MPC_MPI_OBJ) $(MPC_MPI_EXT_T_OBJ)

dclean:
	rm -rf *.yaml *.btr *.o *.json mcxx_*.cpp traces*

fclean: clean dclean
	rm -rf $(SEQ_TARGET) $(OMP_T_TARGET) $(OMP_FOR_TARGET) $(OMP_FOR_MPI_FOR_TARGET) $(OMP_FOR_MPI_TASK_TARGET) $(MPC_T_TARGET) $(MPC_MPI_TARGET) $(MPC_MPI_EXT_T_TARGET)

re: fclean all
