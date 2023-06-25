# ifndef __LULESH_TRACE_H__
#  define __LULESH_TRACE_H__

# define TASK_SET_COLOR(...)
# define TASK_SET_LABEL(...)

# ifdef TRACE
#  include <mpc_omp_task_trace.h>

#  ifdef TRACE_LABEL
#   include <mpc_omp_task_label.h>
#   undef TASK_SET_LABEL
#   define TASK_SET_LABEL(...)   do {                                   \
                                MPC_OMP_TASK_SET_LABEL(__VA_ARGS__);    \
                            } while (0)
#  endif /* TRACE_LABEL */

#  ifdef TRACE_COLOR
#   undef TASK_SET_COLOR
#   define TASK_SET_COLOR(C)    do {                        \
                                    mpc_omp_task_color(C);  \
                                } while (0)
#  endif /* TRACE_COLOR */

# endif /* TRACE */

# endif /* __LULESH_TRACE_H__ */
