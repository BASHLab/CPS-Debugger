#ifndef LQR_SIM_H
#define LQR_SIM_H

/* local includes */
#include "general.h"
#include "plclib.h"
#include "pos.h"
#include "sys_params.h"

typedef struct lqr_sim_context_t lqr_sim_context_t;
typedef struct lqr_sim_outputs_t lqr_sim_outputs_t;
typedef enum lqr_sim_result_t lqr_sim_result_t;

double reference_pt(lqr_sim_params_t *params, const general_drive_out_t *gen_drive_out);

double pou_lqr_sim(const lqr_sim_context_t *ctx,
                             lqr_sim_params_t *params,
                             const general_drive_out_t *gen_drive_out,
                             double xsoll,
                             lqr_sim_outputs_t *lqr_sim_out);


struct lqr_sim_context_t {
    const logicaltime_t dt;
};

#define INIT_LQR_SIM_CONTEXT(cycletime) \
    { .dt = cycletime }

struct lqr_sim_outputs_t {
    double LQ_F_soll;  // Force SW [N]
    double SI_v_soll;  // Velocity SW [m/S]
    double SI_v_soll_old;
};

#define INIT_LQR_SIM_OUTPUT() \
    { .LQ_F_soll = 0.0, .SI_v_soll = 0.0, .SI_v_soll_old = 0.0 }



#endif /* LQR_SIM_H */