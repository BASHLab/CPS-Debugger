#ifndef STANDUP_RELIABLE_H
#define STANDUP_RELIABLE_H

#include <stdbool.h>

/* local includes */
#include "plclib.h"
#include "sys_params.h"
// #include "pos.h"
#include "common.h"
#include "general.h"

enum state_t { ENABLE, MOVE, DONE };

typedef struct standup_context_t standup_context_t;
typedef struct standup_output_t standup_output_t;

double pou_standup_rel(const general_drive_out_t *gen_drive_out,
                 standup_context_t *ctx, 
				 const su_sim_params_t *params, standup_output_t *standup_out,
                 ui_status_motion_t *ui_status_motion);

struct standup_context_t {
    const logicaltime_t dt;
    bool prev_enable;
    enum state_t state;
    delay_timer_t tmr;
    //logicaltime_t prev_tmrl_T;
        
    bool prev_tmr_enable;
    int index;
    bool idx8done;
    bool first_push;
};

#define INIT_STANDUP_CONTEXT(cycletime)                                     \
    {                                                                       \
        .dt = cycletime, .prev_enable = false, .state = ENABLE, .index = 0, \
        .idx8done = false, .tmr = INIT_DELAY_TIMER(cycletime), .first_push = true,               \
    }

struct standup_output_t {
    double x_soll;      // POS sw [m]
    double v_soll;       // max velocity [m/S]
    double a_soll;       // max acceleration [m/S²]
    bool su_ready;      // Standup done
    bool su_pndlum_ok;  // position of pendulum and velocity of
    //                       // axis in a good range
};



#endif /* STANDUP_RELIABLE_H */
