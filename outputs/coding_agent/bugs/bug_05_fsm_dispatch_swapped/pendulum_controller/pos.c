/* local includes */

#include <math.h>

#include "general.h"
#include "sys_params.h"

/* self include */
#include <stdio.h>

#include "common.h"
#include "pos.h"

#define DEBUG 0

static void set_pos_local(const double x_soll, const double v_max,
                          const double a_max, pos_t *pos_local) {
    pos_local->x_soll = x_soll;
    pos_local->v_max = v_max;
    pos_local->a_max = a_max;
    return;
}

static void compute_mode_dependent_params(const position_mode_t pos_mode,
                                          const main_output_t *main_out,
                                          const standup_output_t *standup_out,
                                          pos_t *pos_local) {
    
    switch (pos_mode) {
        case POS_MANUAL:
            set_pos_local(main_out->x_soll, main_out->v_max, main_out->a_max,
                          pos_local);
            break;

        case POS_STANDUP:
            if (DEBUG)
                printf("in pos. x_soll=%f v_max=%f a_max=%f\n",
                       standup_out->x_soll, standup_out->v_soll,
                       standup_out->a_soll);
            set_pos_local(standup_out->x_soll, standup_out->v_soll,
                          standup_out->a_soll, pos_local);
            break;

        default:
            set_pos_local(0, 0, 0, pos_local);
            // pos_enable = false;
            break;
    }
    return;
}

static double compute_new_vel(const logicaltime_t cycletime,
                            const double v_soll_old,
                            const pos_params_t *pos_params,
                            const pos_t *pos_local,
                            const general_drive_out_t *gen_drive_out
                            ) {
    double v_soll_tmp;
    double velocity = 0.0;
    double delta_x = pos_local->x_soll - gen_drive_out->x_ist;
    double cycletime_secs = cycletime * 0.001;

    v_soll_tmp = (delta_x + pos_params->K_v * gen_drive_out->v_ist) *
                 pos_params->factor;  // ??

    if (DEBUG)
        printf("initial v_soll_tmp:%f, vmax:%f \n", v_soll_tmp,
               pos_local->v_max);
    // ensure that the computed v_soll_tmp lies between -pos_local->vmax,
    // pos_local->vmax
    v_soll_tmp = fmin(pos_local->v_max, v_soll_tmp);
    v_soll_tmp = fmax(-pos_local->v_max, v_soll_tmp);

    // set new v_soll s.t. {-(u + a * t) < v_soll_tmp < (u + a * t)}
    // where u is v_soll_old, t = cycletime, a = pos_local->max

    double at = pos_local->a_max * cycletime_secs;
    double v_new = 0.0;
    if (v_soll_tmp > 0) {
        v_new = fabs(v_soll_old) + at;
        velocity = fmin(v_soll_tmp, v_new);
    } else {
        v_new = -fabs(v_soll_old) - at;
        velocity = fmax(v_soll_tmp, v_new);
    }

    
    

    return velocity;
}

double pou_position(const pos_params_t *pos_params,
                    const general_drive_out_t *gen_drive_out,
                    const position_mode_t pos_mode,
                    const main_output_t *main_out,
                    const standup_output_t *standup_out, pos_context_t *ctx) {
    pos_t pos_local;
    double velocity = 0.0;
    compute_mode_dependent_params(pos_mode, main_out, standup_out, &pos_local);
    
    velocity = compute_new_vel(ctx->dt, ctx->prev_v_soll, pos_params,
                    &pos_local, gen_drive_out);
    ctx->prev_v_soll = velocity;
    
    return velocity;
}
