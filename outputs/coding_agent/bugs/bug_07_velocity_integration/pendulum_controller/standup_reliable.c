/* global includes */
#include <math.h>

/* local includes */
#include "common.h"
#include "general.h"
#include "lqrsim.h"
#include "plclib.h"
#include "sys_params.h"

/* self include */
#include "standup_reliable.h"

/* Debugging */
#include <stdio.h>

#define DEBUG 0

double sign_double(double x) {
    if (x >= 0.0) return 1.0;
    if (x < 0.0) return -1.0;
    return 0.0;  // This should never happen
}

static void compute_velocity_su(const su_sim_params_t *params,
                         standup_output_t *standup_out,
                         const logicaltime_t cycletime) {
    double cycletime_secs = cycletime * 0.001;
    // integrate force to become a velocity value: v = u + a*t where a = cmd_su,
    // t = cycletime
    standup_out->v_soll =
        standup_out->v_soll + standup_out->a_soll * cycletime_secs;
    standup_out->v_soll = fmin(params->v_max, standup_out->v_soll);
    standup_out->v_soll = fmax(-params->v_max, standup_out->v_soll);
    return;
}

static void compute_acceleration_su(const general_drive_out_t *gen_drive_out,
                             const su_sim_params_t *params,
                             standup_output_t *standup_out
                            ) {
    double theta, theta_d, x, safety_xtol;
    double v, safety_vtol;

    safety_xtol = 0.95;  // percentage of max_xdisplay for asymptote of
                         // restricted cart length control
    safety_vtol =
        0.95;  // percentage of max velocity for velocity well asymptote

    theta = gen_drive_out->theta_ist;
    theta_d = gen_drive_out->theta_d_ist;
    x = gen_drive_out->x_ist;  // x=0 in the middle
    v = gen_drive_out->v_ist;

    // Remove jittering at small velicities by putting it to 0
    double theta_d_zero = (fabs(theta_d) < 0.001) ? 0.0 : 1.0;
    double v_zero = (fabs(v) < 0.001) ? 0.0 : 1.0;

    double ext_costheta = -1.0; //(fabs(theta) > ( p1_general_params.calibrated_PI / 20) ) ? -1.0 : 1.0;

    double sign_thetadotcostheta = sign_double(theta_d * cos(theta)) * theta_d_zero;
    //double sign_thetadotcostheta = sign_double(theta_d * ext_costheta) * theta_d_zero;
    double sign_x = sign_double(x);
    double sign_v = sign_double(v) * v_zero;

    // Control command for pure energy control
    double cmd_su_acc = params->K_su * sign_thetadotcostheta;

    // Control command for restricted cart length
    double frac_relpos = fabs(x) / (safety_xtol * params->max_xdispl);
    double minvalue_pos = (0.9999 < frac_relpos) ? 0.9999 : frac_relpos;
    double cmd_cartwell = params->K_cw * sign_x * log(1 - minvalue_pos);

    // Control command for velocity well
    double frac_relvel = fabs(v) / (safety_vtol * params->v_max);
    double minvalue_vel = (0.9999 < frac_relvel) ? 0.9999 : frac_relvel;
    double cmd_velwell = params->K_vw * sign_v * log(1 - minvalue_vel);

    // Combine to find cmd acceleration
    double cmd_su = cmd_su_acc + cmd_cartwell + cmd_velwell;

    standup_out->a_soll = cmd_su;

    return;
}

double pou_standup_rel(const general_drive_out_t *gen_drive_out,
                     standup_context_t *ctx,
                     const su_sim_params_t *params,
                     standup_output_t *standup_out,
                     ui_status_motion_t *ui_status_motion) {
    // Compute acceleration command
    compute_acceleration_su(gen_drive_out, params, standup_out);

    // Compute cart velocity
    compute_velocity_su(params, standup_out, ctx->dt);

    double velocity = 0.0; 

    // First gentle push
    if (ctx->first_push){
        velocity = 1.0;
        ctx->first_push = false;
    }
    else {
        velocity = standup_out->v_soll;
    }

    // if (fabs(gen_drive_out->theta_ist < 0.5))
    //	printf("Theta_ist %f\n", gen_drive_out->theta_ist);
    
    if (fabs(gen_drive_out->x_ist) <= 0.8 * params->max_xdispl) { // Activate switch only if far from end of rail

        if ((fabs(gen_drive_out->theta_ist) < 0.07) && (fabs(gen_drive_out->theta_d_ist) < 0.05)) {
           // printf("Angle in interval, setting flags su_pendulum_ok --new \n");
            standup_out->su_pndlum_ok = true;
            ui_status_motion->standup_ready = true;
            // printf("Yay! Standup done!%lf\n", gen_drive_out->theta_ist);
        }

    }

    // if(standup_out->su_ready) printf("ready\n");
    ctx->prev_enable = true;
    return velocity;
}
