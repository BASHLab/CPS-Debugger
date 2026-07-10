/* global includes */
#include <math.h>
#include <unistd.h>

/* local includes */
#include "sys_params.h"
#include "general.h"
// #include "main.h"
// #include "main_context.h"
/* self include */
#include "lqrsim.h"

/* Debugging */
#include <stdio.h>

#include <time.h>
#include <errno.h>    

#define DELAYMODE false
#define GOODRANGEMODE true


double reference_pt (lqr_sim_params_t *params, const general_drive_out_t *gen_drive_out) {

    // Iterate through the following setpoints
    // There is hysteresis in the cart position, this should be checked. For now, hardcoded fix with different setpoints
    const int NUM_XMODES = 4;
    double xref[NUM_XMODES] = {gen_drive_out->x_ist, 0.0, 0.08, -0.12};

    // Update counters for updating setpoint
    if (params->counter_xref < 4000)
        params->counter_xref++;
    if (params->counter_xref == 250) // First stay in current position and start balancing, then move to 0
        params->xref_switch = 1;
    if (params->counter_xref == 4000) { // First switch after 4 seconds
        params->xref_switch = 2;
        params->counter_xref++;
    }

    // Change setpoint when approaching end of rail
    if ((params->xref_switch == (NUM_XMODES - 2)) && (gen_drive_out->x_ist >= 0.45 * xref[NUM_XMODES - 2]))
        params->xref_switch = NUM_XMODES - 1;
    // There is hysteresis in the cart position, this should be checked. For now, hardcoded fix with 0.8
    if (params->xref_switch == (NUM_XMODES - 1) && (gen_drive_out->x_ist <= 0.4 * xref[NUM_XMODES - 1]))
        params->xref_switch = NUM_XMODES - 2;

    // Assign setpoint
    double xsoll = xref[params->xref_switch];
    return xsoll;
}


double reference_pt_old(lqr_sim_params_t *params, const general_drive_out_t *gen_drive_out) {

    // Iterate through the following setpoints
    // There is hysteresis in the cart position, this should be checked. For now, hardcoded fix with different setpoints
    double xref[] = {0.0, 0.08, -0.12};

    // Wait 4 seconds in 0 before starting the setpoint move
    // Update counters for updating setpoint
    if (params->counter_xref < 4000)
        params->counter_xref++;
    if (params->counter_xref == 4000) { // First switch after 4 seconds
        params->xref_switch = 1;
        params->counter_xref++;
    }

    // Change setpoint when approaching end of rail
    if ((params->xref_switch == 1) && (gen_drive_out->x_ist >= 0.9 * xref[1]))
        params->xref_switch = 2;
    // There is hysteresis in the cart position, this should be checked. For now, hardcoded fix with 0.8
    if (params->xref_switch == 2 && (gen_drive_out->x_ist <= 0.8 * xref[2]))
        params->xref_switch = 1;

    // Assign setpoint
    double xsoll = xref[params->xref_switch];
    return xsoll;
}


static void compute_force(lqr_sim_params_t *params, const general_drive_out_t *gen_drive_out, const double xsoll, 
                        lqr_sim_outputs_t *lqr_sim_out) {

    double A1, A2, A3, A4, F_lqr;
    //printf("x_ist=%f x_soll=%f v_ist=%f\n", gen_drive_out->x_ist, main_out->x_soll, gen_drive_out->v_ist);
    A1 = params->K_x * (gen_drive_out->x_ist - xsoll);
    A2 = params->K_v * gen_drive_out->v_ist;
    A3 = params->K_t * gen_drive_out->theta_ist;
    A4 = params->K_td * gen_drive_out->theta_d_ist;

    // Original LQR parameters
    // 89.0 97.0 361.0 40.0

    // Switch to softer LQR parameters (together with mass change in velocity function)
    if (params->counter_goodrange >= 500) {
        A1 =  100.0 * (gen_drive_out->x_ist - xsoll);
        A2 = (fabs(gen_drive_out->v_ist) > 0.0001) ? 54.0 * gen_drive_out->v_ist : 0.0;
        A3 = (fabs(gen_drive_out->theta_ist) > 0.005) ? 274.0 * gen_drive_out->theta_ist : 0.0;
        A4 = 36.0 * gen_drive_out->theta_d_ist;
    }

    // Original LQR algorithm formula is
    // F = A1 + A2 - A3 - A4
    // Pgh Pendulum angle sign is reversed
    // (clockwise is positive, counterclockwise is negative)
    // So we have to reverse the signs of parameters A3 and A4
    F_lqr = A1 + A2 + A3 + A4;
    F_lqr = fmax (-params->F_max, F_lqr);
    F_lqr = fmin ( params->F_max, F_lqr);
       

    lqr_sim_out->LQ_F_soll =  F_lqr;
}

static void compute_velocity(lqr_sim_params_t *params, const general_drive_out_t *gen_drive_out, const double xsoll,
                         lqr_sim_outputs_t *lqr_sim_out, const logicaltime_t cycletime) {

    double cycletime_secs = cycletime * 0.001;  

    // Update old velocity value
    lqr_sim_out->SI_v_soll_old = lqr_sim_out->SI_v_soll;

    // integrate force to become a velocity value: v = u + a*t where a = F/m,  t = cycletime
    lqr_sim_out->SI_v_soll = lqr_sim_out->SI_v_soll_old + lqr_sim_out->LQ_F_soll * (cycletime_secs / (params->m_axis));

    // Count time within good range
    if ((params->counter_goodrange < 500) && GOODRANGEMODE) {
        params->counter_goodrange++;
    }
    if (params->counter_goodrange >= 500) { // Switch to softer LQR; for the Pgh pendulum we reduced the softening factor from 5.0 to 3.9 to reduce vibrations
        lqr_sim_out->SI_v_soll = lqr_sim_out->SI_v_soll_old + lqr_sim_out->LQ_F_soll * (cycletime_secs / (3.9 + params->m_axis));
    }

    // Reset counter when exceeding good range
    if ((fabs(gen_drive_out->x_ist - xsoll) > 0.05) || (fabs(gen_drive_out->theta_ist) > 0.05)) {
        //params->counter_goodrange = 0;
    }

    // Saturate velocity
    lqr_sim_out->SI_v_soll = fmin (params->v_max, lqr_sim_out->SI_v_soll);
    lqr_sim_out->SI_v_soll = fmax (-params->v_max, lqr_sim_out->SI_v_soll);
    return;
}

double pou_lqr_sim(const lqr_sim_context_t *ctx, lqr_sim_params_t *params, 
                            const general_drive_out_t *gen_drive_out, const double xsoll,
                         lqr_sim_outputs_t *lqr_sim_out) {

    double velocity = 0.0; 

    compute_force(params, gen_drive_out, xsoll, lqr_sim_out);

    compute_velocity (params, gen_drive_out, xsoll, lqr_sim_out, ctx->dt);
    velocity = lqr_sim_out->SI_v_soll;

    return velocity;
}

