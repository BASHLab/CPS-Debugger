/* global includes */
#include <math.h>
#include <stdlib.h>

/* local includes */
#include <stdio.h>

#include "plclib.h"
#include "sys_io.h"
#include "sys_params.h"
#include "ui_io.h"
/* self include */
#include "general.h"

#define DEBUG 0

#if 0
static double calculate_theta_raw(const general_params_t *params,
                                  const double DG_Ana) {
    double theta_raw;
    // offsPend = 21757
    theta_raw = (double)(DG_Ana - params->offsPend);
    // Glättung theta
    return theta_raw;
}

static double calculate_theta(general_drive_context_t *ctx,
                              const general_params_t *params,
                              const double theta_raw) {
    // TODO: move magic constants to params
    double theta_ist;
    // theta_gl = 0.1,
    if (DEBUG) printf("Init theta_raw: %f\n", ctx->prev_theta_raw);
    ctx->prev_theta_raw = ctx->prev_theta_raw * (1.0 - params->theta_gl) +
                          theta_raw * params->theta_gl;
    if (DEBUG) printf("theta_raw: %f \n", theta_raw);
    // calibrated_PI = 2.94836,
    theta_ist =
        -(ctx->prev_theta_raw / 32768.0) * 2 * params->calibrated_PI;  // rad

    return theta_ist;
}

#endif

/*
    Pittsburgh specific function to calculate angle based on Pittsburgh pendulum encoder hardware
    The Pgh encoder is a 13-bit singleturn absolute rotary encoder.
    A full clockwise revolution (i.e. 2 pi radians) goes from 0 to 8191 and back to 0
    Encoder data sheet:
    https://www.pepperl-fuchs.com/usa/en/classid_362.htm?view=productdetails&prodid=24164
*/
static double calculate_theta_ist_axis2(general_drive_context_t *ctx, int encoder_counter_value, bool encoder_calibration) {
    int prev_encoder_value;
    int encoder_value;
    //double encoder_temp;
    double theta_n = 0.0;

    // Load previous value
    prev_encoder_value = ctx->prev_encoder_value;

    if (ctx->set_calibration == true) { //encoder_calibration must be set to true only each time a new swing up is attempted [not in every period]
        ctx->offset = PGH_ENCODER_BITMASK & encoder_counter_value;
        printf("Encoder offset at start: %d \n", ctx->offset); //offset is the value of the encoder counter when the pendulum is in the down position
        ctx->set_calibration = false;
        ctx->encoder_overflow = 0;
    }
    // Shift into calibrated space
    // Subtract offset from encoder value to shift so that the
    // down position is counter value 0 and counter 4096 is fully upright
    encoder_value = (PGH_ENCODER_BITMASK & encoder_counter_value) - ctx->offset;
    
    // If raw encoder value is less than the offset add 8192 to wrap the value around
    if (encoder_value < 0) {
        encoder_value += PGH_ENCODER_FULL_REV;
    }

    // For the angle calculation fully upright should be angle 0
    // So subtract 4096 (AKA 1*Pi) before converting from the counter value to radians
    encoder_value -= PGH_ENCODER_FULL_REV/2;
    ctx->curr_encoder_value = encoder_value;
    theta_n = (double) encoder_value*2*PGH_PI/PGH_ENCODER_FULL_REV;

    // Update previous values
    ctx->prev2_encoder_value = ctx->prev_encoder_value;
    ctx->prev_encoder_value = encoder_value;
    //printf("prev encoder val: %lld, theta_n = %lf \n", ctx->prev_encoder_value, theta_n);
    return(theta_n);
}

/*
static double calculate_theta_ist_axis2(general_drive_context_t *ctx, int encoder_counter_value, bool encoder_calibration) {
    
    //The new 16 bit encoder captures the entire 360 degree angle in 4000 steps
    //It keeps incrementing the counter value until it reaches it max-data level of 65536
    int value, selected_value, prev_encoder_value, overflow_corr;
    double theta_n = 0.0;
    //printf("Encoder counter value:%lld \n", encoder_counter_value);


    // Load previous value
    prev_encoder_value = ctx->prev_encoder_value;

    // Check for encoder overflows
    if (prev_encoder_value > 61436 && encoder_counter_value < 4000) {
        ctx->encoder_overflow++;
        //printf("Overflow ++ detected \n");
    }
    if (prev_encoder_value < 4000 && encoder_counter_value > 61536) {
        ctx->encoder_overflow--;
        //printf("Overflow -- detected \n");
    }

    if (ctx->set_calibration == true) { //encoder_calibration must be set to true only each time a new swing up is attempted [not in every period]
        ctx->offset = encoder_counter_value;
        printf("Encoder offset at start: %d \n", ctx->offset); //offset is the value of the encoder counter when the pendulum is in the upright position
        ctx->set_calibration = false;
        ctx->encoder_overflow = 0;
    }

#if 0 
    int diff_value = abs(ctx->prev_encoder_value - encoder_counter_value);

    //if (diff_value > 50)  
       //printf("Previous value %d, current value %d \n", ctx->prev_encoder_value, encoder_counter_value);

    // Filtering logic for encoder errors
    if((((ctx->prev_encoder_value - encoder_counter_value > 50) && (prev_encoder_value < 65500))
    || ((encoder_counter_value - ctx->prev_encoder_value > 50) && (encoder_counter_value < 65500))) && !ctx->encoder_error) {
       
        ctx->encoder_error = true;
        //printf("!!!!!!!!!!!!!!!!!!!!ENCODER ERROR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! \n");
        //printf("Previous value %d, current value %d \n", ctx->prev_encoder_value, encoder_counter_value);
        //encoder_counter_value = ctx->prev_encoder_value;
    }
    if (ctx->encoder_error && ctx->counting_errors < 50) // reset at second step
    {
        //printf("Previos 2 value %d, Previous value %d, current value %d \n", ctx->prev2_encoder_value, ctx->prev_encoder_value, encoder_counter_value);
        ctx->counting_errors = ctx->counting_errors + 1;

        if (ctx->counting_errors >= 50) {
            ctx->encoder_error = false;
            ctx->counting_errors = 0;
        }
            
    }

#endif

    
    overflow_corr = (ctx->encoder_overflow >= 0) ? (1536) : (-2464);

    value = (encoder_counter_value + overflow_corr * ctx->encoder_overflow - ctx->offset) % 4000;
    selected_value = (value + 4000) % 4000 - 2000;
   
   // takes value -2000 in vertical down position, 0 in upright, <0 left half and >0 right half

    ctx->curr_encoder_value = selected_value;
    theta_n =(double)  (selected_value * 2.0 * p1_general_params.calibrated_PI /4000.0);

    // Update previous values
    ctx->prev2_encoder_value = ctx->prev_encoder_value;
    ctx->prev_encoder_value = encoder_counter_value;
    //printf("prev encoder val: %lld, theta_n = %lf \n", ctx->prev_encoder_value, theta_n);
    return(theta_n);
}
*/

#if 0
static double calculate_theta_d(const general_drive_context_t *ctx,
                                const double theta_ist) {
    double theta_d_ist;
    double cycletime_secs = ctx->dt * 0.001;
    theta_d_ist = (theta_ist - ctx->prev_theta) / cycletime_secs;
    return theta_d_ist;
}
#endif 

static double calculate_theta_d_axis2(general_drive_context_t *ctx,
                                const double theta_ist, const double p2_encoder_delta_k) {
    double theta_d_ist;
    double mod_compensation = 0;
    double cycletime_secs = ctx->dt * 0.001;

    // Compensate discontinuity of derivative of the modulus
    if (theta_ist < -2 && ctx->prev_theta >= 2) {
        mod_compensation = 2 * p2_general_params.calibrated_PI;
    }
    if (theta_ist >= 2 && ctx->prev_theta < -2) {
        mod_compensation = -2 * p2_general_params.calibrated_PI;
    }

    theta_d_ist = (theta_ist - ctx->prev_theta + mod_compensation) / cycletime_secs;
    
    theta_d_ist = ctx->prev_theta_d * (1.0 - p2_encoder_delta_k) + theta_d_ist * p2_encoder_delta_k;
    ctx->prev_theta_d = theta_d_ist; 
    return theta_d_ist;
}

static double calculate_x(const int32_t Dr_PosIW) {
    return (double)((double)Dr_PosIW / (double)1E7);
}

static double calculate_v(const general_drive_context_t *ctx,
                          const general_params_t *params, const double x_ist) {
    double v_tmp;
    double v_ist;
    double cycletime_secs = ctx->dt * 0.001;
    v_tmp = (x_ist - ctx->prev_x) / cycletime_secs;
    v_ist = ctx->prev_v * (1.0 - params->v_gl) + v_tmp * params->v_gl;
    return v_ist;
}

bool get_theta_180(general_drive_context_t *ctx, const general_params_t *params,
                   const double theta_ist) {
    bool theta_180_o;
    // theta ca. 180°, theta_d ca. 0 rad/s
    // theta_180_delay = 100 * MSEC,
    // theta_180_epsilon = 0.235,
    // if really close to the 180 degree or PI radian mark, set to true
    if (fabs(theta_ist - 3.1415926) < params->theta_180_epsilon)
        theta_180_o = true;
    else
        theta_180_o = false;

    bool theta_180;
    // If the pendulum stays at 180 degrees ie. in downward position for 100ms
    // then return true
    theta_180 = turn_on_with_delay(&(ctx->ton_theta_180), theta_180_o,
                                   params->theta_180_delay);
    return theta_180;
}

bool get_theta_d_0(general_drive_context_t *ctx, const general_params_t *params,
                   const double theta_d_ist) {
    /*
    .theta_d_0_delay = 100 * MSEC,
    .theta_d_0_epsilon = 0.03,
    */
    // if delta theta is very low, set nearby to true
    bool nearby;
    if (fabs(theta_d_ist) < params->theta_d_0_epsilon)
        nearby = true;
    else
        nearby = false;

    // if pendulum stays in same position for 100 msec, then return true
    bool theta_d_0;
    theta_d_0 = turn_on_with_delay(&(ctx->ton_theta_d_0), nearby,
                                   params->theta_d_0_delay);
    return theta_d_0;
}

static void get_status_drive(uint16_t Dr_Status,
                             ui_status_pendulum_t *pendulum_status) {
    pendulum_status->i_Dr_Error = GET_BIT(Dr_Status, 13);
    pendulum_status->i_Dr_AF = GET_BIT(Dr_Status, 14);
    pendulum_status->i_Dr_AEin = GET_BIT(Dr_Status, 15);
}

bool get_status_drive_release(uint16_t Dr_Status) {
    return GET_BIT(Dr_Status, 15) && GET_BIT(Dr_Status, 14);
}

bool get_status_drive_error(uint16_t Dr_Status) {
    return GET_BIT(Dr_Status, 13);
}

/* drive release, Antriebsfreigabe */
// bool get_drive_release(general_context_t *ctx, const bool button_Dr_AF,
//                        const bool ST_Ein, const bool s_Dr_AF) {
//     /* button b_Dr_AF is a toogle button */
//     bool rising = rising_edge_trigger(&(ctx->r_trig), button_Dr_AF);
//     bool drive_release;
//     if (ST_Ein) {
//         if (rising)
//             drive_release = !s_Dr_AF;
//         else
//             drive_release = s_Dr_AF;
//     } else {
//         drive_release = false;
//     }
//     return drive_release;
// }

void pou_general_axis(
    const bool ST_Ein,
    const ui_event_enable_pendulum_t *ui_event_pendulum_enable,
    general_axis_out_t *gen_axis_out) {
    gen_axis_out->ST_Ein = ST_Ein;
    if (ST_Ein) {
        if (ui_event_pendulum_enable->present)
            gen_axis_out->s_Dr_AF = ui_event_pendulum_enable->b_Dr_AF;
        // else no change if event is not present
    } else {
        gen_axis_out->s_Dr_AF = false;
    }
}

// operation_mode_t get_operation_mode(const visu_drive_out_t *buttons,
//                                     const bool ST_Ein,
//                                     const general_drive_out_t *gen_drive_out,
//                                     const general_axis_out_t *ax_out) {
//     operation_mode_t op_mode;
//     if (ST_Ein && ax_out->s_Dr_AF && !ax_out->Dr_Error) {
//         op_mode = gen_drive_out->i_BA;
//         if (gen_drive_out->i_BA == BA_IDLE) op_mode = BA_HAND;
//         if (buttons->b_BA_Ha) op_mode = BA_HAND;
//         if (buttons->b_BA_Au) op_mode = BA_AUTO;
//     } else
//         op_mode = BA_IDLE;
//     return op_mode;
// }

// bool get_auto_start(const bool button_Au_Start, const operation_mode_t i_BA)
// {
//     // only true as long as the button_Au_Start is pressed
//     // TODO: We should change this, to be true once switched on, and false
//     when
//     // switch off (on/off switch or a toggle button
//     return (i_BA == BA_AUTO) && button_Au_Start;
// }

const logicaltime_t general_debounce_delay = 50 * MSEC;

bool pou_general_machine(general_machine_context_t *ctx, const sys_machine_input_t *const sys_machine_in,
                         sys_machine_output_t *const sys_machine_out,
                         ui_status_machine_t *ui_status_machine) {
    // debouncing PNOZ_controller_on
    // TODO: clarify the debouncing functionality, and addopt the test case

    bool ctrl_off = turn_off_with_delay(&(ctx->tof), sys_machine_in->p_P_STEin, general_debounce_delay); 
    bool ST_Ein =
       turn_on_with_delay(&(ctx->ton), ctrl_off, general_debounce_delay);
    

    /* output to PNOZ */
    // bool ST_Ein = true;
    bool fuses_ok = sys_machine_in->p_31F1OK && sys_machine_in->p_31F2OK &&
                    sys_machine_in->p_31F4OK;
    bool drives_ok = sys_machine_in->p_BTBA1A2;
    bool emergency_stop = sys_machine_in->p_P_NABet;
    sys_machine_out->p_PLC_OK = fuses_ok && drives_ok;
    sys_machine_out->p_190K1 = !emergency_stop && fuses_ok && drives_ok;
    sys_machine_out->p_190H2 = sys_machine_in->p_P_STEin;

    /* output to application */
    // Door open is handled by PNOZ and signaled to the UI
    bool door_open = !sys_machine_in->p_P_TueZu;
    ui_status_machine->i_ST_Ein = ST_Ein;
    ui_status_machine->i_NotAus = emergency_stop;
    ui_status_machine->i_SchutzTuer = door_open;
    // ST_Ein = true;
    //return sys_machine_in->p_P_STEin;
    return ST_Ein;
}


void pou_general_drive(general_drive_context_t *ctx,
                       const general_params_t *params,
                       const sys_drive_input_t *sys_drive_in,
                       ui_status_pendulum_t *ui_status_pendulum,
                       general_drive_out_t *gen_drive_out) {
    
    // ui_status_machine->i_Dr_Error =
    //     get_status_drive_error(sys_drive_in->Dr_Status);
    // ui_status_machine->i_Dr_AF =
    //     get_status_drive_release(sys_drive_in->Dr_Status);


    

   //bool encoder_calibration = true; 

    get_status_drive(sys_drive_in->Dr_Status, ui_status_pendulum);

    //double theta_raw = calculate_theta_raw(params, sys_drive_in->DG_Ana);
    //printf("Counter: %ld\n", sys_drive_in->encoder_counter);
    gen_drive_out->theta_ist = calculate_theta_ist_axis2(ctx, sys_drive_in->encoder_counter, ctx->set_calibration);
//    printf("Computed angle: %f\n", gen_drive_out->theta_ist);
    //gen_drive_out->theta_ist = calculate_theta(ctx, params, theta_raw);

    // Calculate delta theta
    gen_drive_out->theta_d_ist = calculate_theta_d_axis2(ctx, gen_drive_out->theta_ist, params->p2_encoder_delta_k);
//        calculate_theta_d(ctx, gen_drive_out->theta_ist);
    gen_drive_out->x_ist = calculate_x(sys_drive_in->Dr_PosIW);
    gen_drive_out->v_ist = calculate_v(ctx, params, gen_drive_out->x_ist);

    // check if pendulum stays at 180 degree for 100 ms
    gen_drive_out->theta_180 =
        get_theta_180(ctx, params, gen_drive_out->theta_ist);

    // check if pendulum stays at same position (delta theta is very low) for
    // 100 ms
    gen_drive_out->theta_d_0 =
        get_theta_d_0(ctx, params, gen_drive_out->theta_d_ist);

    /* save values for the next iteration */
    ctx->prev_theta = gen_drive_out->theta_ist;

    ctx->prev_x = gen_drive_out->x_ist;
    ctx->prev_v = gen_drive_out->v_ist;
}
