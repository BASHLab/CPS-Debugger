#ifndef GENERAL_H
#define GENERAL_H

/* global includes */
#include <stdbool.h>
#include <stdint.h>

/* local includes */
#include "plclib.h"
#include "sys_io.h"
#include "sys_params.h"
#include "ui_io.h"

typedef struct general_machine_context_t general_machine_context_t;
typedef struct general_drive_context_t general_drive_context_t;

// typedef struct general_visu_out_t general_visu_out_t;
typedef struct general_drive_out_t general_drive_out_t;
typedef struct general_axis_out_t general_axis_out_t;

// typedef enum operation_mode_t operation_mode_t;

// bool get_drive_release(general_context_t *ctx, const bool button_Dr_AF,
//                        const bool ST_Ein, const bool s_Dr_AF);

// operation_mode_t get_operation_mode(const ui_event_enable_pendulum_t
// *buttons,
//                                     const bool ST_Ein,
//                                     const general_drive_out_t *gen_drive_out,
//                                     const general_axis_out_t *ax_out);

// bool get_auto_start(const bool button_Au_Start, const operation_mode_t i_BA);

extern const logicaltime_t general_debounce_delay;

bool pou_general_machine(general_machine_context_t *ctx, const sys_machine_input_t *const sys_out,
                         sys_machine_output_t *const sys_in,
                         ui_status_machine_t *ui_out);

void pou_general_drive(general_drive_context_t *ctx,
                       const general_params_t *params,
                       const sys_drive_input_t *sys_drive_in,
                       ui_status_pendulum_t *ui_status_pendulum,
                       general_drive_out_t *gen_drive_out);

void pou_general_axis(const bool ST_Ein,
                      const ui_event_enable_pendulum_t *pendulum_enable,
                      general_axis_out_t *axis_out);



struct general_machine_context_t {
    delay_timer_t tof;
    delay_timer_t ton;
};
#define INIT_GENERAL_MACHINE_CTX(cycletime) \
    { .tof = INIT_DELAY_TIMER(cycletime), .ton = INIT_DELAY_TIMER(cycletime) }

struct general_drive_context_t {
    const logicaltime_t dt;
    bool set_calibration;
    //double prev_theta_raw;
    double prev_theta;
    double prev_x;
    double prev_v;  // this is also strange
    double prev_theta_raw;
    double prev_theta_d;
    delay_timer_t tof;
    delay_timer_t ton;
    // edge_trigger_t
    //     r_trig;  // used for button_Dr_AF which is no longer necessary
    delay_timer_t ton_theta_180;
    delay_timer_t ton_theta_d_0;
    int prev_encoder_value;
    int prev2_encoder_value;
    int curr_encoder_value;
    int encoder_overflow;
    int offset;
    bool encoder_error;
    int counting_errors;
};

#define INIT_GENERAL_DRIVE_CONTEXT(cycletime)                      \
    {                                                              \
        .dt = cycletime,  .prev_theta = 0.0, \
        .prev_x = 0.0, .prev_v = 0.0, .set_calibration = true,                             \
        .ton_theta_180 = INIT_DELAY_TIMER(cycletime),              \
        .ton_theta_d_0 = INIT_DELAY_TIMER(cycletime), .offset = 0, .encoder_error = false,  \
        .curr_encoder_value = 0, .prev_encoder_value = 0, .prev2_encoder_value = 0,  .counting_errors = 0,  \
    }

struct general_drive_out_t {
    double x_ist;        // pos IW [m]
    double v_ist;        // vel IW [m/S]
    double theta_ist;    // theta IW [rad]
    double theta_d_ist;  // delta theta IW [rad/S]
    bool theta_180;      // theta ca. 180°
    bool theta_d_0;      // theta_d ca. 0 rad/s
};

#define INIT_GENERAL_DRIVE_OUT() \
    { 0 }

struct general_axis_out_t {
    // This is a copy of the drive input status, no need to copy it around
    uint16_t Dr_Status;  // Drive Staus word
    bool s_Dr_AF;     // Drive soll AF
    bool ST_Ein;
};

#define INIT_GENERAL_AXIS_OUT() \
    { 0 }

// Bits 0-12 are 1's because the encoder is only 13 bits but we read in 16 for
// the sake of minimal interface changes
#define PGH_ENCODER_BITMASK 0x1FFF
#define PGH_PI 3.1415926535

// The Pittsburgh encoder is 13 bits which means a full revolution is 2^13 = 8192 steps
#define PGH_ENCODER_FULL_REV 8192

#endif /* GENERAL_H */
