#ifndef MAIN_H
#define MAIN_H

#include <stdbool.h>

// #include "axis.h"
#include "pos.h"
#include "manual.h"
#include "lqrsim.h"

#include "standup.h"

typedef enum automatic_mode_t automatic_mode_t;

// typedef general_main_output_t main_general_input_t;

typedef struct main_context_t main_context_t;

typedef struct main_axis_out_t main_axis_out_t;

typedef struct main_visu_out_t main_visu_out_t;

enum automatic_mode_t {
    AUTO_IDLE = 0,
    AUTO_STANDUP,
    AUTO_BALANCE,
    AUTO_FAILED,
};

struct main_context_t {
    const logicaltime_t dt;
    operation_mode_t prev_BA;
    enum position_mode_t pos_mode;
    enum automatic_mode_t auto_mode;
    manual_context_t man_ctx;
    manual_output_t man_out;
    standup_context_t su_ctx;
    standup_output_t standup_out;
    standup_visu_out_t su_visu_out;
    lqr_sim_context_t ls_ctx;
    lqr_sim_outputs_t ls_out;
    pos_context_t pos_ctx;
    pos_output_t pos_out;
    // axis_context_t axis_ctx;
};

#define INIT_MAIN_CONTEXT(cycletime)                                          \
    {                                                                         \
        .dt = cycletime, .pos_mode = POS_IDLE, .auto_mode = AUTO_IDLE,        \
        .man_ctx = INIT_MANUAL_CONTEXT(cycletime),                            \
        .man_out = INIT_MANUAL_OUTPUT(),                                      \
        .su_ctx = INIT_STANDUP_CONTEXT(cycletime),                            \
        .standup_out = INIT_STANDUP_POS_OUT(),                                 \
        .su_visu_out = INIT_STANDUP_VISU_OUT(),                               \
        .ls_ctx = INIT_LQR_SIM_CONTEXT(cycletime),                            \
        .ls_out = INIT_LQR_SIM_OUTPUTS(),                                     \
        .pos_ctx = INIT_POS_CONTEXT(cycletime), .pos_out = INIT_POS_OUTPUT(), \
    }

struct main_axis_out_t {
    double Dr_Vel_SW;  // drive velocity Soll Wert
};

#define INIT_MAIN_AXIS_OUT() \
    { 0 }

// TODO: Remove this completely, fjg 11.01.2023
struct main_visu_out_t {
    bool i_Au_Start;  // Start Auto ist
    // Never written anywhere in PLC application
    // bool Both;        // Both Axes in Visu
};

#define INIT_MAIN_VISU_OUT() \
    { 0 }


void pou_main(main_context_t *ctx, const main_params_t *main_params,
              const general_drive_out_t *gen_out, main_axis_out_t *ax_out,
              main_visu_out_t *visu_out, visu_in_t *visu_in);


#endif /* MAIN_H */