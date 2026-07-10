/* global includes */
#include <stdbool.h>
/* local includes */
#include "axis.h"
#include "common.h"
#include "general.h"
#include "lqrsim.h"
#include "main.h"
#include "plclib.h"
#include "pos.h"
#include "standup_reliable.h"
#include "sys_params.h"
#include "ui_io.h"
// #include "sys_io.h"

/* debug */
#include <stdio.h>

const logicaltime_t cycletime = 2 * MSEC;
/* state data structures for plc functions */
//Data structures have changed a lot since this version */
#if 0
main_context_t main_ctx_p1 = INIT_MAIN_CONTEXT(cycletime);
main_context_t main_ctx_p2 = INIT_MAIN_CONTEXT(cycletime);
general_machine_context_t gen_machine_ctx_p1 =
    INIT_GENERAL_MACHINE_CTX(cycletime);
general_drive_context_t gen_drive_ctx_p1 =
    INIT_GENERAL_DRIVE_CONTEXT(cycletime);

int main(void) {
    discrete_time_t clock = INIT_DISCRETE_TIME(cycletime);

    main_ctx_p1.auto_mode = AUTO_STANDUP;
    main_ctx_p2.auto_mode = AUTO_STANDUP;

    general_drive_out_t ge_in_p1;
    general_axis_out_t gen_axout_p1;

    sys_drive_output_t p1_drive_output;
    sys_drive_input_t p1_drive_input;

    main_output_t main_out_p1;
    enable_flags flags_p1;

    standup_out_t su_out_p1;

    ui_event_move_pendulum_t p1_move;
    ui_event_autorun_pendulum_t p1_autorun;
    ui_status_motion_t p1_motion_status;

    pos_output_t pos_out_p1;

    lqr_sim_outputs_t lqr_out_p1;

    sys_machine_input_t sys_in_p1;
    sys_machine_output_t sys_out_p1;

    printf("The acceleration is %f\n", p1_standup_moves[1].acc);
    standup_visu_out_t vout_p1;
    position_mode_t pos_mode;

    ui_event_enable_pendulum_t p1_enable;
    ui_status_machine_t machine_status;
    ui_status_pendulum_t pendulum_status;

    for (int i = 1; i <= 16; i++) {
        printf("loop: %d\n", i);

        if (2 == i) {
            // trigger event
            p1_autorun.b_Au_Start = true;
            p1_autorun.present = true;
        }

        /* io: general */

        bool ST_Ein = pou_general_machine(&sys_in_p1,
                                          &sys_out_p1, &machine_status);
        pou_general_axis(ST_Ein, &p1_enable, &gen_axout_p1);
        pou_general_drive(&gen_drive_ctx_p1, &p1_general_params,
                          &p1_drive_input, &pendulum_status, &ge_in_p1);

        /* control: */
        pou_main(&main_ctx_p1, &ge_in_p1, &p1_manual_params, &p1_standup_params,
                 &su_out_p1, &p1_autorun, &p1_move, &p1_motion_status,
                 &flags_p1, &main_out_p1);

        pou_standup(&main_ctx_p1.su_ctx, p1_standup_moves, &su_out_p1, &vout_p1,
                    &ge_in_p1, &pos_out_p1, flags_p1.su_enable);
        pou_position(flags_p1.pos_enable, &main_ctx_p1.pos_ctx, &p1_pos_params,
                     &ge_in_p1, pos_mode, &main_out_p1, &su_out_p1,
                     &pos_out_p1);
        pou_lqr_sim(flags_p1.lqr_enable, flags_p1.sim_enable,
                    &main_ctx_p1.ls_ctx, &p1_lqr_sim_params, &ge_in_p1,
                    &main_out_p1, &lqr_out_p1);
        pos_mode = main_out_p1.pos_mode;

        /* io: axis */
        pou_axis(flags_p1.axis_enable, ST_Ein, pos_out_p1.v_soll,
                 lqr_out_p1.SI_v_soll, &gen_axout_p1, pos_mode, &p1_drive_input,
                 &p1_drive_output);

        // reset events and update statuses (not needed here)
        // See demo_io.c how to handle events and statuses after each iteration

        p1_autorun.present = false;
        p1_move.present = false;
        p1_enable.present = false;

        tick(&clock);
    }
    return 0;

}
#endif
