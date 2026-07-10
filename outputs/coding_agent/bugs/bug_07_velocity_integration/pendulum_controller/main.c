
/* self include */

#include "main.h"
#include <stdio.h>
#include "common.h"
#include "general.h"
#include "pos.h"

#define DEBUG 0

static void set_manual_mode_params(main_output_t *main_out,
                                   const manual_params_t *params
                                   ) {
    main_out->v_max = params->v_manual;
    main_out->a_max = params->a_manual;
    main_out->pos_mode = POS_MANUAL;
}

static void init_automatic_mode_params(main_output_t *main_out,
                                       const standup_params_t *params,
                                       enable_flags *flags) {
    main_out->v_max = 0.0;
    main_out->a_max = params->a_standup;
    flags->su_enable = true;
}

static void set_BA_Ha_params(
    main_output_t *main_out, const manual_params_t *params,
    const general_drive_out_t *gen_drive_out,
    const ui_event_move_pendulum_t *ui_event_move_pendulum,
    standup_output_t *standup_out, enable_flags *flags) {
    main_out->v_max = params->v_manual;
    main_out->a_max = params->a_manual;
    main_out->x_soll = gen_drive_out->x_ist;
    flags->su_enable = false;
   
    flags->lqr_enable = false;
    flags->pos_enable = true;
    standup_out->su_pndlum_ok = false;
    standup_out->su_ready = false;
    main_out->pos_mode = POS_MANUAL;
    if (ui_event_move_pendulum->b_Pos_Set) {
        main_out->x_soll = ui_event_move_pendulum->Pos_Soll;
    }
}

static void set_automatic_mode_params(
    main_output_t *main_out, const manual_params_t *params,
    const ui_event_move_pendulum_t *ui_event_move_pendulum,
    const standup_output_t *standup_out, enable_flags *flags) {
    if (standup_out->su_pndlum_ok == true) {  // Switching from swingup to lqr
        
        flags->lqr_enable = true;
        flags->su_enable = false;
        flags->pos_enable = false;
        main_out->pos_mode = POS_LQR;
        main_out->v_max = params->v_manual;
        main_out->a_max = params->a_manual;
    if (ui_event_move_pendulum->present) {
            main_out->x_soll = ui_event_move_pendulum->Pos_Soll;
        }

    } else {  // trying to swing up
        
        flags->lqr_enable = false;
        flags->pos_enable = false;
        flags->su_enable = true;
        main_out->pos_mode = POS_STANDUP;
        main_out->v_max = 0;
        main_out->a_max = 0;
        main_out->x_soll = 0;
    }
}




void pou_main(const general_drive_out_t *gen_drive_out,  //
              const manual_params_t *manual_params,
              const standup_params_t *su_params,  //
              standup_output_t *standup_out,      //
              ui_event_autorun_pendulum_t *pendulum_autorun,
              const ui_event_move_pendulum_t *pendulum_move,  //
              enable_flags *flags,  //
              main_output_t *main_out) {
    if (pendulum_autorun->present) {  // new event arrived
        main_out->x_soll = gen_drive_out->x_ist;
        if (pendulum_autorun->b_Au_Start) {
            standup_out->su_pndlum_ok = false;
            init_automatic_mode_params(main_out, su_params, flags);
            pendulum_autorun->present = false; 
        } else {  // not b_Au_Start == fl_Ha
            set_manual_mode_params(main_out, manual_params);
        }
    }

    if (pendulum_autorun->b_Au_Start) {  // pendulum is already trying to swing up BA Auto
        set_automatic_mode_params(main_out, manual_params, 
                                  pendulum_move, standup_out, flags);

    } else {  // --- BA_Hand ---
        set_BA_Ha_params(main_out, manual_params, gen_drive_out, pendulum_move,
                         standup_out, flags);
    }
 
}