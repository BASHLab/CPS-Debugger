
/* local includes */

#include "manual.h"
#include "pos.h"
#include "standup.h"
#include "sys_params.h"
#include "lqrsim.h"
#include "plclib.h"


/* self include */
#include "main_classic.h"

/* for debugging */
#include <stddef.h>








void pou_main(main_context_t *ctx, const main_params_t *params,
              const general_drive_out_t *gen_out, main_axis_out_t *ax_out,
              main_visu_out_t *visu_out) {
    if (BA_IDLE == gen_out->i_BA) {
        // pou_position_reset()
    } else {
        position_mode_t pos_mode = POS_IDLE;
        if (BA_HAND == gen_out->i_BA) {
            if (ctx->prev_BA != BA_HAND) {
                ctx->man_out.v_max = params->man_params->v_manual;
                ctx->man_out.a_max = params->man_params->a_manual;
                ctx->man_out.x_soll = gen_out->x_ist;
            }
            pou_manual(&(ctx->man_ctx), &(ctx->man_out));
            pos_mode = POS_MANUAL;
        } else if (BA_AUTO == gen_out->i_BA) {
            if (ctx->prev_BA != BA_AUTO) {
                ctx->standup_out.v_soll = params->su_params->v_standup;
                ctx->standup_out.a_soll = params->su_params->a_standup;
                ctx->standup_out.x_soll = gen_out->x_ist;
            }
            const automatic_mode_t auto_mode = ctx->auto_mode;
            switch (auto_mode) {
                case AUTO_IDLE: {
                    pou_standup_reset(&(ctx->su_ctx));
                    pou_lqr_sim_reset(&(ctx->ls_out));
                    if (gen_out->i_Au_Start) pos_mode = POS_IDLE;
                    break;
                }
                case AUTO_STANDUP: {
                    standup_result_t res =
                        pou_standup(&(ctx->su_ctx), params->su_moves,
                                    &(ctx->standup_out), &(ctx->su_visu_out));
                    switch (res) {
                        case SU_INPROGRESS:  // no change
                            break;
                        case SU_SUCCESS:
                            ctx->auto_mode = AUTO_BALANCE;
                            break;
                        case SU_FAIL:
                            ctx->auto_mode = AUTO_FAILED;
                            break;
                    }
                    pos_mode = POS_STANDUP;
                    break;
                }
                case AUTO_BALANCE: {
                    lqr_sim_result_t res = pou_lqr_sim(&(ctx->ls_ctx),
                        params->ls_params, gen_out, &(ctx->man_out),  &(ctx->ls_out));
                    switch (res) {
                        case LQR_SIM_INPROGRESS:  // no change
                            break;
                        case LQR_SIM_FAILED:  // currently not handled
                            break;
                    }
                    pos_mode = POS_LQR;
                    break;
                }
                case AUTO_FAILED: {
                    pos_mode = POS_IDLE;
                    break;
                }
            }
        }
        pou_position(&(ctx->pos_ctx), params->pos_params, gen_out, pos_mode,
                     &(ctx->man_out), &(ctx->standup_out), &(ctx->pos_out));

        
        
    
    } 

    ctx->prev_BA = gen_out->i_BA;
}
