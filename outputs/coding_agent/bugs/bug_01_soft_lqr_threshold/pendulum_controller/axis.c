

#include <stdbool.h>

#include "common.h"
#include "lqrsim.h"
#include "sys_io.h"

/* self include */
#include "axis.h"
#include <stdio.h>


static double compute_vsoll(const position_mode_t pos_mode, const double pos_vsoll,
                     const double lqr_vsoll) {
    double v_soll = 0.0;
    if (pos_mode == POS_MANUAL || pos_mode == POS_STANDUP) {
        v_soll = pos_vsoll;
    } else {
        if (pos_mode == POS_LQR) {
            v_soll = lqr_vsoll;
        }
    }
    return v_soll;
}

static void set_drive_control_word(const bool axis_enable, const bool STEin,
                            const sys_drive_input_t *sys_drive_in,
                            const general_axis_out_t *gen_axis_out,
                            sys_drive_output_t *sys_drive_out) {
    int dr_status = sys_drive_in->Dr_Status;

    // Bit 13: if true, drive error
    if ((TEST_BIT(dr_status, 15)) && !(TEST_BIT(dr_status, 14)) &&
        !(TEST_BIT(dr_status, 13)) && STEin) {
        SET_BIT(sys_drive_out->Dr_Control, 14);
        SET_BIT(sys_drive_out->Dr_Control, 13);
        SET_BIT(sys_drive_out->Dr_Control, 9);
        CLEAR_BIT(sys_drive_out->Dr_Control, 8);
    }
    // --- drive error or not ST_Ein ------------------
    if ((TEST_BIT(dr_status, 13)) && !STEin) {
        CLEAR_BIT(sys_drive_out->Dr_Control, 15);
        CLEAR_BIT(sys_drive_out->Dr_Control, 14);
        CLEAR_BIT(sys_drive_out->Dr_Control, 13);
    } else {
        bool val = (gen_axis_out->s_Dr_AF) && (axis_enable);
        if (val == true) {
            SET_BIT(sys_drive_out->Dr_Control, 15);
        } else {
            CLEAR_BIT(sys_drive_out->Dr_Control, 15);
        }
    }
    //if(!TEST_BIT(sys_drive_out->Dr_Control, 14))
    //    printf("bit 15: %d  bit 14:%d control word:%d\n", TEST_BIT(dr_status,15), TEST_BIT(dr_status, 14), sys_drive_out->Dr_Control);
    return;
}

void sys_reset_drive(sys_drive_output_t *sys_drive_out) {
    sys_drive_out->Dr_VelSW = 0;       
}

void pou_axis(const bool axis_enable, const bool STEin, const double pos_vsoll,
              const double lqr_vsoll, const general_axis_out_t *gen_axis_out,
              const position_mode_t pos_mode,
              const sys_drive_input_t *sys_drive_in,
              sys_drive_output_t *sys_drive_out) {
    double v_soll = compute_vsoll(pos_mode, pos_vsoll, lqr_vsoll);

    // --- Write velocity to drive ---
    //  1 m/s = 1000 mm/s = 60000 mm/min, Faktor für DINT: 1000
    sys_drive_out->Dr_VelSW = v_soll * 60000 * 1000;
    set_drive_control_word(axis_enable, STEin, sys_drive_in, gen_axis_out,
                           sys_drive_out);
}