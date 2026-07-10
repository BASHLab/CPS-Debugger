#include <stdbool.h>
#include "common.h"
#include "sys_io.h"
#include "control_word.h"
#include "general.h"
#include <stdio.h>


void reset_drive(sys_drive_output_t *sys_drive_out) {
    sys_drive_out->Dr_VelSW = 0;       
}

void pou_drive_control_word(const bool STEin,  const bool drive_release,
              const uint16_t dr_status, uint16_t *dr_control_word){

    printf("STEin: %d, drive_release: %d, dr_status: %d\n", STEin, drive_release, dr_status);

    bool drive_error = TEST_BIT(dr_status, 13);

    // Bit 13: if true, drive error
    if ((TEST_BIT(dr_status, 15)) && !(TEST_BIT(dr_status, 14)) &&
        !drive_error && STEin) {
        SET_BIT(*dr_control_word, 14);
        SET_BIT(*dr_control_word, 13);
        SET_BIT(*dr_control_word, 9);
        CLEAR_BIT(*dr_control_word, 8);
    }
    // --- drive error or not ST_Ein ------------------
    if (drive_error && !STEin) {
        CLEAR_BIT(*dr_control_word, 15);
        CLEAR_BIT(*dr_control_word, 14);
        CLEAR_BIT(*dr_control_word, 13);
    } else {
        if (drive_release) {
            SET_BIT(*dr_control_word, 15);
        } else {
            CLEAR_BIT(*dr_control_word, 15);
        }
    }
    //printf("dr_control_word: %d %x\n", dr_control_word, dr_control_word);
    return;
}

