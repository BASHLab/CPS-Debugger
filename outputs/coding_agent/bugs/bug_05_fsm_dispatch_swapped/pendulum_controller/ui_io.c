/* global includes */
#include <stdbool.h>
#include <string.h>

/* self include */
#include "ui_io.h"

/* Signals */

void ui_machine_status_reset(ui_status_machine_t *sig) {
    sig->i_NotAus = false;
    sig->i_SchutzTuer = false;
    sig->i_ST_Ein = false;
}

void ui_pendulum_status_reset(ui_status_pendulum_t *sig) {
    sig->i_Dr_AF = false;
    sig->i_Dr_AEin = false;
    sig->i_Dr_Error = false;
}


void ui_move_status_reset(ui_status_move_t *sig) {
    sig->move_started = false;
    sig->position_reached = false;
}

void ui_motion_status_reset(ui_status_motion_t *sig) {
    sig->i_Au_Start = false;
    sig->pendulum_ok = false;
    sig->standup_ready = false;
    sig->standup_failed = false;
    sig->balance_failed = false;
}

/* Events */
