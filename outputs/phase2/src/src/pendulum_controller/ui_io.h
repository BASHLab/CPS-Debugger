#ifndef UI_H
#define UI_H

#include <stdbool.h>

#include "sys_io.h"

/*
    UI attach
*/

typedef struct ui_event_attach_t ui_event_attach_t;

/*
    UI status information for the machine
*/

typedef struct ui_status_machine_t ui_status_machine_t;

/*
    UI control events and status information for pendulums
*/

typedef struct ui_status_pendulum_t ui_status_pendulum_t;
typedef struct ui_event_enable_pendulum_t ui_event_enable_pendulum_t;

/*
    UI control events and status for motions
*/

typedef struct ui_status_move_t ui_status_move_t;
typedef struct ui_status_motion_t ui_status_motion_t;


typedef struct ui_event_move_pendulum_t ui_event_move_pendulum_t;
typedef struct ui_event_autorun_pendulum_t ui_event_autorun_pendulum_t;

/*
    Struct definitions
*/

struct ui_status_machine_t {
    bool i_NotAus;
    bool i_SchutzTuer;
    bool i_ST_Ein;  // Steuerung Ein
    bool _prev_i_NotAus;
    bool _prev_i_SchutzTuer;
    bool _prev_i_ST_Ein;
};

#define INIT_UI_STATUS_MACHINE() \
    { 0 }

void ui_machine_status_reset(ui_status_machine_t *sig);


struct ui_status_pendulum_t {
    const pendulum_id_t pid;
    bool i_Dr_AF;     // Drive Antriebsfreigabe
    bool i_Dr_AEin;   // Drive Antrieb-Ein
    bool i_Dr_Error;  // Drive error
    bool _prev_i_Dr_AF;
    bool _prev_i_Dr_AEin;
    bool _prev_i_Dr_Error;
};

#define INIT_UI_STATUS_PENDULUM(pendulum_id) \
    { .pid = pendulum_id }

void ui_pendulum_status_reset(ui_status_pendulum_t *sig);


struct ui_status_move_t {
    const pendulum_id_t pid;
    bool move_started;
    bool position_reached;
    bool _prev_move_started;
    bool _prev_position_reached;
};

#define INIT_UI_STATUS_MOVE(pendulum_id) \
    { .pid = pendulum_id }

void ui_move_status_reset(ui_status_move_t *sig);


struct ui_status_motion_t {
    const pendulum_id_t pid;
    bool i_Au_Start;      // automatic mode started
    bool pendulum_ok;     // in desired position, hanging without movement
    bool standup_ready;   // standup successfully finished
    bool standup_failed;  // standup not successful
    bool balance_failed;  // balancing not sucessful
    bool _prev_i_Au_Start;
    bool _prev_pendulum_ok;
    bool _prev_standup_ready;
    bool _prev_standup_failed;
    bool _prev_balance_failed;
};

#define INIT_UI_STATUS_MOTION(pendulum_id) \
    { .pid = pendulum_id }

void ui_motion_status_reset(ui_status_motion_t *sig);


struct ui_event_attach_t {
    bool present;
};

#define INIT_EVENT_ATTACH() \
    { 0 }


struct ui_event_enable_pendulum_t {
    const pendulum_id_t pid;
    bool b_Dr_AF;  // switch: Enable drive, button Drive Antriebsfreigabe
    bool present;  // true in one iteration
};

#define INIT_EVENT_ENABLE_PENDULUM(pendulum_id) \
    { .pid = pendulum_id }


struct ui_event_move_pendulum_t {
    const pendulum_id_t pid;
    bool b_Pos_Set;   // button Move: Set sollPos TODO: This is probably not needed, fjg. 
    double Pos_Soll;  // slider value: sollPos
    bool present;
};

#define INIT_EVENT_MOVE_PENDULUM(pendulum_id) \
    { .pid = pendulum_id }


struct ui_event_autorun_pendulum_t {
    const pendulum_id_t pid;
    bool b_Au_Start;  // switch: Run automatic
    bool present;
};

#define INIT_EVENT_AUTORUN_PENDULUM(pendulum_id) \
    { .pid = pendulum_id }

#endif /* UI_H */