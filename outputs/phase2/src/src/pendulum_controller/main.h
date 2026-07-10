#ifndef MAIN_H
#define MAIN_H

#include "common.h"
#include "general.h"
#include "lqrsim.h"
#include "plclib.h"
#include "pos.h"
#include "standup_reliable.h"
#include "sys_params.h"

typedef enum operation_mode_t operation_mode_t;

typedef enum automatic_mode_t automatic_mode_t;

typedef struct enable_flags enable_flags;

typedef struct main_context_t main_context_t;


enum operation_mode_t {
    BA_IDLE = 0,
    BA_HAND,
    BA_AUTO,
};

enum automatic_mode_t {
    AUTO_IDLE = 0,
    AUTO_STANDUP,
    AUTO_BALANCE,
    AUTO_FAILED,
};

struct enable_flags {
    bool lqr_enable;
    bool pos_enable;
    bool su_enable;
};

struct main_context_t {
    //const logicaltime_t dt;
    //operation_mode_t prev_BA;
    //position_mode_t pos_mode;
    //automatic_mode_t auto_mode;
    int prev_au_state;
};

#define INIT_MAIN_CONTEXT()                                       \
    {       .prev_au_state = 0  }
    




void pou_main(
              const general_drive_out_t *gen_out,  //
              const manual_params_t *man_params,
              const standup_params_t *su_params, standup_output_t *su_out,
              ui_event_autorun_pendulum_t *pendulum_autorun,
              const ui_event_move_pendulum_t *pendulum_move,  //
              enable_flags *flags,
              main_output_t *out);


#endif /* MAIN_H */