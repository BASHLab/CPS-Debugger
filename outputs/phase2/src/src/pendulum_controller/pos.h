#ifndef POS_H
#define POS_H

#include <stdbool.h>

#include "common.h"
#include "plclib.h"
#include "standup_reliable.h"
#include "sys_params.h"
#include "general.h"


typedef struct pos_t {
    double x_soll;  // POS sw [m]
    double v_max;   // max velocity [m/S]
    double a_max;   // max acceleration [m/S²]
}pos_t;

typedef struct pos_context_t pos_context_t;

// typedef bool pos_result_t;




double pou_position(const pos_params_t *pos_params, const general_drive_out_t *gen_drive_out, 
                  const position_mode_t pos_mode, const main_output_t *main_out,
                  const standup_output_t *standup_out, pos_context_t *ctx);





struct pos_context_t {
    const logicaltime_t dt;
    double prev_v_soll;
};

#define INIT_POS_CONTEXT(cycletime) \
    { .dt = cycletime }



#define INIT_POS_OUTPUT() \
    { 0 }

#endif /* POS_H */