#ifndef AXIS_H
#define AXIS_H

#include <stdint.h>

/* local includes */
#include "pos.h"

typedef struct axis_context_t axis_context_t;

void pou_axis(bool axis_enable, bool STEin, const double pos_v_soll,
              const double lqr_vsoll, const general_axis_out_t *axis_out,
              const position_mode_t pos_mode, const sys_drive_input_t *drive_in,
              sys_drive_output_t *drive_out);

struct axis_context_t {};

void sys_reset_drive(sys_drive_output_t *sys_drive_out);

#define INIT_AXIS_CONTEXT() \
    {}

#endif /* AXIS_H */