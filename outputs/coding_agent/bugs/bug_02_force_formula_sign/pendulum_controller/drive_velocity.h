#ifndef DRIVE_VELOCITY_H
#define DRIVE_VELOCITY_H
#include <stdint.h>
#include "pos.h"

double pou_compute_drive_velocity(const double pos_vsoll,
              const double lqr_vsoll, 
              const position_mode_t pos_mode);

#endif /* DRIVE_VELOCITY_H */