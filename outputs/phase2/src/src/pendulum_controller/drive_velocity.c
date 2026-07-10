#include "lqrsim.h"
#include "drive_velocity.h"
#include "pos.h"

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

double pou_compute_drive_velocity(const double pos_vsoll,
              const double lqr_vsoll, 
              const position_mode_t pos_mode) {
    
    double velocity = compute_vsoll(pos_mode, pos_vsoll, lqr_vsoll);
    //printf("pos_mode: %d, pos_vsoll: %f, lqr_vsoll: %f, velocity: %f\n", pos_mode, pos_vsoll, lqr_vsoll, velocity);
    return velocity;
}
