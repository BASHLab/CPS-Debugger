/* global includes */
#include <stdbool.h>

/* self include */
#include "dataflow.h"

void dataflow_reset_sensors(dataflow_sensing_t *sensors) {
    sensors->x_ist = 0.0;
    sensors->v_ist = 0.0;
    sensors->theta_ist = 0.0;
    sensors->theta_d_ist = 0.0;
    sensors->theta_180 = false;
    sensors->theta_d_0 = false;
    sensors->number = 0;
}

void dataflow_reset_actuators(dataflow_actuation_t *actuators) {
    actuators->v_soll = 0.0;
    actuators->number = 0;
}