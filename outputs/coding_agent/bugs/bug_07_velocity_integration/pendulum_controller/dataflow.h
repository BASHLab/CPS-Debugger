#ifndef DATAFLOW_H
#define DATAFLOW_H

#include <stdint.h>
#include "sys_io.h"

typedef struct dataflow_sensing_t dataflow_sensing_t;
typedef struct dataflow_actuation_t dataflow_actuation_t;

struct dataflow_sensing_t {
    const pendulum_id_t pid;
    double x_ist;           // pos IW [m]
    double v_ist;           // vel IW [m/S]
    double theta_ist;       // theta IW [rad]
    double theta_d_ist;     // delta theta IW [rad/S]
    bool theta_180;         // theta ca. 180°
    bool theta_d_0;         // theta_d ca. 0 rad/s
    bool pendulum_enabled;  // allows
    uint32_t number;        // message number for debugging
};

#define INIT_DATAFLOW_SENSING(pendulum_id) \
    { .pid = pendulum_id }

void dataflow_reset_sensors(dataflow_sensing_t *sensors);

struct dataflow_actuation_t {
    const pendulum_id_t pid;
    double v_soll; // velocity SW [m/S]
    uint32_t number;     // message number for debugging
};

#define INIT_DATAFLOW_ACTUATION(pendulum_id) \
    { .pid = pendulum_id }

void dataflow_reset_actuators(dataflow_actuation_t *actuators);

#endif /* DATAFLOW_H */