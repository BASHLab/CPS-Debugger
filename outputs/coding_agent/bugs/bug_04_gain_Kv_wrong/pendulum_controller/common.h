#ifndef COMMON_H
#define COMMON_H
// This file contains structures across functions to resolve circular
// dependencies

#include <stdbool.h>

#include "plclib.h"

typedef struct main_output_t main_output_t;
typedef struct pos_output_t pos_output_t;

// typedef struct enable_flags enable_flags;
typedef enum position_mode_t position_mode_t;
// typedef struct manual_context_t manual_context_t;

// struct manual_context_t {
//     logicaltime_t dt;
// 	edge_trigger_t Ha_ctx;
// 	edge_trigger_t Au_ctx;
// };

// #define PI 3.1415926; defined in plclib.h

enum position_mode_t {
    POS_IDLE = 0,
    POS_MANUAL,
    POS_STANDUP,
    POS_LQR,
};

struct pos_output_t {
    double v_soll;  // velocity SW [m/S]
    bool inPos;
};

#define INIT_POS_OUTPUT() \
    { 0 }

struct main_output_t {
    double x_soll;  // POS sollwert [m]
    double v_max;   // max velocity [m/S]
    double a_max;   // max acceleration [m/S²]
    //int au_state;
    position_mode_t pos_mode;
    // enable_flags *enable;
};

#define INIT_MAIN_OUTPUT() \
    { 0 }

// struct enable_flags {
//     bool sim_enable;
//     bool lqr_enable;
//     bool pos_enable;
//     bool su_enable;
//     bool axis_enable;
// };


/*
    Bit operations for unsigned integers.

    Let V be an unsigned integer of N bits lengths.
    Let I be an integer between 0 and N,  0 <= I < N.
    The following macros can be used to set, clear, get and test
    individual bits within V.
*/

// #define SET_BIT(V, I) (V) |= (0x1 << (I))
// #define CLEAR_BIT(V, I) (V) &= ~(0x1 << (I))
// #define GET_BIT(V, I) ((V) & (0x1 << (I)))
// #define TEST_BIT(V, I) (((V) & (0x1 << (I))) != 0)

#endif