#ifndef SYS_PARAMS_H
#define SYS_PARAMS_H

/* local includes */
#include "plclib.h"

/* self include */
#include "sys_params.h"

// Maximum displacement of pendulum cart on the X axis in both directions from the center
#define MAX_X_DISPL 0.18 // [m]

// Safety buffer to give the pendulum controller slack to stop the cart before it
// reaches the end of the track and triggers a hard stop in the drive controller
#define X_SAFETY_BUFFER 0.03 // [m]

/* 
 *  General parameters
 */

typedef struct general_params_t {
    int16_t offsPend;  // offset pendulum
    double theta_gl;   // Glättung
    double theta_gl2;  // Glättung
    double calibrated_PI; // used in rad calculation 
    double v_gl;       // Glättung
    logicaltime_t theta_180_delay;  // on delay [msec]
    double theta_180_epsilon;
    logicaltime_t theta_d_0_delay;  // on dealy [msec]
    double theta_d_0_epsilon; 
    double filter_theta_d;
    double filter_theta;
    double filter_commanded_v;
    double p2_encoder_delta_k;
} general_params_t;

extern const general_params_t p1_general_params;
extern const general_params_t p2_general_params;


/*
    Manual parameters
*/

typedef struct manual_params_t {
    // not used
    // const double x_target_start;  // [m] x position
    const double v_manual;        // [m/s] velocity
    const double a_manual;        // [m/s²] acceleration
} manual_params_t;

extern const manual_params_t p1_manual_params;
extern const manual_params_t p2_manual_params;

/*
 *   Stand up parameters
 */

typedef struct standup_params_t {
    // not used
    // const double x_target_start;  // [m] x position
    const double v_standup;       // [m/s] velocity
    const double a_standup;       // [m/s²] acceleration
} standup_params_t;

extern const standup_params_t p1_standup_params;
extern const standup_params_t p2_standup_params;


struct move_params_t {
    const double pos;          // [m] position
    const double vel;          // [m/s] velocity
    const double acc;          // [m/s²] acceleration
    logicaltime_t time;  // duration
};

#define NO_MOVES 11
typedef struct move_params_t standup_moves_t[NO_MOVES];

extern standup_moves_t p1_standup_moves;
extern standup_moves_t p2_standup_moves;

/*
 *   LQR and Sim parameters
 */

typedef struct lqr_sim_params_t {
    double m_axis;  // [kg]
    double F_max;   // [N]
    double v_max;   // [m/s]
    double K_x;     // Weight of pos
    double K_v;     // Weight of velocity
    double K_t;     // Weight of theta
    double K_td;    // Weight of delta theta
    int counter_xref;
    int counter_goodrange;
    int xref_switch;
    double max_xdispl;
} lqr_sim_params_t;

extern lqr_sim_params_t p1_lqr_sim_params;
extern lqr_sim_params_t p2_lqr_params;


typedef struct pos_params_t {
    const double K_v;           // Weight of velocity soll
    const double factor;        // Factor for velocity soll ??
    const double epsilon_x ;    // ABS(x_soll - x_ist) < epsilon_x
} pos_params_t;

extern const pos_params_t p1_pos_params;
extern const pos_params_t p2_pos_params;

/*
*   Energy-based standup parameters
*/

typedef struct su_sim_params_t {
    const double m_axis;  // [kg]
    const double l_axis;
    const double F_max;   // [N]
    const double v_max;   // [m/s]
    double K_su;     // Weight of pure energy su
    double K_cw;     // Weight of cart boundary well
    double K_vw;
    const double max_xdispl;
} su_sim_params_t;

extern su_sim_params_t su_sim_params;

/*
    Main parameters
*/

typedef struct main_params_t {
    const manual_params_t *man_params;
    const standup_params_t *su_params;
    const standup_moves_t *su_moves;
    const lqr_sim_params_t *ls_params;
    const pos_params_t *pos_params;
} main_params_t;

extern const main_params_t p1_main_params;
extern const main_params_t p2_main_params;

#endif /* SYS_PARAMS_H */
