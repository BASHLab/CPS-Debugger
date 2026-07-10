
/* local includes */
// #include "lqrsim.h"
// #include "main.h"
// #include "plclib.h"
// #include "standup.h"
// #include "pos.h"

/* self include */
#include "sys_params.h"

/*
 *  General parameters
 */


const general_params_t p1_general_params = {
    .offsPend = 21757,  // calibrated
    .theta_gl = 0.1,
    .theta_gl2 = 0.01,
    .calibrated_PI = 3.1415926535,
    .v_gl = 0.08,       // calibrated
    .theta_180_delay = 100 * MSEC,
    .theta_180_epsilon = 0.235,
    .theta_d_0_delay = 100 * MSEC,
    .theta_d_0_epsilon = 0.03,
    .filter_theta_d = 0.8,
    .filter_theta = 1.0,
    .p2_encoder_delta_k = 1.0,

};

const general_params_t p2_general_params = {
    .offsPend = 19845,  // calibrated different to p1
    .theta_gl = 0.1,
    .theta_gl2 = 0.01,
    .calibrated_PI =  3.1415926535,
    .v_gl = 0.1,   // calibrated different to p1
    .theta_180_delay = 100 * MSEC,
    .theta_180_epsilon = 0.235,
    .theta_d_0_delay = 100 * MSEC,
    .theta_d_0_epsilon = 0.03,
    .filter_theta_d = 0.8,
    .filter_theta = 1.0,
    .p2_encoder_delta_k = 1.0,
    
};

/*
    Manual parameters
*/

const manual_params_t p1_manual_params = {
    .v_manual = 0.2,  // [m/s] velocity
    .a_manual = 0.5,  // [m/s²] acceleration
};

const manual_params_t p2_manual_params = {
    .v_manual = 0.2,  // [m/s] velocity
    .a_manual = 0.5,  // [m/s²] acceleration
};

/*
 *   Stand up parameters
 */

const standup_params_t p1_standup_params = {
    // This should be zero, because it is set to 0.0 at auto mode start
    // .v_standup = 2.0,  // [m/s] velocity
    .v_standup = 0.2,  // [m/s] velocity
    .a_standup = 0.5,  // [m/s²] acceleration
};

const standup_params_t p2_standup_params = {
    // This should be zero, because it is set to 0.0 at auto mode start
    // .v_standup = 2.0,  // [m/s] velocity
    .v_standup = 0.2,  // [m/s] velocity
    .a_standup = 0.5,  // [m/s²] acceleration
};

#define V_MAX_ALL 0.4 //original was 0.4
#define A_MAX_ALL 2.0




/*
 *   LQR and Sim parameters
 */

// 89.0 97.0 361.0 40.0
lqr_sim_params_t p1_lqr_sim_params = {
    .m_axis = 2.3,   // [kg]
    .F_max = 100.0,  // [N] 800.0
    // .v_max = 2.0,  global 2.0 is overwritten by 1.0 in pos code
    .v_max = 1.0,  // [m/s]

    // Pittsburgh specific LQR parameters provided by Marco Giani (CR/APT5)
    .K_x = 100.0, // 150.0 alternate value
    .K_v = 97.0,
    .K_t = 280.0, // 376.0 alternate value
    .K_td = 42.0,
    
    .counter_xref = 0,
    .counter_goodrange = 0,
    .xref_switch = 0,
    .max_xdispl = MAX_X_DISPL - X_SAFETY_BUFFER, // [m]
};

lqr_sim_params_t p2_lqr_params = {
    .m_axis = 2.3,   // [kg]
    .F_max = 800.0,  // [N]
    .v_max = 2.0,    // [m/s]

    .K_x = 89.0, //89.0,
    .K_v = 97.0,
    .K_t = 361.0,
    .K_td = 40.0,
    

    .counter_xref = 0,
    .counter_goodrange = 0,
    .xref_switch = 0,
    .max_xdispl = MAX_X_DISPL - X_SAFETY_BUFFER, // [m]
};

/*
 *  Pos parameters
 */

const pos_params_t p1_pos_params = {
    .K_v = 0.03,        // Weight for calculating v_soll
    .factor = 10.0,     // ??
    .epsilon_x = 0.005  // [m],
};

const pos_params_t p2_pos_params = {
    .K_v = 0.03,        // Weight for calculating v_soll
    .factor = 10.0,     // ??
    .epsilon_x = 0.005  // [m],
};

/*
*   Energy-based standup parameters
*/
/*
.m_axis = 2.3,   // [kg]
    .l_axis = 0.2,  // [m]
    .F_max = 800.0,  // [N]
    .v_max = 1.25,  // [m/s]
    .K_su = -1.8,  //keeps the pendulum upwards
    .K_cw = 2.0,  // tries to limit the horizontal movements
    .K_vw = 2.0,  // tries to limit the cart speed at switch to balancing
    


*/

// Swing up parameters that work with the Pittsburgh pendulum
su_sim_params_t su_sim_params = {
    .m_axis = 2.3,   // [kg]
    .l_axis = 0.2,  // [m]
    .F_max = 800.0,  // [N]
    .v_max = 1.25,  // [m/s]
    .K_su = -1.8,  //keeps the pendulum upwards
    .K_cw = 2.0,  // tries to limit the horizontal movements
    .K_vw = 2.0,  // tries to limit the cart speed at switch to balancing
    .max_xdispl = MAX_X_DISPL - X_SAFETY_BUFFER // [m]
};

/*
 * Main parameters
 * These are just collectios of parameters for the instances of POU main
 */

const main_params_t p1_main_params = {.man_params = &p1_manual_params,
                                      .su_params = &p1_standup_params,
                                      .su_moves = &p1_standup_moves,
                                      .ls_params = &p1_lqr_sim_params,
                                      .pos_params = &p1_pos_params};

const main_params_t p2_main_params = {.man_params = &p2_manual_params,
                                      .su_params = &p2_standup_params,
                                      .su_moves = &p2_standup_moves,
                                      .ls_params = &p2_lqr_params,
                                      .pos_params = &p2_pos_params};
