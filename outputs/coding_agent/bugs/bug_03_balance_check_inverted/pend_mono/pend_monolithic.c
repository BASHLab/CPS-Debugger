#include <assert.h>
#include <stdint.h>
#include <stdio.h> // for debugging
#include <time.h>

/* WAXI includes */
#include "waxi/datalayer/definitions.h"
#include "waxi/datalayer/factory.h"
#include "waxi/datalayer/system.h"
#include "waxi/datalayer/user.h"
#include "waxi/datalayer/variant.h"
#include "waxi/scheduler/scheduler.h"
#include "waxi/services/datalayer.h"

/* WAXI SDK */
#include "waxi-c-sdk/datalayer.h"
#include "waxi-c-sdk/logging.h"
#include "waxi-c-sdk/timer.h"
#include "waxi-c-sdk/variant.h"

#include "read_ethercat_data.h"
#include "sys_io.h"
#include "ethercat_map.h"
#include "general.h"
#include "control_word.h"
#include "main.h"
#include "sys_params.h"

#include "control_word.h"
#include "main.h"
#include "drive_velocity.h"
#include "sys_params.h"

#include "main.h"
#include "standup_reliable.h"
#include "log_data.h"
// #include "lqr_sim.h"
#include "pos.h"

/* For this example we only have two states */
typedef enum State
{
    NONE = 0,
    INITIALIZED,
} State;

#define PERIOD_MS 1

// Maximum safe x distance the pendulum cart can go before we stop the algorithm and return to center
#define MAX_X_SAFE (MAX_X_DISPL - X_SAFETY_BUFFER)

// Flag to trigger controller to go into safety mode, stop balancing or swingup algorithm,
// and bring cart back to center before resetting
bool x_safety_error_triggered = false;

/* Some globals */
static waxi_dlr_factory_t factory = WAXI_INVALID_HOST_DATA_HANDLE;
static waxi_dlr_user_t ether_in_iod = WAXI_INVALID_HOST_DATA_HANDLE;
static waxi_dlr_user_t ether_out_iod = WAXI_INVALID_HOST_DATA_HANDLE;
static waxi_dlr_user_t logger_iod = WAXI_INVALID_HOST_DATA_HANDLE;
static int array_index = 0;

// static waxi_dlr_variant_t memory_map = WAXI_INVALID_HOST_DATA_HANDLE;
static State state = NONE;

sys_machine_input_t sys_machine_in = INIT_SYS_MACHINE_INPUT();

// For Pit Pendulum, changed from P2 to P1
sys_drive_input_t sys_drive_in = INIT_SYS_DRIVE_INPUT(P1);
sys_machine_output_t sys_machine_out = INIT_SYS_MACHINE_OUTPUT();

// P2 changed to P1
sys_drive_output_t sys_drive_out = INIT_SYS_DRIVE_OUTPUT(P1);

general_drive_out_t gen_drive_out = INIT_GENERAL_DRIVE_OUT();
const logicaltime_t cycletime = PERIOD_MS * MSEC;
general_machine_context_t gen_machine_ctx_p1 =
    INIT_GENERAL_MACHINE_CTX(cycletime);

general_axis_out_t gen_axis_out = INIT_GENERAL_AXIS_OUT();
standup_context_t su_ctx = INIT_STANDUP_CONTEXT(cycletime);

lqr_sim_context_t lqr_ctx = INIT_LQR_SIM_CONTEXT(cycletime);

standup_output_t standup_out;

main_output_t main_out = INIT_MAIN_OUTPUT();

lqr_sim_outputs_t lqr_sim_out = INIT_LQR_SIM_OUTPUT();

pos_output_t pos_out = INIT_POS_OUTPUT();
pos_context_t pos_ctx = INIT_POS_CONTEXT(cycletime);

enable_flags flags;

ui_status_machine_t machine_status;
ui_status_pendulum_t p1_status;
ui_event_enable_pendulum_t p1_enable_event;
ui_event_autorun_pendulum_t p1_autorun_event;
ui_event_move_pendulum_t p1_move_event;
ui_status_motion_t p1_motion_status;

general_drive_context_t gen_drive_ctx_p1 =
    INIT_GENERAL_DRIVE_CONTEXT(cycletime);
/* Static description of the ethercat input and output */
// Changed these values to match the interface for the pendulum in Pittsburgh
static const char *ether_in_address =
    "fieldbuses/ethercat/master/instances/ethercatmaster/realtime_data/input";
static const char *ether_out_address =
    "fieldbuses/ethercat/master/instances/ethercatmaster/realtime_data/output";
/* The following data could also be generated or dynamically interpreted by
 * using the memory map*/
#ifndef ether_map_revision
#define ether_map_revision 1566571575
#endif
//static const uint32_t ether_map_revision = 1566571575;
//2550826156;

/*
static const char *ether_in_address =
    "fieldbuses/ethercat/master/instances/a620x1/realtime_data/input";
static const char *ether_out_address =
    "fieldbuses/ethercat/master/instances/a620x1/realtime_data/output";
static const uint32_t ether_map_revision = 1307587946;
*/

/*Todo: Add code to create handle to the logger nodes and in balanced mode start logging necessary data*/
static const char *dl_logger_address = "sdk/cpp/datalayer/pendulum-logging/output";

uint64_t packet_counter = 0;
uint32_t iteration = 0;


/*  Initialization functions to the factory, datalayer logger nodes, ethercat input and output nodes
    Also initializes the system parameters and the drive output parameters
*/

static bool init()
{
    const char *context = "init";

    printf("Initialising Wasm callable...\n");

    factory = waxi_service_dlr_factory();
    if (WAXI_INVALID_HOST_DATA_HANDLE == factory)
    {
        LOG_ERROR(context, "Could not get the data layer factory\n");
        return false;
    }

    WAXIDlrResult res = waxi_dlr_factory_open_memory(
        factory, &ether_in_iod, sizeof(ether_in_iod), ether_in_address);
    if (res != WAXI_DL_OK)
    {
        LOG_ERROR(context, "Could not open memory '%s'", ether_in_address);
        return false;
    }
    res = waxi_dlr_factory_open_memory(factory, &ether_out_iod, sizeof(ether_out_iod),
                                       ether_out_address);
    if (res != WAXI_DL_OK)
    {
        LOG_ERROR(context, "Could not open memory '%s'", ether_out_address);
        return false;
    }
    res = waxi_dlr_factory_open_memory(factory, &logger_iod, sizeof(logger_iod), dl_logger_address);
    if (res != WAXI_DL_OK)
    {
        LOG_ERROR(context, "Could not open memory '%s'", dl_logger_address);
        return false;
    }
    // Initialize system parameters
    sys_reset_machine_ouput(&sys_machine_out);
    sys_reset_drive_output(&sys_drive_out);
    ethercat_write(ether_out_iod, &sys_machine_out, &sys_drive_out, ether_map_revision);

    p1_enable_event.b_Dr_AF = false;
    p1_enable_event.present = true;
    p1_autorun_event.present = true;
    p1_autorun_event.b_Au_Start = true;

    iteration = 0;
    x_safety_error_triggered = false;

    // Initialize datalayer logger
    set_end_of_process_flag(logger_iod, LOG_BUFFER_NOT_READY);
    array_index = 0;

    return true;
}

/* Teardown */
static void deinit()
{
    const char *context = "deinit";
    printf("De-initialising Wasm callable...\n");

    // Stop drive controller with a zero velocity command before the task is de-initialized
    reset_drive(&sys_drive_out);
    WAXIDlrResult res = ethercat_write(ether_out_iod, &sys_machine_out, &sys_drive_out, ether_map_revision);
    if (res != WAXI_DL_OK)
    {
        LOG_ERROR(context, "Could not write to ethercat when stopping linear actuator at end of task");
    }

    // Put drive controller in Ab state so it is no longer sending power to the linear actuator
    pou_drive_control_word(true, false, sys_drive_in.Dr_Status, &sys_drive_out.Dr_Control);
    res = ethercat_write(ether_out_iod, &sys_machine_out, &sys_drive_out, ether_map_revision);
    if (res != WAXI_DL_OK)
    {
        LOG_ERROR(context, "Could not write to ethercat when setting drive controller to Ab state at end of task");
    }

    // Write LOG_END_OF_DATA value to the iteration datalayer to signal end of logging
    log_data (array_index, logger_iod, LOG_END_OF_DATA, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0);

    // Set log buffer ready to allow datalayer reader to get the last batch of data
    if (array_index < ARRAY_SIZE / 2)
    {
        set_end_of_process_flag(logger_iod, LOG_FRONT_BUFFER_READY);
    } else
    {
        set_end_of_process_flag(logger_iod, LOG_BACK_BUFFER_READY);
    }

    res = waxi_dlr_factory_close_memory(factory, ether_out_iod);
    if (res != WAXI_DL_OK)
    {
        LOG_ERROR(context, "Could not close output memory '%d'", ether_out_iod);
    }

    res = waxi_dlr_factory_close_memory(factory, ether_in_iod);
    if (res != WAXI_DL_OK)
    {
        LOG_ERROR(context, "Could not close input memory '%d'", ether_in_iod);
    }
}

#define MAX_SETPONTS 4
static const double setpoints[MAX_SETPONTS] = {0.0, 0.05, -0.09, 0.09};
static int setpoint_index = 0;
static int dl_write_counter = 0;
int balanced_counter = 0;

bool stays_balanced(bool is_balanced)
{
    if (is_balanced)
    {
        balanced_counter++;
        if (balanced_counter >= 1500)
            return true;
    }
    else
    {
        balanced_counter = 0;
    }
    return false;
}

static uint64_t ts2ns(struct timespec *ts)
{
    return ts->tv_sec * 1000000000 + ts->tv_nsec;
}

static int toggle_flag = false;
bool already_balanced = false;
int setpoint_counter = 0;
static bool start_new_iteration = false;

/* Tick */
static WAXIDlrResult tick()
{
    const char *context = "pend_mono_tick";
    WAXIDlrResult res = WAXI_DL_OK;
    double velocity = 0.0;
    bool drive_release = false;
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    uint64_t timestamp = ts2ns(&ts);
    uint16_t pendulum_state = LOG_SWING_UP;

    // Clear the datalayer logger flag on the first cycle
    if (packet_counter == 0)
    {
        set_end_of_process_flag(logger_iod, LOG_BUFFER_NOT_READY);
    }

   // For Pit Pendulum changed this to P1 instead of P2
    res = ethercat_read(ether_in_iod, &sys_machine_in, &sys_drive_in, P1, ether_map_revision);
    if (res != WAXI_DL_OK)
    {
        LOG_ERROR(context, "Could not read input memory");
        return res;
    }

    if (!toggle_flag) // simulate a rising edge for these signals and do the initial calibration
    {
        toggle_flag = true;
        gen_drive_ctx_p1.prev_x =
            sys_drive_in.Dr_PosIW / 1E7; // init prev_x
        gen_drive_ctx_p1.prev_v = 0.0;
        gen_drive_ctx_p1.prev_theta_raw = sys_drive_in.DG_Ana - 21757;
    }

    bool ST_Ein =
        pou_general_machine(&gen_machine_ctx_p1, &sys_machine_in,
                            &sys_machine_out, &machine_status);

    // For Pit Pendulum set ST_Ein to true to enable drive control and bypass safety checks
    ST_Ein = true;
    p1_enable_event.b_Dr_AF = true;

    // The code below is used to simulate the rising edge of the enable signal and makes sure STEIN is true for the first 5000 packets
    // Please feel free to remove this code if you do not need this feature. This also leads to a slight delay before the actual swing up starts

    // We don't need to do this delay for the Pittsburgh pendulum.
/*
#if 1
    if (packet_counter == 5000 && ST_Ein)
    {
        p1_enable_event.present = false;
        p1_enable_event.b_Dr_AF = false;
    }
    else if (packet_counter > 5500 && ST_Ein)
    {
        p1_enable_event.present = true;
        p1_enable_event.b_Dr_AF = true;
    }
#endif
*/
    drive_release = p1_enable_event.present && ST_Ein && p1_enable_event.b_Dr_AF;

    pou_general_drive(&gen_drive_ctx_p1, &p1_general_params, &sys_drive_in,
                      &p1_status, &gen_drive_out);
    // LOG_TRACE(context, "Drive out: %f, %f %f", gen_drive_out.theta_ist, gen_drive_out.x_ist, gen_drive_out.v_ist);

    pou_general_axis(ST_Ein, &p1_enable_event, &gen_axis_out);
    packet_counter++;
    {
        pou_main(&gen_drive_out, &p1_manual_params, &p1_standup_params,
                 &standup_out, &p1_autorun_event, &p1_move_event,
                 &flags, &main_out);

        // Check to see if the pendulum cart exceeded the linear x-axis safety buffer
        if (gen_drive_out.x_ist > MAX_X_SAFE || gen_drive_out.x_ist < -(MAX_X_SAFE)) {
            x_safety_error_triggered = true;
        }

        // LOG_TRACE(context, "pos_mode: %d, flags.su: %d flags.lqr: %d, flags.pos:%d", pos_mode, flags.su_enable, flags.lqr_enable, flags.pos_enable);

        if (x_safety_error_triggered)
        {
            pendulum_state = LOG_RESET;

            // Cart is to the right of center so send a small negative velocity command to bring it back to the left
            if (gen_drive_out.x_ist > 0.001)
            {
                velocity = -0.05;
            }
            // Cart is to the left of center so send a small positive velocity command to bring it back to the right
            else if (gen_drive_out.x_ist < -0.001)
            {
                velocity = 0.05;
            }
            // Cart is back to center so clear the error, stop the cart, and try to swing up again
            else
            {
                velocity = 0.0;
                x_safety_error_triggered = false;
                su_ctx.first_push = true;
                standup_out.su_pndlum_ok = false;
                flags.su_enable = true;
                flags.lqr_enable = false;

                // Reset to move pendulum to first setpoint when it is balanced again
                setpoint_counter = 0;
                setpoint_index = 0;
                iteration = (iteration + 1);
            }
        }
        else {
            if (flags.su_enable && !flags.lqr_enable)
            {
                pendulum_state = LOG_SWING_UP;

                velocity =
                    pou_standup_rel(&gen_drive_out, &su_ctx, &su_sim_params,
                                    &standup_out, &p1_motion_status);
            }
            if ((standup_out.su_pndlum_ok == true) || flags.lqr_enable)
            {
                pendulum_state = LOG_BALANCE;

                velocity =
                    pou_lqr_sim(&lqr_ctx, &p1_lqr_sim_params, &gen_drive_out,
                                main_out.x_soll, &lqr_sim_out);

                flags.su_enable = false;

                /*This makes sure we dont start moving the pendulum until it stays balanced for a while*/
                if (!already_balanced)
                {
                    already_balanced = stays_balanced(!standup_out.su_pndlum_ok);
                }
                else
                {
                    double delta_position = 0.009;

                    // Change the setpoint when the current setpoint is reached and stable
                    if (gen_drive_out.x_ist < (main_out.x_soll + delta_position) && gen_drive_out.x_ist > (main_out.x_soll - delta_position))
                    {
                        // Stay at this setpoint for 100 cycles
                        setpoint_counter++;
                        if (setpoint_counter >= 100)
                        {
                            setpoint_index = (setpoint_index + 1) % MAX_SETPONTS;
                            main_out.x_soll = setpoints[setpoint_index];
                            setpoint_counter = 0;

                            // All setpoints have been reached ?
                            if (setpoint_index == 0)
                            {
                                iteration = (iteration + 1);
                            }
                        }
                    } else
		    {
		        setpoint_counter = 0;
		    }
                }
            }
        }
    }

    // Log data to datalayer for every cycle, regardless of system state
    log_data (array_index, logger_iod, pendulum_state, iteration, timestamp,
              main_out.x_soll, gen_drive_out.x_ist, gen_drive_out.v_ist,
              gen_drive_out.theta_ist, gen_drive_out.theta_d_ist);
    array_index = (array_index + 1);

    // If we filled the first half of the datalayer buffer, signal that it is ready to be read
    if (array_index == ARRAY_SIZE / 2)
    {
        set_end_of_process_flag(logger_iod, LOG_FRONT_BUFFER_READY);
    }

    // If we filled the second half of the datalayer buffer, signal that it is ready to be read
    // and loop back to using the first half of the buffer
    if (array_index >= ARRAY_SIZE)
    {
        set_end_of_process_flag(logger_iod, LOG_BACK_BUFFER_READY);
        array_index = 0;
    }

    sys_drive_out.Dr_VelSW = velocity * 60000 * 1000;

    pou_drive_control_word(ST_Ein, gen_axis_out.s_Dr_AF, sys_drive_in.Dr_Status, &sys_drive_out.Dr_Control);

    // LOG_TRACE(context, "Drive in: %d  Drive control word generated: %d vel:%d, input drive release: %d \n", sys_drive_in.Dr_Status, sys_drive_out.Dr_Control, sys_drive_out.Dr_VelSW, drive_release);
    res = ethercat_write(ether_out_iod, &sys_machine_out, &sys_drive_out, ether_map_revision);
    if (res != WAXI_DL_OK)
    {
        LOG_ERROR(context, "Could not write to ethercat");
        return res;
    }

    return res;
}



//The scheduler invokes this function every cycle
//Do not change the code here. This is standard code that is used to interact with the ctrlX Scheduler

__attribute__((export_name("execute"))) waxi_SchedEventResponse execute(
    const waxi_SchedEventType event,
    const waxi_SchedEventPhase phase,
    const waxi_dlr_variant_t param)
{
    const char *context = "execute";
    switch (event)
    {
    case WAXI_SCHED_EVENT_TICK:
        // Just to be sure about the scheduler behavior
        assert(WAXI_SCHED_EVENT_PHASE_NONE == phase);
        assert(INITIALIZED == state);
        if (INITIALIZED == state)
        {
            WAXIDlrResult res = tick();
            if (res != WAXI_DL_OK)
            {
                LOG_ERROR(context, "Tick failed\n");
            }
        }
        return WAXI_SCHED_EVENT_RESP_OKAY;

    case WAXI_SCHED_EVENT_SWITCH_TO_OPERATING:
        switch (phase)
        {
        case WAXI_SCHED_EVENT_PHASE_BEGIN:
            return WAXI_SCHED_EVENT_RESP_OKAY;
        case WAXI_SCHED_EVENT_PHASE_EXECUTE:
            return WAXI_SCHED_EVENT_RESP_OKAY;
        case WAXI_SCHED_EVENT_PHASE_END:
            return WAXI_SCHED_EVENT_RESP_OKAY;
        default:
            // This should never happen
            return WAXI_SCHED_EVENT_RESP_OKAY;
        }

    case WAXI_SCHED_EVENT_SWITCH_TO_SETUP:
        switch (phase)
        {
        case WAXI_SCHED_EVENT_PHASE_BEGIN:
            return WAXI_SCHED_EVENT_RESP_OKAY;
        case WAXI_SCHED_EVENT_PHASE_EXECUTE:
            return WAXI_SCHED_EVENT_RESP_OKAY;
        case WAXI_SCHED_EVENT_PHASE_END:
            return WAXI_SCHED_EVENT_RESP_OKAY;
        default:
            // This should never happen
            return WAXI_SCHED_EVENT_RESP_OKAY;
        }

    case WAXI_SCHED_EVENT_SWITCH_TO_SERVICE:
        switch (phase)
        {
        case WAXI_SCHED_EVENT_PHASE_BEGIN:
            return WAXI_SCHED_EVENT_RESP_OKAY;
        case WAXI_SCHED_EVENT_PHASE_EXECUTE:
            if (NONE == state)
            {
                init();
                state = INITIALIZED;
            }
            return WAXI_SCHED_EVENT_RESP_OKAY;
        case WAXI_SCHED_EVENT_PHASE_END:
            return WAXI_SCHED_EVENT_RESP_OKAY;
        default:
            // This should never happen
            return WAXI_SCHED_EVENT_RESP_OKAY;
        }

    case WAXI_SCHED_EVENT_SWITCH_TO_EXIT:
        switch (phase)
        {
        case WAXI_SCHED_EVENT_PHASE_BEGIN:
            return WAXI_SCHED_EVENT_RESP_OKAY;
        case WAXI_SCHED_EVENT_PHASE_EXECUTE:
            if (INITIALIZED == state)
            {
                deinit();
                state = NONE;
            }
            return WAXI_SCHED_EVENT_RESP_OKAY;
        case WAXI_SCHED_EVENT_PHASE_END:
            return WAXI_SCHED_EVENT_RESP_OKAY;
        default:
            // This should never happen
            return WAXI_SCHED_EVENT_RESP_OKAY;
        }

    case WAXI_SCHED_EVENT_TASK_PROPERTIES_CHANGE:
        switch (phase)
        {
        case WAXI_SCHED_EVENT_PHASE_BEGIN:
            return WAXI_SCHED_EVENT_RESP_OKAY;
        case WAXI_SCHED_EVENT_PHASE_EXECUTE:
            return WAXI_SCHED_EVENT_RESP_OKAY;
        case WAXI_SCHED_EVENT_PHASE_END:
            return WAXI_SCHED_EVENT_RESP_OKAY;
        default:
            // This should never happen
            return WAXI_SCHED_EVENT_RESP_OKAY;
        }

    case WAXI_SCHED_EVENT_GET_CURRENT_STATE:
        assert(WAXI_SCHED_EVENT_PHASE_NONE == phase);
        return WAXI_SCHED_EVENT_RESP_OKAY;

    default:
        // This should never happen
        return WAXI_SCHED_EVENT_RESP_OKAY;
    }
}
