#ifndef LOG_DATA_H
#define LOG_DATA_H


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
#include "waxi-c-sdk/scheduler.h"

#define ARRAY_SIZE 16000
#define SIZE_OF_RECORD 54
#define PROCESS_FLAG_BYTE_OFFSET (ARRAY_SIZE * SIZE_OF_RECORD)

// Special values to log pendulum state changes
#define LOG_SWING_UP 0
#define LOG_BALANCE 1
#define LOG_RESET 2
#define LOG_END_OF_DATA 3

// Special values for the end_of_process_flag to signal which parts of the datalayer buffer to read
#define LOG_BUFFER_NOT_READY 0
#define LOG_FRONT_BUFFER_READY 1
#define LOG_BACK_BUFFER_READY 2

void log_data(const int index, const waxi_dlr_user_t logger_handle, const uint16_t pendulum_state, const uint32_t iteration, uint64_t timestamp, 
              const double target_x, const double current_x, const double velocity, const double angle, const double angular_velocity); 

void set_end_of_process_flag(const waxi_dlr_user_t logger_handle, uint16_t value);
uint16_t read_end_of_process_flag(const waxi_dlr_user_t logger_handle);

#endif  // LOG_DATA_H
