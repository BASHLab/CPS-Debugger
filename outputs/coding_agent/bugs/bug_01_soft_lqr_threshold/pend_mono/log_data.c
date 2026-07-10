#include "log_data.h"
static WAXIDlrResult waxi_end_access(waxi_dlr_user_t iod) {
    WAXIDlrResult res = waxi_dlr_user_end_access(iod);
    if (WAXI_DL_OK != res)
    {
        LOG_ERROR("waxi_end_access", "Could not end access to input memory '%d'", iod);
    }
    return res;
}

void log_data(const int index, const waxi_dlr_user_t logger_handle, const uint16_t pendulum_state, const uint32_t iteration, uint64_t timestamp, 
              const double target_x, const double current_x, const double velocity, const double angle, const double angular_velocity) {
    const char *context = "log_data";
    const uint32_t revision = 0;

    /*Datalayer map
	    uint16_t pendulum_state[ARRAY_SIZE];
        uint32_t iteration[ARRAY_SIZE]; //
        uint64_t timestamp[ARRAY_SIZE]; // of size
        double target_x[ARRAY_SIZE]; // of size
        double position[ARRAY_SIZE]; //
        double velocity[ARRAY_SIZE]; //
        double angle[ARRAY_SIZE]; //
        double angular_velocity[ARRAY_SIZE];
     */
    int offset = 0;
    int prev_record_size = 0;
    if (index >= ARRAY_SIZE) // do not log more than ARRAY_SIZE records
    {
        return;
    }

    WAXIDlrResult res = waxi_dlr_user_begin_access(logger_handle, revision);
    if (WAXI_DL_OK != res)
    {
        LOG_ERROR(context, "Could not begin access to input memory '%d'",
                  logger_handle);
        return;
    }

    offset = sizeof(pendulum_state) * index;
    res = waxi_dlr_user_write_bytes(logger_handle, offset, &pendulum_state, sizeof(pendulum_state));
    if (WAXI_DL_OK != res)
    {
        LOG_ERROR(context, "Could not write bytes  to  memory '%d'", logger_handle);
        waxi_end_access(logger_handle);
        return;
    }
    prev_record_size = sizeof(pendulum_state);
    offset = sizeof(iteration) * index + ARRAY_SIZE * prev_record_size;

    res = waxi_dlr_user_write_bytes(logger_handle, offset, &iteration, sizeof(iteration));
    if (WAXI_DL_OK != res)
    {
        LOG_ERROR(context, "Could not write bytes  to  memory '%d'", logger_handle);
        waxi_end_access(logger_handle);
        return;
    }
    prev_record_size += sizeof(iteration);
    offset = index * sizeof(timestamp) + ARRAY_SIZE * prev_record_size;

    res = waxi_dlr_user_write_bytes(logger_handle, offset, &timestamp, sizeof(timestamp));
    if (WAXI_DL_OK != res)
    {
        LOG_ERROR(context, "Could not write bytes  to  memory '%d'", logger_handle);
        waxi_end_access(logger_handle);
        return;
    }
    prev_record_size += sizeof(timestamp);

    offset = index * sizeof(target_x) + ARRAY_SIZE * prev_record_size;
    res = waxi_dlr_user_write_bytes(logger_handle, offset, &target_x, sizeof(target_x));
    if (WAXI_DL_OK != res)
    {
        LOG_ERROR(context, "Could not write bytes  to  memory '%d'", logger_handle);
        waxi_end_access(logger_handle);
        return;
    }
    prev_record_size += sizeof(target_x);
    offset = index * sizeof(current_x) + ARRAY_SIZE * prev_record_size;
    res = waxi_dlr_user_write_bytes(logger_handle, offset, &current_x, sizeof(current_x));
    if (WAXI_DL_OK != res)
    {
        LOG_ERROR(context, "Could not write bytes  to  memory '%d'", logger_handle);
        waxi_end_access(logger_handle);
        return;
    }
    prev_record_size += sizeof(current_x);
    offset = index * sizeof(velocity) + ARRAY_SIZE * prev_record_size;
    res = waxi_dlr_user_write_bytes(logger_handle, offset, &velocity, sizeof(velocity));
    if (WAXI_DL_OK != res)
    {
        LOG_ERROR(context, "Could not write bytes  to  memory '%d'", logger_handle);
        waxi_end_access(logger_handle);
        return;
    }
    prev_record_size += sizeof(velocity);
    offset = index * sizeof(angle) +  ARRAY_SIZE * prev_record_size;
    res = waxi_dlr_user_write_bytes(logger_handle, offset, &angle, sizeof(angle));
    if (WAXI_DL_OK != res)
    {
        LOG_ERROR(context, "Could not write bytes  to  memory '%d'", logger_handle);
        waxi_end_access(logger_handle);
        return;
    }
    prev_record_size += sizeof(angle);
    offset = index * sizeof(angular_velocity) + ARRAY_SIZE * prev_record_size;
    res = waxi_dlr_user_write_bytes(logger_handle, offset, &angular_velocity, sizeof(angular_velocity));
    if (WAXI_DL_OK != res)
    {
        LOG_ERROR(context, "Could not write bytes  to  memory '%d'", logger_handle);
        waxi_end_access(logger_handle);
        return;
    }
    res = waxi_end_access(logger_handle);
    if (WAXI_DL_OK != res)
    {
        LOG_ERROR(context, "Could not end access to input memory '%d'", logger_handle);
    }
}

uint16_t read_end_of_process_flag(const waxi_dlr_user_t logger_handle)
{
    const char *context = "read_end_of_process_flag";
    const uint32_t revision = 0;
    uint16_t end_of_process = 0;
    int byte_offset = PROCESS_FLAG_BYTE_OFFSET;
    WAXIDlrResult res = waxi_dlr_user_begin_access(logger_handle, revision);
    if (WAXI_DL_OK != res)
    {
        LOG_ERROR(context, "Could not begin access to input memory '%d'",
                  logger_handle);
        return 0;
    }

    res = waxi_dlr_user_read_bytes(logger_handle, byte_offset, &end_of_process, sizeof(end_of_process));
    if (WAXI_DL_OK != res)
    {
        LOG_ERROR(context, "Could not read bytes  from  memory '%d'", logger_handle);
        waxi_end_access(logger_handle);
        return 0;
    }
    res = waxi_end_access(logger_handle);
    if (WAXI_DL_OK != res)
    {
        LOG_ERROR(context, "Could not end access to input memory '%d'", logger_handle);
    }
    return end_of_process;
}

void set_end_of_process_flag(const waxi_dlr_user_t logger_handle, uint16_t value)
{
    const char *context = "set_end_of_process_flag";
    const uint32_t revision = 0;
    WAXIDlrResult res = waxi_dlr_user_begin_access(logger_handle, revision);
    if (WAXI_DL_OK != res)
    {
        LOG_ERROR(context, "Could not begin access to input memory '%d'",
                  logger_handle);
        return;
    }
    int byte_offset = PROCESS_FLAG_BYTE_OFFSET;
    res = waxi_dlr_user_write_bytes(logger_handle, byte_offset, &value, sizeof(value));
    if (WAXI_DL_OK != res)
    {
        LOG_ERROR(context, "Could not write bytes  to  memory '%d'", logger_handle);
        waxi_end_access(logger_handle);
        return;
    }
    res = waxi_end_access(logger_handle);
    if (WAXI_DL_OK != res)
    {
        LOG_ERROR(context, "Could not end access to input memory '%d'", logger_handle);
    }
    return;
}
