#ifndef READ_ETHERCAT_DATA_H
#define READ_ETHERCAT_DATA_H

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

#include "sys_io.h"




WAXIDlrResult ethercat_read(const waxi_dlr_user_t in_handle,
                          sys_machine_input_t *machine_in,
                          sys_drive_input_t *drive_in, const int drive_pid, const uint32_t revision);



WAXIDlrResult ethercat_write(const waxi_dlr_user_t out_handle,
                           const sys_machine_output_t *const machine_out,
                           const sys_drive_output_t *const drive_out, const uint32_t revision);

#endif 