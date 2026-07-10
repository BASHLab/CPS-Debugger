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

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include "sys_io.h"
#include "ethercat_map.h"
#include "read_ethercat_data.h"


static int get_nth_bit_in_byte(uint8_t byte, int n)
{
    return (byte >> n) & 1;
}

static WAXIDlrResult waxi_end_access(waxi_dlr_user_t iod) {
    WAXIDlrResult res = waxi_dlr_user_end_access(iod);
    if (WAXI_DL_OK != res)
    {
        LOG_ERROR("waxi_end_access", "Could not end access to input memory '%d'", iod);

    }
    return res;
}

WAXIDlrResult ethercat_read(const waxi_dlr_user_t ether_in_handle,
                           sys_machine_input_t *machine_in,
                           sys_drive_input_t *drive_in, const int drive_pid, const uint32_t revision)
{
    const char *context = "ethercat_read";

    uint8_t machine_bytes[2];
    WAXIDlrResult res = waxi_dlr_user_begin_access(ether_in_handle, revision);
    if (WAXI_DL_OK != res)
    {
        LOG_ERROR(context, "Could not begin access to input memory '%d'",
                  ether_in_handle);
        return res;
    }
    int machine_byte_offset = 10;
    int machine_byte_num = 2;

    res = waxi_dlr_user_read_bytes(ether_in_handle, machine_byte_offset, machine_bytes,
                                   machine_byte_num);

    if (WAXI_DL_OK != res)
    {
        LOG_ERROR(context, "Could not read bytes  from  memory '%d'", ether_in_handle);
        waxi_end_access(ether_in_handle);
        return res;
    }
 
    res = waxi_dlr_user_read_bytes(ether_in_handle, input_drive_map[drive_pid][IN_Dr_POSIW].bitoffset / 8, &drive_in->Dr_PosIW, sizeof(&drive_in->Dr_PosIW));

    if (WAXI_DL_OK != res)
    {
        LOG_ERROR(context, "Could not read bytes  from  memory '%d'", ether_in_handle);
        waxi_end_access(ether_in_handle);
        return res;
    }

    res = waxi_dlr_user_read_bytes(ether_in_handle, input_drive_map[drive_pid][IN_Dr_Status].bitoffset / 8, &drive_in->Dr_Status, 2);
    uint8_t encoder_counter[2] = {0};

    res = waxi_dlr_user_read_bytes(ether_in_handle, input_drive_map[drive_in->pid][IN_DRIVE_COUNT].bitoffset / 8, &drive_in->encoder_counter, 2);
    if (WAXI_DL_OK != res)
    {
        LOG_ERROR(context, "Could not read bytes  from  memory '%d'", ether_in_handle);
        waxi_end_access(ether_in_handle);
        return res;
    }
    
    res = waxi_end_access(ether_in_handle);

    LOG_TRACE(context, "drive_status_word:%d, encoder_counter:%d, machine_byte1:%d machine_byte2:%d", drive_in->Dr_Status, drive_in->encoder_counter, machine_bytes[0], machine_bytes[1]);
    
    if (res == WAXI_DL_OK)  { 
        machine_in->p_31F1OK = get_nth_bit_in_byte(machine_bytes[0], input_machine_map[IN_p31_F1_OK].bitoffset % 8);
        machine_in->p_31F2OK = get_nth_bit_in_byte(machine_bytes[0], input_machine_map[IN_p31_F2_OK].bitoffset % 8);
        machine_in->p_31F4OK = get_nth_bit_in_byte(machine_bytes[0], input_machine_map[IN_p31_F4_OK].bitoffset % 8);
        machine_in->p_BTBA1A2 = get_nth_bit_in_byte(machine_bytes[0], input_machine_map[IN_p_BTBA1A2].bitoffset % 8);
        machine_in->p_P_STEin = get_nth_bit_in_byte(machine_bytes[1], input_machine_map[IN_P_STEin].bitoffset % 8);
        machine_in->p_P_TueZu = get_nth_bit_in_byte(machine_bytes[1], input_machine_map[IN_P_TueZu].bitoffset % 8);
        machine_in->p_P_NABet = get_nth_bit_in_byte(machine_bytes[1], input_machine_map[IN_P_NABet].bitoffset % 8);
    }
    
    return res;
}

WAXIDlrResult ethercat_write(const waxi_dlr_user_t out_handle,
                            const sys_machine_output_t *const machine_out,
                            const sys_drive_output_t *const drive_out, const uint32_t revision)
{
    const char *context = "ethercat_write";
    uint8_t bytes;
    int machine_byte_offset = 10;
    
    int bit_offset = 0;
    int control_byte_offset = output_drive_map[drive_out->pid][OUT_Dr_Control].bitoffset / 8;
    int velocity_byte_offset = output_drive_map[drive_out->pid][OUT_Dr_VelSW].bitoffset / 8;
    //LOG_TRACE(context, "Tried to write drive control : %d to offset: %d with user handle:%d", drive_out->Dr_Control, control_byte_offset, out_handle);
    WAXIDlrResult res = WAXI_DL_OK;

    res = waxi_dlr_user_begin_access(out_handle, revision);
    if (WAXI_DL_OK != res)
    {
        LOG_ERROR(context, "Could not begin access to output memory '%d'",
                  out_handle);
        waxi_end_access(out_handle);
        return res;
    }
    res = waxi_dlr_user_read_bytes(out_handle, machine_byte_offset, &bytes, 1);
    if (WAXI_DL_OK != res)
    {
       LOG_ERROR(context, "Could not read bytes  from  memory '%d'", out_handle);
        waxi_end_access(out_handle);
        return res;
    }
    LOG_INFO(context, "Input machine bytes:%d %0x\n", bytes, bytes);

    // set nth bit in byte
    bit_offset = output_machine_map[OUT_p_190H2].bitoffset % 8;
    bytes = (bytes & ~(1 << bit_offset)) | (machine_out->p_190H2 << bit_offset);

    bit_offset = output_machine_map[OUT_p_190K1].bitoffset % 8;
    bytes = (bytes & ~(1 << bit_offset)) | (machine_out->p_190K1 << bit_offset);

    bit_offset = output_machine_map[OUT_p_PLC_OK].bitoffset % 8;
    bytes = (bytes & ~(1 << bit_offset)) | (machine_out->p_PLC_OK << bit_offset);

    // This causes an error for the Pittsburgh pendulum.
    // It writes to the same byte offset as the encoder on the ethercat bus
    // and we cannot read the encoder data anymore as it gets stuck in a fixed value.
    /* 
    res = waxi_dlr_user_write_bytes(out_handle, machine_byte_offset, &bytes, 1);
    if (WAXI_DL_OK != res)
    {
        LOG_ERROR(context, "Could not write bytes  to  memory '%d'", out_handle);
        waxi_end_access(out_handle);
        return res;
    }
    */

    res = waxi_dlr_user_write_bytes(out_handle, control_byte_offset, &drive_out->Dr_Control, 2);
    if (WAXI_DL_OK != res)
    {
        LOG_ERROR(context, "Could not write bytes  to  memory '%d'", out_handle);
        waxi_end_access(out_handle);
        return res;
    }

    res = waxi_dlr_user_write_bytes(out_handle, velocity_byte_offset, &drive_out->Dr_VelSW, 4);
    if (WAXI_DL_OK != res)
    {
        LOG_ERROR(context, "Could not write bytes  to  memory '%d'", out_handle);
        waxi_end_access(out_handle);
        return res;
    }
    
    res = waxi_end_access(out_handle);
    return res;
}
