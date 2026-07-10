#ifndef ETHERCAT_MAP_H
#define ETHERCAT_MAP_H

/* global imports*/
#include <stdint.h>
#include <stdio.h>


/* WAXI includes */
#include "waxi/datalayer/definitions.h"
#include "waxi/datalayer/user.h"
#include "waxi/datalayer/variant.h"
/* WAXI C SDK */
#include "waxi-c-sdk/datalayer.h"
#include "waxi-c-sdk/logging.h"

#include "sys_io.h"


/* The following data could also be generated or dynamically interpreted by
 * using the memory map*/

/* bitoffsets for inputs and outputs */
struct ethercat_map_t
{
    int bitoffset;
    int bitsize;
};

enum sys_machine_input_id_t
{
    IN_LSTG_AUS = 0,
    IN_LSTG_EIN,
    IN_p31_F1_OK,
    IN_p31_F2_OK,
    IN_p31_F4_OK,
    IN_p_BTBA1A2,
    IN_P_STEin,
    IN_P_NABet,
    IN_P_TueZu,
    IN_P_TueUeb,
    IN_MACHINE_COUNT,
};



static const struct ethercat_map_t input_machine_map[IN_MACHINE_COUNT] = {
    [IN_LSTG_AUS] = {.bitoffset = 81, .bitsize = 1},
    [IN_LSTG_EIN] = {.bitoffset = 80, .bitsize = 1},
    [IN_p31_F1_OK] = {.bitoffset = 84, .bitsize = 1},
    [IN_p31_F2_OK] = {.bitoffset = 85, .bitsize = 1},
    [IN_p31_F4_OK] = {.bitoffset = 86, .bitsize = 1},
    [IN_p_BTBA1A2] = {.bitoffset = 87, .bitsize = 1},
    [IN_P_STEin] = {.bitoffset = 88, .bitsize = 1},
    [IN_P_NABet] = {.bitoffset = 89, .bitsize = 1},
    [IN_P_TueZu] = {.bitoffset = 90, .bitsize = 1},
    [IN_P_TueUeb] = {.bitoffset = 91, .bitsize = 1},
};

enum sys_drive_input_id_t
{
    IN_DG_Ana = 0,
    IN_Dr_Status,
    IN_Dr_POSIW,
    IN_DRIVE_COUNT,
};

// Changed bit offsets in P1 for Pendulum drive controller in Pittsburgh
static const struct ethercat_map_t input_drive_map[2][4] = {
    [P1] = {[IN_DG_Ana] = {.bitoffset = 80, .bitsize = 16},
            [IN_Dr_Status] = {.bitoffset = 96, .bitsize = 16},
            [IN_Dr_POSIW] = {.bitoffset = 112, .bitsize = 32},
            [IN_DRIVE_COUNT] = {.bitoffset = 80, .bitsize = 16}}, // change the bit offset here

    [P2] = {[IN_DG_Ana] = {.bitoffset = 152, .bitsize = 16},
            [IN_Dr_Status] = {.bitoffset = 264, .bitsize = 16},
            [IN_Dr_POSIW] = {.bitoffset = 280, .bitsize = 32},
            [IN_DRIVE_COUNT] = {.bitoffset = 152, .bitsize = 16}},

};

enum sys_machine_output_id_t
{
    OUT_p_190K1 = 0,
    OUT_p_190H2,
    OUT_p_PLC_OK,
    OUT_MACHINE_COUNT,
};

static const struct ethercat_map_t output_machine_map[OUT_MACHINE_COUNT] = {
    [OUT_p_190K1] = {.bitoffset = 82, .bitsize = 1},
    [OUT_p_190H2] = {.bitoffset = 81, .bitsize = 1},
    [OUT_p_PLC_OK] = {.bitoffset = 80, .bitsize = 1},
};

enum sys_drive_output_id_t
{
    OUT_Dr_Control = 0,
    OUT_Dr_PosSW,
    OUT_Dr_VelSW,
    OUT_DRIVE_COUNT,
};

// Changed bit offsets in P1 for Pendulum drive controller in Pittsburgh
static const struct ethercat_map_t output_drive_map[P_COUNT][OUT_DRIVE_COUNT] = {
    [P1] = {[OUT_Dr_Control] = {.bitoffset = 88, .bitsize = 16},
            [OUT_Dr_PosSW] = {.bitoffset = 104, .bitsize = 32},
            [OUT_Dr_VelSW] = {.bitoffset = 136, .bitsize = 32}},
    [P2] = {[OUT_Dr_Control] = {.bitoffset = 224, .bitsize = 16},
            [OUT_Dr_PosSW] = {.bitoffset = 240, .bitsize = 32},
            [OUT_Dr_VelSW] = {.bitoffset = 272, .bitsize = 32}},
};

#endif