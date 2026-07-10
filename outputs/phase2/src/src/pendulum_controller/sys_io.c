
/* self include */
#include "sys_io.h"

void sys_reset_machine_ouput(sys_machine_output_t *machine_out) {
    machine_out->p_190H2 = false;
    machine_out->p_190K1 = false;
    machine_out->p_PLC_OK = false;
}

void sys_reset_drive_output(sys_drive_output_t *drive_out) {
    drive_out->Dr_Control = 0u;
    drive_out->Dr_VelSW = 0;
}