#ifndef SYS_IO_H
#define SYS_IO_H

#include <stdbool.h>
#include <stdint.h>


typedef enum pendulum_id_t {
    P1 = 0,
    P2 = 1,
    P_COUNT = 2  // used for arrays of pendulum ids
} pendulum_id_t;

/*
 * Physical IO from and to the machine
 *
 * PNOZ - Pilz Notaus zwangsgeführt
 * see:
 * https://www.all-electronics.de/automatisierung/interview-mit-michael-schlecht-pilz.html
 */

typedef struct sys_machine_input_t sys_machine_input_t;
typedef struct sys_machine_output_t sys_machine_output_t;

struct sys_machine_input_t {
    bool p_31F1OK;   // Sicherung OK 		        E4.4	/  31
    bool p_31F2OK;   // Sicherung OK			    E4.5	/  31
    bool p_31F4OK;   // Sicherung OK			    E4.6	/  31
    bool p_BTBA1A2;  // Antriebe  OK			    E4.7	/ 210

    bool p_P_STEin;  // PNOZ Steuerung ein 	        E5.0	/  80
    bool p_P_NABet;  // PNOZ Notaus betätigt	    E5.1	/  80
    bool p_P_TueZu;  // PNOZ Schutztür geschlossen  E5.2	/  80
};
#define INIT_SYS_MACHINE_INPUT() \
    { 0 }


struct sys_machine_output_t {
    bool p_190K1;   // Drives 230V ein		        A1.0	/ 190
    bool p_190H2;   // Lampe Steuerung ein	        A1.1	/ 190
    bool p_PLC_OK;  // PLC_OK an PNOZ		        A1.2	/ 190
};
#define INIT_SYS_MACHINE_OUTPUT() \
    { 0 }

void sys_reset_machine_ouput(sys_machine_output_t *machine_out);

/*
    Physical IO from and to the drives
*/

typedef struct sys_drive_input_t sys_drive_input_t;
typedef struct sys_drive_output_t sys_drive_output_t;

struct sys_drive_input_t {
    const pendulum_id_t pid;
    int16_t DG_Ana;      // Analog Input Pendel
    uint16_t Dr_Status;  // Drive Statusword
    int32_t Dr_PosIW;    // Drive Position Ist-Wert (m / 1E
    int encoder_counter;
};
#define INIT_SYS_DRIVE_INPUT(pendulum_id) \
    { .pid = pendulum_id }

struct sys_drive_output_t {
    const pendulum_id_t pid;
    uint16_t Dr_Control;  // Drive Controlword
    // int32_t Dr_PosSW;     // Drive Position Soll Wert (not used)
    int32_t Dr_VelSW;  // Drive Velocity Soll Wert (m/s * 1E6)?? (mm/min * 1E3)?
};
#define INIT_SYS_DRIVE_OUTPUT(pendulum_id) \
    { .pid = pendulum_id }

void sys_reset_drive_output(sys_drive_output_t *drive_out);

#endif /* SYS_IO_H */
