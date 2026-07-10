#ifndef CONTROL_WORD_H
#define CONTROL_WORD_H

#include <stdint.h>
#include "general.h"
#include "sys_io.h"

void reset_drive(sys_drive_output_t *sys_drive_out);

void pou_drive_control_word(const bool STEin,  const bool drive_release,
              const uint16_t dr_status, uint16_t *dr_control_word);


#endif /*CONTROL_WORD_H*/