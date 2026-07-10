#ifndef PLCLIB_H
#define PLCLIB_H

#include <stdbool.h>
#include <stdint.h>
#include <time.h>


/* PI as used in the original PLC program */
# define PLC_PI		3.1415926	/* not very precise */

/*
    Bit operations for unsigned integers.
    
    Let V be an unsigned integer of N bits lengths.
    Let I be an integer between 0 and N,  0 <= I < N.
    The following macros can be used to set, clear, get and test
    individual bits within V.
*/

#define SET_BIT(V, I) (V) |= (0x1 << (I))
#define CLEAR_BIT(V, I) (V) &= ~(0x1 << (I))
#define GET_BIT(V, I) ((V) & (0x1 << (I)))
#define TEST_BIT(V, I) (((V) & (0x1 << (I))) != 0)


/*
    Time in PLC is a 32-bit signed integer, aka DINT, in milli seconds
*/

typedef int32_t logicaltime_t;

#define MSEC 1
#define SEC (1000 * MSEC)
#define MIN (60 * SEC)
#define HOUR (60 * MIN)
#define DAY (24 * HOUR)

typedef struct discrete_time_t {
    const logicaltime_t dt;
    logicaltime_t t;
} discrete_time_t;

#define INIT_DISCRETE_TIME(cycletime) \
    { .dt = cycletime }

logicaltime_t now(const discrete_time_t *ctx);
logicaltime_t plc_tick(discrete_time_t *ctx);

/*
    Duration
*/

typedef struct duration_t {
    const logicaltime_t dt;
    logicaltime_t duration;
    logicaltime_t elapsed_time;
} duration_t;

#define INIT_DURATION(cycletime) \
    { .dt = cycletime }

bool after_duration(duration_t *ctx, const logicaltime_t relative_time);
bool before_duration(duration_t *ctx, const logicaltime_t relative_time);

/*
    Delay timer
*/


typedef struct delay_timer_t {
    struct timespec start_time;
    const logicaltime_t dt;  // cycle time
    bool prev_signal;        // previous input
    logicaltime_t et;        // elapsed time
    struct timespec elapsed_time;
} delay_timer_t;

#define INIT_DELAY_TIMER(cycletime) \
    { .dt = cycletime }

//logicaltime_t get_elapsed_delay(delay_timer_t *ctx);
bool turn_on_with_delay(delay_timer_t *ctx, bool signal, logicaltime_t delay);
bool turn_off_with_delay(delay_timer_t *ctx, bool signal, logicaltime_t delay);
bool turn_on_with_delay1(delay_timer_t *ctx, bool signal, struct timespec *delay);

void init_physical_delay_timer(delay_timer_t *tmr);
void reset_physical_delay_timer(delay_timer_t *tmr);

//void reset_delay_timer(delay_timer_t *tmr);

/* PLC function blocks */
#define ton(ctx, in, pt) (turn_on_with_delay(ctx, in, pt))
#define tof(ctx, in, pt) (turn_off_with_delay(ctx, in, pt))

/*
    Trigger
*/

typedef struct edge_trigger_t {
    bool prev_signal;  // previous input
} edge_trigger_t;

#define INIT_EDGE_TRIGGER(is_rising) \
    { .prev_signal = is_rising }

bool falling_edge_trigger(edge_trigger_t *ctx, bool signal);
bool rising_edge_trigger(edge_trigger_t *ctx, bool signal);

/* PLC function blocks */
#define f_trig(ctx, clk) (falling_edge_trigger(ctx, clk))
#define r_trig(ctx, clk) (rising_edge_trigger(ctx, clk))

#endif /* PLCLIB_H */
