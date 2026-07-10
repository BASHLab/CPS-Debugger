#include <assert.h>
#include <stdbool.h>
/* Debugging */
#include <stdio.h>
#include <time.h>
#include <stdlib.h>


#include "plclib.h"

/*
    Time
*/

logicaltime_t now(const discrete_time_t *ctx) { return ctx->t; }

logicaltime_t plc_tick(discrete_time_t *ctx) {
    ctx->t += ctx->dt;
    return now(ctx);
}

/*
    Duration
*/

bool after_duration(duration_t *ctx, logicaltime_t relative_time) {
    assert(relative_time > 0 * MSEC);
    bool is_after = false;
    if (ctx->duration > 0 * MSEC) {
        ctx->elapsed_time += ctx->dt;
        if (ctx->elapsed_time >= ctx->duration) is_after = true;
    }
    if (is_after) {
        ctx->duration = 0 * MSEC;
        ctx->elapsed_time = 0 * MSEC;
    } else
        ctx->duration = relative_time;

    return is_after;
}

bool before_duration(duration_t *ctx, logicaltime_t relative_time) {
    assert(relative_time > 0 * MSEC);
    bool is_before = true;
    if (ctx->duration > 0 * MSEC) {
        ctx->elapsed_time += ctx->dt;
        if (ctx->elapsed_time >= ctx->duration) is_before = false;
    }
    if (!is_before) {
        ctx->duration = 0 * MSEC;
        ctx->elapsed_time = 0 * MSEC;
    } else
        ctx->duration = relative_time;

    return is_before;
}

/*
    Delay timer
*/

//logicaltime_t get_elapsed_delay(delay_timer_t *ctx) { return ctx->et; }

bool turn_on_with_delay(delay_timer_t *ctx, bool signal, logicaltime_t delay) {
    assert(delay >= 0 * MSEC);
    bool out = false;
    if (!ctx->prev_signal && signal) {  // first step (rising edge)
        if (ctx->et >= delay) out = true;
    } else if (ctx->prev_signal && signal) {  // further steps, turn on delay
        if (ctx->et < delay) ctx->et += ctx->dt;
        if (ctx->et >= delay) out = true;
    } else if (ctx->prev_signal && !signal) {  // falling edge
        //printf("et set to ZERO");
        ctx->et = 0 * MSEC;
    }
    //printf("# 3prev_signal=%d signal=%d dt=%d et=%d\n", ctx->prev_signal, signal, ctx->dt, ctx->et);
    ctx->prev_signal = signal;
    return out;
}
/*Returns true if time2 > time1*/
bool exceeds_time (struct timespec *time1, struct timespec *time2) {
    if (time2->tv_sec < time1->tv_sec) return false;
    if (time2->tv_sec > time1->tv_sec) return true; 
    if (time2->tv_nsec > time1->tv_nsec) return true; //differ only in nsecs
    
    return false;
}

static void get_time_diff(struct timespec *start, struct timespec *end, struct timespec *diff){
    
    if (end->tv_nsec > start->tv_nsec) {
        diff->tv_nsec = end->tv_nsec - start->tv_nsec;
        diff->tv_sec = end->tv_sec -start->tv_sec;
    } else {
        diff->tv_nsec = end->tv_nsec + 1000000000 - start->tv_nsec;
        diff->tv_sec = end->tv_sec - 1  - start->tv_sec;
    }
    return;
}

bool turn_on_with_delay1(delay_timer_t *ctx, bool signal, struct timespec *delay) {
    //assert(delay >= 0 * MSEC);
    struct timespec now;
    assert ((delay->tv_nsec >= 0) || (delay->tv_sec >= 0));
    bool out = false;
    if (!ctx->prev_signal && signal) {  // first step (rising edge)
        //if (ctx->et >= delay) out = true;
         clock_gettime(CLOCK_MONOTONIC, &(ctx->start_time)); //start the timer
         //ctx->elapsed_time->tv_sec = 0; 
         //ctx->elapsed_time->tv_nsec = 0;
         //printf("Starting timer: now\n");
         //if (et >delay)
        if(!exceeds_time(&(ctx->elapsed_time), delay)) out = true;
    } else if (ctx->prev_signal && signal) {  // further steps, turn on delay
        
        //if delay > elapsed time
        if (exceeds_time(&(ctx->elapsed_time), delay)) {
            clock_gettime(CLOCK_MONOTONIC, &now);
            get_time_diff(&(ctx->start_time), &now, &(ctx->elapsed_time));
            //printf("Elapsed time:%ld delay: %lld secs %ld nsecs\n", ctx->elapsed_time->tv_nsec/1000000, delay->tv_sec, delay->tv_nsec/1000000);
        }
        //if delay less than elapsed time
        if(!exceeds_time(&(ctx->elapsed_time), delay)) out = true;
        //if (ctx->et >= delay) out = true;
    } else if (ctx->prev_signal && !signal) {  // falling edge
        //ctx->et = 0 * MSEC;
        ctx->elapsed_time.tv_nsec = 0;
        ctx->elapsed_time.tv_sec = 0;
        
    }
    //printf("# 3prev_signal=%d signal=%d dt=%d et=%d\n", ctx->prev_signal, signal, ctx->dt, ctx->et);
    ctx->prev_signal = signal;
    return out;
}

void init_physical_delay_timer(delay_timer_t *tmr) {
     
     //tmr->start_time = (struct timespec *)malloc (sizeof(struct timespec));
     //tmr->elapsed_time = (struct timespec *)malloc (sizeof(struct timespec));
     
     tmr->start_time.tv_nsec  = 0;
     tmr->start_time.tv_sec  = 0;
     
     tmr->elapsed_time.tv_sec = 0;
     tmr->elapsed_time.tv_nsec = 0;
}


void reset_physical_delay_timer(delay_timer_t *tmr){
    //free(tmr->start_time);
    //free(tmr->elapsed_time);

}



bool turn_off_with_delay(delay_timer_t *ctx, bool signal, logicaltime_t delay) {
    assert(delay >= 0 * MSEC);
    bool out = true;
    if (ctx->prev_signal && !signal) {  // first step (falling edge)
        if (ctx->et >= delay) out = false;
    } else if (!ctx->prev_signal && !signal) {  // further steps, turn off delay
        if (ctx->et < delay) ctx->et += ctx->dt;
        if (ctx->et >= delay) out = false;
    } else if (!ctx->prev_signal && signal) {  // rising edge
        ctx->et = 0 * MSEC;
    }
    ctx->prev_signal = signal;
    return out;
}

/*
    Trigger
*/

bool falling_edge_trigger(edge_trigger_t *ctx, bool signal) {
    bool falling_edge = false;
    if (ctx->prev_signal && !signal)
        falling_edge = true;
    else
        falling_edge = false;
    ctx->prev_signal = signal;
    return falling_edge;
}

bool rising_edge_trigger(edge_trigger_t *ctx, bool signal) {
    bool rising_edge = false;
    if (!ctx->prev_signal && signal)
        rising_edge = true;
    else
        rising_edge = false;

    ctx->prev_signal = signal;
    return rising_edge;
}