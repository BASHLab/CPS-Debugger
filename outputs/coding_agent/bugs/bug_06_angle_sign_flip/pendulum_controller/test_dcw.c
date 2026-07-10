#include "control_word.h"
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include "plclib.h"

static void print_binary(const uint16_t n) {
    if (n > 1) {
        print_binary(n / 2);
    }
    printf("%d", n % 2);
}


void test_pou_drive_control_word() {
    // Test case 1: Input value is 0
    bool STEin = true;
    bool drive_release = true;
    uint16_t dr_status = 57586;

    uint16_t expected1 = 0;
    print_binary(dr_status);
    printf("\n%x\n", dr_status);
    
    uint16_t result1 = pou_drive_control_word(STEin, drive_release, dr_status);
    printf("result1: %d\n", result1);

    CLEAR_BIT(dr_status, 13);
    result1 = pou_drive_control_word(STEin, drive_release, dr_status);
    printf("result1: %d\n", result1);

    dr_status = 25088;
    result1 = pou_drive_control_word(STEin, drive_release, dr_status);
    printf("result1: %d\n", result1);

    drive_release = false;
        result1 = pou_drive_control_word(STEin, drive_release, dr_status);
    printf("result1: %d\n", result1);

    dr_status = 35073;
    result1 = pou_drive_control_word(STEin, drive_release, dr_status);
    printf("result1: %d\n", result1);

    dr_status = 35329;
    drive_release = true;
    print_binary(dr_status);
    printf("\n%x\n", dr_status);
    result1 = pou_drive_control_word(STEin, drive_release, dr_status);
    printf("result1: %d\n", result1);
    print_binary(result1);


    
    
}

int main() {
    test_pou_drive_control_word();
    return 0;
}

