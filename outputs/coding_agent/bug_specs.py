"""Bug specifications for CPS code debugging agent PoC.

Each bug is a single, precisely specified modification to the Pittsburgh
pendulum controller source code. The 'original' and 'buggy' fields contain
exact strings found in the source files.

Source: outputs/phase2/src/src/ (Pittsburgh deployment variant, byte-for-byte
match with datalogs at Pittsburgh-pendulum-datalogs/extracted/*/code/src/).
"""

BUGS = [
    # ── Easy: direct control-flow change visible in trace ────────────
    {
        "id": "bug_01_soft_lqr_threshold",
        "file": "pendulum_controller/lqrsim.c",
        "original": "if (params->counter_goodrange >= 500) {",
        "buggy":    "if (params->counter_goodrange >= 50) {",
        "description": "Soft LQR gains activate after only 50ms instead of 500ms. "
                       "The controller switches to deadzoned gains (K=100,54,274,36) "
                       "almost immediately, losing the initial aggressive stabilisation.",
        "category": "parameter_fault",
        "affected_function": "pou_lqr_sim",
        "affected_edges": ["40:44:85"],
        "difficulty": "easy",
    },
    {
        "id": "bug_02_force_formula_sign",
        "file": "pendulum_controller/lqrsim.c",
        "original": "F_lqr = A1 + A2 + A3 + A4;",
        "buggy":    "F_lqr = A1 + A2 - A3 - A4;",
        "description": "Force formula sign reverted to the original (non-Pittsburgh) "
                       "convention. The Pittsburgh pendulum has reversed angle sign, "
                       "so the original formula produces wrong correction direction.",
        "category": "logic_fault",
        "affected_function": "pou_lqr_sim",
        "affected_edges": ["40:44:85", "40:281:316"],
        "difficulty": "medium",
    },
    {
        "id": "bug_03_balance_check_inverted",
        "file": "pend_mono/pend_monolithic.c",
        "original": "already_balanced = stays_balanced(standup_out.su_pndlum_ok);",
        "buggy":    "already_balanced = stays_balanced(!standup_out.su_pndlum_ok);",
        "description": "Balance confirmation logic inverted. stays_balanced() is fed "
                       "the negation of su_pndlum_ok, so the system never confirms "
                       "balance and never starts setpoint tracking.",
        "category": "logic_fault",
        "affected_function": "tick",
        "affected_edges": ["22:2283:2285", "22:2283:2738"],
        "difficulty": "easy",
    },
    {
        "id": "bug_04_gain_Kv_wrong",
        "file": "pendulum_controller/sys_params.c",
        "original": "    // Pittsburgh specific LQR parameters provided by Marco Giani (CR/APT5)\n    .K_x = 100.0, // 150.0 alternate value\n    .K_v = 79.0,",
        "buggy":    "    // Pittsburgh specific LQR parameters provided by Marco Giani (CR/APT5)\n    .K_x = 100.0, // 150.0 alternate value\n    .K_v = 97.0,",
        "description": "Velocity gain changed from Pittsburgh-tuned 79.0 to the "
                       "original pendulum value 97.0. Over-damps cart velocity, "
                       "causing excessive braking and oscillations.",
        "category": "parameter_fault",
        "affected_function": "pou_lqr_sim",
        "affected_edges": ["40:44:85"],
        "difficulty": "hard",
    },
    {
        "id": "bug_05_fsm_dispatch_swapped",
        "file": "pend_mono/pend_monolithic.c",
        "original": "if (flags.su_enable && !flags.lqr_enable)",
        "buggy":    "if (!flags.su_enable && flags.lqr_enable)",
        "description": "FSM guard conditions swapped. SWINGUP code runs when LQR is "
                       "enabled and vice versa, causing the wrong controller to run "
                       "in each state.",
        "category": "logic_fault",
        "affected_function": "tick",
        "affected_edges": ["33:69:71", "33:118:259", "22:1209:1636"],
        "difficulty": "medium",
    },
    {
        "id": "bug_06_angle_sign_flip",
        "file": "pendulum_controller/general.c",
        "original": "theta_n = (double) encoder_value*2*PGH_PI/PGH_ENCODER_FULL_REV;",
        "buggy":    "theta_n = (double) -encoder_value*2*PGH_PI/PGH_ENCODER_FULL_REV;",
        "description": "Angle computation negated. Controller corrects in the wrong "
                       "direction, immediately destabilising the pendulum.",
        "category": "parameter_fault",
        "affected_function": "pou_general_drive",
        "affected_edges": [],  # indirect: all downstream edges change
        "difficulty": "hard",
    },
    {
        "id": "bug_07_velocity_integration",
        "file": "pendulum_controller/lqrsim.c",
        "original": "lqr_sim_out->SI_v_soll = lqr_sim_out->SI_v_soll_old + lqr_sim_out->LQ_F_soll * (cycletime_secs / (3.9 + params->m_axis));",
        "buggy":    "lqr_sim_out->SI_v_soll = lqr_sim_out->SI_v_soll_old + lqr_sim_out->LQ_F_soll * (cycletime_secs * (3.9 + params->m_axis));",
        "description": "Velocity integration divides by mass but bug multiplies. "
                       "Velocity commands are ~6x too large (mass=2.3+3.9=6.2), "
                       "causing immediate saturation and loss of control.",
        "category": "logic_fault",
        "affected_function": "pou_lqr_sim",
        "affected_edges": ["40:281:316"],
        "difficulty": "hard",
    },
]
