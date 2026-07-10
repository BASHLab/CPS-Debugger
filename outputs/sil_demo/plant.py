"""plant.py — cart-pole physics + sensor/actuator codec for the SIL loop.

The Bosch controller drives a VELOCITY-SERVO cart (it writes a cart velocity
setpoint Dr_VelSW, not a force), reads a 13-bit absolute pendulum encoder and
a 1e-7 m cart-position word, and computes theta / x / velocities itself. This
module reproduces exactly that byte-level contract (verified against
src/pend_mono + src/pendulum_controller) so the controller sees training-data
quantization, and integrates the plant physics between ticks.

Contract (source-grounded):
  INPUT  region (realtime_data/input, 64 B):
    encoder int16 @10  (13-bit, mask 0x1FFF, 8192 counts/rev)
    Dr_Status uint16 @12  (bit15 AEin, bit14 AF, bit13 error)
    Dr_PosIW  int32  @14  (cart position, units 1e-7 m)
  OUTPUT region (realtime_data/output, 64 B):
    Dr_Control uint16 @11 (drive control word; not needed by the plant)
    Dr_VelSW   int32  @17 (= velocity_setpoint[m/s] * 6.0e7)

Angle convention: controller theta is 0 at upright, +-pi hanging down, with a
down-position offset the controller self-calibrates on the first swing-up tick.
We fix the down raw count at 0; the forward encoder map is the exact inverse of
the controller decode, so theta round-trips regardless of that offset.

The encoder-count direction and the Dr_VelSW velocity direction are genuine
calibration unknowns (which way the physical encoder counts, which way a
positive setpoint moves the cart). They are exposed as enc_sign / vel_sign and
fit by calibrate_plant.py; the SIL smoke test scans all four combinations.
"""
import math
import struct
from dataclasses import dataclass, field

TWO_PI = 2.0 * math.pi
COUNTS_PER_REV = 8192
ENC_MASK = 0x1FFF
COUNTS_PER_RAD = COUNTS_PER_REV / TWO_PI       # 1303.80
POS_COUNTS_PER_M = 1.0e7                        # Dr_PosIW = round(x * 1e7)
VELSW_SCALE = 6.0e7                             # Dr_VelSW = v * 60000 * 1000
STATUS_WORD = 0xC000                            # AEin | AF, no error (probe value)

# EtherCAT byte offsets (source: ethercat_map.h P1 maps).
OFF_ENCODER = 10
OFF_STATUS = 12
OFF_POSIW = 14
OFF_VELSW = 17


@dataclass
class PlantParams:
    """Physical parameters from the Bosch PendulumPhysicalModel (data_init.m),
    identified against real testbench natural-swing measurements. The heavy,
    nonlinear friction is essential: a near-frictionless plant does not bleed
    swing-up energy, so the pendulum whips through vertical and balance never
    captures (the failure mode observed with the first-principles model)."""
    l_eff: float = 0.2685       # L: center of rotation to COM (m)
    m_pend: float = 2.3         # pendulum mass (kg)
    g: float = 9.81
    # Friction (rotary torque model): viscous + near-zero extra + Coulomb.
    b_visc: float = 0.177       # viscous friction torque coeff (N m s/rad)
    b_extra: float = 1.5        # extra friction active at |theta_d| <= 0.3
    coulomb: float = 0.10       # Coulomb friction torque (units uncertain in
                                # the Simulink block; tunable, start modest)
    tau_v: float = 0.0          # 0 = stiff velocity servo (drive tracks setpt)
    enc_sign: int = 1           # encoder count direction (+-1)
    vel_sign: int = 1           # Dr_VelSW -> cart-velocity direction (+-1)
    pos_sign: int = 1           # Dr_PosIW cart-position direction (+-1)
    v_max: float = 1.0          # cart velocity saturation (m/s); data_init vmax
    a_max: float = 0.0          # cart acceleration limit (m/s^2); 0 = unlimited
    dt_control: float = 1.0e-3  # controller tick period (s)
    substeps: int = 8           # RK4 substeps per tick (finer for stiff friction)


@dataclass
class PlantState:
    x: float = 0.0              # cart position (m)
    v: float = 0.0              # cart velocity (m/s)
    phi: float = math.pi        # pendulum angle from upright (rad); pi = down
    phidot: float = 0.0         # angular velocity (rad/s)
    v_cmd: float = 0.0          # last commanded cart velocity (m/s, signed phys)


class CartPole:
    def __init__(self, params: PlantParams = None, state: PlantState = None):
        self.p = params or PlantParams()
        self.s = state or PlantState()

    # ── physics ──────────────────────────────────────────────────────────────
    def _fric_accel(self, phidot):
        """Friction angular deceleration = friction torque / inertia.
        Physical-pendulum inertia I = m_pend * L^2 (COM at L). Torque model:
        viscous + extra-near-zero (Stribeck-like) + smooth Coulomb."""
        p = self.p
        inertia = p.m_pend * p.l_eff * p.l_eff
        b = p.b_visc + (p.b_extra if abs(phidot) <= 0.3 else 0.0)
        torque = b * phidot + p.coulomb * math.tanh(phidot / 0.05)
        return torque / inertia

    def _phiddot(self, phi, phidot, a_cart):
        p = self.p
        return (p.g / p.l_eff) * math.sin(phi) \
            - (math.cos(phi) / p.l_eff) * a_cart \
            - self._fric_accel(phidot)

    def _deriv(self, x, v, phi, phidot, v_cmd):
        p = self.p
        vdot = (v_cmd - v) / p.tau_v            # velocity servo
        a_cart = vdot
        # phi measured from upright: upright (phi=0) is the unstable equilibrium.
        phiddot = self._phiddot(phi, phidot, a_cart)
        return v, vdot, phidot, phiddot

    def step(self, v_cmd_phys: float):
        """Advance one control tick under a held velocity setpoint.

        With tau_v <= 0 the drive is modeled as a STIFF velocity servo: the
        cart velocity tracks the setpoint exactly within the tick, so the
        cart acceleration is a_cart = (v_cmd - v)/dt held constant over the
        tick. This is the limit the shipped LQR assumes (it integrates force
        into a velocity setpoint, v += F*dt/m, expecting the drive to realize
        that velocity), so it is the correct default. tau_v > 0 keeps the
        first-order-servo model for sensitivity studies.
        """
        p = self.p
        s = self.s
        s.v_cmd = max(-p.v_max, min(p.v_max, v_cmd_phys))
        if p.tau_v <= 0.0:
            self._step_stiff()
            return s
        h = p.dt_control / p.substeps
        for _ in range(p.substeps):
            x, v, ph, pd = s.x, s.v, s.phi, s.phidot
            k1 = self._deriv(x, v, ph, pd, s.v_cmd)
            k2 = self._deriv(x + 0.5 * h * k1[0], v + 0.5 * h * k1[1],
                             ph + 0.5 * h * k1[2], pd + 0.5 * h * k1[3], s.v_cmd)
            k3 = self._deriv(x + 0.5 * h * k2[0], v + 0.5 * h * k2[1],
                             ph + 0.5 * h * k2[2], pd + 0.5 * h * k2[3], s.v_cmd)
            k4 = self._deriv(x + h * k3[0], v + h * k3[1],
                             ph + h * k3[2], pd + h * k3[3], s.v_cmd)
            s.x += (h / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            s.v += (h / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
            s.phi += (h / 6.0) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
            s.phidot += (h / 6.0) * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
        # keep phi in (-pi, pi]
        s.phi = (s.phi + math.pi) % TWO_PI - math.pi
        return s

    def _pend_deriv(self, phi, phidot, a_cart):
        return phidot, self._phiddot(phi, phidot, a_cart)

    def _step_stiff(self):
        """Stiff velocity servo: cart velocity reaches v_cmd this tick;
        constant cart acceleration a_cart = (v_cmd - v)/dt over the tick."""
        p = self.p
        s = self.s
        a_cart = (s.v_cmd - s.v) / p.dt_control
        if p.a_max > 0.0:                      # drive torque/accel limit
            a_cart = max(-p.a_max, min(p.a_max, a_cart))
        v_end = s.v + a_cart * p.dt_control     # may not reach v_cmd if limited
        h = p.dt_control / p.substeps
        for _ in range(p.substeps):
            ph, pd = s.phi, s.phidot
            k1 = self._pend_deriv(ph, pd, a_cart)
            k2 = self._pend_deriv(ph + 0.5 * h * k1[0], pd + 0.5 * h * k1[1], a_cart)
            k3 = self._pend_deriv(ph + 0.5 * h * k2[0], pd + 0.5 * h * k2[1], a_cart)
            k4 = self._pend_deriv(ph + h * k3[0], pd + h * k3[1], a_cart)
            s.phi += (h / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            s.phidot += (h / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        s.x += 0.5 * (s.v + v_end) * p.dt_control  # velocity ramp
        s.v = v_end
        s.phi = (s.phi + math.pi) % TWO_PI - math.pi

    # ── sensor synthesis (plant state -> EtherCAT input image) ───────────────
    def encoder_count(self) -> int:
        theta = self.p.enc_sign * self.s.phi
        masked = (round(theta * COUNTS_PER_RAD) + COUNTS_PER_REV // 2) & ENC_MASK
        return masked

    def input_image(self) -> bytes:
        buf = bytearray(64)
        # Write the raw 16-bit count; the controller masks the low 13 bits.
        struct.pack_into("<H", buf, OFF_ENCODER, self.encoder_count())
        struct.pack_into("<H", buf, OFF_STATUS, STATUS_WORD)
        struct.pack_into("<i", buf, OFF_POSIW,
                         int(round(self.p.pos_sign * self.s.x * POS_COUNTS_PER_M)))
        return bytes(buf)

    # ── actuator decode (EtherCAT output image -> commanded velocity) ────────
    def decode_velsw(self, out_image: bytes) -> float:
        (raw,) = struct.unpack_from("<i", out_image, OFF_VELSW)
        return self.p.vel_sign * (raw / VELSW_SCALE)

    # ── ground-truth logged channels (what the controller would compute) ─────
    def logged_theta(self) -> float:
        return self.p.enc_sign * self.s.phi

    def logged_channels(self) -> dict:
        return {
            "position": self.s.x,
            "velocity": self.s.v,
            "angle": self.logged_theta(),
            "angular_velocity": self.p.enc_sign * self.s.phidot,
        }
