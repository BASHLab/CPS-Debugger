"""calibrate_plant.py (C2) -- fit cart-pole plant parameters from real logs.

The SIL controller only balances if the plant model matches the real plant
closely enough that the shipped LQR gains stabilize it. Rather than guess
l_eff / damping, we identify them from the logged trajectories by least
squares on the pendulum equation of motion.

Physics (phi measured from upright, matching the controller's theta=0=upright):
    theta_ddot = (g/l) sin(theta) - (cos(theta)/l) a_cart - b theta_dot
which is LINEAR in the parameters [g/l, 1/l, b]:
    theta_ddot = p0 sin(theta) + p1 (cos(theta) a_cart) + p2 theta_dot
    -> l = -1/p1,  g = p0 * l,  b = -p2.
We fit on BALANCE-phase samples (state==1), where the pendulum is near upright
and the cart is actively driven (rich a_cart for identification). a_cart is the
cart acceleration, obtained by differentiating the logged cart velocity.

Outputs plant_params.json {l_eff, g, b_pend, tau_v, enc_sign} for run_sil.py.
tau_v (velocity-servo time constant) is NOT identifiable from these logs
(the commanded velocity is not logged), so it is set to a fast prior and
flagged; the pendulum dynamics are ~100x slower, so the loop is insensitive
to it within a plausible range.

CPU-only (numpy + pyarrow); no torch. Run inline or via sbatch.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

DATA = "/home/simran/allspark-data-exploration/CPS-Debugger/outputs/fm_candidates/train_data_full"
OUT = Path(__file__).parent / "plant_params.json"
DT = 1.0e-3  # 1 kHz control tick

# Outlier guards: the logged angular_velocity has +-pi-wrap spikes, and the
# finite-difference accelerations blow up at those samples.
MAX_THETA_D = 30.0     # rad/s; balance-phase |theta_d| is far below this
MAX_ACART = 200.0      # m/s^2
MAX_THETA_DD = 3000.0  # rad/s^2


def load_balance_samples(files, max_sessions=None):
    thetas, theta_ds, theta_dds, a_carts = [], [], [], []
    used = 0
    for f in files:
        if max_sessions and used >= max_sessions:
            break
        t = pq.read_table(
            f, columns=["pendulum_state", "velocity", "current_angle",
                        "angular_velocity"]).to_pandas()
        bal = t["pendulum_state"].values == 1.0
        if bal.sum() < 5000:
            continue
        theta = t["current_angle"].values.astype(float)
        theta_d = t["angular_velocity"].values.astype(float)
        v = t["velocity"].values.astype(float)
        # derivatives at the control rate
        theta_dd = np.gradient(theta_d, DT)
        a_cart = np.gradient(v, DT)
        # keep balance-phase, non-wrap, in-range samples
        m = bal.copy()
        m &= np.abs(theta_d) < MAX_THETA_D
        m &= np.abs(a_cart) < MAX_ACART
        m &= np.abs(theta_dd) < MAX_THETA_DD
        m &= np.abs(theta) < 0.8
        thetas.append(theta[m]); theta_ds.append(theta_d[m])
        theta_dds.append(theta_dd[m]); a_carts.append(a_cart[m])
        used += 1
    return (np.concatenate(thetas), np.concatenate(theta_ds),
            np.concatenate(theta_dds), np.concatenate(a_carts))


def fit(theta, theta_d, theta_dd, a_cart):
    # design matrix columns: sin(theta), cos(theta)*a_cart, theta_d
    X = np.column_stack([np.sin(theta), np.cos(theta) * a_cart, theta_d])
    y = theta_dd
    p, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ p
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    g_over_l, neg_inv_l, neg_b = p
    l_eff = -1.0 / neg_inv_l
    g = g_over_l * l_eff
    b = -neg_b
    return dict(l_eff=float(l_eff), g=float(g), b_pend=float(b),
                r2=float(r2), n=int(len(y)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-sessions", type=int, default=8)
    ap.add_argument("--tau-v", type=float, default=0.01)
    args = ap.parse_args()

    files = sorted(glob.glob(f"{DATA}/*.parquet"))
    print(f"[calib] {len(files)} sessions; using up to {args.max_sessions}")
    theta, theta_d, theta_dd, a_cart = load_balance_samples(
        files, args.max_sessions)
    print(f"[calib] {len(theta)} balance-phase samples after filtering")

    res = fit(theta, theta_d, theta_dd, a_cart)
    print(f"[calib] fit R^2 = {res['r2']:.4f} on n={res['n']}")
    print(f"[calib] l_eff = {res['l_eff']:.4f} m   (source hint l_axis=0.20)")
    print(f"[calib] g_est = {res['g']:.4f} m/s^2  (sanity: ~9.81)")
    print(f"[calib] b_pend = {res['b_pend']:.4f} 1/s")

    ok = 0.05 < res["l_eff"] < 1.0 and 5.0 < res["g"] < 15.0
    print(f"[calib] plausibility: {'OK' if ok else 'SUSPECT (check signs/units)'}")

    params = dict(l_eff=res["l_eff"], g=res["g"], b_pend=max(res["b_pend"], 0.0),
                  tau_v=args.tau_v, enc_sign=1, vel_sign=1,
                  _fit_r2=res["r2"], _fit_n=res["n"], _g_est=res["g"])
    OUT.write_text(json.dumps(params, indent=2))
    print(f"[calib] wrote {OUT}")


if __name__ == "__main__":
    main()
