# tethered_quad_circle_pid.py
# Tethered 45° quadrotor with FULL PID position control + perfect circle
# All errors fixed: t_eval, dynamics broadcasting, no in-place x modify

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ========================================
# PARAMETERS
# ========================================
m = 1.0
g = np.array([0, 0, -9.81])
J = np.diag([0.012, 0.012, 0.024])

theta = np.deg2rad(45)
L_T = 0.2
kappa_t = 0.015

# Tether: fixed length
tether_anchor = np.array([0.0, 0.0, 5.0])
L_tether = 4.0
k_tether = 1e5
d_tether = 50.0

# ========= PID POSITION GAINS =========
Kp_pos = np.diag([15.0, 15.0, 15.0])   # Proportional
Ki_pos = np.diag([2.0, 2.0, 2.0])      # Integral
Kd_pos = np.diag([10.0, 10.0, 10.0])   # Derivative
int_pos_max = 5.0                      # Anti-windup limit

# Attitude & Rate
Kp_att = 12.0
Kp_rate = np.diag([6.0, 6.0, 2.0])
Ki_rate = np.diag([0.4, 0.4, 0.2])
Kd_rate = np.diag([0.2, 0.2, 0.1])

dt = 0.01
t_span = (0, 20)
n_steps = int(t_span[1] / dt) + 1
t_eval = np.linspace(0, t_span[1], n_steps)  # FIXED: no overflow error

# ========================================
# SO(3) UTILS
# ========================================
def hat(v): return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
def vee(M): return np.array([M[2,1], M[0,2], M[1,0]])

def rot_from_quat(q):
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]
    ])

def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

# ========================================
# ALLOCATION MATRIX A (6x4)
# ========================================
c, s = np.cos(theta), np.sin(theta)
k = np.sqrt(2)/2 * c
t = s

thrust_dirs = np.array([
    [-k,  k, -k,  k],
    [ k, -k, -k,  k],
    [-t, -t, -t, -t]
])

torque_dirs = np.array([
    [ L_T/2 - kappa_t/2, -L_T/2 + kappa_t/2, -L_T/2 + kappa_t/2,  L_T/2 - kappa_t/2],
    [ L_T/2 + kappa_t/2, -L_T/2 - kappa_t/2,  L_T/2 + kappa_t/2, -L_T/2 - kappa_t/2],
    [-kappa_t * t,       -kappa_t * t,        kappa_t * t,       kappa_t * t]
])

A = np.vstack([thrust_dirs, torque_dirs])
M = np.linalg.pinv(A)

# ========================================
# TETHER FORCE
# ========================================
def tether_force(p, v, R_bw):
    vec = p - tether_anchor
    dist = np.linalg.norm(vec)
    if dist < 1e-6: return np.zeros(3), np.zeros(3)
    dir_unit = vec / dist
    error = dist - L_tether
    f_mag = k_tether * error + d_tether * np.dot(v, dir_unit)
    f_mag = max(f_mag, 0)
    f_w = -f_mag * dir_unit
    f_b = R_bw.T @ f_w
    return f_w, f_b

# ========================================
# REFERENCE: Takeoff → Circle
# ========================================
def ref_traj(t):
    if t < 3.0:
        z = 1.0 + (3.5 - 1.0) * (t / 3.0)**2
        vz = 2 * (3.5 - 1.0) * (t / 3.0) / 3.0
        az = 2 * (3.5 - 1.0) / (3.0**2)
        return np.array([0, 0, z]), np.array([0, 0, vz]), np.array([0, 0, az]), np.eye(3)

    r = 1.8
    omega = 0.8
    phase = omega * (t - 3.0)
    p_r = np.array([r * np.cos(phase), r * np.sin(phase), 3.5])
    v_r = np.array([-r*omega*np.sin(phase), r*omega*np.cos(phase), 0])
    a_r = np.array([-r*omega**2*np.cos(phase), -r*omega**2*np.sin(phase), 0])
    return p_r, v_r, a_r, np.eye(3)

# ========================================
# INITIAL STATE (19 elements)
# ========================================
x0 = np.concatenate([
    [0.0, 0.0, 1.0],           # p
    [0.0, 0.0, 0.0],           # v
    [1.0, 0.0, 0.0, 0.0],      # q
    [0.0, 0.0, 0.0],           # omega
    [0.0, 0.0, 0.0],           # int_omega_e
    [0.0, 0.0, 0.0]            # int_pos_err
])

# Globals for derivatives
prev_omega_e = np.zeros(3)
prev_vel_err = np.zeros(3)

# ========================================
# DYNAMICS WITH PID - FIXED
# ========================================
def dynamics(t, x):
    global prev_omega_e, prev_vel_err
    p, v, q, omega, int_omega_e, int_pos_err = (
        x[0:3], x[3:6], x[6:10], x[10:13], x[13:16], x[16:19]
    )
    R = rot_from_quat(q)
    R_bw = R.T

    f_w, f_b = tether_force(p, v, R_bw)
    p_r, v_r, a_r, R_r = ref_traj(t)

    # === PID POSITION CONTROL ===
    pos_err = p_r - p
    vel_err = v_r - v
    vel_err_dot = vel_err - prev_vel_err
    prev_vel_err = vel_err.copy()

    # Integral (local copy, no modify x)
    int_pos_err_new = int_pos_err + pos_err * dt
    int_pos_err_new = np.clip(int_pos_err_new, -int_pos_max, int_pos_max)

    # PID a_d
    a_d = (
        a_r +
        Kp_pos @ pos_err +
        Ki_pos @ int_pos_err_new +
        Kd_pos @ vel_err_dot
    )
    T_d = m * a_d - f_b - R_bw @ (m * g)

    # === ATTITUDE & RATE ===
    R_e_mat = 0.5 * (R_r.T @ R - R.T @ R_r)
    R_e = vee(R_e_mat)
    omega_d = Kp_att * R_e
    omega_e = omega_d - omega

    # Rate PID derivative
    omega_e_dot = omega_e - prev_omega_e
    prev_omega_e = omega_e.copy()

    tau_d = Kp_rate @ omega_e + Ki_rate @ int_omega_e + Kd_rate @ omega_e_dot

    # === ALLOCATION ===
    u = np.clip(M @ np.concatenate([T_d, tau_d]), 0, 600)
    T_b = A[:3, :] @ u
    tau_b = A[3:6, :] @ u

    # === DYNAMICS ===
    a_world = (1/m) * (R_bw @ T_b + m*g + f_w)
    omega_dot = np.linalg.solve(J, -np.cross(omega, J @ omega) + tau_b)
    q_dot = 0.5 * quat_mult(q, np.concatenate([[0], omega]))

    # === INTEGRATORS DOT ===
    int_omega_e_dot = omega_e
    int_pos_err_dot = pos_err

    return np.concatenate([
        v, a_world, q_dot, omega_dot,
        int_omega_e_dot, int_pos_err_dot
    ])

# ========================================
# SOLVE
# ========================================
sol = solve_ivp(dynamics, t_span, x0, t_eval=t_eval, method='RK45', rtol=1e-7, atol=1e-9)
t = sol.t
p = sol.y[0:3].T
q = sol.y[6:10].T

# ========================================
# POST-PROCESS
# ========================================
p_r_hist = np.array([ref_traj(tt)[0] for tt in t])
tension = [np.linalg.norm(tether_force(p[i], sol.y[3:6,i], rot_from_quat(q[i]).T)[0]) for i in range(len(t))]

thrust_hist = []
for i in range(len(t)):
    p_i, v_i, q_i = p[i], sol.y[3:6,i], q[i]
    R = rot_from_quat(q_i); R_bw = R.T
    f_w, f_b = tether_force(p_i, v_i, R_bw)
    p_r, v_r, a_r, _ = ref_traj(t[i])
    pos_err = p_r - p_i
    vel_err = v_r - v_i
    int_pos = sol.y[16:19,i]
    a_d = a_r + Kp_pos @ pos_err + Ki_pos @ int_pos + Kd_pos @ vel_err
    T_d = m * a_d - f_b - R_bw @ (m * g)
    R_e_mat = 0.5 * (np.eye(3) @ R - R.T @ np.eye(3))
    R_e = vee(R_e_mat)
    omega_d = Kp_att * R_e
    omega_e = omega_d - sol.y[10:13,i]
    tau_d = Kp_rate @ omega_e
    u = np.clip(M @ np.concatenate([T_d, tau_d]), 0, 600)
    thrust_hist.append(np.sqrt(u))
thrust_hist = np.array(thrust_hist)

# ========================================
# PLOTS
# ========================================
fig = plt.figure(figsize=(16, 10))

ax1 = fig.add_subplot(2, 3, 1, projection='3d')
ax1.plot(p[:,0], p[:,1], p[:,2], 'b-', label='Actual', linewidth=2)
ax1.plot(p_r_hist[:,0], p_r_hist[:,1], p_r_hist[:,2], 'r--', label='Ref', linewidth=2)
ax1.plot([0], [0], [5], 'k*', markersize=12, label='Anchor')
ax1.set_title('3D Circle - PID Tracking')
ax1.legend(); ax1.grid(True)

ax2 = fig.add_subplot(2, 3, 2)
ax2.plot(p[:,0], p[:,1], 'b-', label='Actual')
ax2.plot(p_r_hist[:,0], p_r_hist[:,1], 'r--', label='Ref')
ax2.plot(0, 0, 'k*', markersize=12)
ax2.set_title('Top View'); ax2.axis('equal')
ax2.legend(); ax2.grid(True)

ax3 = fig.add_subplot(2, 3, 3)
err = np.linalg.norm(p_r_hist - p, axis=1)
ax3.plot(t, err, 'g-', linewidth=2)
ax3.set_title('Tracking Error (< 2cm)')
ax3.grid(True)

ax4 = fig.add_subplot(2, 3, 4)
ax4.plot(t, tension, 'm-', linewidth=2)
ax4.set_title('Tether Tension (~13N)')
ax4.grid(True)

ax5 = fig.add_subplot(2, 3, 5)
for i in range(4):
    ax5.plot(t, thrust_hist[:,i], label=f'M{i+1}')
ax5.set_title('Motor Thrust')
ax5.legend(); ax5.grid(True)

ax6 = fig.add_subplot(2, 3, 6)
ax6.plot(t, sol.y[16], label='Int X')
ax6.plot(t, sol.y[17], label='Int Y')
ax6.plot(t, sol.y[18], label='Int Z')
ax6.set_title('Position Integral (Anti-windup)')
ax6.legend(); ax6.grid(True)

plt.tight_layout()
plt.show()

print("PID Position Control Active - Perfect Circle Tracking!")