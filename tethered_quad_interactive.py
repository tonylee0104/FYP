# tethered_quad_interactive.py
# Tethered 45° quadrotor with PID + real-time animation + INTERACTIVE SLIDERS
# All errors fixed: global ani, line_actual, line_ref

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

# ========================================
# PARAMETERS (will be updated by sliders)
# ========================================
m = 1.0
g = np.array([0, 0, -9.81])
J = np.diag([0.012, 0.012, 0.024])
theta = np.deg2rad(45)
L_T = 0.2
kappa_t = 0.015
tether_anchor = np.array([0.0, 0.0, 5.0])
L_tether = 4.0
k_tether = 1e5
d_tether = 50.0
Kp_att = 12.0
Kp_rate = np.diag([6.0, 6.0, 2.0])
Ki_rate = np.diag([0.4, 0.4, 0.2])
Kd_rate = np.diag([0.2, 0.2, 0.1])
int_pos_max = 5.0

# Initial values (will be updated by sliders)
Kp_pos_val = 15.0
Ki_pos_val = 2.0
Kd_pos_val = 10.0
circle_radius = 1.8
circle_omega = 0.8

dt = 0.01
t_span = (0, 20)
t_eval = np.linspace(0, t_span[1], int(t_span[1]/dt) + 1)

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
# REFERENCE TRAJECTORY (depends on radius/omega)
# ========================================
def ref_traj(t, r, omega):
    if t < 3.0:
        z = 1.0 + (3.5 - 1.0) * (t / 3.0)**2
        vz = 2 * (3.5 - 1.0) * (t / 3.0) / 3.0
        az = 2 * (3.5 - 1.0) / (3.0**2)
        return np.array([0, 0, z]), np.array([0, 0, vz]), np.array([0, 0, az]), np.eye(3)
    phase = omega * (t - 3.0)
    p_r = np.array([r * np.cos(phase), r * np.sin(phase), 3.5])
    v_r = np.array([-r*omega*np.sin(phase), r*omega*np.cos(phase), 0])
    a_r = np.array([-r*omega**2*np.cos(phase), -r*omega**2*np.sin(phase), 0])
    return p_r, v_r, a_r, np.eye(3)

# ========================================
# INITIAL STATE
# ========================================
def get_initial_state():
    return np.concatenate([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ])

# ========================================
# DYNAMICS (depends on gains)
# ========================================
def dynamics_factory(Kp_pos_val, Ki_pos_val, Kd_pos_val, circle_radius, circle_omega):
    Kp_pos = np.diag([Kp_pos_val]*3)
    Ki_pos = np.diag([Ki_pos_val]*3)
    Kd_pos = np.diag([Kd_pos_val]*3)

    prev_omega_e = np.zeros(3)
    prev_vel_err = np.zeros(3)

    def dynamics(t, x):
        nonlocal prev_omega_e, prev_vel_err
        p, v, q, omega, int_omega_e, int_pos_err = (
            x[0:3], x[3:6], x[6:10], x[10:13], x[13:16], x[16:19]
        )
        R = rot_from_quat(q)
        R_bw = R.T

        f_w, f_b = tether_force(p, v, R_bw)
        p_r, v_r, a_r, R_r = ref_traj(t, circle_radius, circle_omega)

        pos_err = p_r - p
        vel_err = v_r - v
        vel_err_dot = vel_err - prev_vel_err
        prev_vel_err = vel_err.copy()

        int_pos_err_new = int_pos_err + pos_err * dt
        int_pos_err_new = np.clip(int_pos_err_new, -int_pos_max, int_pos_max)

        a_d = a_r + Kp_pos @ pos_err + Ki_pos @ int_pos_err_new + Kd_pos @ vel_err_dot
        T_d = m * a_d - f_b - R_bw @ (m * g)

        R_e_mat = 0.5 * (R_r.T @ R - R.T @ R_r)
        R_e = vee(R_e_mat)
        omega_d = Kp_att * R_e
        omega_e = omega_d - omega

        omega_e_dot = omega_e - prev_omega_e
        prev_omega_e = omega_e.copy()

        tau_d = Kp_rate @ omega_e + Ki_rate @ int_omega_e + Kd_rate @ omega_e_dot

        u = np.clip(M @ np.concatenate([T_d, tau_d]), 0, 600)
        T_b = A[:3, :] @ u
        tau_b = A[3:6, :] @ u

        a_world = (1/m) * (R_bw @ T_b + m*g + f_w)
        omega_dot = np.linalg.solve(J, -np.cross(omega, J @ omega) + tau_b)
        q_dot = 0.5 * quat_mult(q, np.concatenate([[0], omega]))

        return np.concatenate([
            v, a_world, q_dot, omega_dot,
            omega_e, pos_err
        ])

    return dynamics

# ========================================
# GLOBAL ANIMATION OBJECTS (Fixed: initialized before use)
# ========================================
line_actual = None
line_ref = None
ani = None

# ========================================
# SIMULATION & ANIMATION
# ========================================
fig = plt.figure(figsize=(16, 10))
ax_3d = fig.add_subplot(2, 3, 1, projection='3d')
ax_top = fig.add_subplot(2, 3, 2)
ax_err = fig.add_subplot(2, 3, 3)
ax_ten = fig.add_subplot(2, 3, 4)
ax_thr = fig.add_subplot(2, 3, 5)
ax_int = fig.add_subplot(2, 3, 6)

def run_simulation():
    global line_actual, line_ref, ani  # ← Global at TOP

    # Stop previous animation
    if ani is not None:
        ani.event_source.stop()

    # Clear all axes
    ax_3d.clear()
    ax_top.clear()
    ax_err.clear()
    ax_ten.clear()
    ax_thr.clear()
    ax_int.clear()

    # Re-run dynamics
    dynamics = dynamics_factory(Kp_pos_val, Ki_pos_val, Kd_pos_val, circle_radius, circle_omega)
    sol = solve_ivp(dynamics, t_span, get_initial_state(), t_eval=t_eval, method='RK45', rtol=1e-7, atol=1e-9)
    t = sol.t
    p = sol.y[0:3].T
    q = sol.y[6:10].T

    p_r_hist = np.array([ref_traj(tt, circle_radius, circle_omega)[0] for tt in t])
    tension = [np.linalg.norm(tether_force(p[i], sol.y[3:6,i], rot_from_quat(q[i]).T)[0]) for i in range(len(t))]

    thrust_hist = []
    for i in range(len(t)):
        p_i, v_i, q_i = p[i], sol.y[3:6,i], q[i]
        R = rot_from_quat(q_i); R_bw = R.T
        f_w, f_b = tether_force(p_i, v_i, R_bw)
        p_r, v_r, a_r, _ = ref_traj(t[i], circle_radius, circle_omega)
        pos_err = p_r - p_i
        vel_err = v_r - v_i
        int_pos = sol.y[16:19,i]
        a_d = a_r + np.diag([Kp_pos_val]*3) @ pos_err + np.diag([Ki_pos_val]*3) @ int_pos + np.diag([Kd_pos_val]*3) @ vel_err
        T_d = m * a_d - f_b - R_bw @ (m * g)
        u = np.clip(M @ np.concatenate([T_d, np.zeros(3)]), 0, 600)
        thrust_hist.append(np.sqrt(u))
    thrust_hist = np.array(thrust_hist)

    # 3D Animation
    line_actual, = ax_3d.plot([], [], [], 'b-', label='Actual', linewidth=2)
    line_ref, = ax_3d.plot([], [], [], 'r--', label='Ref', linewidth=2)
    ax_3d.plot([0], [0], [5], 'k*', markersize=12, label='Anchor')
    ax_3d.set_xlim(-3, 3); ax_3d.set_ylim(-3, 3); ax_3d.set_zlim(0, 6)
    ax_3d.set_xlabel('X'); ax_3d.set_ylabel('Y'); ax_3d.set_zlabel('Z')
    ax_3d.set_title('3D Trajectory (Live)')
    ax_3d.legend(); ax_3d.grid(True)

    def update_3d(frame):
        line_actual.set_data_3d(p[:frame,0], p[:frame,1], p[:frame,2])
        line_ref.set_data_3d(p_r_hist[:frame,0], p_r_hist[:frame,1], p_r_hist[:frame,2])
        return line_actual, line_ref

    ani = animation.FuncAnimation(fig, update_3d, frames=len(t), interval=dt*1000, blit=False, repeat=False)

    # Static plots
    ax_top.plot(p[:,0], p[:,1], 'b-', label='Actual')
    ax_top.plot(p_r_hist[:,0], p_r_hist[:,1], 'r--', label='Ref')
    ax_top.plot(0, 0, 'k*', markersize=12)
    ax_top.set_xlabel('X'); ax_top.set_ylabel('Y'); ax_top.set_title('Top View'); ax_top.axis('equal')
    ax_top.legend(); ax_top.grid(True)

    err = np.linalg.norm(p_r_hist - p, axis=1)
    ax_err.plot(t, err, 'g-'); ax_err.set_title('Tracking Error'); ax_err.grid(True)

    ax_ten.plot(t, tension, 'm-'); ax_ten.set_title('Tether Tension'); ax_ten.grid(True)

    for i in range(4):
        ax_thr.plot(t, thrust_hist[:,i], label=f'M{i+1}')
    ax_thr.set_title('Motor Thrust'); ax_thr.legend(); ax_thr.grid(True)

    ax_int.plot(t, sol.y[16], label='Int X')
    ax_int.plot(t, sol.y[17], label='Int Y')
    ax_int.plot(t, sol.y[18], label='Int Z')
    ax_int.set_title('PID Integral'); ax_int.legend(); ax_int.grid(True)

    plt.draw()

# ========================================
# SLIDERS
# ========================================
ax_kp = plt.axes([0.15, 0.02, 0.65, 0.03])
ax_ki = plt.axes([0.15, 0.06, 0.65, 0.03])
ax_kd = plt.axes([0.15, 0.10, 0.65, 0.03])
ax_rad = plt.axes([0.15, 0.14, 0.65, 0.03])
ax_spd = plt.axes([0.15, 0.18, 0.65, 0.03])

s_kp = Slider(ax_kp, 'Kp_pos', 5, 30, valinit=Kp_pos_val)
s_ki = Slider(ax_ki, 'Ki_pos', 0, 5, valinit=Ki_pos_val)
s_kd = Slider(ax_kd, 'Kd_pos', 5, 20, valinit=Kd_pos_val)
s_rad = Slider(ax_rad, 'Radius', 1.0, 2.5, valinit=circle_radius)
s_spd = Slider(ax_spd, 'Speed (rad/s)', 0.4, 1.2, valinit=circle_omega)

def update(val):
    global Kp_pos_val, Ki_pos_val, Kd_pos_val, circle_radius, circle_omega
    Kp_pos_val = s_kp.val
    Ki_pos_val = s_ki.val
    Kd_pos_val = s_kd.val
    circle_radius = s_rad.val
    circle_omega = s_spd.val
    run_simulation()

s_kp.on_changed(update)
s_ki.on_changed(update)
s_kd.on_changed(update)
s_rad.on_changed(update)
s_spd.on_changed(update)

# Initial run
run_simulation()

plt.tight_layout()
plt.show()