# EAPS 591 - Project 1
# Symplectic integrator to model the orbits of neptune and pluto

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# define constants

# Units are AU and days and Msun

t_end = 365 * 10 ** 5  # days
Msun = 1.98850e6 # in 10^24 kg
G = 2.98e-4  # Scaled to the AU-days-Msun system
dt = 365 * 2  # days # use a fixed time step

# Calculate the magnitude of a vector

def mag(v):
    if (np.ndim(v) == 1):
        return np.sqrt(np.sum(v ** 2))
    else:
        return np.sqrt(np.sum(v ** 2, axis=0))


# Set of functions for Keplerian Drift

def f(E, M):
    f_val = E - e * np.sin(E) - M
    return f_val


def f_prime(E):
    f_pr = 1 - e * np.cos(E)
    return f_pr


def f_2prime(E):
    f_2pr = e * np.sin(E)
    return f_2pr


def f_3prime(E):
    f_3pr = e * np.cos(E)
    return f_3pr


def deltai1(E, M):
    deltai = -1 * f(E, M) / f_prime(E)
    return deltai


def deltai2(E, M):
    deltai = -1 * f(E, M) / (f_prime(E) + 0.5 * deltai1(E, M) * f_2prime(E))
    return deltai


def deltai3(E, M):
    # deltai2 = deltai2(E)
    deltai = -1 * f(E, M) / (f_prime(E) + (0.5 * deltai2(E, M) * f_2prime(E)) + (deltai2(E, M) ** 2 * f_3prime(E) / 6))
    return deltai

# Danby's f function
def f_generate(E, E_0, r_0, a):
    f = a / r_0 * (np.cos(E - E_0) - 1) + 1

    return f


def f_dot_generate(E, E_0, r, r_0, a, n):
    f_dot = -a ** 2 / (r * r_0) * n * (np.sin(E - E_0))

    return f_dot


# Danby's g function
def g_generate(E, E_0, dt, n):
    g = dt + (np.sin(E - E_0) - (E - E_0)) / n  # t - t_0 = dt

    return g


def g_dot_generate(E, E_0, r, a):
    g_dot = a / r * (np.cos(E - E_0) - 1) + 1

    return g_dot


# convert cartesian to orbital elements
def xy_to_el(r_vec, P_vec, m):

    v_vec = (P_vec / m)
    mu = G * (m + m[0])

    # Calculate h, r, and v magnitudes
    h = 0
    r = 0
    v = 0

    h_vec = np.cross(r_vec.T, v_vec.T)
    h_vec = h_vec.T



    hx = h_vec[0]  # x-component of the angular momentum
    hy = h_vec[1]  # y-component of the angular momentum
    hz = h_vec[2]  # z-component of the angular momentum

    h = mag(h_vec)
    r = mag(r_vec)
    v = mag(v_vec)

    # Calculate Eccentric Anomaly (E) a, e, inclination, lon_asc_node, arg_peri, and true_anom

    a = 2 / r - v ** 2 / mu
    a = np.abs(1 / a)

    e = np.sqrt(1 - (h ** 2) / (a * mu))

    if(np.isnan(a[0])):
        a[0] = 1e-10 #adjust to prevent nan for the sun
        e[0] = 0

    b = np.sqrt(1 - e ** 2) * a
    E = np.arctan2(a * r_vec[1], b * r_vec[0])  # tan E = a * y / (b * x)

    if np.sign(np.vdot(r_vec, v_vec)) < 0.0:
        E = 2 * np.pi - E

#     E = np.arccos(-1 * (mag(Q) - a) / (a * e))

    inclination = np.arccos(hz / h)

    lon_asc_node = np.arctan2(hx, -1 * hy)

    # true_anom = np.arcsin(v * a * (1 - e ** 2) / (h * e) )
    # true_anom = np.arccos((a * (1 - e ** 2) / r - 1) / e)
    true_anom = np.arctan2(v * r / h * (1 + r), (a * (1 - e**2) - r))

    # arg_peri = np.arcsin(r_vec[2] / (r * np.sin(inclination))) - true_anom
    arg_peri = np.arctan2(-1 * hx, hy)

    if(np.isnan(E[0])):
        E[0] = 0 #adjust to prevent nan for the sun
        inclination[0] = 0
        lon_asc_node[0] = 0
        arg_peri[0] = 0
        true_anom[0] = 0

    return E, a, e, inclination, lon_asc_node, arg_peri, true_anom


# Keplerian Drift
# noinspection PyUnreachableCode
def kepler_drift(Q, P, m, a, e, dt, n):
    E, a, e, inclination, lon_asc_node, arg_peri, true_anom = xy_to_el(Q, P, m)  # Convert cartesian to orbital elements
    M = E - e * np.sin(E) # Mean anomaly

    E_tmp = E + deltai3(E, M)  # New Eccentric Anomaly
    r_0 = mag(Q)
    r = a * (1 - e * np.cos(E_tmp))

    # # trig conversion to cartesian

    # Q_tmp = [a * (np.cos(E) - e), a * np.sqrt(1 - e**2) * np.sin(E), Q[:,2]] # Keep z the same
    # P_tmp = m * n * a**2 / r * [-1 * np.sin(E), np.sqrt(1 - e**2) * np.cos(E), 0] # Keep z the same
    # P_tmp[2] = P[2]

    # Danby's conversion to cartesian

    f = f_generate(E_tmp, E, r_0, a)  # (E, E_0, r_0, a)
    g = g_generate(E_tmp, E, dt, n)  # (E, E_0, dt, n)
    f_dot = f_dot_generate(E_tmp, E, r, r_0, a, n)  # (E, E_0, r, r_0, a, n)
    g_dot = g_dot_generate(E_tmp, E, r, a)  # (E, E_0, r, a)

    r = f * Q + g * (P / m)  # f and g are arrays
    v = f_dot * Q + g_dot * P / m

    return r, m * v, E_tmp, arg_peri


# convert barycentric to heliocentric
def bc_to_hc(b, origin_b, origin_h):
    h = b + origin_b - origin_h

    return h  # return heliocentric


# convert heliocentric to barycentric
def hc_to_bc(h, origin_b, origin_h):
    b = h - origin_b + origin_h

    return b  # return barycentric


# Sun Drift
def sun_drift(Q, P, M_sun, dt):
    Q_dot = P / M_sun
    Q_tmp = Q + Q_dot * dt / 2

    return Q_tmp


# Interaction Kick
def int_kick(Q, P, m, dt):
    # acceleration is gravitational acc.

    for i in range(len(m)):
        acc = 0
        for j in range(len(m)):
            if (i == j):
                continue

            acc += G * m[j] * m[i] / np.power(mag(Q[:,i] - Q[:,j]), 3) * (Q[:,j] - Q[:,i])  # Force calculation

        P[:,i] = P[:,i] + acc * dt / 2

    return P


# Calculate Energy
def calc_energy(Q, P, m):

    # Sun values
    Q_0 = Q[:,0]
    P_0 = P[:,0]
    m_0 = m[0]

    # planets values
    Q = Q[:,1:]
    P = P[:,1:]
    m = m[1:]

    # Keplerian Energy
    H_kepler = np.sum(mag(P) ** 2 / (2 * m) - G * m * m_0 / mag(Q))

    # Interaction Energy
    H_int = 0
    for i in range(len(m) - 1):
        for j in range(i + 1, len(m)):

            H_int -= G * m[j] * m[i] / mag(Q[:,i] - Q[:,j])

    # Sun Hamiltonian
    H_sun = np.sum(P) ** 2 / (2 * m_0)

    return H_kepler + H_int + H_sun


# =================================================================================================================
# MAIN

# define variables
E_err = []  # Energy error
E = []  # Eccentric Anomaly
t_arr = []  # time array
M_nep = []  # Mean Anomaly for Neptune
M_plt = []  # Mean anomaly for Pluto
phi = []  # Resonance Angle

# read in data for the planets

# Arrays defined as [[X1, X2, X3, ... XN], [Y1, Y2, Y3, ... YN], ....]; Avoids .T multiplications
# Array shape/axes = (time-step, planet, coordinate)

data = pd.read_csv('midterm_input.csv')

names = np.asarray(data['Object_Name'])
m = np.asarray(data['mass']) / Msun # mass in Msun
n = 2 * np.pi / np.asarray(data['T'])  # mean motion
n[0] = 1e-10 # set mean motion of the sun to ~0
q = np.asarray([data['X'], data['Y'], data['Z']])
p = m * np.asarray([data['VX'], data['VY'], data['VZ']])

# Initialize other arrays

m_tot = np.sum(m)

# CHECK
Q0 = np.sum((m * q), axis = 0) / m_tot  # Calculate sun initial canonical coordinates
P0 = np.sum(p, axis = 0)

Q = np.array(q) - q[:,0]  # Canonical position (Heliocentric)
P = p - (m * P0) / m_tot  # Canonical momentum (Barycentric)

Q[:,0] = Q0
P[:,0] = P0  # Add Sun coordinates into array

E_tmp, a, e, inclination, lon_asc_node, arg_peri, true_anom = xy_to_el(Q, P, m)
M = E_tmp - e * np.sin(E_tmp)
lambda_plt = arg_peri[2] + M[2]
lambda_nep = arg_peri[1] + M[1]
phi_tmp = 3 * lambda_plt - 2 * lambda_nep - arg_peri[2]

t_arr = np.append(t_arr, np.array([0]), axis=0)
E = np.append(E, E_tmp, axis=0)
E_err = np.asarray([0])
phi = np.append(phi, phi_tmp)

# Calc Energy
E_tot_0 = calc_energy(Q, P, m)

num_planets = len(names)  # number of planets

# Fix Array shapes
Q = np.asarray([Q])
P = np.asarray([P])
P_bc = P # Barycentric Momentum to carry over in time steps

q = np.asarray([q])
p = np.asarray([p])

# Do integration
while (t_arr[-1] <= t_end):

    origin_b = np.sum((m * p[-1])) / m_tot  # HOW TO calc barycentric origin
    origin_h = np.sum((m * Q[-1])) / m_tot  # calc heliocentric origin

    t_current = t_arr[-1]

    # for i in range(num_planets):

    # Define temp variables for each time step
    Q_tmp = np.asarray(Q[-1])
    P_tmp = np.asarray(P_bc[-1])
    E_tmp = np.asarray(E[-1])
    t = np.asarray(t_arr[-1])

    if (t % (365 * 5000) == 0):
        print("\n\nAt t = {} years".format(t / 365))  # Print time steps

    # if(t_arr[-1] > 0):
    #     P_tmp = hc_to_bc(P_tmp, origin_b, origin_h)  # Convert heliocentric velocity/momentum to barycentric velocity/momentum

    # Update Sun Drift
    Q_tmp[:,0] = sun_drift(Q_tmp[:,0], P_tmp[:,0], m[0], dt)

    # Update Interaction Velocities Kick
    P_tmp[:,1:] = int_kick(Q_tmp[:,1:], P_tmp[:,1:], m[1:], dt)

    # Update Keplerian positions

    Q_tmp, P_tmp, E_tmp, arg_peri = kepler_drift(Q_tmp, P_tmp, m, a, e, dt, n)
    # Q_tmp[:,1:], P_tmp[:,1:], E_tmp[1:], arg_peri[1:] = kepler_drift(Q_tmp[:,1:], P_tmp[:,1:], m[1:], a[1:], e[1:], dt, n[1:])

    # Update Interaction velocities again
    P_tmp[:,1:] = int_kick(Q_tmp[:,1:], P_tmp[:,1:], m[1:], dt)  # Use updated half-time-step to kick velocities

    # Update Sun again
    Q_tmp[:,0] = sun_drift(Q_tmp[:,0], P_tmp[:,0], m[0], dt)

    P_h = bc_to_hc(P_tmp, origin_b, origin_h)  # Convert barycentric velocity/mom to heliocentric velocity/mom

    # Calculate Energy and Mean Anomaly and Resonance angle
    E_tot = calc_energy(Q_tmp, P_tmp, m)
    M = E_tmp - e * np.sin(E_tmp)
    M_nep_tmp = M[1]
    M_plt_tmp = M[2]
    lambda_nep = M_nep_tmp + arg_peri[1]
    lambda_plt = M_plt_tmp + arg_peri[2]
    phi_tmp = 3 * lambda_plt - 2 * lambda_nep - arg_peri[2]

    Q = np.append(Q, np.asarray([Q_tmp]), axis=0)
    P = np.append(P, np.asarray([P_h]), axis=0)
    P_bc = np.append(P, np.asarray([P_tmp]), axis=0)
    E_err = np.append(E_err, ((E_tot - E_tot_0) / E_tot_0))
    E = np.append(E, np.asarray((E_tmp)), axis=0)
    M_nep = np.append(M_nep, M_nep_tmp)
    M_plt = np.append(M_plt, M_plt_tmp)
    phi = np.append(phi, phi_tmp)
    t_arr = np.append(t_arr, t + dt)

# plots

fig, (ax1, ax2) = plt.subplots(2)

ax1.plot(t_arr, E_err)
ax2.plot(t_arr, phi)

ax1.set_xlabel(f"Time (days); dt = {dt / 365} years")
ax2.set_xlabel(f"Time (days); dt = {dt / 365} years")
ax1.set_ylabel(r'$\Delta \epsilon / \epsilon_0$')
ax2.set_ylabel(r'\phi')

plt.show()

