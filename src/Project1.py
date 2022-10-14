# EAPS 591 - Project 1
# Symplectic integrator to model the orbits of neptune and pluto

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Calculate the magnitude of a vector
def mag(v):
    return np.sqrt(np.sum(v**2, axis = 1))

# Set of functions for Keplerian Drift

def f(E):
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

def deltai1(E):
    deltai = -1 * f(E) / f_prime(E)
    return deltai

def deltai2(E):
    deltai = -1 * f(E) / (f_prime(E) + 0.5 * deltai1(E) * f_2prime(E))
    return deltai

def deltai3(E):
    # deltai2 = deltai2(E)
    deltai = -1 * f(E) / (f_prime(E) + (0.5 * deltai2(E) * f_2prime(E)) + (deltai2(E) ** 2 * f_3prime(E) / 6))
    return deltai

# Danby's f function
def f_generate(E, E_0, r_0, a):

    f = a / r_0 * (np.cos(E - E_0) - 1) + 1

    return f

def f_dot_generate(E, E_0, r, r_0, a, n):

    f_dot = a**2 / (r_0 * r) * n * (np.sin(E - E_0))

    return f_dot

# Danby's g function
def g_generate(E, E_0, t, t_0, n):

    g = dt + (np.sin(E - E_0) - (E - E_0)) / n # t - t_0 = dt

    return g


def g_dot_generate(E, E_0, r, a):

    g_dot = a / r * (np.cos(E - E_0) - 1) + 1

    return g_dot

# convert cartesian to orbital elements
def xy_to_el(Q, a, e):

    b = np.sqrt(1 - e**2) * a
    E = np.arctan2(a * Q[:,1], b * Q[:,0]) # tan E = a * y / (b * x)
    # M = E - e * np.sin(E)

    return E

# Keplerian Drift
def kepler_drift(Q, P, m, a, e, dt, n):

    E = xy_to_el(Q, a, e)  # Convert cartesian to orbital elements

    E_tmp = E + deltai3(E) # New Eccentric Anomaly
    r_0 = mag(Q)
    r = a * (1 - e * np.cos(E_tmp))

    # # trig conversion to cartesian

    # Q_tmp = [a * (np.cos(E) - e), a * np.sqrt(1 - e**2) * np.sin(E), Q[:,2]] # Keep z the same
    # P_tmp = m * n * a**2 / r * [-1 * np.sin(E), np.sqrt(1 - e**2) * np.cos(E), 0] # Keep z the same
    # P_tmp[2] = P[2]

    # Danby's conversion to cartesian

    f = f_generate(E_tmp, E,  r_0, a) # (E, E_0, r_0, a)
    g = g_generate(E_tmp, E, dt, n) # (E, E_0, t, t_0, n)
    f_dot = f_dot_generate(E_tmp, E, r, r_0, a, n) # (E, E_0, r, r_0, a, n)
    g_dot = g_dot_generate(E_tmp, E, r, a) # (E, E_0, r, a)

    r = f * Q + g * P / m  # f and g are arrays
    v = f_dot * Q + g_dot * P / m

    return r, v * m, E_tmp

# convert barycentric to heliocentric
def bc_to_hc(b, origin_b, origin_h):

    h = b + origin_b - origin_h

    return h # return heliocentric

# convert heliocentric to barycentric
def hc_to_bc(h, origin_b, origin_h):

    b = h - origin_b + origin_h

    return b # return barycentric

# Sun Drift
def sun_drift(Q, P, M_sun, dt):

    Q_dot = P / M_sun
    Q_tmp = Q + Q_dot * dt / 2

    return Q_tmp

# Interaction Kick
def int_kick(Q, P, m, dt):
    # acceleration is gravitational acc.

    for i in range(len(Q)):
        acc = 0
        for j in range(len(Q)):
            if (i == j):
                continue
            acc += G * m[j] * m[i] / mag(Q[i] - Q[j])**3 * (Q[j] - Q[i]) # Direction

        P[i] = P[i] + acc * dt / 2

    return P

# Calculate Energy
def calc_energy(Q, P, m):
    # Sun values
    Q_0 = Q[0]
    P_0 = P[0]
    m_0 = m[0]

    #planets values
    Q = Q[1:]
    P = P[1:]
    m = m[1:]

    # Keplerian Energy
    H_kepler = np.sum(mag(P)**2 / (2 * m) - G * m * m_0 / mag(Q))

    # Interaction Energy
    H_int = 0
    for i in range(len(Q) - 1):
        for j in range(i + 1, len(Q)):
            H_int -= G * m[j] * m[i] / mag(Q[i] - Q[j])

    # Sun Hamiltonian
    H_sun = np.sum(P)**2 / (2 * m_0)

    return H_kepler + H_int + H_sun


# =================================================================================================================
# MAIN

# define constants
t_end = 365 * 10**5 # days
G = 1
dt = 365 # days # use a fixed time step
omega_nep = 44.97135 * np.pi / 180 # longitude of periapsis of neptune in radians
omega_plt = 224.06676 * np.pi / 180 # longitude of periapsis of pluto in radians

# define variables
E_err = [] # Energy error
E = [] # Eccentric Anomaly
t_arr = [] # time array
M_nep = [] # Mean Anomaly for Neptune
M_plt = [] # Mean anomaly for Pluto
phi = [] # Resonance Angle

# read in data for the planets

# DEFINE ARRAYS AS [[X, Y, Z], [X, Y, Z], ....] where each subarray is defined as the coordinates for a given body

data = pd.read_csv('data/midterm_input.csv')

# How to do this for Sun, neptune and pluto??

m = np.asarray(data['mass']) # mass in 10^24 kg
q = np.asarray([data['X'], data['Y'], data['Z']]) # position array
p = m * np.asarray([data['VX'], data['VY'], data['VZ']]) # momentum array
a = np.asarray(data['a']) # semi-major axis
e = np.asarray(data['e']) # eccentricity
n = 2 * np.pi / np.asarray(data['T']) # mean motion

# Initialize other arrays

Q0 = np.sum(m * q) / m_tot # Calculate sun initial canonical coordinates
P0 = np.sum(p)

Q = np.asarray(q - Q0) # Canonical position (Heliocentric)
P = np.asarray(p - m * P0 / m_tot) # Canonical momentum (Barycentric)

Q[0] = Q0
P[0] = P0 # Add Sun coordinates into array

t_arr.append(0)
E.append(xy_to_el(Q, a, e))

m_tot = np.sum(m)

# Calc Energy
E_tot_0 = calc_energy(Q, P, m)

num_planets = len(data) # number of planets

# Do integration
while (t_arr[-1] <= t_end):

    origin_b = np.sum(m * p) / m_tot  # HOW TO calc barycentric origin
    origin_h = np.sum(m * q) / m_tot # calc heliocentric origin

    for i in range(num_planets):

        # Define temp variables for each time step
        Q_tmp = Q[-1]
        P_tmp = P[-1]
        E_tmp = E[-1]
        t = t_arr[-1]

        if(t % (365 * 5000) == 0):
            print("At t = {}".format(t)) # Print time steps

        P_tmp = hc_to_bc(P_tmp, origin_b, origin_h) # Convert heliocentric velocity/momentum to barycentric velocity/momentum

        # Update Sun Drift
        Q_tmp[0] = sun_drift(Q_tmp[0], P_tmp[0], m[0], dt)

        # Update Interaction Velocities Kick
        P_tmp[1:] = int_kick(Q_tmp[1:], P_tmp[1:], m[1:], dt)

        # Update Keplerian positions

        Q_tmp, P_tmp, E_tmp = kepler_drift(Q, P, m, a, e, dt, n)

        # Update Interaction velocities again
        P_tmp[1:] = int_kick(Q_tmp[1:], P_tmp[1:], m[1:], dt) # Use updated half-time-step to kick velocities

        # Update Sun again
        Q_tmp[0] = sun_drift(Q_tmp[0], P_tmp[0], m[0], dt)

        P_h = bc_to_hc(P_tmp, origin_b, origin_h) # Convert barycentric velocity/mom to heliocentric velocity/mom

        # Calculate Energy and Mean Anomaly and Resonance angle
        E_tot = calc_energy(Q, P, m)
        M = E_tmp - e * sin(E_tmp)
        M_nep_tmp = M[1]
        M_plt_tmp = M[2]
        lambda_nep = M_nep_tmp + omega_nep
        lambda_plt = M_plt_tmp + omega_plt
        phi_tmp = 3 * lambda_plt - 2 * lambda_nep - omega_plt

        Q.append(Q_tmp)
        P.append(P_h)
        E_err.append((E_tot - E_tot_0) / E_tot_0)
        E.append(E_tmp)
        M_nep.append(M_nep_tmp)
        M_plt.append(M_plt_tmp)
        phi.append(phi_tmp)
        t_arr.append(t + dt)

# plots

fig, (ax1, ax2) = plt.subplots(dpi = 200)

ax1.scatter(t_arr, E_err)
ax2.scatter(t_arr, phi)

ax1.set_xlabel("Time (days)")
ax2.set_xlabel("Time (days)")
ax1.set_ylabel(r'$\Delta \Epsilon / \Epsilon_0$')
ax2.set_ylabel(r'\phi')

plt.show()
