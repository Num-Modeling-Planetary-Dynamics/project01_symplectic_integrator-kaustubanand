# EAPS 591 - Project 1
# Symplectic integrator to model the orbits of neptune and pluto

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# define constants

# Units are AU and days and Msun

t_end = 365.25 * (10 ** 5)  # days
Msun = 1.98850e6 # in 10^24 kg
G = 2.98e-4  # Scaled to the AU-days-Msun system
dt = 365.25 * 5  # days # use a fixed time step
accuracy = 1e-8 # accuracy for Danby solver

# Calculate the magnitude of a vector

def mag(v):
    return np.linalg.norm(v, axis=0)

# Set of functions for Keplerian Drift

def f(E, M, e):
    f_val = E - e * np.sin(E) - M
    return f_val


def f_prime(E, e):
    f_pr = 1 - e * np.cos(E)
    return f_pr


def f_2prime(E, e):
    f_2pr = e * np.sin(E)
    return f_2pr


def f_3prime(E, e):
    f_3pr = e * np.cos(E)
    return f_3pr


def deltai1(E, M, e):
    deltai = -1 * f(E, M, e) / f_prime(E, e)
    return deltai


def deltai2(E, M, e):
    deltai = -1 * f(E, M, e) / (f_prime(E, e) + 0.5 * deltai1(E, M, e) * f_2prime(E, e))
    return deltai


def deltai3(E, M, e):
    # deltai2 = deltai2(E)
    deltai = -1 * f(E, M, e) / (f_prime(E, e) + (0.5 * deltai2(E, M, e) * f_2prime(E, e)) + (deltai2(E, M, e) ** 2 * f_3prime(E, e) / 6))
    return deltai

# Danby's f function
def f_generate(dE, r_0, a):
    f = a / r_0 * (np.cos(dE) - 1.0) + 1.0

    return f


def f_dot_generate(dE, r, r_0, a, n):
    f_dot = -a ** 2 / (r * r_0) * n * (np.sin(dE))

    return f_dot


# Danby's g function
def g_generate(dE, dt, n):
    g = dt + (np.sin(dE) - (dE)) / n  # t - t_0 = dt

    return g


def g_dot_generate(dE, r, a):
    g_dot = a / r * (np.cos(dE) - 1) + 1

    return g_dot

def danby(M, e):
    k = 0.85 # initial guess

    E = M + np.sign(np.sin(M)) * e * k
    for i in range(100): # Break out if a certain accuracy is not achieved after 50 loops
        Enew = E + deltai3(E, M, e)
        error = np.abs((Enew - E) / E)
        error_check = error[1:] <= accuracy # error of only the planets

        if(np.all(error_check)):
            return Enew
        E = Enew

    print(f'E = {E}; Enew = {Enew}')
    print(f'Error = {error}')
    raise RuntimeError("The Danby function did not converge on a solution")

# convert cartesian to orbital elements
def xy_to_el(r_vec, P_vec, m):

    v_vec = (P_vec / m)
    mu = G * (m + m[0])

    h_vec = np.cross(r_vec.T, v_vec.T)
    h_vec = h_vec.T

    hx = h_vec[0]  # x-component of the angular momentum
    hy = h_vec[1]  # y-component of the angular momentum
    hz = h_vec[2]  # z-component of the angular momentum

    # Calculate h, r, and v magnitudes
    h = mag(h_vec)
    r = mag(r_vec)
    v = mag(v_vec)

    # Calculate a, e, inclination, lon_asc_node, arg_peri, and true_anom

    a = 1 / (2 / r - v ** 2 / mu)

    e = np.sqrt(1 - (h ** 2) / (a * mu))

    if(e[1] >= 1.0 or e[2] >= 1.0):
        raise RuntimeError(f'e = {e}; h^2 = {h**2}; a = {a}')

    if(np.isnan(a[0])):
        a[0] = 1e-10 #adjust to prevent nan for the sun
        e[0] = 0

    inclination = np.arccos(hz / h)

    lon_asc_node = np.arctan2(np.sign(hz) * hx, -np.sign(hz) * hy)

    sign = np.sign(np.diag(np.tensordot(r_vec, v_vec, axes=[0, 0])))

    rdot = np.sqrt(v**2 - (h / r) ** 2) * sign

    sf = a * (1 - e**2) * rdot / (h * e)
    cf = (a * (1 - e**2) / r - 1) / e
    true_anom = np.arctan2(sf, cf)
    # true_anom = np.arctan2(rdot * r / h * (1 + r), (a * (1 - e**2) - r))

    sof = r_vec[2] / (r * np.sin(inclination))  # sin(omega + f)
    cof = (r_vec[0] / r + np.sin(lon_asc_node) * sof * np.cos(inclination)) / np.cos(lon_asc_node)  # cos(omega + f)
    arg_peri = np.arctan2(sof, cof) - true_anom
    # arg_peri = np.mod(arg_peri, 2 * np.pi)

    E = np.arccos((1 - r / a) / e)  # calculate Eccentric Anomaly
    E = np.where(sign < 0.0, 2 * np.pi - E, E)

    M = E - e * np.sin(E)

    n = np.sqrt(mu / a ** 3)  # Kepler's 3rd law to get the mean motion

    varpi = arg_peri + lon_asc_node
    lam = arg_peri + lon_asc_node + M

    if(np.isnan(inclination[0])):
        #adjust to prevent nan for the sun
        inclination[0] = 0
        lon_asc_node[0] = 0
        arg_peri[0] = 0
        true_anom[0] = 0

    return a, e, inclination, lon_asc_node, arg_peri, true_anom, E, M, np.rad2deg(lam), np.rad2deg(varpi), n


# Keplerian Drift
# noinspection PyUnreachableCode
def kepler_drift(Q, P, m, dt):
    a, e, inclination, lon_asc_node, arg_peri, true_anom, E, M, lam, varpi, n = xy_to_el(Q, P, m)  # Convert cartesian to orbital elements
    # sign = np.sign(np.diag(np.tensordot(Q, P / m, axes=[0, 0])))

    M = M + n * dt

    # Danby's solver

    E_tmp = danby(M, e)

    dE = E_tmp - E

    # Danby's conversion to cartesian
    r_0 = mag(Q)

    f = f_generate(dE, r_0, a)  # (dE, r_0, a)
    g = g_generate(dE, dt, n)  # (dE, dt, n)

    r = f * Q + g * (P / m)  # f and g are arrays
    r_mag = mag(r)

    f_dot = f_dot_generate(dE, r_mag, r_0, a, n)  # (dE, r, r_0, a, n)
    g_dot = g_dot_generate(dE, r_mag, a)  # (dE, r, a)

    v = f_dot * Q + g_dot * (P / m)

    return r, m * v, E_tmp, arg_peri, e, lam, varpi


# convert barycentric to heliocentric
def bc_to_hc(vec_b, m_tot, m):
    origin_h = -np.sum(m * vec_b, axis = 1) / m[0] # m[0] = mass of the sun
    vec_h = vec_b - origin_h

    return vec_h  # return heliocentric

# convert heliocentric to barycentric
def hc_to_bc(vec_h, m_tot, m):
    origin_b = -np.sum(m * vec_h, axis = 1) / (m_tot)
    vec_b = vec_h + origin_b

    return vec_b  # return barycentric

# Sun Drift
def sun_drift(Q, P, m, dt):
    # Q_dot = P / M_sun
    V = P / m
    Q_dot = np.sum(m * V, axis = 1) / (1) # 1 = Msun in Msun units; m to scale momentum back to velocity
    Q_tmp = (Q.T + Q_dot * dt / 2).T

    return Q_tmp


# Interaction Kick
def int_kick(Q, P, m, dt):
    # acceleration is gravitational acc.

    for i in range(len(m)): # interaction force on this planet
        force = 0
        for j in range(len(m)): # cycle through remaining planets in the system
            if (i == j):
                continue
            r_vec = Q[:, j] - Q[:, i]
            force += G * m[j] * m[i] / np.power(mag(r_vec), 3) * r_vec # Force calculation

        P[:,i] = P[:,i] + force * dt / 2

    # rvec = Q.T
    # for i in range(len(m)):
    #
    #     drvec = rvec - rvec[i, :]
    #     irij3 = np.linalg.norm(drvec, axis=1) ** 3
    #     irij3[i] = 1  # Diagonal should not be included in the sum
    #     irij3 = G * m / irij3
    #
    #     acc = np.sum(drvec.T * irij3, axis=1)
    #
    #     P[:,i] += m[i] * acc * dt / 2

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

data = pd.read_csv('midterm_input.csv') # '../data/midterm_input.csv

names = np.asarray(data['Object_Name'])
m = np.asarray(data['mass']) / Msun # mass in Msun
n = 2 * np.pi / np.asarray(data['T'])  # mean motion
n[0] = 1e-10 # set mean motion of the sun to ~0
q = np.asarray([data['X'], data['Y'], data['Z']])
p = m * np.asarray([data['VX'], data['VY'], data['VZ']])

# Initialize other arrays

m_tot = np.sum(m)

Q0 = np.sum((m * q), axis = 1) / m_tot  # Calculate sun initial canonical coordinates
P0 = np.sum(p, axis = 1)

Q = np.array(q) - q[:,0]  # Canonical position (Heliocentric)
P = hc_to_bc(p, m_tot, m) # Canonical momentum (Barycentric)
V = P / m

Q[:,0] = Q0
P[:,0] = P0  # Add Sun coordinates into array

a, e, inclination, lon_asc_node, arg_peri, true_anom, E_tmp, M, lam, varpi, n = xy_to_el(Q, P, m)

e_arr = e

M_arr = E_tmp - e * np.sin(E_tmp)
lambda_plt = lam[2]
lambda_nep = lam[1]
phi_tmp = np.mod(3 * lambda_plt - 2 * lambda_nep - varpi[2], 360)

ln_arr = [lambda_nep]
lp_arr = [lambda_plt]
vplt_arr = [varpi[2]]

t_arr = np.append(t_arr, np.array([0]), axis=0)
E_err = np.asarray([0])
phi = np.append(phi, phi_tmp)

# Calc Energy
E_tot_0 = calc_energy(Q, P, m)

num_planets = len(names)  # number of planets

# Fix Array shapes
Q = np.asarray([Q])
P = np.asarray([P])
E = np.asarray([E_tmp])
M_arr = np.asarray([M_arr])
e_arr = np.asarray([e_arr])
P_bc = P # Barycentric Momentum to carry over in time steps

q = np.asarray([q])
p = np.asarray([p])

# Do integration
while (t_arr[-1] <= t_end):

    # origin_b = np.sum((m * p[-1]), axis = 0) / m_tot  # calc barycentric origin
    # origin_h = m * Q[-1][0]# calc heliocentric origin

    t_current = t_arr[-1]

    # Define temp variables for each time step
    Q_tmp = np.asarray(Q[-1])
    P_tmp = np.asarray(P_bc[-1])
    E_tmp = np.asarray(E[-1])
    M_tmp = np.asarray(M_arr[-1])
    e_tmp = np.asarray(e_arr[-1])
    t = np.asarray(t_arr[-1])

    if (t % (365.25 * 5000) == 0):
        time_step = np.rint((t_current / 365.25) / 1000) * 1000
        print("\n\nAt t ~ {} years".format(time_step))  # Print time steps

    # if(t % (365.25 * 10000) == 0):
    #     print("check values")

    # Update Linear Drift
    Q_tmp[:,1:] = sun_drift(Q_tmp[:,1:], P_tmp[:,1:], m[1:], dt)

    # Update Interaction Velocities Kick
    P_tmp[:,1:] = int_kick(Q_tmp[:,1:], P_tmp[:,1:], m[1:], dt)

    # Update Keplerian positions

    # M_tmp = M_tmp + n * dt
    # M_tmp = np.mod(M_tmp, 2 * np.pi)

    Q_0 = Q_tmp[:, 0]
    P_0 = P_tmp[:, 0]

    Q_tmp, P_tmp, E_tmp, arg_peri, e_tmp, lam_tmp, varpi_tmp = kepler_drift(Q_tmp, P_tmp, m, dt)

    Q_tmp[:, 0] = Q_0
    P_tmp[:, 0] = P_0  # cancel out Kepler drift effects of sun

    # Update Interaction velocities again
    P_tmp[:,1:] = int_kick(Q_tmp[:,1:], P_tmp[:,1:], m[1:], dt)  # Use updated half-time-step to kick velocities

    # Linear Drift again
    Q_tmp[:,1:] = sun_drift(Q_tmp[:,1:], P_tmp[:,1:], m[1:], dt)

    P_h = bc_to_hc(P_tmp, m_tot, m) # Convert barycentric velocity/mom to heliocentric velocity/mom

    # Calculate Energy and Mean Anomaly and Resonance angle
    E_tot = calc_energy(Q_tmp, P_tmp, m)

    lambda_nep = lam_tmp[1]
    lambda_plt = lam_tmp[2]
    phi_tmp = np.mod(3 * lambda_plt - 2 * lambda_nep - varpi_tmp[2], 360)

    ln_arr.append(lambda_nep)
    lp_arr.append(lambda_plt)
    vplt_arr.append(varpi_tmp[2])

    # append values into respective arrays
    Q = np.append(Q, np.asarray([Q_tmp]), axis=0)
    P = np.append(P, np.asarray([P_h]), axis=0)
    P_bc = np.append(P, np.asarray([P_tmp]), axis=0)
    E_err = np.append(E_err, np.abs((E_tot - E_tot_0) / E_tot_0))
    E = np.append(E, np.asarray([E_tmp]), axis=0)
    M_arr = np.append(M_arr, np.asarray([M_tmp]), axis=0)
    e_arr = np.append(e_arr, np.asarray([e_tmp]), axis=0)
    M_nep = np.append(M_nep, M[1])
    M_plt = np.append(M_plt, M[2])
    phi = np.append(phi, phi_tmp)
    t_arr = np.append(t_arr, t + dt)

# plots

fig, (ax1, ax2) = plt.subplots(2)

ax1.plot(t_arr / 365.25, E_err)
ax2.plot(t_arr / 365.25, phi)

# ax2.plot(t_arr / 365.25, ln_arr, label = 'Lam Neptune', alpha = 0.6)
# ax2.plot(t_arr / 365.25, lp_arr, label = 'Lam Pluto', alpha = 0.6)
# ax2.plot(t_arr / 365.25, vplt_arr, label = 'Varpi Pluto', alpha = 0.6)

ax1.set_xlabel(f"Time (years); dt = {np.round(dt / 365, 0)} years")
ax2.set_xlabel(f"Time (years); dt = {np.round(dt / 365, 0)} years")
ax1.set_ylabel(r'$\Delta \epsilon / \epsilon_0$')
ax2.set_ylabel(r'$\phi$')

# plt.legend()
plt.show()

lineObjects = plt.plot(t_arr / 365.25, e_arr)
plt.xlabel("Time (years)")
plt.ylabel("eccentricity (e)")

plt.legend(lineObjects, names)
plt.show()


