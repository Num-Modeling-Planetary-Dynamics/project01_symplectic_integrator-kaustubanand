import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from astroquery.jplhorizons import Horizons
import astropy.constants as const
import datetime

class Symplectic_Integrator:
    """
    Class for a N-body Symplectic Integrator
    EAPS 591 midterm project by Kaustub Anand

    (n = number of planets in documentation unless mentioned otherwise)

    """
    def __init__(self, names, date):
        """
        Function to initialize the simulation object.
        Sets up the units for the system (AU-Days-Msun)

        Set up initial arrays for (array shape):
            t = time steps
            xv_elements = Cartesian elements; pos vector + vel vector; (t, n, 6)
            m = mass (n)
            msun = mass of the sun (scalar)
            mu = G * (m + msun) parameter (n)
            m_tot = total mass (scalar)
            el_elements = Orbital elements; a, e, i, Omega, omega, f, E, M, lam, varpi, n (mean-motion) ; (t, n, 11)
            rh_vec = temporary heliocentric position vector for calculations (n, 3)
            vb_vec = temporary barycentric velocity vector for calculations (n, 3)
            el_elements_tmp = temporary orbital elements vector for calculations (n, 11)
            energy = Energy (t)
            E_err = Energy error (t)
            phi = Resonance angle (t)
            dt = time step (scalar)
            names = planet/body names (n)
            G = Gravitational constant (scalar)

        Input Parameters:
            date = date to extract Horizons data from; format = YYYY-MM-DD

        """
        self.add_bodies(names)
        n = len(names)

        # Dataset variables
        self.xv_elements = []
        self.m = []
        self.mu = []
        self.energy = []
        self.msun = 1 # Msun units
        self.el_elements = []
        self.dt = 5 * 365.25  # 5 years
        self.G = 2.98e-4 # AU^2 / Days^3

        # Plotting variables
        self.E_err = []
        self.phi = []

        # Temporary variables
        self.rh_vec = np.zeros((n, 3))  # 3 = x, y, z
        self.vb_vec = np.zeros((n, 3))
        self.el_elements_tmp = np.zeros((n, 11))

        # Initialize Horizon data
        self.get_horizon_data(names, date)

        self.m_tot = self.msun + np.sum(self.m)

        for i in range(len(names)):
            # convert heliocentric velocity to barycentric velocity
            self.xv_elements[i][3:6] = self.hc_to_bc(self.xv_elements[i][3:6], self.mu[i])

            # calculate orbital elements
            self.el_elements.append(self.xy_to_el(self.xv_elements[i][0:3], self.xv_elements[i][3:6], self.mu[i]))

        self.xv_elements = [self.xv_elements]
        self.el_elements = [self.el_elements]
        self.calc_energy()
        self.E_err = [0]
        self.phi = [self.calc_phi()]
        self.t = [0]

        return

    def xy_to_el(self, r_vec, v_vec, mu):
        """
        Converts cartesian elements to orbital elements

        Input Parameters:
           r_vec : Position vector (3)
           v_vec : Velocity vector (3)
           mu : mu = G * (m + msun) for the body

        Output Parameters:
            a : semi-major axis
            e : eccentricity
            i : i
            Omega : longitude of ascending node
            omega : argument of periapsis
            f : true anomaly
            E : eccentric anomaly
            M : mean anomaly
            lam : mean longitude
            varpi : longitude of periapsis
            n : mean motion

        """

        h_vec = np.cross(r_vec, v_vec)

        hx = h_vec[0]  # x-component of the angular momentum
        hy = h_vec[1]  # y-component of the angular momentum
        hz = h_vec[2]  # z-component of the angular momentum

        # Calculate h, r, and v magnitudes
        h = self.mag(h_vec)
        r = self.mag(r_vec)
        v = self.mag(v_vec)

        # Calculate a, e, i, Omega, omega, and f

        a = 1 / (2 / r - v ** 2 / mu)

        e = np.sqrt(1 - (h ** 2) / (a * mu))

        if (e >= 1.0) :
            raise RuntimeError(f'e = {e}; h^2 = {h ** 2}; a = {a}')

        i = np.arccos(hz / h)

        Omega = np.mod(np.arctan2(np.sign(hz) * hx, -np.sign(hz) * hy), 2 * np.pi)

        sign = np.sign(np.vdot(r_vec, v_vec))

        rdot = np.sqrt(v ** 2 - (h / r) ** 2) * sign

        sf = a * (1 - e ** 2) * rdot / (h * e)
        cf = (a * (1 - e ** 2) / r - 1) / e
        f = np.arctan2(sf, cf)

        sof = r_vec[2] / (r * np.sin(i))  # sin(omega + f)
        cof = (r_vec[0] / r + np.sin(Omega) * sof * np.cos(i)) / np.cos(Omega)  # cos(omega + f)
        omega = np.arctan2(sof, cof) - f
        omega = np.mod(omega, 2 * np.pi)

        E = np.arccos((1 - r / a) / e) # calculate Eccentric Anomaly
        E = np.where(sign < 0.0, 2 * np.pi - E, E)

        M = E - e * np.sin(E)

        n = np.sqrt(mu / a ** 3)  # Kepler's 3rd law to get the mean motion

        varpi = np.mod(omega + Omega, 2 * np.pi)
        lam = np.mod(omega + Omega + M, 2 * np.pi)

        el_elements = pd.DataFrame([[a, e, i, Omega, omega, f, E, M, lam, varpi, n]], columns = ['a', 'e', 'i', 'Omega', 'omega', 'f', 'E', 'M', 'lam', 'varpi', 'n'])

        return el_elements

    @staticmethod
    def danby(M, e):
        """
        Danby's solvers for Eccentric anomaly (E)

        Input Parameters:
            M : Mean Anomaly
            e : eccentricity

        Output Parameters:
            E : Eccentric Anomaly
        """

        def deltaij(E, M, e, j):
            """
            Calculate delta_ij for a given j

            Input Parameters:
                j : the j-th value to calculate

            Output Parameters:
                delta_ij: the delta_ij value


            """

            def f_i_prime(E, M, e, i):
                """
                Calculate f and/or it's i-th derivative for Danby's E solver

                Input Parameters:
                    i : i-th derivative to calculate

                Output Parameters:
                    f_i : i-th derivative of f
                """
                if (i == 0):
                    f_i = E - e * np.sin(E) - M
                elif (i == 1):
                    f_i = 1 - e * np.cos(E)
                elif (i == 2):
                    f_i = e * np.sin(E)
                elif (i == 3):
                    f_i = e * np.cos(E)
                else:
                    raise RuntimeError(f'no derivative for i = {i}')

                return f_i

            if (j == 1):
                delta_ij = -1 * f_i_prime(E, M, e, 0) / f_i_prime(E, M, e, 1)
            elif (j == 2):
                delta_ij = -1 * f_i_prime(E, M, e, 0) / (
                            f_i_prime(E, M, e, 1) + 0.5 * deltaij(E, M, e, 1) * f_i_prime(E, M, e, 2))
            elif (j == 3):
                delta_ij = -1 * f_i_prime(E, M, e, 0) / (
                            f_i_prime(E, M, e, 1) + (0.5 * deltaij(E, M, e, 2) * f_i_prime(E, M, e, 2)) + (
                                deltaij(E, M, e, 2) ** 2 * f_i_prime(E, M, e, 3) / 6))

            return delta_ij

        k = 0.85  # initial guess
        accuracy = 1e-10

        E = M + np.sign(np.sin(M)) * e * k
        for i in range(100):  # Break out if a certain accuracy is not achieved after 50 loops
            Enew = E + deltaij(E, M, e, 3)
            error = np.abs((Enew - E) / E)
            error_check = error[1:] <= accuracy  # error of only the planets

            if (np.all(error_check)):
                return Enew
            E = Enew

        print(f'E = {E}; Enew = {Enew}')
        print(f'Error = {error}')
        raise RuntimeError("The Danby function did not converge on a solution")


    def calc_energy(self):
        """
        Calculates the total energy of the system

        Output:
            Updates the energy of the system

        """

        # Keplerian Energy
        r_vec = np.array(self.xv_elements[-1])[:, 0:3]
        v_vec = np.array(self.xv_elements[-1])[:, 3:6]
        mass = np.array(self.m)

        P = (mass * v_vec.T).T

        H_kepler = np.sum(self.mag(P) ** 2 / (2 * mass) - self.G * mass * self.msun / self.mag(r_vec))

        # Interaction Energy
        H_int = 0
        for i in range(len(self.names) - 1):
            for j in range(i + 1, len(self.names)):
                dr_vec = r_vec[i] - r_vec[j]
                H_int -= self.G * mass[j] * mass[i] / self.mag(dr_vec)

        # Sun Hamiltonian
        H_sun = self.mag(np.sum(P, axis = 0)) ** 2 / (2 * self.msun)

        self.energy.append(H_sun + H_int + H_kepler)

        return


    def kepler_drift_all(self):
        """
        Calculates the Keplerian drift on all bodies

        Output:
            Performs the keplerian drift on all bodies

        """

        def kepler_drift(self, r_vec, v_vec, mu, dt):
            """
            Calculates the Keplerian drift on 1 body

            Input Parameters:
                r_vec : position vector (3)
                v_vec : velocity vector (3)
                mu : mu = G * (m + msun) (scalar)
                dt : time step (scalar)

            Output Parameters:
                r : updated position vector (3)
                v : updated velocity vector (3)
                el_elements : orbital elements (11)

            """

            el_elements = self.xy_to_el(r_vec, v_vec, mu)

            el_elements['M'] = el_elements['M'] + el_elements['n'] * dt

            # Danby's solver

            E_tmp = self.danby(el_elements['M'], el_elements['e'])

            dE = (E_tmp - el_elements['E'])[0]
            el_elements['E'] = E_tmp

            # Danby's conversion to cartesian
            r_0 = self.mag(r_vec)

            f, g = self.danby_f_and_g(dE, r_0, el_elements['a'][0], el_elements['n'][0], dt)

            r = f * r_vec + g * v_vec  # f and g are arrays
            r_mag = self.mag(r)

            f_dot, g_dot = self.danby_f_and_g_dot(dE, r_mag, r_0, el_elements['a'][0], el_elements['n'][0], dt)

            v = f_dot * r_vec + g_dot * v_vec

            return r, v, el_elements

        for i in range(len(self.names)):

            r_vec = self.rh_vec[i]
            v_vec = self.vb_vec[i]
            mu = self.mu[i]

            r_new, v_new, el_elements_new = kepler_drift(self, r_vec, v_vec, mu, self.dt)

            self.rh_vec[i] = r_new
            self.vb_vec[i] = v_new
            self.el_elements_tmp[i] = el_elements_new

        return

    def int_kick(self):
        """
        Calculates the kick from body-to-body interactions

        Input Parameters:

        Output Parameters:


        """
        for i in range(len(self.names)):  # interaction force on this planet
            acc = 0
            for j in range(len(self.names)):  # cycle through remaining planets in the system
                if (i == j):
                    continue
                dr_vec = self.rh_vec[j] - self.rh_vec[i]
                acc += self.G * self.m[j] / np.power(self.mag(dr_vec), 3) * dr_vec  # Force calculation

            self.vb_vec[i] += acc * self.dt / 2

        return

    def sun_drift(self):
        """
        Calculates the linear-sun drift on the planets

        Output:
            Performs the linear-sun drift on the planets
        """

        r_dot =  np.sum((self.m * self.vb_vec.T).T, axis = 0) / self.msun
        self.rh_vec += r_dot * self.dt


        return

    def hc_to_bc(self, vec_h, m):
        """
        Converts a heliocentric vector to a barycentric vector

        Input Parameters:
            vec_h : Heliocentric vectors (n, 3)
            m : Mass of the body related to the vector (n)

        Output Parameters:
            vec_b : Barycentric vector

        """

        origin_b = -np.sum((m * vec_h.T).T, axis=0) / (self.m_tot)
        vec_b = vec_h + origin_b

        return vec_b

    def bc_to_hc(self, vec_b, m):
        """
        Converts a barycentric vector to a heliocentric vector

        Input Parameters:
            vec_b : Barycentric vectors (n, 3)
            m : Mass of the body related to the vector (n)

        Output Parameters:
            vec_b : Heliocentric vectors (n, 3)

        """
        origin_h = -np.sum((m * vec_b.T).T, axis=0) / self.m_sun  # m[0] = mass of the sun
        vec_h = vec_b - origin_h

        return vec_h

    @staticmethod
    def danby_f_and_g(dE, r_0, a, n, dt):
        """
        Calculates the Danby f and g functions

        Input Parameters:
            dE : Difference in Eccentric Anomaly
            r_0 : magnitude of original position vector
            a : semi-major axis
            n : mean motion
            dt : time step

        Output Parameters:
            f : Danby's f function
            g : Danby's g function

        """

        f = a / r_0 * (np.cos(dE) - 1.0) + 1.0
        g = dt + (np.sin(dE) - (dE)) / n  # t - t_0 = dt

        return f, g

    @staticmethod
    def danby_f_and_g_dot(dE, r, r_0, a, n, dt):
        """
        Calculates the Danby f_dot and g_dot functions

        Input Parameters:
            dE: Difference in Eccentric Anomaly
            r: magnitude of updated position vector
            r_0: magnitude of original position vector
            a: semi-major axis
            n: mean motion
            dt : time step

        Output Parameters:
            f_dot: Danby's f_dot function
            g_dot: Danby's g_dot function
        """
        f_dot = -a ** 2 / (r * r_0) * n * (np.sin(dE))
        g_dot = a / r * (np.cos(dE) - 1) + 1

        return f_dot, g_dot

    @staticmethod
    def deltaij(E, M, e, j):
        """
        Calculate delta_ij for a given j

        Input Parameters:
            j : the j-th value to calculate

        Output Parameters:
            delta_ij: the delta_ij value


        """

        def f_i_prime(E, M, e, i):
            """
            Calculate f and/or it's i-th derivative for Danby's E solver

            Input Parameters:
                i : i-th derivative to calculate

            Output Parameters:
                f_i : i-th derivative of f
            """
            if (i == 0):
                f_i = E - e * np.sin(E) - M
            elif (i == 1):
                f_i = 1 - e * np.cos(E)
            elif (i == 2):
                f_i = e * np.sin(E)
            elif (i == 3):
                f_i = e * np.cos(E)
            else:
                raise RuntimeError(f'no derivative for i = {i}')

            return f_i

        if(j == 1):
            delta_ij = -1 * f_i_prime(E, M, e, 0) / f_i_prime(E, M, e, 1)
        elif(j == 2):
            delta_ij = -1 * f_i_prime(E, M, e, 0) / (f_i_prime(E, M, e, 1) + 0.5 * deltaij(E, M, e, 1) * f_i_prime(E, M, e, 2))
        elif(j == 3):
            delta_ij = -1 * f_i_prime(E, M, e, 0) / (f_i_prime(E, M, e, 1) + (0.5 * deltaij(E, M, e, 2) * f_i_prime(E, M, e, 2)) + (deltaij(E, M, e, 2) ** 2 * f_i_prime(E, M, e, 3) / 6))

        return delta_ij

    @staticmethod
    def mag(v):
        """
        Returns the magnitude of a vector

        Input Parameters:
            v : a vector

        Output Parameters:
            v_mag : magnitude of the vector
        """
        return np.linalg.norm(v, axis = np.ndim(v) - 1)

    def update_dataset(self):
        """
        Updates the final output dataset for plotting and analysis
        """
        xv_tmp = np.append(self.rh_vec, self.vb_vec, axis = 1)

        self.calc_energy()
        self.xv_elements.append(xv_tmp)
        self.el_elements.append(self.el_elements_tmp)
        self.E_err.append(np.abs((self.energy[-1] - self.energy[0]) / self.energy[0]))
        self.phi.append(self.calc_phi())

        return

    def add_bodies(self, names):
        """
        Add the names of the planets/bodies' to simulate to the Class Object

        Input Parameters:
            names : Planet names

        Output:
            updates self.names

        """
        self.names = names

        return

    def get_horizon_data(self, names, date):
        """
        Extract Horizon JPL data for added planets

        Input Parameters:
            names : names of planets to extract JPL data for
            date : Date to extract JPL data from

        Output:
            updates self.r_vec, self.vh_vec and self.m

        """
        planetid = {
            'Sun': '0',
            'Mercury': '1',
            'Venus': '2',
            'Earth': '3',
            'Mars': '4',
            'Jupiter': '5',
            'Saturn': '6',
            'Uranus': '7',
            'Neptune': '8',
            'Pluto': '9'
        }

        # Planet MSun/M ratio, which is used to compute the value of mu for a given set of units
        MSun_over_Mpl = {
            'Sun': 1.0,
            'Mercury': 6023600.0,
            'Venus': 408523.71,
            'Earth': 328900.56,
            'Mars': 3098708.,
            'Jupiter': 1047.3486,
            'Saturn': 3497.898,
            'Uranus': 22902.98,
            'Neptune': 19412.24,
            'Pluto': 1.35e8
        }
        start_date = datetime.date.fromisoformat(date)
        stop_date = start_date + datetime.timedelta(days=1)
        stop_date = stop_date.isoformat()

        for planet in names:
            pldata = Horizons(id = planetid[planet], location = '@sun', epochs = {'start' : date, 'stop' : stop_date, 'step' : '1d' })

            df = pldata.vectors().to_pandas()

            # Extract the cartesian position and velocity vectors at the ephemerides_start_date
            xv_vecnames = ['x', 'y', 'z', 'vx', 'vy', 'vz']
            xv_vec = np.array(df[xv_vecnames].iloc[0])

            self.xv_elements.append(xv_vec)
            self.m.append(1 / MSun_over_Mpl[planet])
            self.mu.append((self.m[-1] + self.msun) * self.G)

        return

    def step(self):
        """
        Step through the simulation

        """
        self.sun_drift()
        self.int_kick()
        self.kepler_drift_all()
        self.int_kick()
        self.sun_drift()

        return

    def simulate(self, t_end):
        """
        Start the N-body simulation

        Input Parameters:
            t_end : time at which to stop the simulation

        Output:
            simulates and updates the related quantities for the planets

        """

        while self.t[-1] <= t_end:
            # Set up temporary vectors
            for i in range(len(self.names)):
                self.rh_vec[i] = self.xv_elements[-1][i][0:3] # stores the latest heliocentric position vector
                self.vb_vec[i] = self.xv_elements[-1][i][3:6] # stores the latest barycentric velocity vector

            self.el_elements_tmp = self.el_elements[-1] # stores the latest set of orbital elementss

            # Step
            self.step()

            # Update dataset
            self.t.append(self.t[-1] + self.dt)
            self.update_dataset()

            time_step = self.t[-1]
            if (time_step % (365.25 * 5000) == 0):
                time_step = np.rint((time_step / 365.25) / 1000) * 1000
                print("\n\nAt t ~ {} years".format(time_step))  # Print time steps

        return

    def plot(self, y):
        """
        Plot the given value vs t

        Input parameters:
            y : name/label of the value to plot on the y-axis

        Output:
            Outputs a plot for y vs x
        """
        if(len(self.data[y].dims) == 2):
            self.data[y].plot(hue = "name")
        else:
            self.data[y].plot()

        plt.show()

        return

    def convert_to_xarray(self):

        planet_data = []
        col_names = ['rhx', 'rhy', 'rhz', 'vbx', 'vby', 'vbz', 'a', 'e', 'i', 'Omega', 'omega', 'f', 'E', 'M', 'lam', 'varpi', 'n']
        for i in range(len(self.names)):
            data = np.array(self.xv_elements)[:,i,:]
            data = np.append(data, np.array(self.el_elements)[:,i,0,:], axis = 1)

            df = pd.DataFrame(data, index = np.array(self.t) / 365.25, columns = col_names)
            df.index.name = 't'
            ds = df.to_xarray()

            # Assign attributes
            for c in col_names[0:3]:
                ds[c] = ds[c].assign_attrs(long_name = f'heliocentric $r_{c[2]}$', units = 'AU')

            for c in col_names[3:6]:
                ds[c] = ds[c].assign_attrs(long_name = f'heliocentric $r_{c[2]}$', units = 'AU/day')

            ds['t'] = ds['t'].assign_attrs(long_name = 'time', units = 'years')
            ds['a'] = ds['a'].assign_attrs(long_name = 'semi-major axis', units = 'AU')
            ds['e'] = ds['e'].assign_attrs(long_name = 'eccentricity', units = '')
            ds['i'] = ds['i'].assign_attrs(long_name = 'inclination', units = 'radians')
            ds['Omega'] = ds['Omega'].assign_attrs(long_name = 'longitude of ascneding node', units='radians')
            ds['omega'] = ds['omega'].assign_attrs(long_name = 'argument of periapsis', units='radians')
            ds['f'] = ds['f'].assign_attrs(long_name = 'true_anomaly', units='radians')
            ds['E'] = ds['E'].assign_attrs(long_name = 'eccentric anomaly', units = 'radians')
            ds['M'] = ds['M'].assign_attrs(long_name = 'mean anomaly', units='radians')
            ds['lam'] = ds['lam'].assign_attrs(long_name = 'mean longitude', units='radians')
            ds['varpi'] = ds['varpi'].assign_attrs(long_name = 'varpi', units='radians')
            ds['n'] = ds['n'].assign_attrs(long_name = 'mean motion', units='radians/day')

            ds['name'] = [self.names[i]]
            ds['name'] = ds['name'].assign_attrs(long_name = 'planet name', mass = self.m[i])
            planet_data.append(ds)

        self.data = xr.concat(planet_data, dim = 'name')

        extra_data = pd.DataFrame(self.energy, index = self.t, columns = ['energy'])
        extra_data['E_err'] = self.E_err
        extra_data['phi'] = np.rad2deg(self.phi)
        extra_data.index.name = 't'
        extra_data_ds = extra_data.to_xarray()

        self.data = self.data.assign(energy = (['t'], extra_data['energy']))
        self.data = self.data.assign(E_err = (['t'], extra_data['E_err']))
        self.data = self.data.assign(phi = (['t'], extra_data['phi']))

        self.data['energy'] = self.data['energy'].assign_attrs(long_name='energy of the system', units='Msun*AU^2/day^2')
        self.data['E_err'] = self.data['E_err'].assign_attrs(long_name='relative energy error', units='dimensionless')
        self.data['phi'] = self.data['phi'].assign_attrs(long_name='Neptune-Pluto resonance angle', units='degrees')

        return

    def calc_phi(self):
        """
        Calculate the resonance angle for Neptune and Pluto

        Output parameters:
            phi : resonance angle

        """
        # find the indices of the planets for which we need to calculate the resonance angle

        index_n = 0 # np.where(np.array(self.names) == 'Neptune')
        index_p = 1 # np.where(np.array(self.names) == 'Pluto')

        el_elements_neptune = (self.el_elements[-1])[index_n]
        el_elements_pluto = (self.el_elements[-1])[index_p]
        lam_n = el_elements_neptune['lam']
        lam_p = el_elements_pluto['lam']
        varpi_p = el_elements_pluto['varpi']

        # calculate resonance angle
        phi = np.mod(3 * lam_p - 2 * lam_n - varpi_p, 2 * np.pi)

        return phi

if __name__ == '__main__':
    """
    Inputs needed:
        date: date to extract JPL Horizons data from
        t_end: time to stop the simulation (days)
    """
    date = '2022-10-12'
    t_end = 365.25 * 10**5
    symp = Symplectic_Integrator(['Neptune', 'Pluto'], date)

    symp.simulate(t_end)

    symp.convert_to_xarray()

    symp.plot('E_err')
    symp.plot('phi')
    symp.plot('e')
