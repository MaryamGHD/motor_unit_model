#!/usr/bin/python
"""
PID Control Algorithm - Dideriksen et al. (2010)

Implements the descending drive estimation algorithm described in:
"An integrative model of motor unit activity during sustained submaximal contractions"
Dideriksen et al., J Appl Physiol 108:1550-1562, 2010

Equations implemented:
- Eq. 12: Error function
- Eq. 13: Required excitatory input (REI)
- Eq. 14: Time-varying derivative gain (Kd)
- Eq. 15: Mean metabolite concentration (MMC)
"""
import numpy as np


class DescendingDriveController:
    """
    PID-based descending drive estimator from Dideriksen et al. (2010).

    Estimates the required excitatory input (REI) to the motoneuron pool
    to maintain a prescribed target force profile during fatiguing contractions.

    Parameters
    ----------
    target_force : float
        Target force as a fraction of MVC (e.g., 0.20 for 20% MVC)
    max_REI : float
        Maximum allowable REI output (Emax of largest motor unit)
    dt : float
        Simulation timestep in seconds (default: 0.001 s = 1 ms)
    update_interval : float
        Controller update interval in seconds (default: 1/3 s = 3 Hz)
    """

    def __init__(self, target_force, max_REI, dt=0.001, update_interval=1 / 3):

        # Fixed gains from Table 1 / Section "Estimation of descending drive"
        self.Kp = 2e-1 # Eq. 13
        self.Ki = 2.5e-1 # Eq. 13
        #self.Kd = 2.5e-4 # Eq. 14 - computed dynamically, starts near zero

        # Timing
        self.dt = dt
        self.update_interval = update_interval  # 3 Hz = ~333 ms
        self.time_since_update = 0.0  # tracks time since last 3 Hz update

        # Force targets
        self.target_force = target_force  # Ft in Eq. 12, in %MVC
        self.max_REI = max_REI  # output clamp upper bound

        self.clear()

    def clear(self):
        """Clears PID computations and resets all internal state."""

        # Error terms
        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0

        # Integral windup guard (in %MVC units)
        self.windup_guard = 50 # 50% MVC max integral accumulation

        # Output
        self.output = 0.0  # current REI value

        # MMC state
        self.current_MMC = 0.0  # mean metabolite concentration (Eq. 15)
        self.time_since_update = 0.0


    def compute_MMC(self, MC_array, V_array):
        """
        Compute mean metabolite concentration across all motor units (Equation 15).

        Parameters
        ----------
        MC_array : ndarray
            Intracellular metabolite concentration for each motor unit (shape: n_motor_units,)
        V_array : ndarray
            Volume of each motor unit intracellular compartment (shape: n_motor_units,)

        Returns
        -------
        float
            Mean metabolite concentration (MMC)
        """
        # Eq. 15: MMC(t) = sum(MC_i(t)) / sum(V_i)
        return np.sum(MC_array) / np.sum(V_array)

    def compute_MMC_placeholder(self, n_motor_units):
        """
        Placeholder MMC computation when metabolite model is not available.
        Returns MMC = 0 (no fatigue).

        Parameters
        ----------
        n_motor_units : int
            Number of motor units

        Returns
        -------
        float
            MMC = 0.0
        """
        return 0.0

    def debug_state(self, actual_force):
        """
        Print detailed PID state for debugging.

        Parameters
        ----------
        actual_force : float
            Current force as fraction of MVC
        """
        error = self.target_force - actual_force

        print(f"\n=== PID DEBUG ===")
        print(f"Target force: {self.target_force * 100:.2f}% MVC")
        print(f"Actual force: {actual_force * 100:.2f}% MVC")
        print(f"Error: {error * 100:.2f}% MVC")
        print(f"\nPID Terms:")
        print(f"  P term: {self.PTerm:.6f}")
        print(f"  I term: {self.ITerm:.6f}")
        print(f"  D term: {self.DTerm:.6f}")
        print(f"\nGains:")
        print(f"  Kp: {self.Kp:.2e}")
        print(f"  Ki: {self.Ki:.2e}")
        print(f"  Kd: {self.Kd:.6f}")
        print(f"\nOutput:")
        print(f"  Raw: {self.PTerm + (self.Ki * self.ITerm) + self.DTerm:.6f}")
        print(f"  Clamped (REI): {self.output:.6f}")
        print(f"  MMC: {self.current_MMC:.2f}")
        print("=" * 40)

    def compute_Kd(self, MMC):
        """
        Compute time-varying derivative gain as a function of mean
        metabolite concentration (Equation 14).

        Parameters
        ----------
        MMC : float
            Mean metabolite concentration across all motor unit compartments

        Returns
        -------
        float
            Derivative gain Kd(t)

        Notes
        -----
        Kd grows exponentially with MMC, which progressively impairs
        the controller's ability to track the target force and produces
        the experimentally observed increase in force variability
        toward task failure.
        """
        # Eq. 14: Kd(t) = 1.6e-2 * exp(3.5e-8 * MMC(t)) - 5.9e-2
        return 1.6e-2 * np.exp(3.5e-3 * MMC) + 5.9e-2

    def update(self, actual_force, MC_array=None, V_array=None):
        """
        Update the PID controller and compute required excitatory input (Equation 13).

        Should be called every simulation timestep (1 ms), but only recomputes
        REI at 3 Hz intervals. Between updates, holds the last REI value.

        Parameters
        ----------
        actual_force : float
            Current force produced by the model as fraction of MVC
        sim_time : float
            Current simulation time in seconds
        MC_array : ndarray
            Intracellular metabolite concentration for each motor unit
        V_array : ndarray
            Volume of each motor unit intracellular compartment

        Returns
        -------
        float
            Required excitatory input (REI) to the motoneuron pool
        """
        # Accumulate time since last 3 Hz update
        self.time_since_update += self.dt

        # Only recompute at 3 Hz (every ~333 ms)
        if self.time_since_update < self.update_interval:
            return self.output

        # Reset update timer
        self.time_since_update = 0.0

        # Eq. 12: error = target force - actual force (both in %MVC)
        error = self.target_force - actual_force

        # Time elapsed since last update (should be ~333 ms)
        delta_time = self.update_interval
        delta_error = error - self.last_error

        # --- Proportional term ---
        self.PTerm = self.Kp * error

        # --- Integral term with windup guard ---
        self.ITerm += error * delta_time
        #self.ITerm = np.clip(self.ITerm, 0.0, self.windup_guard)
        self.ITerm = np.clip(self.ITerm, -self.windup_guard, self.windup_guard)

        # --- Derivative term ---
        # Update MMC and Kd from current metabolite state
        if MC_array is not None and V_array is not None:
            self.current_MMC = self.compute_MMC(MC_array, V_array)
        else:
            # Placeholder: no metabolite model available yet
            self.current_MMC = 0.0

        self.Kd = self.compute_Kd(self.current_MMC)

        if delta_time > 0:
            self.DTerm = self.Kd * (delta_error / delta_time)
        else:
            self.DTerm = 0.0

        # Store error for next update
        self.last_error = error

        # Eq. 13: REI(t) = Kp*e(t) + Ki*integral(e) + Kd*de/dt
        raw_output = self.PTerm + (self.Ki * self.ITerm) + self.DTerm

        # Clamp output to valid physiological range [0, max_REI]
        self.output = np.clip(raw_output, 0.0, self.max_REI)

        return self.output

    def set_target_force(self, target_force):
        """
        Update the target force setpoint.

        Parameters
        ----------
        target_force : float
            New target force as fraction of MVC (e.g., 0.30 for 30% MVC)
        """
        self.target_force = target_force

    def set_windup_guard(self, windup):
        """
        Update the integral windup guard.

        Parameters
        ----------
        windup : float
            Maximum absolute value of the integral term (in %MVC units)
        """
        self.windup_guard = windup

    def set_max_REI(self, max_REI):
        """
        Update the output clamp upper bound.

        Parameters
        ----------
        max_REI : float
            Maximum allowable REI (Emax of largest motor unit)
        """
        self.max_REI = max_REI

    def get_state(self):
        """
        Return current internal state for logging and debugging.

        Returns
        -------
        dict
            Dictionary containing current values of all internal state variables
        """
        return {
            'output': self.output,
            'PTerm': self.PTerm,
            'ITerm': self.ITerm,
            'DTerm': self.DTerm,
            'Kd': self.Kd,
            'MMC': self.current_MMC,
            'last_error': self.last_error
        }

