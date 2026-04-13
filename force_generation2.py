"""
Force Generation Block - Dideriksen et al. (2010)

This module implements the force generation model exactly as described in:
"An integrative model of motor unit activity during sustained submaximal contractions"
Dideriksen et al., J Appl Physiol 108:1550-1562, 2010

Equations implemented:
- Eq. 37: Motor unit twitch force (Fuglevand base + fatigue gains)
- Eq. 38-40: Contraction time gain (T_gain)  [fatigue only]
- Eq. 41-42: Correction factor (CF)           [fatigue only]
- Eq. 43-44: Amplitude gain (P_gain)          [fatigue only]

Usage modes:
- Non-fatigued (MC=None or MC=0): Pure Fuglevand twitch (Eq. 37 with gains = 1)
- Fatigued (MC > 0): Full Dideriksen gains applied
"""

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import matplotlib as mpl

# ── Global bold style ─────────────────────────────────────────
mpl.rcParams['font.weight']        = 'bold'
mpl.rcParams['axes.titleweight']   = 'bold'
mpl.rcParams['axes.labelweight']   = 'bold'
mpl.rcParams['font.size']          = 12
mpl.rcParams['axes.titlesize']     = 14
mpl.rcParams['axes.labelsize']     = 12
mpl.rcParams['xtick.labelsize']    = 11
mpl.rcParams['ytick.labelsize']    = 11
mpl.rcParams['legend.fontsize']    = 10
mpl.rcParams['lines.linewidth']    = 5
class ForceGenerationBlock:
    """
    Force generation block from Dideriksen et al. (2010).

    Converts motor unit spike trains into muscle force through temporal
    and spatial summation of motor unit twitches.

    Parameters
    ----------
    n_motor_units : int
        Total number of motor units (default: 120, as in paper for FDI)
    dt : float
        Time step in seconds (default: 0.001 s = 1 ms, as in paper)
    P_base : float
        Base peak twitch force for the smallest motor unit (arbitrary units)
    T_base : float
        Base contraction time for the smallest motor unit in seconds
        (default: 0.090 s = 90 ms, from Fuglevand et al. 1993)
    RP : float
        Range of peak forces: ratio of largest to smallest MU twitch
        (default: 100, from Fuglevand et al. 1993 for FDI)
    RT : float
        Range of contraction times: ratio of longest to shortest
        (default: 3.0, from Fuglevand et al. 1993 Table 1)
    """

    def __init__(self, n_motor_units=120, dt=0.001, P_base=1.0,
                 T_base=0.090, RP=100.0, RT=3.0):
        self.n = n_motor_units
        self.dt = dt
        self.P_base = P_base
        self.T_base = T_base
        self.RP = RP
        self.RT = RT

        # Metabolite reference for fatigue equations (Table 1)
        self.MC_ref = 1150.0

        # Initialize motor unit properties (twitch amplitudes and times)
        self._initialize_motor_unit_properties()

        # Initialize online stepping state
        self.reset()

    def _initialize_motor_unit_properties(self):
        """
        Initialize motor unit-specific properties following
        Fuglevand et al. (1993) as referenced in the paper.

        - Peak twitch forces: exponentially distributed (Eq. 13 in Fuglevand)
        - Contraction times: inversely related to peak force
        """
        # Motor unit indices (1 to n)
        self.mu_indices = np.arange(1, self.n + 1)

        # --- Peak twitch force distribution ---
        # Fuglevand Eq. 13: P_i = P_1 * exp(b * (i-1))
        # where b = ln(RP) / (n-1)
        b = np.log(self.RP) / (self.n - 1)
        self.P = self.P_base * np.exp(b * (self.mu_indices - 1))

        # --- Contraction time distribution ---
        # Fuglevand: T_i = T_base * (P_1 / P_i)^(1/c)
        # where c = log(RP) / log(RT), RT = range of contraction times
        # RT = 3 from Fuglevand Table 1 (NOT equal to RP)
        c = np.log(self.RP) / np.log(self.RT)
        self.T = self.T_base * (self.P[0] / self.P) ** (1.0 / c)

    # ==========================================================
    # Online step-by-step interface (for closed-loop simulation)
    # ==========================================================

    def reset(self):
        """Clear all internal state for a new simulation."""
        # Each entry is a list of spike times (in seconds) still producing force
        self.twitch_buffers = [[] for _ in range(self.n)]

    def step(self, t, spike_events, MC=None):
        # 1. Record new spikes
        firing_indices = np.where(spike_events)[0]
        for mu_idx in firing_indices:
            self.twitch_buffers[mu_idx].append(t)

        # 2. Determine fatigue mode
        use_fatigue = MC is not None

        # 3. Compute force from all active twitches
        total_force = 0.0

        for mu_idx in range(self.n):
            if len(self.twitch_buffers[mu_idx]) == 0:
                continue

            i = mu_idx + 1
            P_i = self.P[mu_idx]
            T_i = self.T[mu_idx]

            # Match simulate()'s fatigue logic exactly
            if use_fatigue:
                mc_val = MC[mu_idx] if hasattr(MC, '__len__') else MC
                T_gain = self.compute_T_gain(i, mc_val)
                P_gain = self.compute_P_gain(i, mc_val)
                twitch_scale = 1 - P_gain
            else:
                T_gain = self.compute_T_gain(i, 0)
                twitch_scale = 1 - self.compute_P_gain(i, 0)

            T_scaled = T_gain * T_i
            cutoff = 5.0 * T_scaled

            mu_force = 0.0
            still_active = []

            for spike_idx_j, spike_time in enumerate(self.twitch_buffers[mu_idx]):
                t_rel = t - spike_time

                if t_rel > cutoff:
                    continue

                still_active.append(spike_time)

                if t_rel <= 0:
                    continue

                # Match simulate()'s gain_f logic
                gain_f = 1.0 if spike_idx_j == 0 else \
                    self.compute_gain_f(i,
                                        self.twitch_buffers[mu_idx][spike_idx_j] - self.twitch_buffers[mu_idx][
                                            spike_idx_j - 1],
                                        T_scaled=T_scaled)

                normalized = t_rel / T_scaled
                mu_force += gain_f * twitch_scale * P_i * normalized * np.exp(1.0 - normalized)

            self.twitch_buffers[mu_idx] = still_active
            total_force += mu_force

        return total_force

    # ==========================================================
    # Fatigue gain equations (Eqs. 38-44) — used when MC > 0
    # ==========================================================

    def compute_gain_max(self, i):
        # Eq. 39: confirmed i²/n
        ratio= i/self.n

        return (1.66 * (ratio**2) + 0.25 * (i / self.n) - 0.25)
    def compute_delta_gain_mc(self, MC):
        """Eq. 40: Metabolite-dependent gain factor (0 to 1)."""
        return np.tanh(4.0 * ((MC / self.MC_ref) - 0.5)) * 0.5 + 0.5

    def compute_T_gain(self, i, MC):
        """Eq. 38: Contraction time gain (≥ 1, increases with fatigue)."""
        gain_max = self.compute_gain_max(i)
        delta_gain = self.compute_delta_gain_mc(MC)

        return 1.0 + gain_max * delta_gain

    def compute_b_i(self, i):
        """Eq. 42: b_i parameter for correction factor."""
        n_i = i / self.n
        return 0.389 * np.exp(-4.413 * n_i) + 0.935 * np.exp(0.182 * n_i)

    def compute_CF(self, i, MC):
        """Eq. 41: Correction factor for amplitude during fatigue."""
        T_gain = self.compute_T_gain(i, MC)
        delta_gain = self.compute_delta_gain_mc(MC)
        b_i = self.compute_b_i(i)
        return (1.0 / T_gain) * (1.0 + delta_gain * ((1.0 / b_i )- 1.0))

    def compute_h(self, i):
        # -(i - 0.67n) not (-i - 0.67n)
        #s_i = (np.tanh((i - 0.75 * self.n) / (0.34 * self.n)) * 1.04 +
        #np.tanh(-(i - 0.67 * self.n) / (0.37 * self.n)) * 0.95 + 0.97)

        s_i = (np.tanh((i - 0.75 * self.n) / (0.34 * self.n)) * 1.04 +
               np.tanh(-((i - 0.67 * self.n) / (0.37 * self.n))) * 0.95 + 0.97)

        t_i = np.tanh(i / (0.12 * self.n )+ 1.0) * 0.13+ 1.0
        #t_i = np.tanh(i / (0.12 * self.n) + 1.0) *13 + 1.0

        u_i = np.tanh((i - self.n) / (0.04 * self.n) + 1.0) * 0.06 + 1.0
        return (-2.0 * i / self.n + 2.88) * s_i * t_i * u_i
        #n_i = i / self.n
        #return 2.0 * np.exp(-1.75 * n_i)



        # Linear decrease: h(1)≈2.30, h(60)≈0.70, h(120)≈0.40









    def compute_P_gain(self, i, MC):
        CF = self.compute_CF(i, MC)
        h_i = self.compute_h(i)
        ratio=0.5 * h_i


        return (1 * ((np.tanh((MC / self.MC_ref - h_i) / (ratio)) * 0.5 + 0.5)))
    # ==========================================================
    # Single twitch computation (for testing / inspection)
    # ==========================================================
    def compute_twitch(self, t, i, MC=0, gain_f=1.0):
        """
        Compute motor unit twitch force (Eq. 37 Dideriksen / Eq. 18 Fuglevand).

        Parameters
        ----------
        t      : float or ndarray — time after spike (seconds)
        i      : int              — MU index (1-indexed)
        MC     : float or None    — metabolite concentration
                                    None = no fatigue
        gain_f : float            — twitch summation gain (Eqs. 16-17)
                                    computed externally from T_i/ISI
        """
        idx = i - 1
        P = self.P[idx]
        T = self.T[idx]

        # Only apply fatigue gains if MC is provided and positive
        use_fatigue = (MC is not None) and (float(np.asarray(MC).flat[0]) > 0)


        T_gain = self.compute_T_gain(i, MC)

        P_gain = self.compute_P_gain(i, MC)
        twitch_scale = 1- P_gain  # P_gain = force LOSS fraction


        t = np.maximum(t, 0.0)
        T_scaled = T_gain * T
        normalized_time = t / T_scaled

        # Eq. 18: f(t) = gain_f · twitch_scale · P · (t/T) · exp(1 - t/T)
        return gain_f * twitch_scale * P * normalized_time * np.exp(1.0 - normalized_time)

    def compute_gain_f(self, i, ISI, T_scaled=None):
        if ISI is None or ISI <= 0:
            return 1.0

        T_i = T_scaled if T_scaled is not None else self.T[i - 1]
        ratio = T_i / ISI

        if ratio <= 0.4:
            return 1.0

        # Sigmoid region
        S = 1.0 - np.exp(-2.0 * ratio ** 3)
        g = S / ratio

        # ← normalization MUST stay: makes gain_f=1.0 at boundary
        # and INCREASES above 1.0 for higher ratios (more summation)
        S_04 = 1.0 - np.exp(-2.0 * 0.4 ** 3)
        g_04 = S_04 / 0.4

        return g / g_04  # ← this gives gain_f > 1 for ratio > 0.4


    def simulate(self, spike_times, duration, MC=None):
        n_steps = int(duration / self.dt)
        time = np.arange(n_steps) * self.dt
        force_mu = np.zeros((self.n, n_steps))

        # FIXED: proper use_fatigue definition
        use_fatigue = MC is not None

        for mu_idx in range(self.n):
            i = mu_idx + 1
            spikes = spike_times[mu_idx]

            if len(spikes) == 0:
                continue

            P_i = self.P[mu_idx]
            T_i = self.T[mu_idx]

            # FIXED: guard before calling compute functions
            if use_fatigue:
                mc_val = MC[mu_idx] if hasattr(MC, '__len__') else MC
                T_gain = self.compute_T_gain(i, mc_val)
                P_gain = self.compute_P_gain(i, mc_val)
                twitch_scale =1-P_gain

            else:
                T_gain = self.compute_T_gain(i, 0)
                twitch_scale = 1-self.compute_P_gain(i, 0)

            T_scaled = T_gain * T_i

            for spike_idx_j, spike_time in enumerate(spikes):
                gain_f = 1.0 if spike_idx_j == 0 else \
                    self.compute_gain_f(i, spike_time - spikes[spike_idx_j - 1],T_scaled=T_scaled)

                spike_idx = int(spike_time / self.dt)
                if spike_idx >= n_steps:
                    continue

                t_rel = time[spike_idx:] - spike_time
                t_rel = np.maximum(t_rel, 0.0)
                normalized = t_rel / T_scaled
                twitch = gain_f* twitch_scale *P_i* normalized * np.exp(1.0 - normalized)
                force_mu[mu_idx, spike_idx:] += twitch

        return {
            'time': time,
            'force_total': np.sum(force_mu, axis=0),
            'force_mu': force_mu
        }
    # ==========================================================
    # Plotting utility
    # ==========================================================

    def plot_force(self, results, mu_indices=None, title='Motor Unit Force Generation'):
        """Plot force traces similar to Figure 4 in Dideriksen et al. (2010)."""
        fig, ax = plt.subplots(figsize=(10, 6))
        time_ms = results['time'] * 1000

        if mu_indices is not None:
            for mu_idx in mu_indices:
                idx = mu_idx - 1
                ax.plot(time_ms, results['force_mu'][idx, :],
                        label=f'MU{mu_idx}', linewidth=1.5)

        ax.plot(time_ms, results['force_total'],
                label='Total Force', linewidth=2, color='black', linestyle='--')

        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Force (au)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig, ax

    # ==========================================================
    # Debug helper
    # ==========================================================

    def debug_force_computation(self, spike_times, current_time, mu_indices=[0, 1, 2]):
        """Debug helper to show force computation details."""
        print(f"\n=== FORCE DEBUG at t={current_time:.3f}s ===")
        total_force = 0.0

        for mu_idx in mu_indices:
            i = mu_idx + 1
            spikes = spike_times[mu_idx] if mu_idx < len(spike_times) else []

            print(f"\nMU #{i}:")
            print(f"  P = {self.P[mu_idx]:.4f}, T = {self.T[mu_idx]*1000:.1f} ms")
            print(f"  Total spikes: {len(spikes)}")

            if len(spikes) == 0:
                continue

            mu_force = 0.0
            max_duration = 5 * self.T[mu_idx]

            for spike_time in spikes:
                t_since = current_time - spike_time
                if t_since < 0 or t_since > max_duration:
                    continue
                twitch = self.compute_twitch(t=t_since, i=i, MC=None)
                mu_force += twitch

            print(f"  Force contribution: {mu_force:.4f}")
            total_force += mu_force

        print(f"\nTotal force (from {len(mu_indices)} MUs): {total_force:.4f}")
        print("=" * 50)


