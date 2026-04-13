import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.io
from matplotlib.ticker import NullFormatter
plt.rcParams['font.family'] = 'Times New Roman'
class MotorNeuronPool:
    def __init__(self, n_motor_units=120, dt=0.001):
        """
        Initialize Motor Neuron Pool (Dideriksen et al. 2010)

        Parameters:
            n_motor_units: 120 (number of motor units for FDI, from Table 1)
            dt: 0.001 s (1 ms time step, simulation runs at 1000 Hz)
        """
        # ==========================================
        # Basic Parameters
        # ==========================================
        self.n = n_motor_units
        self.dt = dt  # seconds

        # ==========================================
        # Model Parameters (from Table 1)
        # ==========================================
        self.MDR1 = 6.2  # pps (minimum discharge rate, smallest MU)
        self.MDRD = 10.0  # pps (difference in MDR between smallest and largest)
        self.PDR1 = 15.6  # pps (peak discharge rate, smallest MU)
        self.PDRD = 15.6  # pps (difference in PDR between smallest and largest)
        self.HM = 450  # au (MC_es at half-maximal inhibition)

        # Recruitment range parameter (stated in text, not in Table 1)
        self.recruitment_range = 30

        # ==========================================
        # STEP 0: Compute Time-Invariant Constants
        # ==========================================

        # Equation 18: Recruitment threshold excitation
        a = np.log(self.recruitment_range) / self.n
        self.RTE = np.exp(a * np.arange(1, self.n + 1))


        # Equation 19: Minimum discharge rate per motor unit
        self.MDR = self.MDR1 + self.MDRD * (self.RTE / self.RTE[-1])

        # Peak discharge rate per motor unit (analogous to Eq. 19, stated in text)
        self.PDR = self.PDR1 + self.PDRD * (self.RTE / self.RTE[-1])

        # Equation 20: Maximum excitation per motor unit
        self.Emax = self.RTE + (self.PDR - self.MDR)

        # Equation 26: Coefficient k for descending drive proportion
        #k_i = (0.7 * EI_i + 0.3 * RTE_i - 0.3 * MDR_i) / EI_i
        # At any excitation level, this simplifies to a function of RTE and MDR
        # We need to compute this properly, but for initialization we use RTE as reference



        self.k = (0.7 * self.Emax + 0.3 * self.RTE - 0.3 * self.MDR) / self.Emax
        #self.k = np.full(self.n, 0.7)




        # This simplifies to: k = 1.0 - 0.3 * MDR / RTE

        # Equation 30: Maximum inhibition per motor unit
        i_array = np.arange(1, self.n + 1)
        self.Inh_max = 0.78 - 0.14 * (i_array / self.n)

        # Equation 31: Excitation at maximal inhibition (CORRECTED: +RTE not -RTE)
        self.Exc_inh_max = self.PDR * self.Inh_max + self.RTE - self.MDR
        #I shoudl check how the model works in both case -RTE and + RTE there might be a mistake of the paper

        # ==========================================
        # STEP 1: Initialize Dynamic State Variables
        # ==========================================
        self.spike_train_history = [[] for _ in range(self.n)]

        # H-reflex accumulator (Eq. 29 constraint: range [0.6, 1.0])
        self.sum_deltaHR = 1.0  # Neutral H-reflex state (scalar, applies to all MUs)

        # Discharge rate per motor unit (array of length n)
        self.DR = np.zeros(self.n)

        # Spike timing state (arrays of length n)
        self.last_spike_time = np.full(self.n, -np.inf)  # No previous spikes
        self.next_spike_time = np.full(self.n, np.inf)  # No scheduled spikes


    def receive_external_inputs(self, REI_t, MC_es_t, F_t):
        """
        Step 2: Receive External Inputs (Dideriksen et al. 2010)

        Inputs at time t:
            REI_t (float): Required excitatory input from PID controller
            MC_es_t (float): Extracellular metabolite concentration from metabolite model
            F_t (float): Current force from isometric force model

        State variables updated: None (inputs are stored for use in subsequent steps)
        """
        self.REI_t = REI_t  # Required excitatory input
        self.MC_es_t = MC_es_t  # Extracellular metabolite concentration
        self.F_t = F_t  # Current force

    def compute_excitatory_input_per_motor_unit(self):
        """
        Step 3: Compute Excitatory Input per Motor Unit (Dideriksen et al. 2010)

        Equations 21-22:
            if REI(t) < Emax[i]:
                EI[i](t) = REI(t)
            else:
                EI[i](t) = Emax[i]

        Required:
            Input: self.REI_t
            Constants: self.Emax (array of length n)

        Updates: self.EI (array of length n)
        """
        # Vectorized implementation of Eqs. 21-22
        self.EI = np.minimum(self.REI_t, self.Emax)


    def compute_descending_drive(self):
        """
        Step 4: Compute Descending Drive (Dideriksen et al. 2010)

        Equation 23:
            DD[i](t) = k[i] * EI[i](t)

        Required:
            Constants: self.k (array of length n)
            Current: self.EI (array of length n)

        Updates: self.DD (array of length n)
        """
        self.DD = self.k * self.EI

    def update_H_reflex_accumulator(self, t):
        """
        Step 5: Update H-reflex Accumulator (Dideriksen et al. 2010)

        Equations 27-28:
            During contraction [F(t) > 0]:
                ΔHR(t) = exp(-t / (3×10^6 / F(t))) × 0.4 × (-1) / (3×10^6 / F(t)) × F(t)

            During recovery [F(t) = 0]:
                ΔHR(t) = 0.4 / (3×10^5)

        Equation 29 constraint:
            sum_ΔHR(t) ∈ [0.6, 1.0]

        Required:
            Input: self.F_t (current force)
            State: self.sum_deltaHR (previous accumulator value)
            Parameter: t (current time in ms)

        Updates: self.sum_deltaHR (scalar)
        """
        # Compute instantaneous H-reflex change
        if self.F_t > 0:
            # During contraction (Eq. 27)
            tau = 3e6 / self.F_t
            deltaHR = np.exp(-t / tau) * 0.4 * (-1 / tau)
        else:
            # During recovery (Eq. 28)
            deltaHR = 0.4 / 3e5

        # Update accumulator
        self.sum_deltaHR = self.sum_deltaHR + deltaHR * self.dt

        # Enforce constraint [0.6, 1.0]
        if self.sum_deltaHR < 0.6:
            self.sum_deltaHR = 0.6
        elif self.sum_deltaHR > 1.0:
            self.sum_deltaHR = 1.0

    def compute_excitatory_afferent_input(self):
        """
        Step 6: Compute Excitatory Afferent Input (Dideriksen et al. 2010)

        Equation 29:
            EAI[i](t) = (1 - k[i]) × EI[i](t) × sum_ΔHR(t)

        Required:
            Constants: self.k (array of length n)
            Current: self.EI (array of length n)
            State: self.sum_deltaHR (scalar)

        Updates: self.EAI (array of length n)
        """
        self.EAI = (1 - self.k) * self.EI * self.sum_deltaHR

    def compute_inhibitory_input_magnitude(self):
        """
        Step 7: Compute Inhibitory Input Magnitude (Dideriksen et al. 2010)

        Equation 32:
            IIM[i](t) = tanh((MC_es(t) - HM) / (HM / 2)) × 0.5 + 0.5

        Range: IIM[i](t) ∈ [0, 1]

        Required:
            Input: self.MC_es_t (extracellular metabolite concentration)
            Parameter: self.HM = 450 (from Table 1)

        Updates: self.IIM (array of length n, same value for all i)
        """
        # Note: IIM is the same for all motor units (depends only on MC_es)
        IIM_value = np.tanh((self.MC_es_t - self.HM) / (self.HM / 2)) * 0.5 + 0.5

        # Broadcast to array for consistency with other computations
        self.IIM = np.full(self.n, IIM_value)

    def step(self, REI_t, MC_es_t, F_t, t):
        """Single timestep: wraps all individual compute methods."""
        self.receive_external_inputs(REI_t, MC_es_t, F_t)
        self.compute_excitatory_input_per_motor_unit()

        # Inline dynamic k (Bug fix 3 — replaces precomputed k from __init__)
        safe_EI = np.where(self.EI > 0, self.EI, 1.0)
        self.k = np.clip(
            (0.7 * self.EI + 0.3 * self.RTE - 0.3 * self.MDR) / safe_EI,
            0.0, 1.0
        )

        self.compute_descending_drive()
        self.update_H_reflex_accumulator(t)
        self.compute_excitatory_afferent_input()
        self.compute_inhibitory_input_magnitude()
        self.compute_inhibitory_afferent_input()
        self.compute_synaptic_noise_magnitude()
        self.generate_synaptic_noise()
        self.compute_net_synaptic_input()
        self.compute_instantaneous_discharge_rate()
        self.generate_action_potentials(t)
        return self.spike_events.copy(), self.DR.copy()
    def compute_inhibitory_afferent_input(self):
        """
        Step 8: Compute Inhibitory Afferent Input (Dideriksen et al. 2010)

        Equation 33:
            IAI[i] = IIM[i](t) × (Emax[i] - Exc_inh_max[i])

        Required:
            Current: self.IIM (array of length n)
            Constants: self.Emax (array of length n)
                       self.Exc_inh_max (array of length n)

        Updates: self.IAI (array of length n)
        """
        self.IAI = self.IIM * (self.Emax - self.Exc_inh_max)
        #self.IAI = np.zeros(self.n)

    def compute_synaptic_noise_magnitude(self):
        """
        Step 9: Compute Synaptic Noise Magnitude (Dideriksen et al. 2010)

        Equation 36:
            SNM[i](t) = sqrt(DD[i](t)^2 + EAI[i](t)^2 + IAI[i](t)^2)

        Required:
            Current: self.DD (array of length n)
                     self.EAI (array of length n)
                     self.IAI (array of length n)

        Updates: self.SNM (array of length n)
        """
        self.SNM = np.sqrt(self.DD ** 2 + self.EAI ** 2 + self.IAI ** 2)

    def generate_synaptic_noise(self):
        """
        Step 10: Generate Synaptic Noise (Dideriksen et al. 2010)

        Equation 35:
            SN[i](t) = N(0, 1/6) × SNM[i](t)

        where N(0, 1/6) is a random sample from normal distribution
        with mean = 0 and standard deviation = 1/6

        Required:
            Current: self.SNM (array of length n)
            Random number generator

        Updates: self.SN (array of length n)
        """
        # Generate random samples from N(0, 1/6) for each motor unit
        noise_samples = np.random.normal(loc=0.0, scale=1 / 6, size=self.n)


        self.SN = noise_samples * self.SNM


    def compute_net_synaptic_input(self):
        """
        Step 11: Compute Net Synaptic Input (Dideriksen et al. 2010)

        Equation 34:
            NSI[i](t) = DD[i](t) + EAI[i](t) - IAI[i](t) + SN[i](t)

        Required:
            Current: self.DD (array of length n)
                     self.EAI (array of length n)
                     self.IAI (array of length n)
                     self.SN (array of length n)

        Updates: self.NSI (array of length n)
        """

        self.NSI = self.DD + self.EAI - self.IAI + self.SN

    def compute_instantaneous_discharge_rate(self):
        # Check recruitment
        recruited = self.NSI >= self.RTE

        # Paper Eq. 16: DR = NSI - RTE + MDR
        self.DR = self.NSI - self.RTE + self.MDR

        # Clip to valid range
        self.DR = np.clip(self.DR, self.MDR, self.PDR)

        # Non-recruited units fire at 0
        self.DR[self.NSI < self.RTE] = 0.0




    def generate_action_potentials(self, t):
        recruited = self.DR > 0

        ISI = np.zeros(self.n)
        ISI[recruited] = 1.0 / self.DR[recruited]  # ← seconds (was 1000/DR)

        never_scheduled = self.next_spike_time == np.inf
        newly_recruited = recruited & never_scheduled

        if np.any(newly_recruited):
            random_offsets = np.random.uniform(0, 1, self.n) * ISI
            self.next_spike_time[newly_recruited] = t + random_offsets[newly_recruited]

        fires = recruited & (t >= self.next_spike_time)

        self.spike_events = fires
        self.last_spike_time[fires] = t
        self.next_spike_time[fires] = t + ISI[fires]

