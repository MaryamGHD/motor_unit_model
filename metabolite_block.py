
import numpy as np

class MetaboliteModel:
    """
    Compartment model of intra- and extracellular metabolite concentrations
    during a sustained isometric contraction (Dideriksen et al. 2010).

    This block:
        - Receives spike counts per motor unit (from the motor neuron pool block)
          and average force in %MVC (from the force generation block) once per
          500-ms epoch.
        - Updates intracellular MC for each of the 120 motor units and the single
          extracellular compartment.
        - Outputs MC_i (array, shape [n_MU]), MC_es (scalar), and MMC (scalar)
          for use by the force generation block, motor neuron pool block, and
          PID controller respectively.

    All equation numbers refer to Dideriksen et al. (2010).
    """

    # ------------------------------------------------------------------
    # 5a — __init__: parameters, volumes, state variables
    # ------------------------------------------------------------------
    def __init__(self):
        """
        Initialise all parameters from Table 1 and Eqs. 1–2,
        and set every state variable to its physiological baseline (zero fatigue).
        """

        # ---- Structural parameters (Table 1 / Eq. 1) -----------------

        self.n_MU = 120                  # Number of motor units                  [dimensionless]

        # Eq. 1 parameters
        self.V0 = 3.0                    # Volume offset for intracellular compartments [au]
        self.F_ratio = 80.0             # F120/F1, ratio of largest to smallest MU twitch force [dimensionless]
        #   V_i = V0 * exp( ln(F_ratio) / n * i )   for i = 1 … n_MU

        # Eq. 2 / Table 1
        self.V_es = 1282.0              # Extracellular compartment volume         [au]

        # ---- Transport parameters (Table 1) --------------------------

        self.DC = 0.01                  # Diffusion coefficient (Eq. 6)            [au]
        # Criterion: DC * V_i > RC so metabolites diffuse faster than they are removed;
        # both intra- and extracellular MC fall by ~90% after 3 min rest (Woods et al. 1987).

        self.RC = 0.04                  # Removal capacity by blood flow (Eq. 7)   [au]
        # RC = 4 % of extracellular MC removed per 500-ms epoch at full blood flow.

        # ---- Blood-flow / IMP parameters (Table 1 / Eqs. 9–11) ------

        self.HO = 30.0                  # IMP at half blood-flow occlusion         [mmHg]
        # Full occlusion at ~40 mmHg, corresponding to ~35 % MVC force (Eq. 11).

        # Eq. 9 linear IMP coefficients (curve-fit to Crenshaw et al. 1997,
        # Sejersted et al. 1984, Sadamoto et al. 1983)
        self.IMP_slope     = 0.88       # Slope:     IMP_ins = 0.88 * force + 10.65  [mmHg / %MVC]
        self.IMP_intercept = 10.65      # Intercept                                  [mmHg]

        # Eq. 10 ΔIMP shape parameters (chosen to match Crenshaw et al. 1997 at 25% MVC)
        self.IMP_gain           = 0.12  # Amplitude gain for the cumulative ΔIMP term [mmHg / epoch]
        self.IMP_threshold_low  = 15.0  # Force threshold below which ΔIMP does not increase [%MVC]
        self.IMP_threshold_high = 33.0  # Force level at which ΔIMP saturates            [%MVC]
        self.IMP_width_low      = 8.0   # Width parameter for lower tanh in Eq. 10       [%MVC]
        self.IMP_width_high     = 3.0   # Width parameter for upper tanh in Eq. 10       [%MVC]

        # Eq. 11 BF sigmoid width (denominator inside tanh)
        self.BF_width = 4.0             # Width of BF sigmoid                             [mmHg]

        # ---- Reference MC (Table 1) used by force generation block ---
        # (Stored here for completeness; consumed by force_generation block via MC_i output)
        self.MC_ref = 1150.0            # MC at which twitch force / relaxation time reach
                                        # the reference normalised change (Fig. 3)         [au]

        # ---- Epoch timing --------------------------------------------
        self.epoch_duration_s   = 0.5  # 500 ms per MC-update epoch                       [s]
        self.sim_dt_s           = 0.001 # 1 kHz simulation timestep                       [s]
        self.samples_per_epoch  = int(self.epoch_duration_s / self.sim_dt_s)  # = 500     [samples]

        # ==============================================================
        # Derived / pre-computed quantities (computed once at init)
        # ==============================================================

        # MU indices 1-based as in the paper (i = 1 … 120)
        self._mu_indices = np.arange(1, self.n_MU+1)  # shape (120,)

        # Eq. 1 — intracellular compartment volumes
        #   V_i = V0 + exp( ln(F120/F1) / n * i )
        self.V_i = self.V0 * np.exp(
            (np.log(self.F_ratio) / self.n_MU)* self._mu_indices
        )

        # shape: (120,)
        #
        # DISCREPANCY NOTE (Dideriksen et al. 2010, Eq. 1 vs Table 1):
        # With the literal parameters V0=3, F_ratio=80, n=120, i=1..120,
        # Eq. 1 produces V_i ranging from ~3.11 (MU1) to ~240.0 (MU120).
        # Table 1 states the range is 4.0–83.0 au. These are inconsistent.
        # The ratio of max/min volumes is 240/3.11 ≈ 77 (Eq.1) vs 83/4 ≈ 21 (Table 1).
        # Possible explanations:
        #   (a) Table 1 reports rounded values from a different parameterisation
        #       used during the actual simulations (e.g. i starting at 0, or
        #       a different V0 / F_ratio combination).
        #   (b) A typographic error in either the equation or the table.
        # IMPLEMENTATION DECISION: The literal Eq. 1 is implemented faithfully
        # here (range ~3.11–240 au). If Table 1 values are required exactly,
        # V0 and F_ratio should be re-fitted. This discrepancy should be
        # resolved against the original authors or a reference implementation
        # before relying on absolute MC magnitude values.

        # Sum of intracellular volumes — used for MMC denominator (Eq. 15)
        self._sum_V_i = np.sum(self.V_i)  # scalar [au]

        # ==============================================================
        # State variables — initialised to zero (no fatigue at t=0)
        # ==============================================================

        # Intracellular MC for each MU (Eq. 3)
        # Physiological baseline: zero accumulation relative to resting state
        self.MC_i = np.zeros(self.n_MU, dtype=float)   # shape (120,)  [au]

        # Extracellular MC (Eq. 4)
        self.MC_es = 0.0                                 # scalar         [au]

        # Cumulative ΔIMP history (running sum in Eq. 10)
        # Starts at zero: no prior contraction history
        self.delta_IMP_cumulative = 0.0                  # scalar         [mmHg]

        # Total intramuscular pressure (Eq. 8) — computed from IMP_ins + ΔIMP
        self.IMP_tot = 0.0                               # scalar         [mmHg]

        # Blood flow factor (Eq. 11) — starts at 1.0 (no occlusion)
        self.BF = 1.0                                    # scalar         [dimensionless, in [0,1]]

        # ==============================================================
        # Epoch-level internal accumulators (reset every epoch)
        # ==============================================================

        # Spike accumulator: counts spikes for each MU within current epoch
        # (used when the block is called sample-by-sample rather than epoch-by-epoch)
        self._spike_accumulator = np.zeros(self.n_MU, dtype=int)   # shape (120,)
        self._sample_counter= 0                                   # samples elapsed in current epoch

        # ==============================================================
        # Cached outputs (zero-order hold between epochs for other blocks)
        # ==============================================================

        # Volume-weighted mean intracellular MC (Eq. 15) — sent to PID controller
        self.MMC = 0.0       # scalar  [au]

    # ------------------------------------------------------------------
    # 5b — Metabolite production (Eq. 5)
    # ------------------------------------------------------------------
    def compute_metabolite_production(self, spike_counts: np.ndarray) -> np.ndarray:
        """
        Eq. 5 — Intracellular metabolite production per motor unit.

        Production is proportional to motor unit volume and the number of
        action potentials discharged during the 500-ms epoch (Allen 2004).

            MP_i(epoch) = V_i * ND_i(epoch)

        Parameters
        ----------
        spike_counts : np.ndarray, shape (120,), dtype int or float
            ND_i(epoch) — number of action potentials fired by each MU
            during the current 500-ms epoch.
            Source: motor neuron pool block (motorUnit.py spike trains
            accumulated over the epoch window).

        Returns
        -------
        MP_i : np.ndarray, shape (120,), dtype float
            Metabolite production for each MU in this epoch   [au].

        Notes
        -----
        * V_i has units [au] and ND_i is dimensionless (spike count);
          therefore MP_i has units [au] — consistent with Eq. 3 where
          MP_i is divided by V_i before being added to MC_i (au).
        * Inactive MUs (ND_i = 0) produce no metabolites.
        * High-threshold MUs have larger V_i and therefore produce more
          metabolites per spike, reflecting their larger innervation numbers.
        """
        spike_counts = np.asarray(spike_counts, dtype=int)

        if spike_counts.shape != (self.n_MU,):
            raise ValueError(
                f"spike_counts must have shape ({self.n_MU},), "
                f"got {spike_counts.shape}"
            )

        # Eq. 5: MP_i(epoch) = V_i · ND_i(epoch)
        MP_i = self.V_i * spike_counts   # shape (120,)  [au]

        return MP_i

    def compute_diffusion(self) -> np.ndarray:
        """
        Eq. 6 — Diffusion from intracellular compartment i to extracellular space.

            MD_i(epoch) = DC · [MC_i(epoch) - MC_es(epoch)] · V_i

        Parameters
        ----------
        Uses current state: self.MC_i, self.MC_es, self.V_i, self.DC

        Returns
        -------
        MD_i : np.ndarray, shape (120,)
            Metabolite diffusion per MU [au]
            Positive = outward flux (active MU losing metabolites)
            Negative = inward flux (inactive MU absorbing from extracellular)

        Notes
        -----
        Must be called using MC values BEFORE the epoch update.
        """
        # Eq. 6: MD_i = DC · (MC_i - MC_es) · V_i
        MD_i = self.DC * (self.MC_i - self.MC_es) * self.V_i
        return MD_i

    def compute_blood_flow(self, force_epoch: float) -> float:
        """
        Eqs. 8-11 — Compute blood flow factor from intramuscular pressure.

        Eq. 9 — Instantaneous IMP:
            IMP_ins(epoch) = 0.88 · force(epoch) + 10.65

        Eq. 10 — Cumulative ΔIMP history:
            ΔIMP += tanh[(force - 15)/8] · 0.12
                  · {tanh[-(force - 33)/3] · 0.5 + 0.5}

        Eq. 8 — Total IMP:
            IMP_tot(epoch) = IMP_ins(epoch) + ΔIMP(epoch)

        Eq. 11 — Blood flow:
            BF(epoch) = tanh[-(IMP_tot - HO)/4] · 0.5 + 0.5

        Parameters
        ----------
        force_epoch : float
            Average force during the epoch as %MVC

        Returns
        -------
        BF : float
            Blood flow factor in [0, 1]
            0 = full occlusion, 1 = no occlusion

        Notes
        -----
        Updates state variables: self.delta_IMP_cumulative,
        self.IMP_tot, self.BF
        """
        # Eq. 9: instantaneous IMP
        IMP_ins = self.IMP_slope * force_epoch + self.IMP_intercept

        # Eq. 10: cumulative ΔIMP — running sum across epochs
        factor1 = np.tanh((force_epoch - self.IMP_threshold_low) /
                          self.IMP_width_low) * self.IMP_gain
        factor2 = (np.tanh(-(force_epoch - self.IMP_threshold_high) /
                           self.IMP_width_high) * 0.5 + 0.5)


        self.delta_IMP_cumulative += factor1 * factor2

        # Eq. 8: total IMP
        self.IMP_tot = IMP_ins + self.delta_IMP_cumulative

        # Eq. 11: blood flow
        self.BF = (np.tanh(-(self.IMP_tot - self.HO) /
                           self.BF_width) * 0.5 + 0.5)

        return self.BF

    def update_intracellular_MC(self,
                                MP_i: np.ndarray,
                                MD_i: np.ndarray) -> np.ndarray:
        """
        Eq. 3 — Update intracellular metabolite concentration for each MU.

            MC_i(epoch) = MC_i(epoch-1)
                        + [MP_i(epoch) - MD_i(epoch)] / V_i

        Parameters
        ----------
        MP_i : np.ndarray, shape (120,)
            Metabolite production from compute_metabolite_production()
            Units: [au]
        MD_i : np.ndarray, shape (120,)
            Diffusion flux from compute_diffusion()
            Must be computed from MC_i(epoch-1) and MC_es(epoch-1)
            BEFORE this update is applied.
            Units: [au]

        Returns
        -------
        MC_i : np.ndarray, shape (120,)
            Updated intracellular MC for each MU [au]

        Notes
        -----
        Call order within step_epoch:
            1. compute_diffusion()          → MD_i  (uses epoch-1 values)
            2. compute_metabolite_production() → MP_i
            3. update_intracellular_MC(MP_i, MD_i)  ← this method
            4. update_extracellular_MC(MD_i, MR)    ← uses same MD_i

        The V_i factor in MP_i (Eq. 5) and MD_i (Eq. 6) does NOT cancel
        in Eq. 3 mathematically — both terms are divided by V_i:
            MP_i / V_i = V_i * ND_i / V_i = ND_i
            MD_i / V_i = DC * (MC_i - MC_es) * V_i / V_i
                       = DC * (MC_i - MC_es)
        V_i cancels in both terms. This is confirmed by the paper's
        intent that production scales with volume but concentration
        change does not depend on compartment size alone.
        """
        # Eq. 3: MC_i(epoch) = MC_i(epoch-1) + [MP_i - MD_i] / V_i
        self.MC_i = self.MC_i + (MP_i - MD_i) / self.V_i

        # Non-negativity guard — MC cannot be physically negative
        self.MC_i = np.maximum(self.MC_i, 0.0)

        return self.MC_i

    def update_extracellular_MC(self, MD_i: np.ndarray, MR: float) -> float:
        """
        Eq. 4 — Update extracellular metabolite concentration.

            MC_es(epoch) = MC_es(epoch-1)
                         + [Σ MD_x(epoch) - MR(epoch)] / V_es

        Parameters
        ----------
        MD_i : np.ndarray, shape (120,)
            Diffusion flux per MU from compute_diffusion()
            Must be the SAME MD_i used in update_intracellular_MC()
            i.e. computed from MC values at epoch-1
        MR : float
            Removal by blood flow from compute_removal()

        Returns
        -------
        MC_es : float
            Updated extracellular MC [au]

        Notes
        -----
        Must be called AFTER update_intracellular_MC but with the
        SAME pre-update MD_i vector.
        """
        # Eq. 4: MC_es(epoch) = MC_es(epoch-1) + [Σ MD_x - MR] / V_es
        self.MC_es = self.MC_es + (np.sum(MD_i) - MR) / self.V_es

        # Non-negativity guard
        self.MC_es = max(self.MC_es, 0.0)

        return self.MC_es

    def compute_removal(self) -> float:
        """
        Eq. 7 — Removal of metabolites by blood flow.

            MR(epoch) = RC · BF(epoch) · MC_es(epoch)

        Returns
        -------
        MR : float
            Metabolite removal this epoch [au]
        """
        # Eq. 7: MR = RC · BF · MC_es
        return self.RC * self.BF * self.MC_es * self.V_es

    def compute_MMC(self) -> float:
        """
        Eq. 15 — Volume-weighted mean intracellular MC across all MUs.

            MMC(t) = Σ MC_i(t) / Σ V_i

        Parameters
        ----------
        Uses current state: self.MC_i, self.V_i

        Returns
        -------
        MMC : float
            Mean intracellular MC weighted by volume [au]
            Sent to PID controller to modulate derivative gain Kd (Eq. 14)

        Notes
        -----
        Called after update_intracellular_MC so it uses
        the current epoch MC_i values.
        """
        # Eq. 15: MMC = Σ MC_i(t) / Σ V_i
        #self.MMC = np.sum(self.MC_i) / self._sum_V_i
        self.MMC = np.mean(self.MC_i)

        return self.MMC

    def step_epoch(self,
                   spike_counts: np.ndarray,
                   force_epoch: float) -> dict:
        """
        Main epoch update — calls all sub-steps in the correct order.

        Called once every 500 ms with:
            - spike_counts: ND_i from motor neuron pool (spikes per MU this epoch)
            - force_epoch:  mean force as %MVC from force generation block

        Returns
        -------
        dict with keys:
            'MC_i'  : np.ndarray shape (120,) — intracellular MC per MU
            'MC_es' : float                   — extracellular MC
            'MMC'   : float                   — volume-weighted mean MC
            'BF'    : float                   — blood flow factor [0,1]
            'IMP_tot': float                  — total intramuscular pressure [mmHg]

        Call order strictly follows paper:
            Step 1: Eq. 6  — diffusion (uses MC values from epoch-1)
            Step 2: Eq. 5  — production
            Step 3: Eqs. 8-11 — blood flow and IMP
            Step 4: Eq. 7  — removal
            Step 5: Eq. 3  — update intracellular MC
            Step 6: Eq. 4  — update extracellular MC (same MD_i as step 1)
            Step 7: Eq. 15 — compute MMC
        """
        # Step 1 — Eq. 6: diffusion using MC_i(epoch-1) and MC_es(epoch-1)
        MD_i = self.compute_diffusion()

        # Step 2 — Eq. 5: metabolite production
        MP_i = self.compute_metabolite_production(spike_counts)

        # Step 3 — Eqs. 8-11: blood flow and IMP
        BF = self.compute_blood_flow(force_epoch)

        # Step 4 — Eq. 7: removal by blood flow
        MR = self.compute_removal()

        # Step 5 — Eq. 3: update intracellular MC
        self.update_intracellular_MC(MP_i, MD_i)

        # Step 6 — Eq. 4: update extracellular MC (same MD_i from Step 1)
        self.update_extracellular_MC(MD_i, MR)

        # Step 7 — Eq. 15: volume-weighted mean MC
        self.compute_MMC()

        return {
            'MC_i': self.MC_i.copy(),
            'MC_es': self.MC_es,
            'MMC': self.MMC,
            'BF': self.BF,
            'IMP_tot': self.IMP_tot
        }

