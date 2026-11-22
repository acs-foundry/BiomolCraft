#![forbid(unsafe_code)]

//! Mechanistic Human Biologics Minimal-PBPK + FcRn Recycling + Quasi-Steady-State (QSS)
//! Target-Mediated Drug Disposition (TMDD) RK4 ODE Simulator (84-day IV/SC regimens,
//! Receptor Occupancy `RO%`, maintenance-dose solver, and SC volume feasibility).

use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

use crate::adc::AdcError;

/// Biologic modality format with calibrated human (70 kg) FcRn & renal PK parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BiologicsFormat {
    Igg1Wt,
    Igg1YteLs,
    Igg4S228p,
    VhhFcOrHsa,
    Fab,
    ScFv,
    VhhMonomer,
    Adc,
}

impl BiologicsFormat {
    pub fn label(self) -> &'static str {
        match self {
            Self::Igg1Wt => "IgG1-WT (148 kDa)",
            Self::Igg1YteLs => "IgG1-YTE/LS Half-Life Extended (148 kDa)",
            Self::Igg4S228p => "IgG4-S228P (146 kDa)",
            Self::VhhFcOrHsa => "VHH-Fc / VHH-anti-HSA (78 kDa)",
            Self::Fab => "Fab Fragment (48 kDa)",
            Self::ScFv => "scFv (27 kDa)",
            Self::VhhMonomer => "VHH Nanobody Monomer (14.5 kDa)",
            Self::Adc => "ADC (IgG1 + Linker/Payload, 155 kDa)",
        }
    }

    pub fn from_str_lossy(s: &str) -> Self {
        match s
            .trim()
            .to_ascii_lowercase()
            .replace(['-', ' '], "_")
            .as_str()
        {
            "igg1_yte" | "igg1_ls" | "yte" | "ls" | "igg1_yte_ls" => Self::Igg1YteLs,
            "igg4" | "igg4_s228p" => Self::Igg4S228p,
            "vhh_fc" | "vhh_hsa" | "vhh_antihsa" | "nanobody_fc" => Self::VhhFcOrHsa,
            "fab" => Self::Fab,
            "scfv" => Self::ScFv,
            "vhh" | "vhh_mono" | "nanobody" => Self::VhhMonomer,
            "adc" => Self::Adc,
            _ => Self::Igg1Wt,
        }
    }

    /// Returns `(mw_kda, cl_l_day, vc_l, vp_l, q_l_day, f_sc, ka_sc_day, k_deconj_day)`.
    fn human_params(self) -> (f64, f64, f64, f64, f64, f64, f64, f64) {
        match self {
            Self::Igg1Wt => (148.0, 0.21, 3.0, 2.7, 0.55, 0.70, 0.28, 0.0),
            Self::Igg1YteLs => (148.0, 0.068, 3.0, 2.7, 0.55, 0.72, 0.28, 0.0),
            Self::Igg4S228p => (146.0, 0.23, 3.0, 2.6, 0.52, 0.68, 0.28, 0.0),
            Self::VhhFcOrHsa => (78.0, 0.30, 3.2, 2.8, 0.90, 0.72, 0.35, 0.0),
            Self::Fab => (48.0, 4.20, 3.5, 3.2, 5.50, 0.65, 0.55, 0.0),
            Self::ScFv => (27.0, 14.5, 3.6, 3.0, 12.0, 0.60, 0.85, 0.0),
            Self::VhhMonomer => (14.5, 32.0, 3.8, 2.8, 22.0, 0.62, 1.20, 0.0),
            Self::Adc => (155.0, 0.42, 3.1, 2.5, 0.50, 0.50, 0.22, 0.069),
        }
    }
}

/// Route of administration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DosingRoute {
    IvBolus,
    Subcutaneous,
}

impl DosingRoute {
    pub fn label(self) -> &'static str {
        match self {
            Self::IvBolus => "IV",
            Self::Subcutaneous => "SC",
        }
    }

    pub fn from_str_lossy(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "sc" | "subq" | "subcutaneous" => Self::Subcutaneous,
            _ => Self::IvBolus,
        }
    }
}

/// Input parameters for a human Biologics Minimal-PBPK + TMDD simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiologicsPkRequest {
    pub name: String,
    pub format: BiologicsFormat,
    pub route: DosingRoute,
    pub dose_mg: f64,
    pub interval_days: f64,
    pub duration_days: f64,
    /// Binding equilibrium / quasi-steady-state constant `K_ss ≈ K_D` (nM).
    pub kd_nm: f64,
    /// Baseline free target antigen concentration `R0` in central plasma/tissue pool (nM).
    pub target_r0_nm: f64,
    /// Target degradation rate constant `k_deg` (1/day).
    pub target_kdeg_day: f64,
    /// Drug–target complex internalization rate `k_int` (1/day; > k_deg drives TMDD sink).
    pub complex_kint_day: f64,
    /// Optional override for ADC linker deconjugation rate (`ln(2)/t_half_linker`, 1/day).
    pub adc_deconj_rate_day: Option<f64>,
    /// Formulation concentration (mg/mL) for SC injection volume check (default 150 mg/mL).
    pub formulation_mg_ml: f64,
}

impl Default for BiologicsPkRequest {
    fn default() -> Self {
        Self {
            name: "Trastuzumab_IgG1".to_string(),
            format: BiologicsFormat::Igg1Wt,
            route: DosingRoute::IvBolus,
            dose_mg: 420.0, // ~6 mg/kg maintenance in 70 kg adult
            interval_days: 21.0,
            duration_days: 84.0,
            kd_nm: 0.5,
            target_r0_nm: 5.0,
            target_kdeg_day: 0.25,
            complex_kint_day: 0.60,
            adc_deconj_rate_day: None,
            formulation_mg_ml: 150.0,
        }
    }
}

/// Single time point of the simulated PK & Receptor Occupancy profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PkTimePoint {
    pub day: f64,
    pub c_free_nm: f64,
    pub c_free_ug_ml: f64,
    pub c_intact_adc_ug_ml: Option<f64>,
    pub target_total_nm: f64,
    pub receptor_occupancy_pct: f64,
}

/// Complete human Biologics Minimal-PBPK + FcRn + TMDD simulation report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiologicsPkReport {
    pub name: String,
    pub format: String,
    pub route: String,
    pub mw_kda: f64,
    pub dose_mg: f64,
    pub interval_days: f64,
    pub linear_t_half_days: f64,
    pub effective_t_half_days: f64,
    pub cmax_ss_ug_ml: f64,
    pub ctrough_ss_ug_ml: f64,
    pub ro_trough_ss_pct: f64,
    pub tmdd_sink_fraction_pct: f64,
    pub min_dose_mg_for_90pct_ro: f64,
    pub sc_volume_ml_at_formulation: f64,
    pub sc_feasible: bool,
    pub trajectory: Vec<PkTimePoint>,
    pub clinical_caveats: Vec<String>,
}

fn qss_free_conc(c_tot: f64, r_tot: f64, k_ss: f64) -> f64 {
    let ct = c_tot.max(0.0);
    let rt = r_tot.max(0.0);
    let b = ct - rt - k_ss;
    0.5 * (b + (b * b + 4.0 * k_ss * ct).max(0.0).sqrt())
}

struct SimSummary {
    cmax_ss_ug_ml: f64,
    ctrough_ss_ug_ml: f64,
    ro_trough_ss_pct: f64,
    tmdd_sink_frac: f64,
    trajectory: Vec<PkTimePoint>,
}

fn run_rk4_tmdd(
    req: &BiologicsPkRequest,
    dose_mg_override: f64,
    record_trajectory: bool,
) -> SimSummary {
    let (mw_kda, cl, vc, vp, q, f_sc, ka_sc, def_kdeconj) = req.format.human_params();
    let k_deconj = req.adc_deconj_rate_day.unwrap_or(def_kdeconj);
    let is_adc = req.format == BiologicsFormat::Adc || k_deconj > 0.0;

    // Convert dose (mg) -> nmol: dose_mg * 1e6 ng / (mw_kda * 1e3 ng/nmol) = dose_mg * 1000 / mw_kda
    let dose_nmol = dose_mg_override * 1000.0 / mw_kda;
    let k_ss = req.kd_nm.max(1e-4);
    let r0 = req.target_r0_nm.max(0.0);
    let k_deg = req.target_kdeg_day.max(1e-4);
    let k_int = req.complex_kint_day.max(1e-4);
    let k_syn = k_deg * r0;
    let duration = req.duration_days.clamp(7.0, 180.0);
    let interval = req.interval_days.clamp(1.0, 90.0);

    // State y = [a_sc (nmol), c_tot (nM), a_periph (nmol), r_tot (nM), adc_intact_frac (0..1)]
    let mut y = [0.0_f64, 0.0, 0.0, r0, 1.0];
    let dt = 0.025_f64; // 36 minutes step — accurate even for fast VHH clearance
    let n_steps = (duration / dt).round() as usize;
    let record_every = ((duration / 120.0) / dt).round().max(1.0) as usize;

    let mut traj = if record_trajectory {
        Vec::with_capacity(130)
    } else {
        Vec::new()
    };

    let mut next_dose_step = 0usize;
    let dose_step_inc = (interval / dt).round().max(1.0) as usize;
    let last_interval_start = (duration - interval).max(0.0);

    let mut cmax_ss_nm = 0.0_f64;
    let mut ctrough_ss_nm = f64::INFINITY;
    let mut ro_trough_ss = 100.0_f64;
    let mut total_lin_elim = 0.0_f64;
    let mut total_tmdd_elim = 0.0_f64;

    let deriv = |s: [f64; 5]| -> [f64; 5] {
        let a_sc = s[0].max(0.0);
        let c_tot = s[1].max(0.0);
        let a_per = s[2].max(0.0);
        let r_tot = s[3].max(0.0);
        let c_free = qss_free_conc(c_tot, r_tot, k_ss);
        let rc = if r_tot > 0.0 {
            r_tot * c_free / (k_ss + c_free)
        } else {
            0.0
        };
        let d_asc = -ka_sc * a_sc;
        let in_central = (ka_sc * a_sc) / vc;
        let lin_elim = (cl / vc) * c_free;
        let dist_out = (q / vc) * c_free;
        let dist_in = (q / vp) * a_per / vc;
        let tmdd_elim = k_int * rc;
        let d_ctot = in_central - lin_elim - dist_out + dist_in - tmdd_elim;
        let d_aper = q * c_free - (q / vp) * a_per;
        let d_rtot = k_syn - k_deg * (r_tot - rc).max(0.0) - k_int * rc;
        let d_frac = if is_adc {
            -k_deconj * s[4].max(0.0)
        } else {
            0.0
        };
        [d_asc, d_ctot, d_aper, d_rtot, d_frac]
    };

    for step in 0..=n_steps {
        let t = (step as f64) * dt;
        // Apply scheduled dose
        if step == next_dose_step && t < duration - 1e-6 {
            match req.route {
                DosingRoute::IvBolus => {
                    let old_amt = y[1] * vc;
                    let new_amt = old_amt + dose_nmol;
                    if is_adc && new_amt > 1e-9 {
                        y[4] = (old_amt * y[4] + dose_nmol) / new_amt;
                    }
                    y[1] += dose_nmol / vc;
                }
                DosingRoute::Subcutaneous => {
                    y[0] += dose_nmol * f_sc;
                    if is_adc {
                        y[4] = 1.0;
                    }
                }
            }
            next_dose_step += dose_step_inc;
        }

        let c_free = qss_free_conc(y[1], y[3], k_ss);
        let ro_pct = if r0 > 0.0 {
            100.0 * c_free / (k_ss + c_free)
        } else {
            100.0
        };
        let c_ug_ml = c_free * mw_kda / 1000.0;

        if t >= last_interval_start - 1e-6 {
            if c_free > cmax_ss_nm {
                cmax_ss_nm = c_free;
            }
            if t > last_interval_start + dt * 2.0 && c_free < ctrough_ss_nm {
                ctrough_ss_nm = c_free;
                ro_trough_ss = ro_pct;
            }
        }

        if record_trajectory && (step % record_every == 0 || step == n_steps) {
            traj.push(PkTimePoint {
                day: (t * 100.0).round() / 100.0,
                c_free_nm: (c_free * 1000.0).round() / 1000.0,
                c_free_ug_ml: (c_ug_ml * 1000.0).round() / 1000.0,
                c_intact_adc_ug_ml: if is_adc {
                    Some(((c_ug_ml * y[4].clamp(0.0, 1.0)) * 1000.0).round() / 1000.0)
                } else {
                    None
                },
                target_total_nm: (y[3] * 1000.0).round() / 1000.0,
                receptor_occupancy_pct: (ro_pct * 100.0).round() / 100.0,
            });
        }

        if step == n_steps {
            break;
        }

        // Accumulate linear vs TMDD elimination totals for TMDD sink fraction
        let rc = if y[3] > 0.0 {
            y[3] * c_free / (k_ss + c_free)
        } else {
            0.0
        };
        total_lin_elim += (cl * c_free) * dt;
        total_tmdd_elim += (k_int * rc * vc) * dt;

        // RK4 integration step
        let k1 = deriv(y);
        let y2 = [
            y[0] + 0.5 * dt * k1[0],
            y[1] + 0.5 * dt * k1[1],
            y[2] + 0.5 * dt * k1[2],
            y[3] + 0.5 * dt * k1[3],
            y[4] + 0.5 * dt * k1[4],
        ];
        let k2 = deriv(y2);
        let y3 = [
            y[0] + 0.5 * dt * k2[0],
            y[1] + 0.5 * dt * k2[1],
            y[2] + 0.5 * dt * k2[2],
            y[3] + 0.5 * dt * k2[3],
            y[4] + 0.5 * dt * k2[4],
        ];
        let k3 = deriv(y3);
        let y4 = [
            y[0] + dt * k3[0],
            y[1] + dt * k3[1],
            y[2] + dt * k3[2],
            y[3] + dt * k3[3],
            y[4] + dt * k3[4],
        ];
        let k4 = deriv(y4);
        for i in 0..5 {
            y[i] = (y[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])).max(0.0);
        }
    }

    if !ctrough_ss_nm.is_finite() {
        ctrough_ss_nm = cmax_ss_nm;
    }
    let tmdd_sink_frac = if total_lin_elim + total_tmdd_elim > 1e-9 {
        total_tmdd_elim / (total_lin_elim + total_tmdd_elim)
    } else {
        0.0
    };

    SimSummary {
        cmax_ss_ug_ml: cmax_ss_nm * mw_kda / 1000.0,
        ctrough_ss_ug_ml: ctrough_ss_nm * mw_kda / 1000.0,
        ro_trough_ss_pct: ro_trough_ss,
        tmdd_sink_frac,
        trajectory: traj,
    }
}

/// Simulate human Biologics Minimal-PBPK + FcRn + QSS TMDD and compute maintenance dose for `RO >= 90%`.
pub fn simulate_biologics_pk_tmdd(req: &BiologicsPkRequest) -> Result<BiologicsPkReport, AdcError> {
    if req.dose_mg <= 0.0 || req.interval_days <= 0.0 || req.kd_nm <= 0.0 {
        return Err(AdcError::InvalidParam(
            "dose_mg, interval_days, and kd_nm must all be strictly positive".to_string(),
        ));
    }
    let (mw_kda, cl, vc, vp, q, _f_sc, _ka, _kd) = req.format.human_params();
    // Exact 2-compartment terminal beta half-life (linear FcRn elimination)
    let k10 = cl / vc;
    let k12 = q / vc;
    let k21 = q / vp;
    let s = k10 + k12 + k21;
    let beta = 0.5 * (s - (s * s - 4.0 * k10 * k21).max(0.0).sqrt());
    let linear_t_half = if beta > 1e-9 {
        std::f64::consts::LN_2 / beta
    } else {
        21.0
    };

    let main_sim = run_rk4_tmdd(req, req.dose_mg, true);

    // Effective apparent half-life accounting for TMDD sink acceleration
    let eff_t_half = linear_t_half * (1.0 - 0.65 * main_sim.tmdd_sink_frac).max(0.15);

    // Bisection solver for minimum maintenance dose (mg) achieving trough RO >= 90%
    let mut lo = 0.5_f64;
    let mut hi = 4000.0_f64;
    for _ in 0..18 {
        let mid = 0.5 * (lo + hi);
        let sim = run_rk4_tmdd(req, mid, false);
        if sim.ro_trough_ss_pct >= 90.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let min_dose_90 = (hi * 10.0).round() / 10.0;

    let conc = if req.formulation_mg_ml > 10.0 {
        req.formulation_mg_ml
    } else {
        150.0
    };
    let sc_vol_ml = (min_dose_90 / conc * 100.0).round() / 100.0;
    let sc_feasible = sc_vol_ml <= 2.0 && req.format != BiologicsFormat::Adc;

    let mut caveats = vec![
        "70 kg human typical Minimal-PBPK + QSS TMDD projection; individual patient clearance varies ~30–45% CV (body weight, albumin, FcRn polymorphisms, ADA).".to_string(),
        "Does not include ADA-mediated accelerated clearance (cross-check BioCraft MHC-II immunogenicity report).".to_string(),
    ];
    if main_sim.tmdd_sink_frac > 0.25 {
        caveats.push(format!(
            "Strong TMDD nonlinear target-mediated sink ({:.1}% of total clearance) — low doses will exhibit steep sub-proportional exposure.",
            main_sim.tmdd_sink_frac * 100.0
        ));
    }

    Ok(BiologicsPkReport {
        name: req.name.clone(),
        format: req.format.label().to_string(),
        route: req.route.label().to_string(),
        mw_kda,
        dose_mg: req.dose_mg,
        interval_days: req.interval_days,
        linear_t_half_days: (linear_t_half * 100.0).round() / 100.0,
        effective_t_half_days: (eff_t_half * 100.0).round() / 100.0,
        cmax_ss_ug_ml: (main_sim.cmax_ss_ug_ml * 100.0).round() / 100.0,
        ctrough_ss_ug_ml: (main_sim.ctrough_ss_ug_ml * 100.0).round() / 100.0,
        ro_trough_ss_pct: (main_sim.ro_trough_ss_pct * 100.0).round() / 100.0,
        tmdd_sink_fraction_pct: (main_sim.tmdd_sink_frac * 1000.0).round() / 10.0,
        min_dose_mg_for_90pct_ro: min_dose_90,
        sc_volume_ml_at_formulation: sc_vol_ml,
        sc_feasible,
        trajectory: main_sim.trajectory,
        clinical_caveats: caveats,
    })
}

/// Render a clean Markdown Biologics PK & TMDD Report.
#[must_use]
pub fn render_biopk_markdown(r: &BiologicsPkReport) -> String {
    let mut md = String::new();
    let _ = writeln!(md, "# Biologics Minimal-PBPK + TMDD Report: {}\n", r.name);
    let _ = writeln!(
        md,
        "- **Format:** `{}` | **Regimen:** `{:.1} mg {} Q{:.0}D`",
        r.format, r.dose_mg, r.route, r.interval_days
    );
    let _ = writeln!(md, "\n| Parameter | Predicted Value |");
    let _ = writeln!(md, "|---|---:|");
    let _ = writeln!(
        md,
        "| Linear FcRn/Renal `t½` | **{:.2} days** |",
        r.linear_t_half_days
    );
    let _ = writeln!(
        md,
        "| Effective `t½` (with TMDD) | **{:.2} days** |",
        r.effective_t_half_days
    );
    let _ = writeln!(
        md,
        "| Steady-State `Cmax` (Free) | **{:.2} µg/mL** |",
        r.cmax_ss_ug_ml
    );
    let _ = writeln!(
        md,
        "| Steady-State `Ctrough` (Free) | **{:.2} µg/mL** |",
        r.ctrough_ss_ug_ml
    );
    let _ = writeln!(
        md,
        "| Trough Receptor Occupancy (`RO%`) | **{:.1}%** |",
        r.ro_trough_ss_pct
    );
    let _ = writeln!(
        md,
        "| TMDD Target-Sink Clearance Share | {:.1}% |",
        r.tmdd_sink_fraction_pct
    );
    let _ = writeln!(
        md,
        "| Min Maintenance Dose for `RO >= 90%` | **{:.1} mg** (SC vol `{:.2} mL` → {}) |",
        r.min_dose_mg_for_90pct_ro,
        r.sc_volume_ml_at_formulation,
        if r.sc_feasible {
            "Feasible <= 2 mL"
        } else {
            "IV or high-conc/HLE needed"
        }
    );
    let _ = writeln!(md, "\n## Clinical Caveats\n");
    for c in &r.clinical_caveats {
        let _ = writeln!(md, "- {c}");
    }
    md
}
