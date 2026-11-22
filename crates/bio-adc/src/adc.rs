#![forbid(unsafe_code)]

//! Antibody-Drug Conjugate (ADC) Architect:
//! Curated clinical linkers (`GGFG`, `Val-Cit-PAB`, `SMCC`, `CL2A`, `SPDB`) and payloads
//! (`DXd`, `MMAE`, `MMAF`, `DM1`, `SN-38`, `PBD`) + custom SMILES support, DAR Hydrophobic
//! Drug-Load Index (`HDI`), Bystander Permeability scoring, and Conjugation Site heterogeneity.

use std::fmt::Write as _;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AdcError {
    #[error("unknown linker '{0}' (supported: ggfg, vc_pab, smcc, cl2a, spdb)")]
    UnknownLinker(String),
    #[error("unknown payload '{0}' (supported: dxd, mmae, mmaf, dm1, sn38, pbd, or pass --smiles)")]
    UnknownPayload(String),
    #[error("invalid parameter: {0}")]
    InvalidParam(String),
}

/// Curated clinical ADC linker specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkerSpec {
    pub id: String,
    pub name: String,
    pub cleavage_mechanism: String,
    pub is_cleavable: bool,
    pub mw_da: f64,
    pub hydrophilicity_mask_logp: f64,
    pub plasma_t_half_days: f64,
    pub example_drug: String,
}

/// Curated clinical ADC cytotoxic payload specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PayloadSpec {
    pub id: String,
    pub name: String,
    pub mechanism: String,
    pub smiles: String,
    pub mw_da: f64,
    pub clogp: f64,
    pub tpsa_a2: f64,
    pub potency_ic50_nm: f64,
    pub neutral_catabolite_at_ph74: bool,
    pub example_drug: String,
}

/// Conjugation chemistry site strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConjugationStrategy {
    InterchainCys,
    RandomLys,
    SiteSpecificThiomab,
    GlycoConnect,
}

impl ConjugationStrategy {
    pub fn label(self) -> &'static str {
        match self {
            Self::InterchainCys => "Interchain Cysteine (Maleimide)",
            Self::RandomLys => "Random Surface Lysine (NHS/SMCC)",
            Self::SiteSpecificThiomab => "Site-Specific Engineered Cys (THIOMAB)",
            Self::GlycoConnect => "Site-Specific Fc Glycan (GlycoConnect N297)",
        }
    }

    pub fn from_str_lossy(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "lys" | "lysine" | "random_lys" => Self::RandomLys,
            "thiomab" | "site" | "site_specific" => Self::SiteSpecificThiomab,
            "glyco" | "glycoconnect" => Self::GlycoConnect,
            _ => Self::InterchainCys,
        }
    }
}

/// Input request for designing and auditing an ADC construct.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdcDesignRequest {
    pub name: String,
    pub antibody_mw_da: f64,
    pub linker_id: String,
    pub payload_id: String,
    pub custom_payload_smiles: Option<String>,
    pub dar: f64,
    pub conjugation: ConjugationStrategy,
    pub admetcraft_payload_summary: Option<String>,
}

/// Complete ADC developability, conjugation, bystander & CMC risk audit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdcDesignReport {
    pub name: String,
    pub linker: LinkerSpec,
    pub payload: PayloadSpec,
    pub conjugation: String,
    pub dar: f64,
    pub antibody_mw_da: f64,
    pub total_adc_mw_da: f64,
    pub effective_drug_linker_logp: f64,
    pub hydrophobic_drug_load_index: f64,
    pub aggregation_clearance_risk: String,
    pub bystander_score_0_100: f64,
    pub bystander_class: String,
    pub dar_heterogeneity_risk: String,
    pub plasma_linker_stability_t_half_days: f64,
    pub admetcraft_payload_summary: Option<String>,
    pub design_verdict: String,
    pub recommendations: Vec<String>,
}

/// Curated catalog of clinical ADC linkers.
#[must_use]
pub fn linker_catalog() -> Vec<LinkerSpec> {
    vec![
        LinkerSpec {
            id: "ggfg".to_string(),
            name: "MC-GGFG-AM (Deruxtecan Tetrapeptide)".to_string(),
            cleavage_mechanism: "Lysosomal Cathepsin B/L protease-cleavable (self-immolative aminomethylene)".to_string(),
            is_cleavable: true,
            mw_da: 541.55,
            hydrophilicity_mask_logp: -1.65,
            plasma_t_half_days: 14.0,
            example_drug: "Trastuzumab deruxtecan (Enhertu / T-DXd)".to_string(),
        },
        LinkerSpec {
            id: "vc_pab".to_string(),
            name: "MC-Val-Cit-PAB".to_string(),
            cleavage_mechanism: "Lysosomal Cathepsin B dipeptide-cleavable (p-aminobenzylcarbamate)".to_string(),
            is_cleavable: true,
            mw_da: 598.70,
            hydrophilicity_mask_logp: -0.35,
            plasma_t_half_days: 8.5,
            example_drug: "Brentuximab vedotin (Adcetris), Polatuzumab vedotin".to_string(),
        },
        LinkerSpec {
            id: "smcc".to_string(),
            name: "SMCC (Non-Cleavable Thioether)".to_string(),
            cleavage_mechanism: "Non-cleavable; requires full lysosomal proteolytic digestion of mAb to Lys-SMCC-payload".to_string(),
            is_cleavable: false,
            mw_da: 219.24,
            hydrophilicity_mask_logp: -0.10,
            plasma_t_half_days: 22.0,
            example_drug: "Trastuzumab emtansine (Kadcyla / T-DM1)".to_string(),
        },
        LinkerSpec {
            id: "cl2a".to_string(),
            name: "CL2A (pH-Labile PEG8-Carbonate)".to_string(),
            cleavage_mechanism: "Acid-labile hydrolytic carbonate cleavage (extracellular tumor microenvironment + endosome)".to_string(),
            is_cleavable: true,
            mw_da: 942.06,
            hydrophilicity_mask_logp: -2.10,
            plasma_t_half_days: 2.0,
            example_drug: "Sacituzumab govitecan (Trodelvy)".to_string(),
        },
        LinkerSpec {
            id: "spdb".to_string(),
            name: "SPDB (Hindered Disulfide)".to_string(),
            cleavage_mechanism: "Intracellular glutathione (GSH)-reducible disulfide cleavage".to_string(),
            is_cleavable: true,
            mw_da: 213.30,
            hydrophilicity_mask_logp: -0.15,
            plasma_t_half_days: 5.5,
            example_drug: "Mirvetuximab soravtansine (Elahere)".to_string(),
        },
    ]
}

/// Curated catalog of clinical ADC cytotoxic payloads with exact verified SMILES.
#[must_use]
pub fn payload_catalog() -> Vec<PayloadSpec> {
    vec![
        PayloadSpec {
            id: "dxd".to_string(),
            name: "DXd (Deruxtecan / Exatecan-Glycolamide)".to_string(),
            mechanism: "Topoisomerase-I inhibitor (DNA ds-break)".to_string(),
            smiles: "CC[C@@]1(O)C(=O)OCc2c1cc1n(c2=O)Cc2c-1nc1cc(F)c(C)c3c1c2[C@H](CC3)NC(=O)CO".to_string(),
            mw_da: 493.48,
            clogp: 2.25,
            tpsa_a2: 112.0,
            potency_ic50_nm: 0.35,
            neutral_catabolite_at_ph74: true,
            example_drug: "Enhertu (T-DXd), Datopotamab deruxtecan".to_string(),
        },
        PayloadSpec {
            id: "mmae".to_string(),
            name: "MMAE (Monomethyl Auristatin E)".to_string(),
            mechanism: "Microtubule polymerization inhibitor (Vinca domain)".to_string(),
            smiles: "CC[C@H](C)[C@@H]([C@@H](CC(=O)N1CCC[C@H]1[C@@H]([C@@H](C)C(=O)N[C@H](C)[C@H](O)c1ccccc1)OC)OC)N(C)C(=O)[C@H](C(C)C)NC(=O)[C@H](C(C)C)NC".to_string(),
            mw_da: 717.98,
            clogp: 3.10,
            tpsa_a2: 131.8,
            potency_ic50_nm: 0.10,
            neutral_catabolite_at_ph74: true,
            example_drug: "Adcetris, Padcev, Polivy".to_string(),
        },
        PayloadSpec {
            id: "mmaf".to_string(),
            name: "MMAF (Monomethyl Auristatin F)".to_string(),
            mechanism: "Microtubule inhibitor (C-terminal carboxylate — membrane-impermeable)".to_string(),
            smiles: "CC[C@H](C)[C@@H]([C@@H](CC(=O)N1CCC[C@H]1[C@@H]([C@@H](C)C(=O)N[C@@H](Cc1ccccc1)C(=O)O)OC)OC)N(C)C(=O)[C@H](C(C)C)NC(=O)[C@H](C(C)C)NC".to_string(),
            mw_da: 731.96,
            clogp: 1.65,
            tpsa_a2: 149.0,
            potency_ic50_nm: 0.15,
            neutral_catabolite_at_ph74: false,
            example_drug: "Belantamab mafodotin (Blenrep)".to_string(),
        },
        PayloadSpec {
            id: "dm1".to_string(),
            name: "DM1 (Mertansine / Maytansinoid)".to_string(),
            mechanism: "Microtubule polymerization inhibitor (Maytansine site)".to_string(),
            smiles: "CO[C@H]1\\C=C\\C=C(\\C)Cc2cc(OC)c(Cl)c(c2)N(C)C(=O)C[C@H](OC(=O)[C@H](C)N(C)C(=O)CCS)[C@]2(C)O[C@H]2[C@H](C)[C@@H]2OC(=O)N[C@@]1(O)C2".to_string(),
            mw_da: 738.30,
            clogp: 3.35,
            tpsa_a2: 141.0,
            potency_ic50_nm: 0.08,
            neutral_catabolite_at_ph74: true,
            example_drug: "Kadcyla (T-DM1)".to_string(),
        },
        PayloadSpec {
            id: "sn38".to_string(),
            name: "SN-38 (Active Irinotecan Metabolite)".to_string(),
            mechanism: "Topoisomerase-I inhibitor".to_string(),
            smiles: "CC[C@@]1(O)C(=O)OCc2c1cc1n(c2=O)Cc2c-1nc1ccc(O)cc1c2CC".to_string(),
            mw_da: 392.40,
            clogp: 2.65,
            tpsa_a2: 101.6,
            potency_ic50_nm: 1.50,
            neutral_catabolite_at_ph74: true,
            example_drug: "Sacituzumab govitecan (Trodelvy)".to_string(),
        },
        PayloadSpec {
            id: "pbd".to_string(),
            name: "Tesirine / PBD Dimer (SG3199)".to_string(),
            mechanism: "Ultra-potent DNA minor-groove interstrand crosslinker".to_string(),
            smiles: "COc1cc2c(cc1OCCCCCOc1cc3c(cc1OC)C(=O)N1CC(=C)C[C@H]1C=N3)N=C[C@@H]1CC(=C)CN1C2=O".to_string(),
            mw_da: 584.66,
            clogp: 3.85,
            tpsa_a2: 103.0,
            potency_ic50_nm: 0.01,
            neutral_catabolite_at_ph74: true,
            example_drug: "Loncastuximab tesirine (Zynlonta)".to_string(),
        },
    ]
}

fn estimate_custom_smiles_payload(smiles: &str) -> PayloadSpec {
    let mut c = 0usize;
    let mut n = 0usize;
    let mut o = 0usize;
    let mut hal = 0usize;
    for ch in smiles.chars() {
        match ch {
            'C' | 'c' => c += 1,
            'N' | 'n' => n += 1,
            'O' | 'o' => o += 1,
            'F' | 'I' => hal += 1,
            _ => {}
        }
    }
    let mw = (c as f64) * 13.0 + (n as f64) * 14.5 + (o as f64) * 16.5 + (hal as f64) * 24.0;
    let tpsa = (n as f64) * 15.0 + (o as f64) * 18.0;
    let clogp = ((c as f64) * 0.28 - (n as f64) * 0.45 - (o as f64) * 0.35).clamp(-1.0, 6.5);
    let has_acid = smiles.contains("C(=O)O") && !smiles.contains("C(=O)OC");
    PayloadSpec {
        id: "custom".to_string(),
        name: "Custom Small-Molecule Payload".to_string(),
        mechanism: "Custom SMILES (verify mechanism & potency with AdmetCraft)".to_string(),
        smiles: smiles.to_string(),
        mw_da: (mw * 10.0).round() / 10.0,
        clogp: (clogp * 100.0).round() / 100.0,
        tpsa_a2: (tpsa * 10.0).round() / 10.0,
        potency_ic50_nm: 0.5,
        neutral_catabolite_at_ph74: !has_acid,
        example_drug: "Custom candidate".to_string(),
    }
}

/// Design and audit an Antibody-Drug Conjugate (ADC) construct.
pub fn design_adc(req: &AdcDesignRequest) -> Result<AdcDesignReport, AdcError> {
    if !(0.5..=12.0).contains(&req.dar) {
        return Err(AdcError::InvalidParam(format!(
            "DAR must be in [0.5, 12.0], got {}",
            req.dar
        )));
    }
    let lkey = req.linker_id.trim().to_ascii_lowercase().replace('-', "_");
    let linker = linker_catalog()
        .into_iter()
        .find(|l| l.id == lkey || l.name.to_ascii_lowercase().contains(&lkey))
        .ok_or_else(|| AdcError::UnknownLinker(req.linker_id.clone()))?;

    let payload = if let Some(ref smi) = req.custom_payload_smiles {
        if smi.trim().is_empty() {
            return Err(AdcError::InvalidParam(
                "empty custom payload SMILES".to_string(),
            ));
        }
        estimate_custom_smiles_payload(smi.trim())
    } else {
        let pkey = req.payload_id.trim().to_ascii_lowercase().replace('-', "");
        payload_catalog()
            .into_iter()
            .find(|p| p.id == pkey || p.name.to_ascii_lowercase().replace('-', "").contains(&pkey))
            .ok_or_else(|| AdcError::UnknownPayload(req.payload_id.clone()))?
    };

    let mab_mw = if req.antibody_mw_da > 10_000.0 {
        req.antibody_mw_da
    } else {
        148_000.0
    };
    let total_adc_mw = mab_mw + req.dar * (linker.mw_da + payload.mw_da);

    // Hydrophobic Drug-Load Index (HDI): distinguishes hydrophilic-masked DAR 8 (T-DXd ~ 4.8)
    // from unmasked hydrophobic DAR 8 (Val-Cit + MMAE ~ 22.0, which suffers rapid liver clearance!)
    let eff_logp = (payload.clogp + linker.hydrophilicity_mask_logp).max(0.25);
    let hdi = ((req.dar * eff_logp) * 100.0).round() / 100.0;

    let aggregation_clearance_risk = if hdi <= 9.5 {
        "Low (Clinical-Grade Colloidal Stability & Preserved FcRn PK)".to_string()
    } else if hdi <= 14.5 {
        "Moderate (Monitor HIC Retention & Accelerated Hepatic Clearance)".to_string()
    } else {
        "High (Excess Hydrophobic Drug Load — High Aggregation & Fast Clearance Risk)".to_string()
    };

    // Bystander killing permeability score (0..100):
    // Requires BOTH cleavable free-payload release AND neutral, membrane-permeable catabolite (LogP 1.5–4, TPSA < 140)
    let bystander_score = if !linker.is_cleavable {
        // Non-cleavable SMCC releases charged Lys-SMCC-payload catabolite -> minimal bystander effect
        18.0
    } else if !payload.neutral_catabolite_at_ph74 {
        // E.g. MMAF carboxylate -> membrane-impermeable
        22.0
    } else {
        let logp_term = if (1.5..=3.8).contains(&payload.clogp) {
            48.0
        } else {
            30.0
        };
        let tpsa_term = (42.0 - (payload.tpsa_a2 - 95.0).max(0.0) * 0.45).clamp(10.0, 42.0);
        let mw_term = if payload.mw_da <= 550.0 { 10.0 } else { 4.0 };
        (logp_term + tpsa_term + mw_term).clamp(15.0, 98.0)
    };
    let bystander_class = if bystander_score >= 70.0 {
        "High (Active against heterogeneous / antigen-low tumor neighbors)".to_string()
    } else if bystander_score >= 45.0 {
        "Moderate (Partial membrane diffusion)".to_string()
    } else {
        "Low (Charged/non-cleavable catabolite confined to antigen-positive cells)".to_string()
    };

    let dar_heterogeneity_risk = match req.conjugation {
        ConjugationStrategy::SiteSpecificThiomab | ConjugationStrategy::GlycoConnect => {
            "Low (Homogeneous DAR species; tight HIC/LC-MS profile)".to_string()
        }
        ConjugationStrategy::InterchainCys => {
            if (req.dar - 8.0).abs() <= 0.5 {
                "Low (Saturated 8/8 interchain Cys gives uniform DAR-8 species, e.g. T-DXd)"
                    .to_string()
            } else {
                "Moderate (DAR 0/2/4/6/8 statistical mixture; verify DAR-8 subpopulation)"
                    .to_string()
            }
        }
        ConjugationStrategy::RandomLys => {
            "High (Poisson distribution across ~40 surface Lys residues; complex CMC analytics)"
                .to_string()
        }
    };

    let mut recs = Vec::new();
    if hdi > 14.5 {
        recs.push("Reduce DAR to 2–4 or switch to hydrophilic masking linker (GGFG, PEG8, or polysarcosine) to avoid rapid hepatic clearance.".to_string());
    }
    if linker.plasma_t_half_days < 4.0 {
        recs.push("Short plasma linker half-life increases systemic free-payload exposure; verify off-target neutropenia/GI tox window.".to_string());
    }
    if req.conjugation == ConjugationStrategy::InterchainCys && req.dar > 4.5 && eff_logp > 1.5 {
        recs.push("Breaking all 4 interchain disulfides with a lipophilic drug-linker lowers Fab/hinge Tm — validate by nanoDSF & SEC-MALS.".to_string());
    }
    if recs.is_empty() {
        recs.push("Construct falls inside the proven clinical ADC therapeutic design envelope; proceed to HIC-HPLC, DAR LC-MS, and HER2-high/low cytotoxicity panels.".to_string());
    }

    let design_verdict = if hdi <= 9.5 && linker.plasma_t_half_days >= 5.0 {
        "FAVORABLE — Balanced DAR, Hydrophilicity & Plasma Stability".to_string()
    } else if hdi <= 14.5 {
        "CONDITIONAL — Verify HIC Hydrophobicity & Free-Payload Plasma Stability".to_string()
    } else {
        "HIGH RISK — Excessive Hydrophobic Drug Load (Re-engineer Linker or DAR)".to_string()
    };

    Ok(AdcDesignReport {
        name: req.name.clone(),
        linker: linker.clone(),
        payload,
        conjugation: req.conjugation.label().to_string(),
        dar: req.dar,
        antibody_mw_da: mab_mw,
        total_adc_mw_da: (total_adc_mw * 10.0).round() / 10.0,
        effective_drug_linker_logp: (eff_logp * 100.0).round() / 100.0,
        hydrophobic_drug_load_index: hdi,
        aggregation_clearance_risk,
        bystander_score_0_100: bystander_score.round(),
        bystander_class,
        dar_heterogeneity_risk,
        plasma_linker_stability_t_half_days: linker.plasma_t_half_days,
        admetcraft_payload_summary: req.admetcraft_payload_summary.clone(),
        design_verdict,
        recommendations: recs,
    })
}

/// Render a clean Markdown ADC Architect dossier.
#[must_use]
pub fn render_adc_markdown(r: &AdcDesignReport) -> String {
    let mut md = String::new();
    let _ = writeln!(
        md,
        "# Antibody-Drug Conjugate (ADC) Design Report: {}\n",
        r.name
    );
    let _ = writeln!(md, "- **Verdict:** **{}**", r.design_verdict);
    let _ = writeln!(
        md,
        "- **Construct:** `{}` + `{}` | **DAR:** `{:.1}` | **Conjugation:** `{}`",
        r.linker.name, r.payload.name, r.dar, r.conjugation
    );
    let _ = writeln!(
        md,
        "- **Total ADC MW:** `{:.1} Da` (`{:.2} kDa`, mAb `{:.1} kDa`)\n",
        r.total_adc_mw_da,
        r.total_adc_mw_da / 1000.0,
        r.antibody_mw_da / 1000.0
    );
    let _ = writeln!(md, "## Critical ADC Developability Metrics\n");
    let _ = writeln!(md, "| Parameter | Value | Assessment |");
    let _ = writeln!(md, "|---|---:|---|");
    let _ = writeln!(
        md,
        "| Hydrophobic Drug-Load Index (`DAR × eff_LogP`) | **{:.2}** | {} |",
        r.hydrophobic_drug_load_index, r.aggregation_clearance_risk
    );
    let _ = writeln!(
        md,
        "| Bystander Permeability Score | **{:.0}/100** | {} |",
        r.bystander_score_0_100, r.bystander_class
    );
    let _ = writeln!(
        md,
        "| Linker Plasma Stability (`t½`) | **{:.1} d** | {} |",
        r.plasma_linker_stability_t_half_days, r.linker.cleavage_mechanism
    );
    let _ = writeln!(
        md,
        "| DAR Heterogeneity | DAR {:.1} | {} |",
        r.dar, r.dar_heterogeneity_risk
    );
    let _ = writeln!(
        md,
        "| Payload (`{}`) | MW `{:.1}`, cLogP `{:.2}`, TPSA `{:.1}` | {} (`IC50 ~{:.2} nM`) |",
        r.payload.id.to_uppercase(),
        r.payload.mw_da,
        r.payload.clogp,
        r.payload.tpsa_a2,
        r.payload.mechanism,
        r.payload.potency_ic50_nm
    );
    if let Some(ref adm) = r.admetcraft_payload_summary {
        let _ = writeln!(md, "\n## Live AdmetCraft Payload Audit\n\n- {adm}");
    }
    let _ = writeln!(md, "\n## Recommendations\n");
    for rec in &r.recommendations {
        let _ = writeln!(md, "- {rec}");
    }
    md
}
