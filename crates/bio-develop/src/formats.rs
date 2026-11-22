#![forbid(unsafe_code)]

//! Binder Format Decision Matrix (aligned with `researchfw-binder-design-and-format`) and
//! Multi-Host Codon Optimization (`CHO`, *E. coli*, *Pichia pastoris*).

use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

/// Target product profile constraints guiding binder format selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinderFormatGoal {
    pub target_name: String,
    pub require_effector_function: bool,
    pub require_long_half_life: bool,
    pub require_deep_tissue_penetration: bool,
    pub require_subcutaneous: bool,
    pub require_payload_delivery: bool,
    pub require_bispecific: bool,
}

impl Default for BinderFormatGoal {
    fn default() -> Self {
        Self {
            target_name: "Target Antigen".to_string(),
            require_effector_function: false,
            require_long_half_life: true,
            require_deep_tissue_penetration: false,
            require_subcutaneous: true,
            require_payload_delivery: false,
            require_bispecific: false,
        }
    }
}

/// Single evaluated biologics format row in the decision matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormatComparisonRow {
    pub format_id: String,
    pub label: String,
    pub mw_kda: f64,
    pub valency: String,
    pub human_t_half_days: String,
    pub effector_function: String,
    pub expression_host: String,
    pub fc_engineering: String,
    pub tissue_penetration: String,
    pub suitability_score: f64,
    pub key_tradeoff: String,
}

/// Complete format comparison matrix + orthogonal experimental validation plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormatDecisionMatrix {
    pub goal: BinderFormatGoal,
    pub ranked_formats: Vec<FormatComparisonRow>,
    pub orthogonal_wetlab_plan: Vec<String>,
}

/// Compare 10 clinical biologics & binder formats against the user's TPP constraints.
#[must_use]
pub fn compare_binder_formats(goal: &BinderFormatGoal) -> FormatDecisionMatrix {
    struct Spec {
        id: &'static str,
        label: &'static str,
        mw: f64,
        val: &'static str,
        thalf: &'static str,
        eff: &'static str,
        host: &'static str,
        fc: &'static str,
        pen: &'static str,
        has_eff: bool,
        long_hl: bool,
        deep_pen: bool,
        sc_ok: bool,
        is_adc: bool,
        is_bisp: bool,
        tradeoff: &'static str,
    }

    let catalog = [
        Spec {
            id: "igg1_wt",
            label: "IgG1-WT (Full mAb)",
            mw: 148.0,
            val: "Bivalent (2× Fab)",
            thalf: "18–23 d",
            eff: "High (ADCC / CDC / ADCP)",
            host: "CHO (3–8 g/L)",
            fc: "Native human IgG1 Fc",
            pen: "Moderate",
            has_eff: true,
            long_hl: true,
            deep_pen: false,
            sc_ok: true,
            is_adc: false,
            is_bisp: false,
            tradeoff: "Gold-standard CMC & FcRn recycling; active ADCC/CDC unsuitable when target is on healthy tissue.",
        },
        Spec {
            id: "igg1_lalapg",
            label: "IgG1-LALA-PG (Effector-Silenced)",
            mw: 148.0,
            val: "Bivalent (2× Fab)",
            thalf: "18–23 d",
            eff: "Silent (L234A/L235A/P329G)",
            host: "CHO (3–8 g/L)",
            fc: "L234A / L235A / P329G",
            pen: "Moderate",
            has_eff: false,
            long_hl: true,
            deep_pen: false,
            sc_ok: true,
            is_adc: false,
            is_bisp: false,
            tradeoff: "Eliminates FcγR/C1q cytokine storm while retaining full IgG1 thermostability and FcRn half-life.",
        },
        Spec {
            id: "igg1_yte",
            label: "IgG1-YTE / LS (Half-Life Extended)",
            mw: 148.0,
            val: "Bivalent (2× Fab)",
            thalf: "60–90 d (Q3M/Q6M)",
            eff: "Configurable (WT or LALA-PG)",
            host: "CHO (3–7 g/L)",
            fc: "M252Y/S254T/T256E (YTE) or M428L/N434S (LS)",
            pen: "Moderate",
            has_eff: false,
            long_hl: true,
            deep_pen: false,
            sc_ok: true,
            is_adc: false,
            is_bisp: false,
            tradeoff: "Enables quarterly SC dosing via pH 6.0 FcRn affinity; YTE can mildly lower CH2 Tm (~5°C) vs LS.",
        },
        Spec {
            id: "igg4_s228p",
            label: "IgG4-S228P (Hinge-Stabilized)",
            mw: 146.0,
            val: "Bivalent (2× Fab)",
            thalf: "18–21 d",
            eff: "Very Low (Minimal ADCC/CDC)",
            host: "CHO (3–6 g/L)",
            fc: "S228P prevents Fab-arm exchange",
            pen: "Moderate",
            has_eff: false,
            long_hl: true,
            deep_pen: false,
            sc_ok: true,
            is_adc: false,
            is_bisp: false,
            tradeoff: "Standard for checkpoint blockade (Pembrolizumab); slightly lower acid stability than IgG1.",
        },
        Spec {
            id: "fab",
            label: "Fab Fragment",
            mw: 48.0,
            val: "Monovalent (1×)",
            thalf: "0.5–1.5 d (renal)",
            eff: "None (No Fc)",
            host: "E. coli periplasm / CHO",
            fc: "None",
            pen: "High",
            has_eff: false,
            long_hl: false,
            deep_pen: true,
            sc_ok: true,
            is_adc: false,
            is_bisp: false,
            tradeoff: "Zero receptor cross-linking/agonism and high tissue/ocular penetration; rapid systemic clearance.",
        },
        Spec {
            id: "vhh_mono",
            label: "Nanobody (VHH Monomer)",
            mw: 14.5,
            val: "Monovalent (Convex Paratope)",
            thalf: "0.05–0.15 d (renal)",
            eff: "None",
            host: "E. coli / Pichia (high yield)",
            fc: "None",
            pen: "Very High (Cryptic Epitopes)",
            has_eff: false,
            long_hl: false,
            deep_pen: true,
            sc_ok: true,
            is_adc: false,
            is_bisp: false,
            tradeoff: "Accesses recessed enzymatic clefts & solid tumors rapidly; requires half-life extension for chronic systemic therapy.",
        },
        Spec {
            id: "vhh_hsa",
            label: "VHH–anti-HSA / VHH-Fc (Half-Life Extended)",
            mw: 78.0,
            val: "Monovalent or Bivalent",
            thalf: "14–19 d (HSA/FcRn)",
            eff: "Optional (Fc-silent default)",
            host: "Pichia / CHO",
            fc: "Anti-HSA VHH or Fc fusion (Envafolimab)",
            pen: "High",
            has_eff: false,
            long_hl: true,
            deep_pen: true,
            sc_ok: true,
            is_adc: false,
            is_bisp: false,
            tradeoff: "Best-of-both-worlds SC convenience + cryptic epitope reach (Envafolimab); monitor ADA against camelid FR/linker.",
        },
        Spec {
            id: "scfv",
            label: "scFv (VH-(G4S)3-VL)",
            mw: 27.0,
            val: "Monovalent",
            thalf: "0.08–0.25 d",
            eff: "None",
            host: "E. coli / Yeast",
            fc: "None",
            pen: "High",
            has_eff: false,
            long_hl: false,
            deep_pen: true,
            sc_ok: false,
            is_adc: false,
            is_bisp: false,
            tradeoff: "Ideal building block for CAR-T / BiTEs; prone to domain-swapping dimerization & lower Tm without VH44-VL100 disulfide.",
        },
        Spec {
            id: "bispecific_kih",
            label: "Bispecific IgG (Knobs-into-Holes + CrossMab)",
            mw: 148.0,
            val: "1+1 Dual Target",
            thalf: "14–21 d",
            eff: "Configurable (LALA-PG typical for T-cell engagers)",
            host: "CHO (co-expression / redox)",
            fc: "T366W / T366S-L368A-Y407V",
            pen: "Moderate",
            has_eff: false,
            long_hl: true,
            deep_pen: false,
            sc_ok: true,
            is_adc: false,
            is_bisp: true,
            tradeoff: "Enables dual blockade or T-cell redirection; higher CMC chain-pairing purification complexity.",
        },
        Spec {
            id: "adc_igg1",
            label: "ADC (IgG1 + Cleavable Linker + Cytotoxic Payload)",
            mw: 155.0,
            val: "Bivalent + DAR 4–8 Payload",
            thalf: "4–10 d (DAR-dependent)",
            eff: "IgG1 + Payload Bystander",
            host: "CHO + Chemical Conjugation",
            fc: "IgG1 (Interchain Cys or THIOMAB)",
            pen: "Moderate (+ Bystander)",
            has_eff: true,
            long_hl: true,
            deep_pen: false,
            sc_ok: false,
            is_adc: true,
            is_bisp: false,
            tradeoff: "Turns antigen internalization into potent cytotoxicity (T-DXd); requires tight DAR/linker stability & IV oncology dosing.",
        },
    ];

    let mut rows: Vec<FormatComparisonRow> = catalog
        .iter()
        .map(|s| {
            let mut score = 50.0_f64;
            if goal.require_effector_function {
                score += if s.has_eff { 20.0 } else { -18.0 };
            } else if !s.has_eff {
                score += 8.0;
            }
            if goal.require_long_half_life {
                score += if s.long_hl { 20.0 } else { -22.0 };
            }
            if goal.require_deep_tissue_penetration {
                score += if s.deep_pen { 22.0 } else { -8.0 };
            }
            if goal.require_subcutaneous {
                score += if s.sc_ok { 10.0 } else { -15.0 };
            }
            if goal.require_payload_delivery {
                score += if s.is_adc { 35.0 } else { -20.0 };
            } else if s.is_adc {
                score -= 25.0;
            }
            if goal.require_bispecific {
                score += if s.is_bisp { 35.0 } else { -15.0 };
            } else if s.is_bisp {
                score -= 10.0;
            }
            FormatComparisonRow {
                format_id: s.id.to_string(),
                label: s.label.to_string(),
                mw_kda: s.mw,
                valency: s.val.to_string(),
                human_t_half_days: s.thalf.to_string(),
                effector_function: s.eff.to_string(),
                expression_host: s.host.to_string(),
                fc_engineering: s.fc.to_string(),
                tissue_penetration: s.pen.to_string(),
                suitability_score: score.clamp(0.0, 100.0).round(),
                key_tradeoff: s.tradeoff.to_string(),
            }
        })
        .collect();

    rows.sort_by(|a, b| b.suitability_score.total_cmp(&a.suitability_score));

    let orthogonal_wetlab_plan = vec![
        "Binding kinetics & specificity: SPR / BLI (KD, kon, koff) against human + cynomolgus ortholog + irrelevant antigen & isotype negative controls.".to_string(),
        "Colloidal & thermal stability: SEC-MALS (% monomer >= 95%), nanoDSF (Tm1 >= 65°C, Tagg), and freeze-thaw / 40°C 2-week stress.".to_string(),
        "Polyspecificity & SC viscosity screening: AC-SINS (Δλmax < 5 nm), BVP/PSR ELISA, HIC retention time, and high-concentration viscosity (150 mg/mL < 15 cP).".to_string(),
        "Immunogenicity & PTM verification: LC-MS intact/peptide mapping (deamidation/oxidation/glycans) + ex vivo healthy-donor PBMC CD4+ T-cell assay / MAPPs.".to_string(),
        "Functional potency: Target-positive vs target-knockout cell assay (IC50/EC50, internalization for ADCs, ADCC reporter if Fc-active).".to_string(),
    ];

    FormatDecisionMatrix {
        goal: goal.clone(),
        ranked_formats: rows,
        orthogonal_wetlab_plan,
    }
}

/// Expression host for DNA reverse-translation & codon optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExpressionHost {
    Cho,
    EColi,
    Pichia,
}

impl ExpressionHost {
    pub fn label(self) -> &'static str {
        match self {
            Self::Cho => "CHO (Mammalian IgG/VHH-Fc)",
            Self::EColi => "E. coli (Periplasmic Fab/VHH/scFv)",
            Self::Pichia => "Pichia pastoris (Secreted VHH/Protein)",
        }
    }

    pub fn from_str_lossy(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "cho" | "mammalian" | "hek293" => Some(Self::Cho),
            "ecoli" | "e.coli" | "e_coli" | "bacteria" => Some(Self::EColi),
            "pichia" | "yeast" => Some(Self::Pichia),
            _ => None,
        }
    }
}

/// Optimized DNA coding sequence and manufacturability audit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodonOptReport {
    pub host: String,
    pub signal_peptide_recommendation: String,
    pub dna_sequence: String,
    pub length_bp: usize,
    pub gc_percent: f64,
    pub restriction_sites_found: Vec<String>,
    pub homopolymer_alerts: usize,
}

fn preferred_codon(aa: char, host: ExpressionHost) -> &'static str {
    match host {
        ExpressionHost::Cho => match aa {
            'A' => "GCC",
            'R' => "CGG",
            'N' => "AAC",
            'D' => "GAC",
            'C' => "TGC",
            'Q' => "CAG",
            'E' => "GAG",
            'G' => "GGC",
            'H' => "CAC",
            'I' => "ATC",
            'L' => "CTG",
            'K' => "AAG",
            'M' => "ATG",
            'F' => "TTC",
            'P' => "CCC",
            'S' => "AGC",
            'T' => "ACC",
            'W' => "TGG",
            'Y' => "TAC",
            'V' => "GTG",
            _ => "NNN",
        },
        ExpressionHost::EColi => match aa {
            'A' => "GCG",
            'R' => "CGT",
            'N' => "AAC",
            'D' => "GAT",
            'C' => "TGC",
            'Q' => "CAG",
            'E' => "GAA",
            'G' => "GGC",
            'H' => "CAT",
            'I' => "ATT",
            'L' => "CTG",
            'K' => "AAA",
            'M' => "ATG",
            'F' => "TTT",
            'P' => "CCG",
            'S' => "AGC",
            'T' => "ACC",
            'W' => "TGG",
            'Y' => "TAT",
            'V' => "GTG",
            _ => "NNN",
        },
        ExpressionHost::Pichia => match aa {
            'A' => "GCT",
            'R' => "AGA",
            'N' => "AAC",
            'D' => "GAC",
            'C' => "TGT",
            'Q' => "CAA",
            'E' => "GAA",
            'G' => "GGT",
            'H' => "CAC",
            'I' => "ATT",
            'L' => "TTG",
            'K' => "AAG",
            'M' => "ATG",
            'F' => "TTC",
            'P' => "CCA",
            'S' => "TCT",
            'T' => "ACT",
            'W' => "TGG",
            'Y' => "TAC",
            'V' => "GTT",
            _ => "NNN",
        },
    }
}

/// Generate a host-optimized coding DNA sequence and audit GC%, homopolymers, and restriction sites.
#[must_use]
pub fn optimize_codons(protein_seq: &str, host: ExpressionHost) -> CodonOptReport {
    let mut dna = String::with_capacity(protein_seq.len() * 3 + 3);
    for ch in protein_seq.chars() {
        let cu = ch.to_ascii_uppercase();
        if "ACDEFGHIKLMNPQRSTVWY".contains(cu) {
            dna.push_str(preferred_codon(cu, host));
        }
    }
    dna.push_str("TGA");
    let gc = dna.chars().filter(|&c| c == 'G' || c == 'C').count();
    let gc_pct = if dna.is_empty() {
        0.0
    } else {
        ((gc as f64) / (dna.len() as f64) * 1000.0).round() / 10.0
    };
    let sites = [
        ("EcoRI", "GAATTC"),
        ("BamHI", "GGATCC"),
        ("HindIII", "AAGCTT"),
        ("XhoI", "CTCGAG"),
        ("NotI", "GCGGCCGC"),
        ("NdeI", "CATATG"),
        ("BsaI", "GGTCTC"),
    ];
    let restriction_sites_found: Vec<String> = sites
        .iter()
        .filter(|(_, pat)| dna.contains(pat))
        .map(|(n, pat)| format!("{n} ({pat})"))
        .collect();
    let homopolymer_alerts = ["AAAAAA", "TTTTTT", "CCCCCC", "GGGGGG"]
        .iter()
        .filter(|&&p| dna.contains(p))
        .count();

    let signal_peptide_recommendation = match host {
        ExpressionHost::Cho => {
            "Human IgK / mouse IgG1 heavy leader (MGWSCIILFLVATATGVHS)".to_string()
        }
        ExpressionHost::EColi => {
            "PelB periplasmic leader (MKYLLPTAAAGLLLLAAQPAMA) for oxidizing disulfide folding"
                .to_string()
        }
        ExpressionHost::Pichia => {
            "S. cerevisiae α-mating factor pre-pro secretion signal".to_string()
        }
    };

    CodonOptReport {
        host: host.label().to_string(),
        signal_peptide_recommendation,
        length_bp: dna.len(),
        dna_sequence: dna,
        gc_percent: gc_pct,
        restriction_sites_found,
        homopolymer_alerts,
    }
}

/// Render Markdown format comparison table + orthogonal experimental validation checklist.
#[must_use]
pub fn render_format_matrix_markdown(m: &FormatDecisionMatrix) -> String {
    let mut md = String::new();
    let _ = writeln!(
        md,
        "# Biologics Format Decision Matrix: {}\n",
        m.goal.target_name
    );
    let _ = writeln!(
        md,
        "- **Constraints:** Effector=`{}` | Long t½=`{}` | Deep Penetration=`{}` | SC=`{}` | Payload/ADC=`{}` | Bispecific=`{}`\n",
        m.goal.require_effector_function,
        m.goal.require_long_half_life,
        m.goal.require_deep_tissue_penetration,
        m.goal.require_subcutaneous,
        m.goal.require_payload_delivery,
        m.goal.require_bispecific
    );
    let _ = writeln!(
        md,
        "| Rank | Format | Score | MW (kDa) | Human t½ | Effector | Expression | Key Trade-off |"
    );
    let _ = writeln!(md, "|---:|---|---:|---:|---|---|---|---|");
    for (idx, r) in m.ranked_formats.iter().enumerate() {
        let _ = writeln!(
            md,
            "| {} | **{}** | {:.0} | {:.1} | {} | {} | {} | {} |",
            idx + 1,
            r.label,
            r.suitability_score,
            r.mw_kda,
            r.human_t_half_days,
            r.effector_function,
            r.expression_host,
            r.key_tradeoff
        );
    }
    let _ = writeln!(
        md,
        "\n## Orthogonal Wet-Lab Validation Controls (Go / No-Go)\n"
    );
    for c in &m.orthogonal_wetlab_plan {
        let _ = writeln!(md, "- {c}");
    }
    md
}
