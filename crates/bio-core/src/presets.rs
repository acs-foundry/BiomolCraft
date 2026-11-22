#![forbid(unsafe_code)]

//! Curated Clinical Biologics Reference Benchmarks (mAbs, VHH Nanobodies, TAP Negative Controls,
//! and Reference ADCs) with verified published variable-domain sequences.

use serde::{Deserialize, Serialize};

/// Clinical reference biologic benchmark preset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClinicalBiologicPreset {
    pub id: &'static str,
    pub name: &'static str,
    pub target_antigen: &'static str,
    pub format: &'static str,
    pub pdb_id: Option<&'static str>,
    pub clinical_status: &'static str,
    pub vh_or_vhh_seq: &'static str,
    pub vl_seq: Option<&'static str>,
    pub summary: &'static str,
}

/// Curated clinical biologics benchmark suite (positive & negative TAP/immunogenicity controls).
#[must_use]
pub fn clinical_biologic_presets() -> Vec<ClinicalBiologicPreset> {
    vec![
        ClinicalBiologicPreset {
            id: "trastuzumab",
            name: "Trastuzumab (Herceptin)",
            target_antigen: "Human HER2 (ERBB2 Domain IV)",
            format: "IgG1-WT",
            pdb_id: Some("1N8Z"),
            clinical_status: "FDA Approved (1998) — Gold-Standard Clinical Control",
            vh_or_vhh_seq: "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS",
            vl_seq: Some(
                "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK",
            ),
            summary: "Humanized anti-HER2 IgG1 (IGHV3-66 / IGKV1-39); textbook clean TAP developability and low clinical immunogenicity.",
        },
        ClinicalBiologicPreset {
            id: "adalimumab",
            name: "Adalimumab (Humira)",
            target_antigen: "Human TNF-alpha",
            format: "IgG1-WT",
            pdb_id: Some("3WD5"),
            clinical_status: "FDA Approved (2002) — Fully Human Phage-Display Benchmark",
            vh_or_vhh_seq: "EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSAITWNSGHIDYADSVEGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS",
            vl_seq: Some(
                "DIQMTQSPSSLSASVGDRVTITCRASQGIRNYLAWYQQKPGKAPKLLIYAASTLQSGVPSRFSGSGSGTDFTLTISSLQPEDVATYYCQRYNRAPYTFGQGTKVEIK",
            ),
            summary: "Fully human anti-TNFα IgG1 (IGHV3-9 / IGKV1-27); high SC concentration feasibility (100 mg/mL).",
        },
        ClinicalBiologicPreset {
            id: "pembrolizumab",
            name: "Pembrolizumab (Keytruda)",
            target_antigen: "Human PD-1",
            format: "IgG4-S228P",
            pdb_id: Some("5DK3"),
            clinical_status: "FDA Approved (2014) — Effector-Silenced IgG4-S228P",
            vh_or_vhh_seq: "QVQLVQSGVEVKKPGASVKVSCKASGYTFTNYYMYWVRQAPGQGLEWMGGINPSNGGTNFNEKFKNRVTLTTDSSTTTAYMELKSLQFDDTAVYYCARRDYRFDMGFDYWGQGTTVTVSS",
            vl_seq: Some(
                "EIVLTQSPATLSLSPGERATLSCRASKGVSTSGYSYLHWYQQKPGQAPRLLIYLASYLESGVPARFSGSGSGTDFTLTISSLEPEDFAVYYCQHSRDLPLTFGGGTKVEIK",
            ),
            summary: "Humanized anti-PD-1 IgG4-S228P checkpoint inhibitor (note CDR-H2 NG deamidation & CDR-H3 Met oxidation watch-points).",
        },
        ClinicalBiologicPreset {
            id: "bococizumab",
            name: "Bococizumab (RN316 — Negative Control)",
            target_antigen: "Human PCSK9",
            format: "IgG2Δa",
            pdb_id: None,
            clinical_status: "Discontinued Phase 3 (2016) — Immunogenicity & Developability Control",
            vh_or_vhh_seq: "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYYMHWVRQAPGQGLEWMGEISPFGGRTNYNEKFKSRVTMTRDTSTSTVYMELSSLRSEDTAVYYCARERPLYASDLWGQGTTVTVSS",
            vl_seq: Some(
                "DIQMTQSPSSLSASVGDRVTITCRASQGISSALAWYQQKPGKAPKLLIYSASYRYTGVPSRFSGSGSGTDFTFTISSLQPEDIATYYCQQRYSLWRTFGQGTKLEIK",
            ),
            summary: "Landmark clinical negative control: ~48% ADA immunogenicity in Phase 3 + high positive/hydrophobic CDR-L2/L3 polyreactivity.",
        },
        ClinicalBiologicPreset {
            id: "briakinumab",
            name: "Briakinumab (ABT-874 — TAP Positive-Charge Control)",
            target_antigen: "Human IL-12 / IL-23 p40",
            format: "IgG1-WT",
            pdb_id: None,
            clinical_status: "Withdrawn (2011) — Textbook TAP PPC / High-Clearance Negative Control",
            vh_or_vhh_seq: "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAFIRYDGSNKYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCKTHGSHDNWGQGTMVTVSS",
            vl_seq: Some(
                "QSVLTQPPSVSGAPGQRVTISCSGSRSNIGSNTVKWYQQLPGTAPKLLIYYNDQRPSGVPDRFSGSKSGTSASLAITGLQAEDEADYYCQSYDRYTHPALLFGTGTKVTVL",
            ),
            summary: "Canonical Jain/Raybould TAP negative control exhibiting excess positive CDR surface charge (PPC) driving non-specific FcRn/heparin binding and short half-life.",
        },
        ClinicalBiologicPreset {
            id: "envafolimab",
            name: "Envafolimab (KN035 VHH)",
            target_antigen: "Human PD-L1",
            format: "VHH-Fc",
            pdb_id: Some("5JDS"),
            clinical_status: "NMPA Approved (2021) — First Subcutaneous VHH-Fc Oncology Biologic",
            vh_or_vhh_seq: "QVQLQESGGGLVQPGGSLRLSCAASGKMSSRRCMAWFRQAPGKERERVAKLLTTSGSTYLADSVKGRFTISQNNAKSTVYLQMNSLKPEDTAMYYCAADSFEDPTCTLVTSSGAFQYWGQGTQVTVSS",
            vl_seq: None,
            summary: "Camelid anti-PD-L1 nanobody (PDB 5JDS): classic FR2 F37/E44/R45/E47 tetrad + extra CDR1–CDR3 inter-loop disulfide stabilizing a 21-aa convex paratope.",
        },
        ClinicalBiologicPreset {
            id: "caplacizumab",
            name: "Caplacizumab (Cablivi VHH)",
            target_antigen: "Human von Willebrand Factor (vWF A1)",
            format: "VHH-Bivalent",
            pdb_id: None,
            clinical_status: "FDA / EMA Approved (2018) — First Approved Nanobody",
            vh_or_vhh_seq: "EVQLVESGGGLVQPGGSLRLSCAASGRTFSYNPMGWFRQAPGKGRELVAAISRTGGSTYYPDSVEGRFTISRDNAKRMVYLQMNSLRAEDTAVYYCAAAGVRAEDGRVRTLPSEYTFWGQGTQVTVSS",
            vl_seq: None,
            summary: "Humanized camelid anti-vWF VHH domain preserving F37/R45 solubility hallmarks and 21-aa CDR3.",
        },
        ClinicalBiologicPreset {
            id: "t_dxd",
            name: "Trastuzumab deruxtecan (Enhertu / T-DXd)",
            target_antigen: "Human HER2 (High & HER2-Low)",
            format: "ADC (DAR 8, GGFG + DXd)",
            pdb_id: Some("1N8Z"),
            clinical_status: "FDA Approved (2019) — Next-Generation Bystander ADC Benchmark",
            vh_or_vhh_seq: "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS",
            vl_seq: Some(
                "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK",
            ),
            summary: "Trastuzumab IgG1 conjugated via interchain Cys (DAR 8) to enzyme-cleavable GGFG linker + topoisomerase-I inhibitor DXd (high bystander permeability).",
        },
        ClinicalBiologicPreset {
            id: "t_dm1",
            name: "Trastuzumab emtansine (Kadcyla / T-DM1)",
            target_antigen: "Human HER2 (HER2-High)",
            format: "ADC (DAR 3.5, SMCC + DM1)",
            pdb_id: Some("1N8Z"),
            clinical_status: "FDA Approved (2013) — Non-Cleavable Lysine ADC Benchmark",
            vh_or_vhh_seq: "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS",
            vl_seq: Some(
                "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK",
            ),
            summary: "Trastuzumab IgG1 randomly conjugated on Lys (DAR 3.5) via non-cleavable SMCC linker to DM1; releases charged Lys-SMCC-DM1 (minimal bystander effect).",
        },
    ]
}

/// Find a clinical biologic preset by ID or partial name.
#[must_use]
pub fn find_biologic_preset(query: &str) -> Option<ClinicalBiologicPreset> {
    let q = query.trim().to_ascii_lowercase().replace([' ', '-'], "_");
    clinical_biologic_presets().into_iter().find(|p| {
        p.id == q
            || p.name
                .to_ascii_lowercase()
                .replace([' ', '-'], "_")
                .contains(&q)
    })
}
