#![forbid(unsafe_code)]

//! `biomol-modules` — Pluggable Schrödinger Maestro / BioLuminate / Craft-Anything Style
//! Biomolecular Module Registry connecting `BioCraft` compute crates (`bio-core`, `bio-develop`,
//! `bio-interface`, `bio-adc`) + `biomol-rna` directly onto live 3D `Biomolecule` scenes.

use std::collections::HashMap;
use std::fmt::Write as _;

pub use bio_adc::BiologicsFormat;
use bio_adc::{
    AdcDesignReport, AdcDesignRequest, BiologicsPkReport, BiologicsPkRequest, ConjugationStrategy,
    design_adc, render_adc_markdown, render_biopk_markdown, simulate_biologics_pk_tmdd,
};
use bio_core::{IgChainType, RegionKind, annotate_chain, parse_pdb as parse_bio_pdb};
use bio_develop::{DevelopabilityReport, evaluate_developability, render_developability_markdown};
use bio_engine::audit_payload_with_admetcraft;
use bio_interface::{
    AlaScanReport, DeNovoRequest, DeNovoStagePlan, InterfaceReport, analyze_ppi_interface,
    render_denovo_markdown, render_interface_markdown, run_alanine_scan, stage_denovo_binder,
};
use biomol_enzyme::{
    EnzymeEngineeringReport, analyze_enzyme_engineering, enzyme_visual_overlays,
    render_enzyme_markdown,
};
use biomol_physics::{
    AnmReport, GbSaReport, anm_mode_porcupine_overlays, compute_anm_modes, compute_gbsa_energy,
    generate_anm_trajectory_frames, render_anm_markdown, render_gbsa_markdown,
};
use biomol_rna::{NcRnaAnalysisReport, analyze_ncrna_mrna, render_rna_markdown};
use craftbio_report::craft_data_dir;
use craftbio_sandbox::{WasmPluginReport, inspect_wasm_plugin};
use craftbio_scene::{
    ColorScheme, LineOverlay, ProvenanceEntry, ProvenanceKind, RepresentationKind, SphereOverlay,
};
use craftbio_struct::{
    Biomolecule, MapFitReport, ResidueClass, VolumeGrid3D, compute_sasa, extract_isosurface_nodes,
    fit_model_in_map, simulate_cryoem_map, write_pdb,
};
use serde::{Deserialize, Serialize};

/// First-class loadable biomolecular module identifiers inside `BiomolCraft`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModuleId {
    AntibodiesVhh,
    NcRnaMrna,
    EnzymesCatalysis,
    AdcsBioconjugates,
    BiologicsPkTmdd,
    DeNovoCoFold,
}

impl ModuleId {
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::AntibodiesVhh => "Antibodies & VHH (BioLuminate)",
            Self::NcRnaMrna => "ncRNA, mRNA & Nucleic Acids",
            Self::EnzymesCatalysis => "Enzymes & Active Sites",
            Self::AdcsBioconjugates => "ADCs & Conjugation Sites",
            Self::BiologicsPkTmdd => "Biologics PK & TMDD",
            Self::DeNovoCoFold => "De Novo & Co-Fold Bridge",
        }
    }

    #[must_use]
    pub fn short_key(self) -> &'static str {
        match self {
            Self::AntibodiesVhh => "antibodies",
            Self::NcRnaMrna => "ncrna",
            Self::EnzymesCatalysis => "enzymes",
            Self::AdcsBioconjugates => "adc",
            Self::BiologicsPkTmdd => "pk",
            Self::DeNovoCoFold => "denovo",
        }
    }
}

/// Visual & scientific patch produced by executing a `BiomolCraft` module against the active 3D model.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModuleExecutionOutput {
    pub module_name: String,
    pub recommended_repr: Option<RepresentationKind>,
    pub recommended_color: Option<ColorScheme>,
    pub residue_highlights: HashMap<(char, i32), [f32; 3]>,
    pub sphere_overlays: Vec<SphereOverlay>,
    pub line_overlays: Vec<LineOverlay>,
    pub summary_markdown: String,
    pub provenance: Vec<ProvenanceEntry>,
}

/// Detailed antibody/VHH module result bundle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntibodyModuleBundle {
    pub visual: ModuleExecutionOutput,
    pub developability: DevelopabilityReport,
    pub interface: Option<InterfaceReport>,
    pub alascan: Option<AlaScanReport>,
}

/// Execute the **Antibodies & VHH Module** on a 3D `Biomolecule`:
/// Auto-detects VH/VL/VHH vs Antigen chains (or uses explicit chains), paints IMGT CDR1/2/3 in
/// gold/orange/crimson on the 3D ribbon, places glowing Ala-Scan hotspot spheres & H-bond lines!
pub fn run_antibody_module(
    mol: &Biomolecule,
    antigen_override: Option<&[char]>,
    binder_override: Option<&[char]>,
) -> Result<AntibodyModuleBundle, String> {
    let mut binder_chains = Vec::new();
    let mut antigen_chains = Vec::new();
    let mut highlights: HashMap<(char, i32), [f32; 3]> = HashMap::new();
    let mut binder_seq_pairs: Vec<(String, String)> = Vec::new();

    for ch in &mol.chains {
        if ch.dominant_class != ResidueClass::Protein || ch.sequence.len() < 40 {
            continue;
        }
        let ann = annotate_chain(&format!("Chain_{}", ch.id), &ch.sequence);
        let is_ig = ann.chain_type.is_variable_ig();
        let force_binder = binder_override.is_some_and(|b| b.contains(&ch.id));
        let force_antigen = antigen_override.is_some_and(|a| a.contains(&ch.id));

        if force_binder || (!force_antigen && is_ig) {
            binder_chains.push(ch.id);
            binder_seq_pairs.push((format!("Chain_{}", ch.id), ch.sequence.clone()));
            // Paint IMGT CDR1 (gold), CDR2 (orange), CDR3 (crimson) + VHH FR2 hallmarks (cyan)
            let prot_res: Vec<_> = ch
                .residues
                .iter()
                .filter(|r| r.class == ResidueClass::Protein)
                .collect();
            for (ri, nr) in ann.numbered.iter().enumerate() {
                if let Some(res3d) = prot_res.get(ri) {
                    let col = match nr.region {
                        RegionKind::Cdr1 => Some([0.98, 0.75, 0.14]),
                        RegionKind::Cdr2 => Some([0.96, 0.45, 0.12]),
                        RegionKind::Cdr3 => Some([0.92, 0.16, 0.34]),
                        _ if nr.is_vhh_hallmark => Some([0.15, 0.88, 0.96]),
                        _ => None,
                    };
                    if let Some(rgb) = col {
                        highlights.insert((ch.id, res3d.res_seq), rgb);
                    }
                }
            }
        } else if force_antigen || (!force_binder && ann.chain_type == IgChainType::GeneralProtein)
        {
            antigen_chains.push(ch.id);
        }
    }

    if binder_chains.is_empty() {
        return Err("no antibody VH/VL/VHH chains detected in active 3D structure".to_string());
    }

    // Bridge to BioCraft's 3D SASA developability & PPI/Ala-Scan engine cleanly
    let pdb_text = write_pdb(mol);
    let bio_st = parse_bio_pdb(&pdb_text).map_err(|e| e.to_string())?;
    let refs: Vec<(&str, &str)> = binder_seq_pairs
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    let dev = evaluate_developability(&mol.title, &refs, Some(&bio_st));

    let mut spheres = Vec::new();
    let mut lines = Vec::new();
    let mut iface_opt = None;
    let mut ala_opt = None;

    if !antigen_chains.is_empty() {
        let iface = analyze_ppi_interface(&bio_st, &antigen_chains, &binder_chains)
            .map_err(|e| e.to_string())?;
        let ala = run_alanine_scan(&bio_st, &antigen_chains, &binder_chains)
            .map_err(|e| e.to_string())?;

        // Glowing 3D spheres on Ala-Scan hotspots (ddG >= 1.5 kcal/mol)
        for h in ala
            .paratope_hotspots
            .iter()
            .filter(|h| h.ddg_kcal_mol >= 1.5)
            .take(8)
        {
            if let Some(ch) = mol.chain(h.chain)
                && let Some(r) = ch.residues.iter().find(|r| r.res_seq == h.res_seq)
                && let Some(gi) = r.guide_idx
            {
                spheres.push(SphereOverlay {
                    label: format!("{}:{} (+{:.1})", h.chain, h.res_label, h.ddg_kcal_mol),
                    category: "Paratope Hotspot".to_string(),
                    coords: mol.atoms[gi].coords,
                    radius_a: 1.35,
                    rgba: [0.95, 0.22, 0.28, 0.72],
                });
            }
        }
        for h in ala
            .epitope_hotspots
            .iter()
            .filter(|h| h.ddg_kcal_mol >= 1.5)
            .take(8)
        {
            if let Some(ch) = mol.chain(h.chain)
                && let Some(r) = ch.residues.iter().find(|r| r.res_seq == h.res_seq)
                && let Some(gi) = r.guide_idx
            {
                spheres.push(SphereOverlay {
                    label: format!("{}:{} (+{:.1})", h.chain, h.res_label, h.ddg_kcal_mol),
                    category: "Epitope Hotspot".to_string(),
                    coords: mol.atoms[gi].coords,
                    radius_a: 1.25,
                    rgba: [0.98, 0.58, 0.14, 0.72],
                });
            }
        }
        // Dashed cyan/gold lines for key inter-chain H-bonds & salt bridges
        for c in iface.key_contacts.iter().take(16) {
            let bseq: i32 = c
                .binder_res
                .chars()
                .skip(1)
                .collect::<String>()
                .parse()
                .unwrap_or(0);
            let aseq: i32 = c
                .antigen_res
                .chars()
                .skip(1)
                .collect::<String>()
                .parse()
                .unwrap_or(0);
            let ba = mol.atoms.iter().find(|a| {
                a.chain == c.binder_chain && a.res_seq == bseq && a.name == c.binder_atom
            });
            let aa = mol.atoms.iter().find(|a| {
                a.chain == c.antigen_chain && a.res_seq == aseq && a.name == c.antigen_atom
            });
            if let (Some(p1), Some(p2)) = (ba, aa) {
                lines.push(LineOverlay {
                    label: format!("{} {:.2}Å", c.kind, c.distance_a),
                    from: p1.coords,
                    to: p2.coords,
                    dashed: true,
                    rgba: if c.kind == "Salt Bridge" {
                        [0.98, 0.80, 0.18, 0.95]
                    } else {
                        [0.22, 0.84, 0.98, 0.92]
                    },
                });
            }
        }
        iface_opt = Some(iface);
        ala_opt = Some(ala);
    }

    let mut md = render_developability_markdown(&dev);
    if let Some(ref iface) = iface_opt {
        md.push('\n');
        md.push_str(&render_interface_markdown(iface, ala_opt.as_ref()));
    }

    Ok(AntibodyModuleBundle {
        visual: ModuleExecutionOutput {
            module_name: ModuleId::AntibodiesVhh.label().to_string(),
            recommended_repr: Some(RepresentationKind::Cartoon),
            recommended_color: Some(ColorScheme::Chain),
            residue_highlights: highlights,
            sphere_overlays: spheres,
            line_overlays: lines,
            summary_markdown: md,
            provenance: vec![ProvenanceEntry {
                kind: ProvenanceKind::ComputationalPrediction,
                source_app: "biomolcraft::mod-antibodies (via BioCraft crates)".to_string(),
                summary: format!(
                    "Annotated binder chains {:?} vs antigen {:?}; TAP={} ({:.0}/100).",
                    binder_chains,
                    antigen_chains,
                    dev.tap_overall.as_str(),
                    dev.developability_score_0_100
                ),
            }],
        },
        developability: dev,
        interface: iface_opt,
        alascan: ala_opt,
    })
}

/// Execute the **ncRNA & mRNA Module** on any nucleic-acid chain in the active 3D `Biomolecule`.
pub fn run_ncrna_module(
    mol: &Biomolecule,
) -> Result<(ModuleExecutionOutput, Vec<NcRnaAnalysisReport>), String> {
    let mut reports = Vec::new();
    let mut highlights: HashMap<(char, i32), [f32; 3]> = HashMap::new();
    let mut md = String::new();

    for ch in &mol.chains {
        if !matches!(ch.dominant_class, ResidueClass::Rna | ResidueClass::Dna)
            || ch.sequence.len() < 8
        {
            continue;
        }
        let ch_bps: Vec<_> = mol
            .base_pairs
            .iter()
            .filter(|bp| bp.chain_a == ch.id || bp.chain_b == ch.id)
            .cloned()
            .collect();
        let rep = analyze_ncrna_mrna(
            &format!("{}_Chain_{}", mol.id, ch.id),
            if ch.dominant_class == ResidueClass::Rna {
                "ncRNA / RNA 3D Structure"
            } else {
                "DNA 3D Structure"
            },
            &ch.sequence,
            Some(&ch_bps),
        )
        .map_err(|e| e.to_string())?;

        // Highlight paired stems vs unpaired loops on the 3D RNA ribbon
        let nuc_res: Vec<_> = ch
            .residues
            .iter()
            .filter(|r| matches!(r.class, ResidueClass::Rna | ResidueClass::Dna))
            .collect();
        for (idx, ch_db) in rep.fold.dot_bracket.chars().enumerate() {
            if let Some(res3d) = nuc_res.get(idx) {
                let col = match ch_db {
                    '(' | ')' => [0.20, 0.85, 0.62], // stem emerald
                    _ => [0.98, 0.62, 0.22],         // loop/junction amber
                };
                highlights.insert((ch.id, res3d.res_seq), col);
            }
        }
        md.push_str(&render_rna_markdown(&rep));
        md.push('\n');
        reports.push(rep);
    }

    if reports.is_empty() {
        return Err("no RNA/DNA chains >= 8 nt found in active 3D structure".to_string());
    }

    let out = ModuleExecutionOutput {
        module_name: ModuleId::NcRnaMrna.label().to_string(),
        recommended_repr: Some(RepresentationKind::NucleicLadder),
        recommended_color: Some(ColorScheme::Nucleotide),
        residue_highlights: highlights,
        sphere_overlays: Vec::new(),
        line_overlays: Vec::new(),
        summary_markdown: md,
        provenance: vec![ProvenanceEntry {
            kind: ProvenanceKind::ComputationalPrediction,
            source_app: "biomolcraft::mod-ncrna (Zuker-style MFE in-house stacking model)"
                .to_string(),
            summary: format!(
                "Analyzed {} nucleic chain(s) and cross-checked {} 3D crystal base pairs.",
                reports.len(),
                mol.base_pairs.len()
            ),
        }],
    };
    Ok((out, reports))
}

/// Execute the **Enzymes & Protein Engineering Module** (`biomol-enzyme`): detects catalytic
/// triads/dyads, metal/cofactor coordination spheres, 3D translucent substrate access tunnels,
/// and thermostabilizing disulfide bridge proposals directly in the 3D viewport.
#[must_use]
pub fn run_enzyme_module(mol: &Biomolecule) -> ModuleExecutionOutput {
    let (vis, _) = run_enzyme_engineering_module(mol);
    vis
}

/// Full structured `biomol-enzyme` module execution returning both 3D overlays and report.
#[must_use]
pub fn run_enzyme_engineering_module(
    mol: &Biomolecule,
) -> (ModuleExecutionOutput, EnzymeEngineeringReport) {
    let rep = analyze_enzyme_engineering(mol);
    let mut highlights: HashMap<(char, i32), [f32; 3]> = HashMap::new();
    for m in &rep.catalytic_motifs {
        for (idx, (ch, rseq, _)) in m.residues.iter().enumerate() {
            let col = match idx {
                0 => [0.20, 0.90, 0.55],
                1 => [0.96, 0.25, 0.45],
                _ => [0.98, 0.72, 0.16],
            };
            highlights.insert((*ch, *rseq), col);
        }
    }
    for s in &rep.metal_cofactor_sites {
        for (ch, rseq, _, _) in &s.coordinating_residues {
            highlights.entry((*ch, *rseq)).or_insert([0.82, 0.44, 0.98]);
        }
    }
    let (spheres, lines) = enzyme_visual_overlays(&rep);
    let md = render_enzyme_markdown(&rep);
    let vis = ModuleExecutionOutput {
        module_name: ModuleId::EnzymesCatalysis.label().to_string(),
        recommended_repr: Some(RepresentationKind::Cartoon),
        recommended_color: Some(ColorScheme::SecondaryStructure),
        residue_highlights: highlights,
        sphere_overlays: spheres,
        line_overlays: lines,
        summary_markdown: md,
        provenance: vec![ProvenanceEntry {
            kind: ProvenanceKind::ComputationalPrediction,
            source_app: "biomolcraft::biomol-enzyme".to_string(),
            summary: format!(
                "Enzyme audit: {} catalytic motif(s), {} metal/cofactor site(s), {} 3D tunnel(s), {} disulfide proposal(s).",
                rep.catalytic_motifs.len(),
                rep.metal_cofactor_sites.len(),
                rep.tunnels.len(),
                rep.disulfide_proposals.len()
            ),
        }],
    };
    (vis, rep)
}

/// Run ANM Normal Mode Analysis on `mol`, populate smooth animation frames along `mode_1based`,
/// and emit 3D porcupine displacement arrows.
pub fn run_physics_anm_module(
    mol: &mut Biomolecule,
    mode_1based: usize,
    amplitude_a: f64,
    num_frames: usize,
) -> Result<(ModuleExecutionOutput, AnmReport), String> {
    let rep = compute_anm_modes(mol, 13.0, 6)?;
    let n_frames = generate_anm_trajectory_frames(mol, &rep, mode_1based, amplitude_a, num_frames)?;
    let lines = anm_mode_porcupine_overlays(mol, &rep, mode_1based, amplitude_a * 2.0);
    let mut highlights = HashMap::new();
    for rm in &rep.residue_mobility {
        if rm.is_hinge {
            highlights.insert((rm.chain, rm.res_seq), [0.98, 0.80, 0.14]);
        }
    }
    let md = render_anm_markdown(&rep);
    let vis = ModuleExecutionOutput {
        module_name: "Physics — ANM Normal Modes".to_string(),
        recommended_repr: Some(RepresentationKind::Cartoon),
        recommended_color: Some(ColorScheme::Chain),
        residue_highlights: highlights,
        sphere_overlays: Vec::new(),
        line_overlays: lines,
        summary_markdown: md,
        provenance: vec![ProvenanceEntry {
            kind: ProvenanceKind::ComputationalPrediction,
            source_app: "biomolcraft::biomol-physics (ANM)".to_string(),
            summary: format!(
                "ANM {} nodes, mode {} animated across {} frames ({} hinge residues).",
                rep.guide_nodes,
                mode_1based,
                n_frames,
                rep.hinge_residues.len()
            ),
        }],
    };
    Ok((vis, rep))
}

/// Run pure-Rust Still GB/SA solvation, 3D Electrostatic Dipole, and interface `ΔG_bind`.
#[must_use]
pub fn run_physics_gbsa_module(mol: &Biomolecule) -> (ModuleExecutionOutput, GbSaReport) {
    let rep = compute_gbsa_energy(mol, None, None);
    let mut lines = Vec::new();
    let d = rep.dipole.dipole_vector_debye;
    let mag = rep.dipole.dipole_magnitude_debye;
    if mag > 1.0 {
        let scale = 14.0 / mag;
        let c = rep.dipole.center_of_geometry;
        lines.push(LineOverlay {
            label: format!("Dipole {mag:.1} D"),
            from: c,
            to: [
                c[0] + d[0] * scale,
                c[1] + d[1] * scale,
                c[2] + d[2] * scale,
            ],
            dashed: false,
            rgba: [0.30, 0.86, 0.98, 0.98],
        });
    }
    let md = render_gbsa_markdown(&rep);
    let vis = ModuleExecutionOutput {
        module_name: "Physics — GB/SA & Dipole".to_string(),
        recommended_repr: Some(RepresentationKind::Cartoon),
        recommended_color: Some(ColorScheme::ElectrostaticEsp),
        residue_highlights: HashMap::new(),
        sphere_overlays: Vec::new(),
        line_overlays: lines,
        summary_markdown: md,
        provenance: vec![ProvenanceEntry {
            kind: ProvenanceKind::ComputationalPrediction,
            source_app: "biomolcraft::biomol-physics (GB/SA)".to_string(),
            summary: format!(
                "GB/SA total {:.2} kcal/mol, Dipole {:.1} D, dG_bind={:?}.",
                rep.complex_energy.total_kcal,
                rep.dipole.dipole_magnitude_debye,
                rep.delta_g_bind_kcal
            ),
        }],
    };
    (vis, rep)
}

/// Execute the **ADC & Conjugation-Site 3D Module**: maps solvent-exposed surface Cys/Lys sites
/// via Shrake-Rupley SASA + audits the ADC construct with `bio-adc` & live `AdmetCraft`.
pub fn run_adc_module(
    mol: &Biomolecule,
    linker_id: &str,
    payload_id: &str,
    dar: f64,
    run_admetcraft: bool,
) -> Result<(ModuleExecutionOutput, AdcDesignReport), String> {
    let mut req = AdcDesignRequest {
        name: format!("{} ADC", mol.id),
        antibody_mw_da: 148_000.0,
        linker_id: linker_id.to_string(),
        payload_id: payload_id.to_string(),
        custom_payload_smiles: None,
        dar,
        conjugation: ConjugationStrategy::InterchainCys,
        admetcraft_payload_summary: None,
    };
    let mut rep = design_adc(&req).map_err(|e| e.to_string())?;
    if run_admetcraft && let Some(adm) = audit_payload_with_admetcraft(&rep.payload.smiles) {
        req.admetcraft_payload_summary = Some(adm);
        rep = design_adc(&req).map_err(|e| e.to_string())?;
    }

    // Identify top solvent-accessible surface Lys/Cys conjugation sites in 3D via Shrake-Rupley SASA
    let sasa = compute_sasa(&mol.atoms, 1.4, 24);
    let mut sites: Vec<(char, i32, char, f64, [f64; 3])> = Vec::new();
    for ch in &mol.chains {
        for r in &ch.residues {
            if r.class == ResidueClass::Protein && matches!(r.code, 'K' | 'C') {
                let rs: f64 = r.atom_indices.iter().map(|&i| sasa[i]).sum();
                if rs >= 85.0
                    && let Some(gi) = r.guide_idx
                {
                    sites.push((ch.id, r.res_seq, r.code, rs, mol.atoms[gi].coords));
                }
            }
        }
    }
    sites.sort_by(|a, b| b.3.total_cmp(&a.3));

    let mut highlights = HashMap::new();
    let mut spheres = Vec::new();
    let mut md = render_adc_markdown(&rep);
    let _ = writeln!(
        md,
        "\n## Top 3D Solvent-Accessible Conjugation Sites (SASA >= 85 Å²)\n"
    );
    for &(c, rseq, aa, rs, pos) in sites.iter().take(10) {
        highlights.insert((c, rseq), [0.95, 0.25, 0.85]);
        spheres.push(SphereOverlay {
            label: format!("Conj {c}:{aa}{rseq} ({rs:.0} Å²)"),
            category: "ADC Site".to_string(),
            coords: pos,
            radius_a: 1.15,
            rgba: [0.92, 0.28, 0.85, 0.72],
        });
        let _ = writeln!(md, "- `{c}:{aa}{rseq}` (SASA `{rs:.1} Å²`)");
    }

    Ok((
        ModuleExecutionOutput {
            module_name: ModuleId::AdcsBioconjugates.label().to_string(),
            recommended_repr: Some(RepresentationKind::Cartoon),
            recommended_color: Some(ColorScheme::Chain),
            residue_highlights: highlights,
            sphere_overlays: spheres,
            line_overlays: Vec::new(),
            summary_markdown: md,
            provenance: vec![ProvenanceEntry {
                kind: ProvenanceKind::ComputationalPrediction,
                source_app: "biomolcraft::mod-adc (via bio-adc + AdmetCraft)".to_string(),
                summary: format!(
                    "ADC DAR {:.1} HDI={:.2} Bystander={:.0}/100.",
                    rep.dar, rep.hydrophobic_drug_load_index, rep.bystander_score_0_100
                ),
            }],
        },
        rep,
    ))
}

/// Run the Biologics Minimal-PBPK + FcRn + TMDD module.
pub fn run_pk_tmdd_module(
    name: &str,
    format: BiologicsFormat,
    dose_mg: f64,
    interval_days: f64,
) -> Result<(BiologicsPkReport, String), String> {
    let req = BiologicsPkRequest {
        name: name.to_string(),
        format,
        dose_mg,
        interval_days,
        ..BiologicsPkRequest::default()
    };
    let rep = simulate_biologics_pk_tmdd(&req).map_err(|e| e.to_string())?;
    let md = render_biopk_markdown(&rep);
    Ok((rep, md))
}

/// Synthesize a Cryo-EM volumetric electron density map (`molmap`) and project translucent
/// contour isosurface beads directly onto the 3D scene (`craftbio-viewport`).
pub fn run_cryoem_molmap_module(
    mol: &mut Biomolecule,
    resolution_a: f64,
    sigma_level: f64,
) -> Result<(ModuleExecutionOutput, VolumeGrid3D, MapFitReport), String> {
    let grid = simulate_cryoem_map(mol, resolution_a, 1.5).map_err(|e| e.to_string())?;
    let (vis, fit) = run_cryoem_volume_module(mol, &grid, sigma_level, false);
    Ok((vis, grid, fit))
}

/// Render an external or synthesized `VolumeGrid3D` (`.mrc`/`.map`/`.dx`) isosurface contour
/// and evaluate / optimize rigid `fitmap` cross-correlation against `mol`.
#[must_use]
pub fn run_cryoem_volume_module(
    mol: &mut Biomolecule,
    grid: &VolumeGrid3D,
    sigma_level: f64,
    optimize_fit: bool,
) -> (ModuleExecutionOutput, MapFitReport) {
    let fit = fit_model_in_map(mol, grid, if optimize_fit { 16 } else { 0 });
    let iso = extract_isosurface_nodes(grid, sigma_level, 1400);
    let spheres: Vec<SphereOverlay> = iso
        .into_iter()
        .map(|(pt, rad)| SphereOverlay {
            label: format!("{} ({sigma_level:.1}σ)", grid.label),
            category: "Cryo-EM Density".to_string(),
            coords: pt,
            radius_a: rad,
            rgba: [0.42, 0.78, 0.98, 0.24],
        })
        .collect();
    let mut md = String::new();
    let _ = writeln!(md, "# Cryo-EM & Volumetric Map Audit: {}\n", grid.label);
    let _ = writeln!(
        md,
        "- **Grid Dimensions:** `{} × {} × {}` | **Voxel Size:** `{:.2} Å` | **Contour:** `{sigma_level:.2} σ` (`{} surface beads`)",
        grid.dims[0],
        grid.dims[1],
        grid.dims[2],
        grid.voxel_size[0],
        spheres.len()
    );
    let _ = writeln!(
        md,
        "- **Map-to-Model Correlation (`fitmap`):** **`{:.3} → {:.3}`** (`{} atoms`, mean density `{:.2} σ`, **`{:.1}%`** inside `>= 1.0 σ`)",
        fit.correlation_before,
        fit.correlation_after,
        fit.atoms_evaluated,
        fit.mean_density_sigma,
        fit.atoms_inside_1sigma_pct
    );
    let _ = writeln!(
        md,
        "- **Rigid Translation Applied:** `[{:.2}, {:.2}, {:.2}] Å`",
        fit.translation_applied_a[0], fit.translation_applied_a[1], fit.translation_applied_a[2]
    );
    let vis = ModuleExecutionOutput {
        module_name: "Cryo-EM & Volumetric Maps".to_string(),
        recommended_repr: Some(RepresentationKind::Cartoon),
        recommended_color: Some(ColorScheme::Chain),
        residue_highlights: HashMap::new(),
        sphere_overlays: spheres,
        line_overlays: Vec::new(),
        summary_markdown: md,
        provenance: vec![ProvenanceEntry {
            kind: ProvenanceKind::ComputationalPrediction,
            source_app: "biomolcraft::cryoem-volume".to_string(),
            summary: format!(
                "Volumetric map '{}' at {:.1}σ: fitmap CC={:.3}, {:.1}% atoms >= 1σ.",
                grid.label, sigma_level, fit.correlation_after, fit.atoms_inside_1sigma_pct
            ),
        }],
    };
    (vis, fit)
}

/// Stage a zero-weight **De Novo & Co-Fold** (`RFdiffusion` + `ProteinMPNN` + `ESM2` + `Boltz-2/Chai-1`)
/// binder campaign directly from 3D epitope hotspot residues on `target_chain`.
pub fn run_denovo_cofold_module(
    mol: &Biomolecule,
    target_chain: char,
    hotspots: &[String],
    out_dir: Option<&std::path::Path>,
) -> Result<(ModuleExecutionOutput, DeNovoStagePlan), String> {
    let ch = mol
        .chain(target_chain)
        .or_else(|| mol.chains.last())
        .ok_or_else(|| format!("target chain '{target_chain}' not found"))?;
    let protein_res: Vec<_> = ch
        .residues
        .iter()
        .filter(|r| r.class == ResidueClass::Protein)
        .collect();
    if protein_res.is_empty() {
        return Err(format!("chain '{}' contains no protein residues", ch.id));
    }
    let min_r = protein_res.first().map_or(1, |r| r.res_seq);
    let max_r = protein_res.last().map_or(100, |r| r.res_seq);

    // Auto-select top surface-accessible residues on target chain if caller omitted hotspots
    let chosen_hotspots: Vec<String> = if !hotspots.is_empty() {
        hotspots.to_vec()
    } else {
        let sasa = compute_sasa(&mol.atoms, 1.4, 20);
        let mut ranked: Vec<(i32, f64)> = protein_res
            .iter()
            .map(|r| {
                let rs: f64 = r.atom_indices.iter().map(|&i| sasa[i]).sum();
                (r.res_seq, rs)
            })
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        ranked
            .into_iter()
            .take(5)
            .map(|(rseq, _)| format!("{}{rseq}", ch.id))
            .collect()
    };

    let req = DeNovoRequest {
        campaign_name: format!("{}_chain{}_binder", mol.id, ch.id),
        target_pdb_path: format!("{}.pdb", mol.id),
        target_chain: ch.id,
        target_residue_range: format!("{min_r}-{max_r}"),
        hotspot_residues: chosen_hotspots.clone(),
        binder_length_min: 65,
        binder_length_max: 95,
        num_designs: 50,
        mpnn_seqs_per_backbone: 8,
        cofold_backend: "boltz2".to_string(),
    };
    let plan = stage_denovo_binder(&req);
    let md = render_denovo_markdown(&plan);

    if let Some(dir) = out_dir {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        std::fs::write(dir.join("target.pdb"), write_pdb(mol)).map_err(|e| e.to_string())?;
        std::fs::write(dir.join("denovo_plan.md"), &md).map_err(|e| e.to_string())?;
        std::fs::write(dir.join("score_esm2.py"), &plan.esm2_scoring_script)
            .map_err(|e| e.to_string())?;
        std::fs::write(
            dir.join("fixed_chains.jsonl"),
            &plan.proteinmpnn_fixed_chains_jsonl,
        )
        .map_err(|e| e.to_string())?;
    }

    let mut highlights = HashMap::new();
    let mut spheres = Vec::new();
    for hs in &chosen_hotspots {
        let digits: String = hs
            .chars()
            .filter(|c| c.is_ascii_digit() || *c == '-')
            .collect();
        if let Ok(rseq) = digits.parse::<i32>()
            && let Some(r) = protein_res.iter().find(|r| r.res_seq == rseq)
            && let Some(gi) = r.guide_idx
        {
            highlights.insert((ch.id, rseq), [0.98, 0.22, 0.38]);
            spheres.push(SphereOverlay {
                label: format!("Hotspot {hs}"),
                category: "DeNovo Hotspot".to_string(),
                coords: mol.atoms[gi].coords,
                radius_a: 1.65,
                rgba: [0.98, 0.20, 0.38, 0.78],
            });
        }
    }

    let vis = ModuleExecutionOutput {
        module_name: ModuleId::DeNovoCoFold.label().to_string(),
        recommended_repr: Some(RepresentationKind::Cartoon),
        recommended_color: Some(ColorScheme::Chain),
        residue_highlights: highlights,
        sphere_overlays: spheres,
        line_overlays: Vec::new(),
        summary_markdown: md,
        provenance: vec![ProvenanceEntry {
            kind: ProvenanceKind::ComputationalPrediction,
            source_app: "biomolcraft::mod-denovo (RFdiffusion/MPNN/Boltz-2 Staging)".to_string(),
            summary: format!(
                "Staged de novo binder campaign on chain {} with {} hotspots (0 GB weights downloaded).",
                ch.id,
                chosen_hotspots.len()
            ),
        }],
    };
    Ok((vis, plan))
}

/// Declarative external plugin manifest (`$CRAFT_DATA_DIR/plugins/*.json`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeclarativePluginManifest {
    pub id: String,
    pub name: String,
    pub version: String,
    pub description: String,
    pub dsl_macro: String,
}

/// Discovered external plugin entry (either safe declarative JSON macro or validated `.wasm` binary).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredPlugin {
    pub file_name: String,
    pub kind: String,
    pub summary: String,
    pub wasm_report: Option<WasmPluginReport>,
    pub json_manifest: Option<DeclarativePluginManifest>,
}

/// Discover and safely validate declarative `.json` and `.wasm` plugins under `$CRAFT_DATA_DIR/plugins`.
#[must_use]
pub fn discover_installed_plugins() -> Vec<DiscoveredPlugin> {
    let dir = craft_data_dir().join("plugins");
    let Ok(rd) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in rd.flatten() {
        let path = entry.path();
        let fname = entry.file_name().to_string_lossy().to_string();
        match path.extension().and_then(|s| s.to_str()) {
            Some("json") => {
                if let Ok(raw) = std::fs::read_to_string(&path)
                    && let Ok(man) = serde_json::from_str::<DeclarativePluginManifest>(&raw)
                {
                    out.push(DiscoveredPlugin {
                        file_name: fname,
                        kind: "Declarative DSL Macro".to_string(),
                        summary: format!("{} v{} — {}", man.name, man.version, man.description),
                        wasm_report: None,
                        json_manifest: Some(man),
                    });
                }
            }
            Some("wasm") => {
                if let Ok(bytes) = std::fs::read(&path)
                    && let Ok(wrep) = inspect_wasm_plugin(&bytes)
                {
                    out.push(DiscoveredPlugin {
                        file_name: fname,
                        kind: "Validated WASM Module (Read-Only Inspect)".to_string(),
                        summary: format!(
                            "WASM v{}, {} bytes, {} sections, exports: {:?}",
                            wrep.version, wrep.total_bytes, wrep.section_count, wrep.exports
                        ),
                        wasm_report: Some(wrep),
                        json_manifest: None,
                    });
                }
            }
            _ => {}
        }
    }
    out
}
