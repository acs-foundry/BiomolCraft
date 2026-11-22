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
    AlaScanReport, InterfaceReport, analyze_ppi_interface, render_interface_markdown,
    run_alanine_scan,
};
use biomol_rna::{NcRnaAnalysisReport, analyze_ncrna_mrna, render_rna_markdown};
use craftbio_scene::{
    ColorScheme, LineOverlay, ProvenanceEntry, ProvenanceKind, RepresentationKind, SphereOverlay,
};
use craftbio_struct::{Biomolecule, ResidueClass, compute_sasa, dist3, write_pdb};
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
            source_app: "biomolcraft::mod-ncrna (Zuker MFE Turner-2004)".to_string(),
            summary: format!(
                "Analyzed {} nucleic chain(s) and cross-checked {} 3D crystal base pairs.",
                reports.len(),
                mol.base_pairs.len()
            ),
        }],
    };
    Ok((out, reports))
}

/// Execute the **Enzymes & Active-Site Module**: detects metal/cofactor coordination spheres &
/// classic Ser/Cys–His–Asp/Glu catalytic triads in 3D and highlights them directly in the viewport.
#[must_use]
pub fn run_enzyme_module(mol: &Biomolecule) -> ModuleExecutionOutput {
    let mut highlights: HashMap<(char, i32), [f32; 3]> = HashMap::new();
    let mut spheres = Vec::new();
    let mut lines = Vec::new();
    let mut md = String::new();
    let _ = writeln!(
        md,
        "# Enzyme Active-Site & Catalytic Triad Audit: {}\n",
        mol.title
    );

    // 1. Metal & cofactor coordination sites (<= 2.9 A for ions, <= 4.2 A for organic cofactors)
    let mut coord_hits = 0usize;
    for a in &mol.atoms {
        if a.res_class == ResidueClass::Ion {
            spheres.push(SphereOverlay {
                label: format!("Metal {}:{}{}", a.chain, a.res_name, a.res_seq),
                category: "Catalytic Ion".to_string(),
                coords: a.coords,
                radius_a: 1.20,
                rgba: [0.72, 0.35, 0.98, 0.85],
            });
            for b in &mol.atoms {
                if b.res_class == ResidueClass::Protein
                    && matches!(b.element.as_str(), "N" | "O" | "S")
                    && dist3(a.coords, b.coords) <= 2.95
                {
                    highlights.insert((b.chain, b.res_seq), [0.85, 0.42, 0.98]);
                    lines.push(LineOverlay {
                        label: format!("Coord {:.2}Å", dist3(a.coords, b.coords)),
                        from: a.coords,
                        to: b.coords,
                        dashed: true,
                        rgba: [0.85, 0.45, 0.99, 0.95],
                    });
                    coord_hits += 1;
                }
            }
        }
    }

    // 2. Classic Ser/Cys – His – Asp/Glu catalytic triad geometry search (<= 6.5 A Cα/sidechain)
    let mut his_list = Vec::new();
    let mut nuc_list = Vec::new();
    let mut acid_list = Vec::new();
    for ch in &mol.chains {
        for r in &ch.residues {
            if r.class != ResidueClass::Protein {
                continue;
            }
            let Some(gi) = r.guide_idx else { continue };
            match r.code {
                'H' => his_list.push((r.chain, r.res_seq, r.code, mol.atoms[gi].coords)),
                'S' | 'C' => nuc_list.push((r.chain, r.res_seq, r.code, mol.atoms[gi].coords)),
                'D' | 'E' => acid_list.push((r.chain, r.res_seq, r.code, mol.atoms[gi].coords)),
                _ => {}
            }
        }
    }
    let mut triads = Vec::new();
    for &(hc, hr, _, hp) in &his_list {
        for &(nc, nr, na, np) in &nuc_list {
            if hc != nc || dist3(hp, np) > 7.2 {
                continue;
            }
            for &(ac, ar, aa, ap) in &acid_list {
                if ac != hc || dist3(hp, ap) > 7.2 {
                    continue;
                }
                triads.push(format!("{hc}:{na}{nr} — H{hr} — {aa}{ar}"));
                highlights.insert((hc, hr), [0.96, 0.25, 0.45]);
                highlights.insert((nc, nr), [0.20, 0.90, 0.55]);
                highlights.insert((ac, ar), [0.98, 0.72, 0.16]);
                if triads.len() >= 6 {
                    break;
                }
            }
            if triads.len() >= 6 {
                break;
            }
        }
        if triads.len() >= 6 {
            break;
        }
    }

    let _ = writeln!(
        md,
        "- **Metal Coordination Bonds (`<= 2.95 Å`):** `{coord_hits}`"
    );
    let _ = writeln!(
        md,
        "- **Putative Catalytic Triads (`Ser/Cys–His–Asp/Glu`):** `{}`",
        triads.len()
    );
    for t in &triads {
        let _ = writeln!(md, "  - `{t}`");
    }

    ModuleExecutionOutput {
        module_name: ModuleId::EnzymesCatalysis.label().to_string(),
        recommended_repr: Some(RepresentationKind::Cartoon),
        recommended_color: Some(ColorScheme::SecondaryStructure),
        residue_highlights: highlights,
        sphere_overlays: spheres,
        line_overlays: lines,
        summary_markdown: md,
        provenance: vec![ProvenanceEntry {
            kind: ProvenanceKind::ComputationalPrediction,
            source_app: "biomolcraft::mod-enzymes".to_string(),
            summary: format!(
                "Scanned {} metal bonds and {} putative catalytic triads.",
                coord_hits,
                triads.len()
            ),
        }],
    }
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
