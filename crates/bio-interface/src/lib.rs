#![forbid(unsafe_code)]

//! Layer 2 (`bio-interface`): 3D PPI Epitope/Paratope Mapping, Shrake-Rupley `ΔBSA`,
//! Computational Alanine Scanning (`ΔΔG_bind`), Paratope Site-Saturation Mutagenesis,
//! and Opt-In `RFdiffusion` / `ProteinMPNN` / `ESM2` / `DockCraft co-fold` Bridges.

pub mod ai_bridge;
pub mod alascan;
pub mod interface;

pub use ai_bridge::{
    DeNovoRequest, DeNovoStagePlan, ToolAvailability, probe_denovo_toolchain,
    render_denovo_markdown, stage_denovo_binder,
};
pub use alascan::{
    AlaScanReport, AlaScanResidue, MutationProposal, SaturationSiteReport, run_alanine_scan,
    saturation_mutagenesis_site,
};
pub use interface::{
    InterfaceContact, InterfaceReport, InterfaceResidueSummary, analyze_ppi_interface,
    render_interface_markdown,
};

#[cfg(test)]
mod tests {
    use super::*;
    use bio_core::{BioError, parse_pdb};

    fn synthetic_dimer_pdb() -> &'static str {
        "\
TITLE     SYNTHETIC AB-AG SALT BRIDGE & AROMATIC INTERFACE
ATOM      1  N   ASP A   1       0.000   0.000   0.000  1.00 10.00           N
ATOM      2  CA  ASP A   1       1.450   0.000   0.000  1.00 10.00           C
ATOM      3  CB  ASP A   1       2.000   1.400   0.000  1.00 10.00           C
ATOM      4  CG  ASP A   1       3.500   1.500   0.000  1.00 10.00           C
ATOM      5  OD1 ASP A   1       4.100   2.550   0.000  1.00 10.00           O
ATOM      6  OD2 ASP A   1       4.100   0.400   0.000  1.00 10.00           O
ATOM      7  C   ASP A   1       2.000  -0.750   1.200  1.00 10.00           C
ATOM      8  O   ASP A   1       1.300  -1.050   2.150  1.00 10.00           O
ATOM      9  N   LYS B   1       9.800   2.500   0.000  1.00 10.00           N
ATOM     10  CA  LYS B   1       8.600   2.500   0.000  1.00 10.00           C
ATOM     11  CB  LYS B   1       8.000   2.500   1.200  1.00 10.00           C
ATOM     12  CG  LYS B   1       7.400   2.500   0.200  1.00 10.00           C
ATOM     13  CD  LYS B   1       7.000   2.500  -0.200  1.00 10.00           C
ATOM     14  CE  LYS B   1       6.950   2.550   0.000  1.00 10.00           C
ATOM     15  NZ  LYS B   1       6.900   2.550   0.000  1.00 10.00           N
END
"
    }

    #[test]
    fn synthetic_ppi_detects_salt_bridge_bsa_and_ala_hotspot() -> Result<(), BioError> {
        let st = parse_pdb(synthetic_dimer_pdb())?;
        let rep = analyze_ppi_interface(&st, &['B'], &['A'])?;
        assert!(
            rep.total_bsa_a2 > 30.0,
            "expected positive ΔBSA, got {}",
            rep.total_bsa_a2
        );
        assert!(rep.salt_bridge_count >= 1, "expected Asp-Lys salt bridge");

        let ala = run_alanine_scan(&st, &['B'], &['A'])?;
        assert!(!ala.paratope_hotspots.is_empty());
        assert!(
            ala.paratope_hotspots[0].ddg_kcal_mol >= 1.5,
            "salt-bridge Asp1 should be a hotspot, got {}",
            ala.paratope_hotspots[0].ddg_kcal_mol
        );

        let sat = saturation_mutagenesis_site(&st, &['B'], 'A', 1)?;
        assert_eq!(sat.ranked_mutations.len(), 19);
        assert!(sat.ranked_mutations[0].estimated_ddg_kcal_mol < 0.0);
        Ok(())
    }

    #[test]
    fn denovo_staging_builds_valid_rfdiffusion_contig() {
        let plan = stage_denovo_binder(&DeNovoRequest {
            campaign_name: "her2_minibinder".to_string(),
            target_pdb_path: "1N8Z.pdb".to_string(),
            target_chain: 'C',
            target_residue_range: "550-610".to_string(),
            hotspot_residues: vec!["C557".to_string(), "C570".to_string(), "C597".to_string()],
            binder_length_min: 65,
            binder_length_max: 85,
            num_designs: 50,
            mpnn_seqs_per_backbone: 8,
            cofold_backend: "boltz2".to_string(),
        });
        assert_eq!(plan.rfdiffusion_contig, "[C550-610/0 65-85]");
        assert!(
            plan.rfdiffusion_command
                .contains("ppi.hotspot_res=[C557,C570,C597]")
        );
        assert!(
            plan.dockcraft_cofold_command
                .contains("dockcraft-cli co-fold")
        );
    }
}
