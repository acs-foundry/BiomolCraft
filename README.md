# BiomolCraft v0.2.1

**Pure-Rust Biomolecular 3D Visualization, Interactive Modeling, Hybrid GPU Physics & Modular Design Studio** — the self-contained flagship biomolecular hub of the **CraftBio** suite (Schrödinger Maestro / BioLuminate / OpenEye / UCSF ChimeraX / PyMOL replacement), aligned with **`storytold/craft-libs`** (`Craft Link v1` IPC + `craft-mcp-server`) and **`acs-foundry/craftbio-libs`**.

## Architecture & Built-In Loadable Modules (13-Crate Self-Contained Workspace)
- **Core Biomolecular 3D Engine (`craftbio-viewport`, `wgpu 24 + WGSL`):** GPU ray-cast sphere & capsule impostors, DSSP protein ribbons, **ncRNA/mRNA/DNA backbone ribbons + 3D base-pair rungs**, molecular surfaces colored by **3D ESP** / **3D SAP** / **pLDDT**, ChimeraX-style silhouettes, click-to-pick ray casting, interactive trajectory playback timeline (`.dcd` + multi-MODEL PDB), and deterministic headless **4K PNG** export.
- **Interactive 3D Modeling (`biomol-model`):** PyMOL-style Dunbrack rotamer mutagenesis with live steric clash discs & automatic repacking, ChimeraX-style Smith-Waterman + Kabsch iterative MatchMaker alignment, ISOLDE-lite local MM energy minimization, and transactional Undo/Redo.
- **Hybrid 2-Tier Biomolecular Physics (`biomol-physics`):**
  - **Tier 1 (Pure Rust, `0` external deps):** Anisotropic Network Model (**ANM**) normal modes + hinge detection + 3D porcupine arrows + animation frames, Still **GB/SA** solvation + 3D dipole moment + interface `ΔG_bind`, and local Langevin MD.
  - **Tier 2 (Opt-In GPU OpenMM CUDA + `GFN2-xTB`):** Live OpenMM CUDA MD for proteins, antibodies/VHH, enzymes, and RNA/ncRNA exporting binary `.dcd` trajectories loaded 1:1 back into `BiomolCraft`.
- **Self-Contained Biomolecular Design Modules (`biomol-modules`, `bio-*`, `biomol-rna`, `biomol-enzyme`):**
  - **Antibodies & VHH (`bio-core` / `bio-develop` / `bio-interface`):** IMGT/Kabat CDR1–3, VHH hallmarks, germline humanization, TAP 5-metric developability, 3D SAP, HLA-DRB1 immunogenicity, PPI ΔBSA, and computational Ala-Scan hotspots painted directly in 3D.
  - **ncRNA, mRNA, siRNA & Riboswitches (`biomol-rna`):** Deterministic Zuker-style MFE 2D RNA folding over an in-house stacking model (not the Turner-2004 parameter set; the kcal/mol values are an internal scale and are not comparable with RNAfold), G-quadruplex & TLR7/8/`m1Ψ` audit, LNP molar calculator, 2D SVG base-pair arc diagrams, and 2D↔3D crystal concordance.
  - **Enzymes & Protein Engineering (`biomol-enzyme`):** Catalytic triads/dyads, metal/cofactor coordination spheres, ray-marched **3D translucent substrate access tunnels**, thermostabilizing disulfide bridge design, and `ΔΔG_fold`.
  - **ADCs & Biologics PK/TMDD (`bio-adc` / `bio-engine`):** 3D solvent-accessible surface conjugation site mapping + live `AdmetCraft` payload audit + 84-day Minimal-PBPK/FcRn/TMDD simulation.
- **Craft Link v1 Inter-App IPC (`storytold/craft-libs::craft-ipc`) & MCP (`craft-mcp-server`):** Live authenticated local Unix-socket scene & trajectory sync with `DockCraft`, `ChemCraft`, `AdmetCraft`, and ArtCraft apps + **11 hardened MCP tools** (including live PNG viewport rendering!).

## Quick Start
```bash
cargo build --release --workspace
./target/release/biomolcraft-cli inspect --pdb 1Y26.pdb
./target/release/biomolcraft-cli anm --pdb 1N8Z.pdb --mode 1 --out-png 1n8z_anm.png
./target/release/biomolcraft-cli enzyme --pdb 1KE6.pdb --out-png 1ke6_tunnels.png
./target/release/biomolcraft-cli md --pdb 5JDS.pdb --engine openmm --platform CUDA
./target/release/biomolcraft
```

## License
Dual-licensed under `MIT` or `Apache-2.0`.
