# BiomolCraft v0.1.0

**Pure-Rust Biomolecular 3D Visualization, Interactive Modeling, Physics & Modular Design Studio** — the flagship biomolecular hub of the **CraftBio** suite (Schrödinger Maestro / BioLuminate / OpenEye / UCSF ChimeraX / PyMOL replacement), aligned with **`storytold/craft-libs`** (`Craft Link v1` IPC + `craft-mcp-server`) and **`acs-foundry/craftbio-libs`**.

## Architecture & Built-In Loadable Modules
- **Core Biomolecular 3D Engine (`craftbio-viewport`, `wgpu 24 + WGSL`):** GPU ray-cast sphere & capsule impostors, DSSP protein ribbons, **ncRNA/mRNA/DNA backbone ribbons + 3D base-pair rungs**, molecular surfaces colored by **3D ESP** / **3D SAP** / **pLDDT**, ChimeraX-style silhouettes, click-to-pick ray casting, and deterministic headless **4K PNG** export.
- **Interactive 3D Modeling (`biomol-model`):** PyMOL-style Dunbrack rotamer mutagenesis with live steric clash discs & automatic repacking, ChimeraX-style MatchMaker/Kabsch alignment, ISOLDE-lite local MM energy minimization, and transactional Undo/Redo.
- **Pluggable Biomolecular Modules (`biomol-modules`, `biomol-rna`):**
  - **Antibodies & VHH:** Mounts `BioCraft` compute crates (`bio-core`/`bio-develop`/`bio-interface`) to paint IMGT CDR1–3, VHH hallmarks, 3D SAP, and Ala-Scan hotspot spheres directly onto the 3D viewport.
  - **ncRNA, mRNA, siRNA & Riboswitches (`biomol-rna`):** Deterministic Zuker/Turner-2004 MFE 2D RNA folding, G-quadruplex & TLR7/8/`m1Ψ` audit, LNP molar calculator, 2D SVG base-pair arc diagrams, and 2D↔3D crystal base-pair concordance.
  - **Enzymes & Active Sites:** Metal/cofactor coordination spheres & catalytic triad geometry perception in 3D.
  - **ADCs & Biologics PK/TMDD:** 3D solvent-accessible surface conjugation site mapping + live `AdmetCraft` payload audit + 84-day Minimal-PBPK/TMDD.
- **Craft Link v1 Inter-App IPC (`storytold/craft-libs::craft-ipc`) & MCP (`craft-mcp-server`):** Live authenticated local Unix-socket scene sync with `BioCraft`, `DockCraft`, `ChemCraft`, `AdmetCraft`, and ArtCraft apps + 8 MCP tools (including live PNG image rendering!).

## Quick Start
```bash
cargo build --release --workspace
./target/release/biomolcraft-cli inspect --pdb 1Y26.pdb
./target/release/biomolcraft-cli rna-fold --pdb 1Y26.pdb --out-svg 1y26_arc.svg
./target/release/biomolcraft-cli render --pdb 1N8Z.pdb --module antibodies --out-png 1n8z.png
./target/release/biomolcraft
```

## License
Dual-licensed under `MIT` or `Apache-2.0`.
