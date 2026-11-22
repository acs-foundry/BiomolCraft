# AGENTS.md — BiomolCraft Rust House Rules (Aligned with storytold/craft-contrib)

1. **Never crash.** Production code enforces `#![forbid(unsafe_code)]` and denies `clippy::unwrap_used`, `clippy::expect_used`, `clippy::panic`, `clippy::unimplemented`, `clippy::todo`, `clippy::unreachable`.
2. **Clean room & Rust-only native UI.** No Electron, Tauri, or handwritten JS wrappers in the native desktop app; built on `craftbio-libs` (`wgpu 24` + `egui`) and `storytold/craft-libs` (`craft-ipc`, `craft-mcp-server`, `craft-dialogs`).
3. **Modular domain separation.** Biomolecular specialties (Antibodies/VHH, ncRNA/mRNA, Enzymes, ADCs, PK/TMDD, De Novo) plug into `BiomolCraft` as modular engines reusing `BioCraft` compute crates rather than duplicating domain logic.
4. **Scientific honesty.** All computational predictions (rotamer mutagenesis, local MM minimization, Zuker MFE 2D folding, Ala-Scan ΔΔG) are logged with epistemic provenance distinct from experimental coordinates.
