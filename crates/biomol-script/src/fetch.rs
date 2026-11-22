#![forbid(unsafe_code)]

//! Zero-Dependency Allow-Listed Online Structure Fetcher (`RCSB PDB/mmCIF`, `AlphaFoldDB`,
//! `EMDB`, `PubChem 3D`) with Strict ID Validation and Persistent `$CRAFT_DATA_DIR/cache/`.

use std::path::PathBuf;
use std::process::Command;

/// Resolved fetch target returned by `fetch_structure_spec`.
#[derive(Debug, Clone)]
pub struct FetchedAsset {
    pub provider: String,
    pub canonical_id: String,
    pub local_path: PathBuf,
    pub cached_hit: bool,
    pub is_volume_map: bool,
}

fn cache_root() -> PathBuf {
    if let Ok(d) = std::env::var("CRAFT_DATA_DIR")
        && !d.trim().is_empty()
    {
        return PathBuf::from(d).join("cache");
    }
    if let Ok(h) = std::env::var("HOME")
        && !h.trim().is_empty()
    {
        return PathBuf::from(h)
            .join("Datos")
            .join("CraftBio")
            .join("cache");
    }
    std::env::temp_dir().join("craftbio_cache")
}

fn validate_id(raw: &str, max_len: usize) -> Result<String, String> {
    let clean = raw.trim();
    if clean.is_empty() || clean.len() > max_len {
        return Err(format!(
            "invalid fetch identifier '{raw}' (must be 1..={max_len} ASCII alphanumeric chars)"
        ));
    }
    if !clean
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Err(format!(
            "rejected fetch identifier '{raw}' (only ASCII alphanumeric, '_' and '-' allowed)"
        ));
    }
    Ok(clean.to_string())
}

fn curl_download(url: &str, dest: &PathBuf) -> Result<(), String> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("cache mkdir failed: {e}"))?;
    }
    let tmp = dest.with_extension("download.tmp");
    let out = Command::new("curl")
        .args([
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--proto",
            "=https",
            "--max-time",
            "30",
            "--max-filesize",
            "67108864",
            "-o",
        ])
        .arg(&tmp)
        .arg(url)
        .output()
        .map_err(|e| format!("failed to launch curl: {e}"))?;
    if !out.status.success() {
        let _ = std::fs::remove_file(&tmp);
        let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
        return Err(format!("HTTPS fetch failed for {url}: {stderr}"));
    }
    std::fs::rename(&tmp, dest).map_err(|e| format!("cache finalize failed: {e}"))?;
    Ok(())
}

/// Fetch a structure or map from an allow-listed public scientific repository with local caching:
/// - `1N8Z` or `pdb:1N8Z` → RCSB PDB (`files.rcsb.org`, mmCIF with PDB fallback)
/// - `af:P04626` → AlphaFoldDB (`alphafold.ebi.ac.uk`, PDB v4)
/// - `emdb:1234` → EMDB (`ftp.ebi.ac.uk`, `.map` / `.mrc`)
/// - `pubchem:2244` → PubChem 3D (`pubchem.ncbi.nlm.nih.gov`, 3D SDF)
pub fn fetch_structure_spec(spec: &str) -> Result<FetchedAsset, String> {
    let trimmed = spec.trim();
    if trimmed.is_empty() {
        return Err(
            "usage: fetch <PDB_ID | pdb:ID | af:UNIPROT | emdb:ID | pubchem:CID>".to_string(),
        );
    }
    let (scheme, raw_id) = if let Some((s, id)) = trimmed.split_once(':') {
        (s.trim().to_ascii_lowercase(), id.trim())
    } else {
        ("pdb".to_string(), trimmed)
    };
    let root = cache_root();

    match scheme.as_str() {
        "pdb" | "rcsb" => {
            let id = validate_id(raw_id, 12)?.to_ascii_uppercase();
            let cif_path = root.join("rcsb").join(format!("{id}.cif"));
            if cif_path.is_file() {
                return Ok(FetchedAsset {
                    provider: "RCSB PDB".to_string(),
                    canonical_id: id,
                    local_path: cif_path,
                    cached_hit: true,
                    is_volume_map: false,
                });
            }
            let pdb_path = root.join("rcsb").join(format!("{id}.pdb"));
            if pdb_path.is_file() {
                return Ok(FetchedAsset {
                    provider: "RCSB PDB".to_string(),
                    canonical_id: id,
                    local_path: pdb_path,
                    cached_hit: true,
                    is_volume_map: false,
                });
            }
            let cif_url = format!("https://files.rcsb.org/download/{id}.cif");
            if curl_download(&cif_url, &cif_path).is_ok() {
                return Ok(FetchedAsset {
                    provider: "RCSB PDB".to_string(),
                    canonical_id: id,
                    local_path: cif_path,
                    cached_hit: false,
                    is_volume_map: false,
                });
            }
            let pdb_url = format!("https://files.rcsb.org/download/{id}.pdb");
            curl_download(&pdb_url, &pdb_path)?;
            Ok(FetchedAsset {
                provider: "RCSB PDB".to_string(),
                canonical_id: id,
                local_path: pdb_path,
                cached_hit: false,
                is_volume_map: false,
            })
        }
        "af" | "alphafold" => {
            let id = validate_id(raw_id, 20)?.to_ascii_uppercase();
            let dest = root
                .join("alphafold")
                .join(format!("AF-{id}-F1-model_v4.pdb"));
            if dest.is_file() {
                return Ok(FetchedAsset {
                    provider: "AlphaFoldDB".to_string(),
                    canonical_id: id,
                    local_path: dest,
                    cached_hit: true,
                    is_volume_map: false,
                });
            }
            let url = format!("https://alphafold.ebi.ac.uk/files/AF-{id}-F1-model_v4.pdb");
            curl_download(&url, &dest)?;
            Ok(FetchedAsset {
                provider: "AlphaFoldDB".to_string(),
                canonical_id: id,
                local_path: dest,
                cached_hit: false,
                is_volume_map: false,
            })
        }
        "pubchem" | "cid" => {
            let id = validate_id(raw_id, 16)?;
            if !id.chars().all(|c| c.is_ascii_digit()) {
                return Err(format!("PubChem CID must be numeric (got '{id}')"));
            }
            let dest = root.join("pubchem").join(format!("CID_{id}_3d.sdf"));
            if dest.is_file() {
                return Ok(FetchedAsset {
                    provider: "PubChem 3D".to_string(),
                    canonical_id: format!("CID:{id}"),
                    local_path: dest,
                    cached_hit: true,
                    is_volume_map: false,
                });
            }
            let url = format!(
                "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{id}/SDF?record_type=3d"
            );
            curl_download(&url, &dest)?;
            Ok(FetchedAsset {
                provider: "PubChem 3D".to_string(),
                canonical_id: format!("CID:{id}"),
                local_path: dest,
                cached_hit: false,
                is_volume_map: false,
            })
        }
        "emdb" | "emd" => {
            let id = validate_id(raw_id, 12)?;
            let digits: String = id.chars().filter(|c| c.is_ascii_digit()).collect();
            if digits.is_empty() {
                return Err(format!("invalid EMDB ID '{raw_id}'"));
            }
            let dest = root.join("emdb").join(format!("emd_{digits}.map"));
            if dest.is_file() {
                return Ok(FetchedAsset {
                    provider: "EMDB".to_string(),
                    canonical_id: format!("EMD-{digits}"),
                    local_path: dest,
                    cached_hit: true,
                    is_volume_map: true,
                });
            }
            let gz_dest = root.join("emdb").join(format!("emd_{digits}.map.gz"));
            let url = format!(
                "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-{digits}/map/emd_{digits}.map.gz"
            );
            curl_download(&url, &gz_dest)?;
            // Decompress gzip using system gzip -dc (zero extra crate needed)
            let out = Command::new("gzip")
                .args(["-dc"])
                .arg(&gz_dest)
                .output()
                .map_err(|e| format!("failed to gunzip EMDB map: {e}"))?;
            if !out.status.success() || out.stdout.len() < 1024 {
                return Err("failed to decompress downloaded EMDB .map.gz".to_string());
            }
            std::fs::write(&dest, &out.stdout)
                .map_err(|e| format!("failed to write cached EMDB map: {e}"))?;
            let _ = std::fs::remove_file(&gz_dest);
            Ok(FetchedAsset {
                provider: "EMDB".to_string(),
                canonical_id: format!("EMD-{digits}"),
                local_path: dest,
                cached_hit: false,
                is_volume_map: true,
            })
        }
        other => Err(format!(
            "unsupported fetch scheme '{other}' (use pdb:, af:, pubchem:, or emdb:)"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_shell_or_url_injection_identifiers() {
        assert!(fetch_structure_spec("pdb:1n8z;rm -rf /").is_err());
        assert!(fetch_structure_spec("pdb:../../etc/passwd").is_err());
        assert!(fetch_structure_spec("pubchem:not_a_number").is_err());
        assert!(fetch_structure_spec("http:evil.com").is_err());
    }
}
