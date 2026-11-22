#!/usr/bin/env bash
set -euo pipefail
PREFIX="${PREFIX:-$HOME/.local}"
cargo build --release --workspace
install -Dm755 target/release/biomolcraft "$PREFIX/bin/biomolcraft"
install -Dm755 target/release/biomolcraft-cli "$PREFIX/bin/biomolcraft-cli"
install -Dm644 packaging/biomolcraft.desktop "$HOME/.local/share/applications/biomolcraft.desktop"
install -Dm644 packaging/biomolcraft.svg "$HOME/.local/share/icons/hicolor/scalable/apps/biomolcraft.svg"
echo "Installed BiomolCraft to $PREFIX/bin/{biomolcraft,biomolcraft-cli}"
