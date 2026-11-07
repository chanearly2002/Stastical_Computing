#!/usr/bin/env bash
# Download the paper's IHDP (1000 realizations) and Jobs (10 splits) data from the CFR author's site.
# Each dataset goes into its own directory; files are treated as untrusted data (load with allow_pickle=False).
set -euo pipefail
BASE="https://www.fredjo.com/files"
ROOT="$(cd "$(dirname "$0")/.." && pwd)/data"
mkdir -p "$ROOT/ihdp" "$ROOT/jobs"
dl() { # url dest
  [ -s "$2" ] || curl -sS -f -m 600 -o "$2" "$1"
}
dl "$BASE/ihdp_npci_1-1000.train.npz.zip" "$ROOT/ihdp/ihdp_npci_1-1000.train.npz.zip"
dl "$BASE/ihdp_npci_1-1000.test.npz.zip"  "$ROOT/ihdp/ihdp_npci_1-1000.test.npz.zip"
(cd "$ROOT/ihdp" && for z in *.zip; do [ -s "${z%.zip}" ] || unzip -q -o "$z"; done)
dl "$BASE/jobs_DW_bin.new.10.train.npz" "$ROOT/jobs/jobs_DW_bin.new.10.train.npz"
dl "$BASE/jobs_DW_bin.new.10.test.npz"  "$ROOT/jobs/jobs_DW_bin.new.10.test.npz"
ls -la "$ROOT/ihdp" "$ROOT/jobs"
