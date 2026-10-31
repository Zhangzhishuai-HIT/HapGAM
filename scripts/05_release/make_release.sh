#!/bin/bash
# Build the code and results archives, with SHA-256 sums.
#   release/HapGAM_code_<date>.tar.gz     code, environments, project records, README, LICENSE (the GitHub repository)
#   release/HapGAM_results_<date>.tar.gz  fold-level outputs of every reported run, comparison tables, figures
# Input data (genotypes, phenotypes, data splits, known genes) are not included (see docs/DATA_CARD.md).
# usage: bash scripts/05_release/make_release.sh
set -euo pipefail
cd "$(dirname "$0")/../.."   # repository root
D=$(date +%Y%m%d)
OUT=release; mkdir -p "$OUT"
STAGE=$(mktemp -d); trap 'rm -rf "$STAGE"' EXIT
C="$STAGE/HapGAM"; mkdir -p "$C/docs" "$C/tests"
cp docs/README_release.md "$C/README.md"; cp LICENSE "$C/"
rsync_like() { tar cf - --exclude=__pycache__ --exclude='*.pyc' --exclude=jobs/xiyou/sync_from_xiyou.sh "$@" | tar xf - -C "$C"; }
rsync_like src scripts jobs envs
cp tests/*.py "$C/tests/"
cp docs/PREREGISTRATION.md docs/PREREGISTRATION.sha256 docs/PREREGISTRATION_en.md docs/FREEZE.json \
   docs/FREEZE_DEVIATIONS.md docs/DATA_CARD.md "$C/docs/"
cp -r docs/freeze_deviations docs/raw_data_manifests "$C/docs/"
tar czf "$OUT/HapGAM_code_$D.tar.gz" -C "$STAGE" HapGAM
R="$STAGE/HapGAM_results"; mkdir -p "$R"
find runs \( -name '*.json' -o -name '*.pred.tsv' -o -name '*.blocks.tsv' -o -name '*.tuning.tsv' -o -name '*.parts.npz' \) \
  -print0 | tar cf - --null -T - | tar xf - -C "$R"
cp -r results/tables "$R/tables"; mkdir -p "$R/figures"
cp figures/fig*.png figures/fig*.pdf figures/fig*.tsv "$R/figures/"
cat > "$R/README.md" <<'TXT'
# HapGAM results archive
runs/<scenario>/<model>/<trait>/<fold>.json      metrics, chosen hyper-parameters, package code hash ("code")
runs/<scenario>/<model>/<trait>/<fold>.pred.tsv  test predictions (id, group, adjusted phenotype y, prediction)
HapGAM only: .blocks.tsv (selected blocks), .tuning.tsv (inner-CV grid), .parts.npz (per-block test contributions)
runs/_superseded: GBLUP under an earlier definition of the adjusted phenotypes (not used in the article)
tables/: bootstrap comparisons and summary tables; figures/: figures of the article
Scenario and model keys are explained in README.md of the code archive (e.g. hapgam_e = HapGAM;
lockbox_random / lockbox_japan = random / Japanese test set; pc_random = Mendelian traits).
Code hashes: 440783eb7e4ba983 = code fixed before the test sets were evaluated; 989587bae25cd9a9 = the same code with
principal-component caching for DNNGP (same computation); other hashes occur only in cross-validation within the
training set (model development).
TXT
tar czf "$OUT/HapGAM_results_$D.tar.gz" -C "$STAGE" HapGAM_results
(cd "$OUT" && sha256sum "HapGAM_code_$D.tar.gz" "HapGAM_results_$D.tar.gz" > "SHA256SUMS_$D.txt" && cat "SHA256SUMS_$D.txt" && ls -la)
