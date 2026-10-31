#!/bin/bash
# Build the release archives in short steps (each well under a minute), as an alternative to make_release.sh when
# no compute node is available.  Same contents as make_release.sh; the staging directory persists between steps.
#   bash scripts/05_release/make_release_staged.sh code       # release/HapGAM_code_<date>.tar.gz
#   bash scripts/05_release/make_release_staged.sh stage      # copy fold outputs, tables and figures to the stage
#   bash scripts/05_release/make_release_staged.sh compress   # release/HapGAM_results_<date>.tar.gz
#   bash scripts/05_release/make_release_staged.sh sums       # release/SHA256SUMS_<date>.txt
set -euo pipefail
cd "$(dirname "$0")/../.."   # repository root
D=${RELEASE_DATE:-$(date +%Y%m%d)}
OUT=release; mkdir -p "$OUT"
STAGE=${RELEASE_STAGE:-/tmp/hapgam_release_stage_$USER}
case "${1:-}" in
code)
  rm -rf "$STAGE/HapGAM"; C="$STAGE/HapGAM"; mkdir -p "$C/docs" "$C/tests"
  cp docs/README_release.md "$C/README.md"; cp LICENSE "$C/"
  tar cf - --exclude=__pycache__ --exclude='*.pyc' --exclude=jobs/xiyou/sync_from_xiyou.sh \
      src scripts jobs envs | tar xf - -C "$C"
  cp tests/*.py "$C/tests/"
  cp docs/PREREGISTRATION.md docs/PREREGISTRATION.sha256 docs/PREREGISTRATION_en.md docs/FREEZE.json \
     docs/FREEZE_DEVIATIONS.md docs/DATA_CARD.md "$C/docs/"
  cp -r docs/freeze_deviations docs/raw_data_manifests "$C/docs/"
  tar czf "$OUT/HapGAM_code_$D.tar.gz" -C "$STAGE" HapGAM
  ls -la "$OUT/HapGAM_code_$D.tar.gz" ;;
stage)
  R="$STAGE/HapGAM_results"; rm -rf "$R"; mkdir -p "$R/figures"
  find runs \( -name '*.json' -o -name '*.pred.tsv' -o -name '*.blocks.tsv' -o -name '*.tuning.tsv' -o -name '*.parts.npz' \) \
    -print0 | tar cf - --null -T - | tar xf - -C "$R"
  cp -r results/tables "$R/tables"
  cp figures/fig*.png figures/fig*.pdf figures/fig*.tsv "$R/figures/"
  sed -n '/^cat > "\$R\/README.md" <<.TXT.$/,/^TXT$/p' scripts/05_release/make_release.sh | sed '1d;$d' > "$R/README.md"
  du -sh "$R" ;;
compress)
  tar czf "$OUT/HapGAM_results_$D.tar.gz" -C "$STAGE" HapGAM_results
  ls -la "$OUT/HapGAM_results_$D.tar.gz" ;;
sums)
  (cd "$OUT" && sha256sum "HapGAM_code_$D.tar.gz" "HapGAM_results_$D.tar.gz" > "SHA256SUMS_$D.txt" && cat "SHA256SUMS_$D.txt") ;;
*)
  echo "usage: $0 code|stage|compress|sums" >&2; exit 2 ;;
esac
