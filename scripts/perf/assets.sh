#!/bin/bash
# Runtime assets for the perf worktrees, without copying and without writing into the main checkout.
#
#   assets.sh prepare  <worktree>            junction pretrained_models + exiftool_files, hard-link exiftool.exe
#   assets.sh manifest [out.json]            {name,size,sha256} of pretrained_models/* and exiftool.exe
#   assets.sh check    [manifest.json]       recompute; exit 1 on any difference, 2 if the manifest is missing
#   assets.sh teardown <worktree>            rmdir the junctions, del the hard link (never rm -r), then check
#   assets.sh photos-manifest <dir> [out]    {name,size,mtime,sha256} of the scratch photo set (SC10)
#   assets.sh photos-check    <dir> [manifest]
#
# env: PERF_ASSET_SRC   checkout that owns the assets   (default /d/photo-llm)
#      PERF_SCRATCH     scratch root                    (default /d/facet-scratch/perf2)
#
# Teardown uses `rmdir` on a junction because `rm -r` / `git worktree remove` through a junction can
# recurse into the TARGET and delete the real model files (spec R16). Never `git clean` in a worktree.

SRC="${PERF_ASSET_SRC:-/d/photo-llm}"
SCRATCH="${PERF_SCRATCH:-/d/facet-scratch/perf2}"
MANIFEST_DEFAULT="$SCRATCH/assets.manifest.json"
PHOTOS_MANIFEST_DEFAULT="$SCRATCH/photos.manifest.json"

win() { cygpath -w "$1"; }

# one JSON object per line: name, size, sha256 (paths relative to $SRC, sorted)
emit_assets() {
    local f
    ( cd "$SRC" || exit 9
      for f in pretrained_models/* exiftool.exe; do
          [ -f "$f" ] || continue
          printf '%s\t%s\t%s\n' "$f" "$(stat -c %s "$f")" "$(sha256sum "$f" | cut -d' ' -f1)"
      done | LC_ALL=C sort )
}

tsv_to_json() {   # stdin: tab separated columns -> JSON array; $@ = column names
    local cols=("$@") first=1 line i
    printf '[\n'
    while IFS=$'\t' read -r -a fields; do
        [ "$first" = 1 ] || printf ',\n'
        first=0
        printf '  {'
        for i in "${!cols[@]}"; do
            [ "$i" = 0 ] || printf ', '
            case "${cols[$i]}" in
                size|mtime) printf '"%s": %s' "${cols[$i]}" "${fields[$i]}" ;;
                *) printf '"%s": "%s"' "${cols[$i]}" "${fields[$i]}" ;;
            esac
        done
        printf '}'
    done
    printf '\n]\n'
}

cmd="${1:-}"; shift
case "$cmd" in
prepare)
    wt="${1:?usage: assets.sh prepare <worktree>}"
    [ -d "$wt/.git" ] || [ -f "$wt/.git" ] || { echo "not a git worktree: $wt" >&2; exit 2; }
    for d in pretrained_models exiftool_files; do
        if [ -e "$wt/$d" ]; then echo "present: $wt/$d"; continue; fi
        cmd //c mklink //J "$(win "$wt/$d")" "$(win "$SRC/$d")" > /dev/null || { echo "mklink /J failed: $d" >&2; exit 1; }
        echo "junction: $wt/$d -> $SRC/$d"
    done
    if [ -e "$wt/exiftool.exe" ]; then echo "present: $wt/exiftool.exe"; else
        cmd //c mklink //H "$(win "$wt/exiftool.exe")" "$(win "$SRC/exiftool.exe")" > /dev/null || { echo "mklink /H failed" >&2; exit 1; }
        echo "hardlink: $wt/exiftool.exe -> $SRC/exiftool.exe"
    fi
    ;;
manifest)
    out="${1:-$MANIFEST_DEFAULT}"
    mkdir -p "$(dirname "$out")"
    emit_assets | tsv_to_json name size sha256 > "$out"
    echo "wrote $out ($(grep -c '"name"' "$out") entries)"
    ;;
check)
    man="${1:-$MANIFEST_DEFAULT}"
    [ -f "$man" ] || { echo "manifest missing: $man" >&2; exit 2; }
    tmp="$(mktemp "${TMPDIR:-/tmp}/assets.XXXXXX")"
    emit_assets | tsv_to_json name size sha256 > "$tmp"
    if diff -u "$man" "$tmp"; then echo "assets check: OK ($(grep -c '"name"' "$man") entries identical)"; rm -f "$tmp"; exit 0; fi
    rm -f "$tmp"; echo "assets check: DIFFERENT" >&2; exit 1
    ;;
teardown)
    wt="${1:?usage: assets.sh teardown <worktree>}"
    for d in pretrained_models exiftool_files; do
        # a junction is removed by rmdir; refuse to touch a real directory
        if [ -e "$wt/$d" ]; then
            if cmd //c dir /AL "$(win "$wt")" 2> /dev/null | grep -q "<JUNCTION> *$d"; then
                cmd //c rmdir "$(win "$wt/$d")" && echo "removed junction $wt/$d"
            else
                echo "REFUSING: $wt/$d is a real directory, not a junction" >&2; exit 1
            fi
        fi
    done
    [ -e "$wt/exiftool.exe" ] && cmd //c del "$(win "$wt/exiftool.exe")" && echo "removed hardlink $wt/exiftool.exe"
    "$0" check
    ;;
photos-manifest)
    dir="${1:?usage: assets.sh photos-manifest <dir> [out]}"; out="${2:-$PHOTOS_MANIFEST_DEFAULT}"
    mkdir -p "$(dirname "$out")"
    ( cd "$dir" || exit 9
      for f in *; do
          [ -f "$f" ] || continue
          printf '%s\t%s\t%s\t%s\n' "$f" "$(stat -c %s "$f")" "$(stat -c %Y "$f")" "$(sha256sum "$f" | cut -d' ' -f1)"
      done | LC_ALL=C sort ) | tsv_to_json name size mtime sha256 > "$out"
    echo "wrote $out ($(grep -c '"name"' "$out") photos)"
    ;;
photos-check)
    dir="${1:?usage: assets.sh photos-check <dir> [manifest]}"; man="${2:-$PHOTOS_MANIFEST_DEFAULT}"
    [ -f "$man" ] || { echo "manifest missing: $man" >&2; exit 2; }
    tmp="$(mktemp "${TMPDIR:-/tmp}/photos.XXXXXX")"
    "$0" photos-manifest "$dir" "$tmp" > /dev/null
    if diff -u "$man" "$tmp" > /dev/null; then echo "photos check: OK ($(grep -c '"name"' "$man") photos identical)"; rm -f "$tmp"; exit 0; fi
    diff -u "$man" "$tmp" | head -20; rm -f "$tmp"; echo "photos check: DIFFERENT" >&2; exit 1
    ;;
*)
    sed -n '2,13p' "$0"; exit 2
    ;;
esac
