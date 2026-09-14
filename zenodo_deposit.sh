#!/usr/bin/env bash
#
# zenodo_deposit.sh -- package, upload, and publish the COF-999 data record.
#
#   Zenodo records are FLAT: they have no directory structure. The ~505 bulk
#   files therefore ship as a handful of tar archives whose members carry paths
#   relative to the bundle root, so that
#
#       tar -xf <archive> -C /path/to/CO2-behavior-in-COF-999
#
#   restores the layout the README describes, exactly.
#
# SAFETY
#   * Defaults to the Zenodo SANDBOX. Pass --production to touch the real one.
#   * `publish` is irreversible: it mints a permanent DOI, and files in a
#     published record can never be removed. It requires a typed confirmation
#     and is never run by any other subcommand.
#   * Everything before `publish` is a draft and can be deleted freely.
#
# USAGE
#   ./zenodo_deposit.sh plan                      # show the archive plan
#   ./zenodo_deposit.sh metadata-preview          # print the Zenodo metadata as JSON
#   ./zenodo_deposit.sh package                   # build archives + checksums
#   ./zenodo_deposit.sh create                    # NEW draft + metadata + reserved DOI
#   ./zenodo_deposit.sh adopt <id|DOI>            # attach to a draft made in the web UI
#   ./zenodo_deposit.sh upload                    # push archives to the draft
#   ./zenodo_deposit.sh verify                    # compare local vs remote MD5
#   ./zenodo_deposit.sh status                    # show the draft
#   ./zenodo_deposit.sh doi                       # print the reserved DOI
#   ./zenodo_deposit.sh publish                   # MINT THE DOI (irreversible)
#
#   Add --production to any command to use zenodo.org instead of the sandbox.
#
# AUTH
#   Create a token at  https://zenodo.org/account/settings/applications/tokens/new/
#   (sandbox: https://sandbox.zenodo.org/...) with scopes:
#       deposit:write   deposit:actions
#   Then either:
#       export ZENODO_TOKEN=...            (production)
#       export ZENODO_SANDBOX_TOKEN=...    (sandbox)
#   or put it in ~/.config/zenodo/token-production  /  ~/.config/zenodo/token-sandbox
#   with mode 600. The token is never printed or written to the state file.
#
set -euo pipefail

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
# Resolve relative to this script, so the bundle can be cloned or moved anywhere.
SELF_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="${BUNDLE_ROOT:-$SELF_DIR}"
# Staging defaults OUTSIDE the bundle: the archives total ~14 GB and must never
# land inside the git working tree.
STAGING="${STAGING:-$(dirname -- "$SELF_DIR")/_zenodo_upload}"
GITHUB_URL="https://github.com/liping-ai4cat/CO2-behavior-in-COF-999"
PAPER_DOI="10.26434/chemrxiv.15003518/v3"
RECORD_VERSION="1.0.0"
COMPRESS_THREADS="$(nproc)"
# Zenodo's gateway 502s on very large single PUTs. A 2.85 GB part uploads fine,
# 4.13 GB does not, so any group above this threshold is split on file
# boundaries into balanced parts. Each part is a normal tar and extracts
# independently to the same paths -- there is nothing to reassemble.
SPLIT_BYTES="${SPLIT_BYTES:-3000000000}"

PRODUCTION=0
for a in "$@"; do [[ "$a" == "--production" ]] && PRODUCTION=1; done

if (( PRODUCTION )); then
  API="https://zenodo.org/api"; WEB="https://zenodo.org"
  TOKEN_ENV="${ZENODO_TOKEN:-}"; TOKEN_FILE="$HOME/.config/zenodo/token-production"
  STATE="$STAGING/.deposition-production.json"; TAG="PRODUCTION"
else
  API="https://sandbox.zenodo.org/api"; WEB="https://sandbox.zenodo.org"
  TOKEN_ENV="${ZENODO_SANDBOX_TOKEN:-}"; TOKEN_FILE="$HOME/.config/zenodo/token-sandbox"
  STATE="$STAGING/.deposition-sandbox.json"; TAG="SANDBOX"
fi

say()  { printf '%s\n' "$*" >&2; }
pflag(){ (( PRODUCTION )) && printf ' --production' || true; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
rule() { printf '%s\n' "------------------------------------------------------------" >&2; }

# --------------------------------------------------------------------------
# Archive plan.   name | mode | selector
#   mode: dir=<path>      whole directory
#         glob=<pattern>  every file matching -name <pattern> under the bundle
#         raw=<path>      upload the file as-is, no tar
#   Compression: gz for text-like payloads, none where the bytes are already
#   incompressible (PDF, ASE binary trajectories, torch checkpoints).
# --------------------------------------------------------------------------
PLAN=(
  "metadynamics_stationary_points_dry.tar|none|dir=computational_results/Metadynamics/dry/stationary_points_src"
  "metadynamics_stationary_points_humid.tar|none|dir=computational_results/Metadynamics/humid/stationary_points_src"
  "benchmarking_force_component_errors.tar.gz|gz|glob=all_force_component_errors.csv"
  "benchmarking_force_parity_plots.tar|none|glob=*forces_components.pdf"
  "benchmarking_uma_predictions.tar|none|glob=uma.traj"
  "dft_vasp_input_example.tar.gz|gz|dir=DFT_Dataset/VASP-input"
  "dft_train_trajectory.tar|none|file=DFT_Dataset/train/train.traj"
  "metadynamics_humid_example_strided_traj.tar|none|file=computational_results/Metadynamics/humid/example_job/meta_40_stride.traj"
  "FT-uma-s-v3_inference_ckpt_18000.pt|raw|file=Training/fine-tuned-UMA-s-models/FT-uma-s-v3/inference_ckpt_18000.pt"
)

token() {
  local t="$TOKEN_ENV"
  [[ -z "$t" && -r "$TOKEN_FILE" ]] && t="$(tr -d '[:space:]' < "$TOKEN_FILE")"
  [[ -n "$t" ]] || die "No $TAG token. Set the env var or create $TOKEN_FILE (mode 600). See the header of this script."
  printf '%s' "$t"
}

api() { # api <method> <path-or-url> [curl args...]
  local m="$1" u="$2"; shift 2
  [[ "$u" == http* ]] || u="$API$u"
  # GET is idempotent, so retry it through Zenodo's 502/504 spells. Never retry
  # POST/PUT here -- publish in particular must not fire twice.
  local extra=()
  [[ "$m" == GET ]] && extra=(--retry 5 --retry-delay 10 --retry-all-errors)
  curl --fail-with-body -sS --connect-timeout 30 --max-time 300 "${extra[@]}" \
       -X "$m" -H "Authorization: Bearer $(token)" "$@" "$u"
}

# GET returning *valid JSON*; retries when the gateway hands back an HTML error
# page instead. Returns 1 rather than dying, so callers can decide.
api_json() {
  local u="$1" attempt j
  for attempt in 1 2 3 4 5; do
    if j="$(api GET "$u" 2>/dev/null)" && jq -e . >/dev/null 2>&1 <<<"$j"; then
      printf '%s' "$j"; return 0
    fi
    (( attempt < 5 )) && sleep $(( attempt * 10 ))
  done
  return 1
}

remote_md5() { # <filename> -> bare md5 held by Zenodo, empty if absent
  remote_files | awk -v n="$1" '$1==n {print $2; exit}'
}

upload_one() { # <path> <name> <local-md5> <bucket> ; authoritative: verifies server-side
  local f="$1" n="$2" lmd5="$3" bkt="$4"
  local attempt rc rmd5 backoff
  for attempt in 1 2 3 4 5; do
    rc=0
    curl --fail-with-body -sS --progress-bar -X PUT \
         -H "Authorization: Bearer $(token)" \
         --connect-timeout 30 --retry 3 --retry-delay 10 --retry-all-errors \
         --upload-file "$f" "$bkt/$n" >/dev/null || rc=$?
    # Never trust the PUT body: Zenodo's gateway can return 2xx with an empty or
    # non-JSON body after an internal 502. Ask the deposition what it actually holds.
    rmd5="$(remote_md5 "$n" || true)"
    if [[ "$rmd5" == "$lmd5" ]]; then
      say "  ok                     $n  (md5 confirmed by Zenodo)"
      return 0
    fi
    backoff=$(( attempt * 20 ))
    say "  attempt $attempt/5 failed for $n  (curl rc=$rc, Zenodo holds [${rmd5:-nothing}])"
    if (( attempt < 5 )); then say "  retrying in ${backoff}s ..."; sleep "$backoff"; fi
  done
  return 1
}

remote_files() { # -> "<filename> <bare-md5>" per line; empty when the draft has no files.
  # NB: GET /deposit/depositions/<id>/files returns HTTP 500 on a draft with no
  # files yet, so read .files[] off the deposition object instead. Checksums are
  # bare hex there but "md5:<hex>" from the bucket API, so normalise.
  local j
  j="$(api_json "/deposit/depositions/$(dep_id)")" || return 1
  jq -r '.files[]? | "\(.filename // .key) \((.checksum // "") | sub("^md5:";""))"' <<<"$j"
}

dep_id()  { [[ -f "$STATE" ]] || die "No draft recorded for $TAG. Run: $0 create${1:+ }${1:-}"; jq -r '.id' "$STATE"; }
bucket()  { jq -r '.links.bucket' "$STATE"; }

# --------------------------------------------------------------------------
cmd_package() {
  command -v pigz >/dev/null || die "pigz not found"
  [[ -d "$BUNDLE_ROOT" ]] || die "Bundle not found: $BUNDLE_ROOT"
  mkdir -p "$STAGING"
  say "Packaging from: $BUNDLE_ROOT"
  say "Staging to:     $STAGING"
  rule

  local list; list="$(mktemp)"; trap 'rm -f "$list"' RETURN

  for entry in "${PLAN[@]}"; do
    IFS='|' read -r name mode sel <<<"$entry"
    local kind="${sel%%=*}" path="${sel#*=}" out="$STAGING/$name"

    if [[ "$mode" == "raw" ]]; then
      [[ -f "$BUNDLE_ROOT/$path" ]] || die "missing: $path"
      if [[ -f "$out" ]] && [[ "$out" -nt "$BUNDLE_ROOT/$path" ]]; then
        say "  skip (current)  $name"; continue
      fi
      say "  copy            $name"
      cp -f "$BUNDLE_ROOT/$path" "$out"
      continue
    fi

    : > "$list"
    case "$kind" in
      dir)  ( cd "$BUNDLE_ROOT" && find "$path" -type f -print ) | sort > "$list" ;;
      file) printf '%s\n' "$path" > "$list" ;;
      glob) ( cd "$BUNDLE_ROOT" && find . -type f -name "$path" -printf '%P\n' ) | sort > "$list" ;;
      *)    die "bad selector: $sel" ;;
    esac
    local n; n=$(wc -l < "$list")
    (( n > 0 )) || die "selector matched nothing: $sel"

    # How big would this group be? Split if it would exceed SPLIT_BYTES.
    local bytes nparts
    bytes=$( cd "$BUNDLE_ROOT" && tr '\n' '\0' < "$list" | xargs -0 stat -c%s 2>/dev/null | awk '{t+=$1} END {print t+0}' )
    nparts=1
    if [[ "$mode" != "gz" ]] && (( bytes > SPLIT_BYTES )); then
      nparts=$(( (bytes + SPLIT_BYTES - 1) / SPLIT_BYTES ))
    fi

    if (( nparts == 1 )); then
      if [[ -f "$out" ]]; then say "  skip (exists)   $name  ($n members)"; continue; fi
      say "  build           $name  ($n members, $(numfmt --to=iec --suffix=B "$bytes"))"
      if [[ "$mode" == "gz" ]]; then
        tar -c -C "$BUNDLE_ROOT" --files-from="$list" | pigz -p "$COMPRESS_THREADS" -6 > "$out.part"
      else
        tar -c -C "$BUNDLE_ROOT" --files-from="$list" -f "$out.part"
      fi
      mv "$out.part" "$out"
      continue
    fi

    # Balanced split on file boundaries: fill each part up to bytes/nparts.
    say "  split           $name  ($n members, $(numfmt --to=iec --suffix=B "$bytes")) -> $nparts parts"
    local target=$(( bytes / nparts + 1 )) i=1 acc=0 sub
    sub="$(mktemp)"; : > "$sub"
    local base="${name%.tar}"
    while IFS= read -r rel; do
      local fsz; fsz=$(stat -c%s "$BUNDLE_ROOT/$rel")
      if (( acc > 0 && acc + fsz > target && i < nparts )); then
        _emit_part "$sub" "$STAGING/${base}.part${i}of${nparts}.tar"
        i=$((i+1)); acc=0; : > "$sub"
      fi
      printf '%s\n' "$rel" >> "$sub"; acc=$((acc+fsz))
    done < "$list"
    [[ -s "$sub" ]] && _emit_part "$sub" "$STAGING/${base}.part${i}of${nparts}.tar"
    rm -f "$sub"
  done

  rule
  say "Computing checksums..."
  ( cd "$STAGING" && md5sum    -- *.tar *.tar.gz *.pt 2>/dev/null | sort -k2 > MD5SUMS    || true )
  ( cd "$STAGING" && sha256sum -- *.tar *.tar.gz *.pt 2>/dev/null | sort -k2 > SHA256SUMS || true )
  write_contents_doc
  rule
  ( cd "$STAGING" && ls -la --block-size=M -- *.tar *.tar.gz *.pt ZENODO_CONTENTS.md MD5SUMS SHA256SUMS 2>/dev/null ) >&2
  say ""
  say "Total: $(du -sh "$STAGING" | cut -f1) in $(ls -1 "$STAGING" | wc -l) files."
  say "Confirm this fits your Zenodo quota before uploading (the default per-record"
  say "allowance is limited; request an increase from Zenodo support if needed)."
}

_emit_part() { # <file-list> <output.tar>
  local lst="$1" outp="$2"
  if [[ -f "$outp" ]]; then say "    skip (exists)  $(basename "$outp")"; return 0; fi
  say "    build          $(basename "$outp")  ($(wc -l < "$lst") members)"
  tar -c -C "$BUNDLE_ROOT" --files-from="$lst" -f "$outp.part"
  mv "$outp.part" "$outp"
}

write_contents_doc() {
  { cat <<'MD'
# Contents of this data record

Zenodo stores files without directory structure, so the bulk of this deposit is
packaged as tar archives. Every archive stores its members with paths relative
to the root of the `CO2-behavior-in-COF-999` tree, so extracting an archive at
that root restores the exact layout described in the repository README:

```bash
git clone https://github.com/liping-ai4cat/CO2-behavior-in-COF-999
cd CO2-behavior-in-COF-999
for a in /path/to/downloads/*.tar /path/to/downloads/*.tar.gz; do
    tar -xf "$a" -C .
done
cp /path/to/downloads/FT-uma-s-v3_inference_ckpt_18000.pt \
   Training/fine-tuned-UMA-s-models/FT-uma-s-v3/inference_ckpt_18000.pt
```

The repository and this record use identical paths, so after extraction every
path mentioned in the README resolves.

| file | restores |
|---|---|
MD
    for entry in "${PLAN[@]}"; do
      IFS='|' read -r name mode sel <<<"$entry"
      local base="${name%.tar}"
      shopt -s nullglob
      local parts=( "$STAGING/${base}.part"*of*.tar )
      shopt -u nullglob
      if (( ${#parts[@]} )); then
        for pp in "${parts[@]}"; do
          printf '| `%s` | `%s` (part %s of %s) |\n' "$(basename "$pp")" "${sel#*=}" \
            "$(basename "$pp" | sed -E 's/.*part([0-9]+)of([0-9]+)\.tar/\1/')" \
            "$(basename "$pp" | sed -E 's/.*part([0-9]+)of([0-9]+)\.tar/\2/')"
        done
      else
        printf '| `%s` | `%s` |\n' "$name" "${sel#*=}"
      fi
    done
    cat <<'MD'

`MD5SUMS` and `SHA256SUMS` cover every file in this record.
Verify with `md5sum -c MD5SUMS`.
MD
  } > "$STAGING/ZENODO_CONTENTS.md"
}

# --------------------------------------------------------------------------
metadata_json() {
  jq -n --arg ver "$RECORD_VERSION" --arg gh "$GITHUB_URL" --arg pdoi "$PAPER_DOI" '
  {metadata:{
    upload_type:"dataset",
    title:"Molecular origins of CO2 capture behavior in amine-appended nanoporous frameworks: DFT reference data, a fine-tuned UMA-S potential, and simulation results",
    creators:[
      {name:"Liu, Liping"},
      {name:"Zhou, Zihui"},
      {name:"Daglar, Hilal"},
      {name:"Siepmann, Ilja"},
      {name:"Yaghi, Omar M."},
      {name:"Gagliardi, Laura"}
    ],
    description:"<p>Data record supporting the manuscript <em>Molecular origins of CO2 capture behavior in amine-appended nanoporous frameworks</em>.</p><p>Contains the DFT reference dataset used to fine-tune a UMA-S machine-learning interatomic potential for amine-appended covalent organic frameworks (4,469 training / 497 validation / 1,466 held-out test structures with energies, forces and stresses, as ASE trajectories); the production fine-tuned checkpoint FT-uma-s-v3; the MLIP-versus-DFT benchmark output for five models; and the production simulation results, comprising simulated-annealing structure searches, 100 ps equilibrium molecular dynamics with radial-distribution, hydrogen-bond and amine-accessibility analyses, and well-tempered metadynamics of CO2 chemisorption at 80 dry and 79 humid amine sites including the extracted stationary-point structures.</p><p>Analysis and simulation code, together with all small tabular data, is in the companion GitHub repository. Paths in this record match that repository exactly; see ZENODO_CONTENTS.md for extraction instructions.</p><p>VASP POTCAR pseudopotential files are licence-restricted and are not included. The fine-tuned checkpoint is a derivative of Meta FAIR Chemistry UMA-S and is governed by the FAIR Chemistry License, not by this record CC-BY-4.0 licence.</p>",
    access_right:"open",
    license:"cc-by-4.0",
    version:$ver,
    language:"eng",
    keywords:["carbon capture","covalent organic framework","COF-999","machine-learning interatomic potential","UMA","metadynamics","density functional theory","CO2 chemisorption","direct air capture","molecular dynamics"],
    related_identifiers:[
      {relation:"isSupplementTo", identifier:$pdoi,  scheme:"doi"},
      {relation:"isSupplementTo", identifier:$gh,    scheme:"url"}
    ],
    prereserve_doi:true
  }}'
}

cmd_create() {
  mkdir -p "$STAGING"
  if [[ -f "$STATE" ]]; then
    say "A $TAG draft already exists (id $(jq -r .id "$STATE")). Use 'metadata' to update it,"
    say "or delete $STATE and the draft on Zenodo to start over."
    exit 1
  fi
  say "Creating $TAG draft deposition..."
  api POST "/deposit/depositions" -H "Content-Type: application/json" -d '{}' > "$STATE"
  local id; id=$(jq -r '.id' "$STATE")
  api PUT "/deposit/depositions/$id" -H "Content-Type: application/json" -d "$(metadata_json)" > "$STATE"
  rule
  say "Draft id:     $id"
  say "Edit online:  $WEB/deposit/$id"
  say "Reserved DOI: $(jq -r '.metadata.prereserve_doi.doi // "(none)"' "$STATE")"
  say ""
  say "The reserved DOI is stable: you can put it in the manuscript now, before publishing."
}

cmd_adopt() {
  local raw="${1:-}"
  [[ -n "$raw" ]] || die "usage: $0 adopt <deposition-id | DOI> [--production]"
  local id="${raw##*zenodo.}"; id="${id##*/}"
  [[ "$id" =~ ^[0-9]+$ ]] || die "Could not read a numeric deposition id from '$raw'"
  mkdir -p "$STAGING"
  say "Fetching $TAG deposition $id ..."
  local j; j="$(api GET "/deposit/depositions/$id")"
  local st; st="$(jq -r '.state' <<<"$j")"
  local sub; sub="$(jq -r '.submitted' <<<"$j")"
  if [[ "$sub" == "true" ]]; then
    die "Deposition $id is already published. Files in a published record cannot be replaced; use Zenodo's 'New version' instead."
  fi
  printf '%s' "$j" > "$STATE"
  rule
  say "Adopted draft $id  (state: $st)"
  say "  title:  $(jq -r '.metadata.title // "(unset)"' <<<"$j")"
  say "  DOI:    $(jq -r '.metadata.prereserve_doi.doi // .doi // "(none reserved)"' <<<"$j")"
  say "  files:  $(jq -r '.files | length' <<<"$j") already on the draft"
  say "  bucket: $(jq -r '.links.bucket // "(none)"' <<<"$j")"
  rule
  say "Next:  $0 package$(pflag)   then   $0 metadata$(pflag)   then   $0 upload$(pflag)"
  say "('metadata' overwrites the web-UI metadata with the block in this script -- review"
  say " it first with '$0 metadata-preview', or skip it to keep what you typed in the UI.)"
}

cmd_metadata() {
  local id; id=$(dep_id)
  say "Updating metadata on $TAG draft $id..."
  api PUT "/deposit/depositions/$id" -H "Content-Type: application/json" -d "$(metadata_json)" > "$STATE"
  say "Done. Reserved DOI: $(jq -r '.metadata.prereserve_doi.doi // "(none)"' "$STATE")"
}

# --------------------------------------------------------------------------
cmd_upload() {
  local id bkt; id=$(dep_id); bkt=$(bucket)
  [[ "$bkt" == http* ]] || die "No bucket URL in $STATE; re-run 'create'."
  [[ -d "$STAGING" ]] || die "Nothing packaged. Run: $0 package"

  local remote
  remote="$(remote_files)" || die "Zenodo's API is not responding with valid JSON.
       This is a server-side outage, not a problem with your draft or your files.
       Check https://status.zenodo.org and re-run: $0 upload$(pflag)"

  shopt -s nullglob
  local files=( "$STAGING"/*.tar "$STAGING"/*.tar.gz "$STAGING"/*.pt \
                "$STAGING"/ZENODO_CONTENTS.md "$STAGING"/MD5SUMS "$STAGING"/SHA256SUMS )
  shopt -u nullglob
  (( ${#files[@]} )) || die "No packaged files found in $STAGING"

  say "Uploading ${#files[@]} files to $TAG draft $id"
  rule
  for f in "${files[@]}"; do
    local n; n="$(basename "$f")"
    local lmd5; lmd5="$(md5sum "$f" | cut -d' ' -f1)"
    if grep -qx -- "$n $lmd5" <<<"$remote"; then
      say "  ok (already uploaded)  $n"; continue
    fi
    say "  uploading              $n  ($(du -h "$f" | cut -f1))"
    upload_one "$f" "$n" "$lmd5" "$bkt" \
      || die "Giving up on $n after 5 attempts. Zenodo's gateway is returning errors.
       Wait a few minutes and re-run: $0 upload$(pflag)
       Completed files are skipped, so this resumes where it stopped.
       If this one file keeps failing, rebuild it smaller:
         rm $STAGING/$n && SPLIT_BYTES=1500000000 $0 package"
  done
  rule
  say "Upload finished. Now run: $0 verify$(pflag)"
}

cmd_verify() {
  local id; id=$(dep_id)
  local remote
  remote="$(remote_files)" || die "Zenodo's API is not responding; retry verify shortly."
  local bad=0 n_ok=0
  shopt -s nullglob
  for f in "$STAGING"/*.tar "$STAGING"/*.tar.gz "$STAGING"/*.pt \
           "$STAGING"/ZENODO_CONTENTS.md "$STAGING"/MD5SUMS "$STAGING"/SHA256SUMS; do
    local n lmd5; n="$(basename "$f")"; lmd5="$(md5sum "$f" | cut -d' ' -f1)"
    if grep -qx -- "$n $lmd5" <<<"$remote"; then
      printf '  MATCH     %s\n' "$n" >&2; n_ok=$((n_ok+1))
    else
      printf '  MISMATCH  %s (local md5:%s)\n' "$n" "$lmd5" >&2; bad=$((bad+1))
    fi
  done
  shopt -u nullglob
  rule
  local n_remote; n_remote=$(grep -c . <<<"$remote" || true)
  say "$n_ok verified, $bad mismatched; $n_remote file(s) present on Zenodo."
  (( bad == 0 )) || die "Verification failed -- do not publish."
  say "All files verified."
}

cmd_status() {
  local id; id=$(dep_id)
  api GET "/deposit/depositions/$id" | jq '{
    id, state, submitted,
    title: .metadata.title,
    doi: (.doi // .metadata.prereserve_doi.doi),
    version: .metadata.version,
    files: [.files[]? | {filename, filesize, checksum}],
    n_files: (.files | length // 0),
    total_bytes: ([.files[]?.filesize] | add // 0),
    web: .links.html
  }'
}

cmd_doi() { dep_id >/dev/null; jq -r '.doi // .metadata.prereserve_doi.doi // "none reserved"' "$STATE"; }

# --------------------------------------------------------------------------
cmd_publish() {
  local id; id=$(dep_id)
  local info; info="$(api_json "/deposit/depositions/$id")" || die "Cannot read the draft; Zenodo may be down."
  local nf doi ttl
  nf=$(jq '.files | length' <<<"$info")
  doi=$(jq -r '.metadata.prereserve_doi.doi // "(unreserved)"' <<<"$info")
  ttl=$(jq -r '.metadata.title' <<<"$info")

  rule
  say "About to PUBLISH on $TAG"
  say "  id:    $id"
  say "  title: $ttl"
  say "  files: $nf"
  say "  DOI:   $doi"
  rule
  say "This is IRREVERSIBLE. Publishing mints a permanent DOI and makes the record"
  say "public; files in a published Zenodo record can never be deleted, only a new"
  say "version added. Confirm the file list and metadata first:  $0 status$(pflag)"
  rule
  if (( PRODUCTION )); then
    say "Type exactly:  PUBLISH TO ZENODO"
  else
    say "Type exactly:  PUBLISH TO SANDBOX"
  fi
  local reply; read -r -p "> " reply
  local want; want=$( (( PRODUCTION )) && echo "PUBLISH TO ZENODO" || echo "PUBLISH TO SANDBOX" )
  [[ "$reply" == "$want" ]] || die "Not confirmed; nothing published."

  api POST "/deposit/depositions/$id/actions/publish" > "$STATE"
  rule
  say "PUBLISHED."
  say "  DOI:    $(jq -r '.doi' "$STATE")"
  say "  Record: $(jq -r '.links.record_html // .links.html' "$STATE")"
  say ""
  say "Now replace 10.5281/zenodo.XXXXXXX in README.md (2 places) with the DOI above."
}

# --------------------------------------------------------------------------
sub="${1:-}"; [[ "$sub" == --* ]] && sub=""
case "$sub" in
  package)  cmd_package  ;;
  metadata-preview) metadata_json | jq . ;;
  plan)     say "BUNDLE_ROOT = $BUNDLE_ROOT"; say "STAGING     = $STAGING"; say "TARGET      = $TAG ($API)"; rule
            for e in "${PLAN[@]}"; do IFS='|' read -r n m sl <<<"$e"
              k="${sl%%=*}"; v="${sl#*=}"
              case "$k" in
                dir)  c=$(find "$BUNDLE_ROOT/$v" -type f 2>/dev/null | wc -l) ;;
                file) c=$([[ -f "$BUNDLE_ROOT/$v" ]] && echo 1 || echo 0) ;;
                glob) c=$(cd "$BUNDLE_ROOT" && find . -type f -name "$v" 2>/dev/null | wc -l) ;;
              esac
              printf '  %-46s %-4s %5s file(s)  %s\n' "$n" "$m" "$c" "$sl"
            done ;;
  create)   cmd_create   ;;
  adopt)    _aid=""; for _a in "$@"; do [[ "$_a" == --* || "$_a" == adopt ]] && continue; _aid="$_a"; break; done
            cmd_adopt "$_aid" ;;
  metadata) cmd_metadata ;;
  upload)   cmd_upload   ;;
  verify)   cmd_verify   ;;
  status)   cmd_status   ;;
  doi)      cmd_doi      ;;
  publish)  cmd_publish  ;;
  *) sed -n '2,40p' "$0" >&2; exit 1 ;;
esac
