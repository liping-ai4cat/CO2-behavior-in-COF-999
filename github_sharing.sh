#!/usr/bin/env bash
#
# github_sharing.sh -- publish the code half of this bundle to GitHub.
#
#   The repository and the Zenodo data record share one directory layout. This
#   script commits and pushes only what .gitignore leaves tracked (~190 MB);
#   the ~12 GB of bulk goes to Zenodo via ./zenodo_deposit.sh.
#
# SAFETY
#   * `check` runs first and is also run automatically before `push`. A failing
#     check blocks the push.
#   * `push` publishes. It requires a typed confirmation. Content pushed to a
#     public repository can be cached, forked or indexed even if deleted later.
#   * Force-pushing is refused outright; there is no flag to enable it.
#   * Nothing here creates the GitHub repository -- `gh` is not installed on
#     this host. Create it in the web UI first, EMPTY (no README, no .gitignore,
#     no licence), then run `remote` and `push`.
#
# USAGE
#   ./github_sharing.sh check                 # preflight only, writes nothing
#   ./github_sharing.sh init [email]          # git init + first commit
#       The optional e-mail is set repo-locally BEFORE the first commit, so it
#       is the address recorded in the public history. Your global git config
#       is not modified. You cannot set this beforehand with `git config`,
#       because that needs a repository to already exist.
#   ./github_sharing.sh remote <url>          # set origin (ssh or https)
#   ./github_sharing.sh push                  # PUBLISH (typed confirmation)
#   ./github_sharing.sh tag v1.0.0            # annotated tag for a Zenodo release
#   ./github_sharing.sh status
#
# GETTING A CODE DOI (after the first push)
#   1. Log in to https://zenodo.org with GitHub, open Settings -> GitHub, and
#      flip this repository ON. Zenodo only archives releases created *after*
#      the switch is on.
#   2. Create a release on GitHub (Releases -> Draft a new release), tag v1.0.0.
#   3. Zenodo archives it and mints a code DOI, separate from the data DOI.
#   4. Put both DOIs in README.md and in the manuscript.
#
set -euo pipefail

SELF_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SELF_DIR"

DEFAULT_REMOTE="https://github.com/liping-ai4cat/CO2-behavior-in-COF-999.git"
BRANCH="main"
MAX_FILE_BYTES=$((100*1024*1024))   # GitHub hard limit, per file
WARN_FILE_BYTES=$((50*1024*1024))   # GitHub starts warning here
WARN_REPO_BYTES=$((1024*1024*1024)) # GitHub recommends staying under ~1 GB

say()  { printf '%s\n' "$*" >&2; }
ok()   { printf '  \033[32mPASS\033[0m  %s\n' "$*" >&2; }
warn() { printf '  \033[33mWARN\033[0m  %s\n' "$*" >&2; WARNS=$((WARNS+1)); }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$*" >&2; FAILS=$((FAILS+1)); }
rule() { printf '%s\n' "------------------------------------------------------------" >&2; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
human(){ numfmt --to=iec --suffix=B "$1" 2>/dev/null || echo "$1 bytes"; }

# Files git would track: everything not matched by .gitignore.
tracked_list() {
  if [[ -d .git ]]; then
    git ls-files -z --cached --others --exclude-standard
  else
    # Probe without writing anything into the work tree: the throwaway index
    # and object store live in a temp GIT_DIR that is deleted immediately.
    local tmp; tmp="$(mktemp -d)"
    GIT_DIR="$tmp/git" GIT_WORK_TREE="$PWD" git init -q >/dev/null 2>&1 || true
    GIT_DIR="$tmp/git" GIT_WORK_TREE="$PWD" git ls-files -z --others --exclude-standard
    rm -rf "$tmp"
  fi
}

# --------------------------------------------------------------------------
cmd_check() {
  FAILS=0; WARNS=0
  say "Preflight for: $SELF_DIR"
  rule

  [[ -f .gitignore ]] && ok ".gitignore present" || bad ".gitignore MISSING -- 12 GB of bulk would be committed"
  [[ -f README.md  ]] && ok "README.md present"  || bad "README.md missing"
  [[ -f LICENSE    ]] && ok "LICENSE present"    || warn "LICENSE missing"

  local n=0 total=0 big=() warnbig=()
  while IFS= read -r -d '' f; do
    [[ -f "$f" ]] || continue
    local sz; sz=$(stat -c%s "$f")
    n=$((n+1)); total=$((total+sz))
    (( sz > MAX_FILE_BYTES ))  && big+=("$f ($(human "$sz"))")
    (( sz > WARN_FILE_BYTES && sz <= MAX_FILE_BYTES )) && warnbig+=("$f ($(human "$sz"))")
  done < <(tracked_list)

  say "  ....  $n files, $(human "$total") would be committed"
  if (( ${#big[@]} )); then
    bad "${#big[@]} file(s) exceed GitHub's 100 MB hard limit -- the push WILL be rejected:"
    printf '           %s\n' "${big[@]}" >&2
  else
    ok "no tracked file exceeds 100 MB"
  fi
  (( ${#warnbig[@]} )) && { warn "${#warnbig[@]} file(s) over 50 MB (GitHub will warn):"; printf '           %s\n' "${warnbig[@]}" >&2; } || ok "no tracked file over 50 MB"
  (( total > WARN_REPO_BYTES )) && warn "repository over 1 GB ($(human "$total")) -- GitHub recommends staying below this" || ok "repository size is comfortable"

  # symlinks: git stores the link TEXT, not the target -- a clone elsewhere gets a dangling link
  local nsym; nsym=$(find . -path ./.git -prune -o -type l -print | wc -l)
  (( nsym == 0 )) && ok "no symlinks (git would commit these as text, not data)" \
                  || { bad "$nsym symlink(s) found -- they would commit as dangling links:"; find . -path ./.git -prune -o -type l -print | sed 's/^/           /' >&2; }

  # licensed / private material
  local npot; npot=$(find . -name 'POTCAR*' -not -path './.git/*' | wc -l)
  (( npot == 0 )) && ok "no VASP POTCAR (licence-restricted)" || bad "$npot POTCAR file(s) present -- must not be published"

  # secrets
  local hits
  hits=$(grep -rIlE 'ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{22,}|BEGIN [A-Z ]*PRIVATE KEY|AKIA[0-9A-Z]{16}|(ZENODO|GITHUB)_TOKEN=[A-Za-z0-9]' . \
         --exclude-dir=.git --exclude=github_sharing.sh --exclude=zenodo_deposit.sh 2>/dev/null || true)
  [[ -z "$hits" ]] && ok "no API tokens or private keys" || { bad "possible secrets:"; printf '%s\n' "$hits" | sed 's/^/           /' >&2; }

  hits=$(grep -rIoE '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}' . --exclude-dir=.git 2>/dev/null \
         | grep -v '@users\.noreply\.github\.com' | cut -d: -f1 | sort -u || true)
  [[ -z "$hits" ]] && ok "no e-mail addresses in tracked content" \
    || { warn "e-mail address(es) in:"; printf '%s\n' "$hits" | sed 's/^/           /' >&2; }

  # the commit author e-mail also becomes public, permanently
  local ce; ce=$(git config --get user.email 2>/dev/null || echo "")
  if [[ -z "$ce" ]]; then
    bad "git user.email is not set"
  elif [[ "$ce" == *"noreply"* ]]; then
    ok "commit e-mail is a noreply address ($ce)"
  else
    warn "commits will carry $ce, permanently and publicly, in every commit."
    say  "           If you scrubbed that address from the files, set a repo-local"
    say  "           alias before committing:"
    say  "             git config user.email 'liping-ai4cat@users.noreply.github.com'"
  fi

  # placeholders still to fill
  local ph; ph=$(grep -noE 'zenodo\.XXXXXXX|<TITLE>|<AUTHORS>|<YEAR>|CORRESPONDING AUTHOR' README.md 2>/dev/null | head -8 || true)
  [[ -z "$ph" ]] && ok "no unfilled placeholders in README.md" \
    || { warn "unfilled placeholder(s) in README.md:"; printf '%s\n' "$ph" | sed 's/^/           /' >&2; }

  rule
  if (( FAILS )); then say "$FAILS failure(s), $WARNS warning(s) -- fix the failures before pushing."; return 1; fi
  say "0 failures, $WARNS warning(s). Safe to commit."
}

# --------------------------------------------------------------------------
cmd_init() {
  [[ -d .git ]] && die "Already a git repository. Use 'commit'."
  local email="${1:-}"
  cmd_check || die "Preflight failed; not initialising."
  git init -q -b "$BRANCH"
  # Set the commit identity BEFORE the first commit: the author e-mail is baked
  # into every commit and is public forever. Repo-local, so your global config
  # is untouched.
  if [[ -n "$email" ]]; then
    git config user.email "$email"
    say "Commit e-mail for this repository: $email"
  else
    say "Commit e-mail for this repository: $(git config --get user.email) (inherited from your global config)"
    say "To use a different one, delete .git and re-run:  $0 init <email>"
  fi
  git add -A
  git -c core.pager=cat commit -q -m "COF-999 CO2 capture: data, models and analysis code

Code and small tabular data supporting 'Molecular origins of CO2 capture
behavior in amine-appended nanoporous frameworks'.

Bulk data (model checkpoint, DFT training trajectory, metadynamics
stationary points, per-component benchmark tables) is deposited in the
companion Zenodo record under identical paths; see README.md."
  say "Initialised on '$BRANCH' with $(git ls-files | wc -l) files."
  say "Next: ./github_sharing.sh remote [url]   then   ./github_sharing.sh push"
}

cmd_commit() {
  [[ -d .git ]] || die "Not a git repository. Run 'init' first."
  cmd_check || die "Preflight failed; not committing."
  git add -A
  git diff --cached --quiet && { say "Nothing to commit."; return 0; }
  git -c core.pager=cat commit -q -m "${1:-Update data-sharing bundle}"
  say "Committed. $(git ls-files | wc -l) files tracked."
}

cmd_remote() {
  [[ -d .git ]] || die "Not a git repository. Run 'init' first."
  local url="${1:-$DEFAULT_REMOTE}"
  if git remote get-url origin >/dev/null 2>&1; then
    say "origin was $(git remote get-url origin)"; git remote set-url origin "$url"
  else
    git remote add origin "$url"
  fi
  say "origin -> $(git remote get-url origin)"
  say "Create the repository EMPTY on github.com first (no README/licence), or the push will be rejected."
}

cmd_push() {
  [[ -d .git ]] || die "Not a git repository. Run 'init' first."
  git remote get-url origin >/dev/null 2>&1 || die "No origin. Run: $0 remote <url>"
  cmd_check || die "Preflight failed; refusing to push."

  local url n sz
  url=$(git remote get-url origin)
  n=$(git ls-files | wc -l)
  sz=$(git ls-files -z | xargs -0 stat -c%s 2>/dev/null | awk '{s+=$1} END {print s+0}')
  rule
  say "About to PUSH"
  say "  remote: $url"
  say "  branch: $BRANCH"
  say "  files:  $n  ($(human "$sz"))"
  say "  commit: $(git log -1 --oneline 2>/dev/null || echo none)"
  rule
  say "This publishes. Anything pushed to a public repository may be cached,"
  say "forked or indexed by third parties even if you delete it afterwards."
  say ""
  say "Type exactly:  PUSH TO GITHUB"
  local reply; read -r reply
  [[ "$reply" == "PUSH TO GITHUB" ]] || die "Not confirmed; nothing pushed."

  git push -u origin "$BRANCH"      # never --force, by design
  rule
  say "Pushed. To mint a code DOI, follow 'GETTING A CODE DOI' at the top of this script."
}

cmd_tag() {
  [[ -d .git ]] || die "Not a git repository."
  local t="${1:?usage: $0 tag v1.0.0}"
  git tag -a "$t" -m "Release $t: data and code accompanying the COF-999 CO2 capture manuscript"
  say "Created tag $t. Publish it with:  git push origin $t"
  say "Then draft a GitHub release from that tag so Zenodo archives it."
}

cmd_status() {
  [[ -d .git ]] || { say "Not a git repository yet. Run: $0 init"; return 0; }
  say "branch:  $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '(no commits)')"
  say "origin:  $(git remote get-url origin 2>/dev/null || echo '(none)')"
  say "tracked: $(git ls-files | wc -l) files"
  say "commits: $(git rev-list --count HEAD 2>/dev/null || echo 0)"
  git -c core.pager=cat status --short | head -20 >&2
}

sub="${1:-check}"; shift || true
case "$sub" in
  check)  cmd_check  ;;
  init)   cmd_init "${1:-}" ;;
  commit) cmd_commit "$@" ;;
  remote) cmd_remote "$@" ;;
  push)   cmd_push   ;;
  tag)    cmd_tag    "$@" ;;
  status) cmd_status ;;
  *) sed -n '2,40p' "$0" >&2; exit 1 ;;
esac
