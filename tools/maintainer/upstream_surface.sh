#!/usr/bin/env bash
# Measure the upstream merge surface of a local customization branch.
#
# The maintainer lines (mobile-quasar, mobile-quasar-yarn) carry local renderer, YaRN and
# froggeric work on top of upstream. Before every upstream sync this script answers two
# questions:
#
#   1. how much of the branch's own diff lands in files upstream also edits (the "seam"), and
#      how often upstream has touched each of those files recently;
#   2. what `git merge upstream/master` would conflict on (an in-memory rehearsal, no work tree).
#
# Usage:
#   tools/maintainer/upstream_surface.sh [options]
#
#   --upstream <rev>   upstream branch to merge and rank churn against (default upstream/master)
#   --parent <rev>     long-lived line this branch forks from (default mobile-quasar-yarn)
#   --line <rev>       customization branch to measure (default HEAD)
#   --base <rev>       override the owned base (default merge-base of --parent and --line)
#   --churn <n>        upstream commit window for per-file churn (default 200)
#   --owned <regex>    branch-owned path prefixes (default: the froggeric v22.5 module)
#   --strict           exit 1 when the seam budget is exceeded
#   --max-files <n>    strict seam file budget (default 15)
#   --max-lines <n>    strict seam net-added-line budget (default 60)
#   -h, --help         this message
#
# Read-only: the script never checks out, merges, or writes to the work tree. The rehearsal
# uses `git merge-tree --write-tree`, which requires git >= 2.38.

set -euo pipefail

upstream=upstream/master
parent=mobile-quasar-yarn
line=HEAD
base=""
churn=200
strict=0
max_files=15
max_lines=60
owned='^(src/product/chat_style\.h|src/targets/qwen3_6/impl/frontend/froggeric_v22_5_|src/targets/qwen3_6/impl/frontend/render_fragment\.|src/targets/qwen3_6/impl/frontend/tool_call_json_parser\.)'

usage() { sed -n '2,27p' "$0"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --upstream) upstream=$2; shift 2;;
    --parent)   parent=$2;   shift 2;;
    --line)     line=$2;     shift 2;;
    --base)     base=$2;     shift 2;;
    --churn)    churn=$2;    shift 2;;
    --owned)    owned=$2;    shift 2;;
    --strict)   strict=1;    shift;;
    --max-files) max_files=$2; shift 2;;
    --max-lines) max_lines=$2; shift 2;;
    -h|--help)  usage; exit 0;;
    *) echo "unknown argument: $1 (try --help)" >&2; exit 2;;
  esac
done

require_rev() {
  if ! git rev-parse --verify --quiet "$1^{commit}" >/dev/null; then
    echo "unknown revision: $1" >&2
    exit 2
  fi
}
require_rev "$upstream"
require_rev "$parent"
require_rev "$line"
[ -n "$base" ] || base=$(git merge-base "$parent" "$line")
require_rev "$base"
upstream_base=$(git merge-base "$upstream" "$line")

echo "== owned surface: $base..$line (parent=$parent) =="
new=0
modified=0
added=0
removed=0
seam_modified=0
seam_added=0
seam_removed=0
while read -r a d f; do
  [ -n "${f:-}" ] || continue
  # Binary files report "-" instead of a line count.
  [[ "$a" =~ ^[0-9]+$ ]] || a=0
  [[ "$d" =~ ^[0-9]+$ ]] || d=0
  if [[ "$f" =~ $owned ]]; then
    new=$((new + 1))
    continue
  fi
  if git cat-file -e "$base:$f" 2>/dev/null; then
    modified=$((modified + 1))
    added=$((added + a))
    removed=$((removed + d))
    if [[ "$f" =~ ^(src|apps|include)/ ]]; then
      seam_modified=$((seam_modified + 1))
      seam_added=$((seam_added + a))
      seam_removed=$((seam_removed + d))
    fi
    n=$(git log --oneline -"$churn" "$upstream" -- "$f" | wc -l | tr -d ' ')
    printf '%4s|%5s|%5s| %s\n' "$n" "+$a" "-$d" "$f"
  else
    new=$((new + 1))
  fi
done < <(git diff --numstat "$base".."$line" | sort -k1 -rn)

echo "-- totals: owned-new=$new modified-existing=$modified +$added/-$removed"
echo "-- seam (src/apps/include): modified=$seam_modified +$seam_added/-$seam_removed"
echo "-- upstream delta: $upstream_base..$upstream ($(git rev-list --count "$upstream_base".."$upstream") commit(s))"

# Conflicted files for one revision, from the in-memory merge of $upstream. `--name-only`
# prints the merged tree id first, then one conflicted path per line up to a blank line.
conflict_files() {
  local rev=$1
  local out
  out=$(git merge-tree --write-tree --name-only "$upstream" "$rev" 2>/dev/null || true)
  printf '%s\n' "$out" | sed -n '2,/^$/p' | grep -v '^$' || true
}

line_conflicts=$(conflict_files "$line" | sort)
parent_conflicts=$(conflict_files "$parent" | sort)
introduced=$(comm -23 <(printf '%s\n' "$line_conflicts") <(printf '%s\n' "$parent_conflicts") |
             grep -v '^$' || true)
preexisting=$(comm -12 <(printf '%s\n' "$line_conflicts") <(printf '%s\n' "$parent_conflicts") |
              grep -v '^$' || true)

echo "== rehearsal: merge $upstream into $line =="
if [ -z "$line_conflicts" ]; then
  echo "  (no conflicts)"
else
  echo "  conflicted files:"
  printf '%s\n' "$line_conflicts" | sed 's/^/    /'
fi
if [ -n "$preexisting" ]; then
  echo "  pre-existing on $parent (not introduced here):"
  printf '%s\n' "$preexisting" | sed 's/^/    /'
fi
if [ -n "$introduced" ]; then
  echo "  introduced by this branch:"
  printf '%s\n' "$introduced" | sed 's/^/    /'
fi

if [ "$strict" -eq 1 ]; then
  budget_ok=1
  if [ "$seam_modified" -gt "$max_files" ]; then
    echo "BUDGET: seam files $seam_modified > $max_files" >&2
    budget_ok=0
  fi
  if [ $((seam_added - seam_removed)) -gt "$max_lines" ]; then
    echo "BUDGET: seam net lines $((seam_added - seam_removed)) > $max_lines" >&2
    budget_ok=0
  fi
  [ "$budget_ok" -eq 1 ] || exit 1
fi
