#!/usr/bin/env bash
# Thin wrapper around ssh/rsync to the Brev box.
#
# Resolves the box's CURRENT ip from ~/.brev/ssh_config (run `brev refresh` if
# the instance was restarted) so scripts keep working across IP changes.
#
#   ./infra/remote.sh ssh '<cmd>'        run a command on the box
#   ./infra/remote.sh sync               rsync this repo -> $REMOTE_ROOT (code only)
#   ./infra/remote.sh pull <remote> <local>   copy a file/dir back from the box
#   ./infra/remote.sh push <local> <remote>   copy a file/dir to the box
#   ./infra/remote.sh host               print user@ip
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/box.env"
eval BOX_KEY="$BOX_KEY"   # expand $HOME in the key path

HOST="$(awk -v inst="$BREV_INSTANCE" '
  $1=="Host" && $2==inst {f=1; next}
  f && tolower($1)=="hostname" {print $2; exit}
' "$HOME/.brev/ssh_config")"
[ -n "${HOST:-}" ] || { echo "Cannot resolve $BREV_INSTANCE in ~/.brev/ssh_config — run: brev refresh" >&2; exit 1; }

SSH_OPTS=(-i "$BOX_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR)
TARGET="${BOX_USER}@${HOST}"
RSH="ssh ${SSH_OPTS[*]}"

cmd="${1:-}"; shift || true
case "$cmd" in
  ssh)  ssh "${SSH_OPTS[@]}" "$TARGET" "$@" ;;
  sync) rsync -az --delete --exclude '.git' --exclude 'outputs' --exclude '*.pem' \
          --exclude 'episodes/*/stills'   --exclude 'episodes/*/voice' \
          --exclude 'episodes/*/clips'    --exclude 'episodes/*/captions' \
          --exclude 'episodes/*/final'    --exclude 'episodes/*/.tmp' \
          -e "$RSH" "$HERE/../" "$TARGET:$REMOTE_ROOT/" ;;
  pull) rsync -azP -e "$RSH" "$TARGET:$1" "$2" ;;
  push) rsync -azP -e "$RSH" "$1" "$TARGET:$2" ;;
  host) echo "$TARGET" ;;
  *) echo "usage: remote.sh {ssh <cmd>|sync|pull <remote> <local>|push <local> <remote>|host}" >&2; exit 1 ;;
esac
