#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p external
checkout() {
  local url="$1" dest="$2" revision="$3"
  if [[ ! -d "$dest/.git" ]]; then git clone "$url" "$dest"; fi
  if [[ -n "$(git -C "$dest" status --porcelain)" ]]; then
    printf 'Refusing to modify dirty checkout: %s\n' "$dest" >&2; exit 1
  fi
  git -C "$dest" fetch origin "$revision"
  git -C "$dest" checkout --detach "$revision"
}
checkout https://github.com/alibaba-damo-academy/RynnVLA-002.git external/worldvla ebdc36dc5cfdc3288e19a7b117b2ff07b135c8be
checkout https://github.com/alibaba-damo-academy/RynnVLA-002.git external/rynnvla e548ccc2977ba4fc06013de18192fa6245e454a4
checkout https://github.com/Lifelong-Robot-Learning/LIBERO.git external/LIBERO 8f1084e3132a39270c3a13ebe37270a43ece2a01
printf 'External checkouts ready. Install external/LIBERO in each model environment.\n'
