#!/usr/bin/env bash
# Builds fly-doctrines.odmod. Needs a wasm32-capable clang and an OpenDoctrines
# checkout for sdk/gearbox.h and tools/pack_odmod.sh.
set -eu
here="$(cd "$(dirname "$0")" && pwd)"
OD="${OD:-$HOME/CLionProjects/OpenDoctrines}"
[ -f "$OD/sdk/gearbox.h" ] || { echo "set OD= to an OpenDoctrines checkout"; exit 1; }

CC="${CC:-}"
if [ -z "$CC" ]; then
    for c in "$(command -v wasm32-wasi-clang || true)" \
             "$(command -v clang || true)" \
             /opt/homebrew/Cellar/emscripten/*/libexec/llvm/bin/clang; do
        [ -x "$c" ] || continue
        if "$c" --print-targets 2>/dev/null | grep -qi wasm32; then CC="$c"; break; fi
    done
fi
[ -n "$CC" ] || { echo "no wasm32-capable clang found (set CC=)"; exit 1; }

# -nostdlib: there is no libc in the sandbox. --allow-undefined: the gearbox
# imports are resolved by the host, not at link time.
# -fno-builtin -ffreestanding: without them the optimiser recognises a
# hand-written string-length loop and rewrites it into a call to strlen, which
# does not exist here -- the module then imports env.strlen and the host refuses
# it. The failure appears at LOAD, long after a clean compile, so it is worth
# closing at the flag rather than by avoiding loop shapes.
"$CC" --target=wasm32 -nostdlib -ffreestanding -fno-builtin -O2 -std=c11 \
      -I "$OD/sdk" -I "$here/src" \
      -Wl,--no-entry -Wl,--allow-undefined -Wl,--export-dynamic \
      -o "$here/mod.wasm" "$here/src/mod.c" "$here/src/fly_brain.c"

python3 "$OD/tools/wasm_imports.py" "$here/mod.wasm" >/dev/null
"$OD/tools/pack_odmod.sh" "$here" "$here/fly-doctrines.odmod"
