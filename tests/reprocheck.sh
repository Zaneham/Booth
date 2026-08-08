#!/bin/sh
# Same input twice, same bytes out. Separate processes, so ASLR and any
# pointer-ordered container shows up as a mismatch.
set -u

root=$(git rev-parse --show-toplevel)
cd "$root"

kath=./kath.exe
[ -x "$kath" ] || kath=./kath
[ -x "$kath" ] || { echo "reprocheck: no kath binary, run make first" >&2; exit 2; }

modes="--amdgpu --nvidia-ptx --ir"
fail=0
checked=0
skipped=0

for f in tests/*.cu; do
  for m in $modes; do
    a=$("$kath" $m "$f" 2>/dev/null | sha256sum 2>/dev/null | cut -d' ' -f1)
    # A file the backend refuses is not a reproducibility failure.
    if [ -z "$a" ]; then skipped=$((skipped + 1)); continue; fi
    b=$("$kath" $m "$f" 2>/dev/null | sha256sum 2>/dev/null | cut -d' ' -f1)
    checked=$((checked + 1))
    if [ "$a" != "$b" ]; then
      fail=$((fail + 1))
      echo "NOT REPRODUCIBLE: $m $f" >&2
      echo "  $a" >&2
      echo "  $b" >&2
    fi
  done
done

if [ "$fail" -ne 0 ]; then
  echo >&2
  echo "reprocheck: $fail of $checked runs differed" >&2
  exit 1
fi

echo "reprocheck: $checked runs reproducible ($skipped skipped)"
