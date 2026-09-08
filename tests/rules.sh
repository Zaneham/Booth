#!/bin/sh
# rules.sh -- the Power of Ten, as a ratchet rather than a gate.
#
# Booth does not pass JPL's rules today and a check that fails on every run is
# a check nobody reads. So the counts below are recorded in tests/rules.tbl and
# this fails when one rises, not when it is above zero. Lower a number, commit
# the new baseline, and it can never go back up.
#
# Rule 7 is the exception and is checked outright, because it is the one that
# cost a day: pp_emit_char guarded its write, dropped the byte when full, and
# returned void, so nothing downstream could tell the buffer had been truncated
# and the lexer read off the end of it.
#
# src/mlir/vendor is Certik's and is not ours to hold to this.

set -e
cd "$(dirname "$0")/.."

SRC=$(ls src/*.c src/*/*.c src/*.h src/*/*.h 2>/dev/null | grep -v '/vendor/')
TBL=tests/rules.tbl
bad=0

count() {
    case "$1" in
    goto)  grep -hoE '\b(goto|setjmp|longjmp)\b' $SRC 2>/dev/null | wc -l ;;
    alloc) grep -hoE '\b(malloc|calloc|realloc|free)[[:space:]]*\(' $SRC 2>/dev/null | wc -l ;;
    fnptr) grep -hoE '\([[:space:]]*\*[[:space:]]*[a-z_]+[[:space:]]*\)[[:space:]]*\(' $SRC 2>/dev/null | wc -l ;;
    long)  awk '/^[a-z_].*\(|^static .*\(/{n=0;f=1} f{n++} /^\}/{if(f&&n>60)c++;f=0} END{print c+0}' $SRC ;;
    comment) awk '{s=$0; sub(/^[ \t]+/,"",s); if (s ~ /^(\/\*|\* |\*\/|\/\/)/) c++} END{print c+0}' $SRC ;;
    snprintf) grep -h 'snprintf[[:space:]]*(' $SRC 2>/dev/null \
                | grep -cvE '(=|\(|&&|\|\||<|>|!|return)[[:space:]]*snprintf' ;;
    esac
}

echo "[rules] ratchet, from $TBL"
while read -r name limit label; do
    [ -z "$name" ] && continue
    case "$name" in \#*) continue ;; esac
    now=$(count "$name" | tr -d ' ')
    if [ "$now" -gt "$limit" ]; then
        echo "  RISEN  $label: $now, baseline $limit"
        bad=1
    elif [ "$now" -lt "$limit" ]; then
        echo "  fallen $label: $now, baseline $limit, lower the baseline"
    else
        echo "  held   $label: $now"
    fi
done < "$TBL"

echo "[rules] every code with a message has a translation"
miss=$(awk -F'[ *]+' '/\/\* E[0-9]{3} \*\/[[:space:]]*"/{print $3}' src/fe/bc_err.c \
       | while read -r c; do
             grep -q "^$c=" lang/en.txt || echo "  $c has a message in bc_err.c and no entry in lang/en.txt"
         done)
if [ -n "$miss" ]; then echo "$miss"; bad=1; else echo "  clean"; fi

echo "[rules] 7  a function that can fail says so"
py=$(command -v python3 || command -v python) || py=
if [ -n "$py" ]; then
    hits=$("$py" - "$@" <<'EOF'
import re, glob
pat = re.compile(r'static\s+void\s+(\w+)\s*\([^)]{0,200}\)\s*\{(.{0,500}?)\n\}', re.S)
grd = re.compile(r'if\s*\([^)]{0,120}<\s*[^)]{0,60}(max|cap)\w*\s*\)')
out = []
for f in sorted(glob.glob('src/*.c') + glob.glob('src/*/*.c')):
    if '/vendor/' in f:
        continue
    t = open(f, encoding='utf-8', errors='replace').read()
    for m in pat.finditer(t):
        if grd.search(m.group(2)):
            out.append('  %s: %s guards on capacity and returns void' % (f, m.group(1)))
print('\n'.join(out))
EOF
)
    if [ -n "$hits" ]; then
        echo "$hits"
        bad=1
    else
        echo "  clean"
    fi
else
    echo "  skipped, no python"
fi

if [ "$bad" -ne 0 ]; then
    echo "rules: failed"
    exit 1
fi
echo "rules: clean"
