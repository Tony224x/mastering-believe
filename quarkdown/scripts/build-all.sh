#!/usr/bin/env bash
#
# build-all.sh — equivalent bash de build-all.ps1 (Linux / macOS / CI).
#
# Build tous les sites Quarkdown du repo (un par domaine ayant un
# 01-theory-qd/main.qd), ou un domaine cible via --domain. Chaque site
# est compile dans quarkdown/output-site/<domain>/ puis post-traite par
# post-build-fix-links.py pour rendre le bundle portable (file:// +
# WKWebView KaView).
#
# Le mode watch (-Watch cote PowerShell) n'est PAS porte ici : c'est une
# commodite de dev local. Utiliser build-all.ps1 -Domain X -Watch, ou
# `quarkdown c <main.qd> -p -w` directement.
#
# Prerequis : `java` (17+) et `quarkdown` (2.x) dans le PATH. Overrides
# optionnels QUARKDOWN_JAVA_HOME / QUARKDOWN_BIN_DIR (cf. build-all.ps1).
#
# Usage :
#   quarkdown/scripts/build-all.sh                 # tous les domaines
#   quarkdown/scripts/build-all.sh --domain agentic-ai
#   quarkdown/scripts/build-all.sh --out /tmp/site # base de sortie custom
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DOMAINS_DIR="$REPO_ROOT/domains"
OUTPUT_BASE="$REPO_ROOT/quarkdown/output-site"
POST_BUILD="$REPO_ROOT/quarkdown/post-build-fix-links.py"

DOMAIN=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --domain) DOMAIN="${2:-}"; shift 2 ;;
        --out)    OUTPUT_BASE="${2:-}"; shift 2 ;;
        -h|--help)
            sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *) echo "error: argument inconnu '$1'" >&2; exit 2 ;;
    esac
done

# Overrides locaux optionnels (parite avec build-all.ps1).
if [[ -n "${QUARKDOWN_JAVA_HOME:-}" && -d "${QUARKDOWN_JAVA_HOME}" ]]; then
    export JAVA_HOME="$QUARKDOWN_JAVA_HOME"
    export PATH="$JAVA_HOME/bin:$PATH"
fi
if [[ -n "${QUARKDOWN_BIN_DIR:-}" && -d "${QUARKDOWN_BIN_DIR}" ]]; then
    export PATH="$QUARKDOWN_BIN_DIR:$PATH"
fi

command -v quarkdown >/dev/null 2>&1 || {
    echo "error: 'quarkdown' introuvable dans le PATH (cf. quarkdown/README.md)" >&2
    exit 127
}

# Resolution de la liste (domaine, main.qd, output) a builder.
declare -a NAMES=() MAINS=() OUTS=()
declare -a SKIPPED=()

add_domain() {
    local name="$1" main="$2"
    NAMES+=("$name")
    MAINS+=("$main")
    OUTS+=("$OUTPUT_BASE/$name")
}

if [[ -n "$DOMAIN" ]]; then
    main="$DOMAINS_DIR/$DOMAIN/01-theory-qd/main.qd"
    if [[ ! -f "$main" ]]; then
        echo "ERROR: pas de 01-theory-qd/main.qd pour le domaine '$DOMAIN'" >&2
        echo "       attendu : $main" >&2
        echo "       fix     : python quarkdown/scripts/scaffold-domain.py $DOMAIN" >&2
        exit 1
    fi
    add_domain "$DOMAIN" "$main"
else
    for dir in "$DOMAINS_DIR"/*/; do
        name="$(basename "$dir")"
        main="$dir/01-theory-qd/main.qd"
        if [[ -f "$main" ]]; then
            add_domain "$name" "$main"
        else
            SKIPPED+=("$name")
        fi
    done
fi

if [[ ${#NAMES[@]} -eq 0 ]]; then
    echo "warning: aucun domaine avec 01-theory-qd/main.qd trouve." >&2
    if [[ ${#SKIPPED[@]} -gt 0 ]]; then
        echo "Domains sans 01-theory-qd/ : ${SKIPPED[*]}"
        echo "Pour scaffold : python quarkdown/scripts/scaffold-domain.py <domain>"
    fi
    exit 0
fi

declare -a OK=() FAILED=()
for i in "${!NAMES[@]}"; do
    name="${NAMES[$i]}"; main="${MAINS[$i]}"; out="${OUTS[$i]}"
    echo ""
    echo "==> Building $name"
    echo "    main.qd : $main"
    echo "    output  : $out"
    if quarkdown c "$main" --out "$out"; then
        echo "    post-process links..."
        if python "$POST_BUILD" "$out"; then
            OK+=("$name")
        else
            echo "ERROR: post-process echoue pour $name" >&2
            FAILED+=("$name")
        fi
    else
        echo "ERROR: build Quarkdown echoue pour $name" >&2
        FAILED+=("$name")
    fi
done

echo ""
echo "============================================"
echo " Recap"
echo "============================================"
[[ ${#OK[@]}      -gt 0 ]] && echo " OK    : ${OK[*]}"
[[ ${#FAILED[@]}  -gt 0 ]] && echo " FAIL  : ${FAILED[*]}"
[[ ${#SKIPPED[@]} -gt 0 ]] && echo " SKIP  : ${SKIPPED[*]} (no 01-theory-qd/)"
echo ""

[[ ${#FAILED[@]} -gt 0 ]] && exit 1
exit 0
