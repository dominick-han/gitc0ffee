#!/bin/bash
# test_e2e.sh — end-to-end tests using the built gitc0ffee binary
#
# Creates a temp git repo, makes commits, runs gitc0ffee, and verifies
# the resulting hashes actually start with the requested prefix.

set -e

BIN="${1:-build/gitc0ffee}"
PASS=0
FAIL=0
RUN=0

if [ ! -x "$BIN" ]; then
    echo "Error: $BIN not found or not executable. Run 'make' first."
    exit 1
fi

# Resolve to absolute path before we cd into temp dirs
BIN="$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")"

TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

setup_repo() {
    rm -rf "$TMPDIR/repo"
    mkdir -p "$TMPDIR/repo"
    cd "$TMPDIR/repo"
    git init -q
    git config user.name "Test"
    git config user.email "test@test.com"
    echo "init" > file.txt
    git add file.txt
}

check() {
    local label="$1" prefix="$2" hash="$3"
    RUN=$((RUN + 1))
    local actual="${hash:0:${#prefix}}"
    if [ "$actual" = "$prefix" ]; then
        echo "ok $RUN - $label: $hash starts with $prefix"
        PASS=$((PASS + 1))
    else
        echo "not ok $RUN - $label: $hash does NOT start with $prefix"
        FAIL=$((FAIL + 1))
    fi
}

echo "TAP version 13"

# --- Basic prefix lengths ---

setup_repo
git commit -q -m "test short prefix"
HASH=$("$BIN" --prefix aa 2>/dev/null)
check "2-nibble prefix" "aa" "$HASH"

setup_repo
git commit -q -m "test odd prefix"
HASH=$("$BIN" --prefix bad 2>/dev/null)
check "3-nibble odd prefix" "bad" "$HASH"

setup_repo
git commit -q -m "test 4 nibble"
HASH=$("$BIN" --prefix f00d 2>/dev/null)
check "4-nibble prefix" "f00d" "$HASH"

setup_repo
git commit -q -m "test 6 nibble"
HASH=$("$BIN" --prefix c0ffee 2>/dev/null)
check "6-nibble prefix" "c0ffee" "$HASH"

# --- 8-nibble prefix (the real test, ~2s) ---

setup_repo
git commit -q -m "test 8 nibble"
HASH=$("$BIN" --prefix deadbeef 2>/dev/null)
check "8-nibble prefix" "deadbeef" "$HASH"

# --- --update-ref actually updates HEAD ---

setup_repo
git commit -q -m "test update-ref"
HASH=$("$BIN" --prefix c0ffee --update-ref 2>/dev/null)
HEAD=$(git rev-parse HEAD)
RUN=$((RUN + 1))
if [ "$HASH" = "$HEAD" ]; then
    echo "ok $RUN - --update-ref: HEAD matches returned hash"
    PASS=$((PASS + 1))
else
    echo "not ok $RUN - --update-ref: HEAD=$HEAD != returned=$HASH"
    FAIL=$((FAIL + 1))
fi

# --- Re-run on already-salted commit ---

HASH2=$("$BIN" --prefix beef --update-ref 2>/dev/null)
check "re-run with different prefix" "beef" "$HASH2"
HEAD2=$(git rev-parse HEAD)
RUN=$((RUN + 1))
if [ "$HASH2" = "$HEAD2" ]; then
    echo "ok $RUN - re-run: HEAD updated correctly"
    PASS=$((PASS + 1))
else
    echo "not ok $RUN - re-run: HEAD=$HEAD2 != returned=$HASH2"
    FAIL=$((FAIL + 1))
fi

# --- Long commit message ---

setup_repo
git commit -q -m "$(python3 -c "print('A' * 2000)")"
HASH=$("$BIN" --prefix c0ffee 2>/dev/null)
check "long commit message" "c0ffee" "$HASH"

# --- Multi-line commit message ---

setup_repo
git commit -q -m "feat: big feature

This is a detailed description of the feature.

- Point 1
- Point 2
- Point 3"
HASH=$("$BIN" --prefix c0ffee 2>/dev/null)
check "multi-line message" "c0ffee" "$HASH"

# --- Commit message stays clean (no visible artifacts) ---

setup_repo
git commit -q -m "clean message test"
"$BIN" --prefix aa --update-ref 2>/dev/null
MSG=$(git log -1 --format="%s")
RUN=$((RUN + 1))
if [ "$MSG" = "clean message test" ]; then
    echo "ok $RUN - message subject preserved"
    PASS=$((PASS + 1))
else
    echo "not ok $RUN - message subject changed: '$MSG'"
    FAIL=$((FAIL + 1))
fi

# --- git cat-file validates the object ---

HASH=$(git rev-parse HEAD)
RUN=$((RUN + 1))
if git cat-file -t "$HASH" | grep -q commit; then
    echo "ok $RUN - git cat-file validates object"
    PASS=$((PASS + 1))
else
    echo "not ok $RUN - git cat-file failed"
    FAIL=$((FAIL + 1))
fi

# --- Error cases ---

RUN=$((RUN + 1))
if "$BIN" --prefix xyz 2>/dev/null; then
    echo "not ok $RUN - should reject non-hex prefix"
    FAIL=$((FAIL + 1))
else
    echo "ok $RUN - rejects non-hex prefix"
    PASS=$((PASS + 1))
fi

RUN=$((RUN + 1))
if "$BIN" --version 2>&1 | grep -q "gitc0ffee"; then
    echo "ok $RUN - --version works"
    PASS=$((PASS + 1))
else
    echo "not ok $RUN - --version broken"
    FAIL=$((FAIL + 1))
fi

# --- Summary ---

echo "1..$RUN"
if [ $FAIL -gt 0 ]; then
    echo "# FAILED $FAIL of $RUN"
    exit 1
else
    echo "# All $PASS tests passed"
fi
