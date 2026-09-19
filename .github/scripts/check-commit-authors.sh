#!/usr/bin/env bash
#
# Fail if any commit in a range names an AI agent as an author or co-author.
#
# A commit records who made a change. Work done with an assistant is the work of
# the person who directed and reviewed it, and this repository does not record
# the assistant anywhere in the commit: not as author, not as committer, and not
# in a Co-Authored-By trailer.
#
#   check-commit-authors.sh [range]     default: origin/main..HEAD
#   check-commit-authors.sh --self-test
#
set -euo pipefail

# Matched case-insensitively against author/committer name and email.
FORBIDDEN='claude|codex|anthropic|openai'

scan() {
    local range="$1" found=0 sha author_name author_email committer_name committer_email
    # A unit separator keeps names containing spaces or commas intact.
    while IFS=$'\037' read -r sha author_name author_email committer_name committer_email; do
        [ -n "$sha" ] || continue
        local culprit="" where=""
        for field in "$author_name" "$author_email" "$committer_name" "$committer_email"; do
            if printf '%s' "$field" | grep -Eiq "$FORBIDDEN"; then
                culprit="$field"; where="author/committer"
                break
            fi
        done
        if [ -z "$culprit" ]; then
            # Trailers too: a Co-Authored-By naming an assistant records it just
            # as durably as the author field does.
            local trailer
            trailer="$(git log -1 --format='%(trailers:key=Co-Authored-By,valueonly)' "$sha" \
                       | grep -Ei "$FORBIDDEN" | head -1 || true)"
            if [ -n "$trailer" ]; then
                culprit="$trailer"; where="Co-Authored-By trailer"
            fi
        fi
        if [ -n "$culprit" ]; then
            found=1
            printf 'FAIL %s\n' "$(git log -1 --format='%h %s' "$sha")"
            printf '     author    %s <%s>\n' "$author_name" "$author_email"
            printf '     committer %s <%s>\n' "$committer_name" "$committer_email"
            printf '     matched   %s  (in the %s)\n\n' "$culprit" "$where"
        fi
    done < <(git log --no-merges --format="%H%x1f%an%x1f%ae%x1f%cn%x1f%ce" "$range")
    return "$found"
}

self_test() {
    # A check that cannot demonstrate catching the thing it checks for is not a
    # check. This builds a throwaway repository with one acceptable commit and
    # one unacceptable one, and asserts the scan agrees.
    local work
    work="$(mktemp -d)"
    trap 'rm -rf "$work"' RETURN
    (
        cd "$work"
        git init -q .
        git config user.name "A Person"; git config user.email "person@example.com"
        echo one > file; git add file
        git commit -q -m "A change a person made"
        git tag clean
        echo two > file; git add file
        git -c user.name="Claude" -c user.email="noreply@anthropic.com" \
            commit -q -m "A change attributed to the assistant"
        git tag bad-author
        echo three > file; git add file
        git commit -q -m "A change crediting the assistant

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
    ) >/dev/null

    local failures=0
    if ! ( cd "$work" && scan "clean" ) >/dev/null 2>&1; then
        echo "self-test FAILED: rejected a commit naming only a person"
        failures=1
    fi
    if ( cd "$work" && scan "clean..bad-author" ) >/dev/null 2>&1; then
        echo "self-test FAILED: accepted a commit authored by an assistant"
        failures=1
    fi
    if ( cd "$work" && scan "bad-author..HEAD" ) >/dev/null 2>&1; then
        echo "self-test FAILED: accepted a Co-Authored-By trailer naming an assistant"
        failures=1
    fi
    if [ "$failures" -eq 0 ]; then
        echo "self-test passed: a person alone is accepted; authorship and trailers are not"
    fi
    return "$failures"
}

if [ "${1:-}" = "--self-test" ]; then
    self_test
    exit $?
fi

range="${1:-origin/main..HEAD}"
echo "Checking commit authorship over ${range}"
if scan "$range"; then
    echo "OK: every commit is authored and committed by a person."
else
    cat <<'MESSAGE'
A commit records who made a change. Work done with an assistant is the work of the
person who directed and reviewed it, and this repository does not record the
assistant in the commit at all -- not as author, not as committer, not in a
Co-Authored-By trailer.

To correct the commits listed above, rewrite them and force-push:

    git filter-branch -f --msg-filter \
        "grep -viE '^Co-Authored-By:.*(claude|codex|anthropic|openai)' || true" \
        --env-filter 'export GIT_AUTHOR_NAME="$(git config user.name)"
                      export GIT_AUTHOR_EMAIL="$(git config user.email)"' \
        <base>..HEAD
    git push --force-with-lease
MESSAGE
    exit 1
fi
