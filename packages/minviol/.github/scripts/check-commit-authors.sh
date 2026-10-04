#!/usr/bin/env bash
#
# Fail if any commit in a range was authored or committed by an AI agent identity.
#
# A commit records who made a change. Work done with an assistant is still the
# work of the person who directed and reviewed it, so this repository's
# convention is that a person is the author and the assistant is credited in a
# Co-Authored-By trailer. Two commits arrived authored by "Claude
# <noreply@anthropic.com>" instead, which is what this exists to catch.
#
# The trailer is deliberately NOT checked. Flagging it would reject every commit
# in the repository, including the ones that follow the convention correctly.
# Only the author and committer identity fields are examined.
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
        local culprit=""
        for field in "$author_name" "$author_email" "$committer_name" "$committer_email"; do
            if printf '%s' "$field" | grep -Eiq "$FORBIDDEN"; then
                culprit="$field"
                break
            fi
        done
        if [ -n "$culprit" ]; then
            found=1
            printf 'FAIL %s\n' "$(git log -1 --format='%h %s' "$sha")"
            printf '     author    %s <%s>\n' "$author_name" "$author_email"
            printf '     committer %s <%s>\n' "$committer_name" "$committer_email"
            printf '     matched   %s\n\n' "$culprit"
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
        # The convention: a person authors, the assistant is credited in a trailer.
        git commit -q -m "A change a person made

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
        git tag clean
        echo two > file; git add file
        git -c user.name="Claude" -c user.email="noreply@anthropic.com" \
            commit -q -m "A change attributed to the assistant"
    ) >/dev/null

    local failures=0
    if ! ( cd "$work" && scan "clean" ) >/dev/null 2>&1; then
        echo "self-test FAILED: rejected a commit that follows the convention"
        echo "  (a Co-Authored-By trailer must not count as authorship)"
        failures=1
    fi
    if ( cd "$work" && scan "clean..HEAD" ) >/dev/null 2>&1; then
        echo "self-test FAILED: accepted a commit authored by the assistant"
        failures=1
    fi
    if [ "$failures" -eq 0 ]; then
        echo "self-test passed: the trailer is allowed, the authorship is not"
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
A commit records who made a change. Work done with an assistant is still the work
of the person who directed and reviewed it, so a person must be the author; credit
the assistant with a Co-Authored-By trailer instead, which this check allows.

To correct the commits listed above, re-author them and force-push:

    git rebase --onto <base> <base> HEAD \
        --exec 'git commit --amend --no-edit --reset-author'
    git push --force-with-lease
MESSAGE
    exit 1
fi
