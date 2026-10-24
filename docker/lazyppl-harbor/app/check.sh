#!/bin/bash
# Quick type-check without full compilation.
# Run this while editing Submission.hs to catch type errors early.
set -euo pipefail

if [ ! -f /app/Submission.hs ]; then
    echo "Submission.hs not found at /app/Submission.hs"; exit 1
fi

ghc -fno-code \
    -i/app \
    /app/Submission.hs \
    && echo "Type check passed" \
    || { echo "Type check FAILED"; exit 1; }
