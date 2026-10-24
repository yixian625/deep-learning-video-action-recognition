#!/bin/bash
# Run from the repo root (src/ and test/ are linted; source-roots in pylintrc is relative to it)
cd "$(dirname "$0")/.."
# Lint the code with pylint/pylintrc and write the full report (score, statistics, messages)
# to pylint/pylint_report.txt
{
    echo "pylint report"
    echo "Generated: $(date '+%Y-%m-%d %H:%M %Z')"
    echo "Command:   python -m pylint --rcfile pylint/pylintrc src test --reports=y --persistent=n"
    python -m pylint --version | head -2 | sed 's/^/Version:   /'
    echo
    python -m pylint --rcfile pylint/pylintrc src test --reports=y --persistent=n
} > pylint/pylint_report.txt
echo "Saved pylint/pylint_report.txt ($(grep -o 'rated at [0-9.]*/10' pylint/pylint_report.txt))"
