#!/bin/sh

# Delete anything the setup and build steps may have created. Safe to re-run.
cd "$(dirname "$0")"

if [ -d .venv ]; then
    rm -rf .venv
fi

# Remove generated Core ML packages (exerciser.mlpackage and any variants).
for pkg in *.mlpackage; do
    [ -e "$pkg" ] && rm -rf "$pkg"
done

echo "Cleaned up: removed .venv and *.mlpackage"
