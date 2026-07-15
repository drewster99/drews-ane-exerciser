#!/bin/sh

# Clean up old stuff
if [ -d .venv ]; then
    rm -rf .venv
fi
if [ -d exerciser.mlpackage ]; then
    rm -rf exerciser.mlpackage
fi

# Set up python virtual environment and set up whatever we need
./setup.sh

# Activate virtual environment
source .venv/bin/activate

# Create a new CoreML package
python makemodel.py

# Run the tests and print the comparison
python try_ane.py

# Done
exit 0
