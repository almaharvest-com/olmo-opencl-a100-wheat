#!/usr/bin/env bash
# Rebuilds olmo_inputs.tgz from the two folders next to it. Run it in WSL, from this folder.
# The bundle keeps both folder names at the top level, which is what "g13_setup.sh tools" expects.
set -euo pipefail
cd "$(dirname "$0")"
tar czf olmo_inputs.tgz OLMO_opencl Olmoearth-retraining-for-wheat-field
ls -l olmo_inputs.tgz
