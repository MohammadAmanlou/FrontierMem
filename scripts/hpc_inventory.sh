#!/usr/bin/env bash
# Safe, read-only SU-HPC inventory for FrontierMem.
# Intended to run on an HPC LOGIN node; no heavy computation.
set -euo pipefail

printf '\n== identity and host ==\n'
whoami
hostname
date -Is

printf '\n== storage ==\n'
printf 'HOME: %s\n' "$HOME"
df -h "$HOME" || true
command -v quota >/dev/null && quota -s || true

printf '\n== partitions ==\n'
if command -v sinfo >/dev/null; then
  sinfo -o '%P | availability=%a | limit=%l | nodes=%D | state=%t | gres=%G'
  printf '\n== node inventory (first 40 rows) ==\n'
  sinfo -N -o '%N | %P | %t | cpus=%c | memory_mb=%m | gres=%G' | head -n 40 || true
else
  printf 'sinfo unavailable; check whether you are on an HPC login node.\n'
fi

printf '\n== your Slurm jobs ==\n'
if command -v squeue >/dev/null; then
  squeue -u "$USER" -o '%.12i %.14P %.18j %.8T %.12M %.30R' || true
fi

printf '\n== selected partition limits ==\n'
if command -v scontrol >/dev/null; then
  for p in cuda short_mdbf mid_mdbf; do
    printf '\n-- %s --\n' "$p"
    scontrol show partition "$p" 2>&1 | head -n 8 || true
  done
fi

printf '\n== development tools ==\n'
for cmd in git python3 pip3 sbatch srun nvidia-smi; do
  command -v "$cmd" || true
done
python3 -V 2>&1 || true
if command -v module >/dev/null; then
  module -t avail 2>&1 | head -n 50 || true
fi

printf '\nNo GPU workload was run by this script. GPU type and VRAM must be checked via a Slurm GPU allocation.\n'
