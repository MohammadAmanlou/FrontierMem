# Sabancı University HPC connection and FrontierMem setup

This guide summarizes the **2026-09-16 SU-Net guest VPN message** and **2026-10-26 Sine Mete HPC access message** sent to the account owner. **Never commit, copy, or share either password.** The linked University's documentation takes priority over a dated email if access policies change.

## Access facts (no secrets)
- VPN login: `guest1131@sabanciuniv.edu` (the @domain suffix is required).
- HPC SSH username: `guest1131`.
- Allowed HPC hosts: `tosun.sabanciuniv.edu`, `sakura.sabanciuniv.edu`, `tortoise.sabanciuniv.edu`.
- HPC email announced partitions/QoS: `cuda`, `long_mdbf`, `longer_mdbf`, `mid_mdbf`, `short_mdbf`.
- Email's initial test: `srun -N2 -n2 --partition=short_mdbf --qos=short_mdbf hostname`.
- The VPN and HPC passwords may be different. HPC first-login policy requires changing the initial HPC password with `passwd`.

Documentation:
- VPN client: https://mysu.sabanciuniv.edu/it/en/globalprotect-vpn-connection-guide
- HPC Quickstart: https://su-hpc-tutorials.readthedocs.io/en/latest/
- Slurm: https://su-hpc-tutorials.readthedocs.io/en/latest/slurm/
- Support: https://hc.suticket.sabanciuniv.edu/servicedesk/customer/portal/233/group/268/create/341

## 0. Connection from Windows
1. Install the University-approved GlobalProtect VPN following the official link above. The exact VPN portal is in the University guide, not the original account email.
2. Connect with the SU-Net VPN login above and your own VPN password; confirm `Connected`.
3. Open PowerShell/Windows Terminal (or PuTTY/MobaXterm), then:
   ```powershell
   ssh guest1131@tosun.sabanciuniv.edu
   ```
4. On first connection verify the host key against a trusted University source, not merely the unknown SSH prompt. Use the HPC password from your Sine Mete email. Terminal password input is invisible.
5. Change the initial HPC password with `passwd` if it has not already been changed. Do not reuse the VPN password.
6. If direct SSH fails, follow University's quickstart for the `flow.sabanciuniv.edu` jump host or request HPC IT assistance; a jump host is not an alternative to authorized remote access.

## 1. Confirm cluster access and capacity (login node only)
```bash
whoami
hostname
srun -N2 -n2 --partition=short_mdbf --qos=short_mdbf hostname
sinfo -o '%P | availability=%a | limit=%l | nodes=%D | state=%t | gres=%G'
squeue -u "$USER"
scontrol show partition cuda
df -h "$HOME"
```
The email test requests 2 scheduler nodes, not GPUs. It can wait in the queue. **Never run model training on a login node**.

### Prepare repository
```bash
mkdir -p ~/research
cd ~/research
git clone https://github.com/MohammadAmanlou/FrontierMem.git
cd FrontierMem
git switch research/decision-sufficient-memory-2026-11
bash scripts/hpc_inventory.sh | tee hpc_inventory.txt
```
If GitHub cannot be reached from the cluster, transfer a source archive using SFTP/SCP from your trusted computer after connecting VPN. Do not upload secrets, personal data, or model weights to the public repository.

## 2. Verify an ALLOCATED GPU
From the checked-out repository:
```bash
sbatch scripts/hpc_gpu_probe.sbatch
squeue -u "$USER"
```
On completion, read `fm-gpu-probe-<JOBID>.out` and `fm-gpu-probe-<JOBID>.err`.
The job prints the GPU model, memory, CUDA visibility, Python and PyTorch availability.
The template requests `--partition=cuda --qos=cuda --gres=gpu:1` as indicated by University docs and the email. On `InvalidAccount`, `InvalidQOS`, `InvalidPartition`, or `Resources` issues, inspect `scontrol show partition cuda`, `sinfo`, and your assigned association with IT, rather than guessing an account.

## 3. Python/LLM environment, after GPU allocation is verified
Use `module avail` to inspect available Python, CUDA and frameworks. Prefer the HPC-provided version or virtual environment/Apptainer permitted by IT. Downloading Qwen checkpoints and CUDA packages can be large and may require Internet access or a mirror; arrange a quota-aware directory before installation.

A Qwen3-4B experiment uses CUDA GPU resources; start with a tiny forward-only check **inside Slurm**. Do not presume all partitions have the same GPU model or that 4-bit QLoRA will work until CUDA and package compatibility are verified.

For the three-week study, prioritize existing RPEval/CAID checkpoints, then a matched Pointwise/CAID/CAID+S2Pref binary/Dual-Axis run. Unreviewed silver S2Pref training candidates are not approved gold. Official BenchPreS and S2Pref heldout test are never used to optimize training or memory selection.

## 4. Trouble diagnosis
- `ssh: Could not resolve hostname`: verify VPN, DNS, hostname spelling, or ask IT.
- `Connection timed out`: VPN policy/firewall, host health, try University's documented jump-host procedure.
- `Permission denied (publickey,password)`: check **HPC** login name/password (not VPN) and account status.
- `InvalidAccount/InvalidQOS`: show scheduler message to HPC support; do not randomly change ACLs.
- Job `PD (Resources)`: waiting in queue, not necessarily broken.
- Job `OUT_OF_MEMORY` or CUDA OOM: reduce batch/sequence length, increase GPU or RAM request only with quota.
- `torch.cuda.is_available() == False`: ensure commands run inside an allocated GPU job, not the login node.

When opening a support ticket, supply cluster name, username (not password), JobID, selected partition/QoS, exact errors, timestamp, and research advisor. University explicitly prohibits running large workloads on login nodes or storing long-lived research data solely in shared temporary spaces.
