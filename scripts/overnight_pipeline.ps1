### Overnight pipeline:
###   1. PURE intra (60/40 RhythmFormer protocol) - 약 2시간
###   2. PURE→UBFC cross-dataset - 약 4시간
###   3. Bland-Altman + scatter plots 생성 (3 results: intra UBFC, intra PURE, cross)

$ErrorActionPreference = 'Continue'
$env:PYTHONIOENCODING = 'utf-8'

# ----- 1. PURE intra 60/40 -----
Write-Output "===== [1/3] PURE intra 60/40 시작 $(Get-Date) ====="
if (-not (Test-Path results/intra_pure_biphysformer)) { New-Item -ItemType Directory -Path results/intra_pure_biphysformer -Force | Out-Null }
python scripts/run_intra_pure_biphysformer.py 2>&1 | Out-File -FilePath results/intra_pure_biphysformer/run.log -Encoding utf8
Write-Output "===== [1/3] PURE intra 완료 $(Get-Date) ====="

# ----- 2. PURE→UBFC cross-dataset -----
Write-Output "===== [2/3] PURE->UBFC cross 시작 $(Get-Date) ====="
Remove-Item results/cross_biphysformer/log.txt,results/cross_biphysformer/run.log,results/cross_biphysformer/checkpoints,results/cross_biphysformer/waveforms -Recurse -Force -ErrorAction SilentlyContinue
# Modify EXPERIMENTS list to ONLY PURE→UBFC (skip UBFC→PURE)
python -c @'
import re
p = "scripts/run_cross_biphysformer.py"
with open(p, "r", encoding="utf-8") as f:
    code = f.read()
# Comment out UBFC→PURE line if present, keep only PURE→UBFC
new_code = re.sub(
    r"(\s+\('UBFC-rPPG', UBFC_PATH, 'PURE', PURE_PATH\),)",
    r"#\1  # disabled for overnight pipeline",
    code, count=1)
with open(p, "w", encoding="utf-8") as f:
    f.write(new_code)
print("modified script to PURE->UBFC only")
'@
python scripts/run_cross_biphysformer.py 2>&1 | Out-File -FilePath results/cross_biphysformer/run.log -Encoding utf8
Write-Output "===== [2/3] PURE->UBFC cross 완료 $(Get-Date) ====="

# ----- 3. Plots 생성 -----
Write-Output "===== [3/3] Plots 생성 시작 $(Get-Date) ====="

# Intra UBFC E4 (이미 best, 60/40)
python scripts/generate_plots.py `
    --ckpt results/intra_ubfc_biphysformer/checkpoints/UBFC-rPPG_to_UBFC-rPPG_epoch4.pt `
    --test_dataset UBFC-rPPG --split_range 0.6,1.0 `
    --name intra_UBFC_epoch4 `
    --out_dir results/plots/biphysformer 2>&1 | Out-File -FilePath results/plots/log_ubfc.txt -Encoding utf8

# Intra PURE - best epoch will be selected after training
# Default: try epoch 4, 7, 9 (these were good in 60/20/20 run)
for ($e in 1..10) {
    $ck = "results/intra_pure_biphysformer/checkpoints/PURE_to_PURE_epoch$e.pt"
    if (Test-Path $ck) {
        python scripts/generate_plots.py `
            --ckpt $ck --test_dataset PURE --split_range 0.6,1.0 `
            --name "intra_PURE_epoch$e" `
            --out_dir results/plots/biphysformer 2>&1 | Out-File -FilePath "results/plots/log_pure_e$e.txt" -Encoding utf8
    }
}

# Cross PURE→UBFC - all epochs
for ($e in 1..10) {
    $ck = "results/cross_biphysformer/checkpoints/PURE_to_UBFC-rPPG_epoch$e.pt"
    if (Test-Path $ck) {
        python scripts/generate_plots.py `
            --ckpt $ck --test_dataset UBFC-rPPG `
            --name "cross_PURE_to_UBFC_epoch$e" `
            --out_dir results/plots/biphysformer 2>&1 | Out-File -FilePath "results/plots/log_cross_e$e.txt" -Encoding utf8
    }
}

Write-Output "===== ALL DONE $(Get-Date) ====="
