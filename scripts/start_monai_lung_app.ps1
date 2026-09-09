$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PythonExe = if ($env:CONDA_PREFIX) {
    Join-Path $env:CONDA_PREFIX "python.exe"
} else {
    (Get-Command python -ErrorAction Stop).Source
}

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable not found. Activate the lung_app environment first."
}

Set-Location -LiteralPath $ProjectRoot

& $PythonExe -m monailabel.main start_server `
    --app "monai_apps\lung_monai_app" `
    --studies "data\qc\fail_qc\images" `
    --conf "models" "all" `
    --conf "anatomy_device" "cpu" `
    --conf "anatomy_num_threads" "2" `
    --conf "lesion_device" "cpu" `
    --conf "lesion_num_threads" "2" `
    --conf "lesion_shift_pixels" "64" `
    --conf "lesion_mask_logit_threshold" "1.0" `
    --conf "lesion_flip_display_vertical" "false" `
    --conf "cxformer_device" "cpu" `
    --conf "cxformer_num_threads" "2"
