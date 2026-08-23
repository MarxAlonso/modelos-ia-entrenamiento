param(
    [string]$Prompt = "",
    [int]$Pasos = 5000
)
$rutaSrc = ($PSScriptRoot -replace '^([A-Za-z]):', '/mnt/$1' -replace '\\', '/').ToLower()
$scriptBash = "$rutaSrc/gpu_env.sh"
$scriptPy = "$rutaSrc/lenguaje_mini_gpt.py"
if ($Prompt -ne "") {
    wsl -d Ubuntu-22.04 -u root -- bash $scriptBash $scriptPy --generar $Prompt
} else {
    wsl -d Ubuntu-22.04 -u root -- bash $scriptBash $scriptPy --pasos $Pasos
}
