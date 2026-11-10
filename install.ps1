param(
    # MaiBot root directory (the folder that contains bot.py).
    [Parameter(Mandatory = $true)][string]$MaiBotPath,
    [switch]$Force
)
$ErrorActionPreference = "Stop"
$source = $PSScriptRoot
# Prefer an existing installation with the same stable manifest ID, including ZIP installs.
if (-not (Test-Path (Join-Path $MaiBotPath "bot.py"))) { throw "MaiBot bot.py not found: $MaiBotPath" }
$existing = @(Get-ChildItem -LiteralPath (Join-Path $MaiBotPath "plugins") -Directory -ErrorAction SilentlyContinue | Where-Object {
    $manifest = Join-Path $_.FullName "_manifest.json"
    if (Test-Path $manifest) {
        try { (Get-Content -LiteralPath $manifest -Raw -Encoding UTF8 | ConvertFrom-Json).id -eq "local.gpt-sovits-tts" } catch { $false }
    }
})
if ($existing.Count -gt 1) { throw "Duplicate GPT-SoVITS installations found. Keep exactly one before updating." }
$target = if ($existing.Count -eq 1) { $existing[0].FullName } else { Join-Path $MaiBotPath "plugins/local.gpt-sovits-tts" }
if ([IO.Path]::GetFullPath($source) -eq [IO.Path]::GetFullPath($target)) { throw "Run this script from the development or release directory, not the installed plugin itself." }
if ((Test-Path $target) -and -not $Force) { throw "Target already exists. Back it up and use -Force to update; config.toml is preserved." }
New-Item -ItemType Directory -Force -Path $target | Out-Null
foreach ($name in @("plugin.py", "tts_client.py", "config_models.py", "setup_service.py", "model_catalog.py", "native_config.py", "role_service.py", "role_library.py", "wizard_progress.py", "auto_voice.py", "emotion_judge.py", "tts_log.py", "_manifest.json", "webui.json", "requirements.txt", "config.example.toml", "README.md", "LICENSE")) {
    Copy-Item -LiteralPath (Join-Path $source $name) -Destination (Join-Path $target $name) -Force
}
$webui = Join-Path $target "webui"
New-Item -ItemType Directory -Force -Path $webui | Out-Null
Copy-Item -LiteralPath (Join-Path $source "webui/setup.html") -Destination (Join-Path $webui "setup.html") -Force
# Plugin logo declared in _manifest.json (display.icon).
$assets = Join-Path $target "assets"
New-Item -ItemType Directory -Force -Path $assets | Out-Null
Copy-Item -LiteralPath (Join-Path $source "assets/maibot-microphone-chibi-256.png") -Destination (Join-Path $assets "maibot-microphone-chibi-256.png") -Force
# Usage documents and their screenshots.
$docs = Join-Path $target "docs"
New-Item -ItemType Directory -Force -Path (Join-Path $docs "images") | Out-Null
Copy-Item -Path (Join-Path $source "docs/*.md") -Destination $docs -Force
Copy-Item -Path (Join-Path $source "docs/images/*.png") -Destination (Join-Path $docs "images") -Force
$config = Join-Path $target "config.toml"
if (-not (Test-Path $config)) { Copy-Item (Join-Path $source "config.example.toml") $config }
Write-Host "Installed to: $target"
Write-Host "Reload the plugin in MaiBot, then open the 'GPT-SoVITS 语音设置' workspace and follow the five setup steps."
Write-Host "No service was restarted. An existing config.toml is kept."
