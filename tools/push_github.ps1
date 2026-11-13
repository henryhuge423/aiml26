param(
    [string]$Remote = 'origin',
    [string]$Branch = 'main',
    [switch]$Direct
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$remoteUrl = & git -C $projectRoot remote get-url $Remote
if ($LASTEXITCODE -ne 0 -or $remoteUrl -notmatch '^https://github\.com/henryhuge423/aiml26(?:\.git)?/?$') {
    throw 'The remote does not match the configured Eye Sentinel GitHub repository.'
}
$tokenFile = Join-Path $projectRoot '.secrets/github.token'
if (-not (Test-Path -LiteralPath $tokenFile -PathType Leaf)) {
    throw 'GitHub token file was not found in .secrets.'
}

$helperFile = Join-Path ([System.IO.Path]::GetTempPath()) ('eye-sentinel-askpass-' + [guid]::NewGuid().ToString('N') + '.cmd')
$oldAskPass = $env:GIT_ASKPASS
$oldTokenFile = $env:EYE_SENTINEL_TOKEN_FILE
$oldPrompt = $env:GIT_TERMINAL_PROMPT
try {
    @'
@echo off
echo %~1 | findstr /I "username" >nul
if %errorlevel%==0 (echo x-access-token) else (powershell.exe -NoProfile -NonInteractive -Command "[Console]::Out.Write([IO.File]::ReadAllText($env:EYE_SENTINEL_TOKEN_FILE).Trim())")
'@ | Set-Content -LiteralPath $helperFile -Encoding ascii
    $env:GIT_ASKPASS = $helperFile
    $env:EYE_SENTINEL_TOKEN_FILE = $tokenFile
    $env:GIT_TERMINAL_PROMPT = '0'
    $gitArgs = @('-c', 'credential.helper=')
    if ($Direct) { $gitArgs += @('-c', 'http.proxy=', '-c', 'https.proxy=') }
    $gitArgs += @('push', $Remote, $Branch)
    Push-Location -LiteralPath $projectRoot
    try {
        & git @gitArgs
        if ($LASTEXITCODE -ne 0) { throw "Git push failed with exit code $LASTEXITCODE." }
    } finally { Pop-Location }
} finally {
    $env:GIT_ASKPASS = $oldAskPass
    $env:EYE_SENTINEL_TOKEN_FILE = $oldTokenFile
    $env:GIT_TERMINAL_PROMPT = $oldPrompt
    Remove-Item -LiteralPath $helperFile -Force -ErrorAction SilentlyContinue
}
