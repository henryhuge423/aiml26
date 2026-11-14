$ErrorActionPreference = 'Stop'
$taskProject = Split-Path -Parent $PSScriptRoot
$taskEnvironment = Join-Path $taskProject '.envs/training'
$taskLogs = Join-Path $taskProject 'runs/expanded_20261113/pipeline'
New-Item -ItemType Directory -Path $taskLogs -Force | Out-Null
$taskLaunchFile = Join-Path $taskLogs 'latest_launch.json'
if (Test-Path -LiteralPath $taskLaunchFile) {
    $taskPrevious = Get-Content -LiteralPath $taskLaunchFile -Raw | ConvertFrom-Json
    $taskExisting = Get-CimInstance Win32_Process -Filter ('ProcessId = ' + $taskPrevious.pid)
    if ($taskExisting -and $taskExisting.CommandLine -like '*run_expansion_experiments_20261113.py*') {
        throw 'The experiment driver is already running.'
    }
}
$env:PATH = (Join-Path $taskEnvironment 'Library/bin') + ';' + (Join-Path $taskEnvironment 'Scripts') + ';' + $taskEnvironment + ';' + $env:PATH
$env:PYTHONNOUSERSITE = '1'
$taskStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$taskStdout = Join-Path $taskLogs ('driver_' + $taskStamp + '.out.log')
$taskStderr = Join-Path $taskLogs ('driver_' + $taskStamp + '.err.log')
$taskScript = Join-Path $taskProject 'tools/run_expansion_experiments_20261113.py'
$taskProcess = Start-Process -FilePath (Join-Path $taskEnvironment 'python.exe') -ArgumentList @('-u', ('"{0}"' -f $taskScript)) -WorkingDirectory $taskProject -WindowStyle Hidden -RedirectStandardOutput $taskStdout -RedirectStandardError $taskStderr -PassThru
$taskLaunch = [ordered]@{ pid = $taskProcess.Id; started_at = (Get-Date).ToUniversalTime().ToString('o'); stdout = $taskStdout; stderr = $taskStderr }
$taskLaunch | ConvertTo-Json | Set-Content -Encoding utf8 -LiteralPath $taskLaunchFile
$taskLaunch | ConvertTo-Json
