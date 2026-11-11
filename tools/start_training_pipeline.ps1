$ErrorActionPreference = 'Stop'
$taskProject = Split-Path -Parent $PSScriptRoot
$taskEnvironment = Join-Path $taskProject '.envs/training'
$env:PATH = (Join-Path $taskEnvironment 'Library/bin') + ';' + (Join-Path $taskEnvironment 'Scripts') + ';' + $taskEnvironment + ';' + $env:PATH
$env:PYTHONNOUSERSITE = '1'
$taskLogs = Join-Path $taskProject 'runs/formal_v3/pipeline'
New-Item -ItemType Directory -Path $taskLogs -Force | Out-Null
$taskStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$taskStdout = Join-Path $taskLogs ('driver_' + $taskStamp + '.out.log')
$taskStderr = Join-Path $taskLogs ('driver_' + $taskStamp + '.err.log')
$taskScript = Join-Path $taskProject 'tools/run_training_pipeline.py'
$taskProcess = Start-Process -FilePath (Join-Path $taskEnvironment 'python.exe') -ArgumentList @('-u', ('"{0}"' -f $taskScript)) -WorkingDirectory $taskProject -WindowStyle Hidden -RedirectStandardOutput $taskStdout -RedirectStandardError $taskStderr -PassThru
$taskLaunch = [ordered]@{ pid = $taskProcess.Id; started_at = (Get-Date).ToUniversalTime().ToString('o'); stdout = $taskStdout; stderr = $taskStderr }
$taskLaunch | ConvertTo-Json | Set-Content -Encoding utf8 -LiteralPath (Join-Path $taskLogs 'latest_launch.json')
$taskLaunch | ConvertTo-Json
