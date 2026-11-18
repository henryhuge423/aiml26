$ErrorActionPreference = 'Stop'
$taskProject = Split-Path -Parent $PSScriptRoot
$taskEnvironment = Join-Path $taskProject '.envs/training'
$taskLogs = Join-Path $taskProject 'runs/followup_20261115/pipeline'
New-Item -ItemType Directory -Path $taskLogs -Force | Out-Null
$taskLaunchFile = Join-Path $taskLogs 'latest_launch.json'
$taskDrivers = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'run_(followup_experiments_20261115|expansion_experiments_20261113|training_pipeline)\.py|train_model\.py' }
if ($taskDrivers) { throw 'A disease training driver or worker is already running.' }
$env:PATH = (Join-Path $taskEnvironment 'Library/bin') + ';' + (Join-Path $taskEnvironment 'Scripts') + ';' + $taskEnvironment + ';' + $env:PATH
$env:PYTHONNOUSERSITE = '1'
$taskStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$taskStdout = Join-Path $taskLogs ('driver_' + $taskStamp + '.out.log')
$taskStderr = Join-Path $taskLogs ('driver_' + $taskStamp + '.err.log')
$taskScript = Join-Path $taskProject 'tools/run_followup_experiments_20261115.py'
$taskProcess = Start-Process -FilePath (Join-Path $taskEnvironment 'python.exe') -ArgumentList @('-u', ('"{0}"' -f $taskScript)) -WorkingDirectory $taskProject -WindowStyle Hidden -RedirectStandardOutput $taskStdout -RedirectStandardError $taskStderr -PassThru
$taskLaunch = [ordered]@{ pid = $taskProcess.Id; started_at = (Get-Date).ToUniversalTime().ToString('o'); stdout = $taskStdout; stderr = $taskStderr }
$taskLaunch | ConvertTo-Json | Set-Content -Encoding utf8 -LiteralPath $taskLaunchFile
$taskLaunch | ConvertTo-Json
