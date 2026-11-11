$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$interpreter = Join-Path $projectRoot '.envs\training\python.exe'
$stateRoot = Join-Path $projectRoot 'data\audit\eyeq_acquisition_20261110'
$env:PYTHONNOUSERSITE = '1'
$env:PATH = (Join-Path $projectRoot '.envs\training\Library\bin') + ';' + $env:PATH
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdout = Join-Path $stateRoot "import_$stamp.out.log"
$stderr = Join-Path $stateRoot "import_$stamp.err.log"
$process = Start-Process -FilePath $interpreter -ArgumentList @('-u', 'tools/import_eyeq_download.py') -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
$process.PriorityClass = 'BelowNormal'
$launch = [ordered]@{pid=$process.Id;started_at=(Get-Date).ToUniversalTime().ToString('o');stdout=$stdout;stderr=$stderr}
$launch | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $stateRoot 'import_launch.json') -Encoding utf8
$launch | ConvertTo-Json
