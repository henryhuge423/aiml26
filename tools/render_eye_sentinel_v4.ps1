$ErrorActionPreference = 'Stop'
$projectDir = Split-Path $PSScriptRoot -Parent
$previewDir = Join-Path $PSScriptRoot 'build_reports/v4_preview'
New-Item -ItemType Directory -Path $previewDir -Force | Out-Null
$pptApp = New-Object -ComObject PowerPoint.Application
try {
    $deck = $pptApp.Presentations.Open((Join-Path $projectDir 'archive/proposal_versions/Eye_Sentinel_Project_Proposal_v4.pptx'), $true, $false, $false)
    $deck.Export($previewDir, 'PNG', 1600, 900)
    $deck.SaveAs((Join-Path $previewDir 'proposal_preview.pdf'), 32)
    $overflow = @()
    foreach ($slide in $deck.Slides) {
        foreach ($shape in $slide.Shapes) {
            if ($shape.HasTextFrame -eq -1 -and $shape.TextFrame.HasText -eq -1) {
                if ($shape.TextFrame2.TextRange.BoundHeight -gt ($shape.Height + 2)) {
                    $overflow += [pscustomobject]@{slide=$slide.SlideIndex; text=$shape.TextFrame.TextRange.Text}
                }
            }
        }
    }
    ConvertTo-Json -InputObject $overflow -Depth 3 | Set-Content -LiteralPath (Join-Path $previewDir 'overflow.json') -Encoding utf8
    Write-Output ('Overflow count: ' + $overflow.Count)
    $deck.Close()
} finally {
    $pptApp.Quit()
}
