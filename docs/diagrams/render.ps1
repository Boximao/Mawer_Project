# Renders every .svg in this folder to a 2x PNG using headless Chrome.
# Usage:  powershell -ExecutionPolicy Bypass -File docs\diagrams\render.ps1

$dir = $PSScriptRoot
$chrome = "C:\Program Files\Google\Chrome\Application\chrome.exe"
if (-not (Test-Path $chrome)) { $chrome = "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" }
if (-not (Test-Path $chrome)) { throw "No Chrome or Edge found for rendering." }

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)

foreach ($svgFile in Get-ChildItem -Path $dir -Filter *.svg) {
    $svg = [System.IO.File]::ReadAllText($svgFile.FullName, [System.Text.Encoding]::UTF8)
    if ($svg -notmatch 'width="(\d+)"\s+height="(\d+)"') { throw "No width/height on $($svgFile.Name)" }
    $w = $Matches[1]; $h = $Matches[2]

    # Chrome screenshots an HTML document, so inline the SVG into a zero-margin page.
    $tmp = Join-Path $dir "_render_tmp.html"
    $html = "<!doctype html><html><head><meta charset=""utf-8""><style>html,body{margin:0;padding:0;background:#f5f7fb;}svg{display:block;}</style></head><body>$svg</body></html>"
    [System.IO.File]::WriteAllText($tmp, $html, $utf8NoBom)

    $png = Join-Path $dir ($svgFile.BaseName + ".png")
    $url = "file:///" + ($tmp -replace '\\', '/')
    & $chrome --headless=new --disable-gpu --hide-scrollbars --force-device-scale-factor=2 `
        --window-size="$w,$h" --screenshot="$png" $url 2>&1 | Out-Null
    Remove-Item $tmp -Force
    Write-Host "rendered $($svgFile.BaseName).png  ($w x $h @2x)"
}
