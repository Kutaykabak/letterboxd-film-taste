# Renders the share cards in share.html to PNG and PDF with headless Chrome.
# Usage (from the project root):  powershell -ExecutionPolicy Bypass -File reports\share\render_share.ps1
# Needs: Google Chrome, an internet connection (Google Fonts), and site/data.js + site/catalog.js.

$chrome = "C:\Program Files\Google\Chrome\Application\chrome.exe"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$url = "file:///" + ($here -replace '\\', '/') + "/share.html"
# separate profile so this never touches an open Chrome window
$profile = Join-Path $env:TEMP "letterboxd-share-chrome"
$common = @("--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
            "--user-data-dir=$profile", "--virtual-time-budget=10000", "--run-all-compositor-stages-before-draw")

function Render-Png($card, $w, $h, $scale, $out) {
    & $chrome @common "--force-device-scale-factor=$scale" "--window-size=$w,$h" "--screenshot=$here\$out" "$url`?card=$card" 2>$null | Out-Null
    Write-Output "$out written"
}

Render-Png "og"    1200 630  1 "og.png"
Render-Png "og-tr" 1200 630  1 "og-tr.png"
Render-Png "post"  1080 1350 2 "linkedin-post.png"
Render-Png "model" 1080 1350 2 "model-vs-me.png"

& $chrome @common "--no-pdf-header-footer" "--print-to-pdf=$here\linkedin-carousel.pdf" "$url`?card=carousel" 2>$null | Out-Null
Write-Output "linkedin-carousel.pdf written"
