# Convert the submission Word files to PDF with Microsoft Word (Windows only).
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\docx_to_pdf.ps1
$dir = Join-Path (Split-Path -Parent $PSScriptRoot) "paper\submission"
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    foreach ($name in @("manuscript", "supplementary")) {
        $src = [string](Join-Path $dir "$name.docx")
        $dst = [string](Join-Path $dir "$name.pdf")
        $doc = $word.Documents.Open($src, $false, $true)
        $doc.ExportAsFixedFormat($dst, 17)
        $doc.Close([ref]0)
        Write-Output "wrote $dst"
    }
} finally {
    $word.Quit()
}
