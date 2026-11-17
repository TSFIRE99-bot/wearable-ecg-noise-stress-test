# Convert the submission Word files to PDF with Microsoft Word (Windows only).
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\docx_to_pdf.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\docx_to_pdf.ps1 -Folder paper\submission\physiol_meas
# Every .docx in the folder (except Word lock files) is exported next to itself as .pdf.
param([string]$Folder = "paper\submission\physiol_meas")
$root = Split-Path -Parent $PSScriptRoot
$dir = if ([System.IO.Path]::IsPathRooted($Folder)) { $Folder } else { Join-Path $root $Folder }
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    foreach ($f in Get-ChildItem -Path $dir -Filter "*.docx" | Where-Object { $_.Name -notlike '~$*' }) {
        $src = [string]$f.FullName
        $dst = [string]([System.IO.Path]::ChangeExtension($f.FullName, ".pdf"))
        $doc = $word.Documents.Open($src, $false, $true)
        $doc.ExportAsFixedFormat($dst, 17)
        $doc.Close([ref]0)
        Write-Output "wrote $dst"
    }
} finally {
    $word.Quit()
}
