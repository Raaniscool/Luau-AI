<#
.SYNOPSIS
Creates or replaces a DukeOTR desktop shortcut for a portable release.

.DESCRIPTION
This is a one-time setup helper, not a normal launch path.  The resulting Desktop\DukeOTR.lnk
points directly to DukeOTR.exe and uses the embedded application icon.  It does not start the
application, contact Ollama, or modify any model files.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$TargetPath,

    [string]$ShortcutPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$target = [System.IO.Path]::GetFullPath($TargetPath)
if (-not (Test-Path -LiteralPath $target -PathType Leaf)) {
    throw "DukeOTR executable was not found: $target"
}
if ([System.IO.Path]::GetFileName($target) -ine "DukeOTR.exe") {
    throw "TargetPath must point to the packaged DukeOTR.exe, not a directory or another executable: $target"
}

if ([string]::IsNullOrWhiteSpace($ShortcutPath)) {
    $desktop = [Environment]::GetFolderPath([Environment+SpecialFolder]::DesktopDirectory)
    if ([string]::IsNullOrWhiteSpace($desktop)) {
        throw "Windows did not report a Desktop directory for the current user. Pass -ShortcutPath explicitly."
    }
    $ShortcutPath = Join-Path $desktop "DukeOTR.lnk"
}

$shortcut = [System.IO.Path]::GetFullPath($ShortcutPath)
$shortcutDirectory = Split-Path -Parent $shortcut
if (-not (Test-Path -LiteralPath $shortcutDirectory -PathType Container)) {
    New-Item -ItemType Directory -Path $shortcutDirectory -Force | Out-Null
}

$wsh = New-Object -ComObject WScript.Shell
$link = $wsh.CreateShortcut($shortcut)
$link.TargetPath = $target
$link.WorkingDirectory = Split-Path -Parent $target
$link.IconLocation = "$target,0"
$link.Description = "DukeOTR local AI workspace"
$link.Save()

Write-Host "Created DukeOTR desktop shortcut: $shortcut"
Write-Host "Target: $target"
