<#
.SYNOPSIS
Builds the portable Windows DukeOTR desktop release.

.DESCRIPTION
Uses PyInstaller against the existing tkinter application; it does not rewrite the UI, contact
Ollama, or bundle/download Qwen3-4B.  The output is a self-contained *folder* with
DukeOTR.exe, which can be copied as a unit to a permanent user location.  Passing
-Install -CreateDesktopShortcut deploys that folder under the current user's LocalAppData and
creates Desktop\DukeOTR.lnk for ordinary double-click launching.
#>

[CmdletBinding()]
param(
    # Paths may be absolute or relative to the repository root.
    [string]$OutputDirectory = "dist",
    [string]$WorkDirectory = "build\pyinstaller",
    [string]$BuildVenvDirectory = ".venv-windows-build",
    [switch]$SkipTests,
    [switch]$Install,
    [switch]$CreateDesktopShortcut,
    [string]$InstallDirectory
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($env:OS -ne "Windows_NT") {
    throw "DukeOTR's PyInstaller release must be built on Windows. PyInstaller does not cross-compile DukeOTR.exe from this host."
}

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$desktopEntrypoint = Join-Path $repoRoot "desktop_app\__main__.py"
$specFile = Join-Path $repoRoot "packaging\dukeotr.spec"
$requirementsFile = Join-Path $repoRoot "requirements\windows-build.txt"
$releaseVerifier = Join-Path $repoRoot "scripts\verify_windows_release.py"
$shortcutScript = Join-Path $repoRoot "scripts\create_desktop_shortcut.ps1"

foreach ($required in @($desktopEntrypoint, $specFile, $requirementsFile, $releaseVerifier, $shortcutScript)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required release file is missing: $required"
    }
}

function Resolve-RepositoryPath {
    param([Parameter(Mandatory = $true)][string]$PathValue)
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return [System.IO.Path]::GetFullPath($PathValue)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $repoRoot $PathValue))
}

function Invoke-CheckedProcess {
    param(
        [Parameter(Mandatory = $true)][string]$Executable,
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [Parameter(Mandatory = $true)][string]$Description
    )

    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE."
    }
}

$pyLauncher = Get-Command py -ErrorAction SilentlyContinue
$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if ($null -ne $pyLauncher) {
    $bootstrapPython = $pyLauncher.Source
    $bootstrapArgs = @("-3")
} elseif ($null -ne $pythonCommand) {
    $bootstrapPython = $pythonCommand.Source
    $bootstrapArgs = @()
} else {
    throw "Python 3 was not found. Install CPython 3.10+ with Tcl/Tk, then rerun this build script."
}

$outputRoot = Resolve-RepositoryPath $OutputDirectory
$workRoot = Resolve-RepositoryPath $WorkDirectory
$buildVenv = Resolve-RepositoryPath $BuildVenvDirectory
if ($workRoot -ieq [System.IO.Path]::GetPathRoot($workRoot)) {
    throw "Refusing to clean a filesystem root as WorkDirectory: $workRoot"
}
$builderPython = Join-Path $buildVenv "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $builderPython -PathType Leaf)) {
    Write-Host "Creating isolated release-build environment: $buildVenv"
    Invoke-CheckedProcess -Executable $bootstrapPython -Arguments ($bootstrapArgs + @("-m", "venv", $buildVenv)) -Description "Python virtual environment creation"
}

$pythonSupported = (& $builderPython -c "import sys; print(int(sys.version_info >= (3, 10)))").Trim()
if ($LASTEXITCODE -ne 0 -or $pythonSupported -ne "1") {
    throw "Use CPython 3.10 or newer to build DukeOTR."
}
$pythonBits = (& $builderPython -c "import struct; print(struct.calcsize('P') * 8)").Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Could not inspect the release-build Python interpreter."
}
if ($pythonBits -ne "64") {
    throw "Use a 64-bit CPython interpreter to build the 64-bit DukeOTR Windows release; this interpreter is $pythonBits-bit."
}
$tkVersion = (& $builderPython -c "import tkinter; print(tkinter.TkVersion)").Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Tcl/Tk is unavailable in this Python installation. Rerun the CPython installer, enable Tcl/Tk, then rebuild DukeOTR."
}
Write-Host "Using CPython with Tcl/Tk $tkVersion ($pythonBits-bit) for the portable release"

Write-Host "Installing pinned build-only dependency from $requirementsFile"
Invoke-CheckedProcess -Executable $builderPython -Arguments @("-m", "pip", "install", "--disable-pip-version-check", "--requirement", $requirementsFile) -Description "PyInstaller installation"

Push-Location $repoRoot
try {
    if (-not $SkipTests) {
        Write-Host "Running desktop and packaging tests before release assembly"
        Invoke-CheckedProcess -Executable $builderPython -Arguments @("-m", "unittest", "tests.test_desktop_core", "tests.test_windows_packaging", "-v") -Description "Desktop release tests"
    }

    $packageDirectory = Join-Path $outputRoot "DukeOTR"
    if (Test-Path -LiteralPath $packageDirectory) {
        Remove-Item -LiteralPath $packageDirectory -Recurse -Force
    }
    if (Test-Path -LiteralPath $workRoot) {
        Remove-Item -LiteralPath $workRoot -Recurse -Force
    }
    New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
    New-Item -ItemType Directory -Path $workRoot -Force | Out-Null

    Write-Host "Building portable DukeOTR.exe with the checked-in PyInstaller recipe"
    Invoke-CheckedProcess -Executable $builderPython -Arguments @(
        "-m", "PyInstaller", "--noconfirm", "--clean",
        "--distpath", $outputRoot,
        "--workpath", $workRoot,
        $specFile
    ) -Description "PyInstaller packaging"

    Invoke-CheckedProcess -Executable $builderPython -Arguments @(
        $releaseVerifier, "--package-dir", $packageDirectory
    ) -Description "Portable release layout verification"

    $packageExecutable = Join-Path $packageDirectory "DukeOTR.exe"
    if (-not (Test-Path -LiteralPath $packageExecutable -PathType Leaf)) {
        throw "PyInstaller returned success but DukeOTR.exe is missing: $packageExecutable"
    }

    # A desktop shortcut should target a stable copied release, not a potentially moved/deleted
    # repository build directory. Requesting one therefore also requests deployment.
    if ($CreateDesktopShortcut) {
        $Install = $true
    }

    if ($Install) {
        if ([string]::IsNullOrWhiteSpace($InstallDirectory)) {
            $localAppData = [Environment]::GetFolderPath([Environment+SpecialFolder]::LocalApplicationData)
            if ([string]::IsNullOrWhiteSpace($localAppData)) {
                throw "Windows did not report LocalAppData. Pass -InstallDirectory explicitly."
            }
            $destination = Join-Path $localAppData "Programs\DukeOTR"
        } else {
            $destination = Resolve-RepositoryPath $InstallDirectory
        }
        $destination = [System.IO.Path]::GetFullPath($destination)
        $destinationLeaf = Split-Path -Leaf ($destination.TrimEnd([char[]]@('\', '/')))
        if ($destination -ieq [System.IO.Path]::GetPathRoot($destination) -or $destinationLeaf -ine "DukeOTR") {
            throw "For safety, InstallDirectory must be a non-root folder named DukeOTR (for example D:\\Apps\\DukeOTR): $destination"
        }

        if (Test-Path -LiteralPath $destination) {
            Remove-Item -LiteralPath $destination -Recurse -Force
        }
        New-Item -ItemType Directory -Path $destination -Force | Out-Null
        Get-ChildItem -LiteralPath $packageDirectory -Force | Copy-Item -Destination $destination -Recurse -Force
        $installedExecutable = Join-Path $destination "DukeOTR.exe"
        if (-not (Test-Path -LiteralPath $installedExecutable -PathType Leaf)) {
            throw "Portable release copy did not contain DukeOTR.exe: $installedExecutable"
        }
        Write-Host "Installed portable DukeOTR folder: $destination"

        if ($CreateDesktopShortcut) {
            & $shortcutScript -TargetPath $installedExecutable
            if ($LASTEXITCODE -ne 0) {
                throw "Desktop shortcut creation failed with exit code $LASTEXITCODE."
            }
        }
    }

    Write-Host ""
    Write-Host "Release build complete."
    Write-Host "Build output: $packageDirectory"
    Write-Host "Executable: $packageExecutable"
    if ($Install) {
        Write-Host "Portable installation: $destination"
    } else {
        Write-Host "To deploy it elsewhere, copy the entire DukeOTR folder, not DukeOTR.exe alone."
        Write-Host "To create a shortcut later: .\scripts\create_desktop_shortcut.ps1 -TargetPath '$packageExecutable'"
    }
    Write-Host "No Ollama model, conversation history, training data, or checkpoint was included in this package."
} finally {
    Pop-Location
}
