param(
    [string]$Target = "",
    [string]$Arch = "sm_86"
)
$ErrorActionPreference = "Stop"
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
if (!(Test-Path $vswhere)) { throw "Visual Studio Build Tools not found" }
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (!$vs) { throw "Visual Studio C++ workload not found" }
$dev = Join-Path $vs "Common7\Tools\VsDevCmd.bat"
$nvcc = (Get-Command nvcc).Source
$targetArg = if ($Target) { " $Target" } else { "" }
$cmd = "call `"$dev`" -arch=x64 && make$targetArg NVCC=`"$nvcc`" ARCH=-arch=$Arch"
cmd.exe /d /s /c $cmd
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
