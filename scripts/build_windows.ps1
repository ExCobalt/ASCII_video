param([string]$PythonPath)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $PythonPath) {
    $PythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw 'Create .venv and install the locked build dependencies before building.'
}
& $PythonPath -c "import sys, struct; assert sys.platform == 'win32' and struct.calcsize('P') == 8, 'Build requires Windows x64 Python'"
if ($LASTEXITCODE -ne 0) { throw 'Build requires Windows x64 Python.' }

Push-Location $projectRoot
try {
    # Optional builds must not accidentally reuse an older DLL after a compiler error.
    Get-ChildItem -LiteralPath (Join-Path $projectRoot 'src') -Filter '_glyphs*.pyd' -File |
        ForEach-Object { Remove-Item -LiteralPath $_.FullName }
    Get-ChildItem -LiteralPath (Join-Path $projectRoot 'src') -Filter '_diffusion*.pyd' -File |
        ForEach-Object { Remove-Item -LiteralPath $_.FullName }
    $nativeBuild = Join-Path $projectRoot ('build\native-' + [guid]::NewGuid().ToString('N'))
    & $PythonPath setup.py build_ext --inplace --force `
        --build-lib (Join-Path $nativeBuild 'lib') --build-temp (Join-Path $nativeBuild 'temp')
    if ($LASTEXITCODE -ne 0) { throw 'Native extension build failed.' }
    & $PythonPath -c "import sys; from pathlib import Path; sys.path.insert(0, 'src'); import raster, diffusion; assert raster._compose is not None and raster._compose_planes is not None and diffusion._native_quantize is not None, 'Release requires both native accelerators'; assert all(Path(sys.modules[name].__file__).resolve().parent == Path('src').resolve() for name in ('_glyphs', '_diffusion')), 'Unexpected native module outside src'; print('Native raster and diffusion acceleration: enabled')"
    if ($LASTEXITCODE -ne 0) { throw 'Required native accelerators are unavailable.' }
} finally {
    Pop-Location
}

# The contrib hook collects av.libs; our hook selects x64 PortAudio without ASIO.
# https://github.com/pyinstaller/pyinstaller-hooks-contrib/blob/master/_pyinstaller_hooks_contrib/stdhooks/hook-av.py
# https://github.com/pyinstaller/pyinstaller-hooks-contrib/blob/master/_pyinstaller_hooks_contrib/stdhooks/hook-sounddevice.py
$previousCache = $env:PYINSTALLER_CONFIG_DIR
$previousPath = $env:PATH
$env:PYINSTALLER_CONFIG_DIR = Join-Path $projectRoot 'build\pyinstaller-cache'
try {
    & $PythonPath (Join-Path $projectRoot 'scripts\collect_licenses.py') `
        --output (Join-Path $projectRoot 'build\release-licenses')
    if ($LASTEXITCODE -ne 0) { throw 'License collection failed.' }
    # Do not collect DLLs from unrelated applications on the invoking shell's PATH.
    $basePrefix = & $PythonPath -c "import sys; print(sys.base_prefix)"
    if ($LASTEXITCODE -ne 0) { throw 'Unable to locate the Python runtime.' }
    $windowsDirectory = [Environment]::GetFolderPath([Environment+SpecialFolder]::Windows)
    $systemDirectory = [Environment]::GetFolderPath([Environment+SpecialFolder]::System)
    $env:PATH = @(
        (Split-Path -Parent (Resolve-Path -LiteralPath $PythonPath).Path),
        $basePrefix,
        (Join-Path $basePrefix 'DLLs'),
        $systemDirectory,
        $windowsDirectory
    ) -join ';'
    & $PythonPath -m PyInstaller --noconfirm --onefile --windowed --noupx `
        --name ascii-video --paths (Join-Path $projectRoot 'src') `
        --exclude-module _ssl --exclude-module _hashlib --exclude-module Cython --exclude-module cython `
        --additional-hooks-dir (Join-Path $projectRoot 'scripts\pyinstaller_hooks') `
        --runtime-hook (Join-Path $projectRoot 'scripts\pyinstaller_windows_runtime.py') `
        --hidden-import PIL._tkinter_finder `
        --icon (Join-Path $projectRoot 'assets\ascii-video.ico') `
        --add-data "$(Join-Path $projectRoot 'assets');assets" `
        --add-data "$(Join-Path $projectRoot 'THIRD_PARTY_NOTICES.md');." `
        --add-data "$(Join-Path $projectRoot 'build\release-licenses');licenses" `
        --workpath (Join-Path $projectRoot 'build\windows') `
        --specpath (Join-Path $projectRoot 'build\windows') `
        --distpath (Join-Path $projectRoot 'dist') `
        (Join-Path $projectRoot 'src\main.py')
    if ($LASTEXITCODE -ne 0) { throw 'PyInstaller build failed.' }
    $executable = Join-Path $projectRoot 'dist\ascii-video.exe'
    & $executable --help | Out-Host
    if ($LASTEXITCODE -ne 0) { throw 'Executable --help failed.' }
    $expectedVersion = & $PythonPath -c "import sys,tomllib; print('ascii-video',tomllib.load(open(sys.argv[1],'rb'))['project']['version'])" (Join-Path $projectRoot 'pyproject.toml')
    $version = & $executable --version | Out-String
    if ($LASTEXITCODE -ne 0 -or $version.Trim() -ne $expectedVersion) { throw "Unexpected executable version: $version" }
    Write-Output $version.Trim()
    & $PythonPath (Join-Path $projectRoot 'scripts\package_windows.py') --check-executable
    if ($LASTEXITCODE -ne 0) { throw 'Executable libraries or license notices differ from the reviewed inventory.' }
    & $PythonPath (Join-Path $projectRoot 'tests\check_packaged_export.py') --executable $executable
    if ($LASTEXITCODE -ne 0) { throw 'Executable silent MP4 export failed.' }
    Get-Item -LiteralPath $executable | Select-Object FullName, Length
} finally {
    $env:PYINSTALLER_CONFIG_DIR = $previousCache
    $env:PATH = $previousPath
}
