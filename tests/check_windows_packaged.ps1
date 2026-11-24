param(
    [Parameter(Mandatory)][string]$Executable,
    [Parameter(Mandatory)][string]$InputVideo,
    [Parameter(Mandatory)][string]$OutputDirectory,
    [string]$ExpectedVersion
)

# Run in an interactive Windows guest with a short video containing audio.
# Only the terminal player can open an audio device; that invocation is muted.
$ErrorActionPreference = 'Stop'
$exe = (Resolve-Path -LiteralPath $Executable).Path
$video = (Resolve-Path -LiteralPath $InputVideo).Path
$out = (New-Item -ItemType Directory -Force -Path $OutputDirectory).FullName

function Invoke-PackagedCommand([string[]]$CommandArgs) {
    $start = New-Object Diagnostics.ProcessStartInfo
    $start.FileName = $exe
    $start.Arguments = ($CommandArgs | ForEach-Object { '"' + $_ + '"' }) -join ' '
    $start.WorkingDirectory = $out
    $start.UseShellExecute = $false
    $start.RedirectStandardInput = $true
    $start.RedirectStandardOutput = $true
    $start.RedirectStandardError = $true
    $process = [Diagnostics.Process]::Start($start)
    try {
        $process.StandardInput.Close()
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        if (-not $process.WaitForExit(120000)) { throw "Timeout: $CommandArgs" }
        if ($process.ExitCode -ne 0) { throw "Failed: $CommandArgs - $($stderr.Result)" }
        $stdout.Result.Trim()
    } finally {
        if (-not $process.HasExited) { & taskkill.exe /PID $process.Id /T /F | Out-Null }
        $process.Dispose()
    }
}

$version = Invoke-PackagedCommand @('--version')
if ($version -notmatch '^ascii-video \d+\.\d+\.\d+$' -or
    ($ExpectedVersion -and $version -ne "ascii-video $ExpectedVersion")) {
    throw "Unexpected executable version: $version"
}
foreach ($format in 'mp4/h264','mp4/hevc','mp4/av1','webm/vp9','webm/av1') {
    $container, $codec = $format.Split('/')
    $output = Join-Path $out "$codec.$container"
    Invoke-PackagedCommand @('render', $video, '--output', $output, '--container', $container,
        '--codec', $codec, '--resolution', '128x64', '--columns', '16', '--color', 'source',
        '--dither', 'blue-noise', '--overwrite') | Out-Null
    if ((Get-Item -LiteralPath $output).Length -eq 0) { throw "Empty export: $output" }
}

$gui = Start-Process -FilePath $exe -WorkingDirectory $out -PassThru
try {
    $deadline = [DateTime]::UtcNow.AddSeconds(45)
    do {
        if ($gui.HasExited) { throw "GUI exited before opening its window: $($gui.ExitCode)" }
        # A PyInstaller onefile bootloader starts the Python application as a child.
        $children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $($gui.Id)"
        $window = @($gui; foreach ($child in $children) {
            Get-Process -Id $child.ProcessId -ErrorAction SilentlyContinue
        }) | Where-Object { $_.MainWindowHandle -and $_.MainWindowTitle -like '*ASCII Video*' } |
            Select-Object -First 1
        if (-not $window) { Start-Sleep -Milliseconds 200 }
    } until ($window -or [DateTime]::UtcNow -ge $deadline)
    if (-not $window) { throw 'No GUI window appeared' }
    if (-not $window.CloseMainWindow() -or -not $gui.WaitForExit(15000) -or $gui.ExitCode) {
        throw 'GUI did not close cleanly'
    }
} finally {
    if (-not $gui.HasExited) { & taskkill.exe /PID $gui.Id /T /F | Out-Null }
    $gui.Dispose()
}

$status = Join-Path $out ("player-" + [guid]::NewGuid() + ".txt")
$launcher = Join-Path $out ("player-" + [guid]::NewGuid() + ".cmd")
$stderr = Join-Path $out 'player-error.txt'
$previous = @{}
foreach ($name in 'TERM','ASCII_VIDEO_EXIT_STATUS','ASCII_CHECK_EXE','ASCII_CHECK_INPUT','ASCII_CHECK_STDERR','ASCII_CHECK_LAUNCHER') {
    $previous[$name] = [Environment]::GetEnvironmentVariable($name)
}
try {
    # This private Windows console has VT output even on a headless CI host.
    $env:TERM = $null
    $env:ASCII_VIDEO_EXIT_STATUS = $status
    $env:ASCII_CHECK_EXE = $exe
    $env:ASCII_CHECK_INPUT = $video
    $env:ASCII_CHECK_STDERR = $stderr
    $env:ASCII_CHECK_LAUNCHER = $launcher
    # Keep a console client attached until the windowed onefile app has finished.
    '@"%ASCII_CHECK_EXE%" play "%ASCII_CHECK_INPUT%" --font-scale 1 --no-audio 2>"%ASCII_CHECK_STDERR%"' |
        Set-Content -Encoding Ascii -LiteralPath $launcher
    $arguments = 'cmd.exe /d /v:off /c call "%ASCII_CHECK_LAUNCHER%"'
    $console = Start-Process -FilePath "$env:SystemRoot\System32\conhost.exe" -ArgumentList $arguments -WorkingDirectory $out -PassThru
    try {
        if (-not $console.WaitForExit(90000) -or -not (Test-Path -LiteralPath $status) -or
            (Get-Content -Raw -LiteralPath $status).Trim() -ne '0') { throw "Terminal playback failed. See $stderr" }
    } finally {
        if (-not $console.HasExited) { & taskkill.exe /PID $console.Id /T /F | Out-Null }
        $console.Dispose()
    }
} finally {
    foreach ($name in $previous.Keys) { [Environment]::SetEnvironmentVariable($name, $previous[$name]) }
    Remove-Item -LiteralPath $launcher
}
Write-Output "$version passed five exports, GUI startup and shutdown, and muted console playback."
