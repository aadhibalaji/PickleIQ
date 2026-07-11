# Start the dev stack, guaranteeing it is THIS working tree's code that gets served.
#
# The failure this prevents: a dev server left running from an earlier session (or an
# earlier branch) keeps holding 5173/8420. Vite then quietly falls back to 5174 and the
# backend fails to bind, so the browser keeps talking to the stale stack and your code
# changes appear to do nothing. Free the ports first and that cannot happen.
#
#   npm run dev        (wired to this script)
#   powershell -ExecutionPolicy Bypass -File dev.ps1

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$ports = 5173, 8420

# Snapshot the holders of BOTH ports before killing anything: killing one server makes
# `concurrently -k` tear down its sibling, so a PID found now may be gone a moment later.
$targets = foreach ($port in $ports) {
    $conn = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
    foreach ($c in $conn) {
        $proc = Get-Process -Id $c.OwningProcess -ErrorAction SilentlyContinue
        if (-not $proc) { continue }

        # Only ever reap our own dev servers. If something else holds the port, stop and
        # say so rather than force-killing a process the user cares about.
        if ($proc.ProcessName -notin 'node', 'python') {
            throw "Port $port is held by $($proc.ProcessName) (PID $($proc.Id)), which is not a dev server. Refusing to kill it."
        }
        $proc
    }
}

foreach ($proc in $targets) {
    # The sibling cascade may already have taken this one down; that is success, not failure.
    if (-not (Get-Process -Id $proc.Id -ErrorAction SilentlyContinue)) { continue }
    Write-Host "freeing stale $($proc.ProcessName) (PID $($proc.Id), started $($proc.StartTime))" -ForegroundColor Yellow
    # /T so uvicorn's --reload worker and any other children go too, not just the parent.
    # No stderr redirect here: in PS 5.1 that wraps native stderr in an ErrorRecord, which
    # under ErrorActionPreference=Stop aborts this script.
    taskkill /PID $proc.Id /T /F | Out-Null
}

# A killed process can hold its socket briefly. Starting before it lets go is exactly the
# bind failure we are trying to prevent, so wait for the ports to actually clear.
foreach ($port in $ports) {
    for ($i = 0; $i -lt 50; $i++) {
        if (-not (Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue)) { break }
        Start-Sleep -Milliseconds 100
    }
    if (Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue) {
        throw "Port $port is still held after 5s. Run: Get-NetTCPConnection -LocalPort $port"
    }
}

Write-Host "ports $($ports -join ' + ') clear - starting backend + frontend" -ForegroundColor Green
npm run dev:servers
