<#
.SYNOPSIS
    Build tous les sites Quarkdown du repo, ou un domaine cible.

.DESCRIPTION
    Itere sur domains/*/01-theory-qd/main.qd et build chaque site dans
    quarkdown/output-site/<domain>/. Applique systematiquement le
    post-process post-build-fix-links.py pour rendre les bundles
    portables (file:// + WKWebView KaView).

    Un domaine sans 01-theory-qd/ est ignore (silencieux). Pour scaffold
    un nouveau domaine : python quarkdown/scripts/scaffold-domain.py <domain>.

.PARAMETER Domain
    Build uniquement ce domaine. Echoue si 01-theory-qd/main.qd absent.

.PARAMETER Watch
    Live-preview avec auto-reload (-p -w cote Quarkdown). Necessite
    -Domain pour cibler 1 site.

.EXAMPLE
    pwsh quarkdown/scripts/build-all.ps1
    Build tous les domaines qui ont un 01-theory-qd/.

.EXAMPLE
    pwsh quarkdown/scripts/build-all.ps1 -Domain agentic-ai
    Build uniquement agentic-ai.

.EXAMPLE
    pwsh quarkdown/scripts/build-all.ps1 -Domain agentic-ai -Watch
    Live-preview agentic-ai dans le browser.
#>

[CmdletBinding()]
param(
    [string]$Domain,
    [switch]$Watch
)

$ErrorActionPreference = "Stop"

# Resolution des chemins (script dans quarkdown/scripts/)
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
$DomainsDir = Join-Path $RepoRoot "domains"
$OutputBase = Join-Path $RepoRoot "quarkdown\output-site"
$PostBuildScript = Join-Path $RepoRoot "quarkdown\post-build-fix-links.py"

# Configuration JAVA_HOME et PATH si overrides locaux disponibles
# Override par variables d'env :
#   $env:QUARKDOWN_JAVA_HOME = "C:\path\to\jdk"
#   $env:QUARKDOWN_BIN_DIR   = "C:\path\to\quarkdown\bin"
# Sinon on assume que `java` et `quarkdown` sont deja dans le PATH.
if ($env:QUARKDOWN_JAVA_HOME -and (Test-Path $env:QUARKDOWN_JAVA_HOME)) {
    $env:JAVA_HOME = $env:QUARKDOWN_JAVA_HOME
    $env:Path = "$env:JAVA_HOME\bin;$env:Path"
}
if ($env:QUARKDOWN_BIN_DIR -and (Test-Path $env:QUARKDOWN_BIN_DIR)) {
    $env:Path = "$env:QUARKDOWN_BIN_DIR;$env:Path"
}

# Validation Watch + Domain
if ($Watch -and -not $Domain) {
    Write-Error "Le mode -Watch necessite -Domain <name> (un seul site a la fois)."
    exit 2
}

# Build un site
function Invoke-DomainBuild {
    param(
        [string]$DomainName,
        [string]$MainQd,
        [string]$OutputDir,
        [switch]$WatchMode
    )

    Write-Host ""
    Write-Host "==> Building $DomainName" -ForegroundColor Cyan
    Write-Host "    main.qd : $MainQd"
    Write-Host "    output  : $OutputDir"

    $args = @("c", $MainQd, "--out", $OutputDir)
    if ($WatchMode) { $args += @("-p", "-w") }

    & quarkdown @args
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Quarkdown build failed for $DomainName (exit $LASTEXITCODE)"
        return $false
    }

    # Watch mode : Quarkdown serve indefiniment, on n'enchaine pas le post-process.
    if ($WatchMode) { return $true }

    Write-Host "    post-process links..." -ForegroundColor DarkGray
    & python $PostBuildScript $OutputDir
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Post-process failed for $DomainName"
        return $false
    }
    return $true
}

# Resolution des domaines a builder
$domainsToBuild = @()
$skipped = @()

if ($Domain) {
    $mainQd = Join-Path $DomainsDir "$Domain\01-theory-qd\main.qd"
    if (-not (Test-Path $mainQd)) {
        Write-Host "ERROR: no 01-theory-qd/main.qd found for domain '$Domain'" -ForegroundColor Red
        Write-Host "       expected at : $mainQd" -ForegroundColor Red
        Write-Host "       fix         : python quarkdown/scripts/scaffold-domain.py $Domain" -ForegroundColor Yellow
        exit 1
    }
    $domainsToBuild += [PSCustomObject]@{
        Name = $Domain
        MainQd = $mainQd
        OutputDir = Join-Path $OutputBase $Domain
    }
} else {
    Get-ChildItem -Path $DomainsDir -Directory | ForEach-Object {
        $candidate = Join-Path $_.FullName "01-theory-qd\main.qd"
        if (Test-Path $candidate) {
            $domainsToBuild += [PSCustomObject]@{
                Name = $_.Name
                MainQd = $candidate
                OutputDir = Join-Path $OutputBase $_.Name
            }
        } else {
            $skipped += $_.Name
        }
    }
}

if ($domainsToBuild.Count -eq 0) {
    Write-Warning "Aucun domaine avec 01-theory-qd/main.qd trouve."
    if ($skipped.Count -gt 0) {
        Write-Host "Domains sans 01-theory-qd/ : $($skipped -join ', ')"
        Write-Host "Pour scaffold : python quarkdown/scripts/scaffold-domain.py <domain>"
    }
    exit 0
}

# Build
$succeeded = @()
$failed = @()
foreach ($d in $domainsToBuild) {
    $ok = Invoke-DomainBuild -DomainName $d.Name -MainQd $d.MainQd -OutputDir $d.OutputDir -WatchMode:$Watch
    if ($ok) { $succeeded += $d.Name } else { $failed += $d.Name }
}

# Recap
Write-Host ""
Write-Host "============================================" -ForegroundColor Yellow
Write-Host " Recap"
Write-Host "============================================" -ForegroundColor Yellow
if ($succeeded.Count -gt 0) {
    Write-Host " OK    : $($succeeded -join ', ')" -ForegroundColor Green
}
if ($failed.Count -gt 0) {
    Write-Host " FAIL  : $($failed -join ', ')" -ForegroundColor Red
}
if ($skipped.Count -gt 0) {
    Write-Host " SKIP  : $($skipped -join ', ') (no 01-theory-qd/)" -ForegroundColor DarkGray
}
Write-Host ""

if ($failed.Count -gt 0) { exit 1 }
exit 0
