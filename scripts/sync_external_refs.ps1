param(
    [string]$Root = "external_refs",
    [switch]$PullLatest
)

$ErrorActionPreference = "Stop"

$repoUrl = "https://github.com/leggedrobotics/robotic_world_model"
$repoName = "robotic_world_model"
$rootPath = Resolve-Path "." | ForEach-Object { Join-Path $_ $Root }
$repoPath = Join-Path $rootPath $repoName

if (!(Test-Path $rootPath)) {
    New-Item -ItemType Directory -Path $rootPath -Force | Out-Null
}

function Invoke-Git {
    param(
        [Parameter(Mandatory=$true)][string[]]$Args
    )
    & git @Args
    if ($LASTEXITCODE -ne 0) {
        throw "git command failed: git $($Args -join ' ')"
    }
}

if (!(Test-Path $repoPath)) {
    Invoke-Git -Args @("clone", "--depth", "1", $repoUrl, $repoPath)
} elseif ($PullLatest.IsPresent) {
    Invoke-Git -Args @("-C", $repoPath, "fetch", "--depth", "1", "origin")
    $remoteInfo = & git -C $repoPath remote show origin
    if ($LASTEXITCODE -ne 0) {
        throw "git command failed: git -C $repoPath remote show origin"
    }
    $defaultBranch = ($remoteInfo | Select-String "HEAD branch" | ForEach-Object { $_.ToString().Split(":")[-1].Trim() })
    if ([string]::IsNullOrWhiteSpace($defaultBranch)) {
        $defaultBranch = "main"
    }
    Invoke-Git -Args @("-C", $repoPath, "checkout", $defaultBranch)
    Invoke-Git -Args @("-C", $repoPath, "pull", "--ff-only", "origin", $defaultBranch)
}

if (!(Test-Path $repoPath)) {
    throw "Reference repository missing after sync attempt: $repoPath"
}

$headCommit = (& git -C $repoPath rev-parse HEAD)
if ($LASTEXITCODE -ne 0) {
    throw "git command failed: git -C $repoPath rev-parse HEAD"
}
$branch = (& git -C $repoPath rev-parse --abbrev-ref HEAD)
if ($LASTEXITCODE -ne 0) {
    throw "git command failed: git -C $repoPath rev-parse --abbrev-ref HEAD"
}
$manifest = @{
    references = @(
        @{
            name = $repoName
            url = $repoUrl
            path = $repoPath
            branch = $branch
            commit = $headCommit
            synced_utc = (Get-Date).ToUniversalTime().ToString("o")
        }
    )
}

$manifestPath = Join-Path $rootPath "manifest.json"
$manifest | ConvertTo-Json -Depth 5 | Out-File -FilePath $manifestPath -Encoding utf8
Write-Output (@{
    root = $rootPath
    repo = $repoPath
    manifest = $manifestPath
    commit = $headCommit
} | ConvertTo-Json -Depth 4)
