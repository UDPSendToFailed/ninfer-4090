$ErrorActionPreference = 'Stop'

$ReleaseTag = 'v0.9.0'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$BuildRoot = Join-Path $RepoRoot 'build-ninja'
$DistRoot = Join-Path $RepoRoot 'dist'
$ProductName = "ninfer-rtx4090-windows-x64-$ReleaseTag"
$ProductRoot = Join-Path $DistRoot $ProductName
$ArchivePath = Join-Path $DistRoot "$ProductName.zip"
$ChecksumPath = Join-Path $DistRoot "SHA256SUMS-$ReleaseTag.txt"

$Products = @(
    @{ Source = 'apps\ninfer.exe'; Destination = 'ninfer.exe' },
    @{ Source = 'apps\ninfer-serve.exe'; Destination = 'ninfer-serve.exe' },
    @{ Source = 'bench\ninfer_bench.exe'; Destination = 'ninfer_bench.exe' }
)

New-Item -ItemType Directory -Force -Path $DistRoot | Out-Null
$resolvedDist = (Resolve-Path -LiteralPath $DistRoot).Path
$resolvedProductParent = [System.IO.Path]::GetFullPath((Split-Path -Parent $ProductRoot))
if ($resolvedProductParent -ne $resolvedDist -or (Split-Path -Leaf $ProductRoot) -ne $ProductName) {
    throw "Refusing to package outside the expected dist directory: $ProductRoot"
}
if (Test-Path -LiteralPath $ProductRoot) { Remove-Item -LiteralPath $ProductRoot -Recurse -Force }
if (Test-Path -LiteralPath $ArchivePath) { Remove-Item -LiteralPath $ArchivePath -Force }
New-Item -ItemType Directory -Path $ProductRoot | Out-Null

foreach ($product in $Products) {
    $source = Join-Path $BuildRoot $product.Source
    if (-not (Test-Path -LiteralPath $source)) { throw "Missing release product: $source" }
    Copy-Item -LiteralPath $source -Destination (Join-Path $ProductRoot $product.Destination)
}
Get-ChildItem -LiteralPath (Join-Path $BuildRoot 'apps') -Filter '*.dll' | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination $ProductRoot
}
Copy-Item -LiteralPath (Join-Path $RepoRoot 'LICENSE') -Destination $ProductRoot
Copy-Item -LiteralPath (Join-Path $RepoRoot 'README.md') -Destination $ProductRoot
Copy-Item -LiteralPath (Join-Path $RepoRoot ('RELEASE_NOTES_' + $ReleaseTag.Replace('v','') + '.md')) -Destination $ProductRoot

$innerHashes = Get-ChildItem -LiteralPath $ProductRoot -File | Sort-Object Name | ForEach-Object {
    $hash = Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256
    "$($hash.Hash.ToLowerInvariant())  $($_.Name)"
}
$innerHashes | Set-Content -LiteralPath (Join-Path $ProductRoot 'SHA256SUMS.txt') -Encoding ascii
Compress-Archive -LiteralPath $ProductRoot -DestinationPath $ArchivePath -CompressionLevel Optimal
$archiveHash = Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256
"$($archiveHash.Hash.ToLowerInvariant())  $(Split-Path -Leaf $ArchivePath)" |
    Set-Content -LiteralPath $ChecksumPath -Encoding ascii

Get-Item -LiteralPath $ArchivePath, $ChecksumPath |
    Select-Object Name, @{ Name = 'SizeMB'; Expression = { [math]::Round($_.Length / 1MB, 2) } }
