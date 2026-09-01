# 把 0001-add-phoneme-durations-header.patch 应用到本地 Bert-VITS2 仓。
# 用法：pwsh tools\tts_patch\apply.ps1 -RepoPath D:\Bert-VITS2
#
# 代理无 D:\Bert-VITS2 写权限，本脚本须由用户手工运行。
param(
    [Parameter(Mandatory = $true)]
    [string]$RepoPath
)

$ErrorActionPreference = "Stop"
$patchFile = Join-Path $PSScriptRoot "0001-add-phoneme-durations-header.patch"
$versionFile = Join-Path $PSScriptRoot "VERSION"

if (-not (Test-Path $RepoPath)) {
    throw "目标仓路径不存在：$RepoPath"
}
if (-not (Test-Path $patchFile)) {
    throw "找不到补丁文件：$patchFile"
}

Push-Location $RepoPath
try {
    $head = (git log -1 --format=%H 2>$null)
    $versionText = Get-Content $versionFile -Raw
    if ($head -and $versionText -notmatch [regex]::Escape($head)) {
        Write-Warning "目标仓 HEAD（$head）与 VERSION 记录的核验 commit 不一致——" `
            "补丁按行号/上下文锚点生成，版本漂移可能导致应用失败或悄悄对不上。" `
            "继续尝试应用，不阻断（git apply 若真对不上会自行报错退出）。"
    }
    git apply --check $patchFile
    git apply $patchFile
    Write-Host "已应用：$patchFile → $RepoPath"
}
finally {
    Pop-Location
}
