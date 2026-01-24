$ErrorActionPreference = 'Stop'
$branches = git for-each-ref --format='%(refname:short)' refs/remotes/origin | Where-Object {$_ -ne 'origin/HEAD' -and $_ -ne 'origin/main'}
foreach ($remote in $branches) {
    Write-Host "Processing $remote"
    git checkout main
    git pull
    git merge --no-commit --no-ff $remote
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Conflicts for $remote"
        $conflicts = git diff --name-only --diff-filter=U
        foreach ($f in $conflicts) {
            Write-Host "Resolving $f"
            $mainTime = [int](git log -1 --format=%ct main)
            $remoteTime = [int](git log -1 --format=%ct $remote)
            if ($remoteTime -gt $mainTime) {
                Write-Host "Choosing theirs (newer) for $f"
                git checkout --theirs -- $f
            } else {
                Write-Host "Choosing ours (main) for $f"
                git checkout --ours -- $f
            }
            git add $f
        }
        git rm --cached --ignore-unmatch src/__pycache__/model.cpython-312.pyc
        git commit -m "Merge $remote — auto-resolve conflicts preferring newest branch"
        git push
    } else {
        Write-Host "Merged cleanly: $remote"
        git commit -m "Merge $remote — merged cleanly" 2>$null
        git push
    }
}
