$repo = (Get-Location).Path

# Fix 1: common/prompt_utils.py
$path = Join-Path $repo 'common/prompt_utils.py'
$text = Get-Content -Raw -Path $path
$old = 'parts.append(f"\n\n[Current question]\n{query}\n\nAnswer:")'
$new = 'parts.append(f"\n\n[Current question]\n{query}\n\nAnswer concisely in 1-2 sentences, then write <END>.\n\nAnswer:")'
if ($text.Contains($old) -and -not ($text.Contains('<END>'))) {
    $text = $text.Replace($old, $new)
    Set-Content -Path $path -Value $text
    Write-Host '✅ prompt_utils.py patched'
} else {
    Write-Host '⚠️ check manually:' $text.Contains('<END>')
}

# Fix 2: a1_naive_rag.py
$path = Join-Path $repo 'baselines/a1_naive_rag.py'
$text = Get-Content -Raw -Path $path
$old = @'
        sampling_params = SamplingParams(
            max_tokens=256,

            # Deterministic baseline.
            temperature=0.0,
        )'@
$new = @'
        sampling_params = SamplingParams(
            max_tokens=256,

            # Deterministic baseline.
            temperature=0.0,
            stop=["<END>", "[End of answer]", "\n\nQuestion:", "\n\n[Current question]"],
        )'@
if ($text.Contains($old)) {
    $text = $text.Replace($old, $new)
    Set-Content -Path $path -Value $text
    Write-Host '✅ a1_naive_rag.py patched'
} elseif ($text.Contains('stop=')) {
    Write-Host '✅ a1_naive_rag.py already has stop='
} else {
    Write-Host '⚠️ a1 pattern mismatch'
}

# Fix 3-7: baaki 5 files
$files = @(
    'baselines/a2_prefix_cache_rag.py',
    'baselines/a3_agent_memory.py',
    'baselines/a5_naive_full_reuse.py',
    'gaps/gap1_cache_aware_retrieval.py',
    'gaps/gap2_selective_recompute.py'
)
$old2 = 'SamplingParams(max_tokens=256)'
$new2 = 'SamplingParams(max_tokens=256, temperature=0.0, stop=["<END>", "[End of answer]", "\n\nQuestion:", "\n\n[Current question]"])'
foreach ($f in $files) {
    $path = Join-Path $repo $f
    $t = Get-Content -Raw -Path $path
    if ($t.Contains($old2)) {
        $t = $t.Replace($old2, $new2)
        Set-Content -Path $path -Value $t
        Write-Host "✅ $f patched"
    } elseif ($t.Contains('stop=')) {
        Write-Host "✅ $f already has stop="
    } else {
        Write-Host "⚠️ $f pattern mismatch"
    }
}
