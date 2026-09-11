from pathlib import Path

repo = Path('.')

# Fix 1
p = repo / 'common' / 'prompt_utils.py'
text = p.read_text(encoding='utf-8') if p.exists() else ''
old = 'parts.append(f"\\n\\n[Current question]\\n{query}\\n\\nAnswer:")'
new = 'parts.append(f"\\n\\n[Current question]\\n{query}\\n\\nAnswer concisely in 1-2 sentences, then write <END>.\\n\\nAnswer:")'
if old in text and '<END>' not in text:
    p.write_text(text.replace(old, new), encoding='utf-8')
    print('✅ prompt_utils.py patched')
else:
    print('⚠️ check manually:', '<END>' in text)

# Fix 2
p = repo / 'baselines' / 'a1_naive_rag.py'
text = p.read_text(encoding='utf-8')
old = '''        sampling_params = SamplingParams(
            max_tokens=256,

            # Deterministic baseline.
            temperature=0.0,
        )'''
new = '''        sampling_params = SamplingParams(
            max_tokens=256,

            # Deterministic baseline.
            temperature=0.0,
            stop=["<END>", "[End of answer]", "\\n\\nQuestion:", "\\n\\n[Current question]"],
        )'''
if old in text:
    p.write_text(text.replace(old, new), encoding='utf-8')
    print('✅ a1_naive_rag.py patched')
elif 'stop=' in text:
    print('✅ a1_naive_rag.py already has stop=')
else:
    print('⚠️ a1 pattern mismatch')

# Fix 3-7
files = [
    repo / 'baselines' / 'a2_prefix_cache_rag.py',
    repo / 'baselines' / 'a3_agent_memory.py',
    repo / 'baselines' / 'a5_naive_full_reuse.py',
    repo / 'gaps' / 'gap1_cache_aware_retrieval.py',
    repo / 'gaps' / 'gap2_selective_recompute.py',
]
old2 = 'SamplingParams(max_tokens=256)'
new2 = 'SamplingParams(max_tokens=256, temperature=0.0, stop=["<END>", "[End of answer]", "\\n\\nQuestion:", "\\n\\n[Current question]"])'
for p in files:
    t = p.read_text(encoding='utf-8')
    if old2 in t:
        p.write_text(t.replace(old2, new2), encoding='utf-8')
        print(f'✅ {p} patched')
    elif 'stop=' in t:
        print(f'✅ {p} already has stop=')
    else:
        print(f'⚠️ {p} pattern mismatch')
