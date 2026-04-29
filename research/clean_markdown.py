import sys
import re

replacements = {
    '≥': '>=',
    '≤': '<=',
    '≈': '~',
    'α': 'alpha',
    '→': '->',
    '【': '[',
    '】': ']',
    '｜': '|',
    '：': ':',
    '−': '-',
    '–': '-',
    '—': '--',
    'Δ': 'Delta',
    'β': 'beta',
    'γ': 'gamma',
    'μ': 'mu',
    'σ': 'sigma',
    '×': 'x',
    '÷': '/',
    '≠': '!=',
    '±': '+-',
    '∈': 'in',
    '∞': 'infinity',
    '∑': 'sum',
    '∏': 'prod',
    '√': 'sqrt',
    '∂': 'd',
    'λ': 'lambda',
    'ε': 'epsilon',
    'θ': 'theta',
    'ω': 'omega',
    '…': '...',
    '‧': '.',
    '•': '*',
    '·': '*',
    '・': '*',
    '"': '"',
    '"': '"',
    ''': "'",
    ''': "'",
    '–': '-',
    '—': '--',
    '−': '-',
}

markdown_replacements = {
    '²': '^2',
    '³': '^3',
    '⁴': '^4',
}

input_file = sys.argv[1]
output_file = sys.argv[2] if len(sys.argv) > 2 else 'cleaned.md'

with open(input_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace unicode characters
for old, new in replacements.items():
    content = content.replace(old, new)

for old, new in markdown_replacements.items():
    content = content.replace(old, new)

with open(output_file, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Cleaned markdown written to {output_file}")

# Check for remaining non-ascii characters
remaining = set()
for ch in content:
    if ord(ch) > 127:
        remaining.add(ch)

if remaining:
    print(f"Remaining non-ASCII characters: {sorted(remaining)}")
