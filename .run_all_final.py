import subprocess
from pathlib import Path

root = Path(r'd:/OCULOXPLAIN/OculoXplain')
py = root / '.venv' / 'Scripts' / 'python.exe'
scripts = sorted([p.name for p in root.glob('*.py') if not p.name.startswith('.')])
report = root / 'run_all_report_final.txt'

fails = 0
timeouts = 0
passes = 0
with report.open('w', encoding='utf-8', buffering=1) as f:
    f.write('=== RUN-ALL SUMMARY ===\n')
    f.write(f'Total scripts: {len(scripts)}\n')
    f.flush()
    for i, name in enumerate(scripts, 1):
        try:
            p = subprocess.run([str(py), name], cwd=str(root), capture_output=True, text=True, timeout=8, env={**dict(), **__import__('os').environ, 'PYTHONUTF8':'1'})
            status = 'PASS' if p.returncode == 0 else f'FAIL({p.returncode})'
            tail = ((p.stdout or '') + '\n' + (p.stderr or '')).strip()
        except subprocess.TimeoutExpired as e:
            status = 'TIMEOUT'
            tail = ((e.stdout or '') + '\n' + (e.stderr or '')).strip()

        if status == 'PASS':
            passes += 1
        elif status == 'TIMEOUT':
            timeouts += 1
        else:
            fails += 1

        line = f'[{i}/{len(scripts)}] {name}: {status}'
        print(line, flush=True)
        f.write(line + '\n')
        if status != 'PASS':
            f.write(f'--- {name} :: {status} ---\n')
            if len(tail) > 1200:
                tail = tail[-1200:]
            f.write((tail if tail else '(no output)') + '\n\n')
        f.flush()

print(f'FINAL_COUNTS PASS={passes} FAIL={fails} TIMEOUT={timeouts}', flush=True)
print(f'Report: {report}', flush=True)
