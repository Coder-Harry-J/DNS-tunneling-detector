import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from app import load_models, predict_all, SCENARIOS

load_models()
test_ids = ['benign_browse', 'attack_base64_heavy', 'attack_audio_exfil', 'benign_cloud', 'attack_dga']
for sid in test_ids:
    s = next(x for x in SCENARIOS if x['id'] == sid)
    r = predict_all(s['features'])
    line = f"[{s['category'].upper():<7}] {s['label']:<35}"
    for m, v in r['models'].items():
        line += f"  {m}={v['label']}({v['probability']:.1f}%)"
    print(line)
print("\n[OK] All predictions work.")
