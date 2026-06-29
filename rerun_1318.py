"""Train 4 seeds × 2 modes on original 13:18 data files.
Data already in data_baseline/ and data_rag/ (restored from old/ backup).
"""
import sys, os, torch, numpy as np, io
from pathlib import Path
from datetime import datetime
os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

OLD = Path(r"D:/Program Files/CC-Agent/Service-Robots/Bert_Chinese_Text/Bert-Chinese-Text-Classification-Pytorch")
SAVED = OLD / "THUCNews" / "saved_dict"
LOG = Path(r"D:/Program Files/CC-Agent/Service-Robots/Log.md")
SAVED.mkdir(parents=True, exist_ok=True)

def wlog(msg):
    t = datetime.now().strftime('%m-%d %H:%M:%S')
    with open(LOG,'a',encoding='utf-8') as f: f.write(f'\n---\n\n## {t} | {msg}\n'); f.flush(); os.fsync(f.fileno())

wlog('[13:18-FULL] 4种子×2模式训练启动')

sys.path.insert(0, str(OLD))
from importlib import import_module
import utils as bert_utils; import train_eval
bm = import_module("models.bert"); ds_dir = str(OLD/'THUCNews')

results = {}
for seed in [1, 42, 1188, 999]:
    for mode, label, subdir_name, pad in [
        ('baseline', 'Baseline', 'data_baseline', 64),
        ('rag', 'RAG', 'data_rag', 128)
    ]:
        subdir = OLD / 'THUCNews' / subdir_name
        ckpt = SAVED / f'bert_{mode}_seed{seed}.ckpt'
        lf = SAVED / f'bert_{mode}_seed{seed}.log'
        ml = f'[{label} seed={seed}]'
        wlog(f'{ml} 开始训练')

        np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

        cfg = bm.Config(ds_dir)
        cfg.train_path = str(subdir/'train.txt'); cfg.dev_path = str(subdir/'dev.txt')
        cfg.test_path = str(subdir/'test.txt'); cfg.pad_size = pad; cfg.save_path = str(ckpt)

        td, dd, ttd = bert_utils.build_dataset(cfg)
        model = bm.Model(cfg).to(cfg.device)

        class ET:
            def __init__(self, fh, l): self.fh = fh; self.l = l; self.cur = []
            def write(self, s):
                self.fh.write(s); self.fh.flush(); self.cur.append(s)
                for line in s.split('\n'):
                    if line.startswith('Epoch ['):
                        best = [l for l in self.cur if 'Iter:' in l and l.strip().endswith('*')]
                        if best: wlog(f'{self.l} | {line.strip()} best: {best[-1].strip()}')
                        self.cur = []
                return len(s)
            def flush(self): self.fh.flush()
            def __getattr__(self, n): return getattr(self.fh, n)

        o = sys.stdout
        with open(lf, 'w', encoding='utf-8') as fh:
            sys.stdout = ET(fh, ml)
            train_eval.train(cfg, model,
                bert_utils.build_iterator(td, cfg),
                bert_utils.build_iterator(dd, cfg),
                bert_utils.build_iterator(ttd, cfg))
        sys.stdout = o

        acc = '?'; f1 = '?'
        for l in open(lf, 'r', encoding='utf-8', errors='replace').readlines():
            if 'Test Acc:' in l: acc = l.split('Test Acc:')[1].strip().replace('%','').split()[0]
            if 'macro avg' in l: f1 = l.strip().split()[-2]
        results[f'{mode}_{seed}'] = (acc, f1)
        wlog(f'{ml} 完成 — Acc={acc}% F1={f1}')

# Summary
tbl = '\n## [13:18-FULL] 4-Seed汇总\n\n| Seed | Baseline | F1 | +RAG | F1 | Δ |\n|------|:--:|:--:|:--:|:--:|:--:|\n'
b_vals, r_vals = [], []
for s in [1, 42, 1188, 999]:
    ba = float(results.get(f'baseline_{s}', ('0','0'))[0])
    bf = results.get(f'baseline_{s}', ('0','0'))[1]
    ra = float(results.get(f'rag_{s}', ('0','0'))[0])
    rf = results.get(f'rag_{s}', ('0','0'))[1]
    b_vals.append(ba); r_vals.append(ra)
    tbl += f'| {s} | {ba:.2f}% | {bf} | {ra:.2f}% | {rf} | {ra-ba:+.2f} |\n'
tbl += f'| **均值±σ** | {np.mean(b_vals):.2f}%±{np.std(b_vals):.2f} | — | {np.mean(r_vals):.2f}%±{np.std(r_vals):.2f} | — | {np.mean(r_vals)-np.mean(b_vals):+.2f} |\n'
wlog(tbl)
print(tbl)
wlog('[13:18-FULL] 完成')
