# Drift monitor: DINOv2 (deployed model) + ConvNeXt (canary sensor). Batch-level alert from raw embeddings.
import numpy as np, joblib, torch, torch.nn as nn
from scipy.optimize import nnls

BK, ACT, REF_B = ['dino', 'convnext'], 0.05, 100

def _ks_vec(X, S):
    B, n = len(X), len(S); Xs = np.sort(X, 0)
    Fr = np.stack([np.searchsorted(S[:, j], Xs[:, j], side='right') for j in range(S.shape[1])], 1) / n
    up, lo = np.arange(1, B + 1)[:, None] / B, np.arange(B)[:, None] / B
    return np.maximum(np.abs(up - Fr), np.abs(lo - Fr)).max(0)

def _ae(dim):
    return nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Linear(256, 32), nn.ReLU(),
                         nn.Linear(32, 256), nn.ReLU(), nn.Linear(256, dim))

class DriftMonitor:
    def __init__(self, objects_path, artifacts_path):
        self.obj, self.art = joblib.load(objects_path), joblib.load(artifacts_path)
        self.ae = {}
        for key, o in self.obj.items():
            ae = _ae(o['dim']); ae.load_state_dict(o['ae']); self.ae[key] = ae.eval()

    def _sensor(self, dd, k, emb):
        o = self.obj[(dd, k)]
        z = (np.asarray(emb, np.float32) - o['mu']) / o['sd']
        P = o['pca'].transform(z)
        with torch.no_grad():
            x = torch.tensor(z, dtype=torch.float32); err = ((self.ae[(dd, k)](x) - x) ** 2).mean(1).numpy()
        kv = _ks_vec(P, o['S'])
        return kv.mean(), err.mean(), np.concatenate([P.mean(0), P.std(0), kv, [kv.mean(), err.mean()]])

    def _label_shift(self, cond_logits):
        pred = np.asarray(cond_logits)[:, :11].argmax(1)
        q = np.bincount(pred, minlength=11) / len(pred)
        w, _ = nnls(self.art['C'], q); w /= w.sum()
        return w @ self.art['acc_c'] - self.art['pref'] @ self.art['acc_c']

    def assess(self, domain, emb, cond_logits=None):
        # emb: {'dino': (B, 768), 'convnext': (B, 768)} raw embeddings of one batch (same extraction as notebook 03a).
        # cond_logits: DINOv2 condition-head logits (B, >=11), BDD only; used for the label-shift impact estimate.
        parts = [self._sensor(domain, k, emb[k]) for k in BK]
        s = np.array([parts[0][0], parts[0][1], parts[1][0], parts[1][1]])
        B = len(emb[BK[0]]); tb = self.art.get('thr_by_B', {}).get(domain)
        if tb:
            sizes = sorted(tb); Bt = max([z for z in sizes if z <= B] or [sizes[0]]); thr = tb[Bt]['both']   # largest calibrated size <= B (conservative)
        else:
            Bt, thr = REF_B, self.art['thr'][domain]['both']
        fired = s > thr
        by = [k for k, hit in (('dino', fired[:2].any()), ('convnext', fired[2:].any())) if hit]
        out = dict(level='none', fired_by=by, cause=None, impact=0.0, scores=s.round(4).tolist(), batch_size=B, thr_batch_size=Bt)
        notes = []
        if tb and B < min(tb): notes.append(f'batch is smaller than the smallest calibrated size ({min(tb)}); false alarms may exceed 1%')
        if B != REF_B: notes.append(f'impact models were trained on {REF_B}-image batches; treat the estimate as indicative')
        if notes: out['note'] = '; '.join(notes)
        if not by: return out
        x = np.concatenate([p[2] for p in parts])[None]
        cause = self.art['att'][domain].predict(x)[0]
        if cause == 'mix_shift' and cond_logits is not None: impact = self._label_shift(cond_logits)
        else: impact = -self.art['reg'][domain]['both'].predict(x)[0]
        out.update(level='act' if impact <= -ACT else 'watch', cause=cause, impact=round(float(impact), 3))
        return out
