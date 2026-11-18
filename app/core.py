# Embedding + drift analysis engine for the dashboard (CPU). Needs drift_pipeline.py next to this file.
import numpy as np, torch, torch.nn as nn
from scipy.optimize import nnls
from PIL import Image
from torchvision import models, transforms as T
from transformers import AutoConfig, Dinov2Model
from drift_pipeline import DriftMonitor, BK

SIZE = {'places': (224, 224), 'bdd': (384, 216)}      # (width, height) of the images the detectors were fitted on
TF = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class MultiTask(nn.Module):
    def __init__(s, bb, d, dino=False):
        super().__init__()
        s.bb, s.dino = bb, dino
        s.h1, s.h2, s.h3 = nn.Linear(d, 48), nn.Linear(d, 11), nn.Linear(d, 2)

    def forward(s, x):
        if s.dino:
            h = s.bb(pixel_values=x).last_hidden_state
            z = torch.cat([h[:, 0], h[:, 1:].mean(1)], 1)
        else:
            z = s.bb(x)
        return z, s.h1(z), s.h2(z), s.h3(z)


def prepare(img, domain):
    # scale to cover the target size, then centre-crop (images already at the native size are untouched)
    img = img.convert('RGB'); w, h = SIZE[domain]
    if img.size == (w, h): return img
    s = max(w / img.width, h / img.height)
    img = img.resize((max(w, round(img.width * s)), max(h, round(img.height * s))), Image.BICUBIC)
    l, t = (img.width - w) // 2, (img.height - h) // 2
    return img.crop((l, t, l + w, t + h))


class Engine:
    def __init__(self, objects, artifacts, dino_pt, convnext_pt, dino_cfg):
        self.monitor = DriftMonitor(objects, artifacts)
        cx = models.convnext_tiny(weights=None); cx.classifier[2] = nn.Identity()
        self.nets = {'convnext': MultiTask(cx, 768),
                     'dino': MultiTask(Dinov2Model(AutoConfig.from_pretrained(str(dino_cfg))), 768, True)}
        for k, pt in (('convnext', convnext_pt), ('dino', dino_pt)):
            self.nets[k].load_state_dict(torch.load(pt, map_location='cpu')); self.nets[k].eval()

    @torch.no_grad()
    def embed(self, imgs, bs=25):
        out = {k: {'emb': [], 'cond': []} for k in self.nets}
        for i in range(0, len(imgs), bs):
            x = torch.stack([TF(im) for im in imgs[i:i + bs]])
            for k, m in self.nets.items():
                z, _, c, _ = m(x); out[k]['emb'].append(z); out[k]['cond'].append(c)
        return {k: {n: torch.cat(v).numpy() for n, v in d.items()} for k, d in out.items()}

    def analyze(self, imgs, domain):
        r = self.embed([prepare(i, domain) for i in imgs])
        res = self.monitor.assess(domain, {k: r[k]['emb'] for k in BK}, r['dino']['cond'] if domain == 'bdd' else None)
        if res['cause'] == 'mix_shift':                      # estimated scene mix of this batch vs the usual mix
            art = self.monitor.art
            q = np.bincount(r['dino']['cond'][:, :11].argmax(1), minlength=11) / len(imgs)
            w, _ = nnls(art['C'], q)
            res['scene_mix'] = {'now': (w / w.sum()).tolist(), 'usual': np.asarray(art['pref']).tolist()}
        return res
