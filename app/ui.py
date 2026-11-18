# Dashboard widgets: layout helpers and the plain-language explanation of an alert.
import html, pandas as pd, streamlit as st

NAMES = ['clear_day', 'clear_night', 'clear_twilight', 'overcast_day', 'overcast_twilight', 'rainy_day', 'rainy_night',
         'rainy_twilight', 'snow_fog_day', 'snow_fog_night', 'snow_fog_twilight']
DOMAIN_NAME = {'bdd': 'BDD100K driving scenes', 'places': 'Places365 scenes'}
SENSOR = {'dino': 'DINOv2 (the deployed model)', 'convnext': 'ConvNeXt (the independent canary)'}
CAUSES = {'gaussian_noise': 'grainy, noisy images', 'shot_noise': 'grainy images, as from a camera in very low light',
          'impulse_noise': 'speckled images (random black and white dots)', 'defocus_blur': 'out-of-focus, blurry images',
          'glass_blur': 'a frosted-glass style blur', 'motion_blur': 'smeared images, as if the camera or subject was moving',
          'zoom_blur': 'a radial smear, as if zooming while shooting', 'snow': 'snow falling in front of the camera',
          'frost': 'frost or ice on the lens', 'fog': 'foggy, low-visibility images', 'brightness': 'images brighter than usual',
          'contrast': 'images with washed-out or distorted contrast', 'elastic_transform': 'warped, wobbly images',
          'pixelate': 'blocky, low-resolution images', 'jpeg_compression': 'heavily compressed images with visible blocks'}
CHECKS = [('DINOv2', 'Feature distribution (K-S)'), ('DINOv2', 'Reconstruction error'),
          ('ConvNeXt', 'Feature distribution (K-S)'), ('ConvNeXt', 'Reconstruction error')]
GROUPS = {'Time of day': [('Day', lambda n: n.endswith('_day')), ('Night', lambda n: n.endswith('_night')),
                          ('Twilight', lambda n: n.endswith('_twilight'))],
          'Weather': [('Clear', lambda n: n.startswith('clear')), ('Overcast', lambda n: n.startswith('overcast')),
                      ('Rain', lambda n: n.startswith('rainy')), ('Snow / fog', lambda n: n.startswith('snow_fog'))]}

CSS = ('<style>'
       '#MainMenu, footer, [data-testid="stMainMenu"], [data-testid="stDecoration"], [data-testid="stToolbarActions"], .stAppDeployButton, .stDeployButton {display:none !important;}'
       '[data-testid="stSidebarCollapseButton"], [data-testid="stSidebarCollapsedControl"] {display:none !important;}'
       '[data-testid="stHeader"] {background:transparent;}'
       '.block-container {padding-top:1.6rem; padding-bottom:1rem; max-width:1300px;}'
       '.card {border:1px solid rgba(128,128,128,.28); border-radius:12px; padding:14px 18px; background:rgba(128,128,128,.07); margin-bottom:14px;}'
       '.card h4 {margin:0 0 10px 0; font-size:.74rem; letter-spacing:.09em; text-transform:uppercase; opacity:.6; font-weight:600;}'
       '.card p {margin:0 0 8px 0; line-height:1.45; font-size:.93rem;}'
       '.foot {margin-top:8px; font-size:.76rem; opacity:.6; line-height:1.35;}'
       '.status {display:flex; align-items:center; justify-content:space-between; gap:24px; border-left:7px solid var(--c);}'
       '.status.none {--c:#2ea043;} .status.watch {--c:#d29922;} .status.act {--c:#f85149;}'
       '.lvl {font-size:2.3rem; font-weight:800; line-height:1; color:var(--c); letter-spacing:.02em;}'
       '.head {font-size:1.05rem; font-weight:600; margin-top:6px;}'
       '.sub {font-size:.82rem; opacity:.65; margin-top:3px;}'
       '.kpi {text-align:right; font-size:2.1rem; font-weight:700; white-space:nowrap;}'
       '.kpi span {display:block; font-size:.74rem; font-weight:400; opacity:.65;}'
       '.srow {margin-bottom:11px;} .slab {font-size:.82rem; margin-bottom:4px;}'
       '.track {position:relative; height:8px; border-radius:4px; background:rgba(128,128,128,.25);}'
       '.fill {height:8px; border-radius:4px;} .fill.ok {background:#2ea043;} .fill.bad {background:#f85149;}'
       '.limit {position:absolute; left:70%; top:-3px; bottom:-3px; width:2px; background:currentColor; opacity:.75;}'
       '.sval {font-size:.74rem; opacity:.65; margin-top:3px;}'
       '.cols {display:flex; gap:36px;} .cols > div {flex:1;}'
       '.gt {font-size:.82rem; font-weight:600; margin-bottom:6px;}'
       '.mrow {display:flex; align-items:center; gap:10px; margin-bottom:7px;} .ml {width:74px; font-size:.82rem;}'
       '.mb {flex:1;} .mb div {height:6px; border-radius:3px; margin:2px 0;}'
       '.u {background:rgba(128,128,128,.55);} .n {background:#4c8dff;}'
       '.mv {width:92px; font-size:.78rem; text-align:right; opacity:.8; white-space:nowrap;}'
       '.step {display:flex; gap:12px; margin-bottom:8px; font-size:.93rem;}'
       '.step b {min-width:22px; height:22px; border-radius:11px; background:#4c8dff; color:#fff; text-align:center; line-height:22px; font-size:.8rem;}'
       '.chip {display:inline-block; padding:2px 10px; border-radius:10px; font-size:.78rem; font-weight:700; margin-right:8px; color:#fff;}'
       '</style>')


def inject_css(): st.markdown(CSS, unsafe_allow_html=True)


def esc(s): return html.escape(str(s))


def story(res):
    n, lvl, imp, cause = res['batch_size'], res['level'], res['impact'], res['cause']
    if lvl == 'none':
        return [f'We compared your {n} images with normal data for this domain and found nothing unusual. '
                'The deployed model should perform as it normally does.']
    who = 'Both checking models' if len(res['fired_by']) == 2 else SENSOR[res['fired_by'][0]]
    out = [f'We compared your {n} images with normal data. {who} found that this batch looks different from what the system was built on.']
    if cause == 'mix_shift':
        out.append('The pictures themselves look normal. What changed is the mix of scenes: the batch has a different share of '
                   'day, night, twilight or weather conditions than usual.')
    else:
        out.append(f'The most likely cause is {CAUSES.get(cause, cause)}. Changes like this make pictures harder for the model to read.')
    if lvl == 'act':
        out.append(f'Because of this, we expect the deployed model to be about {abs(imp) * 100:.0f} percentage points less accurate '
                   'on a batch like this than on normal data.')
        out.append('What to do: find out what changed in the data source (camera, lighting, weather, time of day or location). '
                   'If the change is lasting, collect and label images from the new conditions and retrain or fine-tune the model.')
    else:
        out.append(f'The change is not expected to hurt accuracy much (estimated {imp * 100:+.1f} points; the action level is a loss of 5 points), '
                   'so there is no need to act yet.')
        out.append('What to do: keep an eye on the next few batches. If the shift continues or grows, treat it as an action alert.')
    return out


def status_card(res, domain, n):
    lvl, cause = res['level'], res['cause']
    head = 'No drift detected' if lvl == 'none' else 'Drift detected: ' + ('the scene mix changed' if cause == 'mix_shift' else CAUSES.get(cause, cause))
    who = ' + '.join(SENSOR[k].split(' ')[0] for k in res['fired_by']) or 'no sensor'
    kpi = '-' if lvl == 'none' else f"{res['impact'] * 100:+.1f} pts"
    return (f'<div class="card status {lvl}"><div><div class="lvl">{lvl.upper()}</div><div class="head">{esc(head)}</div>'
            f'<div class="sub">{DOMAIN_NAME[domain]} &middot; {n} images &middot; flagged by {esc(who)}</div></div>'
            f'<div class="kpi">{kpi}<span>expected accuracy change</span></div></div>')


def story_card(res):
    ps = ''.join(f'<p><b>What to do:</b>{esc(p[11:])}</p>' if p.startswith('What to do:') else f'<p>{esc(p)}</p>' for p in story(res))
    return f'<div class="card"><h4>What happened</h4>{ps}</div>'


def sensor_card(scores, thr):
    rows = ''
    for (s, c), v, t in zip(CHECKS, scores, thr):
        w = min(v / t * 70, 100); cls = 'bad' if v > t else 'ok'
        rows += (f'<div class="srow"><div class="slab"><b>{s}</b> &middot; {c}</div>'
                 f'<div class="track"><div class="fill {cls}" style="width:{w:.0f}%"></div><div class="limit"></div></div>'
                 f'<div class="sval">score {v:.3f} &nbsp;/&nbsp; limit {t:.3f}</div></div>')
    return (f'<div class="card"><h4>Sensor checks</h4>{rows}'
            '<div class="foot">The bar is the batch score and the line is the limit. A bar past the line raises the alert.</div></div>')


def mix_card(mix):
    usual, now = dict(zip(NAMES, mix['usual'])), dict(zip(NAMES, mix['now']))
    cols = ''
    for title, items in GROUPS.items():
        rows = ''
        for label, f in items:
            u, w = sum(v for n, v in usual.items() if f(n)), sum(v for n, v in now.items() if f(n))
            rows += (f'<div class="mrow"><div class="ml">{label}</div><div class="mb"><div class="u" style="width:{u * 100:.0f}%"></div>'
                     f'<div class="n" style="width:{w * 100:.0f}%"></div></div><div class="mv">{u * 100:.0f}% &rarr; {w * 100:.0f}%</div></div>')
        cols += f'<div><div class="gt">{title}</div>{rows}</div>'
    return ('<div class="card"><h4>Scene mix: usual vs this batch (estimated)</h4>'
            f'<div class="cols">{cols}</div>'
            '<div class="foot">Grey = usual share, blue = this batch. The shares are estimated from the model&#39;s own predictions, '
            'so group totals (such as all twilight scenes) are more reliable than any single scene type.</div></div>')


def render(res, domain, thr, thumbs, n):
    st.markdown(status_card(res, domain, n), unsafe_allow_html=True)
    if res.get('note'): st.caption('Note: ' + res['note'])
    c1, c2 = st.columns([3, 2])
    c1.markdown(story_card(res), unsafe_allow_html=True)
    c2.markdown(sensor_card(res['scores'], thr), unsafe_allow_html=True)
    if res.get('scene_mix'): st.markdown(mix_card(res['scene_mix']), unsafe_allow_html=True)
    t1, t2 = st.tabs(['Batch preview', 'Raw output'])
    with t1:
        st.caption(f'First {len(thumbs)} of {n} images in the batch')
        st.image(thumbs, width=110)
    with t2: st.json(res)


def empty_state():
    steps = ''.join(f'<div class="step"><b>{i}</b><div>{t}</div></div>' for i, t in enumerate(
        ['Pick a domain in the sidebar.', 'Choose a sample batch, or upload about 100 images.', 'Click <b>Analyze batch</b>.'], 1))
    chips = ('<span class="chip" style="background:#2ea043">NONE</span> nothing unusual &nbsp; '
             '<span class="chip" style="background:#d29922">WATCH</span> drift, little expected loss &nbsp; '
             '<span class="chip" style="background:#f85149">ACT</span> drift, accuracy likely to drop')
    st.markdown(f'<div class="card"><h4>Drift monitor</h4>{steps}</div>'
                f'<div class="card"><h4>Alert levels</h4><p>{chips}</p></div>', unsafe_allow_html=True)


def about(samples_csv):
    st.markdown('**How it works**')
    st.markdown('1. **Embed.** Two models turn every image into a list of 768 numbers that describes what is in it. DINOv2 is the model being protected; ConvNeXt is an independent second opinion (the canary).  \n'
                '2. **Compare with normal.** The numbers are squeezed to 32 using a projection learned on clean, normal images.  \n'
                '3. **Score the batch.** Two checks per model: a K-S test (is the spread of the features different from normal?) and an autoencoder (does the batch look unusual to a model that only ever saw normal images?).  \n'
                '4. **Flag.** Each check has a limit, set so that a normal batch of 100 images crosses it only about 1% of the time. Any check over its limit raises an alert.  \n'
                '5. **Explain.** A random forest names the most likely cause: one of 15 image problems (blur, fog, noise...) or a change in the mix of scenes.  \n'
                '6. **Estimate impact.** For image problems, a regression model predicts how much accuracy the deployed model loses. For scene-mix changes, the deployed model\'s known accuracy on each scene type is combined with the estimated new mix.  \n'
                '7. **Tier.** NONE = no drift. WATCH = drift, but expected loss under 5 points. ACT = expected loss of 5 points or more.')
    st.markdown('**Results**')
    st.table(pd.DataFrame([
        ('False alarms on normal batches (1% budget)', '0.5-2% at 30, 50 and 100 images'),
        ('Cause attribution on corruptions (held out)', 'Places 96.1%, BDD 99.9%'),
        ('Accuracy-loss estimate, R2 (held out)', 'Places 0.88-0.89, BDD 0.83-0.84'),
        ('Scene-mix impact error (mean absolute)', '0.03-0.06 accuracy (corruption model alone: 0.05-0.12)'),
        ('Scene-mix shifts recognised as such (leave-one-group-out)', 'all six groups'),
    ], columns=['Result', 'Value']).set_index('Result'))
    st.markdown('**Known limitations.** Rain-heavy scene mixes are the weakest case (the estimate can come out as watch instead of act). '
                'The BDD test set is small, so expect roughly 1-4% false alarms on new BDD data. Impact models were trained on batches of 100 images, '
                'so estimates for other sizes are indicative. The scene-mix estimate exists only for BDD.')
    if samples_csv.exists():
        st.markdown('**Sample batches** and the alert each one gives')
        st.dataframe(pd.read_csv(samples_csv), hide_index=True)
