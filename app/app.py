from pathlib import Path
import streamlit as st
from PIL import Image
from core import Engine, SIZE
import ui

HERE = Path(__file__).parent; MD, SAMPLES = HERE / 'models', HERE / 'samples'
st.set_page_config(page_title='Drift Monitor', layout='wide', initial_sidebar_state='expanded')
ui.inject_css()


@st.cache_resource(show_spinner='Loading models (first run only)...')
def engine():
    return Engine(MD / 'detector_objects.joblib', MD / 'drift_artifacts.joblib',
                  MD / 'multitask_dino.pt', MD / 'multitask_convnext.pt', MD / 'dinov2_config')


def pretty(name):
    return name.replace('_mix_75', ' mix (75%)').replace('_s1', ' severity 1').replace('_s3', ' severity 3').replace('_', ' ').capitalize()


DOMAINS = {'bdd': 'BDD100K driving scenes', 'places': 'Places365 scenes'}
with st.sidebar:
    st.markdown('### Drift Monitor')
    st.caption('DINOv2 (deployed) with a ConvNeXt canary')
    domain = st.radio('Domain', list(DOMAINS), format_func=DOMAINS.get)
    folders = sorted(p.name for p in (SAMPLES / domain).glob('*') if p.is_dir()) if (SAMPLES / domain).exists() else []
    source = st.radio('Input', ['Sample batch', 'Upload images'], horizontal=True) if folders else 'Upload images'
    if source == 'Sample batch':
        pick = st.selectbox('Batch', folders, format_func=pretty)
        paths = sorted(p for p in (SAMPLES / domain / pick).iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
        sources = paths
    else:
        sources = st.file_uploader('Images', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True, label_visibility='collapsed') or []
    w, h = SIZE[domain]
    st.caption(f'{len(sources)} images ready. Resized to {w}x{h} before analysis. Best with 100 images; under 30 the alert is only indicative.')
    run = st.button('Analyze batch', type='primary', disabled=len(sources) == 0)

if run:
    imgs = [Image.open(s) for s in sources]
    eng = engine()
    with st.spinner('Embedding images and checking for drift...'):
        res = eng.analyze(imgs, domain)
    thumbs = []
    for im in imgs[:12]:
        t = im.convert('RGB'); t.thumbnail((240, 240)); thumbs.append(t)
    st.session_state['last'] = dict(res=res, domain=domain, n=len(imgs), thumbs=thumbs,
                                    thr=eng.monitor.art['thr_by_B'][domain][res['thr_batch_size']]['both'])

tab_a, tab_b = st.tabs(['Analysis', 'Method and results'])
with tab_a:
    last = st.session_state.get('last')
    if last: ui.render(last['res'], last['domain'], last['thr'], last['thumbs'], last['n'])
    else: ui.empty_state()
with tab_b:
    ui.about(SAMPLES / 'EXPECTED.csv')
