# ===== Palco Aberto · funções de preparo das músicas =====
import os, re, json, glob, subprocess, unicodedata, zipfile, time, traceback
import numpy as np

ENTRADA = '/content/musicas'
SAIDA = '/content/pacote'

def slug(t):
    t = unicodedata.normalize('NFD', t.lower())
    t = ''.join(c for c in t if unicodedata.category(c) != 'Mn')
    return re.sub(r'[^a-z0-9]+', '-', t).strip('-') or 'musica'

def ffmpeg_load(path, sr, ch):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-f', 'f32le', '-ac', str(ch), '-ar', str(sr), '-'],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).reshape(-1, ch).T.copy()

def salvar_mp3(audio, sr, destino, kbps):
    import soundfile as sf
    tmp = destino + '.wav'
    sf.write(tmp, np.clip(audio.T, -1, 1), sr)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', tmp, '-b:a', f'{kbps}k', destino], check=True)
    os.remove(tmp)

def ler_tags(path):
    title = artist = None
    try:
        from mutagen import File
        f = File(path, easy=True)
        if f is not None and f.tags:
            title = (f.tags.get('title') or [None])[0]
            artist = (f.tags.get('artist') or [None])[0]
    except Exception:
        pass
    base = os.path.splitext(os.path.basename(path))[0]
    base = re.sub(r'\s*[\(\[][^\)\]]*(official|oficial|lyric|letra|video|vídeo|audio|áudio|clipe|hd|4k|ao vivo)[^\)\]]*[\)\]]', '', base, flags=re.I)
    base = base.replace('_', ' ').strip()
    if not (title and artist):
        if ' - ' in base:
            a, t = base.split(' - ', 1)
            artist = artist or a.strip(); title = title or t.strip()
        else:
            title = title or base; artist = artist or ''
    return title.strip(), (artist or '').strip()

# ---------- 1. separar voz e instrumental (Demucs) ----------
_demucs = None
def separar(path, device):
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    global _demucs
    if _demucs is None:
        _demucs = get_model('htdemucs').to(device).eval()
    m = _demucs
    wav = torch.from_numpy(ffmpeg_load(path, m.samplerate, 2))
    ref = wav.mean(0); mean, std = ref.mean(), ref.std() + 1e-8
    with torch.no_grad():
        fontes = apply_model(m, ((wav - mean) / std)[None].to(device), device=device, split=True, overlap=0.25, progress=False)[0]
    fontes = (fontes * std + mean).cpu().numpy()
    voz = fontes[m.sources.index('vocals')]
    inst = fontes.sum(0) - voz
    return inst, voz, m.samplerate

# ---------- 2. letra sincronizada (LRCLIB) ----------
LRC_RE = re.compile(r'\[(\d+):(\d+(?:\.\d+)?)\]')
def parse_lrc(txt, dur):
    ev = []
    for raw in txt.splitlines():
        stamps = LRC_RE.findall(raw)
        text = LRC_RE.sub('', raw).strip()
        for mm, ss in stamps:
            ev.append((int(mm) * 60 + float(ss), text))
    ev.sort(key=lambda e: e[0])
    linhas = []
    for i, (t, x) in enumerate(ev):
        if not x or re.fullmatch(r'[♪♫\s\.\-…]*', x):
            continue
        fim = ev[i + 1][0] if i + 1 < len(ev) else dur
        linhas.append((t, fim, x))
    return linhas

def buscar_letra(title, artist, dur):
    import requests
    H = {'User-Agent': 'PalcoAberto/1.0 (notebook de karaoke)'}
    try:
        r = requests.get('https://lrclib.net/api/get', headers=H, timeout=20,
                         params={'track_name': title, 'artist_name': artist, 'duration': round(dur)})
        if r.ok:
            j = r.json()
            if j.get('syncedLyrics') or j.get('instrumental'):
                return j
        r = requests.get('https://lrclib.net/api/search', headers=H, timeout=20, params={'q': f'{artist} {title}'.strip()})
        if r.ok:
            c = [x for x in r.json() if x.get('syncedLyrics')]
            c.sort(key=lambda x: abs((x.get('duration') or 0) - dur))
            if c and abs((c[0].get('duration') or 0) - dur) <= 10:
                return c[0]
    except Exception as e:
        print('   LRCLIB indisponível:', e)
    return None

# ---------- 3. tempo de cada palavra (Whisper via stable-ts) ----------
_wm = None
def modelo_whisper(nome, device):
    global _wm
    import stable_whisper
    if _wm is None:
        _wm = stable_whisper.load_model(nome, device=device)
    return _wm

def detectar_idioma(model, voz16):
    try:
        import whisper as ow
        n = len(voz16)
        seg = voz16[n // 3: n // 3 + 16000 * 30] if n > 16000 * 45 else voz16
        seg = ow.pad_or_trim(np.ascontiguousarray(seg, dtype=np.float32))
        mel = ow.log_mel_spectrogram(seg, n_mels=model.dims.n_mels).to(model.device)
        _, probs = model.detect_language(mel)
        if isinstance(probs, list):
            probs = probs[0]
        return max(probs, key=probs.get)
    except Exception as e:
        print('   não detectei o idioma, usando pt:', e)
        return 'pt'

def mapear_palavras(texto, palavras_alinhadas, linhas):
    """Casa as palavras do alinhamento com as palavras da letra pela posição no texto."""
    spans, pos = [], 0
    for w, s, e in palavras_alinhadas:
        w = w.strip()
        if not w:
            continue
        i = texto.find(w, pos)
        if i < 0 or i - pos > 60:
            continue
        spans.append((i, i + len(w), float(s), float(e)))
        pos = i + len(w)
    out, off, j = [], 0, 0
    for (ls, le, txt) in linhas:
        words = []
        for m in re.finditer(r'\S+', txt):
            a, b = off + m.start(), off + m.end()
            while j < len(spans) and spans[j][1] <= a:
                j += 1
            hit = [sp for sp in spans[j:j + 10] if sp[0] < b and sp[1] > a]
            words.append({'t': m.group(), 's': min(h[2] for h in hit) if hit else None, 'e': max(h[3] for h in hit) if hit else None})
        out.append({'ls': ls, 'le': le, 'words': words})
        off += len(txt) + 1
    return out

def completar(linhas):
    """Preenche palavras sem tempo e corrige linhas que o alinhamento errou, usando o tempo do LRC."""
    final = []
    for l in linhas:
        ws, ls, le = l['words'], l['ls'], l['le']
        known = [w for w in ws if w['s'] is not None]
        bad = not known or (ls is not None and abs(known[0]['s'] - ls) > 4)
        if bad:
            if ls is None:
                continue
            fim = min(le, ls + 0.42 * len(ws) + 0.9) if le else ls + 0.42 * len(ws) + 0.9
            pesos = np.array([len(w['t']) + 2 for w in ws], float)
            cuts = ls + np.concatenate([[0], np.cumsum(pesos)]) / pesos.sum() * (fim - ls)
            for k, w in enumerate(ws):
                w['s'], w['e'] = float(cuts[k]), float(cuts[k + 1]) - 0.02
        else:
            k = 0
            while k < len(ws):
                if ws[k]['s'] is not None:
                    k += 1; continue
                a = k
                while k < len(ws) and ws[k]['s'] is None:
                    k += 1
                ini = ws[a - 1]['e'] if a > 0 else (ws[k]['s'] - 0.35 * (k - a) if k < len(ws) else ls)
                fim = ws[k]['s'] if k < len(ws) else ini + 0.35 * (k - a)
                if fim <= ini:
                    fim = ini + 0.2 * (k - a)
                passo = (fim - ini) / (k - a)
                for q in range(a, k):
                    ws[q]['s'] = ini + passo * (q - a); ws[q]['e'] = ini + passo * (q - a + 1) - 0.01
        prev = None
        for w in ws:
            if prev is not None and w['s'] < prev:
                w['s'] = prev
            w['e'] = max(w['e'], w['s'] + 0.08)
            w['e'] = min(w['e'], w['s'] + 4.0)
            prev = w['e']
        final.append({'s': round(ws[0]['s'], 3), 'e': round(ws[-1]['e'], 3),
                      'words': [{'t': w['t'], 's': round(w['s'], 3), 'e': round(w['e'], 3)} for w in ws]})
    final.sort(key=lambda l: l['s'])
    return final

def alinhar(model, voz_path, linhas, idioma):
    texto = '\n'.join(l[2] for l in linhas)
    res = model.align(voz_path, texto, language=idioma)
    aw = [(w.word, w.start, w.end) for w in res.all_words()]
    return completar(mapear_palavras(texto, aw, linhas))

def transcrever(model, voz_path, idioma):
    res = model.transcribe(voz_path, language=idioma)
    linhas, atual = [], []
    def fecha():
        if atual:
            linhas.append({'s': round(atual[0]['s'], 3), 'e': round(atual[-1]['e'], 3), 'words': list(atual)})
            atual.clear()
    for seg in res.segments:
        for w in seg.words:
            t = w.word.strip()
            if not t:
                continue
            if atual and (len(atual) >= 8 or w.start - atual[-1]['e'] > 1.0):
                fecha()
            atual.append({'t': t, 's': round(float(w.start), 3), 'e': round(float(w.end), 3)})
        fecha()
    return linhas

# ---------- 4. melodia da voz original (CREPE) ----------
def melodia(voz, sr, device):
    import torch, torchaudio, torchcrepe
    mono = voz.mean(0) if voz.ndim > 1 else voz
    x16 = torchaudio.functional.resample(torch.from_numpy(np.ascontiguousarray(mono, dtype=np.float32))[None], sr, 16000)
    hop = 160
    f0, per = torchcrepe.predict(x16, 16000, hop, fmin=65, fmax=1000, model='full' if device == 'cuda' else 'tiny',
                                 decoder=torchcrepe.decode.viterbi, return_periodicity=True, batch_size=1024, device=device)
    per = torchcrepe.filter.median(per, 3)
    f0, per = f0[0].cpu().numpy(), per[0].cpu().numpy()
    a = x16[0].numpy(); n = len(f0)
    pad = np.pad(a, (0, hop * n + 1024))
    rms = np.sqrt(np.array([np.mean(pad[i * hop:i * hop + 1024] ** 2) for i in range(n)]) + 1e-12)
    db = 20 * np.log10(rms)
    voiced = (per > 0.45) & (db > np.percentile(db, 98) - 32)
    midi = 69 + 12 * np.log2(np.maximum(f0, 1e-3) / 440)
    t = np.arange(n) * hop / 16000
    return t, midi, voiced

def segmentar(t, midi, voiced, janelas, min_dur=0.09, salto=0.75, hop=0.01):
    """Transforma a curva de afinação em notas (início, fim, altura) dentro das linhas da letra."""
    notas = []
    def fecha(c):
        if c and c[1] - c[0] + hop >= min_dur:
            notas.append({'s': round(c[0], 3), 'e': round(c[1] + hop, 3), 'p': round(float(np.median(c[2])), 2)})
    for ws, we in janelas:
        idx = np.where((t >= ws - 0.15) & (t <= we + 0.4))[0]
        cur = None
        for i in idx:
            if not voiced[i]:
                fecha(cur); cur = None; continue
            v = float(midi[i])
            if cur is None:
                cur = [t[i], t[i], [v]]
            else:
                med = float(np.median(cur[2][-8:]))
                if abs(v - med) > salto or t[i] - cur[1] > 0.06:
                    fecha(cur); cur = [t[i], t[i], [v]]
                else:
                    cur[1] = t[i]; cur[2].append(v)
        fecha(cur)
    notas.sort(key=lambda n: n['s'])
    junt = []
    for n in notas:
        if junt and n['s'] - junt[-1]['e'] < 0.05 and abs(n['p'] - junt[-1]['p']) < 0.5:
            junt[-1]['e'] = max(junt[-1]['e'], n['e'])
        elif junt and n['s'] < junt[-1]['e']:
            n['s'] = junt[-1]['e']
            if n['e'] - n['s'] >= min_dur:
                junt.append(n)
        else:
            junt.append(n)
    return junt

# ---------- tudo junto ----------
def processar(path, device, whisper_nome='medium', idioma=None):
    import soundfile as sf
    t0 = time.time()
    title, artist = ler_tags(path)
    print(f'▶ {artist + " - " if artist else ""}{title}')
    pasta = os.path.join(SAIDA, slug(f'{artist}-{title}'))
    os.makedirs(pasta, exist_ok=True)
    print('   separando a voz (Demucs)…')
    inst, voz, sr = separar(path, device)
    dur = inst.shape[1] / sr
    salvar_mp3(inst, sr, os.path.join(pasta, 'instrumental.mp3'), 160)
    voz_mono = voz.mean(0)
    salvar_mp3(voz_mono[None], sr, os.path.join(pasta, 'vocals.mp3'), 96)
    voz_path = os.path.join(pasta, '_voz.wav'); sf.write(voz_path, voz_mono, sr)
    model = modelo_whisper(whisper_nome, device)
    lang = idioma or detectar_idioma(model, ffmpeg_load(voz_path, 16000, 1)[0])
    print('   idioma:', lang, '· buscando letra no LRCLIB…')
    lr = buscar_letra(title, artist, dur)
    linhas_lrc = parse_lrc(lr['syncedLyrics'], dur) if lr and lr.get('syncedLyrics') else []
    if linhas_lrc:
        fonte = 'lrclib'
        print(f'   letra encontrada ({len(linhas_lrc)} linhas) · alinhando palavra por palavra…')
        lines = alinhar(model, voz_path, linhas_lrc, lang)
    else:
        fonte = 'whisper'
        print('   letra não encontrada · o Whisper vai escrever a letra ouvindo a voz…')
        lines = transcrever(model, voz_path, lang)
    print('   tirando a melodia (CREPE)…')
    t, midi, voiced = melodia(voz_mono, sr, device)
    notes = segmentar(t, midi, voiced, [(l['s'], l['e']) for l in lines])
    meta = {
        'version': 1, 'title': title, 'artist': artist, 'duration': round(dur, 3), 'language': lang,
        'lyricsSource': fonte, 'audio': {'instrumental': 'instrumental.mp3', 'vocals': 'vocals.mp3'},
        'lines': lines, 'notes': notes
    }
    with open(os.path.join(pasta, 'song.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False)
    os.remove(voz_path)
    print(f'   pronto em {time.time() - t0:.0f}s · {len(lines)} linhas · {len(notes)} notas · letra: {fonte}')
    return meta, (t, midi, voiced)
