#!/usr/bin/env python3
"""Build a karaoke-style .ass subtitle file from timed transcript segments.

Input is a JSON list of segments, each the exact line for one shot plus its
absolute start/end on the final timeline (the assembler computes these from clip
durations):

    [{"text": "Bună dimineața!", "start": 0.0, "end": 3.4}, ...]

By default word timing inside each segment is split evenly (no extra deps, fully
deterministic — the transcript is already known/exact, so this looks fine on short
reel shots). If --audio is given AND `whisperx` is importable, words are instead
force-aligned to the audio for tighter karaoke; it silently falls back to the even
split if whisperx or the alignment model isn't available.

Usage:
    python scripts/gen_subs.py --segments segs.json --out captions.ass \
        --play-res 1080x1920 [--audio full.wav --language ro]
"""
import argparse
import json


def _cs(t):
    """seconds -> ASS time 'H:MM:SS.cs' (centiseconds)."""
    t = max(0.0, float(t))
    h = int(t // 3600); t -= h * 3600
    m = int(t // 60); t -= m * 60
    s = int(t)
    cs = int(round((t - s) * 100))
    if cs == 100:
        s += 1; cs = 0
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def even_words(text, start, end):
    """(word, w_start, w_end) split evenly across [start, end]."""
    words = text.split()
    if not words:
        return []
    step = (end - start) / len(words)
    return [(w, start + i * step, start + (i + 1) * step) for i, w in enumerate(words)]


def aligned_words(segments, audio, language, device):
    """Force-align words to audio with whisperx. Returns per-segment word lists,
    or None if whisperx is unavailable / fails."""
    try:
        import whisperx
    except Exception as e:
        print(f"[subs] whisperx not available ({e}); using even split")
        return None
    try:
        a = whisperx.load_audio(audio)
        model_a, meta = whisperx.load_align_model(language_code=language, device=device)
        segs = [{"text": s["text"], "start": s["start"], "end": s["end"]} for s in segments]
        res = whisperx.align(segs, model_a, meta, a, device, return_char_alignments=False)
        # Group the flat word list back under each input segment by time overlap.
        words = res.get("word_segments", [])
        out = []
        for s in segments:
            ws = [(w["word"], w.get("start", s["start"]), w.get("end", s["end"]))
                  for w in words if w.get("start") is not None
                  and w["start"] >= s["start"] - 0.25 and w["start"] <= s["end"] + 0.25]
            out.append(ws or even_words(s["text"], s["start"], s["end"]))
        return out
    except Exception as e:
        print(f"[subs] whisperx align failed ({e}); using even split")
        return None


ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Karaoke,{font},{fs},&H0000FFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,{outline},{shadow},2,80,80,{marginv},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def chunk_cards(words, max_chars, max_words):
    """Group a segment's words into short on-screen 'cards' (reel-style pop
    captions) so a card always fits one line instead of dumping the whole
    sentence. Start a new card when adding the next word would exceed either the
    character budget or the word count."""
    cards, cur, n = [], [], 0
    for w in words:
        wl = len(w[0])
        if cur and (len(cur) >= max_words or n + 1 + wl > max_chars):
            cards.append(cur)
            cur, n = [], 0
        cur.append(w)
        n += wl + (1 if n else 0)
    if cur:
        cards.append(cur)
    return cards


def build_ass(segments, words_per_seg, play_w, play_h, font, font_size,
              max_chars, max_words):
    out = [ASS_HEADER.format(w=play_w, h=play_h, font=font, fs=font_size,
                             outline=max(2, font_size // 16),
                             shadow=max(1, font_size // 32),
                             marginv=int(play_h * 0.18))]
    for seg, words in zip(segments, words_per_seg):
        if not words:
            continue
        cards = chunk_cards(words, max_chars, max_words)
        for i, card in enumerate(cards):
            disp_start = card[0][1]
            # Hold each card until the next one starts (last card runs to seg end)
            # so captions never blink to black between cards.
            disp_end = cards[i + 1][0][1] if i + 1 < len(cards) else seg["end"]
            chunks = []
            for w, ws, we in card:
                k = max(1, int(round((we - ws) * 100)))  # karaoke duration in cs
                chunks.append(f"{{\\kf{k}}}{w} ")
            line = "".join(chunks).rstrip()
            out.append(f"Dialogue: 0,{_cs(disp_start)},{_cs(disp_end)},Karaoke,,0,0,0,,{line}")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", required=True, help="JSON list of {text,start,end}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--play-res", default="1080x1920")
    ap.add_argument("--font", default="Arial")
    ap.add_argument("--font-size", type=int, default=84)
    ap.add_argument("--max-chars", type=int, default=18, help="max chars per caption card")
    ap.add_argument("--max-words", type=int, default=3, help="max words per caption card")
    ap.add_argument("--audio", default=None, help="optional wav for whisperx alignment")
    ap.add_argument("--language", default="ro")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    with open(args.segments) as f:
        segments = [s for s in json.load(f) if s.get("text", "").strip()]
    play_w, play_h = (int(x) for x in args.play_res.lower().split("x"))

    words_per_seg = None
    if args.audio:
        words_per_seg = aligned_words(segments, args.audio, args.language, args.device)
    if words_per_seg is None:
        words_per_seg = [even_words(s["text"], s["start"], s["end"]) for s in segments]

    ass = build_ass(segments, words_per_seg, play_w, play_h, args.font, args.font_size,
                    args.max_chars, args.max_words)
    import os
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(ass)
    print(f"[subs] wrote {args.out} ({len(segments)} segment(s))")


if __name__ == "__main__":
    main()
