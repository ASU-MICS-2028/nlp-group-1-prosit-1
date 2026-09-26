"""
Builds the 10-minute Prosit 1 group deck on the official Ashesi "Presentation Red" template, with the same
template and slide helpers as the ICS553 Machine Learning Prosit 1 deck (scripts/build_ashesi_deck.py there).

Every number on a slide is read from a result file when the deck is built, and each slide's speaker notes
name the file and key it comes from (see reports/claims_table.md), so rebuilding after a rerun updates the
slides instead of leaving stale numbers behind.

Run from the repo root:  python presentation/build_deck.py
Writes presentation/Prosit1_Language_Models.pptx.
"""

import json
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_TICK_LABEL_POSITION, XL_TICK_MARK
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TEMPLATE = HERE / "ashesi_presentation_red.pptx"
OUT = HERE / "Prosit1_Language_Models.pptx"
REPO_URL = "github.com/ASU-MICS-2028/nlp-group-1-prosit-1"

RED, GOLD, INK, MUTED = "AB3D3F", "FEBA5A", "262626", "595959"
CARD, GRID, GREY, WHITE = "F3F3F3", "D9D9D9", "A6A6A6", "FFFFFF"
FONT = "Candara"  # template body font; titles keep the layouts' Poppins
EWE_FONT = "Arial"  # has every Ewe letter (checked with fontTools); Candara may not
EWE_LETTERS = set("ɖƒɣŋɔɛʋƉƑƔŊƆƐƲ̃")
TITLE, TEXT_HEAVY, TABLE, CLOSING = 0, 3, 7, 8  # template layout indices
W = 12.09  # content width from the 0.62" margin; the wordmark sits below y = 6.5"


# ---------------------------------------------------------------- text helpers (as in the ICS553 builder)
def rgb(h):
    return RGBColor.from_string(h)


def P(*runs, bullet=False, after=None, before=None, align=None, s=None, c=None):
    """One paragraph: runs are str or (str, style) with style keys b, c, s, i."""
    return dict(runs=runs, bullet=bullet, after=after, before=before, align=align, s=s, c=c)


def B(t, **st):
    return (t, {"b": True, **st})


def bulletize(p, color=RED, indent=0.24):
    ppr = p._p.get_or_add_pPr()
    ppr.set("marL", str(Inches(indent)))
    ppr.set("indent", str(-Inches(indent)))
    for tag in ("a:buNone", "a:buClr", "a:buFont", "a:buChar", "a:buAutoNum"):
        for el in ppr.findall(qn(tag)):
            ppr.remove(el)
    etree.SubElement(etree.SubElement(ppr, qn("a:buClr")), qn("a:srgbClr")).set("val", color)
    etree.SubElement(ppr, qn("a:buFont")).set("typeface", "Arial")
    etree.SubElement(ppr, qn("a:buChar")).set("char", "•")


def fill(tf, paras, size=14, color=INK, align=PP_ALIGN.LEFT, space=6):
    tf.clear()
    for i, para in enumerate(paras):
        para = para if isinstance(para, dict) else P(para)
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = para["align"] or align
        p.space_after = Pt(space if para["after"] is None else para["after"])
        if para["before"] is not None:
            p.space_before = Pt(para["before"])
        for run in para["runs"]:
            t, st = (run, {}) if isinstance(run, str) else run
            r = p.add_run()
            r.text = t
            f = r.font
            f.name = EWE_FONT if EWE_LETTERS & set(t) else FONT
            f.size = Pt(st.get("s", para["s"] or size))
            f.bold = st.get("b", False)
            f.italic = st.get("i", False)
            f.color.rgb = rgb(st.get("c", para["c"] or color))
        if para["bullet"]:
            bulletize(p)


def text(slide, x, y, w, h, paras, anchor=MSO_ANCHOR.TOP, **kw):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    fill(tf, paras, **kw)
    return tb


def card(slide, x, y, w, h, paras, bg=CARD, pad=0.22, anchor=MSO_ANCHOR.TOP, **kw):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.adjustments[0] = 0.05
    s.fill.solid()
    s.fill.fore_color.rgb = rgb(bg)
    s.line.fill.background()
    s.shadow.inherit = False
    tf = s.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(pad)
    tf.margin_top = tf.margin_bottom = Inches(pad * 0.8)
    if paras:
        fill(tf, paras, **kw)
    return s


def title(slide, idx, t, size=None, width=None):
    ph = slide.placeholders[idx]
    if width:  # override all four so the inherited position is kept
        ph.left, ph.top, ph.width, ph.height = ph.left, ph.top, Inches(width), ph.height
    ph.text_frame.text = t
    if size:
        ph.text_frame.paragraphs[0].runs[0].font.size = Pt(size)
    return ph


def drop(slide, *idxs):
    for i in idxs:
        el = slide.placeholders[i]._element
        el.getparent().remove(el)


def notes(slide, t):
    slide.notes_slide.notes_text_frame.text = t


# ---------------------------------------------------------------- charts and tables (as in the ICS553 builder)
def bar_chart(slide, x, y, w, h, cats, vals, colors, fmt="#,##0", horizontal=False, axis=True):
    cd = CategoryChartData()
    cd.categories = cats
    cd.add_series("value", vals)
    kind = XL_CHART_TYPE.BAR_CLUSTERED if horizontal else XL_CHART_TYPE.COLUMN_CLUSTERED
    ch = slide.shapes.add_chart(kind, Inches(x), Inches(y), Inches(w), Inches(h), cd).chart
    ch.has_legend = False
    ch.has_title = False
    ch.font.name = FONT
    ch.font.size = Pt(12)
    ch.font.color.rgb = rgb(INK)
    plot = ch.plots[0]
    plot.gap_width = 60
    series = plot.series[0]
    series.invert_if_negative = False
    series.format.fill.solid()  # added: some viewers ignore per-bar colours, so set the series colour too
    series.format.fill.fore_color.rgb = rgb(GREY if GREY in colors else colors[0])
    for pt, c in zip(series.points, colors):
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = rgb(c)
    for dpt in series._element.findall(qn("c:dPt")):  # PowerPoint inverts negative points without this
        flag = etree.Element(qn("c:invertIfNegative"))
        flag.set("val", "0")
        dpt.find(qn("c:idx")).addnext(flag)
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format, dl.number_format_is_linked = fmt, False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size, dl.font.bold = Pt(13), True
    va, ca = ch.value_axis, ch.category_axis
    va.has_major_gridlines = axis
    if axis:
        va.major_gridlines.format.line.color.rgb = rgb(GRID)
    va.format.line.fill.background()
    va.major_tick_mark = XL_TICK_MARK.NONE
    va.tick_labels.font.size = Pt(11)
    va.tick_labels.font.color.rgb = rgb(MUTED)
    va.tick_labels.number_format, va.tick_labels.number_format_is_linked = fmt, False
    va.visible = axis
    ca.format.line.color.rgb = rgb(GREY)
    ca.major_tick_mark = XL_TICK_MARK.NONE
    ca.tick_labels.font.size = Pt(12)
    if horizontal:
        ca.tick_label_position = XL_TICK_LABEL_POSITION.LOW
    return ch


def table(slide, x, y, col_w, rows, row_h, size=12, right_cols=(), bold_rows=()):
    tbl = slide.shapes.add_table(
        len(rows), len(col_w), Inches(x), Inches(y), Inches(sum(col_w)), Inches(sum(row_h))
    ).table
    tbl.first_row, tbl.horz_banding = True, False
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h[i])
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(RED if i == 0 else (WHITE if i % 2 else CARD))
            cell.margin_left = cell.margin_right = Inches(0.1)
            cell.margin_top = cell.margin_bottom = Inches(0.04)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            bold = i == 0 or j == 0 or i in bold_rows
            align = PP_ALIGN.RIGHT if j in right_cols else PP_ALIGN.LEFT
            fill(
                cell.text_frame,
                [P((val, {"b": bold}))],
                size=size + (1 if i == 0 else 0),
                color=WHITE if i == 0 else INK,
                align=align,
                space=0,
            )
    return tbl


# ---------------------------------------------------------------- the numbers
def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def down(old, new):
    """Percentage fall from old to new, one decimal."""
    return f"{100 * (old - new) / old:.1f}%"


def up(old, new):
    return f"{100 * (new - old) / old:.1f}%"


def lower_by(old, new):
    """Whole-percent fall, for the headline comparisons."""
    return f"{100 * (old - new) / old:.0f}%"


# ---------------------------------------------------------------- the deck
def build():
    uni = load("results/section_b_ngram/unified.json")
    word = uni["results"]["Unicode Word"]  # one row per order N = 1..6
    best = uni["best_order_by_val"]
    lstm = load("results/section_b_lstm/lstm_vs_ngram.json")
    lora = load("results/section_c_llm/lora_results.json")
    base, std, masked = (lora["results"][k] for k in ("base", "standard", "masked"))
    dist3 = load("results/section_c_llm/decoding_benchmark.json")["strategy_averages"]
    ewe = load("data/processed/unified/stats.json")
    agri = load("data/processed/domain_english/stats.json")
    d2_words = load("data/processed/dataset_2_json/stats.json")["train_words"]

    prs = Presentation(TEMPLATE)
    n_template = len(prs.slides)
    add = lambda layout: prs.slides.add_slide(prs.slide_layouts[layout])  # noqa: E731

    # 1. Title
    s = add(TITLE)
    title(s, 0, "Building and adapting language models", size=54)
    fill(
        s.placeholders[1].text_frame,
        [
            P("An Ewe n-gram model and an agricultural English model for Ankora", s=22, c=GOLD, after=10),
            P("ICS554 Natural Language Processing · MICS 2028 · Group 1 · Ashesi University · September 2026", s=16, c=WHITE, after=6),
            P(REPO_URL, s=16, c=WHITE),
        ],
        align=PP_ALIGN.CENTER,
    )
    notes(
        s,
        "Say: Ankora builds speech recognition, and a recogniser needs a language model to choose between "
        "candidate transcripts. We built one for Ewe, a low-resource Ghanaian language, and adapted one to "
        "agricultural English. (About 45 seconds.)\n"
        "The repository link must stay on this slide (assignment requirement). "
        "Add the presenters' names to this slide before presenting.",
    )

    # 2. Three models
    s = add(TEXT_HEAVY)
    title(s, 11, "Three models, kept apart", width=W)
    drop(s, 12, 13)
    text(
        s, 0.62, 1.55, W, 0.5,
        [P("Two deliverables, plus one baseline to test a claim the brief makes.")],
        size=18,
    )
    models = [
        ("N-gram", "Section B", f"Ewe · {ewe['total_unique_sentences']:,} sentences",
         "Counts plus Kneser-Ney smoothing. No neural network.",
         f"Best: {best['Byte-Pair Encoding (BPE)']['per_word_perplexity']:.1f} perplexity per word (BPE subwords)"),
        ("LSTM baseline", "Section B, Question 2", "The same Ewe sentences and tokens",
         "A small network trained from scratch, to test whether n-grams beat neural models on little data.",
         f"Loses on {lstm['2']['train_sentences']} sentences, wins on {lstm['unified']['train_sentences']:,}"),
        ("distilgpt2 + LoRA", "Section C", f"English · {agri['unique_questions']:,} farming questions",
         f"A pretrained model; LoRA trains {100 * std['trainable_params'] / std['total_params']:.2f}% of its weights.",
         f"Answer perplexity {base['answer_ppl']:.2f} → {std['answer_ppl']:.2f}"),
    ]
    for i, (name, section, data, what, result) in enumerate(models):
        card(
            s, 0.62 + i * 4.13, 2.2, 3.83, 3.0,
            [
                P(B(name, c=RED, s=26), after=2),
                P(B(section, s=16), after=10),
                P(data, s=15, after=8),
                P(what, s=14, c=MUTED, after=12),
                P(B(result, s=15, c=RED)),
            ],
        )
    text(
        s, 0.62, 5.45, W, 0.8,
        [
            P(
                B("The LSTM and the LLM are unrelated: ", c=RED),
                "different language, data and model. Perplexities compare within Section B, never across sections.",
            )
        ],
        size=16,
    )
    notes(
        s,
        "Say: There are three models, and it helps to keep them apart. The n-gram is our Section B model for "
        "Ewe. The LSTM exists only to answer Section B's question two, whether n-grams beat neural models, so "
        "it uses exactly the n-gram's data and tokens. Section C is a different model altogether: a "
        "pretrained English model adapted with LoRA. (About 60 seconds.)\n"
        "Sources: data/processed/unified/stats.json (total_unique_sentences); "
        "data/processed/domain_english/stats.json (unique_questions); "
        "results/section_c_llm/lora_results.json (results.standard.trainable_params / total_params).",
    )

    # 3. The Ewe corpus
    s = add(TEXT_HEAVY)
    title(s, 11, "Section B: the Ewe corpus", width=W)
    drop(s, 12, 13)
    for i, (num, lab, sub) in enumerate(
        [
            (f"{ewe['total_unique_sentences']:,}", "sentences from four sources",
             "Deduplicated within and across sources, then split 80/10/10 with seed 42"),
            (f"{ewe['train_words']:,}", "training words", f"{ewe['train_sentences']:,} training sentences"),
            (f"{ewe['pct_mentioning_yehowa']}%", "of sentences mention Yehowa",
             "A large share is Bible and Jehovah's Witnesses text, and the model leans that way"),
        ]
    ):
        card(
            s, 0.62, 1.55 + i * 1.6, 5.9, 1.45,
            [P(B(num, c=RED, s=30), after=0), P(B(lab, s=15), after=2), P(sub, s=12, c=MUTED)],
            anchor=MSO_ANCHOR.MIDDLE,
        )
    card(
        s, 6.81, 1.55, 5.9, 4.65,
        [
            P(B("Getting the letters right", c=RED, s=18), after=10),
            P("Sources: sentence pairs, dictionary examples, spoken image descriptions (University of Ghana, "
              "Waxal) and a large aligned corpus.", bullet=True, after=8),
            P("Unicode NFC, so each letter and tone mark is stored one way. The nasal vowel ", ("ɔ̃", {}),
              " has no single Unicode character, so our tokenizers accept combining marks.", bullet=True, after=8),
            P("Capital eth ", ("Ð", {}), " typed for ", ("Ɖ", {}), " had split ", ("ɖe", {}),
              " into two words; the cleaner maps lookalikes back.", bullet=True, after=8),
            P("Vocabulary from the training split only; words seen once become <unk>.", bullet=True),
        ],
        size=14,
    )
    notes(
        s,
        "Say: We combined four existing Ewe sources. Be upfront that a large share is religious text, which "
        "shows up in what the model generates. Ewe spelling needed care: some nasal vowels are two Unicode "
        "code points, and a lookalike letter was splitting common words. (About 60 seconds.)\n"
        "Sources: data/processed/unified/stats.json (total_unique_sentences, train_words, train_sentences, "
        "pct_mentioning_yehowa); data/README.md for the four sources; claims table rows 1 to 3 and 16.",
    )

    # 4. Context length
    ppl = [r["perplexity"] for r in word]
    val4to6 = [r["val_perplexity"] for r in word[3:6]]
    s = add(TEXT_HEAVY)
    title(s, 11, "Longer context helps, up to about four words", width=W)
    drop(s, 12, 13)
    text(
        s, 0.62, 1.5, 6.4, 0.6,
        [P(B("Test perplexity by n-gram order"), " (Unicode Word tokens, Kneser-Ney; lower is better)")],
        size=14,
    )
    bar_chart(s, 0.62, 2.0, 6.4, 4.2, [f"N = {r['order']}" for r in word], [round(v, 1) for v in ppl],
              [GREY] * 3 + [RED] * 3, fmt="0.0", axis=False)
    card(
        s, 7.33, 1.55, 5.38, 4.65,
        [
            P(B("Why smoothing decides this", c=RED, s=18), after=10),
            P(f"{word[1]['sparsity_pct']:.1f}% of test bigrams and {word[5]['sparsity_pct']:.1f}% of test 6-grams "
              "never occur in training; maximum likelihood gives them probability 0.", bullet=True, after=8),
            P("Kneser-Ney hands an unseen long context's probability down to shorter contexts, so perplexity "
              "levels off instead of rising.", bullet=True, after=8),
            P(f"It beats equal-weight interpolation at every N ≥ 2 ({word[2]['perplexity']:.1f} against "
              f"{word[2]['interpolation_perplexity']:.1f} at N = 3).", bullet=True, after=8),
            P(f"N is chosen on validation: N = 4, 5 and 6 are within {100 * (max(val4to6) / min(val4to6) - 1):.0f}% "
              "of each other.", bullet=True),
        ],
        size=14,
    )
    notes(
        s,
        "Say: More context helps a lot at first, then stops helping at about four words, even though most "
        "long contexts were never seen in training. With correct smoothing a longer context never hurts. Our "
        "first draft showed perplexity rising after N = 4; that turned out to be three bugs, not the language. "
        "(About 60 seconds.)\n"
        "Sources: results/section_b_ngram/unified.json, results['Unicode Word'][N-1]: perplexity, "
        "sparsity_pct, interpolation_perplexity, val_perplexity; claims table rows 4, 5, 8 and 10.",
    )

    # 5. Tokenizers
    order = ["Character", "Whitespace", "Unicode Word", "Ewe Stemmer (affixes kept)", "Byte-Pair Encoding (BPE)"]
    label = {"Ewe Stemmer (affixes kept)": "Ewe stemmer, affixes kept", "Byte-Pair Encoding (BPE)": "BPE subwords"}
    per_word = {t: best[t]["per_word_perplexity"] for t in order}
    s = add(TEXT_HEAVY)
    title(s, 11, "Which unit to count: five tokenizers", width=W)
    drop(s, 12, 13)
    text(
        s, 0.62, 1.5, 6.4, 0.6,
        [P(B("Test perplexity per word"), " (best N per tokenizer; lower is better)")],
        size=14,
    )
    bar_chart(s, 0.62, 2.0, 6.4, 4.2, [label.get(t, t) for t in order], [round(per_word[t], 1) for t in order],
              [GREY] * 4 + [RED], fmt="0.0", horizontal=True, axis=False)
    card(
        s, 7.33, 1.55, 5.38, 4.65,
        [
            P(B("Compare per word, not per token", c=RED, s=18), after=10),
            P(f"Per token, characters look best ({best['Character']['perplexity']:.1f}): a character model chooses "
              f"among {uni['results']['Character'][0]['vocab_size']} symbols, a word model among "
              f"{uni['results']['Unicode Word'][0]['vocab_size']:,}. Only per word do all five score the same text.",
              bullet=True, after=8),
            P("Each <unk> also pays to spell its word, or the tokenizer with the most unknown words looks best.",
              bullet=True, after=8),
            P(f"Keeping Ewe affixes as tokens beats plain words by "
              f"{down(per_word['Unicode Word'], per_word['Ewe Stemmer (affixes kept)'])}; attaching punctuation "
              f"to words costs {100 * (per_word['Whitespace'] / per_word['Unicode Word'] - 1):.0f}%.",
              bullet=True, after=8),
            P("The same ranking holds on each of the four source datasets.", bullet=True),
        ],
        size=14,
    )
    notes(
        s,
        "Say: Tokenizers split the same text into different units, so their per-token perplexities measure "
        "different things. Per word they are comparable, as long as a model that predicts <unk> also pays to "
        "spell the word. On that basis subword units (BPE) are best, and characters are last because six "
        "characters cover only about one word of context. One exception to the ranking: on the smallest "
        "dataset, characters beat whitespace tokens. (About 60 seconds.)\n"
        "Sources: results/section_b_ngram/unified.json, best_order_by_val[...].per_word_perplexity and "
        ".perplexity; results[...][0].vocab_size; per-dataset files results/section_b_ngram/dataset_1..4.json; "
        "claims table rows 6, 11 and 12.",
    )

    # 6. Question 2: n-gram or neural
    d2, full = lstm["2"], lstm["unified"]
    hours = [r["train_seconds"] / 3600 for r in full["runs"]]
    s = add(TEXT_HEAVY)
    title(s, 11, "Question 2: are n-grams better than neural models?", size=30, width=W)
    drop(s, 12, 13)
    for x, e, words, digits, verdict in [
        (0.62, d2, d2_words, 0,
         f"n-gram {lower_by(d2['lstm_per_word_mean'], d2['kn_bpe']['per_word_perplexity'])} lower. "
         f"About {d2['runs'][0]['params'] / d2_words:.0f} LSTM weights per training word is more than the data can pin down."),
        (6.81, full, ewe["train_words"], 1,
         f"LSTM {lower_by(full['kn_bpe']['per_word_perplexity'], full['lstm_per_word_mean'])} lower, and still "
         "improving when our 10-epoch budget ran out."),
    ]:
        text(s, x, 1.5, 5.9, 0.4,
             [P(B(f"{e['train_sentences']:,} training sentences"), f" ({words:,} words)")], size=15)
        kn, nn = e["kn_bpe"]["per_word_perplexity"], e["lstm_per_word_mean"]
        bar_chart(s, x, 1.95, 5.9, 2.75, ["n-gram (Kneser-Ney)", "LSTM (mean of 3 seeds)"],
                  [round(kn, digits), round(nn, digits)], [RED, GREY] if kn < nn else [GREY, RED],
                  fmt="#,##0" if digits == 0 else "0.0", axis=False)
        text(s, x, 4.75, 5.9, 0.7, [P(verdict)], size=14)
    card(
        s, 0.62, 5.5, W, 0.8,
        [
            P(B("Our answer: ", c=RED),
              "n-grams for very small data or tight compute; a neural model once there is enough text. Same BPE "
              f"tokens and test sentences; every seed lands on the same side. Each large LSTM took {min(hours):.1f} "
              f"to {max(hours):.1f} hours on a CPU, all six n-gram orders about 7 minutes.")
        ],
        size=14,
        anchor=MSO_ANCHOR.MIDDLE,
        pad=0.18,
    )
    notes(
        s,
        "Say: We tested the brief's claim instead of assuming it. The LSTM reads exactly the n-gram's tokens "
        "and is scored the same way. On 420 sentences the n-gram wins clearly; on our full corpus the LSTM "
        "wins by about an eighth, at hours of CPU instead of minutes. So the answer depends on how much text "
        "you have. This LSTM is only a Section B baseline; it is not the Section C model. (About 75 seconds.)\n"
        "Sources: results/section_b_lstm/lstm_vs_ngram.json, ['2'] and ['unified']: lstm_per_word_mean, "
        "lstm_per_word_min/max, kn_bpe.per_word_perplexity, runs[].params, runs[].train_seconds, "
        "runs[].best_epoch; training words from data/processed/{dataset_2_json,unified}/stats.json. The n-gram's "
        "7 minutes is 425 s from the sweep log (WORKLOG.md, claims table row 33). Claims table rows 28 to 34.",
    )

    # 7. Section C setup
    s = add(TEXT_HEAVY)
    title(s, 11, "Section C: adapting distilgpt2 to farming questions", size=30, width=W)
    drop(s, 12, 13)
    card(
        s, 0.62, 1.55, 3.83, 4.65,
        [
            P(B("The data", c=RED, s=18), after=10),
            P(B(f"{agri['raw_rows']:,} rows, but only {agri['unique_questions']:,} distinct questions."), after=8),
            P(f"One row per question, kept before splitting: {agri['train_pairs']:,} / {agri['val_pairs']} / "
              f"{agri['test_pairs']}.", bullet=True, after=8),
            P(f"{lora['test_questions_seen_in_train_or_val']} test questions appear in training or validation.",
              bullet=True, after=8),
            P(f"Trained on the first {lora['split_sizes']['train_used']} pairs, a CPU budget.", bullet=True),
        ],
        size=15,
    )
    card(
        s, 4.75, 1.55, 3.83, 4.65,
        [
            P(B("Why LoRA", c=RED, s=18), after=10),
            P(B("From scratch: "), f"{agri['unique_questions']:,} pairs cannot teach a model English.",
              bullet=True, after=8),
            P(B("RAG: "), "looks facts up but leaves the model's probabilities unchanged, and scoring "
              "transcripts needs those probabilities.", bullet=True, after=8),
            P(B("LoRA: "), "freezes the pretrained weights and learns a small low-rank update.", bullet=True),
        ],
        size=15,
    )
    card(
        s, 8.88, 1.55, 3.83, 4.65,
        [
            P(B("The setup", c=RED, s=18), after=6),
            P(B(f"{std['trainable_params']:,}", c=RED, s=34), after=0),
            P(f"trainable weights, {100 * std['trainable_params'] / std['total_params']:.2f}% of "
              f"{std['total_params']:,}", s=13, c=MUTED, after=10),
            P("Rank 8, α = 32, on c_attn, the fused query, key and value projection.", bullet=True, after=8),
            P("Two objectives on identical data: loss on every token (standard), or on answer tokens only "
              "(masked).", bullet=True),
        ],
        size=15,
    )
    notes(
        s,
        "Say: The corpus repeats each question about ten times, so we deduplicated before splitting; our "
        "first split leaked copies of training questions into the test set. We chose LoRA because it changes "
        "the model's own probabilities, which is what scoring transcripts needs, while training a tiny "
        "fraction of the weights on a laptop CPU. (About 60 seconds.)\n"
        "Sources: data/processed/domain_english/stats.json (raw_rows, unique_questions, *_pairs); "
        "results/section_c_llm/lora_results.json (test_questions_seen_in_train_or_val, split_sizes.train_used, "
        "results.standard.trainable_params, total_params); claims table rows 18 to 20 and 26.",
    )

    # 8. Section C results
    s = add(TABLE)
    title(s, 11, f"Results on {lora['split_sizes']['test']} held-out questions", size=30, width=11.01)
    drop(s, 10)
    rows = [["Perplexity (lower is better)", "Full Q&A text", "Answer tokens only", "General English (WikiText-2)"]]
    for name, r in (("distilgpt2 base", base), ("LoRA, loss on all tokens", std), ("LoRA, loss on answers only", masked)):
        rows.append([name, f"{r['full_ppl']:.2f}", f"{r['answer_ppl']:.2f}", f"{r['wikitext_ppl']:.2f}"])
    table(s, 1.16, 1.55, [3.6, 2.2, 2.3, 2.91], rows, [0.5, 0.45, 0.45, 0.45], size=14, right_cols=(1, 2, 3))
    for i, (num, lab) in enumerate(
        [
            (down(base["answer_ppl"], std["answer_ppl"]), "lower answer perplexity: the cleanest sign of domain knowledge"),
            (up(base["wikitext_ppl"], std["wikitext_ppl"]), "higher general-English perplexity: LoRA limits forgetting, it does not prevent it"),
            (down(base["full_ppl"], std["full_ppl"]), "lower full-text perplexity, partly from learning the question format"),
        ]
    ):
        card(
            s, 1.16 + i * 3.73, 3.65, 3.53, 1.55,
            [P(B(num, c=RED, s=30), after=2), P(lab, s=13)],
            anchor=MSO_ANCHOR.MIDDLE,
        )
    text(
        s, 1.16, 5.4, 11.01, 0.8,
        [
            P(B("Standard vs masked: ", c=RED),
              f"on answers they are within noise ({std['answer_ppl']:.2f} vs {masked['answer_ppl']:.2f}, one seed). "
              "Only the standard adapter learns to predict questions, and speech transcripts contain the questions.")
        ],
        size=14,
    )
    notes(
        s,
        "Say: The answer-only column is the fair measure of domain knowledge, because the full-text number "
        "also rewards learning the Question/Answer template. The WikiText column is the price: general English "
        "gets worse. The percentages are for the standard adapter. (About 60 seconds.)\n"
        "Sources: results/section_c_llm/lora_results.json, results.{base,standard,masked}.{full_ppl,answer_ppl,"
        "wikitext_ppl}; percentages computed from those values; claims table rows 21 and 22.",
    )

    # 9. Fluent is not correct
    question = lora["prompts"][2].split("\n")[0].removeprefix("Question: ")
    first = lambda sample: sample.split(". ")[0] + "."  # noqa: E731  first sentence, verbatim
    s = add(TEXT_HEAVY)
    title(s, 11, "Fluent is not the same as correct", width=W)
    drop(s, 12, 13)
    card(
        s, 0.62, 1.55, 6.5, 4.65,
        [
            P(B(question, s=18), after=12),
            P(B("LoRA, all tokens", c=RED), after=2),
            P(f"“{first(std['samples'][2])}”", after=12, s=17),
            P(B("LoRA, answers only", c=RED), after=2),
            P(f"“{first(masked['samples'][2])}”", after=12, s=17),
            P("Seed 42, temperature 0.7. Both sound like an extension officer; neither is right.", s=13, c=MUTED),
        ],
        size=17,
    )
    card(
        s, 7.42, 1.55, 5.29, 4.65,
        [
            P(B("Decoding changes repetition, not truth", c=RED, s=18), after=10),
            P(f"Distinct trigrams per answer: {100 * dist3['unpenalized_sampling']['avg_distinct_3']:.1f}% with plain "
              f"sampling, {100 * dist3['repetition_penalty_only']['avg_distinct_3']:.1f}% with a repetition penalty, "
              f"{100 * dist3['ngram_blocking']['avg_distinct_3']:.1f}% with 3-gram blocking.", bullet=True, after=8),
            P("Blocking guarantees that number by construction; it says nothing about content.", bullet=True, after=8),
            P("At best one of our nine seeded answers is roughly right.", bullet=True, after=8),
            P(B("Use: "), "scoring transcripts in the domain. Never advice to farmers.", bullet=True),
        ],
        size=16,
    )
    notes(
        s,
        "Say: An 82-million-parameter model trained on 500 examples learns how an extension answer sounds "
        "long before it learns agronomy. Decoding tricks stop the loops but not the errors. (About 45 seconds.)\n"
        "Sources: results/section_c_llm/lora_results.json, prompts[2] and results.{standard,masked}.samples[2] "
        "(first sentence, verbatim); results/section_c_llm/decoding_benchmark.json, strategy_averages; "
        "claims table rows 24 and 25.",
    )

    # 10. What we got wrong, and takeaways
    s = add(TEXT_HEAVY)
    title(s, 11, "What we got wrong, and what it taught us", size=30, width=W)
    drop(s, 12, 13)
    card(
        s, 0.62, 1.55, 5.9, 4.65,
        [
            P(B("Caught before submission", c=RED, s=18), after=10),
            P("Our first draft reported a breaking point after N = 4. It was three smoothing bugs; a test that "
              "probabilities sum to 1 now guards against them.", bullet=True, after=8),
            P("We first ranked tokenizers per token, then per word without charging unknown words. Both "
              "rankings were wrong.", bullet=True, after=8),
            P("16 of 100 test pairs in our first Section C split were copies of training pairs. We now "
              "deduplicate before splitting.", bullet=True, after=8),
            P("Every number on these slides is read from a result file written by a committed script.",
              bullet=True),
        ],
        size=16,
    )
    card(
        s, 6.81, 1.55, 5.9, 4.65,
        [
            P(B("Takeaways", c=RED, s=18), after=10),
            P(B("Smoothing decides it. "), "With Kneser-Ney, Ewe n-grams improve up to about four words of "
              "context, then plateau.", bullet=True, after=8),
            P(B("Subwords model Ewe best "), "when tokenizers are compared per word.", bullet=True, after=8),
            P(B("N-gram or neural depends on data. "), f"The n-gram wins on {lstm['2']['train_sentences']} "
              f"sentences; the LSTM wins on {lstm['unified']['train_sentences']:,}.", bullet=True, after=8),
            P(B("LoRA adapts cheaply "), "but makes the model fluent, not reliable.", bullet=True),
        ],
        size=16,
    )
    notes(
        s,
        "Say: Our biggest lessons came from our own mistakes. Each of these was caught by checking the "
        "results against the code, and each check is now part of the repository. (About 60 seconds.)\n"
        "Sources: the three bugs and the 16 leaked pairs are audit findings recorded in WORKLOG.md "
        "(2026-09-21 and 2026-09-22) and in the claims table rows 15, 17 and 26; the takeaways repeat numbers "
        "from slides 4 to 9.",
    )

    # 11. Close
    s = add(CLOSING)
    title(s, 0, "Thank you. Questions?", size=60)
    text(
        s, 1.67, 5.55, 10.0, 0.5,
        [P(f"ICS554 Natural Language Processing · Group 1 · {REPO_URL}", c=GOLD)],
        size=16,
        align=PP_ALIGN.CENTER,
    )
    text(
        s, 1.17, 6.2, 11.0, 0.7,
        [
            P(
                B("AI declaration: ", c=GOLD),
                "we used AI tools to help write code, check results and draft text, and to generate this "
                "presentation from our result files. How AI was used is logged in the project's WORKLOG.md.",
            )
        ],
        size=12,
        color=WHITE,
        align=PP_ALIGN.CENTER,
    )
    notes(
        s,
        "Say: Thank you. Every number in this deck has its source in the speaker notes; we are happy to take "
        "questions on any of them.\n"
        "The footnote is the group's AI declaration; agree its wording as a group. Each member's own AI-use "
        "declaration belongs in their individual reflection.",
    )

    # remove the template's example slides; their parts are dropped on save
    ids = prs.slides._sldIdLst
    for sld in list(ids)[:n_template]:
        prs.part.drop_rel(sld.rId)
        ids.remove(sld)
    # the template was saved in Slide Master view; open the deck in Normal view instead
    view = prs.part.part_related_by(RT.VIEW_PROPS)
    view._blob = view.blob.replace(b'lastView="sldMasterView"', b'lastView="sldView"')
    prs.core_properties.title = "Building and adapting language models"
    prs.core_properties.author = "ICS554 Group 1, Ashesi University"
    prs.save(OUT)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    build()
