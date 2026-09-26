"""Build the 10-minute presentation on the Ashesi turquoise template from results/*.

Run: uv run --with python-pptx python build_slides.py   ->   presentation.pptx
Speaker notes hold what to say. Slide 1 carries the GitHub link (brief requirement).
Charts are native PowerPoint charts built from results/*.json, so they stay editable.
"""
import copy
import json

from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION, XL_MARKER_STYLE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt, Emu

TEMPLATE = "templates/ashesi_presentation template turqouise .pptx"
OUT = "presentation.pptx"
GITHUB_URL = "https://github.com/ASU-MICS-2028/nlp-group-1-prosit-1"
TEAM = "Group 1"

# Template palette
TEAL = RGBColor(0x51, 0x80, 0x8E)      # slide background
DARK = RGBColor(0x25, 0x3A, 0x40)      # panels, text on white
GOLD = RGBColor(0xFE, 0xBA, 0x5A)      # titles
AMBER = RGBColor(0xFB, 0xAE, 0x40)     # accents
CORAL = RGBColor(0xF2, 0x6B, 0x43)     # second accent
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
PALE = RGBColor(0xDC, 0xE8, 0xEB)      # soft text on dark / zebra rows
MUTED = RGBColor(0x5F, 0x6B, 0x6E)
HEAD, BODY = "Poppins", "Poppins"

s4 = json.load(open("results/stage4_scaling.json"))["scaling"]
s5 = json.load(open("results/stage5_domain.json"))
qwen = s5["runs"]["data/models/Qwen2.5-0.5B"]
base = qwen["base"]

prs = Presentation(TEMPLATE)
L = {l.name: l for l in prs.slide_layouts}

# drop the template's sample slides, keep the layouts
sldIdLst = prs.slides._sldIdLst
for sldId in list(sldIdLst):
    prs.part.drop_rel(sldId.rId)
    sldIdLst.remove(sldId)


# ---------------------------------------------------------------- helpers
def _run(p, txt, size, color, bold=False, italic=False, font=BODY):
    r = p.add_run()
    r.text = txt
    f = r.font
    f.size, f.bold, f.italic, f.name = Pt(size), bold, italic, font
    f.color.rgb = color
    return r


def fill(frame, lines, size=16, color=WHITE, space=6, align=None, anchor=None, font=BODY):
    """lines: str, or (str, dict) with keys b, c, s, i, bullet. '- ' prefix = bullet."""
    frame.clear()
    frame.word_wrap = True
    if anchor:
        frame.vertical_anchor = anchor
    for i, line in enumerate(lines):
        opt = {}
        if isinstance(line, tuple):
            line, opt = line
        p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        bullet = line.startswith("- ")
        if bullet:
            line = line[2:]
            pPr = p._p.get_or_add_pPr()
            pPr.set("marL", str(Emu(Inches(0.25))))
            pPr.set("indent", str(-Emu(Inches(0.25))))
            bu = etree.SubElement(pPr, qn("a:buFont"))
            bu.set("typeface", "Arial")
            ch = etree.SubElement(pPr, qn("a:buChar"))
            ch.set("char", "•")
        else:
            pPr = p._p.get_or_add_pPr()
            pPr.set("marL", "0")
            pPr.set("indent", "0")
            etree.SubElement(pPr, qn("a:buNone"))
        _run(p, line, opt.get("s", size), opt.get("c", color), opt.get("b", False), opt.get("i", False), opt.get("f", font))
        p.space_after = Pt(opt.get("after", space))
        if align:
            p.alignment = align
    return frame


def box(slide, x, y, w, h, lines, size=16, color=WHITE, align=None, anchor=MSO_ANCHOR.TOP, margin=0.0, **kw):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = Inches(margin)
    fill(tf, lines, size, color, align=align, anchor=anchor, **kw)
    return tb


SHADOW = ('<a:effectLst xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
          '<a:outerShdw blurRad="190500" dist="38100" dir="5400000" algn="ctr" rotWithShape="0">'
          '<a:srgbClr val="000000"><a:alpha val="18000"/></a:srgbClr></a:outerShdw></a:effectLst>')


def card(slide, x, y, w, h, color=WHITE, radius=0.06, shadow=True):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shp.adjustments[0] = radius
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    shp.line.fill.background()
    spPr = shp._element.spPr
    for e in spPr.findall(qn("a:effectLst")):
        spPr.remove(e)
    if shadow:
        spPr.append(etree.fromstring(SHADOW))
    else:
        spPr.append(etree.fromstring('<a:effectLst xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"/>'))
    shp.text_frame.text = ""
    return shp


def stat(slide, x, y, w, big, label, big_color=GOLD, label_color=WHITE, big_size=44, label_size=13):
    box(slide, x, y, w, 0.85, [(big, {"b": True, "c": big_color, "s": big_size, "f": HEAD})], anchor=MSO_ANCHOR.BOTTOM)
    box(slide, x, y + 0.9, w, 0.9, [label], label_size, label_color)


def notes(slide, say):
    slide.notes_slide.notes_text_frame.text = say


def content_slide(title, tag):
    """Template 'Text + image' layout, with only its title kept, widened across the slide."""
    slide = prs.slides.add_slide(L["Text + image"])
    for ph in list(slide.placeholders):
        if ph.placeholder_format.idx != 0:
            ph._element.getparent().remove(ph._element)
    t = slide.shapes.title
    t.left, t.top, t.width, t.height = Inches(0.68), Inches(1.2), Inches(11.95), Inches(0.95)
    t.text_frame.word_wrap = True
    t.text_frame.vertical_anchor = MSO_ANCHOR.TOP
    fill(t.text_frame, [(title, {"b": True, "c": GOLD, "s": 28, "f": HEAD})], space=0)
    box(slide, 8.4, 0.5, 4.25, 0.35, [(tag, {"b": True, "c": PALE, "s": 11})], align=PP_ALIGN.RIGHT)
    return slide


def table(slide, x, y, w, rows, col_w, size=12, row_h=0.36, highlight=None, first_left=True):
    n_rows, n_cols = len(rows), len(rows[0])
    shape = slide.shapes.add_table(n_rows, n_cols, Inches(x), Inches(y), Inches(w), Inches(row_h * n_rows))
    tbl = shape.table
    # plain table: no template style banding
    tblPr = tbl._tbl.tblPr
    tblPr.set("firstRow", "0")
    tblPr.set("bandRow", "0")
    sid = tblPr.find(qn("a:tableStyleId"))
    if sid is not None:
        sid.text = "{5940675A-B579-460E-94D1-54222C63F5DA}"  # 'No Style, Table Grid'
    for i, cw in enumerate(col_w):
        tbl.columns[i].width = Inches(cw)
    for r, row in enumerate(rows):
        tbl.rows[r].height = Inches(row_h)
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            head = r == 0
            hl = highlight is not None and r == highlight
            cell.fill.solid()
            cell.fill.fore_color.rgb = DARK if head else (RGBColor(0xFF, 0xE9, 0xC4) if hl else (WHITE if r % 2 else RGBColor(0xEE, 0xF4, 0xF5)))
            cell.margin_left = cell.margin_right = Inches(0.1)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.clear()
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if (c == 0 and first_left) else PP_ALIGN.RIGHT
            _run(p, str(val), size, GOLD if head else DARK, bold=head or hl)
            # hairline borders in the card colour
            tcPr = cell._tc.get_or_add_tcPr()
            for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
                ln = etree.SubElement(tcPr, qn(tag))
                ln.set("w", "6350")
                sf = etree.SubElement(ln, qn("a:solidFill"))
                clr = etree.SubElement(sf, qn("a:srgbClr"))
                clr.set("val", "FFFFFF")
            # schema order: borders must come before the fill
            fillel = tcPr.find(qn("a:solidFill"))
            if fillel is not None:
                tcPr.remove(fillel)
                tcPr.append(fillel)
    return shape


def style_chart(chart, size=11, color=DARK):
    chart.font.size = Pt(size)
    chart.font.name = BODY
    chart.font.color.rgb = color


def axis_style(ax, color=MUTED, grid=True, size=11):
    ax.tick_labels.font.size = Pt(size)
    ax.tick_labels.font.color.rgb = color
    ax.format.line.color.rgb = RGBColor(0xC9, 0xD3, 0xD6)
    ax.has_major_gridlines = grid
    if grid:
        ax.major_gridlines.format.line.color.rgb = RGBColor(0xE3, 0xEA, 0xEC)


def log_axis(ax, base=10):
    scaling = ax._element.find(qn("c:scaling"))
    lb = etree.SubElement(scaling, qn("c:logBase"))
    lb.set("val", str(base))
    scaling.remove(lb)
    scaling.insert(0, lb)  # logBase is the first child of c:scaling


def chart_title(chart, txt, size=13):
    chart.has_title = True
    tf = chart.chart_title.text_frame
    tf.clear()
    _run(tf.paragraphs[0], txt, size, DARK, bold=True)


# ================================================================ 1. Title
slide = prs.slides.add_slide(L["Title Slide"])
ph = {p.placeholder_format.idx: p for p in slide.placeholders}
fill(ph[0].text_frame, [("Building and Adapting Language Models", {"b": True, "c": GOLD, "s": 40, "f": HEAD})], space=0)
ph[0].left, ph[0].top, ph[0].width, ph[0].height = Inches(0.5), Inches(1.9), Inches(5.9), Inches(2.1)
fill(ph[10].text_frame, [
    ("An Ewe n-gram model and a domain-tuned English model", {"s": 18, "c": WHITE}),
    (f"ICS554 Natural Language Processing  ·  Prosit 1  ·  {TEAM}", {"s": 14, "c": PALE}),
], space=4)
ph[10].left, ph[10].top, ph[10].width, ph[10].height = Inches(0.5), Inches(4.15), Inches(5.9), Inches(1.0)
ph[11]._element.getparent().remove(ph[11]._element)
# right half: a dark panel with the seven Ewe letters our language filter keys on
panel = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(6.67), 0, Inches(6.67), Inches(7.5))
panel.fill.solid(); panel.fill.fore_color.rgb = DARK; panel.line.fill.background()
box(slide, 7.2, 1.7, 5.6, 1.4, [("ɖ  ƒ  ŋ  ɔ  ɛ  ʋ  ɣ", {"b": True, "c": GOLD, "s": 54})], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
box(slide, 7.4, 3.2, 5.2, 0.9, [("The seven letters that told us a sentence was Ewe", {"c": PALE, "s": 14})], align=PP_ALIGN.CENTER)
box(slide, 7.4, 4.35, 5.2, 0.9, [("“Aƒetɔ, va ɖe mí”", {"i": True, "c": WHITE, "s": 22})], align=PP_ALIGN.CENTER)
box(slide, 7.4, 4.95, 5.2, 0.5, [("our hand-worked example sentence", {"c": PALE, "s": 12})], align=PP_ALIGN.CENTER)
box(slide, 0.5, 5.55, 5.9, 0.7, [("Code", {"b": True, "c": GOLD, "s": 12}), (GITHUB_URL, {"c": WHITE, "s": 12})], space=0)
notes(slide, "10 seconds. Names, and point at the GitHub link (the brief requires it on slide 1).")

# ================================================================ 2. The brief
slide = content_slide("Ankora asked us for two language models", "THE BRIEF")
band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(3.55), Inches(13.33), Inches(2.95))
band.fill.solid(); band.fill.fore_color.rgb = DARK; band.line.fill.background()
cards = [
    ("SECTION B", "An Ewe language model",
     "A low-resource African language. Ankora suggested an n-gram model because there is little text."),
    ("SECTION C", "A domain-tuned English model",
     "A pretrained English model adapted to a domain. We did two: health (PubMed) and agriculture (farming Q&A)."),
    ("OUR METHOD", "Build, fan out, verify",
     "Build each model, test every reasonable variant, and check our numbers against outside tools."),
]
for i, (tag, head, body) in enumerate(cards):
    x = 0.68 + i * 4.08
    card(slide, x, 2.35, 3.8, 2.55)
    box(slide, x + 0.3, 2.6, 3.2, 0.3, [(tag, {"b": True, "c": CORAL, "s": 11})])
    box(slide, x + 0.3, 2.92, 3.2, 0.5, [(head, {"b": True, "c": DARK, "s": 17})])
    box(slide, x + 0.3, 3.62, 3.2, 1.2, [(body, {"c": DARK, "s": 13})])
box(slide, 0.68, 5.25, 11.95, 1.1, [
    ("Who did what", {"b": True, "c": GOLD, "s": 16}),
    ("Eric Sunu: a parallel pipeline (four Ewe sources, an Ewe stemmer tokenizer, distilgpt2 + LoRA, decoding benchmark), in contrib/eric/.  [Add the other members]", {"c": WHITE, "s": 14}),
], space=4)
notes(slide, "30 s. State the two tasks and who did what. One sentence on the approach: build, fan out the options, verify.")

# ================================================================ 3. Data funnel
slide = content_slide("Most of a 4-million-sentence corpus was not Ewe", "SECTION B · DATA")
card(slide, 0.68, 2.25, 5.9, 4.6)
cd = CategoryChartData()
cd.categories = ["Downloaded", "After dedupe", "Ewe (kept)"]
cd.add_series("Sentences", (4408322, 995588, 295198))
gf = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.9), Inches(2.4), Inches(5.45), Inches(4.3), cd)
ch = gf.chart
style_chart(ch)
chart_title(ch, "Sentences left after each cleaning step")
ch.has_legend = False
plot = ch.plots[0]
plot.gap_width = 60
plot.has_data_labels = True
dl = plot.data_labels
dl.number_format, dl.number_format_is_linked = "#,##0", False
dl.position = XL_LABEL_POSITION.OUTSIDE_END
dl.font.size, dl.font.bold, dl.font.color.rgb = Pt(12), True, DARK
ser = plot.series[0]
for i, c in enumerate((TEAL, TEAL, CORAL)):
    pt = ser.points[i]
    pt.format.fill.solid(); pt.format.fill.fore_color.rgb = c
axis_style(ch.category_axis, DARK, grid=False, size=12)
va = ch.value_axis
axis_style(va)
va.visible = False
va.has_major_gridlines = False
va.maximum_scale = 5000000

stat(slide, 7.0, 2.2, 2.7, "77.4%", "of rows were copies. One sentence appeared 1,296 times.")
stat(slide, 9.9, 2.2, 2.8, "70.3%", "of the rest had no Ewe letter: English spam, Japanese, web titles.")
box(slide, 7.0, 4.35, 5.65, 2.5, [
    ("Kept: 295,198 sentences, 6.7% of the download.", {"b": True, "c": WHITE, "s": 15}),
    "- Split 90/5/5 by near-duplicate group: 0 test sentences leak into train.",
    "- Each step tested by leaving it out: no dedupe 41% worse, no filter 25% worse, random split 7% “better” (leakage).",
    "- A language identifier that knows Ewe calls 92% of what we kept Ewe.",
], 13, WHITE, space=5)
notes(slide, "60 s. The size on a dataset page means little. Source: automatic sentence mining from web pages (HuggingFace, 4,408,322 rows). Say what the filter cannot do: it keeps Ga and Pidgin that share Ewe's letters, and nobody who reads Ewe has checked a sample yet.")

# ================================================================ 4. n-gram by hand
slide = content_slide("Training an n-gram model is just counting", "SECTION B · HOW IT WORKS")
card(slide, 0.68, 2.25, 6.3, 4.6, color=DARK, shadow=False)
box(slide, 1.0, 2.5, 5.7, 4.2, [
    ("THE RULE", {"b": True, "c": AMBER, "s": 11}),
    ("P(next word | previous word) = count(pair) ÷ count(previous word)", {"b": True, "c": WHITE, "s": 16, "after": 16}),
    ("WORKED BY HAND", {"b": True, "c": AMBER, "s": 11}),
    ("Five training sentences, one held-out sentence:", {"c": PALE, "s": 13}),
    ("“Aƒetɔ, va ɖe mí”", {"i": True, "c": GOLD, "s": 20}),
    ("P = 2/5 × 1/2 × 1/2 × 2/3 × 1/2 = 1/30", {"c": WHITE, "s": 15}),
    ("Perplexity = 30^(1/5) = 1.974", {"b": True, "c": WHITE, "s": 15, "after": 12}),
    ("We did this on paper first, then made the code check itself against it.", {"c": PALE, "s": 13}),
], space=4)
stat(slide, 7.4, 2.15, 5.2, "≈ 2 choices", "per word: perplexity is the model’s average number of equally likely choices per token. Lower is better.", big_size=38)
box(slide, 7.4, 4.35, 5.25, 2.5, [
    ("The catch", {"b": True, "c": GOLD, "s": 15}),
    "- Perplexity only compares models with the same tokens and test set.",
    "- Split the comma off and the same sentence scores 2.076 (P = 1/80, N = 6): different units, not a worse model.",
    "- So to compare tokenisers we use bits per character.",
], 13, WHITE, space=5)
notes(slide, "45 s. Cut this slide first if short on time.")

# ================================================================ 5. Sparsity
slide = content_slide("Plain counting gives infinite perplexity", "SECTION B · WHY COUNTING FAILS")
stat(slide, 0.68, 2.1, 5.4, "95.5%", "of dev sentences contain a trigram the model never saw. One zero makes the whole sentence “impossible”.", big_size=54)
box(slide, 0.68, 4.0, 5.4, 2.7, [
    ("Higher orders do not help: they recite.", {"b": True, "c": GOLD, "s": 15}),
    "- 3.39M distinct 5-grams in a 4.32M-word corpus: almost every position is unique.",
    "- At order 9, 96% of generated sentences are exact copies of training sentences. Perplexity alone would not show this.",
], 13, WHITE, space=5)
card(slide, 6.6, 2.25, 6.05, 4.6)
box(slide, 6.9, 2.45, 5.5, 0.4, [("Sparsity and memorisation by n-gram order", {"b": True, "c": DARK, "s": 14})])
table(slide, 6.9, 3.0, 5.45, [["order", "seen once", "P = 0 tokens", "copied"],
                               ["1", "55%", "2%", "0%"], ["3", "79%", "42%", "7%"],
                               ["5", "90%", "71%", "46%"], ["9", "94%", "80%", "96%"]],
      col_w=[0.85, 1.55, 1.6, 1.45], size=13, row_h=0.55)
box(slide, 6.9, 5.95, 5.5, 0.7, [("“Copied verbatim” = share of generated sentences found word-for-word in training.", {"c": MUTED, "s": 11})])
notes(slide, "60 s. Sparsity: 3.39M distinct 5-grams in a 4.32M-word corpus, almost every position unique. Memorisation: fluent output at order 9 is copying, which perplexity alone would not show.")

# ================================================================ 6. Smoothing
slide = content_slide("Smoothing: “never seen” means rare, not impossible", "SECTION B · SMOOTHING")
card(slide, 0.68, 2.25, 7.3, 4.6)
methods = [("add-1 (Laplace)", 88754), ("add-k (k = 1e-5)", 14579), ("Good-Turing", 13893), ("interpolation (λ = 0.45)", 127.5),
           ("Katz backoff", 126.0), ("modified Kneser-Ney", 100.8), ("Kneser-Ney (D = 0.8)", 100.4)]
cd = CategoryChartData()
cd.categories = [m for m, _ in methods]
cd.add_series("dev perplexity", [v for _, v in methods])
gf = slide.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.85), Inches(2.4), Inches(6.95), Inches(4.3), cd)
ch = gf.chart
style_chart(ch)
chart_title(ch, "Dev perplexity, order 5 (log scale, lower is better)", 12)
ch.has_legend = False
plot = ch.plots[0]
plot.gap_width = 45
plot.has_data_labels = True
dl = plot.data_labels
dl.number_format, dl.number_format_is_linked = "#,##0.#", False
dl.position = XL_LABEL_POSITION.OUTSIDE_END
dl.font.size, dl.font.color.rgb = Pt(11), DARK
ser = plot.series[0]
for i in range(len(methods)):
    pt = ser.points[i]
    pt.format.fill.solid()
    pt.format.fill.fore_color.rgb = CORAL if i >= 5 else (RGBColor(0xA9, 0xC2, 0xC9) if i < 3 else TEAL)
ca = ch.category_axis
axis_style(ca, DARK, grid=False, size=11)
ca.reverse_order = True
va = ch.value_axis
axis_style(va, size=10)
log_axis(va)
va.minimum_scale, va.maximum_scale = 10, 1000000
va.tick_labels.number_format, va.tick_labels.number_format_is_linked = "#,##0", False
va.visible = False
va.has_major_gridlines = False
box(slide, 8.4, 2.3, 4.25, 4.5, [
    ("Six methods, tuned on dev only (k, λ, D).", {"b": True, "c": GOLD, "s": 15}),
    "- Add-1 spreads probability over every possible sequence, almost none of which occur.",
    "- Kneser-Ney judges a word by how many different contexts it follows.",
    "- Same ranking as Chen & Goodman (1999).",
    "- Our code matches NLTK to 2 decimals and KenLM (128.25 vs our 128.04).",
    ("Final test, run once: 99.8", {"b": True, "c": GOLD, "s": 16}),
], 13, WHITE, space=6)
notes(slide, "60 s. Add-1 spreads probability over every possible sequence, almost none of which occur, and gets worse with order and with more data. Kneser-Ney judges a word by how many different contexts it follows. Mention the unknown-word finding if asked: Katz looked better until we split the score into known and unknown words. Final test perplexity 99.8 (dev 100.8).")

# ================================================================ 7. Tokenisation
slide = content_slide("Which tokens? Bits per character decides, and BPE 2k wins", "SECTION B · TOKENISATION")
card(slide, 0.68, 2.25, 6.8, 4.6)
box(slide, 0.98, 2.45, 6.2, 0.4, [("Best order per tokeniser, dev set", {"b": True, "c": DARK, "s": 14})])
table(slide, 0.98, 2.95, 6.2, [["tokeniser", "perplexity", "bits / char", "best order"],
                                ["characters", "3.2", "1.599", "12"], ["BPE 1k pieces", "13.8", "1.509", "10"],
                                ["BPE 2k pieces", "20.3", "1.496", "8"], ["BPE 16k pieces", "92.8", "1.540", "6"],
                                ["whitespace words*", "100.8", "1.443", "6"], ["lower-cased words*", "47.0", "1.429", "6"]],
      col_w=[2.3, 1.3, 1.3, 1.3], size=12, row_h=0.46, highlight=3)
box(slide, 0.98, 6.25, 6.2, 0.5, [("* before charging word models for spelling unknown words (then 1.516–1.627)", {"c": MUTED, "s": 10})])
stat(slide, 7.9, 2.1, 4.7, "1.496", "bits per character for BPE with 2,000 pieces: the fair winner.")
box(slide, 7.9, 4.35, 4.75, 2.5, [
    "- The two columns rank in opposite orders: characters have the best perplexity and the worst bits/char.",
    "- Word models pay a flat cost for unknown words (2.4% of dev). Charge them for spelling those words and they drop behind.",
    "- Subwords beat words on Ewe, once the comparison was fair.",
], 13, WHITE, space=6)
notes(slide, "45 s. Character perplexity is per guess and character guesses are easy; you need 5x as many of them. Divide by characters instead and every tokenisation shares the denominator.")

# ================================================================ 8. Scaling
slide = content_slide("n-grams beat neural models only below ~10k sentences", "SECTION B · N-GRAM VS NEURAL")
card(slide, 0.68, 2.25, 7.3, 4.6)
sizes = ["1000", "3000", "10000", "30000", "100000"]
cd = CategoryChartData()
cd.categories = ["1k", "3k", "10k", "30k", "100k"]
cd.add_series("n-gram (modified KN)", [round(s4[k]["ngram"]["pp"], 1) for k in sizes])
cd.add_series("LSTM", [round(s4[k]["lstm"]["pp"], 1) for k in sizes])
cd.add_series("tiny transformer", [round(s4[k]["transformer"]["pp"], 1) for k in sizes])
gf = slide.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(0.85), Inches(2.4), Inches(6.95), Inches(4.3), cd)
ch = gf.chart
style_chart(ch)
chart_title(ch, "Dev perplexity vs training sentences (log scale)", 12)
ch.has_legend = True
ch.legend.position = XL_LEGEND_POSITION.TOP
ch.legend.include_in_layout = False
ch.legend.font.size = Pt(11)
for ser, c, mk in zip(ch.plots[0].series, (DARK, CORAL, TEAL), (XL_MARKER_STYLE.CIRCLE, XL_MARKER_STYLE.SQUARE, XL_MARKER_STYLE.TRIANGLE)):
    ser.smooth = False
    ser.format.line.color.rgb = c
    ser.format.line.width = Pt(3)
    ser.marker.style, ser.marker.size = mk, 8
    ser.marker.format.fill.solid(); ser.marker.format.fill.fore_color.rgb = c
    ser.marker.format.line.color.rgb = WHITE
axis_style(ch.category_axis, DARK, grid=False, size=11)
va = ch.value_axis
axis_style(va, size=10)
log_axis(va)
va.minimum_scale, va.maximum_scale = 10, 3000
va.tick_labels.number_format, va.tick_labels.number_format_is_linked = "#,##0", False
box(slide, 8.4, 2.3, 4.25, 4.55, [
    ("Same BPE tokens and the same 325,004 dev tokens, so the curves are comparable.", {"c": PALE, "s": 12, "after": 8}),
    "- n-gram wins at 1k (353 vs 843) and 3k.",
    "- LSTM overtakes at 10k (145 vs 151) and 30k (79 vs 94).",
    "- At 100k a wider LSTM (512 units) scores 46 vs 56: size the network to the data and it keeps winning.",
    "- Same crossover on English, on Wikipedia, and in Eric's separate pipeline: not an Ewe quirk.",
    "- Cost: n-gram < 1 min on CPU; LSTM 236 min at 100k on GPU.",
], 13, WHITE, space=6)
notes(slide, "75 s, the centrepiece. Why neural loses on tiny data (Eric's framing): his small LSTM had about 50 weights per training word, far more than the data can pin down; his large one about 2. His independent four-source pipeline found the same crossover: n-gram 26% better on 420 sentences, LSTM 12% better on 98,808. Three corpora, same crossover. Wikipedia is harder than our Ewe at equal size (343 vs 100 at 100k) because of topic breadth: our Ewe corpus is 35% religious text. On Wikipedia the n-gram keeps improving to 3M sentences (134); unknown words fall to 0.3%. Counting works from the first sentence but can only reuse what it saw. A neural model must learn an embedding per token first, then it generalises. The LSTM stalled at 100k because we kept it at 2M weights; a width sweep showed size must match data (256 best at 10k; 512 memorised).")

# ================================================================ 9. Section C set-up
slide = content_slide("Section C: adapt a pretrained English model to a domain", "SECTION C · SET-UP")
data = [("HEALTH", "1.93M", "words · 8,000 PubMed abstracts"),
        ("AGRICULTURE", "84k", "words · 2,112 farming Q&A documents"),
        ("GENERAL (TEST ONLY)", "500", "Wikipedia paragraphs, to measure forgetting")]
for i, (tag, big, lab) in enumerate(data):
    x = 0.68 + i * 4.08
    card(slide, x, 2.25, 3.8, 1.85)
    box(slide, x + 0.3, 2.42, 3.2, 0.3, [(tag, {"b": True, "c": CORAL, "s": 11})])
    box(slide, x + 0.3, 2.72, 3.2, 0.7, [(big, {"b": True, "c": DARK, "s": 32, "f": HEAD})])
    box(slide, x + 0.3, 3.45, 3.2, 0.6, [(lab, {"c": DARK, "s": 12})])
card(slide, 0.68, 4.4, 11.95, 2.45, color=DARK, shadow=False)
box(slide, 1.0, 4.58, 11.3, 0.35, [("OPTIONS WE WEIGHED", {"b": True, "c": AMBER, "s": 11})])
opts = ["from scratch", "prompting", "RAG", "full fine-tune", "LoRA", "DoRA", "instruction tuning"]
ox = 1.0
for o in opts:
    w = 0.32 + 0.095 * len(o)
    chosen = o == "LoRA"
    pill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(ox), Inches(4.98), Inches(w), Inches(0.42))
    pill.adjustments[0] = 0.5
    pill.fill.solid(); pill.fill.fore_color.rgb = GOLD if chosen else RGBColor(0x3A, 0x55, 0x5C)
    pill.line.fill.background()
    tf = pill.text_frame
    tf.margin_left = tf.margin_right = Inches(0.05); tf.margin_top = tf.margin_bottom = 0
    fill(tf, [(o, {"b": chosen, "c": DARK if chosen else WHITE, "s": 12})], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, space=0)
    ox += w + 0.15
box(slide, 1.0, 5.6, 11.3, 1.2, [
    ("We want a better language model of the domain, so: continued pre-training on raw text with LoRA "
     "(frozen model, small add-on matrices; rank = their size). We also ran full fine-tuning, DoRA and three ranks rather than assume.", {"c": WHITE, "s": 13}),
    ("Base models: Qwen2.5-0.5B, with SmolLM2-135M as a check. Split by document; every model scored on all three test sets.", {"c": PALE, "s": 12}),
], space=6)
notes(slide, "45 s. Why English: no small open model knows Ewe well enough for adaptation to mean anything. Why a base model: it is a plain language model, so perplexity on raw text is fair. We score every model on all three test sets: in-domain gain and general-text forgetting.")

# ================================================================ 10. Learning rate
slide = content_slide("The learning rate was tuned, not guessed", "SECTION C · TRAINING")
card(slide, 0.68, 2.25, 4.9, 4.6, color=DARK, shadow=False)
box(slide, 1.0, 2.5, 4.3, 4.2, [
    ("THE CLUE", {"b": True, "c": AMBER, "s": 11}),
    ("At our first rate (1e-4), LoRA rank 8 came out worse in-domain than rank 2.", {"b": True, "c": WHITE, "s": 17}),
    ("That should not happen. A result that contradicts theory is usually a measurement problem.", {"c": PALE, "s": 14, "after": 14}),
    ("THE FIX", {"b": True, "c": AMBER, "s": 11}),
    ("Three rates, same set-up (LoRA r8, health), chosen on the validation split, never on test.", {"c": WHITE, "s": 14}),
], space=6)
card(slide, 5.95, 2.25, 6.7, 2.75)
table(slide, 6.25, 2.5, 6.1, [["learning rate", "validation", "health test", "general test"],
                               ["1e-5 (chosen)", "7.96", "8.49", "19.73"], ["3e-5", "8.03", "8.57", "20.84"], ["1e-4", "8.49", "9.07", "24.80"]],
      col_w=[1.9, 1.35, 1.4, 1.45], size=13, row_h=0.52, highlight=1)
box(slide, 6.25, 4.6, 6.1, 0.35, [(f"Perplexity. Base model: health {base['health']:.2f}, general {base['general']:.2f}.", {"c": MUTED, "s": 11})])
stat(slide, 5.95, 5.0, 3.0, "+26%", "general-text perplexity at 1e-4", big_size=36, label_size=12)
box(slide, 9.15, 5.4, 3.5, 1.4, [("The bigger the step, the more the model forgot general English, and the less it gained.", {"c": WHITE, "s": 14})], anchor=MSO_ANCHOR.MIDDLE)
notes(slide, "45 s. A result that contradicts theory is usually a measurement problem. This is the Section C version of that lesson.")

# ================================================================ 11. Results
slide = content_slide("Adaptation works, and rank sets how much it forgets", "SECTION C · RESULTS")
card(slide, 0.68, 2.25, 7.3, 4.6)
ag = "agriculture"
methods = [("LoRA r2", "lora_r2"), ("LoRA r8", "lora_r8"), ("LoRA r32", "lora_r32"), ("DoRA r8", "dora_r8"), ("full FT", "full")]
gain, forget = [], []
for _, key in methods:
    r = qwen[f"{ag}/{key}@1e-05"]
    gain.append(round((base[ag] - r[ag]) / base[ag], 3))
    forget.append(round((r["general"] - base["general"]) / base["general"], 3))
cd = CategoryChartData()
cd.categories = [m for m, _ in methods]
cd.add_series("in-domain improvement", gain)
cd.add_series("general-text forgetting", forget)
gf = slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.85), Inches(2.4), Inches(6.95), Inches(4.3), cd)
ch = gf.chart
style_chart(ch)
chart_title(ch, "Adapted to agriculture: change in perplexity (Qwen2.5-0.5B)", 12)
ch.has_legend = True
ch.legend.position = XL_LEGEND_POSITION.TOP
ch.legend.include_in_layout = False
ch.legend.font.size = Pt(11)
plot = ch.plots[0]
plot.gap_width, plot.overlap = 70, 0
plot.has_data_labels = True
dl = plot.data_labels
dl.number_format, dl.number_format_is_linked = "+0%;-0%;0%", False
dl.position = XL_LABEL_POSITION.OUTSIDE_END
dl.font.size, dl.font.color.rgb = Pt(11), DARK
for ser, c in zip(plot.series, (TEAL, CORAL)):
    ser.format.fill.solid(); ser.format.fill.fore_color.rgb = c
    ser.invert_if_negative = False
axis_style(ch.category_axis, DARK, grid=False, size=11)
from pptx.enum.chart import XL_TICK_LABEL_POSITION
ch.category_axis.tick_label_position = XL_TICK_LABEL_POSITION.LOW
va = ch.value_axis
axis_style(va, size=10)
va.tick_labels.number_format, va.tick_labels.number_format_is_linked = "0%", False
va.minimum_scale, va.maximum_scale = -0.1, 0.8
va.major_unit = 0.2
card(slide, 8.4, 2.25, 4.25, 2.1)
h8, a8 = qwen["health/lora_r8@1e-05"], qwen["agriculture/lora_r8@1e-05"]
table(slide, 8.6, 2.42, 3.85, [["LoRA r8", "in-domain", "general"],
                                ["health", f"{base['health']:.2f} → {h8['health']:.2f}", f"{h8['general']:.2f}"],
                                ["agriculture", f"{base[ag]:.2f} → {a8[ag]:.2f}", f"{a8['general']:.2f}"]],
      col_w=[1.2, 1.6, 1.05], size=11, row_h=0.5)
box(slide, 8.6, 3.95, 3.85, 0.35, [(f"Perplexity; base general = {base['general']:.2f}", {"c": MUTED, "s": 10})])
box(slide, 8.4, 4.6, 4.25, 2.3, [
    "- Health barely moves: the base already knows medical English. Agriculture is new to it.",
    "- Full fine-tuning: 80× the weights of LoRA r8, not better.",
    "- Three seeds: a seed moves a number by ≤ 0.3. DoRA = LoRA is a tie; rank-32 forgetting is real.",
    "- Test questions also seen in training (9.6%) removed: unchanged (6.69 vs 6.66).",
], 12, WHITE, space=5)
notes(slide, "75 s. The leak check came from Eric's work: the agriculture corpus repeats questions with different answers, so 11 of our 115 test documents had their question in training; re-scored without them, nothing changed. His answer-only scoring also shows part of any Q&A gain is learning the question template (his: 49.8% full text vs 20.6% on answers). Rank 2 on agriculture: +52% in-domain with no forgetting (general text slightly better). Rank 32: +63% and 20% forgetting. DoRA = LoRA. Same pattern on SmolLM2-135M: health -9%, agriculture -52%. Health training helped agriculture by 10%; agriculture training hurt health by 9%.")

# ================================================================ 12. Lessons and limits
slide = content_slide("What we learned, and what we would not claim", "TAKEAWAYS")
learned = [
    ("Look at your data", "6.7% of the download was usable; 3× more unfiltered data was 25% worse."),
    ("Verify against outside tools", "Smoothing ranking matched the literature; NLTK and KenLM matched our code."),
    ("n-grams win only when tiny", "“Better for low-resource languages” holds only below ~10k sentences."),
    ("Fair comparisons change answers", "Four of our own conclusions flipped once the test was fair."),
    ("Rank is a learn-vs-forget dial", "Full fine-tuning bought nothing over LoRA."),
]
for i, (h, b) in enumerate(learned):
    y = 2.25 + i * 0.92
    card(slide, 0.68, y, 6.9, 0.8)
    num = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.85), Inches(y + 0.14), Inches(0.52), Inches(0.52))
    num.fill.solid(); num.fill.fore_color.rgb = GOLD; num.line.fill.background()
    tf = num.text_frame; tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    fill(tf, [(str(i + 1), {"b": True, "c": DARK, "s": 14})], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, space=0)
    box(slide, 1.55, y + 0.06, 5.9, 0.7, [(h, {"b": True, "c": DARK, "s": 13}), (b, {"c": DARK, "s": 11})], space=0, anchor=MSO_ANCHOR.MIDDLE)
card(slide, 7.9, 2.25, 4.75, 4.6, color=DARK, shadow=False)
box(slide, 8.2, 2.45, 4.2, 4.3, [
    ("LIMITS", {"b": True, "c": AMBER, "s": 11}),
    "- Our Ewe filter is a spelling check; nobody who reads Ewe has audited a sample.",
    "- The neural model was not scaled with the data; small transformers were unstable (one run diverged).",
    "- Section C: 600 untuned steps, a 0.5B base, and Wikipedia as the only general text.",
    "- Fluent is not correct: Eric’s adapted model answered a fall-armyworm question fluently and wrongly.",
    ("NEXT", {"b": True, "c": AMBER, "s": 11}),
    ("Native-speaker audit · scale the neural model with the data · adapt a model that knows Ewe.", {"c": WHITE, "s": 12}),
], 12, WHITE, space=6)
notes(slide, "45 s. The four flipped conclusions: tokeniser ranking (twice), Katz vs Kneser-Ney, the learning rate, and a 1% split 'beating' 5%. End on the limits: it is what the rubric's 'how well do we understand the work' rewards.")

# ================================================================ 13. End
slide = prs.slides.add_slide(L["End Slide"])
for ph in list(slide.placeholders):
    if ph.placeholder_format.idx != 0:
        ph._element.getparent().remove(ph._element)
fill(slide.shapes.title.text_frame, [("Thank you", {"b": True, "c": WHITE, "s": 66, "f": HEAD}),
                                     ("Questions?", {"b": True, "c": GOLD, "s": 40, "f": HEAD})], space=4)
box(slide, 0.62, 6.1, 8.0, 0.5, [(GITHUB_URL, {"c": PALE, "s": 13})])
notes(slide, "Take questions.")

prs.save(OUT)
print(f"wrote {OUT}: {len(prs.slides)} slides")
