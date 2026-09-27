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
ph[10].left, ph[10].top, ph[10].width, ph[10].height = Inches(0.5), Inches(4.1), Inches(5.9), Inches(1.35)
ph[11]._element.getparent().remove(ph[11]._element)
panel = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(6.67), 0, Inches(6.67), Inches(7.5))
panel.fill.solid(); panel.fill.fore_color.rgb = DARK; panel.line.fill.background()
box(slide, 7.3, 2.3, 5.4, 1.0, [("From low-resource Ewe", {"b": True, "c": GOLD, "s": 30, "f": HEAD})])
box(slide, 7.3, 3.2, 5.4, 1.0, [("to specialised domain AI", {"b": True, "c": WHITE, "s": 30, "f": HEAD})])
box(slide, 7.3, 4.5, 5.4, 0.9, [("Group 1  ·  MICS 2028  ·  Ashesi University", {"c": PALE, "s": 15})])
box(slide, 0.5, 5.55, 5.9, 0.7, [("Code", {"b": True, "c": GOLD, "s": 12}), (GITHUB_URL, {"c": WHITE, "s": 12})], space=0)
notes(slide, "15 s. Good morning. Today, Group 1 presents our work on building and adapting language models. We tackled two challenges: building a language model for Ewe, a low-resource African language, and adapting a modern AI model to specialised domains, healthcare and agriculture. Our full code is at the GitHub link on screen.")

# ================================================================ 2. The brief
slide = content_slide("Two challenges from Ankora", "THE BRIEF")
band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(3.55), Inches(13.33), Inches(2.95))
band.fill.solid(); band.fill.fore_color.rgb = DARK; band.line.fill.background()
cards = [
    ("SECTION B", "A low-resource language: Ewe",
     "Very little clean text exists.  Our question: does simple statistical counting beat deep learning when data is scarce?"),
    ("SECTION C", "Specialised domain AI",
     "General English models struggle with technical text.  Our question: can we teach a model medicine and farming on a laptop without breaking its general English?"),
]
for i, (tag, head, body) in enumerate(cards):
    x = 0.68 + i * 6.12
    card(slide, x, 2.35, 5.83, 3.9)
    box(slide, x + 0.35, 2.65, 5.1, 0.3, [(tag, {"b": True, "c": CORAL, "s": 12})])
    box(slide, x + 0.35, 3.0, 5.1, 0.7, [(head, {"b": True, "c": DARK, "s": 22})])
    box(slide, x + 0.35, 3.95, 5.1, 2.2, [(body, {"c": DARK, "s": 16})])
notes(slide, "45 s. Ankora gave us two challenges. First, build a language model for Ewe, a language spoken in Ghana and Togo. The brief suggested statistical counting models because data is scarce, so our question was: does simple counting really beat neural networks when data is limited? Second, take an existing English AI model and adapt it to healthcare and agriculture, making it fluent in medical papers and farming advice without catastrophic forgetting, meaning it should not lose its grasp of standard English.")

# ================================================================ 3. Data funnel
slide = content_slide("Most of a “4-million” corpus was noise", "SECTION B · DATA")
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

stat(slide, 7.0, 2.2, 5.6, "Only 6.7% usable", "295,198 clean Ewe sentences out of 4,408,322 rows.", big_size=40)
box(slide, 7.0, 4.1, 5.65, 2.75, [
    "- Massive duplication: one sentence appeared 1,296 times. Without removing copies, the model was 41% worse.",
    "- Foreign contamination: English spam, Spanish and Japanese in the Ewe column. Without our language filter, the model was 25% worse.",
    "- The leakage trap: a random split let near-copies of test sentences into training and faked a 7.6% gain. We split by near-duplicate group instead.",
], 13, WHITE, space=7)
notes(slide, "60 s. We downloaded an open dataset of 4.4 million sentences, but when we looked at it, only 6.7% was clean Ewe. Over 77% was duplicate copies: one sentence, 'Akpe kakakaka miedze agbagba ŋtɔ', appeared 1,296 times. And 70% of the rest was not Ewe at all: English ads, Spanish, even Japanese. We tested each cleaning step by leaving it out: without removing duplicates the model was 41% worse; without the language filter, 25% worse. More data only helps if it is clean data. Finally, we grouped near-identical sentences before splitting, so no test sentence had a copy in training; a random split would have faked a 7.6% gain.")

# ================================================================ 4. n-gram by hand
slide = content_slide("How an n-gram works: counting word pairs", "SECTION B · HOW IT WORKS")
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
stat(slide, 7.4, 2.15, 5.2, "≈ 2 choices", "per word: a perplexity of about 2 means the model was as unsure as flipping a coin between two words.", big_size=38)
box(slide, 7.4, 4.35, 5.25, 2.5, [
    ("Perplexity = the model's confusion", {"b": True, "c": GOLD, "s": 15}),
    "- Its average number of equally likely choices per word.",
    "- Choosing between 2 words is far better than choosing between 500.",
    ("Lower is always better.", {"b": True, "c": WHITE, "s": 14}),
], 14, WHITE, space=6)
notes(slide, "45 s. An n-gram works like the autocomplete on your phone: to predict the next word, it counts how often words followed each other in the training text. We calculated this by hand on paper before writing any code: on five training sentences, our test sentence had a probability of 1 in 30. To evaluate models we use perplexity, the model's confusion: a perplexity of 2 means that at each step the model was choosing between about two equally likely words. Lower perplexity is always better.")

# ================================================================ 5. Sparsity
slide = content_slide("The zero problem: why counting fails", "SECTION B · WHY COUNTING FAILS")
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
notes(slide, "45 s. Why not stop at pure counting? Because of the zero problem. If a test sentence contains even one word combination that was never in the training text, its probability is zero, and because probabilities multiply, one zero makes the whole sentence impossible: infinite perplexity. On our real Ewe test data, 95.5% of sentences had such an unseen phrase. Making the context longer does not help; it causes memorisation. At order 9, 96% of generated sentences were word-for-word copies of the training data. High-order models sound fluent only because they are reciting.")

# ================================================================ 6. Smoothing
slide = content_slide("Smoothing: nothing is ever impossible", "SECTION B · SMOOTHING")
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
    ("Shave a slice of probability from what we saw; share it with what we did not.", {"b": True, "c": GOLD, "s": 14}),
    "- Add-one spreads it over every possible sequence: 88,754.",
    "- Kneser-Ney, the best (100.4), asks how many different words a word follows, not how often it appears.",
    "- “Francisco” is common but only follows “San”, so in a new context it is a bad guess.",
    "- Our code matches the standard tool KenLM within 0.16%.",
], 13, WHITE, space=6)
notes(slide, "60 s. To fix the zero problem we use smoothing. Think of probability as a pie: we shave a thin slice from the word sequences we saw and share it with the ones we did not, so nothing is ever zero. Simple methods fail badly: add-one scores over 88,000, and even tuned add-k over 14,000, because they spread the pie across millions of sequences that never occur. The best is Kneser-Ney, at 100.4. Its idea is continuation counts: it judges a word by how many different contexts it appears in, not by raw frequency. 'Francisco' is frequent, but it almost only follows 'San', so in a new context it is a poor guess. Our from-scratch code matched the standard research tool, KenLM, within 0.16%.")

# ================================================================ 7. Tokenisation
slide = content_slide("Words vs subwords: the optical illusion", "SECTION B · TOKENISATION")
card(slide, 0.68, 2.25, 6.8, 4.6)
box(slide, 0.98, 2.45, 6.2, 0.4, [("Best order per tokeniser, dev set", {"b": True, "c": DARK, "s": 14})])
table(slide, 0.98, 2.95, 6.2, [["tokeniser", "perplexity", "bits / char", "best order"],
                                ["characters", "3.2", "1.599", "12"], ["BPE 1k pieces", "13.8", "1.509", "10"],
                                ["BPE 2k pieces", "20.3", "1.496", "8"], ["BPE 16k pieces", "92.8", "1.540", "6"],
                                ["whitespace words*", "100.8", "1.443", "6"], ["lower-cased words*", "47.0", "1.429", "6"]],
      col_w=[2.3, 1.3, 1.3, 1.3], size=12, row_h=0.46, highlight=3)
box(slide, 0.98, 6.25, 6.2, 0.5, [("* before charging word models for spelling unknown words (then 1.516–1.627)", {"c": MUTED, "s": 10})])
stat(slide, 7.9, 2.1, 4.7, "1.496", "bits per character for subwords (BPE, 2,000 pieces): the fair winner.")
box(slide, 7.9, 4.35, 4.75, 2.5, [
    "- The illusion: characters look best by perplexity (3.2) but worst by bits per character.",
    "- Word models got a free pass on unknown words. Charged the real cost of spelling them, they fall to 1.516–1.627.",
    "- Bits per character measures compression, so it compares tokenizers fairly.",
], 13, WHITE, space=6)
notes(slide, "60 s. How should we cut text into pieces: words, characters or subwords? By perplexity, characters look far better than words, but that is an illusion: character guesses are tiny, easy guesses. To compare tokenizers fairly we use bits per character, which measures compression. Word models first looked slightly ahead, but they were getting a free pass: when they met an unknown word they paid one small flat cost instead of spelling it out. When we charged them the real cost of spelling unknown words, subwords, BPE with 2,000 pieces, won at 1.496 bits per character.")

# ================================================================ 8. Scaling
slide = content_slide("n-gram vs neural: the 10,000-sentence rule", "SECTION B · N-GRAM VS NEURAL")
card(slide, 0.68, 2.25, 7.3, 4.6)
sizes = ["1000", "3000", "10000", "30000", "100000"]
ng = [s4[k]["ngram"]["pp"] for k in sizes]
rel = lambda key: [round(s4[k][key]["pp"] / n, 3) if key in s4[k] else None for k, n in zip(sizes, ng)]
# Drawn as an image (not a native chart): log axes and missing points render differently in PowerPoint,
# Keynote and Google Slides, and this one must look the same wherever it is opened.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
hexc = lambda c: "#" + str(c)
x = list(range(len(sizes)))
fig, ax = plt.subplots(figsize=(6.95, 4.3), dpi=220)
fig.patch.set_alpha(0); ax.set_facecolor("white")
plt.rcParams["font.family"] = "Avenir Next"
ax.axhspan(0.5, 1.0, color="#E6F0F2", zorder=0)                      # the zone where neural beats the n-gram
ax.axhline(1.0, color=hexc(MUTED), lw=2, ls=(0, (6, 4)), zorder=1)
ax.text(4.2, 1.06, "n-gram", va="bottom", ha="left", fontsize=11, color=hexc(MUTED), fontweight="bold")
ax.text(0.0, 0.62, "below the line: the neural model beats the n-gram", fontsize=10, color=hexc(TEAL), va="bottom")
series = (("LSTM", rel("lstm"), hexc(CORAL), "s", 3.0, 8),
          ("transformer", rel("transformer"), hexc(TEAL), "^", 3.0, 9),
          ("LSTM, sized to the data", rel("lstm_512"), hexc(AMBER), "o", 3.5, 10))
for label, ys, c, mk, lw, ms in series:
    pts = [(i, y) for i, y in zip(x, ys) if y is not None]
    ax.plot([i for i, _ in pts], [y for _, y in pts], color=c, marker=mk, lw=lw, ms=ms, mec="white", mew=1.5, label=label, zorder=3)
ax.set_yscale("log", base=2)
ax.set_ylim(0.5, 6.5)
ax.set_yticks([0.5, 1, 2, 4]); ax.set_yticklabels(["0.5×", "1×", "2×", "4×"])
ax.set_xticks(x); ax.set_xticklabels(["1k", "3k", "10k", "30k", "100k"])
ax.set_xlim(-0.3, 4.9)
ax.set_xlabel("training sentences", fontsize=11, color=hexc(DARK))
ax.set_ylabel("perplexity ÷ n-gram perplexity", fontsize=11, color=hexc(DARK))
ax.tick_params(colors=hexc(DARK), labelsize=11, length=0)
ax.grid(axis="y", color="#E3EAEC", lw=0.8); ax.set_axisbelow(True)
for side in ("top", "right", "left"): ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color("#C9D3D6")
ax.set_title("Each model's perplexity relative to the n-gram", fontsize=13, fontweight="bold", color=hexc(DARK), loc="left", pad=28)
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=10.5, handlelength=1.8, columnspacing=1.2, borderaxespad=0.2)
fig.tight_layout()
fig.savefig("results/stage4_relative.png", dpi=220, transparent=True)
plt.close(fig)
slide.shapes.add_picture("results/stage4_relative.png", Inches(0.85), Inches(2.4), Inches(6.95), Inches(4.3))
box(slide, 8.4, 2.3, 4.25, 4.55, [
    ("Each model's perplexity divided by the n-gram's, on the same tokens and test sentences.", {"c": PALE, "s": 12, "after": 8}),
    "- Under 10k sentences, counting wins: 353 vs 843 for the LSTM at 1,000.",
    "- The LSTM overtakes at 10k (145 vs 151); the transformer by 100k (51 vs 56).",
    "- At 100k, neural wins only when sized to the data: a wider LSTM scores 46.0 vs 55.6; the smaller one falls back level (57).",
    "- Same pattern on English and on Wikipedia.",
    "- Cost: n-gram under a minute on a CPU; LSTM about 4 hours on a GPU.",
], 12, WHITE, space=5)
notes(slide, "75 s. This is the centrepiece: are n-grams really better than neural networks for a low-resource language? The chart shows each model's perplexity divided by the n-gram's, so the n-gram is the flat dashed line at 1, and anything below it beats the n-gram. Our answer: only below about 10,000 sentences. At 1,000 sentences the LSTM is more than twice as bad as the n-gram and the transformer five times as bad. Counting works from the first sentence; a neural network starts from random weights and 1,000 sentences is not enough to learn useful word representations. Around 10,000 sentences the LSTM crosses the line, and the transformer crosses by 100,000. One caveat we measured: at 100,000 sentences our small LSTM ran out of room and drifted back to level with the n-gram; a wider LSTM, sized to the data, stayed clearly ahead, 46 against 56. So neural models win with enough data, if the model grows with it. We saw the same crossover on English and on Wikipedia. And compute matters: the n-gram trained in under a minute on a CPU, the LSTM in about four hours on a GPU.")

# ================================================================ 9. Section C set-up
slide = content_slide("Adapting an LLM: healthcare & agriculture", "SECTION C · SET-UP")
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
box(slide, 1.0, 4.58, 11.3, 0.35, [("WHY LORA: THE STICKY-NOTE METHOD", {"b": True, "c": AMBER, "s": 11})])
box(slide, 1.0, 5.0, 5.4, 1.7, [
    ("Full fine-tuning rewrites all 500 million weights: slow, expensive, and it risks breaking English.", {"c": WHITE, "s": 14}),
], space=4)
box(slide, 6.8, 5.0, 5.5, 1.7, [
    ("LoRA freezes the model and trains small add-on matrices beside it: 0.6% of the weights, on a laptop, and one small adapter per domain.", {"c": WHITE, "s": 14}),
], space=4)
notes(slide, "45 s. Now Section C, domain adaptation. We took Qwen2.5, a half-billion-parameter base model, and adapted it to healthcare, using medical research abstracts, and agriculture, using farmers' questions and answers. How do you adapt a model on a laptop without destroying what it already knows? We used LoRA. Full fine-tuning is like rewriting a printed textbook in pen. LoRA leaves the textbook frozen and attaches small sticky notes on some pages. It trained only 0.6% of the weights, ran on a laptop, and keeps the base model's English safe from being overwritten.")

# ================================================================ 10. Learning rate
slide = content_slide("The step-size trap: moving too fast broke English", "SECTION C · TRAINING")
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
notes(slide, "45 s. When we first trained LoRA we hit a puzzle: the larger adapter, rank 8, did worse than the tiny one, rank 2. When a result contradicts theory, it is usually a measurement or tuning problem. We found the learning rate, 1e-4, was too large: the model overshot and its general-English perplexity rose 26%. We tried three rates, chosen on the validation split and never on the test set. At 1e-5 in-domain learning improved and general English stayed essentially unchanged.")

# ================================================================ 11. Results
slide = content_slide("Results: specialisation vs forgetting", "SECTION C · RESULTS")
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
    "- Agriculture −58%; health only −6%: the base already knew medical English.",
    "- Rank is a dial: rank 32 learned most but was 20% worse on general English.",
    "- Full fine-tuning: 80× the weights of LoRA rank 8, not better overall.",
], 12, WHITE, space=5)
notes(slide, "75 s. First, adaptation helps most where the base model is weakest: agriculture perplexity fell 58%, healthcare only 6%, because the web is full of medical research and the model already knew it, while farmers' question-and-answer text was new to it. Second, adapter rank is a dial for forgetting: rank 2 gained 52% with no loss of general English; rank 8 gained 58% with 3% loss; rank 32 gained 63% but was 20% worse on general English. Third, full fine-tuning changed 80 times more weights than LoRA rank 8, was worse on health and only 4% better on agriculture. We checked these differences are real by repeating runs with different random seeds.")

# ================================================================ 12. Lessons and limits
slide = content_slide("Key takeaways and honest limits", "TAKEAWAYS")
learned = [
    ("Clean data beats big data", "Without cleaning, a 3× bigger corpus made the model 25% worse."),
    ("The 10,000-sentence rule", "n-grams win below about 10k sentences; neural models win above."),
    ("LoRA is efficient", "0.6% of the weights did as well as full fine-tuning."),
]
for i, (h, b) in enumerate(learned):
    y = 2.3 + i * 1.5
    card(slide, 0.68, y, 6.9, 1.3)
    num = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.9), Inches(y + 0.35), Inches(0.6), Inches(0.6))
    num.fill.solid(); num.fill.fore_color.rgb = GOLD; num.line.fill.background()
    tf = num.text_frame; tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    fill(tf, [(str(i + 1), {"b": True, "c": DARK, "s": 16})], align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, space=0)
    box(slide, 1.75, y + 0.15, 5.6, 1.0, [(h, {"b": True, "c": DARK, "s": 16}), (b, {"c": DARK, "s": 13})], space=2, anchor=MSO_ANCHOR.MIDDLE)
card(slide, 7.9, 2.3, 4.75, 4.3, color=DARK, shadow=False)
box(slide, 8.2, 2.5, 4.2, 4.0, [
    ("FLUENT IS NOT CORRECT", {"b": True, "c": AMBER, "s": 12}),
    ("Asked how to control fall armyworm in maize, an adapted model answered: “The fall armyworm in maize affects the development of mites, insects and other insects.”", {"i": True, "c": WHITE, "s": 14, "after": 10}),
    ("Adaptation changes style and vocabulary, not truth. Use these models to score speech transcripts, never to give farmers or patients advice.", {"c": PALE, "s": 13}),
], space=6)
notes(slide, "45 s. To conclude: first, clean data beats big data; a three-times-bigger uncleaned corpus made the model 25% worse. Second, n-grams beat neural networks only below about 10,000 sentences. Third, LoRA adapts a model with 0.6% of its weights while protecting its general English. We end on an honest limit: fluent is not the same as correct. Asked how to control fall armyworm, an adapted model answered in confident prose that was simply wrong. Domain adaptation changes style and vocabulary, not truth. These models are useful for scoring speech-recognition transcripts, but must never give unverified advice in agriculture or medicine.")

# ================================================================ 13. AI declaration
slide = content_slide("AI declaration", "DECLARATION")
card(slide, 0.68, 2.3, 11.95, 3.3)
box(slide, 1.05, 2.55, 11.2, 0.4, [("We used AI tools in this project to:", {"b": True, "c": DARK, "s": 16})])
uses = [("Write and debug code", "for the models, experiments, notebook and charts."),
        ("Explain concepts", "most of the ideas we learned along the way (smoothing, perplexity, LoRA and more)."),
        ("Generate the presentation", "the first version of this slide document, which we then refined.")]
for i, (h, b) in enumerate(uses):
    x = 1.05 + i * 3.8
    box(slide, x, 3.15, 3.5, 2.3, [(h, {"b": True, "c": CORAL, "s": 15}), (b, {"c": DARK, "s": 13})], space=4)
box(slide, 0.68, 5.85, 11.95, 1.0, [
    ("Every number comes from a results file written by a script in our repository, and our key results were checked against hand calculations and independent tools (NLTK, KenLM).", {"c": WHITE, "s": 13}),
], space=4)
notes(slide, "10 s. A note on how we worked: we used AI tools to write and debug code, to explain concepts as we learned them, and to generate the first version of this presentation, which we refined. Every number traces to a script in our repository.")

# ================================================================ 14. End
slide = prs.slides.add_slide(L["End Slide"])
for ph in list(slide.placeholders):
    if ph.placeholder_format.idx != 0:
        ph._element.getparent().remove(ph._element)
fill(slide.shapes.title.text_frame, [("Thank you", {"b": True, "c": WHITE, "s": 66, "f": HEAD}),
                                     ("Questions?", {"b": True, "c": GOLD, "s": 40, "f": HEAD})], space=4)
box(slide, 0.62, 5.6, 8.0, 0.5, [("Group 1  ·  MICS 2028  ·  Ashesi University", {"c": WHITE, "s": 14})])
box(slide, 0.62, 6.1, 8.0, 0.5, [(GITHUB_URL, {"c": PALE, "s": 13})])
notes(slide, "Thank you. We are ready to take your questions.")

prs.save(OUT)
print(f"wrote {OUT}: {len(prs.slides)} slides")
