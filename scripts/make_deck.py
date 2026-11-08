"""Build docs/AutoLab-pitch.pptx (the pitch deck, with the trained-robot videos embedded).

  .venv/bin/python scripts/make_deck.py

Layout is drawn on a 1920x1080 grid (px) and converted: 1 px = 6350 EMU, 1 px of type = 0.5 pt.
"""
from __future__ import annotations

from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "docs" / "media"
OUT = ROOT / "docs" / "AutoLab-pitch.pptx"

IVORY, SAND, CHAR, INK, BODY, MUTED = "FAF9F5", "F0EEE6", "1F1E1D", "141413", "3D3D3A", "6B6A65"
CLAY, CLAY_LIGHT, ON_DARK, ON_DARK_2, LINE, WHITE = "B5532F", "E08A68", "F5F4EE", "D6D4CB", "E8E6DC", "FFFFFF"
SERIF, SANS = "Source Serif 4", "Inter"

px = lambda v: Emu(int(v * 6350))
rgb = lambda h: RGBColor.from_string(h)

prs = Presentation()
prs.slide_width, prs.slide_height = px(1920), px(1080)
BLANK = prs.slide_layouts[6]


def slide(bg, notes=""):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = rgb(bg)
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def text(s, x, y, w, h, runs, size=28, color=BODY, font=SANS, bold=False, italic=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, spacing=1.15):
    """runs: str, or list of paragraphs; a paragraph is str or list of (text, overrides) tuples."""
    tb = s.shapes.add_textbox(px(x), px(y), px(w), px(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paras = [runs] if isinstance(runs, str) else runs
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        for seg in ([(para, {})] if isinstance(para, str) else para):
            t, o = seg
            r = p.add_run()
            r.text = t
            f = r.font
            f.size = Pt(o.get("size", size) / 2)
            f.name = o.get("font", font)
            f.bold = o.get("bold", bold)
            f.italic = o.get("italic", italic)
            f.color.rgb = rgb(o.get("color", color))
    return tb


def box(s, x, y, w, h, fill=WHITE, line=LINE, radius=True, dash=False, line_w=1.0):
    shp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE, px(x), px(y), px(w), px(h))
    if radius:
        shp.adjustments[0] = min(0.5, 20 / min(w, h))
    shp.fill.solid()
    shp.fill.fore_color.rgb = rgb(fill)
    if line:
        shp.line.color.rgb = rgb(line)
        shp.line.width = Pt(line_w)
        if dash:
            shp.line.dash_style = 7  # dash
    else:
        shp.line.fill.background()
    shp.shadow.inherit = False
    shp.text_frame.text = ""
    return shp


def arrow(s, x1, y1, x2, y2, color=CLAY, width=2.25, head=True, dash=False):
    c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, px(x1), px(y1), px(x2), px(y2))
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(width)
    ln = c.line._get_or_add_ln()
    if dash:
        d = etree.SubElement(ln, qn("a:prstDash")); d.set("val", "dash")
    if head:
        t = etree.SubElement(ln, qn("a:tailEnd")); t.set("type", "triangle"); t.set("w", "med"); t.set("len", "med")
    return c


def eyebrow(s, t, color=CLAY, y=128):
    text(s, 128, y, 1664, 40, t.upper(), size=24, color=color, bold=False)


def title(s, t, color=INK, y=180, size=72, h=200):
    text(s, 128, y, 1664, h, t, size=size, color=color, font=SERIF, spacing=1.05)


def footer(s, t, color=MUTED):
    text(s, 128, 980, 1664, 40, t, size=24, color=color)


def picture(s, path, x, y, w, h):
    return s.shapes.add_picture(str(path), px(x), px(y), px(w), px(h))


def movie(s, clip, poster, x, y, w, h):
    return s.shapes.add_movie(str(clip), px(x), px(y), px(w), px(h), poster_frame_image=str(poster), mime_type="video/mp4")


def cards(s, items, x, y, w, h, cols, gap=28, fill=WHITE, line=LINE, title_size=34, body_size=28, number=False):
    rows = (len(items) + cols - 1) // cols
    cw = (w - gap * (cols - 1)) / cols
    ch = (h - gap * (rows - 1)) / rows
    for i, (head, body) in enumerate(items):
        cx, cy = x + (i % cols) * (cw + gap), y + (i // cols) * (ch + gap)
        box(s, cx, cy, cw, ch, fill=fill, line=line)
        ty = cy + 40
        if number:
            text(s, cx + 44, ty, cw - 88, 80, f"{i + 1:02d}", size=72, color=CLAY, font=SERIF)
            ty += 96
        text(s, cx + 44, ty, cw - 88, 60, head, size=title_size, color=INK, bold=True)
        text(s, cx + 44, ty + title_size * 1.6, cw - 88, ch - (ty - cy) - 60, body, size=body_size, color=BODY, spacing=1.25)


# 1 ─ cover
s = slide(CHAR, "AutoLab gives every company its own autonomous research lab. The robot on the right was trained by our agents today, inside sandboxes on Vultr.")
text(s, 128, 300, 900, 40, "VULTR AGENT RUSH · BLAST RADIUS ZERO", size=24, color=CLAY_LIGHT)
text(s, 128, 370, 900, 200, "AutoLab", size=168, color=ON_DARK, font=SERIF)
text(s, 128, 590, 900, 200, "Your own autonomous research lab — on your cloud, with your data.", size=44, color=ON_DARK_2, spacing=1.2)
movie(s, MEDIA / "panda_reach_trained.mp4", MEDIA / "panda_reach_trained.png", 1032, 255, 760, 570)

# 2 ─ opportunity
s = slide(IVORY, "The edge of a frontier lab is not one genius model. It is a loop: agents try many experiments, keep what works and repeat.")
eyebrow(s, "The opportunity")
title(s, "Frontier AI labs pull ahead by letting agents run the research loop.")
cards(s, [("Try hundreds of ideas", "Agents propose experiments and run them in parallel."),
          ("Keep what works", "Weak runs are cut early; strong ones get more compute."),
          ("Repeat, nonstop", "Humans set the goal; the loop does the grinding.")], 128, 520, 1664, 400, 3, number=True)

# 3 ─ gap
s = slide(IVORY, "The gap is not ambition. Enterprises lack the specialised people, the experiment infrastructure, and a safe way to use their private data with AI.")
eyebrow(s, "The problem")
title(s, "Fortune 500 companies can’t run that loop.", h=110)
rows = [("", "Frontier AI lab", "Typical enterprise"), ("Research team", "Dedicated ML researchers", "Few or none"),
        ("Infrastructure", "Clusters, schedulers, evals", "Spreadsheets and one-off scripts"),
        ("Data", "Used freely in-house", "Can’t leave the company"), ("The best model", "Owned", "Rented from a vendor")]
tbl = s.shapes.add_table(len(rows), 3, px(128), px(380), px(1664), px(500)).table
for c, wdt in enumerate((500, 582, 582)):
    tbl.columns[c].width = px(wdt)
for r, row in enumerate(rows):
    for c, val in enumerate(row):
        cell = tbl.cell(r, c)
        cell.fill.solid(); cell.fill.fore_color.rgb = rgb(IVORY if r else SAND)
        cell.text = val
        para = cell.text_frame.paragraphs[0]
        f = para.runs[0].font if para.runs else para.font
        f.size, f.name = Pt(15), SANS
        f.bold = r == 0
        f.color.rgb = rgb(MUTED if c == 0 else (INK if r == 0 else BODY))

# 4 ─ problem
s = slide(SAND, "Three blockers are about capacity. The fourth is about trust: an autonomous agent writes and runs code, and without containment nobody will allow it near company data.")
eyebrow(s, "The problem")
title(s, "Four blockers keep the loop out of reach.", h=110)
cards(s, [("No specialised team", "ML researchers are scarce, and expensive to hire and keep."),
          ("No experiment infrastructure", "Running hundreds of trials needs schedulers, tracking and evaluation."),
          ("Data can’t leave", "Privacy rules and contracts forbid sending it to outside AI vendors."),
          ("Agents that run code are risky", "Security teams won’t let AI execute code next to company systems.")],
      128, 380, 1664, 570, 2, fill=IVORY, line="E1DED2")

# 5 ─ solution
s = slide(CHAR, "One sentence to remember: own lab, own cloud, own data, own model.")
eyebrow(s, "The solution", color=CLAY_LIGHT, y=300)
text(s, 128, 360, 1664, 500, [[("AutoLab gives every company its own autonomous research lab — on its own cloud and its own data — so the best model for its problem is one it ", {}),
                               ("owns", {"italic": True, "color": CLAY_LIGHT}), (".", {})]], size=80, color=ON_DARK, font=SERIF, spacing=1.1)

# 6 ─ loop
s = slide(IVORY, "Every step except Ask is done by agents. The LLM only plans and decides; training code runs in sandboxes, and the numbers come from code, not the LLM.")
eyebrow(s, "How it works")
title(s, "One question in, a verified model out.", h=110)
steps = [("Ask", "State the goal in plain English."), ("Plan", "Agents pick candidates and bet on outcomes first."),
         ("Train", "Each run in its own locked sandbox."), ("Steer", "Cut weak runs, rescue or branch strong ones."),
         ("Verify", "Re-test the winner on situations it never saw.")]
cw, gap = (1664 - 4 * 40) / 5, 40
for i, (h, b) in enumerate(steps):
    x = 128 + i * (cw + gap)
    box(s, x, 420, cw, 380)
    text(s, x + 32, 452, cw - 64, 80, f"{i + 1}", size=56, color=CLAY, font=SERIF)
    text(s, x + 32, 540, cw - 64, 50, h, size=32, color=INK, bold=True)
    text(s, x + 32, 600, cw - 64, 190, b, size=26, color=BODY, spacing=1.25)
    if i < 4:
        arrow(s, x + cw + 4, 610, x + cw + gap - 4, 610)

# 7 ─ agents architecture
s = slide(IVORY, "The lead agent is an LLM on Vultr Serverless Inference. It only plans and decides; it never executes code. The scheduler turns each experiment into its own gVisor sandbox on a Vultr VM, so runs are isolated from each other and from the company. Results stream back so the lead can cut weak runs, rescue promising ones or branch a leader. A separate fresh sandbox re-tests the winner.")
eyebrow(s, "Architecture · the agents")
title(s, "One lead agent, many sandboxed training runs.", size=64, h=100)
box(s, 128, 400, 270, 220)
text(s, 156, 428, 220, 50, "You", size=32, color=INK, bold=True)
text(s, 156, 486, 220, 120, "Ask one question in plain English.", size=24, color=BODY, spacing=1.25)
box(s, 470, 330, 430, 380, fill=CHAR, line=None)
text(s, 500, 360, 380, 50, "Lead agent", size=34, color=ON_DARK, bold=True)
text(s, 500, 420, 380, 220, ["Plans runs, places bets", "Watches every curve", "Cuts, rescues, branches", "Verifies, writes report"],
     size=26, color=ON_DARK_2, spacing=1.35)
text(s, 500, 650, 380, 40, "Vultr Serverless Inference", size=24, color=CLAY_LIGHT)
box(s, 970, 370, 300, 300)
text(s, 998, 398, 250, 50, "Scheduler", size=32, color=INK, bold=True)
text(s, 998, 458, 250, 200, ["Queues runs", "1 sandbox per run", "Streams results"], size=26, color=BODY, spacing=1.35)
box(s, 1340, 300, 452, 460, fill=SAND, line=CLAY, dash=True, line_w=1.5)
text(s, 1364, 318, 420, 40, "Vultr VM · gVisor sandboxes", size=24, color=CLAY, bold=True)
for i, (h, b) in enumerate([("Run A · baseline", "trains, films checkpoints"), ("Run B · faster learning", "trains, films checkpoints"),
                            ("Verifier", "re-tests the winner, fresh box")]):
    y = 370 + i * 128
    box(s, 1364, y, 404, 112, line="E1DED2")
    text(s, 1384, y + 14, 370, 40, h, size=26, color=INK, bold=True)
    text(s, 1384, y + 56, 370, 40, b, size=24, color=MUTED)
arrow(s, 398, 510, 470, 510); arrow(s, 900, 510, 970, 510); arrow(s, 1270, 510, 1340, 510)
arrow(s, 1566, 760, 1566, 840, color=MUTED, head=False, dash=True)
arrow(s, 1566, 840, 685, 840, color=MUTED, head=False, dash=True)
arrow(s, 685, 840, 685, 714, color=MUTED, dash=True)
text(s, 760, 852, 760, 40, "metrics · videos · checkpoints flow back", size=24, color=MUTED, align=PP_ALIGN.CENTER)
footer(s, "Each sandbox: 2 CPUs · 3 GB · read-only · no internet · destroyed after the run. The lead agent never runs code itself.")

# 8 ─ trust
s = slide(SAND, "Each result in the report links to the run that produced it and to a SHA-256 fingerprint of the saved model. Predictions are written before training and scored afterwards.")
eyebrow(s, "Built for trust")
title(s, "Safe to switch on, easy to audit.", h=110)
cards(s, [("Sandboxed by default", "gVisor isolation, no internet, read-only system, CPU and memory caps."),
          ("Data stays home", "Runs in the company’s own cloud account; agents see scores, not raw data."),
          ("Verified, not lucky", "Several runs per finalist, re-tested on unseen situations, ranges reported."),
          ("Every number traceable", "Bets written before training, a full decision log, model fingerprints.")],
      128, 380, 1664, 570, 2, fill=IVORY, line="E1DED2")

# 9 ─ vultr
s = slide(IVORY, "Three parts. Vultr Serverless Inference is the brain. The control plane schedules and verifies. A Vultr VM runs each experiment in its own gVisor sandbox with no network access. No GPUs are needed.")
eyebrow(s, "Architecture")
title(s, "Everything runs on Vultr.", h=110)
parts = [("Serverless Inference", "The agents’ reasoning: plans, bets, decisions.", False),
         ("AutoLab control", "Schedules runs, verifies results, writes the report.", False),
         ("Sandboxed VM", "gVisor containers, no network, 3 experiments at once.", True)]
bw, gap = (1664 - 2 * 96) / 3, 96
for i, (h, b, dark) in enumerate(parts):
    x = 128 + i * (bw + gap)
    box(s, x, 420, bw, 360, fill=CHAR if dark else WHITE, line=None if dark else LINE)
    text(s, x + 40, 470, bw - 80, 60, h, size=34, color=ON_DARK if dark else INK, bold=True)
    text(s, x + 40, 540, bw - 80, 220, b, size=28, color=ON_DARK_2 if dark else BODY, spacing=1.25)
    if i < 2:
        arrow(s, x + bw + 10, 600, x + bw + gap - 10, 600, width=3)
footer(s, "Measured: ~950 training steps per second per sandbox · 8 vCPU worker · about $0.13 per hour")

# 10 ─ robots (videos)
s = slide(IVORY, "One plain-English question produced this study: 3 setups, 2 robots, 3 runs each, re-tested on new situations. The clips play in PowerPoint; each is the best run.")
eyebrow(s, "Output · one study")
title(s, "Agents taught two robots to move.", h=110)
for i, (name, clip, poster, big, best) in enumerate([("Hopper", "hopper_trained.mp4", "hopper_trained.png", "2,850", "3,366"),
                                                      ("Walker2d", "walker2d_trained.mp4", "walker2d_trained.png", "1,439", "3,198")]):
    x = 128 + i * 852
    movie(s, MEDIA / clip, MEDIA / poster, x, 390, 400, 400)
    text(s, x + 432, 450, 380, 40, name, size=28, color=MUTED)
    text(s, x + 432, 500, 380, 120, big, size=96, color=INK, font=SERIF)
    text(s, x + 432, 620, 380, 80, "typical reward on unseen tests", size=26, color=BODY)
    text(s, x + 432, 700, 380, 40, f"Clip: best run, {best}", size=24, color=MUTED)
footer(s, "3 agent setups × 2 robots × 3 runs each · 42 sandboxed jobs · 4.2M training steps · 2.7 vCPU-hours · no GPU")

# 11 ─ panda
s = slide(SAND, "A Franka Panda arm. Two agents trained side by side; the winner succeeded 9 times out of 10 on new targets.")
movie(s, MEDIA / "panda_reach_trained.mp4", MEDIA / "panda_reach_trained.png", 128, 210, 880, 660)
text(s, 1104, 260, 688, 40, "OUTPUT · ROBOT ARM", size=24, color=CLAY)
text(s, 1104, 320, 688, 230, "90%", size=200, color=INK, font=SERIF)
text(s, 1104, 570, 688, 140, "of reaches hit a target the arm had never seen.", size=36, color=INK, spacing=1.2)
text(s, 1104, 730, 688, 140, "2 agents, about 4 minutes. The agent without input normalisation managed 40%.", size=28, color=BODY, spacing=1.25)

# 12 ─ containment
s = slide(CHAR, "We fed the sandbox code that tries to exfiltrate data and escape. gVisor blocked both, live, while the real experiments kept running.")
eyebrow(s, "Output · blast radius zero", color=CLAY_LIGHT)
title(s, "We attacked our own lab mid-study.", color=ON_DARK, h=110)
rows = [("Malicious code tried to", "Sandbox result"), ("Reach the internet", "Blocked — “Network is unreachable”"),
        ("Write outside its sandbox", "Blocked — “Permission denied”"), ("Everything else", "Capped: 3 GB memory, 256 processes, 2 CPUs")]
for r, (a, b) in enumerate(rows):
    y = 400 + r * 110
    text(s, 128, y, 640, 60, a, size=30, color="9A988F" if r == 0 else ON_DARK)
    text(s, 768, y, 1024, 60, b, size=30, color="9A988F" if r == 0 else ON_DARK_2)
    arrow(s, 128, y + 80, 1792, y + 80, color="3A3936", width=1, head=False)
footer(s, "The code check was switched off on purpose, so the sandbox had to hold alone. The study kept training.", color="9A988F")

# 13 ─ deliverables
s = slide(IVORY, "The company walks away with an asset, not a subscription: the model file, the evidence, and the record of how it was chosen.")
eyebrow(s, "Output · what you keep")
title(s, "Not a demo. A model you own.", h=110)
cards(s, [("The winning model", "A file in your cloud, with its fingerprint."), ("Proof you can watch", "Video of every agent at every checkpoint."),
          ("A plain-English report", "The answer, its likely range, which bets held."), ("A full audit trail", "Every agent decision, with its reason.")],
      128, 380, 1664, 520, 2)

# 14 ─ close
s = slide(CHAR, "Close on the vision: the loop that frontier labs run, available to every company, inside its own walls.")
text(s, 128, 300, 1664, 260, [[("Every company, its own ", {}), ("research lab", {"italic": True, "color": CLAY_LIGHT}), (".", {})]],
     size=104, color=ON_DARK, font=SERIF, spacing=1.05)
text(s, 128, 590, 1600, 140, "Today: robots learning to move. The same engine fits any model a company should own — support routing, predictive maintenance, forecasting.",
     size=36, color=ON_DARK_2, spacing=1.3)
text(s, 128, 780, 1600, 40, "AutoLab · built on Vultr Serverless Inference and Vultr Cloud Compute", size=28, color="9A988F")

prs.save(OUT)
print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {len(prs.slides)} slides)")
