"""Draw the schematics of both circuits into docs/figures/.

Component names and values are read from `ecgfd.circuit`, so the labels follow the
simulated netlist; the placement of the symbols is fixed here. Check a drawing
against `ecgfd --circuit <name> netlist` after changing a topology.

    pip install schemdraw
    python scripts/draw_schematics.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import schemdraw  # noqa: E402
import schemdraw.elements as elm  # noqa: E402

from ecgfd.circuit import INTEGRATED, REFERENCE, Circuit  # noqa: E402
from ecgfd.config import REPO_ROOT  # noqa: E402

OUT_DIR = REPO_ROOT / "docs" / "figures"
RAIL = {"integrated": 1.75, "reference": 3.5}  # half distance between the input rails


def _value(p) -> str:
    unit = "Ω" if p.kind == "R" else "F"
    for scale, prefix in ((1e6, "M"), (1e3, "k"), (1, ""), (1e-3, "m"), (1e-6, "µ"), (1e-9, "n")):
        if p.value >= scale:
            return f"{p.value / scale:g} {prefix}{unit}"
    return f"{p.value / 1e-12:g} p{unit}"


class Sheet:
    """Small drawing vocabulary on top of schemdraw, with absolute coordinates."""

    def __init__(self, circuit: Circuit):
        self.c = circuit
        self.d = schemdraw.Drawing(show=False)
        self.d.config(fontsize=11, bgcolor="white")

    def part(self, name: str, a, b, loc: str = "top"):
        p = self.c.passive(name)
        element = elm.Resistor() if p.kind == "R" else elm.Capacitor()
        self.d += element.endpoints(a, b).label(f"{name}\n{_value(p)}", loc=loc)

    def wire(self, *points):
        for a, b in zip(points, points[1:], strict=False):
            self.d += elm.Line().endpoints(a, b)

    def dot(self, p):
        self.d += elm.Dot().at(p)

    def tag(self, p, text: str, loc: str = "right"):
        """Named net, connected by name to the other tags with the same text."""
        self.d += elm.Dot(open=True).at(p).label(text, loc=loc)

    def ref(self, p):
        """Signal reference: ground, or the mid-supply net of the single-supply circuit."""
        if self.c.ref == "0":
            self.d += elm.Ground().right().at(p)
        else:
            self.tag(p, "vref", "bottom")

    def text(self, p, text: str, size: int = 13):
        self.d += elm.Label().right().at(p).label(text, halign="left", fontsize=size)

    def opamp(self, name: str, plus, plus_on_top: bool = True):
        """Op-amp with its + input at `plus`; returns the placed element."""
        op = elm.Opamp(leads=True).right()
        if plus_on_top:
            op = op.flip()
        op = op.anchor("in2").at(plus).label(name, loc="center", ofst=(-0.4, 0))
        self.d += op
        return op

    def follower(self, name: str, plus):
        op = self.opamp(name, plus)
        out, inn = op.out, op.in1
        low = inn[1] - 1.0
        self.wire(out, (out[0], low), (inn[0] - 0.4, low), (inn[0] - 0.4, inn[1]), inn)
        self.dot(out)
        return op

    def save(self, stem: str):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        self.d.save(str(OUT_DIR / f"{stem}.svg"))
        self.d.save(str(OUT_DIR / f"{stem}.png"), dpi=130)


def input_network(s: Sheet, n: dict[str, str], x_end: float):
    """Electrode terminals, test sources and input network. Returns (inp, inn) rail ends."""
    y = RAIL[s.c.name]
    s.tag((0, y), "LA electrode", "left")
    s.tag((0, -y), "RA electrode", "left")
    s.d += elm.SourceV().endpoints((0, y), (2.5, y)).label("Vcal\n1 mV pulse", loc="top")
    s.wire((0, -y), (2.5, -y))
    s.part(n["rp_la"], (2.5, y), (5, y))
    s.part(n["rp_ra"], (2.5, -y), (5, -y), loc="bottom")
    s.wire((5, y), (x_end, y))
    s.wire((5, -y), (x_end, -y))

    # bias resistors, joined at the reference
    s.part(n["rb_la"], (6, y), (6, 0), loc="bottom")
    s.part(n["rb_ra"], (6, -y), (6, 0), loc="top")
    # common-mode RFI capacitors, joined at ground
    s.part(n["c_la"], (8.5, y), (8.5, 0), loc="bottom")
    s.part(n["c_ra"], (8.5, -y), (8.5, 0), loc="top")
    for x in (6, 8.5):
        s.dot((x, y)), s.dot((x, -y)), s.dot((x, 0))
    s.wire((6, 0), (6.6, 0))
    s.ref((6.6, 0)) if s.c.ref == "0" else s.tag((6.6, 0), "vref")
    s.wire((8.5, 0), (9.1, 0))
    s.d += elm.Ground().right().at((9.1, 0))

    x = 10.5
    if "c_diff" in n:
        s.part(n["c_diff"], (x, y), (x, -y), loc="bottom")
        s.dot((x, y)), s.dot((x, -y))
        x += 2.0
    s.d += elm.SourceI().endpoints((x, -y), (x, y)).label("Ilo\nlead-off test", loc="bottom")
    s.dot((x, y)), s.dot((x, -y))
    s.d += elm.Label().at((x_end - 0.3, y + 0.35)).label("inp", fontsize=10)
    s.d += elm.Label().at((x_end - 0.3, -y - 0.35)).label("inn", fontsize=10)
    return (x_end, y), (x_end, -y)


def back_half(s: Sheet, start, n: dict[str, str]):
    """High-pass, gain stage and Sallen-Key low-pass, from the INA output to the ADC."""
    x, y = start
    s.dot(start)
    s.d += elm.Label().at((x, y + 0.4)).label("ina_out", fontsize=10)
    hp = (x + 2.5, y)
    s.part(n["c_hp"], start, hp)
    s.dot(hp)
    s.part(n["r_hp"], hp, (hp[0], y - 2.5), loc="top")
    s.ref((hp[0], y - 2.5))
    s.wire(hp, (x + 4.5, y))
    gain = s.opamp(n["u_gain"], (x + 4.5, y))

    gfb = (gain.in1[0] - 0.5, y - 2.6)
    s.wire(gain.in1, (gfb[0], gain.in1[1]), gfb)
    s.dot(gfb)
    s.part(n["r_gref"], gfb, (gfb[0], y - 5.1), loc="bottom")
    s.ref((gfb[0], y - 5.1))
    gout = gain.out
    s.part(n["r_gfb"], gfb, (gout[0], gfb[1]), loc="bottom")
    s.wire((gout[0], gfb[1]), gout)
    s.dot(gout)

    gx, gy = gout
    sk1, sk2 = (gx + 3.0, gy), (gx + 6.0, gy)
    s.part(n["r_sk1"], gout, sk1, loc="bottom")
    s.part(n["r_sk2"], sk1, sk2)
    s.dot(sk1), s.dot(sk2)
    s.part(n["c_sk2"], sk2, (sk2[0], gy - 2.5), loc="top")
    s.d += elm.Ground().right().at((sk2[0], gy - 2.5))
    s.wire(sk2, (gx + 7.2, gy))
    buf = s.follower(n["u_sk"], (gx + 7.2, gy))
    top = gy + 2.2
    s.wire(sk1, (sk1[0], top))
    s.part(n["c_sk1"], (sk1[0], top), (buf.out[0], top))
    s.wire((buf.out[0], top), buf.out)
    s.wire(buf.out, (buf.out[0] + 1.5, buf.out[1]))
    s.tag((buf.out[0] + 1.5, buf.out[1]), "out → ADC")


def rld_integrator(s: Sheet, cm, n: dict[str, str]):
    """Driven-right-leg integrator from the common-mode node `cm` to the RL electrode."""
    s.dot(cm)
    s.d += elm.Label().at((cm[0], cm[1] - 0.4)).label("cm", fontsize=10)
    op = s.opamp(n["u_rld"], (cm[0] + 1.0, cm[1] - 1.24), plus_on_top=False)
    s.wire(cm, op.in1)
    out = op.out
    for name, height in ((n["c_rld"], 1.8), (n["r_rld"], 3.6)):
        s.wire(cm, (cm[0], cm[1] + height))
        s.part(name, (cm[0], cm[1] + height), (out[0], cm[1] + height))
        s.wire((out[0], cm[1] + height), out)
    s.dot(out)

    plus = (op.in2[0] - 0.6, op.in2[1])
    s.wire(op.in2, plus)
    low = (plus[0], plus[1] - 2.6)
    s.d += elm.SourceV().endpoints(low, plus).label("Vcmt\nCM test", loc="top")
    s.ref(low)
    end = (out[0] + 3.2, out[1])
    s.part(n["r_lim"], out, end)
    s.tag(end, "RL electrode")
    s.d += elm.Label().at((out[0] + 0.2, out[1] - 0.4)).label("rld_out", fontsize=10, halign="left")


def electrode_inset(s: Sheet, origin):
    x, y = origin
    s.text((x, y + 1.9), "Skin-electrode model (LA, RA and RL)", 11)
    s.tag((x, y), "body", "left")
    s.d += elm.SourceV().endpoints((x, y), (x + 2.2, y)).label("Ehc", loc="bottom").reverse()
    s.d += elm.Resistor().endpoints((x + 2.2, y), (x + 4.7, y)).label("Rs", loc="bottom")
    s.d += elm.Resistor().endpoints((x + 4.7, y), (x + 7.2, y)).label("Rd", loc="bottom")
    s.wire((x + 4.7, y), (x + 4.7, y + 1.2))
    s.d += elm.Capacitor().endpoints((x + 4.7, y + 1.2), (x + 7.2, y + 1.2)).label("Cd")
    s.wire((x + 7.2, y + 1.2), (x + 7.2, y))
    s.dot((x + 4.7, y)), s.dot((x + 7.2, y))
    s.wire((x + 7.2, y), (x + 8.0, y))
    s.tag((x + 8.0, y), "lead")


def draw_integrated() -> Sheet:
    s = Sheet(INTEGRATED)
    y = RAIL["integrated"]
    s.text((0, y + 5.2), "integrated: integrated INA + discrete network, single 3.3 V supply", 14)
    names = {"rp_la": "R1", "rp_ra": "R2", "rb_la": "R3", "rb_ra": "R4",
             "c_la": "C1", "c_ra": "C2", "c_diff": "C3"}
    inp, inn = input_network(s, names, x_end=14.0)

    ina = elm.Ic(
        size=(5.0, 5.2),
        pins=[
            elm.IcPin(name="IN+", side="left", slot="3/4", anchorname="inp"),
            elm.IcPin(name="IN−", side="left", slot="2/4", anchorname="inn"),
            elm.IcPin(name="RG+", side="top", slot="1/2", anchorname="rgp"),
            elm.IcPin(name="RG−", side="top", slot="2/2", anchorname="rgn"),
            elm.IcPin(name="REF", side="bot", slot="1/1", anchorname="ref"),
            elm.IcPin(name="OUT", side="right", slot="1/1", anchorname="out"),
        ],
    ).right().at((15.0, -2.6)).label("U1\nINA", loc="center")
    s.d += ina
    s.wire(inp, (inp[0], ina.inp[1]), ina.inp)
    s.wire(inn, (inn[0], ina.inn[1]), ina.inn)
    top = ina.rgp[1] + 2.4
    s.part("R5", ina.rgp, (ina.rgp[0], top), loc="top")
    s.part("R6", ina.rgn, (ina.rgn[0], top), loc="bottom")
    mid = (0.5 * (ina.rgp[0] + ina.rgn[0]), top)
    s.wire((ina.rgp[0], top), (ina.rgn[0], top))
    s.dot(mid)
    s.wire(mid, (mid[0], top + 0.6))
    s.tag((mid[0], top + 0.6), "mid", "top")
    s.tag(ina.ref, "vref", "bottom")
    s.wire(ina.out, (ina.out[0] + 0.8, ina.out[1]))
    back_half(s, (ina.out[0] + 0.8, ina.out[1]), {
        "c_hp": "C5", "r_hp": "R10", "u_gain": "U4", "r_gref": "R11", "r_gfb": "R12",
        "r_sk1": "R13", "r_sk2": "R14", "c_sk1": "C6", "c_sk2": "C7", "u_sk": "U5",
    })

    # second row: driven right leg, mid-supply reference, electrode model
    row = -10.5
    s.text((0, row + 5.2), "Driven right leg", 11)
    s.tag((0, row), "mid", "left")
    s.wire((0, row), (1.0, row))
    buf = s.follower("U2", (1.0, row))
    cm = (buf.out[0] + 3.0, buf.out[1])
    s.part("R7", buf.out, cm, loc="bottom")
    rld_integrator(s, cm, {"u_rld": "U3", "c_rld": "C4", "r_rld": "R8", "r_lim": "R9"})

    x = 19.5
    s.text((x - 0.5, row + 5.2), "Mid-supply reference", 11)
    s.d += elm.Vdd().right().at((x, row + 3.0)).label("VCC 3.3 V")
    s.part("R15", (x, row + 3.0), (x, row), loc="bottom")
    s.part("R16", (x, row), (x, row - 3.0), loc="bottom")
    s.part("C8", (x + 1.8, row), (x + 1.8, row - 3.0), loc="bottom")
    s.wire((x, row - 3.0), (x + 1.8, row - 3.0))
    s.d += elm.Ground().right().at((x, row - 3.0))
    s.dot((x, row)), s.dot((x + 1.8, row))
    s.wire((x, row), (x + 4.0, row))
    ref = s.follower("U6", (x + 4.0, row))
    s.wire(ref.out, (ref.out[0] + 1.0, ref.out[1]))
    s.tag((ref.out[0] + 1.0, ref.out[1]), "vref")

    electrode_inset(s, (30.5, row))
    return s


def draw_reference() -> Sheet:
    s = Sheet(REFERENCE)
    y = RAIL["reference"]
    s.text((0, y + 2.6), "reference: discrete three-op-amp INA, ±5 V supplies", 14)
    names = {"rp_la": "R1", "rp_ra": "R2", "rb_la": "R3", "rb_ra": "R4", "c_la": "C1", "c_ra": "C2"}
    inp, inn = input_network(s, names, x_end=12.5)

    u1 = s.opamp("U1", inp, plus_on_top=True)
    u2 = s.opamp("U2", inn, plus_on_top=False)
    xo = u1.out[0]
    fb1, fb2 = (xo, 1.0), (xo, -1.0)
    s.part("R5", u1.out, fb1, loc="bottom")
    s.part("R7", fb1, fb2, loc="bottom")
    s.part("R6", fb2, u2.out, loc="bottom")
    for op, fb in ((u1, fb1), (u2, fb2)):
        s.wire(op.in1, (op.in1[0] - 0.5, op.in1[1]), (op.in1[0] - 0.5, fb[1]), fb)
        s.dot(fb), s.dot(op.out)
    s.d += elm.Label().at((xo - 0.1, u1.out[1] + 0.4)).label("o1", fontsize=10)
    s.d += elm.Label().at((xo - 0.1, u2.out[1] - 0.4)).label("o2", fontsize=10)

    # difference stage
    p, q = (xo + 3.6, u1.out[1]), (xo + 3.6, u2.out[1])
    s.wire(u1.out, (xo + 0.8, u1.out[1]))
    s.wire(u2.out, (xo + 0.8, u2.out[1]))
    s.part("R10", (xo + 0.8, u1.out[1]), p)
    s.part("R8", (xo + 0.8, u2.out[1]), q, loc="bottom")
    u3 = s.opamp("U3", (xo + 5.2, 0.62), plus_on_top=True)
    s.wire(p, (p[0], u3.in2[1]), u3.in2)
    s.wire(q, (q[0], u3.in1[1]), u3.in1)
    s.dot(p), s.dot(q)
    s.part("R11", p, (xo + 6.6, p[1]))
    s.wire((xo + 6.6, p[1]), (xo + 6.6, p[1] - 0.4))
    s.d += elm.Ground().right().at((xo + 6.6, p[1] - 0.4))
    s.part("R9", q, (u3.out[0], q[1]), loc="bottom")
    s.wire((u3.out[0], q[1]), u3.out)
    s.wire(u3.out, (u3.out[0] + 1.0, u3.out[1]))
    back_half(s, (u3.out[0] + 1.0, u3.out[1]), {
        "c_hp": "C4", "r_hp": "R16", "u_gain": "U5", "r_gref": "R17", "r_gfb": "R18",
        "r_sk1": "R19", "r_sk2": "R20", "c_sk1": "C5", "c_sk2": "C6", "u_sk": "U6",
    })

    # second row: driven right leg and electrode model
    row = -12.5
    s.text((0, row + 5.2), "Driven right leg", 11)
    cm = (4.0, row)
    for name, net, dy in (("R12", "o1", 1.2), ("R13", "o2", -1.2)):
        s.tag((0, row + dy), net, "left")
        s.part(name, (0, row + dy), (3.0, row + dy), loc="top" if dy > 0 else "bottom")
        s.wire((3.0, row + dy), (3.0, row))
    s.wire((3.0, row), cm)
    s.dot((3.0, row))
    rld_integrator(s, cm, {"u_rld": "U4", "c_rld": "C3", "r_rld": "R14", "r_lim": "R15"})
    electrode_inset(s, (17.0, row))
    return s


def main() -> None:
    for draw in (draw_integrated, draw_reference):
        sheet = draw()
        sheet.save(f"schematic_{sheet.c.name}")
        print(f"wrote {OUT_DIR / f'schematic_{sheet.c.name}'}.svg and .png")


if __name__ == "__main__":
    main()
