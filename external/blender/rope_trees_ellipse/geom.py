"""Geometry and timeline for the rope-between-two-trees animation
(Assignment 6, pp-21).  Pure Python: shared by the Blender
scene script and the overlay compositor.

Units are feet.  The trees stand in the vertical plane y = 0, with x
along the ground from the first tree (x = 0) to the second (x = 100)
and z up.  A = (0, 100), B = (100, 50) in that plane; rope 120 ft.
"""
import math

FPS = 30
AX, AZ = 0.0, 100.0
BX, BZ = 100.0, 50.0
ROPE = 120.0
CX, CZ = (AX + BX) / 2, (AZ + BZ) / 2
AB = math.hypot(BX - AX, BZ - AZ)            # 111.80
SEMI_A = ROPE / 2                            # 60
FOCAL_C = AB / 2                             # 55.90
SEMI_B = math.sqrt(SEMI_A**2 - FOCAL_C**2)   # 21.79
U = ((BX - AX) / AB, (BZ - AZ) / AB)         # major-axis direction
V = (-U[1], U[0])                            # minor-axis direction


def ell(t):
    """Point of the ellipse |AP| + |PB| = 120 (same parametrization as
    the figure in the solution)."""
    c, s = math.cos(t), math.sin(t)
    return (CX + SEMI_A * c * U[0] + SEMI_B * s * V[0],
            CZ + SEMI_A * c * U[1] + SEMI_B * s * V[1])


def t_at_x(x, lo=3.5, hi=6.2):
    """Parameter of the lower-arc point with the given x (x is monotone
    increasing in t on this arc)."""
    for _ in range(80):
        mid = (lo + hi) / 2
        if ell(mid)[0] < x:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


T0 = t_at_x(0.0)      # bottle against the first tree
T1 = t_at_x(100.0)    # bottle against the second tree
X_START, X_END = 4.0, 96.0
TS = t_at_x(X_START)
TE = t_at_x(X_END)


def dist(p, q):
    return math.hypot(p[0] - q[0], p[1] - q[1])


# ---------------------------------------------------------------- easing
def clamp01(u):
    return 0.0 if u < 0 else 1.0 if u > 1 else u


def smooth(u):
    u = clamp01(u)
    return u * u * (3 - 2 * u)


def smoother(u):
    u = clamp01(u)
    return u * u * u * (u * (6 * u - 15) + 10)


def ramp(t, t0, t1):
    """0 before t0, 1 after t1, smooth in between (seconds)."""
    if t1 <= t0:
        return 1.0 if t >= t1 else 0.0
    return smooth((t - t0) / (t1 - t0))


def fade(t, a0, a1, b0=None, b1=None):
    """Fade in over [a0, a1] and (optionally) out over [b0, b1]."""
    v = ramp(t, a0, a1)
    if b0 is not None:
        v *= 1 - ramp(t, b0, b1)
    return v


def lerp(a, b, u):
    return a + (b - a) * u


def lerp3(a, b, u):
    return tuple(lerp(x, y, u) for x, y in zip(a, b))


# ---------------------------------------------------------------- timeline
# Scene boundaries, in seconds.
S2, S3, S4, S5, S6, S7, END = 4.0, 11.0, 29.0, 36.0, 48.0, 56.0, 64.0
RAPPEL = (11.6, 28.4)
LOOP = (38.0, 47.4)

# Camera poses: (location, target, lens mm).
CAM = {
    "c0":    ((178.0, -150.0, 20.0), (40.0, 0.0, 62.0), 30.0),
    "c2":    ((132.0, -190.0, 40.0), (46.0, 0.0, 63.0), 28.0),
    "c3":    ((-30.0, -196.0, 52.0), (54.0, 0.0, 66.0), 31.0),
    "face":  ((62.0, -235.0, 73.0), (62.0, 0.0, 73.0), 50.0),
    "final": ((156.0, -178.0, 34.0), (48.0, 0.0, 66.0), 30.0),
}


def cam_blend(k0, k1, u):
    (l0, t0, f0), (l1, t1, f1) = CAM[k0], CAM[k1]
    return lerp3(l0, l1, u), lerp3(t0, t1, u), lerp(f0, f1, u)


def camera(t):
    if t < S3:
        return cam_blend("c0", "c2", smooth(t / S3))
    if t < S4:
        return cam_blend("c2", "c3", smooth((t - S3) / (S4 - S3)))
    if t < S7:
        return cam_blend("c3", "face", smoother((t - S4) / 3.5))
    return cam_blend("face", "final", smoother((t - S7) / 3.6))


def p_param(t):
    """Unwrapped ellipse parameter of P at time t (seconds)."""
    if t < LOOP[0]:
        u = (t - RAPPEL[0]) / (RAPPEL[1] - RAPPEL[0])
        u = clamp01(u)
        # sine ease: the child speeds up down the slope and slows at the end
        u = 0.5 - 0.5 * math.cos(math.pi * u)
        return t_at_x(lerp(X_START, X_END, u))
    u = smooth((t - LOOP[0]) / (LOOP[1] - LOOP[0]))
    return TE + 2 * math.pi * u


def state(frame):
    t = frame / FPS
    loc, tgt, lens = camera(t)
    pt = p_param(t)
    P = ell(pt)
    # Solid trail: union of traversed pieces of the reachable arc [T0, T1].
    if t < LOOP[0]:
        solid_main = (TS, pt) if pt > TS + 1e-6 else None
        solid_pre = None
    else:
        solid_main = (TS, min(pt, T1))
        solid_pre = (T0, min(pt - 2 * math.pi, TS)) if pt - 2 * math.pi > T0 else None
    dash_upto = min(pt, T0 + 2 * math.pi) if pt > T1 else None
    # Inward label direction during the ride (between the rope pieces),
    # outward during the loop (the rider is gone).
    lab_in = fade(t, S3, S3 + 0.8, LOOP[0] - 1.6, LOOP[0] - 1.1) + fade(t, S7 + 0.3, S7 + 1.0)
    lab_out = fade(t, LOOP[0] - 0.9, LOOP[0] - 0.4, S7 - 0.1, S7 + 0.3)
    return {
        "t": t,
        "cam_loc": loc, "cam_tgt": tgt, "lens": lens,
        "P": P, "pt": pt,
        "solid_main": solid_main, "solid_pre": solid_pre, "dash_upto": dash_upto,
        "rider": 1 - fade(t, S5, S5 + 1.2) + fade(t, S7 + 0.3, S7 + 1.3),
        "trees": 1 - 0.78 * fade(t, S5, S5 + 1.5, S7, S7 + 2.0),
        "labAB": fade(t, S2 + 0.3, S2 + 1.1),
        "labP_in": lab_in, "labP_out": lab_out,
        "dims": fade(t, S2 + 1.2, S2 + 2.2, S3, S3 + 0.8),
        "rope_label": fade(t, S2 + 4.6, S2 + 5.4, S3, S3 + 0.8),
        "plane": 0.16 * fade(t, S7 + 1.5, S7 + 3.0),
    }
