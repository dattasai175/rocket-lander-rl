#!/usr/bin/env python3
"""
Train an AI to land a rocket on your laptop.

A from-scratch version of the classic Lunar Lander reinforcement-learning
project: a 2D rocket with a main engine and two side thrusters, a neural-net
policy, and *reward shaping* to make the learning problem tractable.

Only NumPy and Pillow are required (no Box2D / Gymnasium / PyTorch):

    pip install numpy pillow
    python rocket_lander_rl.py                 # train with shaping, write out/landing_shaped.gif
    python rocket_lander_rl.py --compare       # shaped vs. sparse reward side by side
    python rocket_lander_rl.py --no-shaping    # watch the agent fail to learn

How it works
------------
* Environment : 2D lander physics, 8-number observation (x, y, vx, vy, angle,
                angular velocity, left-leg contact, right-leg contact),
                4 actions (noop, left thruster, main engine, right thruster).
* Reward      : shaped = change in a "potential" (closer to pad, slower,
                upright, legs down) - fuel cost, plus +100 for a soft landing
                and -100 for a crash.  Sparse = only the +100 / -100.
* Learner     : evolution strategies on a small MLP policy (a black-box RL
                method: no backprop needed, and the whole population of
                landers is simulated in parallel as one NumPy batch).
"""
import argparse
import os
import time

import numpy as np
from PIL import Image, ImageDraw

# --------------------------------------------------------------------------
# Physics / task constants (world units: pad centred at x = 0, ground at y = 0)
# --------------------------------------------------------------------------
DT = 0.04              # seconds per step
GRAVITY = 0.6
MAIN_ACC = 1.5         # main-engine acceleration along the body axis
SIDE_ACC = 0.12        # lateral acceleration from a side thruster
SIDE_TORQUE = 5.0      # angular acceleration from a side thruster
LEG = 0.12             # half-width of the landing legs
PAD_HALF = 0.25        # half-width of the landing pad
START_Y = 1.5
MAX_STEPS = 500
SAFE_VY, SAFE_VX, SAFE_ANGLE = 0.45, 0.35, 0.25

ACTIONS = ("noop", "left", "main", "right")
OBS_DIM, N_ACTIONS = 8, 4


class LanderBatch:
    """`n` independent landers stepped together with vectorised NumPy."""

    def __init__(self, n, shaping=True):
        self.n, self.shaping = n, shaping

    # -- helpers -----------------------------------------------------------
    def _tips(self):
        d = LEG * np.sin(self.ang)
        return self.y - d, self.y + d            # left, right foot height

    def _potential(self):
        legL, legR = (t <= 0.04 for t in self._tips())
        dist = np.hypot(self.x, self.y)
        speed = np.hypot(self.vx, self.vy)
        return (-100 * dist - 100 * speed - 100 * np.abs(self.ang)
                + 10 * legL + 10 * legR)

    def obs(self):
        legL, legR = (t <= 0.04 for t in self._tips())
        return np.stack([self.x, self.y, self.vx, self.vy, 2 * self.ang,
                         self.w, legL, legR], axis=-1).astype(np.float32)

    # -- gym-style API -----------------------------------------------------
    def reset(self, rng):
        n = self.n
        self.x = rng.uniform(-0.6, 0.6, n)
        self.y = np.full(n, START_Y)
        self.vx = rng.uniform(-0.3, 0.3, n)
        self.vy = rng.uniform(-0.3, 0.0, n)
        self.ang = rng.uniform(-0.2, 0.2, n)
        self.w = rng.uniform(-0.3, 0.3, n)
        self.alive = np.ones(n, bool)
        self.safe = np.zeros(n, bool)       # touched down softly
        self.on_pad = np.zeros(n, bool)     # ... and on the pad
        self.crashed = np.zeros(n, bool)
        self.fuel = np.zeros(n)
        self.t = 0
        self.prev_pot = self._potential()
        return self.obs()

    def step(self, action):
        a = np.asarray(action)
        alive = self.alive
        main = (a == 2) & alive
        side = ((a == 1).astype(float) - (a == 3).astype(float)) * alive
        s, c = np.sin(self.ang), np.cos(self.ang)

        ax = -s * MAIN_ACC * main + c * SIDE_ACC * side
        ay = -GRAVITY + c * MAIN_ACC * main + s * SIDE_ACC * side
        alpha = -SIDE_TORQUE * side
        m = alive.astype(float)
        self.vx += ax * DT * m
        self.vy += ay * DT * m
        self.w += alpha * DT * m
        self.x += self.vx * DT * m
        self.y += self.vy * DT * m
        self.ang += self.w * DT * m
        self.t += 1

        low = np.minimum(*self._tips())
        touched = alive & (low <= 0)
        oob = alive & ((np.abs(self.x) > 1.6) | (self.y > 2.4))
        safe = (touched & (np.abs(self.vy) < SAFE_VY) & (np.abs(self.vx) < SAFE_VX)
                & (np.abs(self.ang) < SAFE_ANGLE))
        crashed = (touched & ~safe) | oob
        self.y = np.where(touched, self.y - low, self.y)   # rest feet on ground
        self.safe |= safe
        self.on_pad |= safe & (np.abs(self.x) <= PAD_HALF)
        self.crashed |= crashed

        reward = 100.0 * safe - 100.0 * crashed
        if self.shaping:
            pot = self._potential()
            fuel = 0.3 * main + 0.03 * (side != 0)
            reward = reward + (pot - self.prev_pot) * alive - fuel
            self.prev_pot = pot
            self.fuel += fuel
        reward = reward * alive

        done = alive & (touched | oob | (self.t >= MAX_STEPS))
        self.alive = alive & ~done
        return self.obs(), reward, done


# --------------------------------------------------------------------------
# Policy: tiny MLP, parameters packed in one flat vector so ES can perturb it
# --------------------------------------------------------------------------
HIDDEN = 16
N_PARAMS = OBS_DIM * HIDDEN + HIDDEN + HIDDEN * N_ACTIONS + N_ACTIONS


def unpack(params):
    """params: (P, N_PARAMS) -> W1 (P,8,H), b1 (P,1,H), W2 (P,H,4), b2 (P,1,4)"""
    p = params.shape[0]
    i = 0
    W1 = params[:, i:i + OBS_DIM * HIDDEN].reshape(p, OBS_DIM, HIDDEN); i += OBS_DIM * HIDDEN
    b1 = params[:, i:i + HIDDEN].reshape(p, 1, HIDDEN); i += HIDDEN
    W2 = params[:, i:i + HIDDEN * N_ACTIONS].reshape(p, HIDDEN, N_ACTIONS); i += HIDDEN * N_ACTIONS
    b2 = params[:, i:i + N_ACTIONS].reshape(p, 1, N_ACTIONS)
    return W1, b1, W2, b2


def act(params, obs):
    """params (P, N_PARAMS), obs (P, K, 8) -> greedy actions (P, K)."""
    W1, b1, W2, b2 = unpack(params)
    h = np.tanh(np.einsum("pkd,pdh->pkh", obs, W1) + b1)
    return np.argmax(np.einsum("pkh,pha->pka", h, W2) + b2, axis=-1)


def rollout(params, shaping, seed, episodes, record=False):
    """Run every parameter vector for `episodes` episodes, all in parallel.

    Returns (total reward, outcome, trajectory) with shapes (P, episodes),
    (P, episodes), list.  outcome: 0 timed out, 1 crashed, 2 landed softly
    off the pad, 3 landed softly on the pad.  The trajectory is only filled
    when record=True (used for rendering).
    """
    p = params.shape[0]
    env = LanderBatch(p * episodes, shaping)
    env.reset(np.random.default_rng(seed))
    # Every policy gets the same start states -> fair comparison.
    for name in ("x", "y", "vx", "vy", "ang", "w"):
        setattr(env, name, np.tile(getattr(env, name)[:episodes], p))
    env.prev_pot = env._potential()

    obs, total, traj = env.obs(), np.zeros(env.n), []
    while env.alive.any():
        a = act(params, obs.reshape(p, episodes, OBS_DIM)).reshape(-1)
        if record:
            traj.append((env.x.copy(), env.y.copy(), env.ang.copy(), a, env.alive.copy()))
        obs, r, _ = env.step(a)
        total += r
    if record:
        traj.append((env.x.copy(), env.y.copy(), env.ang.copy(),
                     np.zeros(env.n, int), env.alive.copy()))
    outcome = np.where(env.on_pad, 3, np.where(env.safe, 2, np.where(env.crashed, 1, 0)))
    return total.reshape(p, episodes), outcome.reshape(p, episodes), traj


# --------------------------------------------------------------------------
# Training: evolution strategies (antithetic sampling, rank shaping, Adam)
# --------------------------------------------------------------------------
def train(generations=150, pop=128, episodes=4, sigma=0.15, lr=0.05,
          shaping=True, seed=0, log=print):
    rng = np.random.default_rng(seed)
    theta = rng.standard_normal(N_PARAMS) * 0.3
    m = np.zeros(N_PARAMS); v = np.zeros(N_PARAMS)
    history = []
    t0 = time.time()
    for g in range(1, generations + 1):
        eps = rng.standard_normal((pop // 2, N_PARAMS))
        eps = np.concatenate([eps, -eps])
        returns, _, _ = rollout(theta + sigma * eps, shaping,
                                seed=10_000 * seed + g, episodes=episodes)
        fit = returns.mean(1)
        ranks = np.empty(pop); ranks[np.argsort(fit)] = np.arange(pop)
        w = ranks / (pop - 1) - 0.5                        # centred ranks
        grad = (w @ eps) / (pop * sigma)
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad ** 2
        step = (m / (1 - 0.9 ** g)) / (np.sqrt(v / (1 - 0.999 ** g)) + 1e-8)
        theta = theta + lr * step - 0.005 * lr * theta     # ascend + tiny decay

        if g == 1 or g % 10 == 0 or g == generations:
            r, success = evaluate(theta, shaping, episodes=100, seed=777)
            history.append((g, r, success))
            log(f"gen {g:4d} | reward {r:8.1f} | on-pad landings {success*100:5.1f}% "
                f"| {time.time()-t0:5.1f}s")
    return theta, history


def evaluate(theta, shaping=True, episodes=200, seed=12345):
    """Mean shaped-or-sparse return and % of safe on-pad landings."""
    r, outcome, _ = rollout(theta[None], shaping, seed, episodes)
    return float(r.mean()), float((outcome == 3).mean())


# --------------------------------------------------------------------------
# Rendering (Pillow only)
# --------------------------------------------------------------------------
W, H = 360, 480
SCALE = 270.0                       # pixels per world unit
GROUND_Y = H - 48


def to_px(x, y):
    return W / 2 + x * SCALE * 0.9, GROUND_Y - y * SCALE * 0.9


def _rot(pts, ang, cx, cy):
    s, c = np.sin(ang), np.cos(ang)
    return [to_px(cx + px * c - py * s, cy + px * s + py * c) for px, py in pts]


_STARS = np.random.default_rng(3).uniform(0, 1, (60, 2))


def draw_frame(x, y, ang, action, trail, caption, step):
    img = Image.new("RGB", (W, H), (8, 10, 28))
    d = ImageDraw.Draw(img)
    for sx, sy in _STARS:
        d.point((sx * W, sy * (GROUND_Y - 20)), fill=(170, 175, 210))
    # trail
    if len(trail) > 1:
        d.line([to_px(tx, ty) for tx, ty in trail], fill=(90, 110, 170), width=1)
    # exhaust flames (drawn first, behind the rocket)
    if action == 2:
        d.polygon(_rot([(-0.035, 0.0), (0.035, 0.0), (0.0, -0.17)], ang, x, y),
                  fill=(255, 150, 30))
        d.polygon(_rot([(-0.02, 0.0), (0.02, 0.0), (0.0, -0.10)], ang, x, y),
                  fill=(255, 235, 140))
    if action in (1, 3):
        sgn = -1 if action == 1 else 1          # left thruster sits on the left
        d.polygon(_rot([(sgn * 0.05, 0.27), (sgn * 0.05, 0.31), (sgn * 0.12, 0.29)],
                       ang, x, y), fill=(190, 225, 255))
    # ground + pad
    d.rectangle([0, GROUND_Y, W, H], fill=(52, 50, 58))
    l, r = to_px(-PAD_HALF, 0)[0], to_px(PAD_HALF, 0)[0]
    d.rectangle([l, GROUND_Y, r, GROUND_Y + 7], fill=(235, 190, 40))
    for px in (l, r):
        d.line([px, GROUND_Y, px, GROUND_Y - 22], fill=(235, 190, 40), width=2)
        d.polygon([(px, GROUND_Y - 22), (px + (8 if px > W/2 else -8), GROUND_Y - 18),
                   (px, GROUND_Y - 14)], fill=(220, 60, 60))
    # legs
    for sgn in (-1, 1):
        d.line(_rot([(sgn * 0.04, 0.06), (sgn * LEG, 0.0)], ang, x, y),
               fill=(190, 190, 200), width=3)
    # body, nose cone, fins
    d.polygon(_rot([(-0.05, 0.03), (0.05, 0.03), (0.05, 0.28), (-0.05, 0.28)],
                   ang, x, y), fill=(232, 234, 240))
    d.polygon(_rot([(-0.05, 0.28), (0.05, 0.28), (0.0, 0.40)], ang, x, y),
              fill=(200, 60, 60))
    d.polygon(_rot([(-0.05, 0.03), (-0.09, 0.0), (-0.05, 0.12)], ang, x, y),
              fill=(120, 124, 140))
    d.polygon(_rot([(0.05, 0.03), (0.09, 0.0), (0.05, 0.12)], ang, x, y),
              fill=(120, 124, 140))
    d.text((10, 8), caption, fill=(230, 230, 240))
    d.text((10, 22), f"step {step:3d}  action: {ACTIONS[action]}", fill=(150, 160, 190))
    return img


OUTCOME_TEXT = ("OUT OF TIME", "CRASHED", "LANDED (off pad)", "LANDED ON PAD!")


def episode_frames(traj, idx, caption, outcome, stride=2):
    """Turn one recorded episode into a list of PIL frames."""
    frames, trail = [], []
    for step, (xs, ys, angs, acts, alive) in enumerate(traj):
        x, y, ang, a = xs[idx], ys[idx], angs[idx], int(acts[idx])
        trail.append((x, y + 0.2))
        if step % stride == 0 or not alive[idx]:
            frames.append(draw_frame(x, y, ang, a if alive[idx] else 0, trail,
                                     caption, step))
        if not alive[idx]:
            ImageDraw.Draw(frames[-1]).text((10, 36), OUTCOME_TEXT[outcome],
                                            fill=(255, 220, 90))
            break
    return frames


def banner(text_lines, size=(W, H)):
    img = Image.new("RGB", size, (8, 10, 28))
    d = ImageDraw.Draw(img)
    y = size[1] // 2 - 8 * len(text_lines)
    for line in text_lines:
        d.text((24, y), line, fill=(235, 235, 245)); y += 18
    return img


def render_gif(theta_before, theta_after, shaping, path, n_after=4, seed=2024):
    frames = []
    # one attempt from an untrained policy, then a few from the trained one
    for label, th, n in (("untrained policy", theta_before, 1),
                         ("trained policy", theta_after, n_after)):
        _, outcome, traj = rollout(th[None], shaping, seed, n, record=True)
        for i in range(n):
            ok_frames = episode_frames(traj, i, f"{label}  -  episode {i+1}",
                                       int(outcome[0, i]))
            last = ok_frames[-1]
            frames += ok_frames + [last] * 12            # hold the final pose
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=40,
                   loop=0, optimize=True)
    return len(frames)


# --------------------------------------------------------------------------
# Tiny line chart (Pillow) for the learning curves
# --------------------------------------------------------------------------
def plot_curves(curves, path):
    """curves: {label: [(gen, reward, success), ...]} -> PNG with success %."""
    w, h, pad = 640, 360, 50
    img = Image.new("RGB", (w, h), (250, 250, 252))
    d = ImageDraw.Draw(img)
    gmax = max(c[-1][0] for c in curves.values())
    d.rectangle([pad, 20, w - 20, h - pad], outline=(120, 120, 130))
    for pct in range(0, 101, 25):
        yy = h - pad - (h - pad - 20) * pct / 100
        d.line([pad, yy, w - 20, yy], fill=(225, 225, 232))
        d.text((8, yy - 6), f"{pct}%", fill=(70, 70, 80))
    colors = [(30, 110, 210), (220, 80, 50), (40, 150, 80)]
    for (label, hist), col in zip(curves.items(), colors):
        pts = [(pad + (w - 20 - pad) * g / gmax,
                h - pad - (h - pad - 20) * s) for g, _, s in hist]
        d.line(pts, fill=col, width=3)
        d.text((pad + 10, 26 + 14 * list(curves).index(label)), f"- {label}", fill=col)
    d.text((w // 2 - 70, h - 28), "generation  (y: % safe landings on pad)", fill=(70, 70, 80))
    img.save(path)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--generations", type=int, default=150)
    ap.add_argument("--population", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-shaping", action="store_true",
                    help="train on the sparse +100/-100 reward only")
    ap.add_argument("--compare", action="store_true",
                    help="train shaped AND sparse, plot both learning curves")
    ap.add_argument("--out", default="out", help="output directory")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    modes = ([("shaped reward", True), ("sparse reward", False)] if args.compare
             else [("sparse reward", False)] if args.no_shaping
             else [("shaped reward", True)])
    curves, trained = {}, {}
    for label, shaping in modes:
        print(f"\n=== training with {label} ===")
        theta, hist = train(args.generations, args.population, shaping=shaping,
                            seed=args.seed)
        curves[label], trained[label] = hist, (theta, shaping)

    for label, (theta, shaping) in trained.items():
        # judge every agent by the same yardstick: safe landings on the pad
        _, success = evaluate(theta, shaping=False, episodes=500)
        print(f"\n[{label}] safe on-pad landings over 500 fresh episodes: "
              f"{success*100:.1f}%")

    plot_curves(curves, os.path.join(args.out, "learning_curve.png"))
    init = np.random.default_rng(args.seed).standard_normal(N_PARAMS) * 0.3
    for label, (theta, shaping) in trained.items():
        tag = "shaped" if shaping else "sparse"
        n = render_gif(init, theta, shaping, os.path.join(args.out, f"landing_{tag}.gif"))
        np.save(os.path.join(args.out, f"policy_{tag}.npy"), theta)
        print(f"wrote {args.out}/landing_{tag}.gif ({n} frames)")
    print(f"wrote {args.out}/learning_curve.png")


if __name__ == "__main__":
    main()
