
# app.py
# Streamlit interactive visual tool for "Clustering K-Anonymity (CKA)" demo
# Paper referenced: "Location Privacy Protection for the Internet of Things with Edge Computing Based on Clustering K-Anonymity"
# Jiang et al., Sensors 2024, 24, 6153

import math
import random
from typing import List, Tuple

import numpy as np
import streamlit as st
import plotly.graph_objects as go

# Utility functions

def sector_index(center: Tuple[float, float], point: Tuple[float, float], n_sectors: int) -> int:
    """
    Assign a point to one of N angular sectors (0..N-1) around center, clockwise from +x axis.
    """
    dx, dy = point[0] - center[0], point[1] - center[1]
    ang = math.atan2(dy, dx)  # [-pi, pi]
    if ang < 0:
        ang += 2 * math.pi
    sector = int((ang / (2 * math.pi)) * n_sectors) % n_sectors
    return sector


def circle_points(cx, cy, r, steps=256):
    t = np.linspace(0, 2 * np.pi, steps)
    return cx + r * np.cos(t), cy + r * np.sin(t)


def compute_k_function_metrics(points: np.ndarray, beta: float = 2.0):
    """
    Implements the DK and AK metrics from the paper (variance of min distances and combined index).
    points: (K, 2) array of selected K locations (real + virtual).
    """
    if len(points) < 2:
        return 0.0, 0.0

    # pairwise distances
    from scipy.spatial.distance import cdist
    D = cdist(points, points)
    np.fill_diagonal(D, np.inf)

    dmin = D.min(axis=1)  # (d(p)_min)
    dmax = D.max(axis=1)  # (d(p)_max)

    DK = np.var(dmin)
    AK = beta * np.mean(dmax) - DK
    return float(DK), float(AK)


def run_cka(
    users: np.ndarray,
    eaves: np.ndarray,
    head_idx: int,
    K: int,
    N: int,
    gamma: float,
    R: float,
    W: float,
    H: float,
    rng: np.random.Generator,
):
    """
    Execute one CKA selection around the chosen head node.
    Returns dict with:
      - head (x,y)
      - responders indices for users/eaves within R
      - selected_real_points (array)
      - virtual_points (array)
      - selected_all_points (array length K)
      - data_efficiency, P_est, DK, AK
    """
    head = users[head_idx]
    # Responders within communication radius (both legal users and eavesdroppers)
    def within_R(arr):
        return arr[np.linalg.norm(arr - head, axis=1) <= R]

    user_resp = within_R(users)
    # Exclude head duplicate if present
    # Keep the head as a real candidate too (paper assumes head can include itself)
    eav_resp = within_R(eaves)

    # Assign sectors to responders
    sectors_users = [sector_index(head, tuple(p), N) for p in user_resp]
    sectors_eavs = [sector_index(head, tuple(p), N) for p in eav_resp]

    # Partition responders by sector
    users_by_sector = {i: [] for i in range(N)}
    eavs_by_sector = {i: [] for i in range(N)}
    for p, s in zip(user_resp, sectors_users):
        users_by_sector[s].append(p)
    for p, s in zip(eav_resp, sectors_eavs):
        eavs_by_sector[s].append(p)

    # Per-paper threshold T(K,N) = floor(K/N) - 1
    T = (K // N) - 1

    # Cap on selected real nodes total: round(gamma * K / N)
    real_cap = int(round(gamma * K / N))

    selected_real = []
    virtual_pts = []

    # helper: random point in a sector annulus around head (0..R) but keep inside canvas
    def random_point_in_sector(sector_id):
        # pick radius in [R/3, R] to spread dummies; angle within sector range
        r = rng.uniform(R * 0.33, R)
        ang0 = (2 * math.pi / N) * sector_id
        ang1 = (2 * math.pi / N) * (sector_id + 1)
        theta = rng.uniform(ang0, ang1)
        x = head[0] + r * math.cos(theta)
        y = head[1] + r * math.sin(theta)
        # clip to canvas
        x = float(np.clip(x, 0, W))
        y = float(np.clip(y, 0, H))
        return np.array([x, y])

    # First pass: for each sector, ensure up to floor(K/N) entries, preferring real nodes if available,
    # but never exceeding total real_cap across sectors. When real are insufficient (<= T), create virtuals
    # to reach floor(K/N).
    per_sector_quota = K // N
    real_used = 0

    for i in range(N):
        can_use_reals = real_used < real_cap
        user_pool = users_by_sector[i].copy()
        eav_pool = eavs_by_sector[i].copy()

        # Combine user and (optionally) eaves into "real-like" pool for anonymity structure
        # But prefer legal users first (for "data efficiency" metric in our viz we'll only count real = legal users)
        # For selection, we treat both as "real" in the K set from attacker-perspective, but mark legal separately.
        combined_pool = user_pool + eav_pool

        if len(combined_pool) <= T:
            # Need to fill virtuals up to per_sector_quota
            # Use whatever real we have (but respect real_cap)
            take = min(len(combined_pool), per_sector_quota)
            if can_use_reals and take > 0:
                # choose random subset if more than quota
                choose_idx = rng.choice(len(combined_pool), size=take, replace=False) if len(combined_pool) > take else np.arange(take)
                chosen = [combined_pool[j] for j in choose_idx]
                selected_real.extend(chosen)
                real_used += sum(1 for c in chosen if any(np.array_equal(c, u) for u in user_pool))  # count only legal users
            # fill remaining with virtuals
            need_virtual = per_sector_quota - take
            for _ in range(need_virtual):
                virtual_pts.append(random_point_in_sector(i))
        else:
            # We have more than T, random select up to quota, but don't exceed real_cap (counting only legal users)
            if can_use_reals:
                # pick tentative candidates
                if len(combined_pool) > per_sector_quota:
                    choose_idx = rng.choice(len(combined_pool), size=per_sector_quota, replace=False)
                    chosen = [combined_pool[j] for j in choose_idx]
                else:
                    chosen = combined_pool
                # enforce real_cap w.r.t. legal users
                # keep all chosen but if legal real exceed cap, convert overflow to virtuals
                legal_in_chosen = [c for c in chosen if any(np.array_equal(c, u) for u in user_pool)]
                overflow = max(0, (real_used + len(legal_in_chosen)) - real_cap)
                if overflow > 0:
                    # remove 'overflow' legal choices and replace with virtuals
                    rng.shuffle(legal_in_chosen)
                    for k in range(overflow):
                        removed = legal_in_chosen[k]
                        # remove one occurrence from chosen
                        for idx, elem in enumerate(chosen):
                            if np.array_equal(elem, removed):
                                chosen.pop(idx)
                                break
                        virtual_pts.append(random_point_in_sector(i))
                    real_used += len(legal_in_chosen) - overflow
                else:
                    real_used += len(legal_in_chosen)

                selected_real.extend(chosen)
            else:
                # Only virtuals allowed if cap reached
                for _ in range(per_sector_quota):
                    virtual_pts.append(random_point_in_sector(i))

    # Handle remainder K - N*floor(K/N): assign at most one extra (virtual) per distinct sector
    remainder = K - N * (K // N)
    for i in range(remainder):
        virtual_pts.append(random_point_in_sector(i % N))

    # Build final K-set: include selected_real (could be >K due to logic, trim) + virtuals (to reach K)
    selected_all = selected_real + virtual_pts
    if len(selected_all) > K:
        # random trim to exactly K, but try to keep the structure roughly balanced
        idxs = rng.choice(len(selected_all), size=K, replace=False)
        selected_all = [selected_all[i] for i in idxs]
    elif len(selected_all) < K:
        # pad with additional virtuals uniformly across sectors
        for i in range(K - len(selected_all)):
            virtual_pts.append(random_point_in_sector(i % N))
            selected_all.append(virtual_pts[-1])

    selected_all_arr = np.array(selected_all)

    # Helper to check if a point equals any row in an array (float tolerant)
    def isin_rows(arr: np.ndarray, point: np.ndarray, tol=1e-9) -> bool:
        if arr.size == 0:
            return False
        return bool(np.any(np.all(np.isclose(arr, point, atol=tol), axis=1)))

    # Metrics masks
    legal_mask = np.array([isin_rows(users, p) for p in selected_all_arr])
    eav_mask   = np.array([isin_rows(eaves, p) for p in selected_all_arr])
    virtual_mask = ~(legal_mask | eav_mask)

    # For completeness (not strictly needed for plotting), collect arrays by type
    selected_real_arr = selected_all_arr[legal_mask | eav_mask]
    virtual_arr = selected_all_arr[virtual_mask]

    # Data efficiency = proportion of legal user locations within the K-set (not counting eaves or virtuals)
    data_eff = float(legal_mask.sum()) / float(K)

    # Theoretical P(K,N) from paper: round(gamma*K/N)/K
    P_est = int(round(gamma * K / N)) / K

    # DK and AK on the K points (positions)
    try:
        DK, AK = compute_k_function_metrics(selected_all_arr, beta=st.session_state.beta_for_AK if 'beta_for_AK' in st.session_state else 2.0)
    except Exception:
        DK, AK = 0.0, 0.0

    # Labels for plotting
    kinds = np.where(legal_mask, "legal_real", np.where(eav_mask, "eaves_real", "virtual")).tolist()

    return {
        "head": head,
        "user_resp": user_resp,
        "eav_resp": eav_resp,
        "selected_all": selected_all_arr,
        "selected_real": selected_real_arr,
        "virtual": virtual_arr,
        "data_eff": data_eff,
        "P_est": P_est,
        "DK": DK,
        "AK": AK,
        "kinds": kinds,
    }

# Streamlit UI

st.set_page_config(page_title="CKA — Clustering K-Anonymity Visual Tool", layout="wide")

st.title("📍 CKA — Clustering K-Anonymity (Interactive Visual Tool)")
st.caption("Based on *Location Privacy Protection for the IoT with Edge Computing using Clustering K-Anonymity (CKA), Sensors 2024*.")

with st.sidebar:
    st.header("Controls")
    # NOTE: Widgets below have explicit keys so Quick Controls can sync

    seed = st.number_input("Random Seed", min_value=0, max_value=10_000_000, value=42, step=1, key='seed')
    rng = np.random.default_rng(seed)

    W = st.number_input("Region Width (m)", 50, 1000, 100, 10, key='W')
    H = st.number_input("Region Height (m)", 50, 1000, 100, 10, key='H')

    num_users = st.slider("Number of Legal Users", 10, 1000, 120, 10, key='num_users')
    num_eaves = st.slider("Number of Eavesdroppers", 0, 200, 12, 1, key='num_eaves')

    R = st.slider("Head Communication Radius R", 10, int(min(W, H)), int(min(W, H) * 0.35), 1, key='R')

    st.divider()
    st.subheader("CKA Parameters")
    K = st.slider("Anonymity Degree K", 3, 64, 12, 1, key='K')
    N = st.slider("Number of Clusters (zones) N", 2, 24, 6, 1, key='N')
    gamma = st.slider("γ (real-cap weight)", 0.1, float(N), 1.5, 0.1, key='gamma')

    st.slider("β (AK adjustment factor)", 0.1, 20.0, 2.0, 0.1, key="beta_for_AK")

    st.divider()
    st.subheader("Head Node Selection")
    head_mode = st.selectbox("Pick head node", ["Random user", "Nearest to center"], key='head_mode')

    st.divider()
    st.subheader("Display Options")
    show_sectors = st.checkbox("Show sector boundaries", True, key='show_sectors')
    show_circle = st.checkbox("Show communication radius", True, key='show_circle')
    show_eaves = st.checkbox("Show eavesdroppers", True, key='show_eaves')
    show_responders = st.checkbox("Highlight responders within R", True, key='show_responders')

# Fetch possibly-updated values from session_state (Quick Controls take precedence)
seed = st.session_state.get('seed', seed)
W = st.session_state.get('W', W)
H = st.session_state.get('H', H)
num_users = st.session_state.get('num_users_quick', st.session_state.get('num_users', num_users))
num_eaves = st.session_state.get('num_eaves_quick', st.session_state.get('num_eaves', num_eaves))
R = st.session_state.get('R_quick', st.session_state.get('R', R))
K = st.session_state.get('K_quick', st.session_state.get('K', K))
N = st.session_state.get('N_quick', st.session_state.get('N', N))
gamma = st.session_state.get('gamma_quick', st.session_state.get('gamma', gamma))

# Randomly place users & eaves within [0,W]x[0,H]
users = rng.uniform([0, 0], [W, H], size=(num_users, 2))
eaves = rng.uniform([0, 0], [W, H], size=(num_eaves, 2))

# Choose head index
if head_mode == "Nearest to center":
    center = np.array([W / 2, H / 2])
    head_idx = int(np.argmin(np.linalg.norm(users - center, axis=1)))
else:
    head_idx = int(rng.integers(0, num_users))

# Run CKA once
results = run_cka(users, eaves, head_idx, K, N, gamma, R, W, H, rng)

# Build the main plot

fig = go.Figure()

# Sector boundaries
if show_sectors:
    head = results["head"]
    for i in range(N):
        ang = (2 * math.pi / N) * i
        x2 = head[0] + R * math.cos(ang)
        y2 = head[1] + R * math.sin(ang)
        fig.add_trace(go.Scatter(
            x=[head[0], x2], y=[head[1], y2],
            mode="lines",
            line=dict(width=1, dash="dot"),
            showlegend=False,
            hoverinfo="skip"
        ))

# Communication circle
if show_circle:
    cx, cy = circle_points(results["head"][0], results["head"][1], R, steps=256)
    fig.add_trace(go.Scatter(
        x=cx, y=cy,
        mode="lines",
        line=dict(width=1),
        name="Comm Radius R"
    ))

# All users
fig.add_trace(go.Scatter(
    x=users[:,0], y=users[:,1],
    mode="markers",
    marker=dict(size=7, symbol="circle"),
    name="Legal Users"
))

# Eavesdroppers
if show_eaves and len(eaves) > 0:
    fig.add_trace(go.Scatter(
        x=eaves[:,0], y=eaves[:,1],
        mode="markers",
        marker=dict(size=9, symbol="x"),
        name="Eavesdroppers"
    ))

# Head node
fig.add_trace(go.Scatter(
    x=[results["head"][0]], y=[results["head"][1]],
    mode="markers",
    marker=dict(size=14, symbol="star"),
    name="Head Node"
))

# Responders within R (optional highlight)
if show_responders:
    if len(results["user_resp"]) > 0:
        fig.add_trace(go.Scatter(
            x=results["user_resp"][:,0], y=results["user_resp"][:,1],
            mode="markers",
            marker=dict(size=9, symbol="circle-open"),
            name="User Responders (≤R)"
        ))
    if len(results["eav_resp"]) > 0 and show_eaves:
        fig.add_trace(go.Scatter(
            x=results["eav_resp"][:,0], y=results["eav_resp"][:,1],
            mode="markers",
            marker=dict(size=9, symbol="x-thin-open"),
            name="Eaves Responders (≤R)"
        ))

# Selected K locations (color by kind: legal_real, eaves_real, virtual)
kinds = results["kinds"]
sel = results["selected_all"]
if len(sel) > 0:
    # Plot virtuals
    mask_virtual = [k == "virtual" for k in kinds]
    if any(mask_virtual):
        pts = sel[mask_virtual]
        fig.add_trace(go.Scatter(
            x=pts[:,0], y=pts[:,1],
            mode="markers",
            marker=dict(size=11, symbol="diamond-open"),
            name="Selected Virtuals"
        ))
    # Plot eaves real
    mask_eav = [k == "eaves_real" for k in kinds]
    if any(mask_eav):
        pts = sel[mask_eav]
        fig.add_trace(go.Scatter(
            x=pts[:,0], y=pts[:,1],
            mode="markers",
            marker=dict(size=11, symbol="x"),
            name="Selected Eaves (Real)"
        ))
    # Plot legal real
    mask_legal = [k == "legal_real" for k in kinds]
    if any(mask_legal):
        pts = sel[mask_legal]
        fig.add_trace(go.Scatter(
            x=pts[:,0], y=pts[:,1],
            mode="markers",
            marker=dict(size=12, symbol="circle"),
            name="Selected Legal (Real)"
        ))

fig.update_layout(
    width=900, height=700,
    xaxis=dict(range=[0, W], title="X"),
    yaxis=dict(range=[0, H], title="Y", scaleanchor="x", scaleratio=1),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
)

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("Spatial View")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Metrics")
    st.metric("K (anonymity degree)", K)
    st.metric("N (clusters)", N)
    st.metric("γ (real-cap weight)", gamma)
    st.metric("Theoretical P(K,N) = round(γK/N)/K", f"{results['P_est']:.3f}")
    st.metric("Data Efficiency (legal_real / K)", f"{results['data_eff']:.3f}")
    st.metric("DK (var of min dists)", f"{results['DK']:.4f}")
    st.metric("AK (β·mean max dist − DK)", f"{results['AK']:.4f}")

st.divider()

# Quick Controls (editable inline) — sync to sidebar keys

def _sync_quick_to_main():
    for key, qkey in [
        ('K', 'K_quick'),
        ('N', 'N_quick'),
        ('gamma', 'gamma_quick'),
        ('num_users', 'num_users_quick'),
        ('num_eaves', 'num_eaves_quick'),
        ('R', 'R_quick'),
    ]:
        if qkey in st.session_state:
            st.session_state[key] = st.session_state[qkey]
    st.rerun()

with st.container():
    st.subheader("⚡ Quick Controls")
    qc1, qc2, qc3 = st.columns(3)
    with qc1:
        st.slider("K (anonymity)", 3, 64, st.session_state.get('K', 12), 1, key='K_quick')
        st.slider("N (clusters)", 2, 24, st.session_state.get('N', 6), 1, key='N_quick')
    with qc2:
        st.slider("γ (real cap)", 0.1, float(max(2, st.session_state.get('N', 6))), st.session_state.get('gamma', 1.5), 0.1, key='gamma_quick')
        st.slider("Users", 10, 1000, st.session_state.get('num_users', 120), 10, key='num_users_quick')
    with qc3:
        st.slider("Eavesdroppers", 0, 200, st.session_state.get('num_eaves', 12), 1, key='num_eaves_quick')
        st.slider("Radius R", 10, int(min(st.session_state.get('W', 100), st.session_state.get('H', 100))), st.session_state.get('R', int(min(st.session_state.get('W', 100), st.session_state.get('H', 100))*0.35)), 1, key='R_quick')

st.caption("Tip: Use the **Quick Controls** above (or the sidebar) to change K, N, γ, users/eaves, and R in real time.")




with st.expander("What is happening here?"):
    st.markdown("""
- **Goal**: Protect user **location privacy** while still enabling **Location‑Based Services (LBS)** at the edge.
- **CKA idea**: Split the region around the **head node** into **N zones** (clusters). For each request of size **K**, include
  a **balanced mix** of real and **virtual (dummy)** locations, so an eavesdropper cannot easily infer the true location.
- **Real cap**: The total number of real locations placed into the K‑set is limited to **round(γ·K/N)**.
- **Threshold**: Each zone targets **⌊K/N⌋** entries; when a zone has too few real responders, we **synthesize virtuals**.
- **Why it helps**:
  - Lowers **eavesdropping success probability** vs. traditional K‑anonymity (real‑only).
  - **Always succeeds** in forming a K‑sized request even in sparse settings (thanks to virtuals).
  - More **stable** against changes in user/eaves density, and resists **narrow‑region attacks** by spreading points across zones.
- **Metrics**:
  - **P(K,N)** (theoretical): `round(γK/N)/K` — upper bound on attacker guessing the real location in the K‑set.
  - **Data Efficiency**: fraction of selected **legal real** points among K (lower → more privacy; higher → more efficient data use).
  - **DK**: variance of each point’s minimum distance to others (lower → more uniform spread).
  - **AK**: `β·mean(max distance) − DK` — a combined score to capture both spread and uniformity.
""")

st.info("Tip: Change **K**, **N**, **γ**, the counts of users/eaves, and **R** to see how privacy–efficiency trade‑offs behave.")


# Footer note with quick instructions
st.caption("Run: `streamlit run app.py`  • Requires: streamlit, numpy, plotly, scipy")
