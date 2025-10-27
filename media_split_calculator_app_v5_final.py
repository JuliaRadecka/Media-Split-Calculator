# -*- coding: utf-8 -*-
# media_split_calculator_app_v4_9.py

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# ==========================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================

def _parse_opt_float(txt: str):
    txt = (txt or "").strip().replace(",", ".")
    if not txt:
        return None
    try:
        return float(txt)
    except Exception:
        return None

def _apply_platform_bounds(df: pd.DataFrame, platform_bounds: dict) -> pd.DataFrame:
    df = df.copy()
    if "placement" not in df.columns:
        return df
    if "minimum spend" not in df.columns:
        df["minimum spend"] = 0.0
    if "maximum spend" not in df.columns:
        df["maximum spend"] = 1e9
    for key, mm in platform_bounds.items():
        if not mm:
            continue
        mn = mm.get("min")
        mx = mm.get("max")
        if mn is None and mx is None:
            continue
        mask = df["placement"].astype(str).str.lower().str.contains(key)
        if mn is not None:
            df.loc[mask, "minimum spend"] = mn
        if mx is not None:
            df.loc[mask, "maximum spend"] = mx
    return df

def _filter_by_blacklist(df: pd.DataFrame, blacklist: list) -> pd.DataFrame:
    if not blacklist or "placement" not in df.columns:
        return df
    return df[~df["placement"].isin(blacklist)].copy()

def _filter_by_categories(df: pd.DataFrame, ordered_categories: list, other_label: str = "other"):
    if not ordered_categories or "category" not in df.columns:
        return df, ordered_categories
    allowed = [c.lower() for c in ordered_categories if c and isinstance(c, str)]
    if not allowed:
        return df, ordered_categories
    cat_series = df["category"].astype(str).str.lower()
    mask = cat_series.isin([c for c in allowed if c != other_label]) | (cat_series == other_label)
    return df[mask].copy(), ordered_categories

def allocate_budget(df: pd.DataFrame, total_budget: float = 240.0, alpha: float = 1.6,
                    beta: float = 1.0, other_share: float = 10.0):
    if df.empty:
        return df.copy(), pd.DataFrame(), 0.0
    work = df.copy()
    for col, default in [
        ("commercial priority", 0.25),
        ("category priority", 5.0),
        ("placement priority", 5.0),
        ("minimum spend", 0.0),
        ("maximum spend", 1e9),
    ]:
        work[col] = pd.to_numeric(work.get(col, default), errors="coerce").fillna(default)
    other_mask = work.get("category", "").astype(str).str.lower() == "other"
    other_budget = float(total_budget) * (float(other_share) / 100.0)
    main_budget = float(total_budget) - other_budget
    df_main = work[~other_mask].copy()
    df_other = work[other_mask].copy()
    if df_main.empty:
        if not df_other.empty:
            df_other["recommended budget"] = other_budget / len(df_other)
        final = pd.concat([df_other], ignore_index=True)
        summary = final.groupby("category", as_index=False)["recommended budget"].sum()
        summary["share_%"] = (summary["recommended budget"] / float(total_budget)) * 100.0
        return final, summary, 0.0
    df_main["W"] = (df_main["commercial priority"] ** float(alpha)) * ((1.0 / df_main["placement priority"]) ** float(beta))
    df_main["recommended budget"] = df_main["minimum spend"]
    remaining = main_budget - df_main["recommended budget"].sum()
    if remaining < -1e-9:
        st.error("Минимальные пороги превышают доступный бюджет основного пула (без OTHER). Урежьте min.")
        remaining = 0.0
    for _ in range(200):
        if remaining <= 1e-6:
            break
        df_main["available"] = df_main["maximum spend"] - df_main["recommended budget"]
        eligible = df_main["available"] > 1e-12
        total_w = df_main.loc[eligible, "W"].sum()
        if total_w <= 1e-12:
            break
        inc = (df_main.loc[eligible, "W"] / total_w) * remaining
        inc = np.minimum(inc, df_main.loc[eligible, "available"])
        df_main.loc[eligible, "recommended budget"] += inc
        remaining = main_budget - df_main["recommended budget"].sum()
    main_sum = df_main["recommended budget"].sum()
    if main_sum > 0:
        df_main["recommended budget"] = (df_main["recommended budget"] / main_sum) * main_budget
    if not df_other.empty:
        df_other["recommended budget"] = other_budget / len(df_other)
    final = pd.concat([df_main, df_other], ignore_index=True)
    summary = final.groupby("category", as_index=False)["recommended budget"].sum()
    summary["share_%"] = (summary["recommended budget"] / float(total_budget)) * 100.0
    df_valid = final[final["recommended budget"].fillna(0) > 0].copy()
    if not df_valid.empty:
        df_valid["contribution"] = df_valid["recommended budget"] * df_valid["commercial priority"]
        total_margin = (df_valid["contribution"].sum() / df_valid["recommended budget"].sum()) * 100.0
    else:
        total_margin = 0.0
    return final, summary, float(total_margin)

st.set_page_config(page_title="Media Split Calculator v5", layout="wide")
st.title("📊 Media Split Calculator — Fixed Bounds (v5)")

DATA_XLSX_PATH = "калькулятор.xlsx"
df_input = pd.read_excel(DATA_XLSX_PATH)

st.subheader("⚙️ Calculation Parameters")
c1, c2, c3, c4 = st.columns(4)
with c1:
    total_budget = st.number_input("Total Budget (mln ₽)", min_value=0.01, value=240.0, step=1.0)
with c2:
    alpha = st.slider("α — Agency Profit Weight", min_value=0.5, max_value=3.0, value=1.6, step=0.1)
with c3:
    beta = st.slider("β — Client Priority Weight", min_value=0.5, max_value=3.0, value=1.0, step=0.1)
with c4:
    other_share = st.slider("Free Float Share (%)", min_value=0.0, max_value=50.0, value=10.0, step=1.0)

st.markdown("**Platform Budget (mln ₽, min/max) — optional**")
p1, p2, p3, p4 = st.columns(4)
with p1:
    st.caption("Yandex")
    y_min = st.text_input("min (mln ₽)", key="yb_min", label_visibility="collapsed", placeholder="min")
    y_max = st.text_input("max (mln ₽)", key="yb_max", label_visibility="collapsed", placeholder="max")
with p2:
    st.caption("DA")
    da_min = st.text_input("min (mln ₽)", key="da_min", label_visibility="collapsed", placeholder="min")
    da_max = st.text_input("max (mln ₽)", key="da_max", label_visibility="collapsed", placeholder="max")
with p3:
    st.caption("VK")
    vk_min = st.text_input("min (mln ₽)", key="vk_min", label_visibility="collapsed", placeholder="min")
    vk_max = st.text_input("max (mln ₽)", key="vk_max", label_visibility="collapsed", placeholder="max")
with p4:
    st.caption("MTS")
    mts_min = st.text_input("min (mln ₽)", key="mts_min", label_visibility="collapsed", placeholder="min")
    mts_max = st.text_input("max (mln ₽)", key="mts_max", label_visibility="collapsed", placeholder="max")

platform_bounds = {
    "yandex": {"min": _parse_opt_float(y_min), "max": _parse_opt_float(y_max)},
    "da": {"min": _parse_opt_float(da_min), "max": _parse_opt_float(da_max)},
    "vk": {"min": _parse_opt_float(vk_min), "max": _parse_opt_float(vk_max)},
    "mts": {"min": _parse_opt_float(mts_min), "max": _parse_opt_float(mts_max)},
}

st.markdown("**Category Priorities — optional**")
all_categories = ["CTV", "ECOM", "MOB", "OLV PREM", "OLV PRG", "OTHER", "SOCIAL"]
if "cat_order" not in st.session_state:
    st.session_state.cat_order = []
def _toggle_cat(cat_key):
    chosen = st.session_state.get(cat_key, False)
    name = cat_key.replace("cat_", "")
    actual = next((c for c in all_categories if c.replace(" ", "").lower() == name), None)
    if not actual:
        return
    if chosen and actual not in st.session_state.cat_order:
        st.session_state.cat_order.append(actual)
    if (not chosen) and actual in st.session_state.cat_order:
        st.session_state.cat_order.remove(actual)
cat_cols = st.columns(len(all_categories))
for i, cat in enumerate(all_categories):
    key = "cat_" + cat.replace(" ", "").lower()
    cat_cols[i].checkbox(label=cat, key=key, value=(cat in st.session_state.cat_order), on_change=_toggle_cat, args=(key,))
if st.session_state.cat_order:
    st.info("Category order: " + " ➜ ".join([f"{i+1}. {c}" for i, c in enumerate(st.session_state.cat_order)]))
else:
    st.caption("No category priority set (all categories allowed; OTHER handled by Free Float).")

st.markdown("**Placements — Black List (optional)**")
all_placements = sorted(df_input.get("placement", pd.Series([], dtype=str)).astype(str).unique().tolist())
blacklist = st.multiselect("Exclude placements from calculation", options=all_placements, default=[])

st.markdown("---")
# (Остальная часть интерфейса и таблиц идёт аналогично предыдущему коду)
