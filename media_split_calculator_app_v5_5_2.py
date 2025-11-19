
# -*- coding: utf-8 -*-
# Media Split Calculator — v5.5.2
# Changes over v5.4.9:
# 1) Correct margin after Excel upload in Edit:
#    - We enrich the uploaded "Split by Placement" with **commercial priority**
#      merged from the internal catalog (src_df) by (placement, category),
#      then compute margin on those values. If nothing found -> fallback 0.25.
#    - Margin is rendered via a single placeholder so it doesn't duplicate after Save&Recalculate.
# 2) All previous fixes kept (safe Series handling, eye-menu columns, summary-based % in blue banner).

import streamlit as st, pandas as pd, numpy as np
from io import BytesIO
import re, hashlib, ast
from datetime import datetime

APP_VERSION = "v5.5.2"
SCHEMA_VERSION = "2025-11-05.11"

def _norm_text(s: str) -> str:
    if s is None: return ""
    s = str(s).replace("\u00A0"," ").replace("\u2009"," ")
    s = re.sub(r"\s+"," ", s).strip().lower()
    return s.replace("–","-").replace("—","-").replace("−","-")

def make_stable_id(placement: str, category: str) -> np.int64:
    key = f"{_norm_text(placement)}|{_norm_text(category)}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return np.int64(int(h, 16))



PLATFORM_CANON = {
    'yandex': 'Yandex',
    'vk': 'VK',
    'da': 'DA',
    'mts': 'MTS',
}

def classify_platform(placement: str) -> str:
    """
    Determine platform (instrument) by placement name.
    """
    s = _norm_text(placement)
    if not s:
        return ""

    # Yandex
    if "yandex" in s or "яндекс" in s:
        return "Yandex"

    # VK
    if "vk" in s or "vkontakte" in s or "вконтакт" in s:
        return "VK"

    # Digital Alliance
    if "digital alliance" in s or "digialliance" in s or " da " in s:
        return "DA"

    # MTS
    if "mts" in s or "мтс" in s:
        return "MTS"

    # Fallback: use placement itself
    return str(placement).strip()


def add_platform_key(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure presence of platform_key column on a DataFrame.
    platform_key = high-level platform/instrument (Yandex, VK, DA, MTS, etc.).
    """
    df = df.copy()
    if 'placement' not in df.columns:
        return df
    if 'platform_key' not in df.columns:
        df['platform_key'] = df['placement'].astype(str).apply(classify_platform)
    else:
        mask = df['platform_key'].isna() | (df['platform_key'].astype(str).str.strip() == "")
        df.loc[mask, 'platform_key'] = df.loc[mask, 'placement'].astype(str).apply(classify_platform)
    return df


def ensure_mode():
    st.session_state.setdefault('mode','filters')
    st.session_state.setdefault('edit_source', None)
    st.session_state.setdefault('_pending_recalc', False)
    st.session_state.setdefault('cat_order', [])
    st.session_state.setdefault('show_optional_params', False)
    # cat_level1/2/3 and bl_selected will be created lazily by widgets
    # to avoid Streamlit warnings about default values.
    st.session_state.setdefault('total_budget_cache', 240.0)
    st.session_state.setdefault('alpha_cache', 1.6)
    st.session_state.setdefault('beta_cache', 1.0)
    st.session_state.setdefault('other_share_cache', 10.0)
    for key in ['y_min','y_max','da_min','da_max','vk_min','vk_max','mts_min','mts_max']:
        st.session_state.setdefault(key, 0.0)
    st.session_state.setdefault('platform_bounds', {
        'yandex': {'min': 0.0, 'max': 0.0},
        'da': {'min': 0.0, 'max': 0.0},
        'vk': {'min': 0.0, 'max': 0.0},
        'mts': {'min': 0.0, 'max': 0.0},
    })
    st.session_state.setdefault('last_summary_message', None)
    st.session_state.setdefault('last_margin_message', None)
    st.session_state.setdefault('edit_banner_text', None)
    st.session_state.setdefault('edit_banner_type', 'info')
    st.session_state.setdefault('edited_df', None)
    st.session_state.setdefault('calculated_df', None)
    st.session_state.setdefault('scenario_total_budget', None)

def ensure_stable_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if '__id' not in df.columns or df['__id'].isna().any():
        df['__id'] = [make_stable_id(p,c) for p,c in zip(df['placement'].astype(str), df['category'].astype(str))]
    return df

def seriesize(obj, length_hint=0, dtype='float64'):
    if isinstance(obj, pd.Series): return obj
    if obj is None: return pd.Series([np.nan]*length_hint, dtype=dtype)
    try:
        if hasattr(obj, '__len__') and not np.isscalar(obj):
            return pd.Series(obj, dtype=dtype)
    except Exception:
        pass
    return pd.Series([obj], dtype=dtype)

def _series_or_default(df, col, default):
    if col in df.columns:
        s = pd.to_numeric(seriesize(df[col], len(df), 'float64'), errors='coerce')
    else:
        s = pd.Series([np.nan]*len(df), dtype='float64')
    return s.fillna(default)

# --- Category UI groups and priorities ---

UI_GROUPS = {
    "CTV": ["CTV"],
    "OLV PREM": ["OLV PREM"],
    "PRG": ["PRG"],
    "SOCIAL+TG": ["SOCIAL", "TG"],
    "ECOM": ["ECOM"],
    "MOB": ["MOB", "CPA"],
    "Geoservices": ["Geomedia", "Geoperfom"],
    "Promopages": ["Promopages"],
    "Bloggers": ["Bloggers"],
    "Direct": ["Direct", "РСЯ"],
}

CATEGORY_ALIAS = {
    "soc": "social",
    "smm": "social",
    "social media": "social",
    "socialmedia": "social",
    "social-media": "social",
    "соц": "social",
    "социальные сети": "social",
    "tg": "tg",
    "telegram": "tg",
    "telegram ads": "tg",
    "telegram-ads": "tg",
    "телеграм": "tg",
    "тг": "tg",
}

CARD_STYLE = (
    "background-color: rgba(255,255,255,0.05); padding: 24px 28px; "
    "border-radius: 12px; border: 1px solid rgba(255,255,255,0.08); "
    "box-shadow: 0 8px 24px rgba(0,0,0,0.35); width: 100%;"
)


def render_card(title, body_fn):
    """Draw a card with shared style and render body_fn inside."""
    card = st.container()
    card.markdown("<div class='msc-card'>", unsafe_allow_html=True)
    if title:
        card.markdown(f"<h3 class='msc-card__title'>{title}</h3>", unsafe_allow_html=True)
    inner = card.container()
    with inner:
        body_fn()
    card.markdown("</div>", unsafe_allow_html=True)


def toggle_optional_panel():
    st.session_state['show_optional_params'] = not st.session_state.get('show_optional_params', False)


def normalize_category_key(value) -> str:
    key = _norm_text(value)
    if not key:
        return key
    if key in CATEGORY_ALIAS:
        return CATEGORY_ALIAS[key]
    if "social" in key:
        return "social"
    if "tg" == key or key.startswith("tg ") or "telegram" in key:
        return "tg"
    return key


def format_summary_banner(total_loaded, total_budget, other_share):
    total_budget = float(total_budget) if total_budget is not None else 0.0
    pct = (total_loaded / total_budget * 100.0) if total_budget else 100.0
    other_share = float(other_share)
    return (
        f"✅ Бюджет успешно распределён: {total_loaded:.2f} млн ₽ "
        f"({pct:.0f}%), Free Float Share ({other_share:.0f}%)"
    )


def format_margin_message(total_margin):
    return f"### 💰 Общая маржинальность сплита: **{float(total_margin):.2f}%**"


def remember_status_banners(total_loaded, total_budget, other_share, total_margin):
    st.session_state['last_summary_message'] = format_summary_banner(total_loaded, total_budget, other_share)
    st.session_state['last_margin_message'] = format_margin_message(total_margin)


def get_selected_ui_groups():
    """
    Возвращает список выбранных UI-групп категорий с учётом трёх уровней.
    Порядок: Level 1 → Level 2 → Level 3, без дублей.
    Если уровни пусты (старые сценарии), используется cat_order.
    """
    lvl1 = st.session_state.get("cat_level1", []) or []
    lvl2 = st.session_state.get("cat_level2", []) or []
    lvl3 = st.session_state.get("cat_level3", []) or []
    combined = lvl1 + lvl2 + lvl3
    seen = set()
    ordered = []
    for g in combined:
        if g not in seen:
            seen.add(g)
            ordered.append(g)
    if ordered:
        return ordered
    # fallback для старых версий
    return st.session_state.get("cat_order", []) or []


def filter_by_categories(df, picked_ui_groups):
    """
    Фильтрация по выбранным категориям:
    - Если заданы уровни (cat_level1/2/3), используем их.
    - Иначе (legacy) используем cat_order.
    Всегда добавляем category == 'other'.
    Возвращает:
      df2, order_map (для сортировки в do_recalc)
    """
    cats_lower = df["category"].astype(str).map(normalize_category_key)

    # 1) Новый режим: три уровня приоритетов
    lvl1 = st.session_state.get("cat_level1", []) or []
    lvl2 = st.session_state.get("cat_level2", []) or []
    lvl3 = st.session_state.get("cat_level3", []) or []
    has_levels = bool(lvl1 or lvl2 or lvl3)

    if has_levels:
        level_map = {}  # real_category(lower) -> level (1,2,3)
        for level, groups in [(1, lvl1), (2, lvl2), (3, lvl3)]:
            for group in groups:
                for real in UI_GROUPS.get(group, [group]):
                    norm_real = normalize_category_key(real)
                    level_map[norm_real] = level

        if not level_map:
            return df.copy(), None

        keep_cats = set(level_map.keys())
        mask = cats_lower.isin(keep_cats) | (cats_lower == "other")
        df2 = df[mask].copy()

        order_map = {cat: lvl for cat, lvl in level_map.items()}
        # 'other' всегда в конец
        order_map["other"] = 4
        return df2, order_map

    # 2) Legacy-режим: используем cat_order (поддержка старых экспортов)
    picked = st.session_state.get("cat_order", []) or picked_ui_groups or []
    if not picked:
        return df.copy(), None

    picked_lower = [normalize_category_key(c) for c in picked]
    mask = cats_lower.isin(picked_lower) | (cats_lower == "other")
    df2 = df[mask].copy()
    order_map = {normalize_category_key(c): i for i, c in enumerate(picked, start=1)}
    order_map["other"] = len(picked) + 1
    return df2, order_map

def apply_blacklist(df, blacklist):
    """Blacklist now works by platform_key (instrument), not raw placement rows."""
    if not blacklist:
        return df
    df = add_platform_key(df)
    return df[~df['platform_key'].isin(blacklist)].copy()

def apply_platform_bounds(df, bounds):
    """Platform Budget (min/max) now operates on platform groups (platform_key).
    Group min/max are split evenly across rows of the platform so that
    allocate_budget can respect them as per-row min/max.
    """
    df = add_platform_key(df)
    df = df.copy()
    if 'minimum spend' not in df:
        df['minimum spend'] = 0.0
    if 'maximum spend' not in df:
        df['maximum spend'] = 1e9

    for key, mm in (bounds or {}).items():
        if not isinstance(mm, dict):
            continue
        mn = float(mm.get('min', 0) or 0.0)
        mx = float(mm.get('max', 0) or 0.0)
        if mn <= 0 and mx <= 0:
            continue

        key_str = str(key).lower()
        canon = PLATFORM_CANON.get(key_str, None)

        pk = df['platform_key'].astype(str).str.lower()
        if canon is not None:
            mask = pk == canon.lower()
        else:
            mask = pk.str.contains(key_str)

        n_rows = int(mask.sum())
        if n_rows == 0:
            continue

        if mn > 0:
            per_min = mn / n_rows
            current_min = pd.to_numeric(df.loc[mask, 'minimum spend'], errors='coerce').fillna(0.0)
            df.loc[mask, 'minimum spend'] = np.maximum(current_min, per_min)
        if mx > 0:
            per_max = mx / n_rows
            current_max = pd.to_numeric(df.loc[mask, 'maximum spend'], errors='coerce').fillna(1e9)
            df.loc[mask, 'maximum spend'] = np.minimum(current_max, per_max)

    return df

def _ensure_other_summary(summary_df, other_budget):
    try:
        if summary_df is None or summary_df.empty:
            if other_budget is None:
                return summary_df
            return pd.DataFrame([{'category': 'other', 'recommended budget': float(other_budget)}])

        summary_df = summary_df.copy()
        cats = summary_df['category'].astype(str).map(normalize_category_key)
        mask = cats == 'other'
        if mask.any():
            total_other = pd.to_numeric(
                seriesize(summary_df.loc[mask, 'recommended budget'], mask.sum()), errors='coerce'
            ).fillna(0).sum()
            merged = pd.DataFrame([
                {'category': 'other', 'recommended budget': float(total_other)}
            ])
            remainder = summary_df.loc[~mask].copy()
            summary_df = pd.concat([remainder, merged], ignore_index=True)
        elif other_budget is not None:
            extra = pd.DataFrame([{'category': 'other', 'recommended budget': float(other_budget)}])
            summary_df = pd.concat([summary_df, extra], ignore_index=True)
        return summary_df
    except Exception:
        return summary_df

def allocate_budget(df, total_budget=240.0, alpha=1.6, beta=1.0, other_share=10.0, use_gates=None):
    meta = {'scaled_mins': False, 'scale_coef': 1.0}
    df = df.copy()

    df['commercial priority'] = _series_or_default(df, 'commercial priority', 0.25)
    df['category priority']   = _series_or_default(df, 'category priority',   5.0)
    df['placement priority']  = _series_or_default(df, 'placement priority',  5.0)
    df['minimum spend']       = _series_or_default(df, 'minimum spend',       0.0)
    df['maximum spend']       = _series_or_default(df, 'maximum spend',       1e9)

    other_mask   = df['category'].astype(str).map(normalize_category_key)=='other'
    other_budget = float(total_budget)*(float(other_share)/100.0)
    main_budget  = float(total_budget)-other_budget

    use_gates = bool((df['category priority'].notna().any() or df['placement priority'].notna().any())) if use_gates is None else use_gates
    if use_gates:
        cat_gate = df['category priority']<=3
        free_gate = df['category priority']>=4
        gate_mask = ((cat_gate & (df['placement priority']<=2)) | free_gate)
    else:
        gate_mask = True

    df_main = df[gate_mask & (~other_mask)].copy()
    if df_main.empty:
        empty_df = df.assign(**{'recommended budget': np.nan, 'W': np.nan, 'available': np.nan})
        empty_s  = pd.DataFrame({'category': pd.Series(dtype='object'),'recommended budget': pd.Series(dtype='float'),'share_%': pd.Series(dtype='float')})
        return empty_df, empty_s, 0.0, meta

    df_main['W'] = (df_main['commercial priority']**float(alpha)) * ((1.0/df_main['placement priority'])**float(beta))
    df_main['recommended budget'] = df_main['minimum spend']

    remaining = main_budget - df_main['recommended budget'].sum()
    if remaining < -1e-9:
        total_min = max(df_main['recommended budget'].sum(), 1e-9)
        scale = max(main_budget, 0.0) / total_min
        df_main['recommended budget'] *= scale
        meta['scaled_mins'] = True; meta['scale_coef'] = float(scale); remaining = 0.0

    for _ in range(160):
        if remaining <= 1e-9: break
        df_main['available'] = df_main['maximum spend'] - df_main['recommended budget']
        elig = df_main['available'] > 0
        total_w = df_main.loc[elig,'W'].sum()
        if total_w <= 0: break
        inc = (df_main.loc[elig,'W']/total_w) * remaining
        inc = np.minimum(inc, df_main.loc[elig,'available'])
        df_main.loc[elig,'recommended budget'] += inc
        remaining = main_budget - df_main['recommended budget'].sum()

    sum_main = df_main['recommended budget'].sum()
    if sum_main > 0:
        df_main['recommended budget'] = (df_main['recommended budget']/sum_main)*main_budget

    df_other = df[other_mask].copy()
    if not df_other.empty:
        df_other['recommended budget'] = other_budget/len(df_other)

    df_rest = df[~df.index.isin(df_main.index) & ~df.index.isin(df_other.index)].copy()
    df_rest['recommended budget'] = np.nan

    df_final = pd.concat([df_main, df_other, df_rest], ignore_index=True)

    summary = df_final.groupby('category', as_index=False)['recommended budget'].sum()
    summary = _ensure_other_summary(summary, other_budget)
    summary['share_%'] = (summary['recommended budget']/float(total_budget)*100.0) if float(total_budget)>0 else 0.0

    df_valid = df_final[df_final['recommended budget'].fillna(0)>0].copy()
    if df_valid.empty: total_margin = 0.0
    else:
        df_valid['contribution'] = df_valid['recommended budget']*df_valid['commercial priority']
        total_margin = (df_valid['contribution'].sum()/df_valid['recommended budget'].sum())*100.0

    st.session_state['last_meta_scaled_mins'] = meta['scaled_mins']
    st.session_state['last_meta_scale_coef'] = meta['scale_coef']
    return df_final, summary, float(total_margin), meta


def allocate_with_manual_overrides(df, locked_budgets, total_budget, alpha, beta, other_share):
    """Allocate budget while respecting manual overrides for specific placements."""
    locked_budgets = {
        np.int64(k): float(v)
        for k, v in (locked_budgets or {}).items()
        if v is not None and not pd.isna(v)
    }
    if not locked_budgets:
        return allocate_budget(df, total_budget, alpha, beta, other_share, use_gates=None)

    df = ensure_stable_ids(df)
    order_map = {rid: idx for idx, rid in enumerate(df['__id'])}
    mask_locked = df['__id'].isin(locked_budgets.keys())
    locked_df = df[mask_locked].copy()
    locked_df['recommended budget'] = locked_df['__id'].map(locked_budgets)
    for aux in ['W', 'available']:
        if aux not in locked_df.columns:
            locked_df[aux] = np.nan

    remaining_df = df[~mask_locked].copy()
    total_budget = float(total_budget)
    other_share = float(other_share)

    locked_total = float(
        pd.to_numeric(seriesize(locked_df.get('recommended budget'), len(locked_df)), errors='coerce').fillna(0).sum()
    )
    target_other_amount = total_budget * (other_share / 100.0)
    locked_other_amount = 0.0
    if not locked_df.empty and 'category' in locked_df.columns:
        cat_norm = locked_df['category'].astype(str).map(normalize_category_key)
        locked_other_amount = float(
            pd.to_numeric(
                seriesize(locked_df.loc[cat_norm == 'other', 'recommended budget'], len(locked_df)),
                errors='coerce'
            ).fillna(0).sum()
        )

    effective_total = max(total_budget - locked_total, 0.0)
    remaining_other_amount = max(target_other_amount - locked_other_amount, 0.0)
    if effective_total > 0:
        remaining_other_amount = min(remaining_other_amount, effective_total)
        effective_other_share = (remaining_other_amount / effective_total) * 100.0
    else:
        effective_other_share = 0.0

    df_calc, _, _, meta = allocate_budget(
        remaining_df,
        total_budget=effective_total,
        alpha=alpha,
        beta=beta,
        other_share=effective_other_share,
        use_gates=None,
    )
    meta['locked_budget'] = locked_total
    meta['locked_rows'] = len(locked_budgets)

    df_result = pd.concat([locked_df, df_calc], ignore_index=True, sort=False)
    if '__id' in df_result.columns:
        df_result['_ord'] = df_result['__id'].map(order_map).fillna(1e6)
        df_result = df_result.sort_values('_ord').drop(columns=['_ord'])

    summary = df_result.groupby('category', as_index=False)['recommended budget'].sum()
    actual_other_total = 0.0
    if 'category' in summary.columns:
        mask_other = summary['category'].astype(str).map(normalize_category_key) == 'other'
        actual_other_total = float(summary.loc[mask_other, 'recommended budget'].sum())
    summary = _ensure_other_summary(summary, actual_other_total if actual_other_total > 0 else None)
    if total_budget > 0:
        summary['share_%'] = (summary['recommended budget'] / total_budget) * 100.0
    else:
        summary['share_%'] = 0.0

    df_valid = df_result[df_result['recommended budget'].fillna(0) > 0].copy()
    if df_valid.empty:
        total_margin = 0.0
    else:
        df_valid['contribution'] = df_valid['recommended budget'] * df_valid['commercial priority']
        total_margin = (df_valid['contribution'].sum() / df_valid['recommended budget'].sum()) * 100.0

    return df_result, summary, float(total_margin), meta

def apply_editor_to_base(base_df, edited_df, editable_cols):
    base = base_df.copy()
    patch = edited_df[['__id', *editable_cols]].copy()
    patch = patch.dropna(subset=['__id']).drop_duplicates(subset='__id', keep='last')
    merged = base.merge(patch, on='__id', how='left', suffixes=('', '__edit'))
    for col in editable_cols:
        e = col+'__edit'
        if e in merged:
            merged[col] = merged[e].combine_first(merged[col]); merged.drop(columns=[e], inplace=True)
    return merged


def apply_category_priorities(df, picked_ui_groups):
    """
    Присваиваем числовой приоритет категориям в зависимости от уровня:
    - Priority 1 → 1
    - Priority 2 → 2
    - Priority 3 → 3
    Если уровни не заданы (legacy), используем порядок из cat_order / picked_ui_groups.
    Категории без явного уровня получают priority = 4 (как в v5.5).
    """
    lvl1 = st.session_state.get("cat_level1", []) or []
    lvl2 = st.session_state.get("cat_level2", []) or []
    lvl3 = st.session_state.get("cat_level3", []) or []
    has_levels = bool(lvl1 or lvl2 or lvl3)

    df = df.copy()
    mapping = {}

    if has_levels:
        # Чем меньше число — тем выше приоритет. При дубликатах берём самый высокий уровень.
        for level, groups in [(1, lvl1), (2, lvl2), (3, lvl3)]:
            for group in groups:
                for real in UI_GROUPS.get(group, [group]):
                    key = normalize_category_key(real)
                    mapping[key] = min(mapping.get(key, level), level)
    else:
        # Legacy: используем cat_order / picked_ui_groups как линейный порядок
        order = st.session_state.get("cat_order", []) or picked_ui_groups or []
        if order:
            mapping = {normalize_category_key(c): i for i, c in enumerate(order, start=1)}

    mapped = df["category"].astype(str).map(normalize_category_key).map(mapping)
    if "category priority" in df.columns:
        fallback = pd.to_numeric(
            seriesize(df["category priority"], len(df)), errors="coerce"
        )
        mapped = mapped.where(~mapped.isna(), fallback)
    # Не выбранные категории → приоритет 4 (как в v5.5, вне гейтов 1–3)
    mapped = seriesize(mapped, len(df)).fillna(4.0)
    df["category priority"] = mapped
    return df

def export_csv(df):
    return df.drop(columns=[c for c in ['commercial priority','W','available','__id'] if c in df.columns], errors='ignore').to_csv(index=False).encode('utf-8')

def export_excel(df_split, df_sum, base_df_with_ids):
    desired = ['placement','category','category priority','placement priority','minimum spend','maximum spend','recommended budget']
    split_df = df_split[[c for c in desired if c in df_split.columns]].copy()

    # Summary: переименование 'other' -> 'FREE FLOAT'
    df_sum_x = df_sum.copy()
    if not df_sum_x.empty and 'category' in df_sum_x.columns:
        free_label = 'FREE FLOAT'
        df_sum_x['category'] = df_sum_x['category'].replace(
            {'other': free_label, 'Other': free_label, 'OTHER': free_label}
        )

    bio = BytesIO()
    with pd.ExcelWriter(bio, engine='openpyxl') as w:
        split_df.to_excel(w, index=False, sheet_name='Split by Placement')
        df_sum_x.to_excel(w, index=False, sheet_name='Summary by Category')
        meta = pd.DataFrame([{
            'schema_version': SCHEMA_VERSION,
            'app_version': APP_VERSION,
            'exported_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_budget': float(
                st.session_state.get(
                    'scenario_total_budget',
                    st.session_state.get('total_budget_cache', 240.0)
                )
            ),
            'alpha': float(st.session_state.get('alpha_cache', 1.6)),
            'beta': float(st.session_state.get('beta_cache', 1.0)),
            'free_float_%': float(st.session_state.get('other_share_cache', 10.0)),
            'selected_categories': ",".join(get_selected_ui_groups()),
        }])
        meta.to_excel(w, index=False, sheet_name='_meta')
        start = len(meta)+2
        id_map = base_df_with_ids[['__id','placement','category']].copy()
        id_map.to_excel(w, index=False, sheet_name='_meta', startrow=start)
    bio.seek(0)
    return bio.getvalue()

def margin_from_current_budgets(df) -> float:
    df = df.copy()
    if 'recommended budget' not in df: return 0.0
    rb = pd.to_numeric(seriesize(df['recommended budget'], len(df)), errors='coerce').fillna(0)
    cp = pd.to_numeric(seriesize(df.get('commercial priority', 0.25), len(df)), errors='coerce').fillna(0.25)
    mix = pd.DataFrame({'rb': rb, 'cp': cp})
    mix = mix[mix['rb'] > 0]
    if mix.empty: return 0.0
    return float((mix['rb']*mix['cp']).sum() / mix['rb'].sum() * 100.0)

# ---- App ----
st.set_page_config(page_title='📊 Media Split Calculator v5.5.2', layout='wide')
st.title('📊 Media Split Calculator — v5.5.2')
st.markdown(
    f"""
    <style>
    .msc-card {{{CARD_STYLE} margin-bottom: 28px;}}
    .msc-card__title {{
        margin-top: 0;
        margin-bottom: 1.25rem;
        font-weight: 600;
    }}
    .msc-banner {{
        margin-bottom: 0.75rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)
FILE_PATH = 'калькулятор.xlsx'
try:
    src_df = pd.read_excel(FILE_PATH)
    src_df = add_platform_key(src_df)
except Exception as e:
    st.error(f'Не удалось прочитать исходный файл {FILE_PATH}: {e}'); st.stop()

# Prepare CP lookup from catalog for later merge on upload
src_cp = src_df[['placement','category']].copy()
src_cp['commercial priority'] = pd.to_numeric(seriesize(src_df.get('commercial priority', 0.25), len(src_df)), errors='coerce').fillna(0.25)
src_cp['_plc'] = src_cp['placement'].astype(str).map(_norm_text)
src_cp['_cat'] = src_cp['category'].astype(str).map(_norm_text)
src_cp = src_cp.drop_duplicates(subset=['_plc','_cat'], keep='last')

ensure_mode()


def mark_for_recalc():
    """Set recalc flag only in result mode (for mode='result')."""
    if st.session_state.get('mode') == 'result':
        st.session_state._pending_recalc = True


def do_recalc():
    df0 = src_df.copy()
    df0 = apply_blacklist(df0, st.session_state.get('bl_selected', []))
    df0 = apply_platform_bounds(df0, st.session_state.get('platform_bounds', {}))
    picked = get_selected_ui_groups()
    df0, order_map = filter_by_categories(df0, picked)
    df0 = apply_category_priorities(df0, picked)
    df0 = ensure_stable_ids(df0)
    df_res, summary, total_margin, meta = allocate_budget(
        df0,
        total_budget=float(st.session_state.get('total_budget_cache', 240.0)),
        alpha=float(st.session_state.get('alpha_cache', 1.6)),
        beta=float(st.session_state.get('beta_cache', 1.0)),
        other_share=float(st.session_state.get('other_share_cache', 10.0)),
        use_gates=None
    )
    if order_map and 'category' in df_res:
        df_res['_ord'] = df_res['category'].astype(str).map(normalize_category_key).map(order_map).fillna(1e6)
        df_res = df_res.sort_values(by=['_ord','recommended budget'], ascending=[True, False]).drop(columns=['_ord'])
    st.session_state.df_result = df_res.copy()
    st.session_state.calculated_df = df_res.copy()
    st.session_state.summary = summary.copy()
    st.session_state.total_margin = float(total_margin)
    st.session_state.base_df = ensure_stable_ids(df_res.copy())
    st.session_state._pending_recalc = False
    total_loaded = float(pd.to_numeric(seriesize(df_res.get('recommended budget'), len(df_res)), errors='coerce').fillna(0).sum())
    # фиксируем сценарный бюджет для дальнейшего Edit
    st.session_state['scenario_total_budget'] = total_loaded
    other = float(st.session_state.get('other_share_cache', 10.0))
    remember_status_banners(total_loaded, total_loaded, other, total_margin)


# Controls (filters UI)
if st.session_state.mode != 'edit':
    # --- Card: Calculation Parameters ---
    with st.container(border=True):
        st.markdown("### ⚙️ Calculation Parameters")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.number_input(
                "Total Budget (mln ₽)",
                min_value=0.0,
                step=10.0,
                key="total_budget_cache",
                on_change=mark_for_recalc,
            )
        with c2:
            st.slider(
                "α — Agency Profit Weight",
                min_value=1.0,
                max_value=2.5,
                step=0.1,
                key="alpha_cache",
                on_change=mark_for_recalc,
            )
        with c3:
            st.slider(
                "β — Client Priority Weight",
                min_value=0.5,
                max_value=2.0,
                step=0.1,
                key="beta_cache",
                on_change=mark_for_recalc,
            )
        with c4:
            st.slider(
                "Free Float Share (%)",
                min_value=0.0,
                max_value=30.0,
                step=1.0,
                key="other_share_cache",
                on_change=mark_for_recalc,
            )

        btn_label = "⬆️ Hide Optional" if st.session_state.get("show_optional_params") else "⬇️ Optional"
        st.button(
            btn_label,
            key="toggle_optional_btn",
            on_click=toggle_optional_panel,
        )

    # --- Card: Optional Parameters (by toggle) ---
    if st.session_state.get("show_optional_params", False):
        with st.container(border=True):
            st.markdown("### ⚙️ Optional Parameters")

            st.markdown("**Platform Budget (min/max)**")
            p1, p2, p3, p4 = st.columns(4)
            for label, min_k, max_k, col in [
                ("Yandex", "y_min", "y_max", p1),
                ("DA", "da_min", "da_max", p2),
                ("VK", "vk_min", "vk_max", p3),
                ("MTS", "mts_min", "mts_max", p4),
            ]:
                with col:
                    st.caption(label)
                    st.number_input(
                        "min (mln ₽)",
                        key=min_k,
                        step=10.0,
                        on_change=mark_for_recalc,
                    )
                    st.number_input(
                        "max (mln ₽)",
                        key=max_k,
                        step=10.0,
                        on_change=mark_for_recalc,
                    )

            st.session_state.platform_bounds = {
                "yandex": {
                    "min": st.session_state.get("y_min", 0.0),
                    "max": st.session_state.get("y_max", 0.0),
                },
                "da": {
                    "min": st.session_state.get("da_min", 0.0),
                    "max": st.session_state.get("da_max", 0.0),
                },
                "vk": {
                    "min": st.session_state.get("vk_min", 0.0),
                    "max": st.session_state.get("vk_max", 0.0),
                },
                "mts": {
                    "min": st.session_state.get("mts_min", 0.0),
                    "max": st.session_state.get("mts_max", 0.0),
                },
            }

            st.markdown("**Category Priorities (Priority 1/2/3 — multiselect)**")
            ui_groups = list(UI_GROUPS.keys())
            c_lvl1, c_lvl2, c_lvl3 = st.columns(3)
            with c_lvl1:
                st.multiselect(
                    "Priority 1",
                    options=ui_groups,
                    key="cat_level1",
                    on_change=mark_for_recalc,
                )
            with c_lvl2:
                st.multiselect(
                    "Priority 2",
                    options=ui_groups,
                    key="cat_level2",
                    on_change=mark_for_recalc,
                )
            with c_lvl3:
                st.multiselect(
                    "Priority 3",
                    options=ui_groups,
                    key="cat_level3",
                    on_change=mark_for_recalc,
                )

            st.markdown("**Black List (multiselect)**")
            platform_series = src_df["platform_key"].dropna().astype(str).map(lambda s: s.strip())
            all_plc = sorted(platform_series.unique().tolist())
            st.multiselect(
                "Exclude platforms from calculation",
                options=all_plc,
                key="bl_selected",
                on_change=mark_for_recalc,
            )

if st.session_state.mode == 'filters':
    a,b = st.columns(2)
    with a:
        if st.button('🧮 Calculate'): do_recalc(); st.session_state.mode='result'; st.rerun()
    with b:
        if st.button('📂 Upload a file (.xlsx)'):
            st.session_state['edit_source']='upload'; st.session_state.mode='edit'; st.rerun()

elif st.session_state.mode == 'result':
    if st.session_state.get('_pending_recalc', False):
        do_recalc(); st.rerun()

    df_result = st.session_state.get('df_result'); summary = st.session_state.get('summary')
    total_margin = st.session_state.get('total_margin', 0.0)
    if df_result is None or summary is None:
        st.info('Нет результата расчёта. Сначала выполните Calculate.')
    else:
        total_loaded = float(pd.to_numeric(seriesize(df_result.get('recommended budget'), len(df_result)), errors='coerce').fillna(0).sum())
        scenario_tb = float(st.session_state.get('scenario_total_budget', total_loaded))
        st.session_state['scenario_total_budget'] = scenario_tb
        other = float(st.session_state.get('other_share_cache', 10.0))
        remember_status_banners(total_loaded, scenario_tb, other, total_margin)
        status_msg = st.session_state.get('last_summary_message')
        if status_msg:
            st.success(status_msg)

        st.subheader('📈 Recommended Split by Placement')
        base_cols = ['placement','category','recommended budget']
        all_adv  = [c for c in ['category priority','placement priority','minimum spend','maximum spend'] if c in df_result.columns]
        column_order = base_cols
        table_df = df_result[base_cols + all_adv].copy()
        # Не показываем чисто служебные строки Free Float в UI по плейсментам
        if 'placement' in table_df.columns and 'category' in table_df.columns:
            cat_norm = table_df['category'].astype(str).map(normalize_category_key)
            plc_norm = table_df['placement'].astype(str).str.strip().str.lower()
            mask_ff = cat_norm.eq('other') & plc_norm.isin(['free float','free_float','свободно'])
            table_df = table_df[~mask_ff].copy()
        total_row = {c:'' for c in table_df.columns}
        total_row['placement'] = 'ИТОГО'; total_row['recommended budget']=pd.to_numeric(table_df['recommended budget'], errors='coerce').fillna(0).sum()
        table_df = pd.concat([table_df, pd.DataFrame([total_row])], ignore_index=True)
        st.data_editor(table_df, use_container_width=True, hide_index=True, disabled=True,
                       column_config={'recommended budget': st.column_config.NumberColumn('recommended budget', format='%.6f')},
                       column_order=column_order,
                       key='result_table_v550')

        st.subheader('📊 Summary by Category')
        sum_df = st.session_state.get('summary', pd.DataFrame())
        if not sum_df.empty:
            sum_df = sum_df.copy()
            free_label = 'FREE FLOAT'
            if 'category' in sum_df.columns:
                sum_df['category'] = sum_df['category'].replace(
                    {'other': free_label, 'Other': free_label, 'OTHER': free_label}
                )
            tot = {
                'category': 'ИТОГО',
                'recommended budget': float(sum_df['recommended budget'].sum()),
                'share_%': 100.0,
            }
            sum_df = pd.concat([sum_df, pd.DataFrame([tot])], ignore_index=True)
            st.dataframe(sum_df.round(2), use_container_width=True)

        margin_msg = st.session_state.get('last_margin_message', format_margin_message(total_margin))
        st.markdown(margin_msg)

        ccsv, cxlsx = st.columns(2)
        with ccsv: st.download_button('💾 Download Results (CSV)', data=export_csv(df_result), file_name='split_by_placement.csv', mime='text/csv')
        with cxlsx: st.download_button('💾 Download Results (.xlsx)', data=export_excel(df_result, st.session_state.get('summary', pd.DataFrame()), st.session_state.base_df),
                                       file_name='media_split_results.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        if st.button('✏️ Edit Calculated Table'): st.session_state.mode='edit'; st.rerun()

elif st.session_state.mode == 'edit':
    st.subheader('✏️ Edit Calculated Table')
    uploaded = st.file_uploader('Upload Excel with _meta to restore previous editor state (.xlsx only)', type=['xlsx'])
    src_flag = st.session_state.get('edit_source'); waiting_for_file = (src_flag=='upload' and uploaded is None)

    base_df = st.session_state.get('base_df'); current_df = st.session_state.get('df_result')

    def _restore_cat_order_from_meta(meta_df):
        """Восстанавливаем выбор категорий из _meta.selected_categories.

        Для совместимости:
        - cat_order заполняется списком категорий;
        - cat_level1 получает тот же список (L1), если уровни ещё не заданы.
        """
        try:
            raw_series = seriesize(meta_df.get('selected_categories', pd.Series(dtype='object')))
            raw = raw_series.iloc[0] if len(raw_series) > 0 else None
            if isinstance(raw, str) and raw.strip():
                if raw.strip().startswith('['):
                    lst = ast.literal_eval(raw)
                else:
                    lst = [x.strip() for x in raw.split(',') if x.strip()]
                if lst:
                    st.session_state.cat_order = lst
                    # если уровни ещё не заданы, считаем, что это Level 1
                    if not (st.session_state.get("cat_level1") or st.session_state.get("cat_level2") or st.session_state.get("cat_level3")):
                        st.session_state["cat_level1"] = lst
                    return True
        except Exception:
            pass
        return False

    if uploaded is not None:
        try:
            xls = pd.ExcelFile(uploaded)
            split_df = pd.read_excel(xls, 'Split by Placement')

            # try to read summary for correct "100%" reference
            total_from_summary = None
            if 'Summary by Category' in xls.sheet_names:
                try:
                    sum_df_file = pd.read_excel(xls, 'Summary by Category')
                    s = seriesize(sum_df_file.get('recommended budget', pd.Series(dtype='float64')), len(sum_df_file))
                    total_from_summary = float(pd.to_numeric(s, errors='coerce').fillna(0).sum())
                except Exception:
                    total_from_summary = None

            if '_meta' in xls.sheet_names:
                meta_sheet = pd.read_excel(xls, '_meta'); _restore_cat_order_from_meta(meta_sheet)
                try:
                    id_map = pd.read_excel(xls, '_meta', skiprows=len(meta_sheet)+2)
                    if {'__id','placement','category'}.issubset(id_map.columns):
                        base_df = split_df.merge(id_map[['__id','placement','category']], on=['placement','category'], how='left')
                        base_df = ensure_stable_ids(base_df) if base_df['__id'].isna().any() else base_df
                    else:
                        base_df = ensure_stable_ids(split_df)
                except Exception: base_df = ensure_stable_ids(split_df)
            else:
                base_df = ensure_stable_ids(split_df)

            # --- NEW: enrich with commercial priority from internal catalog for correct margin ---
            base_df['_plc'] = base_df['placement'].astype(str).map(_norm_text)
            base_df['_cat'] = base_df['category'].astype(str).map(_norm_text)
            base_df = base_df.merge(src_cp[['_plc','_cat','commercial priority']], on=['_plc','_cat'], how='left')
            base_df['commercial priority'] = pd.to_numeric(base_df['commercial priority'], errors='coerce').fillna(0.25)
            base_df.drop(columns=['_plc','_cat'], inplace=True)

            # banner numbers
            rb_series = pd.to_numeric(seriesize(base_df.get('recommended budget', pd.Series(dtype='float64')), len(base_df)), errors='coerce').fillna(0)
            total_loaded = float(rb_series.sum())
            other = float(st.session_state.get('other_share_cache', 10.0))

            # фиксируем сценарный бюджет по загруженному файлу
            st.session_state['scenario_total_budget'] = total_loaded
            st.session_state['total_budget_cache'] = total_loaded

            st.session_state['edit_banner_text'] = (
                f"📥 Бюджет успешно подгружен: {total_loaded:.2f} млн ₽ (100%), Free Float Share ({other:.0f}%)"
            )
            st.session_state['edit_banner_type'] = 'info'

            # correct margin from uploaded file (with CP merged from catalog)
            st.session_state['last_margin_message'] = format_margin_message(margin_from_current_budgets(base_df))

            base_df = apply_category_priorities(base_df, get_selected_ui_groups())
            st.session_state.base_df = base_df.copy()

            # Treat uploaded file as current calculated result so that
            # downloads work even before a new Save & Recalculate run.
            st.session_state.df_result = base_df.copy()
            st.session_state.calculated_df = base_df.copy()
            st.session_state.total_margin = margin_from_current_budgets(base_df)

            # Try to restore summary-by-category from the uploaded workbook,
            # otherwise build it from the split sheet.
            if total_from_summary is not None and total_from_summary > 0 and 'Summary by Category' in xls.sheet_names:
                try:
                    sum_df_upd = sum_df_file.copy()
                    if 'recommended budget' in sum_df_upd.columns and 'share_%' not in sum_df_upd.columns:
                        sum_df_upd['share_%'] = (pd.to_numeric(sum_df_upd['recommended budget'], errors='coerce').fillna(0) / float(total_from_summary)) * 100.0
                    st.session_state.summary = sum_df_upd
                except Exception:
                    pass
            if st.session_state.get('summary') is None:
                try:
                    if 'category' in base_df.columns and 'recommended budget' in base_df.columns:
                        sum_df_upd = base_df.groupby('category', as_index=False)['recommended budget'].sum()
                        total_rb = float(pd.to_numeric(sum_df_upd['recommended budget'], errors='coerce').fillna(0).sum())
                        if total_rb > 0:
                            sum_df_upd['share_%'] = (sum_df_upd['recommended budget'] / total_rb) * 100.0
                        st.session_state.summary = sum_df_upd
                except Exception:
                    pass
        except Exception as e:
            st.error(f'Не удалось прочитать файл: {e}')

    if waiting_for_file:
        if st.button('⬅️ Back'): st.session_state['edit_source']=None; st.session_state.mode='filters'; st.rerun()
        st.stop()

    if base_df is None and current_df is not None:
        st.session_state.base_df = ensure_stable_ids(apply_category_priorities(current_df.copy(), get_selected_ui_groups()))
        base_df = st.session_state.base_df

    if base_df is None:
        df0 = src_df.copy()
        df0 = apply_blacklist(df0, st.session_state.get('bl_selected', []))
        df0 = apply_platform_bounds(df0, st.session_state.get('platform_bounds', {}))
        picked = get_selected_ui_groups()
        df0, _ = filter_by_categories(df0, picked)
        base_df = ensure_stable_ids(apply_category_priorities(df0, picked))
        st.session_state.base_df = base_df.copy()

    editable_cols = ['category priority','placement priority','minimum spend','maximum spend','recommended budget']
    show_cols = ['__id','placement','category', *editable_cols]
    show_cols = [c for c in show_cols if c in base_df.columns]
    editor_df = base_df[show_cols].copy()

    # Hide synthetic FREE FLOAT / OTHER rows from the manual editor:
    # they are handled via the Free Float Share slider, not per-placement editing.
    if {'placement','category'}.issubset(editor_df.columns):
        cat_norm = editor_df['category'].astype(str).map(normalize_category_key)
        plc_norm = editor_df['placement'].astype(str).str.strip().str.lower()
        ff_mask = cat_norm.eq('other') & (
            plc_norm.isin(['free float','free_float','свободно']) |
            plc_norm.str.contains('free float', na=False) |
            plc_norm.str.contains('free_float', na=False) |
            plc_norm.str.contains('свобод', na=False)
        )
        editor_df = editor_df[~ff_mask].copy()

    if not st.session_state.get('edit_banner_text') and st.session_state.get('last_summary_message'):
        st.session_state['edit_banner_text'] = st.session_state['last_summary_message']
        st.session_state['edit_banner_type'] = 'success'

    banner_text = st.session_state.get('edit_banner_text')
    if banner_text:
        banner_type = st.session_state.get('edit_banner_type', 'success')
        if banner_type == 'info':
            st.info(banner_text)
        elif banner_type == 'warning':
            st.warning(banner_text)
        else:
            st.success(banner_text)

    edited = st.data_editor(editor_df, use_container_width=True, num_rows='fixed',
                            disabled=['__id','placement','category'],
                            column_config={'recommended budget': st.column_config.NumberColumn('recommended budget', format='%.6f')},
                            key='editor_table_v550')
    st.session_state['edited_df'] = edited.copy()

    margin_msg = st.session_state.get('last_margin_message')
    if margin_msg:
        st.markdown(margin_msg)

    c1,c2 = st.columns(2)
    with c1:
        if st.button('🔄 Save & Recalculate'):
            edited_for_calc = st.session_state.get('edited_df')
            if edited_for_calc is None:
                edited_for_calc = edited.copy()
            base_rb = pd.to_numeric(seriesize(editor_df.get('recommended budget'), len(editor_df)), errors='coerce')
            edited_rb = pd.to_numeric(seriesize(edited_for_calc.get('recommended budget'), len(edited_for_calc)), errors='coerce')
            manual_mask = (~edited_rb.isna()) & (base_rb.isna() | (np.abs(edited_rb - base_rb) > 1e-9))
            locked_map = dict(zip(edited_for_calc.loc[manual_mask, '__id'], edited_rb[manual_mask]))

            base_applied = apply_editor_to_base(base_df, edited_for_calc, editable_cols)
            picked = get_selected_ui_groups()
            df_in, _ = filter_by_categories(base_applied, picked)
            # если после фильтрации категорий таблица пустая — берём базу целиком
            if df_in.empty:
                df_in = base_applied.copy()
            df_in = apply_blacklist(df_in, st.session_state.get('bl_selected', []))
            df_in = apply_platform_bounds(df_in, st.session_state.get('platform_bounds', {}))
            df_in = apply_category_priorities(df_in, picked)
            scenario_tb = float(
                st.session_state.get(
                    'scenario_total_budget',
                    st.session_state.get('total_budget_cache', 240.0)
                )
            )
            df_res, summary, total_margin, meta = allocate_with_manual_overrides(
                df_in,
                locked_map,
                total_budget=scenario_tb,
                alpha=float(st.session_state.get('alpha_cache', 1.6)),
                beta=float(st.session_state.get('beta_cache', 1.0)),
                other_share=float(st.session_state.get('other_share_cache', 10.0)),
            )
            st.session_state.base_df = ensure_stable_ids(df_res.copy())
            st.session_state.df_result = df_res.copy()
            st.session_state.calculated_df = df_res.copy()
            st.session_state.summary = summary.copy()
            st.session_state.total_margin = float(total_margin)

            total_loaded = float(pd.to_numeric(seriesize(df_res.get('recommended budget'), len(df_res)), errors='coerce').fillna(0).sum())
            # обновляем сценарный бюджет по результату пересчёта
            st.session_state['scenario_total_budget'] = total_loaded
            other = float(st.session_state.get('other_share_cache', 10.0))
            remember_status_banners(total_loaded, total_loaded, other, total_margin)
            st.session_state['edit_banner_text'] = st.session_state.get('last_summary_message')
            st.session_state['edit_banner_type'] = 'success'
            refreshed_cols = [c for c in show_cols if c in st.session_state.base_df.columns]
            st.session_state['edited_df'] = st.session_state.base_df[refreshed_cols].copy()
            # After saving & recalculation we rerun the app so that
            # the green success banner and updated download buttons
            # immediately reflect the new result.
            st.rerun()
    with c2:
        if st.button('⬅️ Back'):
            origin = st.session_state.get('edit_source')
            st.session_state['edit_source']=None
            st.session_state['edit_banner_text'] = None
            st.session_state.mode = 'result' if origin!='upload' else 'filters'
            st.rerun()

    cur_df = st.session_state.get('calculated_df')
    if cur_df is None:
        cur_df = st.session_state.get('df_result')
    cur_sum = st.session_state.get('summary')
    if cur_df is not None and cur_sum is not None:
        d1,d2 = st.columns(2)
        with d1: st.download_button('💾 Download Result (CSV)', data=export_csv(cur_df), file_name='split_by_placement.csv', mime='text/csv')
        with d2: st.download_button('💾 Download Result (.xlsx)', data=export_excel(cur_df, cur_sum, st.session_state.base_df),
                                    file_name='media_split_results_edited.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
