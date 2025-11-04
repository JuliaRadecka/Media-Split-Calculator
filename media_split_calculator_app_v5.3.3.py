# -*- coding: utf-8 -*-
# Media Split Calculator — v5.3 (routing, stable __id, UI tweaks, exports)
# Changes vs v5.6:
# • STRICT routing: filters → result → edit. No cross-rendered buttons or shared blocks.
# • Distinct data scopes per mode; no reliance on globals inside button handlers.
# • Start screen: "📂 Upload a file (.xlsx)" jumps to edit without calc; no warnings until user interacts.
# • Stable __id via canonicalized (placement|category) + SHA1-based 48-bit int. Alias table lives in code.
# • Result table shown via st.data_editor with only 3 columns visible by default; others available via column "eye".
# • Exports: exact column set/order on "Split by Placement"; _meta includes schema_version/app_version and full context.
# • Edit: "Save & Recalculate" button emoji → 🔄; margin title style unified with result.
# • Editor hybrid sourcing: fast path from last calc/_meta; robust fallback to src_df + current filters when needed.
#
# Source data is read from "калькулятор.xlsx" in the working directory.

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import re, hashlib
from datetime import datetime

APP_VERSION = "v5.7"
SCHEMA_VERSION = "2025-11-03.1"

# ---------- alias / canonicalization ----------
# Extend as needed. Keys are canonical slugs, values are lists of aliases.
ALIASES_PLACEMENT = {
    "yandex_direct": ["яндекс директ", "яндекс.директ", "yandex direct", "yd", "direct"],
    "yandex_rsya": ["рся", "rsya", "yandex rsya"],
    "vk_ads": ["вк", "vk", "vk ads", "вконтакте"],
    "mts": ["mts", "мтс", "mts advertising"],
    "da": ["da", "digital alliance", "цифровой альянс"],
}
ALIASES_CATEGORY = {
    "ctv": ["ctv"],
    "olv_prem": ["olv prem", "olv premium", "olv"],
    "media": ["media", "медиа"],
    "prg": ["prg", "programmatic"],
    "social": ["social", "соцсети", "социальные сети"],
    "ecom": ["ecom", "e-commerce", "интернет-магазины"],
    "mob": ["mob", "mobile"],
    "geomedia": ["geomedia"],
    "geoperfom": ["geoperfom", "geoperformance", "геоперфом"],
    "promopages": ["promopages", "промостраницы"],
    "rsya": ["рся", "rsya"],
    "direct": ["direct", "директ"],
    "cpa": ["cpa"],
    "them": ["them", "тематики", "тематические"],
    "other": ["other", "прочее", "другое"],
}

def _norm_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).replace("\u00A0", " ").replace("\u2009", " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    s = s.replace("–","-").replace("—","-").replace("−","-")
    return s

def _slug_from_alias(s: str, table: dict) -> str:
    s_norm = _norm_text(s)
    for canon, aliases in table.items():
        for a in aliases:
            if _norm_text(a) == s_norm:
                return canon
    # fallback: normalized slug (safe)
    return re.sub(r"[^a-z0-9_]+", "_", s_norm)

def canon_placement(x): return _slug_from_alias(x, ALIASES_PLACEMENT)
def canon_category(x):  return _slug_from_alias(x, ALIASES_CATEGORY)

def make_stable_id(placement: str, category: str) -> np.int64:
    """Deterministic 48-bit integer id based on canonicalized placement|category."""
    key = f"{canon_placement(placement)}|{canon_category(category)}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]  # 48 bits hex
    return np.int64(int(h, 16))

def ensure_mode():
    if 'mode' not in st.session_state:
        st.session_state.mode = 'filters'

def set_mode(m):
    st.session_state.mode = m
    st.rerun()

def ensure_stable_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if '__id' not in df.columns or df['__id'].isna().any():
        df['__id'] = [
            make_stable_id(p, c)
            for p, c in zip(df['placement'].astype(str), df['category'].astype(str))
        ]
    return df

def apply_editor_to_base(base_df, edited_df, editable_cols):
    base = base_df.set_index('__id').copy()
    patch = edited_df[['__id', *editable_cols]].set_index('__id').copy()
    for col in editable_cols:
        if col in patch.columns:
            patch[col] = pd.to_numeric(patch[col], errors='coerce')
    for col in editable_cols:
        if col in base.columns and col in patch.columns:
            base.loc[patch.index, col] = patch[col]
    base = base.reset_index()
    return base

def collect_meta():
    return {
        'schema_version': SCHEMA_VERSION,
        'app_version': APP_VERSION,
        'exported_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_budget': float(st.session_state.get('total_budget_cache', 240.0)),
        'alpha': float(st.session_state.get('alpha_cache', 1.6)),
        'beta': float(st.session_state.get('beta_cache', 1.0)),
        'free_float_%': float(st.session_state.get('other_share_cache', 10.0)),
        'selected_categories': st.session_state.get('cat_order', []),
        'blacklist': st.session_state.get('bl_selected', []),
        'platform_bounds': st.session_state.get('platform_bounds', {}),
        'scaled_mins': bool(st.session_state.get('last_meta_scaled_mins', False)),
        'scale_k': float(st.session_state.get('last_meta_scale_coef', 1.0)),
        # alias tables snapshot (optional for debugging/data lineage)
        'aliases_present': True,
    }

def export_csv(df):
    export_df = df.drop(columns=[c for c in ['commercial priority','W','available','__id'] if c in df.columns], errors='ignore')
    return export_df.to_csv(index=False).encode('utf-8')

def export_excel(df_split, df_sum, base_df_with_ids):
    # enforce exact columns & order on "Split by Placement"
    desired_cols = ['placement','category','category priority','placement priority','minimum spend','maximum spend','recommended budget']
    split_cols = [c for c in desired_cols if c in df_split.columns]
    split_df = df_split[split_cols].copy()

    _xls = BytesIO()
    with pd.ExcelWriter(_xls, engine='openpyxl') as writer:
        split_df.to_excel(writer, index=False, sheet_name='Split by Placement')
        df_sum.to_excel(writer, index=False, sheet_name='Summary by Category')
        # _meta with params + id map
        meta_df = pd.DataFrame([collect_meta()])
        meta_df.to_excel(writer, index=False, sheet_name='_meta')
        start_row = len(meta_df) + 2
        id_map = base_df_with_ids[['__id','placement','category']].copy()
        id_map.to_excel(writer, index=False, sheet_name='_meta', startrow=start_row)
    _xls.seek(0)
    return _xls.getvalue()

# ---------- domain helpers ----------

def filter_by_categories(df, picked):
    if not picked:
        return df.copy(), None
    picked_lower = [c.lower() for c in picked]
    mask = df['category'].astype(str).str.lower().isin(picked_lower) | (df['category'].astype(str).str.lower() == 'other')
    df2 = df[mask].copy()
    order_map = {c.lower(): i for i, c in enumerate(picked, start=1)}
    order_map['other'] = len(picked) + 1
    return df2, order_map

def apply_blacklist(df, blacklist):
    if not blacklist:
        return df.copy()
    return df[~df['placement'].isin(blacklist)].copy()

def apply_platform_bounds(df, bounds):
    df = df.copy()
    if 'minimum spend' not in df.columns:
        df['minimum spend'] = 0.0
    if 'maximum spend' not in df.columns:
        df['maximum spend'] = 1e9
    for key, mm in (bounds or {}).items():
        if not isinstance(mm, dict):
            continue
        mn = float(mm.get('min', 0) or 0)
        mx = float(mm.get('max', 0) or 0)
        if mn <= 0 and mx <= 0:
            continue
        mask = df['placement'].astype(str).str.lower().str.contains(key)
        if mn > 0:
            df.loc[mask, 'minimum spend'] = mn
        if mx > 0:
            df.loc[mask, 'maximum spend'] = mx
    return df

def _ensure_other_summary(summary_df, other_budget):
    try:
        if other_budget is None:
            return summary_df
        cats = summary_df['category'].astype(str).str.lower()
        if 'other' not in set(cats):
            extra = pd.DataFrame([{'category': 'other', 'recommended budget': float(other_budget)}])
            summary_df = pd.concat([summary_df, extra], ignore_index=True)
        return summary_df
    except Exception:
        return summary_df

def allocate_budget(df, total_budget=240.0, alpha=1.6, beta=1.0, other_share=10.0, use_gates=None):
    meta = {'scaled_mins': False, 'scale_coef': 1.0}
    df = df.copy()

    if use_gates is None:
        has_cat = ('category priority' in df.columns) and df['category priority'].notna().any()
        has_plc = ('placement priority' in df.columns) and df['placement priority'].notna().any()
        use_gates = bool(has_cat or has_plc)

    for col, default in [
        ('commercial priority', 0.25),
        ('category priority',   5.0),
        ('placement priority',  5.0),
        ('minimum spend',       0.0),
        ('maximum spend',       1e9),
    ]:
        df[col] = pd.to_numeric(df.get(col, default), errors='coerce').fillna(default)

    other_mask   = df['category'].astype(str).str.lower() == 'other'
    other_budget = float(total_budget) * (float(other_share) / 100.0)
    main_budget  = float(total_budget) - other_budget

    gate_mask = (df['category priority'] <= 3) & (df['placement priority'] <= 2) if use_gates else True
    df_main = df[gate_mask & (~other_mask)].copy()
    if df_main.empty:
        empty_df = df.assign(**{'recommended budget': np.nan, 'W': np.nan, 'available': np.nan})
        empty_summary = pd.DataFrame({'category': pd.Series(dtype='object'),
                                      'recommended budget': pd.Series(dtype='float'),
                                      'share_%': pd.Series(dtype='float')})
        return empty_df, empty_summary, 0.0, meta

    df_main['W'] = (df_main['commercial priority'] ** float(alpha)) * ((1.0 / df_main['placement priority']) ** float(beta))
    df_main['recommended budget'] = df_main['minimum spend']

    remaining = main_budget - df_main['recommended budget'].sum()
    if remaining < -1e-9:
        total_min = max(df_main['recommended budget'].sum(), 1e-9)
        scale = max(main_budget, 0.0) / total_min
        df_main['recommended budget'] = df_main['recommended budget'] * scale
        meta['scaled_mins'] = True
        meta['scale_coef'] = float(scale)
        remaining = 0.0

    for _ in range(160):
        if remaining <= 1e-9:
            break
        df_main['available'] = df_main['maximum spend'] - df_main['recommended budget']
        elig = df_main['available'] > 0
        total_w = df_main.loc[elig, 'W'].sum()
        if total_w <= 0:
            break
        inc = (df_main.loc[elig, 'W'] / total_w) * remaining
        inc = np.minimum(inc, df_main.loc[elig, 'available'])
        df_main.loc[elig, 'recommended budget'] += inc
        remaining = main_budget - df_main['recommended budget'].sum()

    sum_main = df_main['recommended budget'].sum()
    if sum_main > 0:
        df_main['recommended budget'] = (df_main['recommended budget'] / sum_main) * main_budget

    df_other = df[other_mask].copy()
    if not df_other.empty:
        df_other['recommended budget'] = other_budget / len(df_other)

    df_rest = df[~df.index.isin(df_main.index) & ~df.index.isin(df_other.index)].copy()
    df_rest['recommended budget'] = np.nan

    df_final = pd.concat([df_main, df_other, df_rest], ignore_index=True)

    summary = df_final.groupby('category', as_index=False)['recommended budget'].sum()
    summary = _ensure_other_summary(summary, other_budget)
    if float(total_budget) > 0:
        summary['share_%'] = (summary['recommended budget'] / float(total_budget)) * 100.0
    else:
        summary['share_%'] = 0.0

    df_valid = df_final[df_final['recommended budget'].fillna(0) > 0].copy()
    if df_valid.empty:
        total_margin = 0.0
    else:
        df_valid['contribution'] = df_valid['recommended budget'] * df_valid['commercial priority']
        total_margin = (df_valid['contribution'].sum() / df_valid['recommended budget'].sum()) * 100.0

    # store last meta flags for export
    st.session_state['last_meta_scaled_mins'] = meta['scaled_mins']
    st.session_state['last_meta_scale_coef'] = meta['scale_coef']

    return df_final, summary, total_margin, meta

# ---------- app ----------

st.set_page_config(page_title='📊 Media Split Calculator v5.3', layout='wide')
st.title('📊 Media Split Calculator — v5.3')

# Load source
FILE_PATH = 'калькулятор.xlsx'
try:
    src_df = pd.read_excel(FILE_PATH)
except Exception as e:
    st.error(f'Не удалось прочитать исходный файл {FILE_PATH}: {e}')
    st.stop()

ensure_mode()

# Controls header & filters (hidden in Edit)
if st.session_state.mode != 'edit':
    st.subheader('⚙️ Calculation Parameters')
col1, col2, col3, col4 = st.columns(4)
with col1:
    total_budget = st.number_input('Total Budget (mln ₽)', min_value=0.0, value=float(st.session_state.get('total_budget_cache', 240.0)), step=10.0)
    st.session_state.total_budget_cache = float(total_budget)
with col2:
    alpha = st.slider('α — Agency Profit Weight', 1.0, 2.5, float(st.session_state.get('alpha_cache', 1.6)), 0.1)
    st.session_state.alpha_cache = float(alpha)
with col3:
    beta = st.slider('β — Client Priority Weight', 0.5, 2.0, float(st.session_state.get('beta_cache', 1.0)), 0.1)
    st.session_state.beta_cache = float(beta)
with col4:
    other_share = st.slider('Free Float Share (%)', 0.0, 30.0, float(st.session_state.get('other_share_cache', 10.0)), 1.0)
    st.session_state.other_share_cache = float(other_share)

# Platform bounds
st.markdown('**Platform Budget (mln ₽, min/max) — optional**')
p1, p2, p3, p4 = st.columns(4)
with p1:
    st.caption('Yandex')
    y_min = st.number_input('min (mln ₽)', key='y_min', value=float(st.session_state.get('y_min', 0.0)), step=10.0)
    y_max = st.number_input('max (mln ₽)', key='y_max', value=float(st.session_state.get('y_max', 0.0)), step=10.0)
with p2:
    st.caption('DA')
    da_min = st.number_input('min (mln ₽)', key='da_min', value=float(st.session_state.get('da_min', 0.0)), step=10.0)
    da_max = st.number_input('max (mln ₽)', key='da_max', value=float(st.session_state.get('da_max', 0.0)), step=10.0)
with p3:
    st.caption('VK')
    vk_min = st.number_input('min (mln ₽)', key='vk_min', value=float(st.session_state.get('vk_min', 0.0)), step=10.0)
    vk_max = st.number_input('max (mln ₽)', key='vk_max', value=float(st.session_state.get('vk_max', 0.0)), step=10.0)
with p4:
    st.caption('MTS')
    mts_min = st.number_input('min (mln ₽)', key='mts_min', value=float(st.session_state.get('mts_min', 0.0)), step=10.0)
    mts_max = st.number_input('max (mln ₽)', key='mts_max', value=float(st.session_state.get('mts_max', 0.0)), step=10.0)

platform_bounds = {'yandex': {'min': y_min, 'max': y_max},
                   'da': {'min': da_min, 'max': da_max},
                   'vk': {'min': vk_min, 'max': vk_max},
                   'mts': {'min': mts_min, 'max': mts_max}}
st.session_state.platform_bounds = platform_bounds

# Category priorities
st.markdown('**Category Priorities — optional**')
all_cats = ['CTV', 'OLV PREM', 'Media', 'PRG', 'SOCIAL', 'ECOM', 'MOB', 'Geomedia', 'Geoperfom', 'Promopages', 'РСЯ', 'Direct', 'CPA', 'THEM']
if 'cat_order' not in st.session_state:
    st.session_state.cat_order = []

def _toggle_cat(cat_key):
    chosen = st.session_state.get(cat_key, False)
    label_raw = cat_key.replace('cat_', '')
    label = next((c for c in all_cats if c.replace(' ', '_') == label_raw), label_raw.replace('_',' ').upper())
    if chosen and label not in st.session_state.cat_order:
        st.session_state.cat_order.append(label)
    if (not chosen) and label in st.session_state.cat_order:
        st.session_state.cat_order.remove(label)

cat_cols = st.columns(len(all_cats))
order_map_show = {c: i for i, c in enumerate(st.session_state.cat_order, start=1)}
for i, cat in enumerate(all_cats):
    key = 'cat_' + cat.replace(' ', '_')
    prefix = f"{order_map_show.get(cat, '□')}  "
    cat_cols[i].checkbox(prefix + cat, key=key, value=(cat in st.session_state.cat_order),
                         on_change=_toggle_cat, args=(key,))

# Black list
st.markdown('**Placements — Black List (optional)**')
plc_series = src_df['placement'].dropna().astype(str).map(lambda s: s.strip()).replace({'': np.nan, 'None': np.nan, 'none': np.nan, 'nan': np.nan})
all_placements = sorted(plc_series.dropna().unique().tolist())
if 'bl_selected' not in st.session_state:
    st.session_state.bl_selected = []
chosen_names = st.multiselect('Exclude placements from calculation', options=all_placements, default=st.session_state.bl_selected)
st.session_state.bl_selected = chosen_names
blacklist = chosen_names

st.markdown('---')

# ---------- ROUTING ----------

if st.session_state.mode == 'filters':
    cA, cB = st.columns(2)
    with cA:
        if st.button('🧮 Calculate'):
            st.session_state['edit_source'] = 'calc'
            df0 = src_df.copy()
            df0 = apply_blacklist(df0, blacklist)
            df0 = apply_platform_bounds(df0, platform_bounds)
            picked = st.session_state.cat_order
            df0, order_map = filter_by_categories(df0, picked)
            df0 = ensure_stable_ids(df0)
            # assign current dfs
            st.session_state.base_df = df0.copy()
            df_result, summary, total_margin, meta = allocate_budget(
                df0,
                total_budget=float(st.session_state.total_budget_cache),
                alpha=float(st.session_state.alpha_cache),
                beta=float(st.session_state.beta_cache),
                other_share=float(st.session_state.other_share_cache),
                use_gates=None
            )
            if order_map and 'category' in df_result.columns:
                df_result['_cat_ord'] = df_result['category'].astype(str).str.lower().map(order_map).fillna(1e6)
                df_result = df_result.sort_values(by=['_cat_ord', 'recommended budget'],
                                                  ascending=[True, False]).drop(columns=['_cat_ord'])
            st.session_state.df_result = df_result.copy()
            st.session_state.summary = summary.copy()
            st.session_state.total_margin = float(total_margin)
            set_mode('result')
    with cB:
        if st.button('📂 Upload a file (.xlsx)'):
            st.session_state['edit_source'] = 'upload'
            # jump to edit without any warnings; user will decide to upload
            set_mode('edit')

elif st.session_state.mode == 'result':
    df_result = st.session_state.get('df_result')
    summary   = st.session_state.get('summary')
    total_margin = st.session_state.get('total_margin')
    if df_result is None or summary is None:
        st.info('Нет результата расчёта. Сначала выполните Calculate.')
    else:
        if st.session_state.get('total_margin') is not None:
            st.success(f\"✅ Бюджет успешно распределён: {float(st.session_state.get('total_budget_cache',0)):.2f} млн ₽ (100%)\")
        st.subheader('📈 Recommended Split by Placement')
        # Build table; show only 3 columns by default, others hidden under the eye
        all_cols = [c for c in ['placement','category','recommended budget','category priority','placement priority','minimum spend','maximum spend'] if c in df_result.columns]
        table_df = df_result[all_cols].copy()
        # add total row
        total_row = {col: '' for col in all_cols}
        if 'placement' in total_row: total_row['placement'] = 'ИТОГО'
        if 'recommended budget' in total_row: total_row['recommended budget'] = table_df['recommended budget'].sum()
        table_df = pd.concat([table_df, pd.DataFrame([total_row])], ignore_index=True)

        st.data_editor(
            table_df,
            use_container_width=True,
            hide_index=True,
            disabled=True,
            column_order=[c for c in ['placement','category','recommended budget'] if c in table_df.columns],
            column_config={
                'recommended budget': st.column_config.NumberColumn('recommended budget', format='%.6f')
            },
            key='result_table_v57'
        )

        st.subheader('📊 Summary by Category')
        sum_df = summary.copy()
        tot = {'category':'ИТОГО','recommended budget': float(sum_df['recommended budget'].sum()), 'share_%': 100.0}
        sum_df = pd.concat([sum_df, pd.DataFrame([tot])], ignore_index=True)
        st.dataframe(sum_df.round(2), use_container_width=True)

        if total_margin is not None:
            st.success(f"✅ Бюджет успешно распределён: {float(st.session_state.get('total_budget_cache',0)):.2f} млн ₽ (100%)")
            st.markdown(f"### 💰 Общая маржинальность сплита: **{float(total_margin):.2f}%**")

        col_csv, col_xlsx = st.columns(2)
        with col_csv:
            st.download_button('💾 Download Results (CSV)',
                               data=export_csv(df_result),
                               file_name='split_by_placement.csv', mime='text/csv')
        with col_xlsx:
            st.download_button('💾 Download Results (.xlsx)',
                               data=export_excel(df_result, summary, st.session_state.base_df),
                               file_name='media_split_results.xlsx',
                               mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        if st.button('✏️ Edit Calculated Table'):
            set_mode('edit')

elif st.session_state.mode == 'edit':
    st.subheader('✏️ Edit Calculated Table')
    # Always offer uploader; no warnings before user tries to use it
    uploaded = st.file_uploader('Upload Excel with _meta to restore previous editor state (.xlsx only)', type=['xlsx'])
    src_flag = st.session_state.get('edit_source')
    waiting_for_file = (src_flag == 'upload' and uploaded is None)

    # Decide editor source (fast path + fallback):
    base_df    = st.session_state.get('base_df')
    current_df = st.session_state.get('df_result')

    # If user uploaded an Excel produced by the app — reconstruct editor state
    if uploaded is not None:
        try:
            xls = pd.ExcelFile(uploaded)
            split_df = pd.read_excel(xls, 'Split by Placement')
            if '_meta' in xls.sheet_names:
                meta_sheet = pd.read_excel(xls, '_meta')
                try:
                    id_map = pd.read_excel(xls, '_meta', skiprows=len(meta_sheet)+2)
                    if {'__id','placement','category'}.issubset(id_map.columns):
                        base_df = split_df.merge(id_map[['__id','placement','category']], on=['placement','category'], how='left')
                        base_df = ensure_stable_ids(base_df) if base_df['__id'].isna().any() else base_df
                    else:
                        base_df = ensure_stable_ids(split_df)
                except Exception:
                    base_df = ensure_stable_ids(split_df)
            else:
                # Fallback: try to match by normalized placement|category (no ids)
                split_df['_key'] = split_df['placement'].map(_norm_text) + '|' + split_df['category'].map(_norm_text)
                cur = (current_df.copy() if current_df is not None else src_df.copy())
                cur['_key'] = cur['placement'].map(_norm_text) + '|' + cur['category'].map(_norm_text)
                base_df = cur.merge(split_df[['_key','category priority','placement priority','minimum spend','maximum spend','recommended budget']],
                                    on='_key', how='left', suffixes=('','_edited'))
                base_df = ensure_stable_ids(base_df.drop(columns=['_key']))
            st.session_state.base_df = base_df.copy()
            st.success('Файл загружен. Состояние редактора восстановлено.')
        except Exception as e:
            st.error(f'Не удалось прочитать файл: {e}')

    if waiting_for_file:
        cols = st.columns(2)
        with cols[1]:
            if st.button('⬅️ Back'):
                st.session_state['edit_source'] = None
                st.session_state.mode = 'filters'
                st.rerun()
        st.stop()

    # If still no base_df, but there is a calc result — quick path from result
    if base_df is None and current_df is not None:
        st.session_state.base_df = ensure_stable_ids(current_df.copy())
        base_df = st.session_state.base_df

    # If still none — robust fallback from raw source + current filters
    if base_df is None:
        df0 = src_df.copy()
        df0 = apply_blacklist(df0, st.session_state.get('bl_selected', []))
        df0 = apply_platform_bounds(df0, st.session_state.get('platform_bounds', {}))
        picked = st.session_state.get('cat_order', [])
        df0, _ = filter_by_categories(df0, picked)
        base_df = ensure_stable_ids(df0)
        st.session_state.base_df = base_df.copy()

    # Assign "current" dfs just before editor render
    editable_cols = ['category priority','placement priority','minimum spend','maximum spend','recommended budget']
    show_cols = ['__id','placement','category', *editable_cols]
    show_cols = [c for c in show_cols if c in base_df.columns]
    editor_df = base_df[show_cols].copy()

    edited = st.data_editor(
        editor_df,
        use_container_width=True,
        num_rows='fixed',
        disabled=['__id','placement','category'],
        column_config={'recommended budget': st.column_config.NumberColumn('recommended budget', format='%.6f')},
        key='editor_table_v57'
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button('🔄 Save & Recalculate'):
            base_applied = apply_editor_to_base(base_df, edited, editable_cols)
            picked = st.session_state.get('cat_order', [])
            df_in, order_map = filter_by_categories(base_applied, picked)
            df_in = apply_blacklist(df_in, st.session_state.get('bl_selected', []))
            df_in = apply_platform_bounds(df_in, st.session_state.get('platform_bounds', {}))
            df_result, summary, total_margin, meta = allocate_budget(
                df_in,
                total_budget=float(st.session_state.total_budget_cache),
                alpha=float(st.session_state.alpha_cache),
                beta=float(st.session_state.beta_cache),
                other_share=float(st.session_state.other_share_cache),
                use_gates=None
            )
            st.session_state.base_df = base_applied.copy()
            st.session_state.df_result = df_result.copy()
            st.session_state.summary = summary.copy()
            st.session_state.total_margin = float(total_margin)
            st.success(f"✅ Бюджет успешно распределён: {float(st.session_state.get('total_budget_cache',0)):.2f} млн ₽ (100%)")
            st.markdown(f"### 💰 Общая маржинальность сплита: **{float(total_margin):.2f}%**")
    with c2:
        if st.button('⬅️ Back'):
            origin = st.session_state.get('edit_source')
            st.session_state['edit_source'] = None
            if origin == 'upload':
                set_mode('filters')
            else:
                set_mode('result')

    # Downloads for EDIT state (edited/current)
    cur_df = st.session_state.get('df_result')
    cur_sum = st.session_state.get('summary')
    if cur_df is not None and cur_sum is not None:
        d1, d2 = st.columns(2)
        with d1:
            st.download_button('💾 Download Result (CSV)',
                               data=export_csv(cur_df),
                               file_name='split_by_placement.csv',
                               mime='text/csv')
        with d2:
            st.download_button('💾 Download Result (.xlsx)',
                               data=export_excel(cur_df, cur_sum, st.session_state.base_df),
                               file_name='media_split_results_edited.xlsx',
                               mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
