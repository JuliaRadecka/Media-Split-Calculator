# -*- coding: utf-8 -*-
# media_split_calculator_app_v5.py
# v5: правки ТОЛЬКО по фильтрам (Platform Budget / Category Priorities / Black List)

import streamlit as st
import pandas as pd
import numpy as np

# -------------------- Расчёт (оставляем как в 4.9) --------------------
def allocate_budget(df, total_budget=240.0, alpha=1.6, beta=1.0, other_share=10.0):
    df = df.copy()

    # Числовые поля
    for col, default in [
        ('commercial priority', 0.25),
        ('category priority',   5.0),
        ('placement priority',  5.0),
        ('minimum spend',       0.0),
        ('maximum spend',       1e9),
    ]:
        df[col] = pd.to_numeric(df.get(col, default), errors='coerce').fillna(default)

    # OTHER — отдельный кошелёк
    other_mask   = df['category'].astype(str).str.lower() == 'other'
    other_budget = float(total_budget) * (float(other_share) / 100.0)
    main_budget  = float(total_budget) - other_budget

    # Базовая логика 4.9
    df_main = df[(df['category priority'] <= 3) & (df['placement priority'] <= 2) & (~other_mask)].copy()
    if df_main.empty:
        st.error('Нет площадок, удовлетворяющих условиям фильтрации.')
        return df, pd.DataFrame(), None

    df_main['W'] = (df_main['commercial priority'] ** float(alpha)) * ((1.0 / df_main['placement priority']) ** float(beta))
    df_main['recommended budget'] = df_main['minimum spend']
    remaining = main_budget - df_main['recommended budget'].sum()
    if remaining < 0:
        st.error('Минимальные пороги превышают основной бюджет.')
        return df, pd.DataFrame(), None

    for _ in range(120):
        if remaining <= 1e-6:
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

    if df_main['recommended budget'].sum() > 0:
        df_main['recommended budget'] = (df_main['recommended budget'] / df_main['recommended budget'].sum()) * main_budget

    df_other = df[other_mask].copy()
    if not df_other.empty:
        df_other['recommended budget'] = other_budget / len(df_other)

    df_rest = df[~df.index.isin(df_main.index) & ~df.index.isin(df_other.index)].copy()
    df_rest['recommended budget'] = np.nan

    df_final = pd.concat([df_main, df_other, df_rest], ignore_index=True)

    summary = df_final.groupby('category', as_index=False)['recommended budget'].sum()
    summary['share_%'] = (summary['recommended budget'] / float(total_budget)) * 100.0

    # Условная маржинальность
    df_valid = df_final[df_final['recommended budget'].fillna(0) > 0].copy()
    if df_valid.empty:
        total_margin = 0.0
    else:
        df_valid['contribution'] = df_valid['recommended budget'] * df_valid['commercial priority']
        total_margin = (df_valid['contribution'].sum() / df_valid['recommended budget'].sum()) * 100.0

    return df_final, summary, total_margin


# -------------------- Помощники фильтров --------------------
def apply_platform_bounds(df, bounds):
    """
    0 трактуем как «не задано». Если >0 — перезаписываем min/max для тех placement,
    где название содержит ключ (без регистра).
    """
    df = df.copy()
    if 'placement' not in df.columns:
        return df

    if 'minimum spend' not in df.columns:
        df['minimum spend'] = 0.0
    if 'maximum spend' not in df.columns:
        df['maximum spend'] = 1e9

    for key, mm in bounds.items():
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


def filter_by_categories(df, picked):
    """
    Если выбраны категории — в расчёт идут только они.
    OTHER всегда остаётся (кошелёк Free Float) и считается следующим приоритетом.
    Возвращаем (df_filtered, order_map) для возможной сортировки.
    """
    if not picked:
        return df.copy(), None

    picked_lower = [c.lower() for c in picked]
    cat_ser = df['category'].astype(str).str.lower()
    mask = cat_ser.isin(picked_lower) | (cat_ser == 'other')
    df2 = df[mask].copy()

    order_map = {c.lower(): i for i, c in enumerate(picked, start=1)}
    order_map['other'] = len(picked) + 1
    return df2, order_map


def apply_blacklist(df, blacklist):
    """Исключаем выбранные площадки (точное совпадение)."""
    if not blacklist:
        return df
    return df[~df['placement'].isin(blacklist)].copy()


# -------------------- UI --------------------
st.set_page_config(page_title='Media Split Calculator v5', layout='wide')
st.title('📊 Media Split Calculator — Fixed Bounds (v5)')

FILE_PATH = 'калькулятор.xlsx'
df = pd.read_excel(FILE_PATH)

st.subheader('⚙️ Calculation Parameters')
col1, col2, col3, col4 = st.columns(4)
with col1:
    total_budget = st.number_input('Total Budget (mln ₽)', min_value=0.0, value=240.0, step=1.0)
with col2:
    alpha = st.slider('α — Agency Profit Weight', 1.0, 2.5, 1.6, 0.1)
with col3:
    beta = st.slider('β — Client Priority Weight', 0.5, 2.0, 1.0, 0.1)
with col4:
    other_share = st.slider('Free Float Share (%)', 0.0, 30.0, 10.0, 1.0)

# ---- (1) Platform Budget (mln ₽, min/max) — optional (фильтр до кнопок)
st.markdown('**Platform Budget (mln ₽, min/max) — optional**')
p1, p2, p3, p4 = st.columns(4)
with p1:
    st.caption('Yandex')
    y_min = st.number_input('min', key='y_min', value=0.0, step=1.0, label_visibility='collapsed')
    y_max = st.number_input('max', key='y_max', value=0.0, step=1.0, label_visibility='collapsed')
with p2:
    st.caption('DA')
    da_min = st.number_input('min', key='da_min', value=0.0, step=1.0, label_visibility='collapsed')
    da_max = st.number_input('max', key='da_max', value=0.0, step=1.0, label_visibility='collapsed')
with p3:
    st.caption('VK')
    vk_min = st.number_input('min', key='vk_min', value=0.0, step=1.0, label_visibility='collapsed')
    vk_max = st.number_input('max', key='vk_max', value=0.0, step=1.0, label_visibility='collapsed')
with p4:
    st.caption('MTS')
    mts_min = st.number_input('min', key='mts_min', value=0.0, step=1.0, label_visibility='collapsed')
    mts_max = st.number_input('max', key='mts_max', value=0.0, step=1.0, label_visibility='collapsed')

platform_bounds = {
    'yandex': {'min': y_min,  'max': y_max},
    'da':     {'min': da_min, 'max': da_max},
    'vk':     {'min': vk_min, 'max': vk_max},
    'mts':    {'min': mts_min,'max': mts_max},
}

# ---- (2) Category Priorities — optional
st.markdown('**Category Priorities — optional**')
all_cats = ['CTV', 'ECOM', 'MOB', 'OLV PREM', 'OLV PRG', 'OTHER', 'SOCIAL']
if 'cat_order' not in st.session_state:
    st.session_state.cat_order = []

def _toggle_cat(cat_key):
    chosen = st.session_state.get(cat_key, False)
    label_raw = cat_key.replace('cat_', '')
    label = label_raw.replace('_', ' ').upper()
    for c in all_cats:
        if c.replace(' ', '_') == label_raw:
            label = c
            break
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

# ---- (3) Placements — Black List (optional)
st.markdown('**Placements — Black List (optional)**')
all_placements = sorted(df['placement'].astype(str).unique().tolist())
blacklist = st.multiselect('Exclude placements from calculation', options=all_placements, default=[])

st.markdown('---')

# ---- Кнопки (как в 4.9) ниже всех фильтров
if 'mode' not in st.session_state:
    st.session_state.mode = 'default'

if st.session_state.mode == 'default':
    colA, colB = st.columns(2)
    with colA:
        if st.button('🧮 Calculate'):
            st.session_state.mode = 'calculate'
    with colB:
        if st.button('✏️ Edit Input Data'):
            st.session_state.mode = 'edit'

elif st.session_state.mode == 'edit':
    st.subheader('✏️ Edit Input Data')
    edited_df = st.data_editor(df, num_rows='dynamic', use_container_width=True, key='edit_table')
    if st.button('⬆️ Back to Main Menu'):
        st.session_state.mode = 'default'
        st.session_state.edited_df = edited_df

elif st.session_state.mode == 'calculate':
    df_to_use = st.session_state.get('edited_df', df).copy()
    df_to_use = apply_blacklist(df_to_use, blacklist)
    df_to_use = apply_platform_bounds(df_to_use, platform_bounds)
    df_to_use, order_map = filter_by_categories(df_to_use, st.session_state.cat_order)

    df_result, summary, total_margin = allocate_budget(
        df_to_use,
        total_budget=float(total_budget),
        alpha=float(alpha),
        beta=float(beta),
        other_share=float(other_share)
    )

    if order_map and 'category' in df_result.columns:
        df_result['_cat_ord'] = df_result['category'].astype(str).str.lower().map(order_map).fillna(1e6)
        df_result = df_result.sort_values(by=['_cat_ord', 'recommended budget'],
                                          ascending=[True, False]).drop(columns=['_cat_ord'])

    st.subheader('📈 Recommended Split by Placement')
    base_cols = [c for c in ['placement', 'category', 'recommended budget', 'minimum spend', 'maximum spend'] if c in df_result.columns]
    st.dataframe(df_result[base_cols].round(2), use_container_width=True)

    st.subheader('📊 Summary by Category')
    st.dataframe(summary.round(2), use_container_width=True)

    if total_margin is not None:
        st.caption(f'Overall margin proxy: {total_margin:.2f}%')

    if st.button('⬆️ Back to Edit Mode'):
        st.session_state.mode = 'edit'
