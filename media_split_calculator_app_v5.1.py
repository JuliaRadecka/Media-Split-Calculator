
# -*- coding: utf-8 -*-
# media_split_calculator_app_v5.py
# v5: фильтры + UI улучшения (CSV/Excel, ИТОГО, success-блок, глазик через column_order)

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# -------------------- Расчёт (как в 4.9) --------------------
def allocate_budget(df, total_budget=240.0, alpha=1.6, beta=1.0, other_share=10.0):
    df = df.copy()
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

    df_valid = df_final[df_final['recommended budget'].fillna(0) > 0].copy()
    if df_valid.empty:
        total_margin = 0.0
    else:
        df_valid['contribution'] = df_valid['recommended budget'] * df_valid['commercial priority']
        total_margin = (df_valid['contribution'].sum() / df_valid['recommended budget'].sum()) * 100.0

    return df_final, summary, total_margin

# -------------------- Recompute from edited result --------------------
def recompute_from_result(df_res, total_budget):
    df_res = df_res.copy()
    # Summary
    if 'category' in df_res.columns and 'recommended budget' in df_res.columns:
        summary2 = df_res.groupby('category', as_index=False)['recommended budget'].sum()
        summary2['share_%'] = (summary2['recommended budget'] / float(total_budget)) * 100.0
    else:
        summary2 = pd.DataFrame()
    # Margin
    total_margin2 = None
    if {'recommended budget','commercial priority'}.issubset(df_res.columns):
        df_valid = df_res[df_res['recommended budget'].fillna(0) > 0].copy()
        if df_valid.empty:
            total_margin2 = 0.0
        else:
            df_valid['contribution'] = df_valid['recommended budget'] * df_valid['commercial priority']
            total_margin2 = (df_valid['contribution'].sum() / df_valid['recommended budget'].sum()) * 100.0
    return summary2, total_margin2


# -------------------- Помощники фильтров --------------------
def apply_platform_bounds(df, bounds):
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
st.session_state.total_budget_cache = float(total_budget)
with col1:
    st.session_state.total_budget_cache = float(total_budget)
with col2:
    alpha = st.slider('α — Agency Profit Weight', 1.0, 2.5, 1.6, 0.1)
with col3:
    beta = st.slider('β — Client Priority Weight', 0.5, 2.0, 1.0, 0.1)
with col4:
    other_share = st.slider('Free Float Share (%)', 0.0, 30.0, 10.0, 1.0)

# ---- Platform Budget
st.markdown('**Platform Budget (mln ₽, min/max) — optional**')
p1, p2, p3, p4 = st.columns(4)
with p1:
    st.caption('Yandex')
    y_min = st.number_input('min (mln ₽)', key='y_min', value=0.0, step=1.0, help='Минимальный бюджет для Yandex')
    y_max = st.number_input('max (mln ₽)', key='y_max', value=0.0, step=1.0, help='Максимальный бюджет для Yandex')
with p2:
    st.caption('DA')
    da_min = st.number_input('min (mln ₽)', key='da_min', value=0.0, step=1.0, help='Минимальный бюджет для DA')
    da_max = st.number_input('max (mln ₽)', key='da_max', value=0.0, step=1.0, help='Максимальный бюджет для DA')
with p3:
    st.caption('VK')
    vk_min = st.number_input('min (mln ₽)', key='vk_min', value=0.0, step=1.0, help='Минимальный бюджет для VK')
    vk_max = st.number_input('max (mln ₽)', key='vk_max', value=0.0, step=1.0, help='Максимальный бюджет для VK')
with p4:
    st.caption('MTS')
    mts_min = st.number_input('min (mln ₽)', key='mts_min', value=0.0, step=1.0, help='Минимальный бюджет для MTS')
    mts_max = st.number_input('max (mln ₽)', key='mts_max', value=0.0, step=1.0, help='Максимальный бюджет для MTS')

platform_bounds = {
    'yandex': {'min': y_min,  'max': y_max},
    'da':     {'min': da_min, 'max': da_max},
    'vk':     {'min': vk_min, 'max': vk_max},
    'mts':    {'min': mts_min,'max': mts_max},
}

# ---- Category Priorities
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

# ---- Black List (multiselect с квадратиками в лейбле)
st.markdown('**Placements — Black List (optional)**')
all_placements = sorted(df['placement'].astype(str).unique().tolist())
if 'bl_selected' not in st.session_state:
    st.session_state.bl_selected = []
opts, defs = [], []
for name in all_placements:
    marked = name in st.session_state.bl_selected
    label = ('☑ ' if marked else '☐ ') + name
    opts.append(label)
    if marked:
        defs.append(label)
chosen_labels = st.multiselect('Exclude placements from calculation', options=opts, default=defs)
chosen_names = [lbl[2:] for lbl in chosen_labels]
st.session_state.bl_selected = chosen_names
blacklist = chosen_names

st.markdown('---')

# ---- Кнопки
if 'mode' not in st.session_state:
    st.session_state.mode = 'default'
colA, colB = st.columns(2)
with colA:
    calc_clicked = st.button('🧮 Calculate')
with colB:
    edit_clicked = st.button('✏️ Edit Input Data')
if calc_clicked:
    st.session_state.mode = 'calculate'
if edit_clicked:
    st.session_state.mode = 'edit'

if st.session_state.mode == 'edit':
    st.subheader('✏️ Edit Input Data')
    edited_df = st.data_editor(df, num_rows='dynamic', use_container_width=True, key='edit_table')
    if st.button('⬆️ Back to Main Menu'):
        st.session_state.mode = 'default'
        st.session_state.edited_df = edited_df


if st.session_state.mode == 'edit_result':
    st.subheader('✏️ Edit Calculated Table')
    df_res = st.session_state.get('df_result')
    if df_res is None or df_res.empty:
        st.info('Нет сохранённого результата. Сначала выполните расчёт.')
    else:
        up_col, _ = st.columns([1,3])
        with up_col:
            uploaded = st.file_uploader('Загрузить предыдущую выгрузку (CSV/XLSX)', type=['csv','xlsx'])
        if uploaded is not None:
            try:
                if uploaded.name.lower().endswith('.csv'):
                    df_res = pd.read_csv(uploaded)
                else:
                    try:
                        df_res = pd.read_excel(uploaded, sheet_name='Placement split')
                    except Exception:
                        df_res = pd.read_excel(uploaded)
            except Exception as e:
                st.error(f'Не удалось прочитать файл: {e}')
        edited = st.data_editor(df_res, use_container_width=True, num_rows='dynamic')
        c1, c2 = st.columns([1,1])
        with c1:
            if st.button('💾 Apply Edits'):
                st.session_state.df_result = edited.copy()
                new_summary, new_margin = recompute_from_result(
                    st.session_state.df_result,
                    float(st.session_state.get('total_budget_cache', 0) or 0) or float(240.0)
                )
                st.session_state.summary = new_summary
                st.session_state.total_margin = new_margin
                st.session_state.mode = 'view_result'
        with c2:
            if st.button('⬅️ Cancel'):
                st.session_state.mode = 'calculate'

elif st.session_state.mode == 'view_result':
    st.subheader('📈 Recommended Split by Placement (edited)')
    df_result = st.session_state.get('df_result', pd.DataFrame()).copy()
    summary = st.session_state.get('summary', pd.DataFrame()).copy()
    total_margin = st.session_state.get('total_margin', None)

    if not df_result.empty:
        all_cols = ['placement', 'category', 'recommended budget',
                    'category priority', 'placement priority', 'minimum spend', 'maximum spend']
        available_cols = [c for c in all_cols if c in df_result.columns]
        table_df = df_result[available_cols].copy()
        total_row = {col: '' for col in available_cols}
        if 'placement' in total_row: total_row['placement'] = 'ИТОГО'
        if 'recommended budget' in total_row: total_row['recommended budget'] = table_df['recommended budget'].sum()
        table_df = pd.concat([table_df, pd.DataFrame([total_row])], ignore_index=True)
        base_order = [c for c in ['placement', 'category', 'recommended budget'] if c in available_cols]
        st.dataframe(table_df, use_container_width=True, column_order=base_order)

    if not summary.empty:
        st.subheader('📊 Summary by Category')
        tot = {'category': 'ИТОГО',
               'recommended budget': summary['recommended budget'].sum(),
               'share_%': 100.0}
        sum_df = pd.concat([summary, pd.DataFrame([tot])], ignore_index=True)
        st.dataframe(sum_df.round(2), use_container_width=True)

    if total_margin is not None:
        st.markdown(f"### 💰 Общая маржинальность сплита: **{float(total_margin)

# ---- Выгрузки + Edit Calculated Table
# CSV — только Placement split (как и в 4.9)
csv_bytes = df_result.to_csv(index=False).encode('utf-8')

dl1, sp1, dl2, sp2, dl3 = st.columns([1, 0.05, 1, 0.05, 1])
with dl1:
    st.download_button('💾 Download Result (CSV)',
                       data=csv_bytes,
                       file_name='media_split_result_v5.csv',
                       mime='text/csv')
with dl2:
    xls = BytesIO()
    try:
        with pd.ExcelWriter(xls, engine='xlsxwriter') as writer:
            df_result.to_excel(writer, index=False, sheet_name='Placement split')
            summary.to_excel(writer, index=False, sheet_name='Summary by category')
    except ModuleNotFoundError:
        with pd.ExcelWriter(xls, engine='openpyxl') as writer:
            df_result.to_excel(writer, index=False, sheet_name='Placement split')
            summary.to_excel(writer, index=False, sheet_name='Summary by category')
    xls.seek(0)
    st.download_button('💾 Download Result (Excel)',
                       data=xls.getvalue(),
                       file_name='media_split_result_v5.xlsx',
                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
with dl3:
    if st.button('✏️ Edit Calculated Table'):
        st.session_state.df_result = df_result.copy()
        st.session_state.summary = summary.copy()
        st.session_state.total_margin = total_margin
        st.session_state.mode = 'edit_result'

:.2f}%**")

    if st.button('⬅️ Back to Calculation'):
        st.session_state.mode = 'calculate'


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

    # ---- Баннер успеха (как в 4.9)
    allocated = float(df_result['recommended budget'].fillna(0).sum())
    percent = (allocated / float(total_budget) * 100.0) if total_budget > 0 else 0.0
    st.success(f'Бюджет успешно распределён: {allocated:.2f} млн ₽ ({percent:.0f}%)')

    # ---- Таблица по площадкам (база + доп.колонки через "глазик")
    st.subheader('📈 Recommended Split by Placement')
    all_cols = ['placement', 'category', 'recommended budget',
                'category priority', 'placement priority', 'minimum spend', 'maximum spend']
    available_cols = [c for c in all_cols if c in df_result.columns]
    table_df = df_result[available_cols].copy()

    # Строка ИТОГО
    total_row = {col: '' for col in available_cols}
    if 'placement' in total_row: total_row['placement'] = 'ИТОГО'
    if 'recommended budget' in total_row: total_row['recommended budget'] = table_df['recommended budget'].sum()
    table_df = pd.concat([table_df, pd.DataFrame([total_row])], ignore_index=True)

    base_order = [c for c in ['placement', 'category', 'recommended budget'] if c in available_cols]
    st.dataframe(table_df, use_container_width=True, column_order=base_order)

    # ---- Сводка по категориям + ИТОГО
    st.subheader('📊 Summary by Category')
    sum_df = summary.copy()
    tot = {'category': 'ИТОГО',
           'recommended budget': sum_df['recommended budget'].sum(),
           'share_%': 100.0}
    sum_df = pd.concat([sum_df, pd.DataFrame([tot])], ignore_index=True)
    st.dataframe(sum_df.round(2), use_container_width=True)

    # ---- Маржинальность
    if total_margin is not None:
        st.markdown(f"### 💰 Общая маржинальность сплита: **{total_margin:.2f}%**")

        
# ---- [downloads removed for reinsertion]
    if st.button('✏️ Edit Calculated Table'):
        st.session_state.df_result = df_result.copy()
        st.session_state.summary = summary.copy()
        st.session_state.total_margin = total_margin
        st.session_state.mode = 'edit_result'

# Кнопка возврата
back_col, _ = st.columns([1,3])
with back_col:
    if st.button('⬆️ Back to Edit Mode'):
        st.session_state.mode = 'edit'
