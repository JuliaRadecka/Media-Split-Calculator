
# -*- coding: utf-8 -*-
# Media Split Calculator — v5.4 (final ui/export/meta patch)
# Changes vs your 'рабочая' version:
# 1) Two main export buttons are side-by-side; "Edit Calculated Table" moved to the next row.
# 2) CSV/Excel: sheet with placements is named exactly "Split by Placement" (CSV logic mirrors it).
#    Removed columns: 'commercial priority', 'W', 'available' from exports.
#    Excel additionally contains a service sheet "_meta" to allow loading this file in Edit later.
# 3) In Edit Calculated Table: compact uploader (≈1/4 row, left), Save/Cancel + two export buttons in tight pairs.
#    Exports from Edit also include "_meta" and use the *edited* table.
# 4) Black List: chips in the row show only names (no pseudo-checkboxes). Selection persists in session_state.
# 5) No 'учитывать пороги' checkbox anywhere.
#
# Note: This file keeps your existing allocation logic & parameters structure. If you see 'TODO' markers,
#       they are safe fallbacks if a specific optional column is absent in your dataset.

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# ---------- Helpers
def _norm_key(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace("\u00A0", " ", regex=False).str.replace("\t", " ", regex=False)
    return s.str.strip().str.lower()

def apply_blacklist(df: pd.DataFrame, blacklist):
    if not blacklist:
        return df
    return df[~df['placement'].isin(blacklist)].copy()

def filter_by_categories(df: pd.DataFrame, picked):
    if not picked:
        return df.copy(), None
    picked_lower = [c.lower() for c in picked]
    cat_ser = df['category'].astype(str).str.lower()
    mask = cat_ser.isin(picked_lower) | (cat_ser == 'other')
    df2 = df[mask].copy()
    order_map = {c.lower(): i for i, c in enumerate(picked, start=1)}
    order_map['other'] = len(picked) + 1
    return df2, order_map

def allocate_budget(df, total_budget=240.0, alpha=1.6, beta=1.0, other_share=10.0, use_gates=None):
    # (Same robust allocator as in prior v5.4; trimmed comments.)
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

    # start from mins
    df_main['recommended budget'] = df_main['minimum spend']
    remaining = main_budget - df_main['recommended budget'].sum()
    if remaining < -1e-9:
        total_min = max(df_main['recommended budget'].sum(), 1e-9)
        scale = max(main_budget, 0.0) / total_min
        df_main['recommended budget'] = df_main['recommended budget'] * scale
        meta['scaled_mins'] = True
        meta['scale_coef'] = float(scale)
        remaining = 0.0

    for _ in range(120):
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
    if float(total_budget) > 0:
        summary['share_%'] = (summary['recommended budget'] / float(total_budget)) * 100.0
    else:
        summary['share_%'] = 0.0

    df_valid = df_final[df_final['recommended budget'].fillna(0) > 0].copy()
    total_margin = 0.0 if df_valid.empty else (df_valid.eval("recommended budget * `commercial priority`").sum() / df_valid['recommended budget'].sum()) * 100.0

    return df_final, summary, total_margin, meta

def _export_tables(df_result, summary, meta_dict, edited=False):
    # Strip service columns in export
    drop_cols = [c for c in ['commercial priority', 'W', 'available'] if c in df_result.columns]
    export_df = df_result.drop(columns=drop_cols, errors='ignore').copy()

    # CSV: only the "Split by Placement" sheet logic
    csv_bytes = export_df.to_csv(index=False).encode('utf-8')

    # Excel with _meta
    xls = BytesIO()
    with pd.ExcelWriter(xls, engine='xlsxwriter') as writer:
        export_df.to_excel(writer, index=False, sheet_name='Split by Placement')
        summary.to_excel(writer, index=False, sheet_name='Summary by Category')
        pd.DataFrame([meta_dict]).T.rename(columns={0: 'value'}).to_excel(writer, sheet_name='_meta')  # key/value style
    xls.seek(0)
    xls_bytes = xls.getvalue()

    csv_name  = f"media_split_result_v5{'_edited' if edited else ''}.csv"
    xls_name  = f"media_split_result_v5{'_edited' if edited else ''}.xlsx"
    return csv_bytes, xls_bytes, csv_name, xls_name

# ---------- App
st.set_page_config(page_title='Media Split Calculator v5.4', layout='wide')
st.title('📊 Media Split Calculator — v5.4')

# Load user base (left as in your working app — adjust path as needed)
FILE_PATH = 'калькулятор.xlsx'
try:
    base_df = pd.read_excel(FILE_PATH)
except Exception:
    st.error('Не удалось прочитать исходные данные (калькулятор.xlsx). Поместите файл рядом с приложением.')
    st.stop()

# Cache params we show in exports
def _set_cache(k, v): st.session_state[k] = v

st.subheader('⚙️ Calculation Parameters')
c1, c2, c3, c4 = st.columns(4)
with c1:
    total_budget = st.number_input('Total Budget (mln ₽)', min_value=0.0, value=300.0, step=10.0, key='tb')
    _set_cache('total_budget', float(total_budget))
with c2:
    alpha = st.slider('α — Agency Profit Weight', 1.0, 2.5, 1.6, 0.1, key='alpha')
    _set_cache('alpha', float(alpha))
with c3:
    beta = st.slider('β — Client Priority Weight', 0.5, 2.0, 1.0, 0.1, key='beta')
    _set_cache('beta', float(beta))
with c4:
    other_share = st.slider('Free Float Share (%)', 0.0, 30.0, 10.0, 1.0, key='other')
    _set_cache('other_share', float(other_share))

# Category Priorities — optional (keep your order)
all_cats = ['CTV','OLV PREM','Media','PRG','SOCIAL','ECOM','MOB','Geomedia','Geoperfom','Promopages','РСЯ','Direct','CPA','THEM']
if 'cat_order' not in st.session_state:
    st.session_state.cat_order = []
st.markdown('**Category Priorities — optional**')
ccols = st.columns(len(all_cats))
def _toggle(cat):
    chosen = st.session_state.get(cat, False)
    label = cat.replace('_', ' ')
    if chosen and label not in st.session_state.cat_order:
        st.session_state.cat_order.append(label)
    if (not chosen) and label in st.session_state.cat_order:
        st.session_state.cat_order.remove(label)

for i, cat in enumerate(all_cats):
    ckey = f'cat_{cat.replace(" ", "_")}'
    ccols[i].checkbox(cat, key=ckey, value=(cat in st.session_state.cat_order), on_change=_toggle, args=(ckey,))

# Black list — chips show only names
st.markdown('**Placements — Black List (optional)**')
clean_pl = base_df['placement'].dropna().astype(str).map(str.strip)
clean_pl = clean_pl.replace({'': np.nan, 'None': np.nan, 'none': np.nan, 'nan': np.nan})
all_placements = sorted(clean_pl.dropna().unique().tolist())
if 'bl_selected' not in st.session_state:
    st.session_state.bl_selected = []
bl = st.multiselect('Exclude placements from calculation', options=all_placements, default=st.session_state.bl_selected, key='bl_ms')
st.session_state.bl_selected = bl

st.markdown('---')

# Calculate / Edit flow
if 'mode' not in st.session_state: st.session_state.mode = 'default'
btns = st.columns(2)
with btns[0]:
    calc = st.button('🧮 Calculate', use_container_width=True)
with btns[1]:
    go_edit = st.button('✏️ Edit Input Data', use_container_width=True)
if calc: st.session_state.mode = 'calculate'
if go_edit: st.session_state.mode = 'edit'

# EDIT INPUT
if st.session_state.mode == 'edit':
    st.subheader('✏️ Edit Input Data')
    st.session_state.edited_df = st.data_editor(base_df, num_rows='dynamic', use_container_width=True, key='input_editor')
    st.info('Изменения применяются к следующему расчёту.')
    st.button('⬅️ Back', on_click=lambda: st.session_state.update(mode='default'))

# CALCULATE
if st.session_state.mode == 'calculate':
    df = st.session_state.get('edited_df', base_df).copy()
    df = apply_blacklist(df, st.session_state.bl_selected)
    df, order_map = filter_by_categories(df, st.session_state.cat_order)

    # Allocation
    df_result, summary, total_margin, meta = allocate_budget(
        df, total_budget=st.session_state.total_budget, alpha=st.session_state.alpha,
        beta=st.session_state.beta, other_share=st.session_state.other_share, use_gates=None
    )

    st.success(f'Бюджет успешно распределён: {st.session_state.total_budget:.2f} млн ₽ (100%)')

    # Display tables
    st.subheader('📈 Recommended Split by Placement')
    show_cols = [c for c in ['placement','category','recommended budget','category priority','placement priority','minimum spend','maximum spend'] if c in df_result.columns]
    st.dataframe(df_result[show_cols], use_container_width=True)
    st.subheader('📊 Summary by Category')
    st.dataframe(summary.round(2), use_container_width=True)
    st.markdown(f"### 💰 Общая маржинальность сплита: **{float(total_margin):.2f}%**")

    # --- MAIN EXPORTS (two buttons together), then edit button on next row
    meta_export = {
        'total_budget': st.session_state.total_budget,
        'alpha': st.session_state.alpha,
        'beta': st.session_state.beta,
        'other_share': st.session_state.other_share,
        'categories_order': ','.join(st.session_state.cat_order),
        'blacklist': ','.join(st.session_state.bl_selected),
        'mode': 'calculate'
    }
    csv_bytes, xls_bytes, csv_name, xls_name = _export_tables(df_result, summary, meta_export, edited=False)
    b1, spacer, b2 = st.columns([1,0.05,1])
    with b1:
        st.download_button('💾 Download Result (CSV)', data=csv_bytes, file_name=csv_name, mime='text/csv', use_container_width=True)
    with b2:
        st.download_button('💾 Download Result (Excel)', data=xls_bytes, file_name=xls_name,
                           mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', use_container_width=True)

    st.container().button('✏️ Edit Calculated Table', use_container_width=False,
                          on_click=lambda: st.session_state.update(mode='edit_result',
                                                                  df_result=df_result.copy(),
                                                                  summary=summary.copy(),
                                                                  last_calc_base=st.session_state.get('edited_df', base_df).copy()))

# EDIT CALCULATED TABLE
if st.session_state.mode == 'edit_result':
    st.subheader('✏️ Edit Calculated Table')
    base = st.session_state.get('last_calc_base')
    current = st.session_state.get('df_result')

    if base is None or current is None or current.empty:
        st.info('Нет результата для редактирования. Сначала выполните расчёт.')
    else:
        # Prepare editable frame once
        if "editor_df" not in st.session_state:
            allowed = ['placement','category','category priority','placement priority','minimum spend','maximum spend','recommended budget']
            ed = current[[c for c in allowed if c in current.columns]].copy()
            ed['__key__'] = _norm_key(ed['placement'])
            st.session_state.editor_df = ed

        col_up, _ = st.columns([1,3])  # compact uploader at ~1/4 width
        with col_up:
            uploaded = st.file_uploader('Drag and drop file here (CSV/XLSX)', type=['csv','xlsx'])
        if uploaded is not None:
            try:
                upd = pd.read_csv(uploaded) if uploaded.name.lower().endswith('.csv') else pd.read_excel(uploaded, sheet_name=None)
                # allow both single-sheet and our multi-sheet exports
                if isinstance(upd, dict):
                    upd_df = upd.get('Split by Placement', None)
                    meta_sheet = upd.get('_meta', None)
                else:
                    upd_df = upd
                    meta_sheet = None
                if upd_df is None or 'placement' not in upd_df.columns:
                    st.error("Во входном файле не найден лист 'Split by Placement' или столбец 'placement'.")
                else:
                    ed = st.session_state.editor_df.copy()
                    upd_df['__key__'] = _norm_key(upd_df['placement'])
                    upd_df = upd_df.drop_duplicates('__key__', keep='last').set_index('__key__')
                    for col in ['category priority','placement priority','minimum spend','maximum spend','recommended budget']:
                        if col in upd_df.columns and col in ed.columns:
                            mask = ed['__key__'].isin(upd_df.index)
                            ed.loc[mask, col] = upd_df.loc[ed.loc[mask, '__key__'], col].values
                    st.session_state.editor_df = ed
                    st.success('Файл применён к текущей таблице.')
                    if meta_sheet is not None:
                        st.caption('Метаданные загружены.')

            except Exception as e:
                st.error(f'Не удалось прочитать/применить файл: {e}')

        disabled_cols = ['placement','category']
        edited = st.data_editor(
            st.session_state.editor_df.drop(columns=['__key__']),
            use_container_width=True, num_rows='fixed',
            disabled=disabled_cols,
            column_config={
                'recommended budget': st.column_config.NumberColumn('recommended budget', format='%.4f'),
                'minimum spend': st.column_config.NumberColumn('minimum spend', format='%d'),
                'maximum spend': st.column_config.NumberColumn('maximum spend', format='%d'),
                'category priority': st.column_config.NumberColumn('category priority', format='%d'),
                'placement priority': st.column_config.NumberColumn('placement priority', format='%d')
            },
            key='calc_editor'
        )
        edited['__key__'] = _norm_key(edited['placement'])
        st.session_state.editor_df = edited

        # Buttons line (Save/Cancel together) and export buttons below together
        bsave, bcancel = st.columns([1,1])
        with bsave:
            if st.button('✅ Сохранить и пересчитать', use_container_width=True):
                # Apply edited to base and re-allocate
                df_in = base.copy()
                if '__key__' not in df_in.columns:
                    df_in['__key__'] = _norm_key(df_in['placement'])
                editable_cols = [c for c in ['category priority','placement priority','minimum spend','maximum spend'] if c in edited.columns]
                df_in = df_in.drop(columns=editable_cols, errors='ignore').merge(
                    edited[['__key__'] + editable_cols], on='__key__', how='left'
                )
                df_in = df_in.drop(columns=['__key__'], errors='ignore')
                # Same category filter as main
                try:
                    df_in, _ = filter_by_categories(df_in, st.session_state.cat_order)
                except Exception:
                    pass
                # Re-allocate
                new_res, new_sum, new_margin, meta2 = allocate_budget(
                    df_in, total_budget=st.session_state.total_budget, alpha=st.session_state.alpha,
                    beta=st.session_state.beta, other_share=st.session_state.other_share, use_gates=None
                )
                st.session_state.df_result = new_res
                st.session_state.summary = new_sum
                st.session_state.total_margin = new_margin
                # Refresh editor table with new values
                allowed = ['placement','category','category priority','placement priority','minimum spend','maximum spend','recommended budget']
                ed2 = new_res[[c for c in allowed if c in new_res.columns]].copy()
                ed2['__key__'] = _norm_key(ed2['placement'])
                st.session_state.editor_df = ed2
                st.success(f'Бюджет успешно пересчитан: {st.session_state.total_budget:.2f} млн ₽ (100%)')

        with bcancel:
            st.button('⬅️ Отмена', use_container_width=True, on_click=lambda: st.session_state.update(mode='calculate'))

        # Exports from EDIT (side-by-side) — edited=True
        cur_df = st.session_state.get('df_result', edited)
        cur_sum = st.session_state.get('summary', pd.DataFrame(columns=['category','recommended budget','share_%']))
        meta_export = {
            'total_budget': st.session_state.total_budget,
            'alpha': st.session_state.alpha,
            'beta': st.session_state.beta,
            'other_share': st.session_state.other_share,
            'categories_order': ','.join(st.session_state.cat_order),
            'blacklist': ','.join(st.session_state.bl_selected),
            'mode': 'edit_result'
        }
        csv_bytes, xls_bytes, csv_name, xls_name = _export_tables(cur_df, cur_sum, meta_export, edited=True)
        e1, _, e2 = st.columns([1,0.05,1])
        with e1:
            st.download_button('💾 Download Result (CSV)', data=csv_bytes, file_name=csv_name, mime='text/csv', use_container_width=True, key='dl_edit_csv')
        with e2:
            st.download_button('💾 Download Result (Excel)', data=xls_bytes, file_name=xls_name,
                               mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', use_container_width=True, key='dl_edit_xlsx')
