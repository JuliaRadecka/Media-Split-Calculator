
# -*- coding: utf-8 -*-
# Media Split Calculator — v5.4 (A–F UI/Export/Meta patch)
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

def _norm_key(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace("\u00A0", " ", regex=False).str.replace("\t", " ", regex=False)
    return s.str.strip().str.lower()

def apply_blacklist(df: pd.DataFrame, blacklist):
    if not blacklist:
        return df.copy()
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
    if df_valid.empty:
        total_margin = 0.0
    else:
        df_valid['contribution'] = df_valid['recommended budget'] * df_valid['commercial priority']
        total_margin = (df_valid['contribution'].sum() / df_valid['recommended budget'].sum()) * 100.0
    return df_final, summary, total_margin, meta

def _export(df_result, summary, meta_dict, edited=False):
    drop_cols = [c for c in ['commercial priority','W','available'] if c in df_result.columns]
    export_df = df_result.drop(columns=drop_cols, errors='ignore').copy()
    csv_bytes = export_df.to_csv(index=False).encode('utf-8')
    csv_name  = f"split_by_placement{'_edited' if edited else ''}.csv"
    from io import BytesIO
    xls = BytesIO()
    with pd.ExcelWriter(xls, engine='xlsxwriter') as writer:
        export_df.to_excel(writer, index=False, sheet_name='Split by Placement')
        summary.to_excel(writer, index=False, sheet_name='Summary by Category')
        pd.DataFrame([meta_dict]).to_excel(writer, index=False, sheet_name='meta')
    xls.seek(0)
    xls_bytes = xls.getvalue()
    xls_name  = f"media_split_result_v5{'_edited' if edited else ''}.xlsx"
    return csv_bytes, xls_bytes, csv_name, xls_name

# ----------------- App -----------------
st.set_page_config(page_title='Media Split Calculator v5.4', layout='wide')
st.title('📊 Media Split Calculator — v5.4')

if 'mode' not in st.session_state:
    st.session_state.mode = 'default'

FILE_PATH = 'калькулятор.xlsx'
try:
    base_df = pd.read_excel(FILE_PATH)
except Exception:
    st.error('Не удалось прочитать исходные данные (калькулятор.xlsx). Поместите файл рядом с приложением.')
    st.stop()

st.subheader('⚙️ Calculation Parameters')
c1, c2, c3, c4 = st.columns(4)
with c1:
    total_budget = st.number_input('Total Budget (mln ₽)', min_value=0.0, value=300.0, step=10.0)
    st.session_state.total_budget_cache = float(total_budget)
with c2:
    alpha = st.slider('α — Agency Profit Weight', 1.0, 2.5, 1.6, 0.1, key='alpha_slider')
    st.session_state.alpha_cache = float(alpha)
with c3:
    beta = st.slider('β — Client Priority Weight', 0.5, 2.0, 1.0, 0.1, key='beta_slider')
    st.session_state.beta_cache = float(beta)
with c4:
    other_share = st.slider('Free Float Share (%)', 0.0, 30.0, 10.0, 1.0, key='other_share_slider')
    st.session_state.other_share_cache = float(other_share)

all_cats = ['CTV','OLV PREM','Media','PRG','SOCIAL','ECOM','MOB','Geomedia','Geoperfom','Promopages','РСЯ','Direct','CPA','THEM']
if 'cat_order' not in st.session_state:
    st.session_state.cat_order = []
st.markdown('**Category Priorities — optional**')
ccols = st.columns(len(all_cats))
def _toggle(cat_label):
    if cat_label in st.session_state.cat_order:
        st.session_state.cat_order.remove(cat_label)
    else:
        st.session_state.cat_order.append(cat_label)
for i, cat in enumerate(all_cats):
    ckey = f'cat_{cat.replace(" ", "_")}'
    ccols[i].checkbox(cat, key=ckey, value=(cat in st.session_state.cat_order),
                      on_change=_toggle, args=(cat,))

st.markdown('**Placements — Black List (optional)**')
clean_pl = base_df['placement'].dropna().astype(str).map(str.strip)
clean_pl = clean_pl.replace({'': np.nan, 'None': np.nan, 'none': np.nan, 'nan': np.nan})
all_placements = sorted(clean_pl.dropna().unique().tolist())
if 'bl_selected' not in st.session_state:
    st.session_state.bl_selected = []
bl = st.multiselect('Exclude placements from calculation', options=all_placements,
                    default=st.session_state.bl_selected, key='bl_ms')
st.session_state.bl_selected = bl

st.markdown('---')

act1, act2 = st.columns(2)
with act1:
    if st.button('🧮 Считать сейчас', use_container_width=True):
        st.session_state.mode = 'calculate'
with act2:
    if st.button('📂 Перейти к правкам (загрузить файл)', use_container_width=True):
        st.session_state.mode = 'edit_result'
        for k in ['editor_df','df_result','summary','total_margin','last_calc_base']:
            st.session_state.pop(k, None)

if st.session_state.mode == 'calculate':
    df = st.session_state.get('edited_df', base_df).copy()
    df = apply_blacklist(df, st.session_state.bl_selected)
    df, order_map = filter_by_categories(df, st.session_state.cat_order)
    df_result, summary, total_margin, meta = allocate_budget(
        df, total_budget=st.session_state.total_budget_cache,
        alpha=st.session_state.alpha_cache, beta=st.session_state.beta_cache,
        other_share=st.session_state.other_share_cache, use_gates=None
    )
    st.success(f'Бюджет успешно распределён: {st.session_state.total_budget_cache:.2f} млн ₽ (100%)')
    if meta.get('scaled_mins'):
        st.info(f'Minimum spend был масштабирован коэффициентом {meta.get("scale_coef",1.0):.4f} для укладки в бюджет.')
    st.subheader('📈 Recommended Split by Placement')
    show_cols = [c for c in ['placement','category','recommended budget','category priority','placement priority','minimum spend','maximum spend'] if c in df_result.columns]
    st.dataframe(df_result[show_cols], use_container_width=True)
    st.subheader('📊 Summary by Category')
    st.dataframe(summary.round(2), use_container_width=True)
    st.markdown(f"### 💰 Общая маржинальность сплита: **{float(total_margin):.2f}%**")
    meta_export = {
        'total_budget': st.session_state.total_budget_cache,
        'alpha': st.session_state.alpha_cache,
        'beta': st.session_state.beta_cache,
        'other_share': st.session_state.other_share_cache,
        'categories_order': ','.join(st.session_state.cat_order),
        'blacklist': ','.join(st.session_state.bl_selected),
        'mode': 'calculate'
    }
    csv_bytes, xls_bytes, csv_name, xls_name = _export(df_result, summary, meta_export, edited=False)
    c1, s1, c2, s2, c3 = st.columns([1,0.05,1,0.05,1])
    with c1:
        st.download_button('💾 Download Results (CSV)', data=csv_bytes, file_name=csv_name, mime='text/csv', use_container_width=True)
    with c2:
        st.download_button('💾 Download Results (Excel)', data=xls_bytes, file_name=xls_name,
                           mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', use_container_width=True)
    st.write("")
    if st.button('✏️ Edit Calculated Table', use_container_width=False):
        st.session_state.mode = 'edit_result'
        st.session_state.df_result = df_result.copy()
        st.session_state.summary = summary.copy()
        st.session_state.total_margin = total_margin
        st.session_state.last_calc_base = st.session_state.get('edited_df', base_df).copy()

if st.session_state.mode == 'edit_result':
    st.subheader('✏️ Edit Calculated Table')
    base = st.session_state.get('last_calc_base', None)
    current = st.session_state.get('df_result', None)
    up_col, table_col = st.columns([1,3])
    with up_col:
        uploaded = st.file_uploader('Загрузить файл с результатами (CSV/XLSX)', type=['csv','xlsx'])
    if uploaded is not None:
        try:
            if uploaded.name.lower().endswith('.csv'):
                loaded = pd.read_csv(uploaded)
                meta_sheet = None
            else:
                loaded_sheets = pd.read_excel(uploaded, sheet_name=None)
                loaded = loaded_sheets.get('Split by Placement', None)
                meta_sheet = loaded_sheets.get('meta', None)
            if loaded is None or 'placement' not in loaded.columns:
                st.error("В файле не найден лист/таблица 'Split by Placement' со столбцом 'placement'.")
            else:
                st.session_state.editor_df = loaded.copy()
                if meta_sheet is not None:
                    try:
                        meta_row = meta_sheet.iloc[0].to_dict()
                        st.session_state.total_budget_cache = float(meta_row.get('total_budget', st.session_state.total_budget_cache))
                        st.session_state.alpha_cache = float(meta_row.get('alpha', st.session_state.alpha_cache))
                        st.session_state.beta_cache = float(meta_row.get('beta', st.session_state.beta_cache))
                        st.session_state.other_share_cache = float(meta_row.get('other_share', st.session_state.other_share_cache))
                        cats = meta_row.get('categories_order', '')
                        st.session_state.cat_order = [c for c in str(cats).split(',') if c]
                        bls = meta_row.get('blacklist', '')
                        st.session_state.bl_selected = [p for p in str(bls).split(',') if p]
                        st.info('Метаданные восстановлены из файла.')
                    except Exception:
                        st.caption('Метаданные в файле не прочитались — работаем только с таблицей.')
                st.success('Текущая таблица в редакторе взята из загруженного файла.')
        except Exception as e:
            st.error(f'Не удалось прочитать файл: {e}')
    if 'editor_df' not in st.session_state:
        if current is not None:
            allowed = ['placement','category','category priority','placement priority','minimum spend','maximum spend','recommended budget']
            st.session_state.editor_df = current[[c for c in allowed if c in current.columns]].copy()
        else:
            st.session_state.editor_df = pd.DataFrame(columns=['placement','category','category priority','placement priority','minimum spend','maximum spend','recommended budget'])
    with table_col:
        ed = st.data_editor(
            st.session_state.editor_df,
            use_container_width=True, num_rows='dynamic',
            disabled=['placement','category'],
            column_config={
                'recommended budget': st.column_config.NumberColumn('recommended budget', format='%.4f'),
                'minimum spend': st.column_config.NumberColumn('minimum spend', format='%d'),
                'maximum spend': st.column_config.NumberColumn('maximum spend', format='%d'),
                'category priority': st.column_config.NumberColumn('category priority', format='%d'),
                'placement priority': st.column_config.NumberColumn('placement priority', format='%d')
            },
            key='calc_editor'
        )
        st.session_state.editor_df = ed.copy()
    a1, a2 = st.columns(2)
    with a1:
        if st.button('✅ Сохранить и пересчитать', use_container_width=True):
            df_in = st.session_state.editor_df.copy()
            try:
                df_filt, _ = filter_by_categories(df_in, st.session_state.cat_order)
            except Exception:
                df_filt = df_in.copy()
            new_res, new_sum, new_margin, meta2 = allocate_budget(
                df_filt,
                total_budget=st.session_state.total_budget_cache,
                alpha=st.session_state.alpha_cache,
                beta=st.session_state.beta_cache,
                other_share=st.session_state.other_share_cache,
                use_gates=None
            )
            st.session_state.df_result = new_res
            st.session_state.summary = new_sum
            st.session_state.total_margin = new_margin
            st.success(f'Бюджет успешно пересчитан: {st.session_state.total_budget_cache:.2f} млн ₽ (100%)')
    with a2:
        if st.button('⬅️ Вернуться', use_container_width=True):
            st.session_state.mode = 'calculate'
    cur_df = st.session_state.get('df_result', st.session_state.editor_df.copy())
    cur_sum = st.session_state.get('summary', pd.DataFrame(columns=['category','recommended budget','share_%']))
    meta_export = {
        'total_budget': st.session_state.total_budget_cache,
        'alpha': st.session_state.alpha_cache,
        'beta': st.session_state.beta_cache,
        'other_share': st.session_state.other_share_cache,
        'categories_order': ','.join(st.session_state.cat_order),
        'blacklist': ','.join(st.session_state.bl_selected),
        'mode': 'edit_result'
    }
    csv_b, xls_b, csv_n, xls_n = _export(cur_df, cur_sum, meta_export, edited=True)
    d1, d2 = st.columns(2)
    with d1:
        st.download_button('💾 Download Result (CSV)', data=csv_b, file_name=csv_n, mime='text/csv', use_container_width=True, key='dl_edit_csv')
    with d2:
        st.download_button('💾 Download Result (Excel)', data=xls_b, file_name=xls_n,
                           mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', use_container_width=True, key='dl_edit_xlsx')
