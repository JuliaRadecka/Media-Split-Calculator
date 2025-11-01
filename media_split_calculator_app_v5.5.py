# -*- coding: utf-8 -*-
# media_split_calculator_app_v5.4_fixed_blacklist_edit.py
# (see comments at top for changes)
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import re

def _norm_key(series):
    s = series.astype(str).str.replace("\u00A0", " ", regex=False).str.replace("\t", " ", regex=False)
    return s.str.strip().str.lower()

def apply_editor_to_base(full_base, editor_df, cols_to_apply=None):
    full_base = full_base.copy()
    full_base["__key__"] = _norm_key(full_base["placement"])
    editor = editor_df.copy()
    if "__key__" not in editor.columns:
        editor["__key__"] = _norm_key(editor["placement"])
    editor = editor.drop_duplicates("__key__", keep="last").set_index("__key__")
    if cols_to_apply is None:
        cols_to_apply = ["category priority","placement priority","minimum spend","maximum spend","recommended budget"]
    cols_to_apply = [c for c in cols_to_apply if c in full_base.columns and c in editor.columns]
    if not cols_to_apply:
        full_base.drop(columns=["__key__"], inplace=True, errors="ignore")
        return full_base
    mask = full_base["__key__"].isin(editor.index)
    for col in cols_to_apply:
        full_base.loc[mask, col] = editor.loc[full_base.loc[mask, "__key__"], col].values
    full_base.drop(columns=["__key__"], inplace=True, errors="ignore")
    return full_base


def _ensure_row_ids(df):
    df = df.copy()
    if '__id' not in df.columns:
        df['__id'] = np.arange(len(df), dtype=np.int64)
    return df

def _norm_text(s):
    if s is None: return ""
    s = str(s).replace("\u00A0"," ").replace("\u2009"," ")
    s = re.sub(r"\s+"," ", s).strip().lower()
    s = s.replace("–","-").replace("—","-").replace("−","-")
    return s

def _bi_key(df):
    a = df['placement'].map(_norm_text) if 'placement' in df.columns else pd.Series([""]*len(df))
    b = df['category'].map(_norm_text) if 'category' in df.columns else pd.Series([""]*len(df))
    return a + "|" + b
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

def apply_platform_bounds(df, bounds):
    df = df.copy()
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
    try:
        summary = _ensure_other_summary(summary, other_budget if 'other_budget' in locals() else None)
    except Exception:
        pass
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

st.set_page_config(page_title='Media Split Calculator v5.4', layout='wide')
st.title('📊 Media Split Calculator — v5.4')

FILE_PATH = 'калькулятор.xlsx'
try:
    df = pd.read_excel(FILE_PATH)
except Exception:
    st.stop()

st.subheader('⚙️ Calculation Parameters')
col1, col2, col3, col4 = st.columns(4)
with col1:
    total_budget = st.number_input('Total Budget (mln ₽)', min_value=0.0, value=240.0, step=10.0)
    st.session_state.total_budget_cache = float(total_budget)
with col2:
    alpha = st.slider('α — Agency Profit Weight', 1.0, 2.5, 1.6, 0.1)
    st.session_state.alpha_cache = float(alpha)
with col3:
    beta = st.slider('β — Client Priority Weight', 0.5, 2.0, 1.0, 0.1)
    st.session_state.beta_cache = float(beta)
with col4:
    other_share = st.slider('Free Float Share (%)', 0.0, 30.0, 10.0, 1.0)
    st.session_state.other_share_cache = float(other_share)

st.markdown('**Platform Budget (mln ₽, min/max) — optional**')
p1, p2, p3, p4 = st.columns(4)
with p1:
    st.caption('Yandex')
    y_min = st.number_input('min (mln ₽)', key='y_min', value=0.0, step=10.0)
    y_max = st.number_input('max (mln ₽)', key='y_max', value=0.0, step=10.0)
with p2:
    st.caption('DA')
    da_min = st.number_input('min (mln ₽)', key='da_min', value=0.0, step=10.0)
    da_max = st.number_input('max (mln ₽)', key='da_max', value=0.0, step=10.0)
with p3:
    st.caption('VK')
    vk_min = st.number_input('min (mln ₽)', key='vk_min', value=0.0, step=10.0)
    vk_max = st.number_input('max (mln ₽)', key='vk_max', value=0.0, step=10.0)
with p4:
    st.caption('MTS')
    mts_min = st.number_input('min (mln ₽)', key='mts_min', value=0.0, step=10.0)
    mts_max = st.number_input('max (mln ₽)', key='mts_max', value=0.0, step=10.0)

platform_bounds = {
    'yandex': {'min': y_min,  'max': y_max},
    'da':     {'min': da_min, 'max': da_max},
    'vk':     {'min': vk_min, 'max': vk_max},
    'mts':    {'min': mts_min,'max': mts_max},
}

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

st.markdown('**Placements — Black List (optional)**')
plc_series = df['placement'].dropna().astype(str).map(lambda s: s.strip()).replace({'': np.nan, 'None': np.nan, 'none': np.nan, 'nan': np.nan})
all_placements = sorted(plc_series.dropna().unique().tolist())
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

if 'mode' not in st.session_state:
    st.session_state.mode = 'default'
if st.session_state.mode in ['default','edit']:
    cA, cB = st.columns(2)
    with cA:  calc_clicked = st.button('🧮 Calculate')
    with cB:  edit_clicked = st.button('✏️ Edit Input Data')
    if calc_clicked: st.session_state.mode = 'calculate'
    if edit_clicked: st.session_state.mode = 'edit'

if st.session_state.mode == 'edit':
    st.subheader('✏️ Edit Input Data')
    edited_df = st.data_editor(df, num_rows='dynamic', use_container_width=True, key='edit_table')
    if st.button('⬆️ Back to Main Menu'):
        st.session_state.mode = 'default'
        st.session_state.edited_df = edited_df

if st.session_state.mode == 'edit_result':
    if st.session_state.get('last_recalc_text'):
        st.success(st.session_state['last_recalc_text'])
    st.subheader('✏️ Edit Calculated Table')

    base = st.session_state.get('last_calc_base')
    current = st.session_state.get('df_result')

    if (base is None or (hasattr(base, 'empty') and base.empty)) and (current is None or current.empty):
        st.info('Нет сохранённого результата. Сначала выполните расчёт.')
    else:
        if "editor_run_id" not in st.session_state: st.session_state.editor_run_id = 0
        if "run_id" not in st.session_state:       st.session_state.run_id = 1
        if st.session_state.get("editor_run_id", 0) != st.session_state.get("run_id", 0):
            source = current if current is not None else base
            allowed = ['__id','placement','category','category priority','placement priority','minimum spend','maximum spend','recommended budget']
            editor_df = source[[c for c in allowed if c in source.columns]].copy()
            editor_df["__key__"] = _norm_key(editor_df["placement"])
            st.session_state.editor_df = editor_df
            st.session_state.original_rb = editor_df.set_index("__key__")['recommended budget'].copy() if 'recommended budget' in editor_df.columns else None
            st.session_state.editor_run_id = st.session_state.run_id

        editor_df = st.session_state.editor_df.copy()
        st.caption('Редактируйте приоритет площадки и min/max. "placement" и "category" недоступны для редактирования. При загрузке файла его данные применятся к текущей таблице.')

        uploaded = st.file_uploader('Upload a results file (.xlsx) — uses _meta to restore settings', type=['xlsx'])
        if uploaded is not None:
            try:
                upd = pd.read_csv(uploaded) if uploaded.name.lower().endswith('.csv') else pd.read_excel(uploaded)
                if 'placement' not in upd.columns:
                    st.error("Во входном файле нет столбца 'placement'. Правки не применены.")
                else:
                    upd['__key__'] = _norm_key(upd['placement'])
                    upd = upd.drop_duplicates('__key__', keep='last').set_index('__key__')
                    editable_cols = [c for c in ['category priority','placement priority','minimum spend','maximum spend','recommended budget'] if c in upd.columns and c in editor_df.columns]
                    mask = editor_df['__key__'].isin(upd.index)
                    for col in editable_cols:
                        editor_df.loc[mask, col] = upd.loc[editor_df.loc[mask, '__key__'], col].values
                    st.session_state.editor_df = editor_df
                    st.success('Файл применён к текущей таблице.')
            except Exception as e:
                st.error(f'Не удалось прочитать/применить файл: {e}')

        disabled_cols = ['__id','placement','category']
        column_cfg = {
            'recommended budget': st.column_config.NumberColumn('recommended budget', format='%.4f'),
            'minimum spend': st.column_config.NumberColumn('minimum spend', format='%d'),
            'maximum spend': st.column_config.NumberColumn('maximum spend', format='%d'),
            'category priority': st.column_config.NumberColumn('category priority', format='%d'),
            'placement priority': st.column_config.NumberColumn('placement priority', format='%d')
        }
        edited = st.data_editor(
            editor_df.drop(columns=['__key__']),
            use_container_width=True, num_rows='fixed',
            disabled=disabled_cols, column_config=column_cfg,
            key='editor_calc_table_v2'
        )
        edited['__key__'] = _norm_key(edited['placement'])
        st.session_state.editor_df = edited

        c1, c2 = st.columns([1,1])
        with c1:
            if st.button('✅ Save & Recalculate'):
                edited_df = edited.copy()
                for _c in ['category priority','placement priority','minimum spend','maximum spend']:
                    if _c in edited_df.columns:
                        edited_df[_c] = pd.to_numeric(edited_df[_c], errors='coerce')
                base_df = st.session_state.get('last_calc_base').copy()
                if '__key__' not in edited_df.columns: edited_df['__key__'] = _norm_key(edited_df['placement'])
                if '__key__' not in base_df.columns:  base_df['__key__']  = _norm_key(base_df['placement'])
                editable_cols = [c for c in ['category priority','placement priority','minimum spend','maximum spend'] if c in edited_df.columns]
                base_df = base_df.drop(columns=editable_cols, errors='ignore').merge(
                    edited_df[['__key__']+editable_cols], on='__key__', how='left'
                )
                st.session_state['last_calc_base'] = base_df
                full_base = base.copy() if base is not None else st.session_state.get('edited_df', df).copy()
                full_base = apply_editor_to_base(full_base, edited,
                    cols_to_apply=['category priority','placement priority','minimum spend','maximum spend'])

                total_budget = float(st.session_state.get('total_budget_cache', 240.0))
                alpha        = float(st.session_state.get('alpha_cache', 1.6))
                beta         = float(st.session_state.get('beta_cache', 1.0))
                other_share  = float(st.session_state.get('other_share_cache', 10.0))

                df_in = full_base.copy()
                try:
                    picked = st.session_state.get('cat_order', [])
                    df_in, _ = filter_by_categories(df_in, picked)
                except Exception:
                    pass
                for col in ['placement','category','category priority','placement priority','minimum spend','maximum spend','recommended budget']:
                    if col not in df_in.columns: df_in[col] = np.nan

                new_df_result, new_summary, new_margin, meta = allocate_budget(
                    df_in, total_budget=total_budget, alpha=alpha, beta=beta, other_share=other_share, use_gates=None
                )

                try:
                    if '__key__' not in new_df_result.columns:
                        new_df_result['__key__'] = _norm_key(new_df_result['placement'])
                    new_df_result = pd.merge(
                        edited_df[['__key__','placement','category','category priority','placement priority','minimum spend','maximum spend']],
                        new_df_result[['__key__','recommended budget']],
                        on='__key__', how='left'
                    )
                    new_df_result['recommended budget'] = new_df_result['recommended budget'].fillna(0.0)
                except Exception:
                    pass

                st.session_state.df_result   = new_df_result
                st.session_state.summary     = new_summary
                st.session_state.total_margin= new_margin
                st.session_state.run_id     += 1
                new_editor = new_df_result[['placement','category','category priority','placement priority','minimum spend','maximum spend','recommended budget']].copy()
                new_editor["__key__"] = _norm_key(new_editor["placement"])
                st.session_state.editor_df   = new_editor
                st.session_state.original_rb = new_editor.set_index("__key__")['recommended budget'].copy()

                st.session_state['last_recalc_text'] = f'Бюджет успешно пересчитан: {float(total_budget):.2f} млн ₽ (100%)'
                try:
                    if new_margin is not None:
                        st.markdown(f"### 💰 Общая маржинальность сплита: **{float(new_margin):.2f}%**")
                except Exception:
                    pass
                if meta.get('scaled_mins'):
                    coef = meta.get('scale_coef', 1.0)
                    st.info(f'Minimum spend был пропорционально масштабирован коэффициентом {coef:.4f}, чтобы уложиться в бюджет.')

        with c2:
            if st.button('⬅️ Back'):
                st.session_state.mode = 'calculate'

        cur_df = st.session_state.get('df_result')
        cur_sum = st.session_state.get('summary')
        if cur_df is not None and cur_sum is not None:
            _csv = cur_df.to_csv(index=False).encode('utf-8')
            dlA, dlB = st.columns([1,1])
            with dlA:
                st.download_button('💾 Download Result (CSV)',
                                   data=_csv,
                                   file_name='media_split_result_v5_edited.csv',
                                   mime='text/csv',
                                   key='dl_edit_csv')
            with dlB:
                _xls = BytesIO()
                try:
                    with pd.ExcelWriter(_xls, engine='xlsxwriter') as writer:
                        cur_df.to_excel(writer, index=False, sheet_name='Split by Placement')
                        cur_sum.to_excel(writer, index=False, sheet_name='Summary by Category')
                except ModuleNotFoundError:
                    with pd.ExcelWriter(_xls, engine='openpyxl') as writer:
                        export_df = cur_df.drop(columns=[c for c in ['commercial priority','W','available'] if c in cur_df.columns], errors='ignore')
                        export_df.to_excel(writer, index=False, sheet_name='Split by Placement')
                        cur_sum.to_excel(writer, index=False, sheet_name='Summary by Category')
                        meta = {
                            'total_budget': float(st.session_state.get('total_budget_cache', 240.0)),
                            'alpha': float(st.session_state.get('alpha_cache', 1.6)),
                            'beta': float(st.session_state.get('beta_cache', 1.0)),
                            'other_share': float(st.session_state.get('other_share_cache', 10.0)),
                            'selected_categories': st.session_state.get('cat_order', []),
                            'blacklist': st.session_state.get('bl_selected', []),
                        }
                        pd.DataFrame([meta]).to_excel(writer, index=False, sheet_name='_meta')
                _xls.seek(0)
                st.download_button('💾 Download Result (Excel)',
                                   data=_xls.getvalue(),
                                   file_name='media_split_result_v5_edited.xlsx',
                                   mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                   key='dl_edit_xlsx')

if st.session_state.mode == 'calculate':
    df_to_use = st.session_state.get('edited_df', df).copy()
    df_to_use = apply_blacklist(df_to_use, blacklist)
    requested_groups = {k: v for k, v in platform_bounds.items() if float(v.get('min',0) or 0) > 0}
    df_to_use = apply_platform_bounds(df_to_use, platform_bounds)
    df_to_use, order_map = filter_by_categories(df_to_use, st.session_state.cat_order)
    df_to_use = _ensure_row_ids(df_to_use)
    st.session_state.base_df = df_to_use.copy()
    for key in requested_groups.keys():
        mask = df_to_use['placement'].astype(str).str.lower().str.contains(key)
        if not mask.any():
            st.warning(f"Для платформы {key.upper()} после фильтров не осталось площадок; group min не применён.")
    df_result, summary, total_margin, meta = allocate_budget(
        df_to_use,
        total_budget=float(total_budget),
        alpha=float(alpha),
        beta=float(beta),
        other_share=float(other_share),
        use_gates=None
    )
    try:
        _left = st.session_state.base_df.copy(); _left['__bikey'] = _bi_key(_left)
        df_result['__bikey'] = _bi_key(df_result)
        df_result = df_result.merge(_left[['__bikey','__id']], on='__bikey', how='left').drop(columns=['__bikey'])
    except Exception:
        pass
    st.success(f'Бюджет успешно распределён: {float(total_budget):.2f} млн ₽ (100%)')
    if meta.get('scaled_mins'):
        st.info(f'Minimum spend был пропорционально масштабирован коэффициентом {meta.get("scale_coef",1.0):.4f}, чтобы уложиться в бюджет.')
    if order_map and 'category' in df_result.columns:
        df_result['_cat_ord'] = df_result['category'].astype(str).str.lower().map(order_map).fillna(1e6)
        df_result = df_result.sort_values(by=['_cat_ord', 'recommended budget'],
                                          ascending=[True, False]).drop(columns=['_cat_ord'])
    st.subheader('📈 Recommended Split by Placement')
    all_cols = ['placement','category','recommended budget','category priority','placement priority','minimum spend','maximum spend']
    available_cols = [c for c in all_cols if c in df_result.columns]
    table_df = df_result[available_cols].copy()
    total_row = {col: '' for col in available_cols}
    if 'placement' in total_row: total_row['placement'] = 'ИТОГО'
    if 'recommended budget' in total_row: total_row['recommended budget'] = table_df['recommended budget'].sum()
    table_df = pd.concat([table_df, pd.DataFrame([total_row])], ignore_index=True)
    base_order = [c for c in ['placement','category','recommended budget'] if c in available_cols]
    st.dataframe(table_df, use_container_width=True, column_order=base_order)
    st.subheader('📊 Summary by Category')
    sum_df = summary.copy()
    if 'recommended budget' not in sum_df.columns: sum_df['recommended budget'] = 0.0
    if 'category' not in sum_df.columns:           sum_df['category'] = pd.Series(dtype='object')
    if 'share_%' not in sum_df.columns:            sum_df['share_%'] = 0.0
    tot = {'category':'ИТОГО','recommended budget': float(sum_df['recommended budget'].sum()), 'share_%': 100.0}
    sum_df = pd.concat([sum_df, pd.DataFrame([tot])], ignore_index=True)
    st.dataframe(sum_df.round(2), use_container_width=True)
    if total_margin is not None:
        st.markdown(f"### 💰 Общая маржинальность сплита: **{float(total_margin):.2f}%**")

if 'df_result' in locals():
    csv_bytes = df_result.drop(columns=[c for c in ['commercial priority','W','available'] if c in df_result.columns], errors='ignore').to_csv(index=False).encode('utf-8')
    dl1, sp1, dl2, sp2, dl3 = st.columns([1,0.02,1,0.02,1])
    with dl1:
        st.download_button('💾 Download Result (CSV)',
                           data=csv_bytes,
                           file_name='media_split_result_v5.csv',
                           mime='text/csv', key='dl_calc_csv_bottom')
    with dl2:
        xls = BytesIO()
        try:
            with pd.ExcelWriter(xls, engine='xlsxwriter') as writer:
                df_result.to_excel(writer, index=False, sheet_name='Split by Placement')
                summary.to_excel(writer, index=False, sheet_name='Summary by Category')
        except ModuleNotFoundError:
            with pd.ExcelWriter(xls, engine='openpyxl') as writer:
                df_result.to_excel(writer, index=False, sheet_name='Split by Placement')
                summary.to_excel(writer, index=False, sheet_name='Summary by Category')
        xls.seek(0)
        st.download_button('💾 Download Result (Excel)',
                           data=xls.getvalue(),
                           file_name='media_split_result_v5.xlsx',
                           mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                           key='dl_calc_xlsx_bottom')
    with dl3:
        if st.button('✏️ Edit Calculated Table'):
            st.session_state.df_result = df_result.copy()
            st.session_state.summary = summary.copy()
            st.session_state.total_margin = total_margin
            st.session_state.last_calc_base = st.session_state.get('edited_df', df).copy()
            st.session_state.mode = 'edit_result'
