
# -*- coding: utf-8 -*-
# Media Split Calculator — v5.5.0
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

APP_VERSION = "v5.5.0"
SCHEMA_VERSION = "2025-11-05.10"

def _norm_text(s: str) -> str:
    if s is None: return ""
    s = str(s).replace("\u00A0"," ").replace("\u2009"," ")
    s = re.sub(r"\s+"," ", s).strip().lower()
    return s.replace("–","-").replace("—","-").replace("−","-")

def make_stable_id(placement: str, category: str) -> np.int64:
    key = f"{_norm_text(placement)}|{_norm_text(category)}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return np.int64(int(h, 16))

def ensure_mode():
    st.session_state.setdefault('mode','filters')
    st.session_state.setdefault('edit_source', None)
    st.session_state.setdefault('_pending_recalc', False)
    st.session_state.setdefault('cat_order', [])

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

def filter_by_categories(df, picked):
    if not picked: return df.copy(), None
    picked_lower = [c.lower() for c in picked]
    mask = df['category'].astype(str).str.lower().isin(picked_lower) | (df['category'].astype(str).str.lower()=='other')
    df2 = df[mask].copy()
    order_map = {c.lower(): i for i,c in enumerate(picked, start=1)}
    order_map['other'] = len(picked)+1
    return df2, order_map

def apply_blacklist(df, blacklist):
    return df if not blacklist else df[~df['placement'].isin(blacklist)].copy()

def apply_platform_bounds(df, bounds):
    df = df.copy()
    if 'minimum spend' not in df: df['minimum spend'] = 0.0
    if 'maximum spend' not in df: df['maximum spend'] = 1e9
    for key, mm in (bounds or {}).items():
        if not isinstance(mm, dict): continue
        mn = float(mm.get('min',0) or 0); mx = float(mm.get('max',0) or 0)
        if mn<=0 and mx<=0: continue
        m = df['placement'].astype(str).str.lower().str.contains(str(key).lower())
        if mn>0: df.loc[m,'minimum spend'] = mn
        if mx>0: df.loc[m,'maximum spend'] = mx
    return df

def _ensure_other_summary(summary_df, other_budget):
    try:
        if other_budget is None: return summary_df
        cats = summary_df['category'].astype(str).str.lower()
        if 'other' not in set(cats):
            extra = pd.DataFrame([{'category':'other','recommended budget':float(other_budget)}])
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

    other_mask   = df['category'].astype(str).str.lower()=='other'
    other_budget = float(total_budget)*(float(other_share)/100.0)
    main_budget  = float(total_budget)-other_budget

    use_gates = bool((df['category priority'].notna().any() or df['placement priority'].notna().any())) if use_gates is None else use_gates
    gate_mask = (df['category priority']<=3) & (df['placement priority']<=2) if use_gates else True

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

def apply_category_priorities_from_order(df, order):
    if not order: return df
    df = df.copy()
    m = {c.lower(): i for i,c in enumerate(order, start=1)}
    mapped = df['category'].astype(str).str.lower().map(m)
    if 'category priority' in df.columns:
        fallback = pd.to_numeric(seriesize(df['category priority'], len(df)), errors='coerce')
        mapped = mapped.where(~mapped.isna(), fallback)
    mapped = seriesize(mapped, len(df)).fillna(5.0)
    df['category priority'] = mapped
    return df

def export_csv(df):
    return df.drop(columns=[c for c in ['commercial priority','W','available','__id'] if c in df.columns], errors='ignore').to_csv(index=False).encode('utf-8')

def export_excel(df_split, df_sum, base_df_with_ids):
    desired = ['placement','category','category priority','placement priority','minimum spend','maximum spend','recommended budget']
    split_df = df_split[[c for c in desired if c in df_split.columns]].copy()
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine='openpyxl') as w:
        split_df.to_excel(w, index=False, sheet_name='Split by Placement')
        df_sum.to_excel(w, index=False, sheet_name='Summary by Category')
        meta = pd.DataFrame([{
            'schema_version': SCHEMA_VERSION,
            'app_version': APP_VERSION,
            'exported_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_budget': float(st.session_state.get('total_budget_cache', 240.0)),
            'alpha': float(st.session_state.get('alpha_cache', 1.6)),
            'beta': float(st.session_state.get('beta_cache', 1.0)),
            'free_float_%': float(st.session_state.get('other_share_cache', 10.0)),
            'selected_categories': ','.join(st.session_state.get('cat_order', [])),
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
st.set_page_config(page_title='📊 Media Split Calculator v5.5.0', layout='wide')
st.title('📊 Media Split Calculator — v5.5.0')
FILE_PATH = 'калькулятор.xlsx'
try:
    src_df = pd.read_excel(FILE_PATH)
except Exception as e:
    st.error(f'Не удалось прочитать исходный файл {FILE_PATH}: {e}'); st.stop()

# Prepare CP lookup from catalog for later merge on upload
src_cp = src_df[['placement','category']].copy()
src_cp['commercial priority'] = pd.to_numeric(seriesize(src_df.get('commercial priority', 0.25), len(src_df)), errors='coerce').fillna(0.25)
src_cp['_plc'] = src_cp['placement'].astype(str).map(_norm_text)
src_cp['_cat'] = src_cp['category'].astype(str).map(_norm_text)
src_cp = src_cp.drop_duplicates(subset=['_plc','_cat'], keep='last')

ensure_mode()

def mark_for_recalc(): st.session_state._pending_recalc = True

def do_recalc():
    df0 = src_df.copy()
    df0 = apply_blacklist(df0, st.session_state.get('bl_selected', []))
    df0 = apply_platform_bounds(df0, st.session_state.get('platform_bounds', {}))
    picked = st.session_state.get('cat_order', [])
    df0, order_map = filter_by_categories(df0, picked)
    df0 = apply_category_priorities_from_order(df0, picked)
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
        df_res['_ord'] = df_res['category'].astype(str).str.lower().map(order_map).fillna(1e6)
        df_res = df_res.sort_values(by=['_ord','recommended budget'], ascending=[True, False]).drop(columns=['_ord'])
    st.session_state.df_result = df_res.copy()
    st.session_state.summary = summary.copy()
    st.session_state.total_margin = float(total_margin)
    st.session_state.base_df = ensure_stable_ids(df_res.copy())
    st.session_state._pending_recalc = False

# Controls (filters UI) — same as 5.4.9
if st.session_state.mode != 'edit':
    st.subheader('⚙️ Calculation Parameters')
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.number_input('Total Budget (mln ₽)', min_value=0.0,
                        value=float(st.session_state.get('total_budget_cache', 240.0)),
                        step=10.0, key='total_budget_cache')
    with c2:
        st.slider('α — Agency Profit Weight', 1.0, 2.5, float(st.session_state.get('alpha_cache', 1.6)), 0.1, key='alpha_cache')
    with c3:
        st.slider('β — Client Priority Weight', 0.5, 2.0, float(st.session_state.get('beta_cache', 1.0)), 0.1, key='beta_cache')
    with c4:
        st.slider('Free Float Share (%)', 0.0, 30.0, float(st.session_state.get('other_share_cache', 10.0)), 1.0, key='other_share_cache')

    st.markdown('**Platform Budget (mln ₽, min/max) — optional**')
    p1,p2,p3,p4 = st.columns(4)
    for label, min_k, max_k, col in [('Yandex','y_min','y_max',p1),('DA','da_min','da_max',p2),('VK','vk_min','vk_max',p3),('MTS','mts_min','mts_max',p4)]:
        with col:
            st.caption(label)
            st.number_input('min (mln ₽)', key=min_k, value=float(st.session_state.get(min_k,0.0)), step=10.0, on_change=mark_for_recalc)
            st.number_input('max (mln ₽)', key=max_k, value=float(st.session_state.get(max_k,0.0)), step=10.0, on_change=mark_for_recalc)

    st.session_state.platform_bounds = {
        'yandex': {'min': st.session_state.get('y_min',0.0), 'max': st.session_state.get('y_max',0.0)},
        'da':     {'min': st.session_state.get('da_min',0.0), 'max': st.session_state.get('da_max',0.0)},
        'vk':     {'min': st.session_state.get('vk_min',0.0), 'max': st.session_state.get('vk_max',0.0)},
        'mts':    {'min': st.session_state.get('mts_min',0.0), 'max': st.session_state.get('mts_max',0.0)},
    }

    st.markdown('**Category Priorities — optional**')
    cats = ['CTV','OLV PREM','Media','PRG','SOCIAL','ECOM','MOB','Geomedia','Geoperfom','Promopages','РСЯ','Direct','CPA','THEM']
    def _toggle_cat(cat_key):
        chosen = st.session_state.get(cat_key, False)
        label_raw = cat_key.replace('cat_','')
        label = next((c for c in cats if c.replace(' ','_')==label_raw), label_raw.replace('_',' ').upper())
        if chosen and label not in st.session_state.cat_order: st.session_state.cat_order.append(label)
        if (not chosen) and label in st.session_state.cat_order: st.session_state.cat_order.remove(label)
        mark_for_recalc()
    cc = st.columns(len(cats))
    ord_show = {c:i for i,c in enumerate(st.session_state.cat_order, start=1)}
    for i, cat in enumerate(cats):
        key = 'cat_'+cat.replace(' ','_')
        prefix = f"{ord_show.get(cat, '\u25A1')}  "
        cc[i].checkbox(prefix+cat, key=key, value=(cat in st.session_state.cat_order), on_change=_toggle_cat, args=(key,))

    st.markdown('**Placements — Black List (optional)**')
    plc_series = src_df['placement'].dropna().astype(str).map(lambda s:s.strip())
    all_plc = sorted(plc_series.unique().tolist())
    st.multiselect('Exclude placements from calculation', options=all_plc, key='bl_selected', on_change=mark_for_recalc)
    st.markdown('---')

# Modes
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
        tb = float(st.session_state.get('total_budget_cache', total_loaded))
        pct = (total_loaded/tb*100.0) if tb>0 else 100.0
        other = float(st.session_state.get('other_share_cache', 10.0))
        st.success(f"✅ Бюджет успешно распределён: {total_loaded:.2f} млн ₽ ({pct:.0f}%), Free Float Share ({other:.0f}%)")

        st.subheader('📈 Recommended Split by Placement')
        base_cols = ['placement','category','recommended budget']
        all_adv  = [c for c in ['category priority','placement priority','minimum spend','maximum spend'] if c in df_result.columns]
        column_order = base_cols
        table_df = df_result[base_cols + all_adv].copy()
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
            tot = {'category':'ИТОГО','recommended budget': float(sum_df['recommended budget'].sum()), 'share_%': 100.0}
            sum_df = pd.concat([sum_df, pd.DataFrame([tot])], ignore_index=True)
            st.dataframe(sum_df.round(2), use_container_width=True)

        st.markdown(f"### 💰 Общая маржинальность сплита: **{float(total_margin):.2f}%**")

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
    margin_box = st.empty()  # single place to render margin

    def _restore_cat_order_from_meta(meta_df):
        try:
            raw_series = seriesize(meta_df.get('selected_categories', pd.Series(dtype='object')))
            raw = raw_series.iloc[0] if len(raw_series)>0 else None
            if isinstance(raw, str) and raw.strip():
                if raw.strip().startswith('['): lst = ast.literal_eval(raw)
                else: lst = [x.strip() for x in raw.split(',') if x.strip()]
                if lst: st.session_state.cat_order = lst; return True
        except Exception: pass
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
            tb_for_banner = total_from_summary if (total_from_summary is not None and total_from_summary>0) else float(st.session_state.get('total_budget_cache', 0))
            pct = (total_loaded/tb_for_banner*100.0) if tb_for_banner>0 else 0.0
            other = float(st.session_state.get('other_share_cache', 10.0))
            if total_from_summary is not None and total_from_summary>0:
                st.session_state['total_budget_cache'] = total_from_summary

            st.info(f"📥 Бюджет успешно подгружен: {total_loaded:.2f} млн ₽ ({pct:.0f}%), Free Float Share ({other:.0f}%)")

            # correct margin from uploaded file (with CP merged from catalog)
            margin_box.markdown(f"### 💰 Общая маржинальность сплита: **{margin_from_current_budgets(base_df):.2f}%**")

            base_df = apply_category_priorities_from_order(base_df, st.session_state.get('cat_order', []))
            st.session_state.base_df = base_df.copy()
        except Exception as e:
            st.error(f'Не удалось прочитать файл: {e}')

    if waiting_for_file:
        if st.button('⬅️ Back'): st.session_state['edit_source']=None; st.session_state.mode='filters'; st.rerun()
        st.stop()

    if base_df is None and current_df is not None:
        st.session_state.base_df = ensure_stable_ids(apply_category_priorities_from_order(current_df.copy(), st.session_state.get('cat_order', [])))
        base_df = st.session_state.base_df

    if base_df is None:
        df0 = src_df.copy()
        df0 = apply_blacklist(df0, st.session_state.get('bl_selected', []))
        df0 = apply_platform_bounds(df0, st.session_state.get('platform_bounds', {}))
        picked = st.session_state.get('cat_order', [])
        df0, _ = filter_by_categories(df0, picked)
        base_df = ensure_stable_ids(apply_category_priorities_from_order(df0, picked))
        st.session_state.base_df = base_df.copy()

    editable_cols = ['category priority','placement priority','minimum spend','maximum spend','recommended budget']
    show_cols = ['__id','placement','category', *editable_cols]
    show_cols = [c for c in show_cols if c in base_df.columns]
    editor_df = base_df[show_cols].copy()

    edited = st.data_editor(editor_df, use_container_width=True, num_rows='fixed',
                            disabled=['__id','placement','category'],
                            column_config={'recommended budget': st.column_config.NumberColumn('recommended budget', format='%.6f')},
                            key='editor_table_v550')

    c1,c2 = st.columns(2)
    with c1:
        if st.button('🔄 Save & Recalculate'):
            base_applied = apply_editor_to_base(base_df, edited, editable_cols)
            picked = st.session_state.get('cat_order', [])
            df_in, _ = filter_by_categories(base_applied, picked)
            df_in = apply_blacklist(df_in, st.session_state.get('bl_selected', []))
            df_in = apply_platform_bounds(df_in, st.session_state.get('platform_bounds', {}))
            df_in = apply_category_priorities_from_order(df_in, picked)
            df_res, summary, total_margin, meta = allocate_budget(
                df_in,
                total_budget=float(st.session_state.get('total_budget_cache', 240.0)),
                alpha=float(st.session_state.get('alpha_cache', 1.6)),
                beta=float(st.session_state.get('beta_cache', 1.0)),
                other_share=float(st.session_state.get('other_share_cache', 10.0)),
                use_gates=None
            )
            st.session_state.base_df = base_applied.copy()
            st.session_state.df_result = df_res.copy()
            st.session_state.summary = summary.copy()
            st.session_state.total_margin = float(total_margin)

            total_loaded = float(pd.to_numeric(seriesize(df_res.get('recommended budget'), len(df_res)), errors='coerce').fillna(0).sum())
            tb = float(st.session_state.get('total_budget_cache', total_loaded))
            pct = (total_loaded/tb*100.0) if tb>0 else 100.0
            other = float(st.session_state.get('other_share_cache', 10.0))
            st.success(f"✅ Бюджет успешно распределён: {total_loaded:.2f} млн ₽ ({pct:.0f}%), Free Float Share ({other:.0f}%)")
            margin_box.markdown(f"### 💰 Общая маржинальность сплита: **{float(total_margin):.2f}%**")
    with c2:
        if st.button('⬅️ Back'):
            origin = st.session_state.get('edit_source')
            st.session_state['edit_source']=None
            st.session_state.mode = 'result' if origin!='upload' else 'filters'
            st.rerun()

    cur_df = st.session_state.get('df_result'); cur_sum = st.session_state.get('summary')
    if cur_df is not None and cur_sum is not None:
        d1,d2 = st.columns(2)
        with d1: st.download_button('💾 Download Result (CSV)', data=export_csv(cur_df), file_name='split_by_placement.csv', mime='text/csv')
        with d2: st.download_button('💾 Download Result (.xlsx)', data=export_excel(cur_df, cur_sum, st.session_state.base_df),
                                    file_name='media_split_results_edited.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
