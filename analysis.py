import streamlit as st
import pandas as pd
import numpy as np
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
import platform
import matplotlib.font_manager as fm
import io

# ==========================================
# 0. 環境設定與中文字體 (強化版)
# ==========================================
def set_plot_font(font_scale=1.2):
    system = platform.system()
    font_priority = [
        'Microsoft JhengHei', 'SimHei', 'LiSu',  # Windows
        'Heiti TC', 'Arial Unicode MS',          # Mac
        'WenQuanYi Micro Hei', 'Noto Sans CJK JP', 'DejaVu Sans' # Linux/Cloud
    ]
    
    selected_font = None
    for font in font_priority:
        try:
            if font in [f.name for f in fm.fontManager.ttflist]:
                selected_font = font
                break
        except:
            continue

    if not selected_font:
        selected_font = 'Microsoft JhengHei'

    plt.rcParams['font.sans-serif'] = [selected_font]
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_theme(style="whitegrid", font=selected_font, font_scale=font_scale)
    return selected_font

# ==========================================
# 1. 核心邏輯：數據處理與統計
# ==========================================

def get_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    return pd.read_excel(file)

def generate_descriptive_table(df_long, dv_name):
    """生成論文常用的 Mean ± SD 表格"""
    desc = df_long.groupby(['Treatment', 'Time'])[dv_name].agg(['mean', 'std']).reset_index()
    desc['Format'] = desc.apply(lambda x: f"{x['mean']:.2f} ± {x['std']:.2f}", axis=1)
    desc_pivot = desc.pivot(index='Treatment', columns='Time', values='Format')
    return desc_pivot

def check_normality(df_long, dv_name):
    """執行 Shapiro-Wilk 常態性檢定"""
    # 確保樣本數足夠才執行
    if len(df_long) < 3: 
        return pd.DataFrame()
    try:
        normality = pg.normality(data=df_long, dv=dv_name, group='Time')
        return normality
    except:
        return pd.DataFrame()

def run_comprehensive_analysis(df, dv_name):
    # 自動識別存在的時間點
    target_times = ['0W', '1W', '2W', '4W', '6W', '8W', '12W', '24W', '1Y'] 
    available_times = [t for t in target_times if t in df.columns]
    
    if 'Subject_Num' not in df.columns or 'Treatment' not in df.columns:
        return None, None, None, "❌ 缺少必要欄位：請確保檔案包含 'Subject_Num' 和 'Treatment'。", None, None, None, None

    # --- 改善率計算 (依據使用者指定的三個區間) ---
    def safe_pct(post, pre):
        # 避免分母為0或極小值
        denom = np.where(pre == 0, 0.1, pre) 
        # (後測 - 前測) / 前測 * 100
        return ((post - pre) / denom) * 100

    df_imp = df.copy()
    imp_cols = []
    
    # 1. 短期改善 (0W -> 12W)
    if '0W' in df.columns and '12W' in df.columns:
        df_imp['短期改善(0-12W)'] = safe_pct(df_imp['12W'], df_imp['0W'])
        imp_cols.append('短期改善(0-12W)')
        
    # 2. 長期改善 (12W -> 24W)
    if '12W' in df.columns and '24W' in df.columns:
        df_imp['長期改善(12-24W)'] = safe_pct(df_imp['24W'], df_imp['12W'])
        imp_cols.append('長期改善(12-24W)')

    # 3. 整體改善 (0W -> 24W)
    if '0W' in df.columns and '24W' in df.columns:
        df_imp['整體改善(0-24W)'] = safe_pct(df_imp['24W'], df_imp['0W'])
        imp_cols.append('整體改善(0-24W)')

    # 如果上述都沒有，但有其他時間點，則做一個通用的 (頭-尾)
    if not imp_cols and len(available_times) >= 2:
        start, end = available_times[0], available_times[-1]
        col_name = f'整體改善({start}-{end})'
        df_imp[col_name] = safe_pct(df_imp[end], df_imp[start])
        imp_cols.append(col_name)

    # 計算各組平均
    if imp_cols:
        imp_stats = df_imp.groupby('Treatment')[imp_cols].mean().round(2)
    else:
        imp_stats = pd.DataFrame()

    # --- 長資料轉換 ---
    df_long = df.melt(id_vars=['Subject_Num', 'Treatment'], value_vars=available_times, 
                      var_name='Time', value_name=dv_name)
    
    df_long['Time_Rank'] = df_long['Time'].apply(lambda x: available_times.index(x))
    df_long = df_long.sort_values(['Subject_Num', 'Time_Rank'])

    # --- 統計執行 ---
    try:
        norm_res = check_normality(df_long, dv_name)
        
        aov = pg.mixed_anova(dv=dv_name, within='Time', between='Treatment', subject='Subject_Num', data=df_long)
        p_inter = aov.loc[aov['Source'] == 'Interaction', 'p-unc'].values[0]
        
        if p_inter < 0.05:
            msg = "🔴 交互作用顯著 (p<0.05)：各組別隨時間變化的趨勢不同。"
        else:
            msg = "✅ 交互作用不顯著 (p>=0.05)：各組別變化趨勢一致。"
            
        ph = pg.pairwise_tests(dv=dv_name, within='Time', between='Treatment', subject='Subject_Num', data=df_long, padjust='bonf').round(4)
        
    except Exception as e:
        aov, ph, msg, norm_res = None, None, f"⚠️ 統計運算錯誤: {str(e)}", None

    desc_stats = generate_descriptive_table(df_long, dv_name)

    return imp_stats, aov, ph, msg, df_long, available_times, desc_stats, norm_res

# ==========================================
# 2. 視覺化函數
# ==========================================

def draw_plots(df_long, imp_stats, dv_name, style_color='tab10'):
    
    # 圖1: 折線圖
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.pointplot(data=df_long, x='Time', y=dv_name, hue='Treatment', 
                  capsize=.15, dodge=0.2, 
                  markers=["o", "s", "D", "^"], 
                  linestyles=["-", "--", "-.", ":"],
                  errorbar='se', 
                  palette=style_color, ax=ax1)
    
    ax1.set_title(f"【{dv_name}】趨勢分析 (Mean ± SE)", fontweight='bold', fontsize=16)
    ax1.set_xlabel("評估時間點", fontsize=12)
    ax1.set_ylabel("分數", fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.legend(title="Treatment", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
    plt.tight_layout()

    # 圖2: 改善率 Bar 圖
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    if not imp_stats.empty:
        # 重置索引以便繪圖
        plot_df = imp_stats.reset_index().melt(id_vars='Treatment', var_name='階段', value_name='改善率(%)')
        
        # 這裡可以自定義階段的順序，確保圖表上依序顯示 短期 -> 長期 -> 整體
        desired_order = ['短期改善(0-12W)', '長期改善(12-24W)', '整體改善(0-24W)']
        # 過濾出實際存在的 column
        order = [col for col in desired_order if col in plot_df['階段'].unique()]
        
        sns.barplot(data=plot_df, x='Treatment', y='改善率(%)', hue='階段', 
                    hue_order=order if order else None, # 指定順序
                    palette='viridis', ax=ax2, edgecolor='black')
        
        for container in ax2.containers:
            ax2.bar_label(container, fmt='%.1f%%', padding=3, fontsize=10, fontweight='bold')
            
        ax2.set_title(f"【{dv_name}】分階段改善率比較", fontweight='bold', fontsize=16)
        ax2.axhline(0, color='gray', linewidth=1)
        ax2.set_ylabel("改善百分比 (%)")
        ax2.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    else:
        ax2.text(0.5, 0.5, "數據不足以計算改善率 (需有 0W, 12W, 24W)", ha='center', fontsize=14)
        
    plt.tight_layout()

    return fig1, fig2

# ==========================================
# 3. Streamlit 介面
# ==========================================

def main():
    st.set_page_config(page_title="臨床數據分析 Pro", page_icon="🩺", layout="wide")
    
    with st.sidebar:
        st.header("⚙️ 設定與說明")
        font_scale = st.slider("字體大小縮放", 0.8, 2.0, 1.2)
        used_font = set_plot_font(font_scale)
        st.caption(f"目前使用字體: {used_font}")
        
        st.info("""
        **檔案格式要求：**
        1. 必須包含 `Subject_Num` 和 `Treatment`
        2. 時間點需包含 `0W`, `12W`, `24W` 以計算完整改善率
        """)

    st.title("🩺 臨床數據自動化分析系統 Pro")
    st.markdown("---")

    uploaded_files = st.file_uploader("📂 上傳 Excel (.xlsx) 或 CSV 檔案", 
                                      type=['xlsx', 'csv'], accept_multiple_files=True)

    if uploaded_files:
        tabs = st.tabs([f"📊 {f.name}" for f in uploaded_files])
        
        for i, file in enumerate(uploaded_files):
            with tabs[i]:
                df = get_data(file)
                dv_name = file.name.split('.')[0]
                
                imp_stats, aov, ph, msg, df_long, available_times, desc_stats, norm_res = run_comprehensive_analysis(df, dv_name)

                if isinstance(msg, str) and msg.startswith("❌"):
                    st.error(msg)
                    st.dataframe(df.head())
                    continue

                c1, c2 = st.columns([1, 2])
                with c1:
                    st.success(f"📅 偵測時間點：{', '.join(available_times)}")
                    st.markdown(f"### {msg}")
                    with st.expander("查看常態性檢定 (Shapiro-Wilk)"):
                        st.dataframe(norm_res)
                
                with c2:
                    st.subheader("📋 階段改善率 (%)")
                    st.dataframe(imp_stats, use_container_width=True)

                st.divider()

                st.subheader("🔢 敘述性統計 (Mean ± SD)")
                st.dataframe(desc_stats, use_container_width=True)

                c3, c4 = st.columns(2)
                with c3:
                    st.subheader("🔬 Mixed ANOVA 結果")
                    if aov is not None:
                        # 修正: 使用正確的欄位名稱 DF1, DF2
                        target_cols = ['Source', 'DF1', 'DF2', 'F', 'p-unc', 'np2']
                        available_cols = [c for c in target_cols if c in aov.columns]
                        st.table(aov[available_cols].style.format({'F': '{:.3f}', 'p-unc': '{:.4f}', 'np2': '{:.3f}'}))
                    else:
                        st.warning("無法執行 ANOVA")

                with c4:
                    st.subheader("🔍 事後比較 (Post-hoc)")
                    if ph is not None:
                        st.dataframe(ph, height=250, use_container_width=True)
                        csv = ph.to_csv(index=False).encode('utf-8-sig')
                        st.download_button("📥 下載 Post-hoc (CSV)", csv, f'{dv_name}_posthoc.csv', 'text/csv')

                st.divider()
                
                st.subheader("📊 高畫質圖表")
                f1, f2 = draw_plots(df_long, imp_stats, dv_name)
                
                cp1, cp2 = st.columns(2)
                with cp1:
                    st.pyplot(f1)
                    img1 = io.BytesIO()
                    f1.savefig(img1, format='png', dpi=300, bbox_inches='tight')
                    st.download_button("📥 下載趨勢圖 (PNG)", img1.getvalue(), f'{dv_name}_trend.png', 'image/png')
                    
                with cp2:
                    st.pyplot(f2)
                    img2 = io.BytesIO()
                    f2.savefig(img2, format='png', dpi=300, bbox_inches='tight')
                    st.download_button("📥 下載改善率圖 (PNG)", img2.getvalue(), f'{dv_name}_imp.png', 'image/png')

if __name__ == "__main__":
    main()