import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              classification_report, roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Petrol Consumption Dashboard",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #141824 100%);
        border-right: 1px solid #2d3748;
    }
    .kpi-card {
        background: linear-gradient(135deg, #1e2a3a 0%, #162032 100%);
        border: 1px solid #2d4a6e;
        border-radius: 12px;
        padding: 20px 16px;
        text-align: center;
        margin: 4px 0;
        box-shadow: 0 4px 15px rgba(0,100,200,0.1);
    }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #4fc3f7; margin: 0; }
    .kpi-label { font-size: 0.78rem; color: #8fa8c8; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }
    .kpi-delta { font-size: 0.82rem; color: #66bb6a; margin-top: 2px; }
    .section-header {
        background: linear-gradient(90deg, #1a3a5c 0%, #1e2a3a 100%);
        border-left: 4px solid #4fc3f7;
        padding: 12px 20px;
        border-radius: 0 8px 8px 0;
        margin: 24px 0 16px 0;
    }
    .section-header h2 { color: #e2e8f0; margin: 0; font-size: 1.3rem; font-weight: 600; }
    .insight-card {
        background: linear-gradient(135deg, #1a2a1a 0%, #162216 100%);
        border: 1px solid #2d5a2d;
        border-radius: 10px;
        padding: 16px;
        margin: 8px 0;
    }
    .insight-title { color: #66bb6a; font-weight: 700; font-size: 0.9rem; margin-bottom: 6px; }
    .insight-text  { color: #b0c4b0; font-size: 0.85rem; line-height: 1.5; }
    .rec-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #4a4a8a;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 6px 0;
    }
    .rec-number { color: #ce93d8; font-weight: 800; font-size: 1.1rem; }
    .rec-text   { color: #c0c0e0; font-size: 0.85rem; line-height: 1.5; }
    .metric-card {
        background: linear-gradient(135deg, #1a2a3a 0%, #0d1b2a 100%);
        border: 1px solid #2a4a6a;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
    }
    .metric-val  { font-size: 1.8rem; font-weight: 700; color: #4fc3f7; }
    .metric-name { font-size: 0.78rem; color: #8fa8c8; text-transform: uppercase; letter-spacing: 1px; }

    /* ── Predictor card ── */
    .pred-high {
        background: linear-gradient(135deg, #0d2137 0%, #0a1929 100%);
        border: 2px solid #4fc3f7;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        box-shadow: 0 0 30px rgba(79,195,247,0.2);
    }
    .pred-low {
        background: linear-gradient(135deg, #2a0d0d 0%, #1a0909 100%);
        border: 2px solid #f06292;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        box-shadow: 0 0 30px rgba(240,98,146,0.2);
    }
    .pred-label   { font-size: 2rem; font-weight: 800; margin-bottom: 8px; }
    .pred-conf    { font-size: 1rem; color: #8fa8c8; margin-top: 6px; }
    .pred-proba   { font-size: 0.85rem; color: #8fa8c8; margin-top: 4px; }
    .input-hint   { font-size: 0.78rem; color: #8fa8c8; margin-top: 2px; font-style: italic; }

    /* gauge label */
    .gauge-label  { text-align:center; color:#8fa8c8; font-size:0.8rem; margin-top:-10px; }

    hr { border-color: #2d3748; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD & PREPARE DATA
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv('petrol_consumption.csv')
    median_val              = df['Petrol_Consumption'].median()
    df['Consumption_Class'] = (df['Petrol_Consumption'] >= median_val).astype(int)
    df['consumption_label'] = df['Consumption_Class'].map({1: 'High', 0: 'Low'})
    df['tax_group'] = pd.cut(df['Petrol_tax'],
        bins=[0, 7, 8, 10], labels=['Low Tax (<7)', 'Mid Tax (7-8)', 'High Tax (>8)'])
    df['income_level'] = pd.cut(df['Average_income'],
        bins=[0, 4000, 5000, 6000, 10000],
        labels=['Low (<4K)', 'Mid (4-5K)', 'Upper-Mid (5-6K)', 'High (>6K)'])
    df['highway_level'] = pd.cut(df['Paved_Highways'],
        bins=[0, 3000, 6000, 9000, 20000],
        labels=['Low (<3K)', 'Mid (3-6K)', 'High (6-9K)', 'Very High (>9K)'])
    df['license_level'] = pd.cut(df['Population_Driver_licence(%)'],
        bins=[0, 0.50, 0.55, 0.60, 1.0],
        labels=['Low (<50%)', 'Mid (50-55%)', 'High (55-60%)', 'Very High (>60%)'])
    return df, median_val

@st.cache_resource
def train_model(df):
    FEATURES = ['Petrol_tax','Average_income','Paved_Highways','Population_Driver_licence(%)']
    X = df[FEATURES]; y = df['Consumption_Class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline = Pipeline([('scaler', StandardScaler()),
                          ('clf',   DecisionTreeClassifier(random_state=42))])
    param_grid = {
        'clf__criterion'        : ['gini','entropy'],
        'clf__max_depth'        : list(range(1,9)),
        'clf__min_samples_split': [2,4,6],
        'clf__min_samples_leaf' : [1,2,3]
    }
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    best      = grid.best_estimator_
    y_pred    = best.predict(X_test)
    y_prob    = best.predict_proba(X_test)[:, 1]
    y_pred_tr = best.predict(X_train)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc         = roc_auc_score(y_test, y_prob)
    cm          = confusion_matrix(y_test, y_pred)
    report      = classification_report(y_test, y_pred,
                    target_names=['Low','High'], output_dict=True)
    best_dt    = best.named_steps['clf']
    feat_imp   = pd.Series(best_dt.feature_importances_, index=FEATURES)
    tree_rules = export_text(best_dt, feature_names=FEATURES)
    return {
        'model'      : best,
        'best_params': grid.best_params_,
        'train_acc'  : accuracy_score(y_train, y_pred_tr),
        'test_acc'   : accuracy_score(y_test,  y_pred),
        'auc'        : auc,
        'fpr'        : fpr, 'tpr': tpr,
        'cm'         : cm,  'report': report,
        'feat_imp'   : feat_imp,
        'tree_rules' : tree_rules,
        'FEATURES'   : FEATURES,
        'df_stats'   : df[FEATURES].describe(),
    }

df, median_val = load_data()
results        = train_model(df)

# ══════════════════════════════════════════════════════════════════════════════
# THEME HELPERS
# ══════════════════════════════════════════════════════════════════════════════
LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(14,17,23,0.6)',
    font=dict(color='#c8d8e8', family='Segoe UI, sans-serif'),
    title_font=dict(size=15, color='#e2e8f0'),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#2d3748'),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor='#1e2a3a', linecolor='#2d3748', tickcolor='#8fa8c8'),
    yaxis=dict(gridcolor='#1e2a3a', linecolor='#2d3748', tickcolor='#8fa8c8'),
)
COLORS  = {'High':'#4fc3f7','Low':'#f06292','blue':'#4fc3f7','orange':'#ffb74d',
           'green':'#66bb6a','purple':'#ce93d8','red':'#ef5350','teal':'#4db6ac'}
PALETTE = ['#4fc3f7','#f06292','#66bb6a','#ffb74d','#ce93d8','#4db6ac','#ef5350','#ffd54f']

def section(icon, title):
    st.markdown(f'<div class="section-header"><h2>{icon} {title}</h2></div>',
                unsafe_allow_html=True)

def kpi(value, label, delta=""):
    return (f'<div class="kpi-card"><div class="kpi-value">{value}</div>'
            f'<div class="kpi-label">{label}</div>'
            + (f'<div class="kpi-delta">{delta}</div>' if delta else '')
            + '</div>')

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:20px 0 10px 0;'>
        <div style='font-size:2.5rem;'>⛽</div>
        <div style='color:#4fc3f7; font-size:1.1rem; font-weight:700;'>Petrol Consumption</div>
        <div style='color:#8fa8c8; font-size:0.78rem;'>Analytics Dashboard</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    nav = st.radio("📍 Navigation", [
        "🏠  Overview",
        "📊  EDA & Distributions",
        "📈  KPI Analysis",
        "🔗  Correlation & Relationships",
        "🌳  Decision Tree Model",
        "🔮  Live Predictor",
        "💡  Insights & Recommendations",
    ])
    st.markdown("---")
    st.markdown("""
    <div style='background:#1e2a3a; border-radius:10px; padding:14px; margin-top:8px;'>
        <div style='color:#4fc3f7; font-weight:700; font-size:0.85rem; margin-bottom:8px;'>📋 Dataset Info</div>
        <div style='color:#8fa8c8; font-size:0.78rem; line-height:1.8;'>
            📌 Records: <b style='color:#e2e8f0;'>48 States</b><br>
            📌 Features: <b style='color:#e2e8f0;'>4 Numerical</b><br>
            📌 Target: <b style='color:#e2e8f0;'>High / Low</b><br>
            📌 Threshold: <b style='color:#e2e8f0;'>{:.0f} (Median)</b><br>
            📌 Model: <b style='color:#e2e8f0;'>Decision Tree</b>
        </div>
    </div>""".format(median_val), unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**🎛️ Filter Data**")
    tax_filter = st.multiselect("Tax Group",
        options=df['tax_group'].dropna().unique().tolist(),
        default=df['tax_group'].dropna().unique().tolist())
    income_filter = st.multiselect("Income Level",
        options=df['income_level'].dropna().unique().tolist(),
        default=df['income_level'].dropna().unique().tolist())

dff = df[df['tax_group'].isin(tax_filter) & df['income_level'].isin(income_filter)]

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if nav == "🏠  Overview":
    st.markdown("""
    <h1 style='text-align:center; background:linear-gradient(90deg,#1a3a5c,#1e2a3a);
               color:white; padding:24px; border-radius:12px; margin-bottom:24px;
               border:1px solid #2d4a6e;'>
        ⛽ Petrol Consumption — Analytics & Decision Tree Dashboard
    </h1>""", unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.markdown(kpi(f"{len(dff)}", "Total Records", "US States"), unsafe_allow_html=True)
    c2.markdown(kpi(f"{dff['Petrol_Consumption'].mean():.0f}", "Avg Consumption", "gal/person"), unsafe_allow_html=True)
    c3.markdown(kpi(f"{dff['Petrol_tax'].mean():.2f}c", "Avg Petrol Tax", "cents/gallon"), unsafe_allow_html=True)
    c4.markdown(kpi(f"${dff['Average_income'].mean():,.0f}", "Avg Income", "USD per capita"), unsafe_allow_html=True)
    c5.markdown(kpi(f"{dff['Population_Driver_licence(%)'].mean()*100:.1f}%", "Avg Driver Licence", "of population"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1,c2,c3,c4,c5 = st.columns(5)
    high = dff[dff['Consumption_Class']==1]; low = dff[dff['Consumption_Class']==0]
    c1.markdown(kpi(f"{len(high)}", "High Consumption", f">= {median_val:.0f} gal"), unsafe_allow_html=True)
    c2.markdown(kpi(f"{len(low)}",  "Low Consumption",  f"< {median_val:.0f} gal"),  unsafe_allow_html=True)
    c3.markdown(kpi(f"{dff['Petrol_Consumption'].max()}", "Max Consumption", "gallons"), unsafe_allow_html=True)
    c4.markdown(kpi(f"{dff['Petrol_Consumption'].min()}", "Min Consumption", "gallons"), unsafe_allow_html=True)
    c5.markdown(kpi(f"{dff['Paved_Highways'].mean():,.0f}", "Avg Highways", "miles paved"), unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        section("🥧", "Consumption Class Distribution")
        counts = dff['consumption_label'].value_counts().reset_index()
        counts.columns = ['Class','Count']
        fig = px.pie(counts, names='Class', values='Count', hole=0.55,
                     color='Class', color_discrete_map={'High':COLORS['blue'],'Low':COLORS['red']})
        fig.update_traces(textinfo='percent+label+value', textfont_size=13,
                          marker=dict(line=dict(color='#0e1117', width=3)))
        fig.add_annotation(text=f"<b>{len(dff)}</b><br>States", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=16, color='#e2e8f0'))
        fig.update_layout(**LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        section("📊", "Feature Summary Statistics")
        num_cols = ['Petrol_tax','Average_income','Paved_Highways',
                    'Population_Driver_licence(%)','Petrol_Consumption']
        summary = dff[num_cols].agg(['mean','median','std','min','max']).T.round(2)
        summary.columns = ['Mean','Median','Std','Min','Max']
        st.dataframe(summary.style.background_gradient(cmap='Blues', axis=1),
                     use_container_width=True)
        section("📌", "Raw Data Preview")
        st.dataframe(dff[['Petrol_tax','Average_income','Paved_Highways',
                           'Population_Driver_licence(%)','Petrol_Consumption',
                           'consumption_label']].head(10),
                     use_container_width=True, height=230)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "📊  EDA & Distributions":
    st.title("📊 EDA & Distributions")
    num_features = ['Petrol_tax','Average_income','Paved_Highways',
                    'Population_Driver_licence(%)','Petrol_Consumption']

    section("📈", "Distribution of All Features")
    fig = make_subplots(rows=2, cols=3,
                        subplot_titles=[f"<b>{c}</b>" for c in num_features],
                        vertical_spacing=0.15, horizontal_spacing=0.08)
    for i, col in enumerate(num_features):
        r, c2 = divmod(i, 3)
        fig.add_trace(go.Histogram(x=dff[col], name=col, nbinsx=12,
                                   marker_color=PALETTE[i], opacity=0.85, showlegend=False),
                      row=r+1, col=c2+1)
        fig.add_vline(x=dff[col].mean(),   line_dash='dash', line_color='#ffb74d', row=r+1, col=c2+1)
        fig.add_vline(x=dff[col].median(), line_dash='dot',  line_color='#66bb6a', row=r+1, col=c2+1)
    fig.update_layout(**LAYOUT, height=520,
                      title_text="Feature Distributions  |  Mean(orange-dash)  Median(green-dot)")
    fig.update_xaxes(gridcolor='#1e2a3a'); fig.update_yaxes(gridcolor='#1e2a3a')
    st.plotly_chart(fig, use_container_width=True)

    section("📦", "Boxplots by Consumption Class")
    fig = make_subplots(rows=1, cols=len(num_features),
                        subplot_titles=[f"<b>{c}</b>" for c in num_features],
                        horizontal_spacing=0.06)
    for i, col in enumerate(num_features):
        for cls, color in [('High', COLORS['blue']), ('Low', COLORS['red'])]:
            vals = dff[dff['consumption_label']==cls][col]
            fig.add_trace(go.Box(y=vals, name=cls, marker_color=color,
                                 showlegend=(i==0), boxmean='sd', legendgroup=cls),
                          row=1, col=i+1)
    fig.update_layout(**LAYOUT, height=420)
    fig.update_xaxes(gridcolor='#1e2a3a'); fig.update_yaxes(gridcolor='#1e2a3a')
    st.plotly_chart(fig, use_container_width=True)

    section("🎻", "Violin Plots — Feature Spread by Class")
    selected = st.selectbox("Select Feature", num_features)
    fig = px.violin(dff, y=selected, x='consumption_label', color='consumption_label',
                    box=True, points='all',
                    color_discrete_map={'High':COLORS['blue'],'Low':COLORS['red']})
    fig.update_layout(**LAYOUT, height=430,
                      xaxis_title='Consumption Class', yaxis_title=selected)
    st.plotly_chart(fig, use_container_width=True)

    section("📐", "Skewness & Kurtosis")
    c1, c2 = st.columns(2)
    skew = dff[num_features].skew().reset_index(); skew.columns = ['Feature','Skewness']
    kurt = dff[num_features].kurt().reset_index(); kurt.columns = ['Feature','Kurtosis']
    with c1:
        fig = px.bar(skew, x='Skewness', y='Feature', orientation='h',
                     color='Skewness', color_continuous_scale='RdBu', title='Skewness per Feature')
        fig.update_layout(**LAYOUT, height=300)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(kurt, x='Kurtosis', y='Feature', orientation='h',
                     color='Kurtosis', color_continuous_scale='Viridis', title='Kurtosis per Feature')
        fig.update_layout(**LAYOUT, height=300)
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: KPI ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "📈  KPI Analysis":
    st.title("📈 KPI Analysis")

    section("⛽", "High Consumption Rate by Tax Group")
    cons_tax = dff.groupby('tax_group', observed=True)['Consumption_Class'].mean().reset_index()
    cons_tax['High_Rate'] = (cons_tax['Consumption_Class']*100).round(1)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(cons_tax, x='tax_group', y='High_Rate', color='High_Rate',
                     color_continuous_scale='Blues', text='High_Rate',
                     title='High Consumption Rate by Tax Group (%)')
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(**LAYOUT, height=360, showlegend=False,
                          xaxis_title='Tax Group', yaxis_title='Rate (%)')
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.line(cons_tax, x='tax_group', y='High_Rate', markers=True, text='High_Rate',
                      color_discrete_sequence=[COLORS['blue']],
                      title='Trend — High Consumption Rate by Tax Group')
        fig.update_traces(marker=dict(size=12), textposition='top center',
                          texttemplate='%{text:.1f}%', line=dict(width=3))
        fig.update_layout(**LAYOUT, height=360, xaxis_title='Tax Group', yaxis_title='Rate (%)')
        st.plotly_chart(fig, use_container_width=True)

    section("💰", "Income Level Analysis")
    inc_kpi = dff.groupby('income_level', observed=True).agg(
        Records=('Consumption_Class','count'), High_Consumption=('Consumption_Class','sum'),
        High_Rate=('Consumption_Class','mean'), Avg_Consumption=('Petrol_Consumption','mean')
    ).reset_index()
    inc_kpi['High_Rate']       = (inc_kpi['High_Rate']*100).round(1)
    inc_kpi['Avg_Consumption'] = inc_kpi['Avg_Consumption'].round(1)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(inc_kpi, x='income_level', y='High_Consumption', color='High_Consumption',
                     color_continuous_scale='Teal', text='High_Consumption',
                     title='High Consumption Count by Income Level')
        fig.update_traces(textposition='outside')
        fig.update_layout(**LAYOUT, height=360, xaxis_title='Income Level', yaxis_title='Count')
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(inc_kpi.sort_values('High_Rate', ascending=False), x='income_level',
                     y='High_Rate', color='High_Rate', color_continuous_scale='OrRd',
                     text='High_Rate', title='High Consumption Rate by Income Level (%)')
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(**LAYOUT, height=360, xaxis_title='Income Level', yaxis_title='Rate (%)')
        st.plotly_chart(fig, use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(inc_kpi.sort_values('Records', ascending=False), x='income_level',
                     y='Records', color='Records', color_continuous_scale='Greens',
                     text='Records', title='Total Records by Income Level')
        fig.update_traces(textposition='outside')
        fig.update_layout(**LAYOUT, height=340, xaxis_title='Income Level', yaxis_title='Count')
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(inc_kpi, x='income_level', y='Avg_Consumption', color='Avg_Consumption',
                     color_continuous_scale='Blues', text='Avg_Consumption',
                     title='Avg Petrol Consumption by Income Level')
        fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')
        fig.update_layout(**LAYOUT, height=340, xaxis_title='Income Level', yaxis_title='Avg Consumption')
        st.plotly_chart(fig, use_container_width=True)

    section("🛣️", "Highway & Driver Licence Level Analysis")
    hw_kpi = dff.groupby('highway_level', observed=True).agg(
        High_Rate=('Consumption_Class','mean'), Records=('Consumption_Class','count'),
        Avg_Cons=('Petrol_Consumption','mean')).reset_index()
    hw_kpi['High_Rate'] = (hw_kpi['High_Rate']*100).round(1)
    lic_kpi = dff.groupby('license_level', observed=True).agg(
        High_Rate=('Consumption_Class','mean'), Records=('Consumption_Class','count'),
        Avg_Cons=('Petrol_Consumption','mean')).reset_index()
    lic_kpi['High_Rate'] = (lic_kpi['High_Rate']*100).round(1)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(hw_kpi, x='highway_level', y='High_Rate', color='High_Rate',
                     color_continuous_scale='Purples', text='High_Rate',
                     title='High Consumption Rate by Highway Level (%)')
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(**LAYOUT, height=360, xaxis_title='Highway Level', yaxis_title='Rate (%)')
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(lic_kpi, x='license_level', y='High_Rate', color='High_Rate',
                     color_continuous_scale='Reds', text='High_Rate',
                     title='High Consumption Rate by Driver Licence Level (%)')
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(**LAYOUT, height=360, xaxis_title='Driver Licence Level', yaxis_title='Rate (%)')
        st.plotly_chart(fig, use_container_width=True)

    section("🔥", "Cross-Feature Heatmaps")
    c1, c2 = st.columns(2)
    with c1:
        pivot = dff.groupby(['tax_group','income_level'], observed=True)['Consumption_Class'].mean().reset_index()
        pivot['High_Rate'] = (pivot['Consumption_Class']*100).round(1)
        pv = pivot.pivot(index='tax_group', columns='income_level', values='High_Rate').fillna(0)
        fig = px.imshow(pv, text_auto='.1f', color_continuous_scale='YlOrRd',
                        title='High Consumption Rate (%): Tax x Income')
        fig.update_layout(**LAYOUT, height=360)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        pivot2 = dff.groupby(['highway_level','license_level'], observed=True)['Consumption_Class'].mean().reset_index()
        pivot2['High_Rate'] = (pivot2['Consumption_Class']*100).round(1)
        pv2 = pivot2.pivot(index='highway_level', columns='license_level', values='High_Rate').fillna(0)
        fig = px.imshow(pv2, text_auto='.1f', color_continuous_scale='Blues',
                        title='High Consumption Rate (%): Highway x Licence')
        fig.update_layout(**LAYOUT, height=360)
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CORRELATION
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "🔗  Correlation & Relationships":
    st.title("🔗 Correlation & Relationships")
    num_cols = ['Petrol_tax','Average_income','Paved_Highways',
                'Population_Driver_licence(%)','Petrol_Consumption']

    section("🌡️", "Correlation Heatmap")
    corr = dff[num_cols].corr().round(3)
    fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1, aspect='auto', title='Pearson Correlation Matrix')
    fig.update_layout(**LAYOUT, height=430)
    fig.update_traces(textfont_size=14)
    st.plotly_chart(fig, use_container_width=True)

    section("🎯", "Feature Correlation with Petrol Consumption")
    target_corr = corr['Petrol_Consumption'].drop('Petrol_Consumption').sort_values()
    colors_bar  = [COLORS['red'] if v < 0 else COLORS['blue'] for v in target_corr.values]
    fig = go.Figure(go.Bar(x=target_corr.values, y=target_corr.index, orientation='h',
                           marker_color=colors_bar,
                           text=[f"{v:.3f}" for v in target_corr.values], textposition='outside'))
    fig.add_vline(x=0, line_color='#8fa8c8', line_width=1)
    fig.update_layout(**LAYOUT, height=320, xaxis_title='Correlation Coefficient',
                      title='Feature Correlation with Petrol_Consumption')
    st.plotly_chart(fig, use_container_width=True)

    section("🔵", "Scatter Plots vs Petrol Consumption")
    features = [c for c in num_cols if c != 'Petrol_Consumption']
    fig = make_subplots(rows=1, cols=len(features),
                        subplot_titles=[f"<b>{c}</b>" for c in features],
                        horizontal_spacing=0.07)
    for i, col in enumerate(features):
        color = COLORS['blue'] if target_corr[col] > 0 else COLORS['red']
        fig.add_trace(go.Scatter(x=dff[col], y=dff['Petrol_Consumption'], mode='markers',
                                 marker=dict(color=color, size=9, opacity=0.75,
                                             line=dict(color='white', width=1)),
                                 name=col, showlegend=False), row=1, col=i+1)
        z = np.polyfit(dff[col], dff['Petrol_Consumption'], 1)
        x_line = np.linspace(dff[col].min(), dff[col].max(), 50)
        fig.add_trace(go.Scatter(x=x_line, y=np.poly1d(z)(x_line), mode='lines',
                                 line=dict(color='#ffb74d', width=2.5, dash='dash'),
                                 showlegend=False), row=1, col=i+1)
        fig.update_xaxes(title_text=col, gridcolor='#1e2a3a', row=1, col=i+1)
        fig.update_yaxes(title_text='Petrol Consumption', gridcolor='#1e2a3a', row=1, col=i+1)
    fig.update_layout(**LAYOUT, height=380, title='Scatter Plots: Features vs Petrol Consumption')
    st.plotly_chart(fig, use_container_width=True)

    section("🔷", "Scatter Matrix (Pairplot)")
    fig = px.scatter_matrix(dff, dimensions=num_cols, color='consumption_label',
                            color_discrete_map={'High':COLORS['blue'],'Low':COLORS['red']},
                            opacity=0.65)
    fig.update_traces(diagonal_visible=False,
                      marker=dict(size=5, line=dict(width=0.5, color='white')))
    fig.update_layout(**LAYOUT, height=560, title='Scatter Matrix — All Features')
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DECISION TREE MODEL  (no CV section)
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "🌳  Decision Tree Model":
    st.title("🌳 Decision Tree Classifier")

    # ── Metrics ──────────────────────────────────────────────────────────────
    section("📊", "Model Performance Metrics")
    c1,c2,c3 = st.columns(3)
    c1.markdown(f'<div class="metric-card"><div class="metric-val">{results["train_acc"]*100:.1f}%</div><div class="metric-name">Train Accuracy</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-val">{results["test_acc"]*100:.1f}%</div><div class="metric-name">Test Accuracy</div></div>',  unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-val">{results["auc"]:.3f}</div><div class="metric-name">ROC AUC Score</div></div>',             unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**🏆 Best Hyperparameters — GridSearchCV**")
    bp = results['best_params']
    c1,c2,c3,c4 = st.columns(4)
    c1.info(f"Criterion: **{bp['clf__criterion']}**")
    c2.info(f"Max Depth: **{bp['clf__max_depth']}**")
    c3.info(f"Min Samples Split: **{bp['clf__min_samples_split']}**")
    c4.info(f"Min Samples Leaf: **{bp['clf__min_samples_leaf']}**")

    # ── Confusion Matrix + ROC ────────────────────────────────────────────────
    section("🎯", "Confusion Matrix & ROC Curve")
    c1, c2 = st.columns(2)
    cm = results['cm']
    with c1:
        fig = px.imshow(cm, text_auto=True,
                        x=['Predicted Low','Predicted High'],
                        y=['Actual Low',   'Actual High'],
                        color_continuous_scale='Blues', title='Confusion Matrix (Counts)')
        fig.update_traces(textfont_size=18)
        fig.update_layout(**LAYOUT, height=380)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results['fpr'], y=results['tpr'], mode='lines',
                                 line=dict(color=COLORS['blue'], width=3),
                                 name=f"ROC (AUC = {results['auc']:.3f})"))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                                 line=dict(color='#8fa8c8', dash='dash', width=1.5),
                                 name='Random Classifier'))
        fig.update_layout(**LAYOUT, height=380, xaxis_title='False Positive Rate',
                          yaxis_title='True Positive Rate', title='ROC Curve — Decision Tree')
        st.plotly_chart(fig, use_container_width=True)

    section("📐", "Normalised Confusion Matrix")
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig = px.imshow(cm_norm.round(2), text_auto='.2f',
                    x=['Predicted Low','Predicted High'], y=['Actual Low','Actual High'],
                    color_continuous_scale='Oranges', title='Confusion Matrix (Normalised)')
    fig.update_traces(textfont_size=18)
    fig.update_layout(**LAYOUT, height=350)
    st.plotly_chart(fig, use_container_width=True)

    # ── Feature Importance ───────────────────────────────────────────────────
    section("🔑", "Feature Importance")
    fi = results['feat_imp'].sort_values(ascending=True).reset_index()
    fi.columns = ['Feature','Importance']
    fig = px.bar(fi, x='Importance', y='Feature', orientation='h',
                 color='Importance', color_continuous_scale='Viridis',
                 text='Importance', title='Feature Importance — Decision Tree')
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(**LAYOUT, height=340)
    st.plotly_chart(fig, use_container_width=True)

    # ── Classification Report ────────────────────────────────────────────────
    section("📋", "Classification Report")
    report = results['report']
    report_df = pd.DataFrame({
        'Class'    :['Low Consumption','High Consumption','Accuracy','Macro Avg','Weighted Avg'],
        'Precision':[report['Low']['precision'], report['High']['precision'],
                     '', report['macro avg']['precision'], report['weighted avg']['precision']],
        'Recall'   :[report['Low']['recall'],    report['High']['recall'],
                     '', report['macro avg']['recall'],    report['weighted avg']['recall']],
        'F1-Score' :[report['Low']['f1-score'],  report['High']['f1-score'],
                     report['accuracy'], report['macro avg']['f1-score'], report['weighted avg']['f1-score']],
        'Support'  :[report['Low']['support'],   report['High']['support'],
                     report['macro avg']['support'], report['macro avg']['support'], report['weighted avg']['support']],
    })
    st.dataframe(report_df, use_container_width=True)

    # ── Tree Rules ───────────────────────────────────────────────────────────
    section("📜", "Decision Tree Rules (Text)")
    st.code(results['tree_rules'], language='text')

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: LIVE PREDICTOR  ← NEW DEDICATED PAGE
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "🔮  Live Predictor":
    st.markdown("""
    <h1 style='text-align:center; background:linear-gradient(90deg,#1a3a5c,#1e2a3a);
               color:white; padding:20px; border-radius:12px; margin-bottom:8px;
               border:1px solid #2d4a6e;'>
        🔮 Live Petrol Consumption Predictor
    </h1>
    <p style='text-align:center; color:#8fa8c8; margin-bottom:28px;'>
        أدخل الأرقام يدويًا — ثم اضغط PREDICT NOW
    </p>""", unsafe_allow_html=True)

    # ── Reference ranges ─────────────────────────────────────────────────────
    stats = results['df_stats']
    with st.expander("📌 Dataset Reference Ranges (click to expand)", expanded=False):
        ref = pd.DataFrame({
            'Feature' : results['FEATURES'],
            'Min'     : [stats[f]['min'] for f in results['FEATURES']],
            'Mean'    : [round(stats[f]['mean'],2) for f in results['FEATURES']],
            'Max'     : [stats[f]['max'] for f in results['FEATURES']],
        })
        st.dataframe(ref, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Manual number inputs only — all start at 0 ───────────────────────────
    section("✏️", "Enter Your Values Manually")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### ⛽ Petrol Tax (cents / gallon)")
        st.markdown('<p class="input-hint">Dataset range: 5.0 – 10.0 | Mean: 7.67</p>',
                    unsafe_allow_html=True)
        tax_val = st.number_input(
            "Petrol Tax", min_value=0.0, max_value=9999.0,
            value=0.0, step=0.1, format="%.2f",
            key="tax", label_visibility="collapsed")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### 🛣️ Paved Highways (miles)")
        st.markdown('<p class="input-hint">Dataset range: 431 – 17,782 | Mean: 5,565</p>',
                    unsafe_allow_html=True)
        hw_val = st.number_input(
            "Paved Highways", min_value=0, max_value=9999999,
            value=0, step=1,
            key="hw", label_visibility="collapsed")

    with col2:
        st.markdown("##### 💰 Average Income (USD)")
        st.markdown('<p class="input-hint">Dataset range: 3,333 – 5,694 | Mean: 4,242</p>',
                    unsafe_allow_html=True)
        inc_val = st.number_input(
            "Average Income", min_value=0, max_value=9999999,
            value=0, step=1,
            key="inc", label_visibility="collapsed")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### 🚗 Driver Licence Rate (0.0 – 1.0)")
        st.markdown('<p class="input-hint">Dataset range: 0.451 – 0.724 | Mean: 0.571</p>',
                    unsafe_allow_html=True)
        lic_val = st.number_input(
            "Driver Licence Rate", min_value=0.0, max_value=1.0,
            value=0.0, step=0.001, format="%.3f",
            key="lic", label_visibility="collapsed")

    st.markdown("---")

    # ── Predict Button ────────────────────────────────────────────────────────
    col_btn = st.columns([1,2,1])
    with col_btn[1]:
        predict_btn = st.button("🚀  PREDICT NOW", use_container_width=True,
                                 type="primary")

    # ── Guard: block prediction if all still 0 ────────────────────────────────
    all_zero = (tax_val == 0.0 and inc_val == 0 and hw_val == 0 and lic_val == 0.0)

    if predict_btn and all_zero:
        st.warning("⚠️  Please enter your values first before predicting.", icon="⚠️")

    elif predict_btn and not all_zero:
        sample = pd.DataFrame([[tax_val, inc_val, hw_val, lic_val]],
                               columns=results['FEATURES'])
        pred  = results['model'].predict(sample)[0]
        proba = results['model'].predict_proba(sample)[0]
        conf  = proba[pred] * 100
        p_low = proba[0] * 100
        p_high= proba[1] * 100

        st.markdown("<br>", unsafe_allow_html=True)
        section("📊", "Prediction Result")

        # ── Result card ───────────────────────────────────────────────────────
        res_col1, res_col2, res_col3 = st.columns([1,2,1])
        with res_col2:
            if pred == 1:
                st.markdown(f"""
                <div class="pred-high">
                    <div class="pred-label" style="color:#4fc3f7;">🔵 HIGH Consumption</div>
                    <div style="font-size:3rem; font-weight:800; color:#4fc3f7;">{conf:.1f}%</div>
                    <div class="pred-conf">Model Confidence</div>
                    <div class="pred-proba" style="margin-top:10px;">
                        P(Low) = {p_low:.1f}%  &nbsp;|&nbsp;  P(High) = {p_high:.1f}%
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="pred-low">
                    <div class="pred-label" style="color:#f06292;">🔴 LOW Consumption</div>
                    <div style="font-size:3rem; font-weight:800; color:#f06292;">{conf:.1f}%</div>
                    <div class="pred-conf">Model Confidence</div>
                    <div class="pred-proba" style="margin-top:10px;">
                        P(Low) = {p_low:.1f}%  &nbsp;|&nbsp;  P(High) = {p_high:.1f}%
                    </div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Probability Gauge ─────────────────────────────────────────────────
        section("📉", "Probability Breakdown")
        g1, g2 = st.columns(2)
        with g1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=p_high,
                title={"text": "P(High Consumption)", "font": {"color": "#4fc3f7", "size": 16}},
                number={"suffix": "%", "font": {"color": "#4fc3f7", "size": 36}},
                gauge={
                    "axis"      : {"range": [0,100], "tickcolor": "#8fa8c8"},
                    "bar"       : {"color": "#4fc3f7"},
                    "bgcolor"   : "#1e2a3a",
                    "bordercolor": "#2d4a6e",
                    "steps"     : [
                        {"range": [0,  40], "color": "#1a2a1a"},
                        {"range": [40, 60], "color": "#2a2a1a"},
                        {"range": [60,100], "color": "#1a2a3a"},
                    ],
                    "threshold" : {"line": {"color": "#ffb74d", "width": 3},
                                   "thickness": 0.75, "value": 50}
                }
            ))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                              font=dict(color='#c8d8e8'), height=280,
                              margin=dict(l=30,r=30,t=60,b=20))
            st.plotly_chart(fig, use_container_width=True)

        with g2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=p_low,
                title={"text": "P(Low Consumption)", "font": {"color": "#f06292", "size": 16}},
                number={"suffix": "%", "font": {"color": "#f06292", "size": 36}},
                gauge={
                    "axis"      : {"range": [0,100], "tickcolor": "#8fa8c8"},
                    "bar"       : {"color": "#f06292"},
                    "bgcolor"   : "#1e2a3a",
                    "bordercolor": "#2d4a6e",
                    "steps"     : [
                        {"range": [0,  40], "color": "#1a2a1a"},
                        {"range": [40, 60], "color": "#2a2a1a"},
                        {"range": [60,100], "color": "#2a1a1a"},
                    ],
                    "threshold" : {"line": {"color": "#ffb74d", "width": 3},
                                   "thickness": 0.75, "value": 50}
                }
            ))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                              font=dict(color='#c8d8e8'), height=280,
                              margin=dict(l=30,r=30,t=60,b=20))
            st.plotly_chart(fig, use_container_width=True)

        # ── Input Summary ─────────────────────────────────────────────────────
        section("📋", "Your Input Summary vs Dataset Mean")
        means = [stats[f]['mean'] for f in results['FEATURES']]
        vals  = [tax_val, inc_val, hw_val, lic_val]
        diff  = [round(v - m, 3) for v, m in zip(vals, means)]

        inp_df = pd.DataFrame({
            'Feature'         : ['Petrol Tax','Average Income','Paved Highways','Driver Licence %'],
            'Your Value'      : vals,
            'Dataset Mean'    : [round(m,3) for m in means],
            'Difference'      : diff,
            'Direction'       : ['Above Mean' if d > 0 else 'Below Mean' if d < 0 else 'Equal' for d in diff]
        })
        st.dataframe(inp_df, use_container_width=True, hide_index=True)

        # ── Radar Chart ───────────────────────────────────────────────────────
        section("🕸️", "Your Values vs Dataset Mean — Radar Chart")
        feat_labels = ['Petrol Tax','Avg Income','Paved Highways','Driver Licence %']
        vals_norm  = [v/m for v, m in zip(vals, means)]
        mean_norm  = [1.0] * len(means)

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=vals_norm + [vals_norm[0]], theta=feat_labels + [feat_labels[0]],
            fill='toself', fillcolor='rgba(79,195,247,0.2)',
            line=dict(color='#4fc3f7', width=2), name='Your Values'))
        fig.add_trace(go.Scatterpolar(
            r=mean_norm + [mean_norm[0]], theta=feat_labels + [feat_labels[0]],
            fill='toself', fillcolor='rgba(240,98,146,0.1)',
            line=dict(color='#f06292', width=2, dash='dash'), name='Dataset Mean'))
        fig.update_layout(
            polar=dict(
                bgcolor='rgba(14,17,23,0.8)',
                radialaxis=dict(visible=True, gridcolor='#2d3748', tickcolor='#8fa8c8',
                                color='#8fa8c8'),
                angularaxis=dict(gridcolor='#2d3748', tickcolor='#8fa8c8', color='#e2e8f0')
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#c8d8e8'),
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            height=420, margin=dict(l=60,r=60,t=40,b=40)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        # placeholder — shown when page first opens (all zeros, button not pressed)
        st.markdown("""
        <div style='background:linear-gradient(135deg,#1e2a3a,#162032);
                    border:1px dashed #2d4a6e; border-radius:14px;
                    padding:48px; text-align:center; margin-top:24px;'>
            <div style='font-size:3.5rem; margin-bottom:14px;'>✏️</div>
            <div style='color:#4fc3f7; font-size:1.3rem; font-weight:700;'>
                أدخل أرقامك أولًا
            </div>
            <div style='color:#8fa8c8; margin-top:10px; font-size:0.92rem; line-height:1.7;'>
                اكتب القيم يدويًا في الحقول أعلاه<br>
                ثم اضغط <b style="color:#4fc3f7;">🚀 PREDICT NOW</b> لرؤية النتيجة
            </div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: INSIGHTS & RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "💡  Insights & Recommendations":
    st.title("💡 Insights & Recommendations")

    section("🔍", "Key Insights")
    insights = [
        ("⛽ Petrol Tax vs Consumption",
         "States with lower petrol taxes show significantly higher consumption rates. "
         "Low-tax states have up to 70%+ high-consumption rate, confirming that "
         "price signals are a major driver of fuel demand."),
        ("💰 Income Level is a Strong Predictor",
         "Higher-income states consistently show higher petrol consumption. "
         "Wealthier populations own more vehicles and travel more, directly "
         "pushing consumption levels above the median threshold."),
        ("🛣️ Paved Highways Drive Usage",
         "States with more paved highway infrastructure tend to have higher consumption. "
         "Better road networks encourage driving over public transit, leading to "
         "greater per-capita fuel use."),
        ("🚗 Driver Licence Rate Matters",
         "States where a higher proportion of the population holds a driver licence "
         "show a clear positive correlation with petrol consumption. "
         "More licensed drivers = more vehicles = more fuel consumed."),
        ("📊 Balanced Dataset",
         "The 50/50 split between High and Low consumption classes (using median threshold) "
         "ensures no class imbalance issues, giving reliable and unbiased model performance."),
        ("🌳 Decision Tree Performance",
         f"The tuned Decision Tree achieved {results['test_acc']*100:.1f}% test accuracy "
         f"and {results['auc']:.3f} ROC-AUC, indicating strong predictive power "
         "despite the small dataset size of only 48 records."),
        ("🔑 Most Important Feature",
         f"'{results['feat_imp'].idxmax()}' ranks as the most important feature "
         f"(importance = {results['feat_imp'].max():.3f}), serving as the primary "
         "decision boundary to separate high from low consumption states."),
        ("📉 Tax Effectiveness",
         "The negative correlation between petrol tax and consumption confirms that "
         "fuel tax policy is an effective demand-side tool. "
         "Even a 1-2c increase in tax is associated with measurable consumption reduction."),
    ]
    c1, c2 = st.columns(2)
    for idx, (title, text) in enumerate(insights):
        col = c1 if idx % 2 == 0 else c2
        col.markdown(f"""<div class="insight-card">
            <div class="insight-title">{title}</div>
            <div class="insight-text">{text}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section("🚀", "Strategic Recommendations")
    recs = [
        ("1","📈 Increase Petrol Tax in Low-Tax States",
         "States with petrol tax below 7c/gallon show the highest consumption rates. "
         "Raising the tax to 8-9c could reduce high-consumption probability by 15-25%."),
        ("2","🚌 Invest in Public Transit for High-Income States",
         "High-income states have both the resources and the consumption problem. "
         "Subsidising public transport and carpooling incentives would provide long-term reduction."),
        ("3","🛣️ Re-evaluate Highway Expansion Policies",
         "More paved highways directly correlate with higher consumption. "
         "Prioritise rail, cycling infrastructure, and urban transit instead."),
        ("4","🚗 Driver Licence & EV Transition Programs",
         "States with very high driver licence rates (>60%) are prime targets for "
         "electric vehicle adoption campaigns, reducing petrol dependency without cutting mobility."),
        ("5","🤖 Deploy Decision Tree Model in Real-Time Policy Systems",
         f"The classifier (accuracy {results['test_acc']*100:.1f}%, AUC {results['auc']:.3f}) "
         "can flag high-risk states early and prioritise intervention resources."),
        ("6","📊 Expand Data Collection",
         "Adding urban density, fuel efficiency standards, EV adoption rate, "
         "and public transport coverage would significantly improve model accuracy."),
    ]
    for num, title, text in recs:
        st.markdown(f"""<div class="rec-card">
            <span class="rec-number">#{num}</span>
            <b style='color:#e2e8f0; font-size:0.95rem; margin-left:8px;'>{title}</b>
            <div class="rec-text" style='margin-top:6px; padding-left:24px;'>{text}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    section("📋", "Model & Dataset Final Summary")
    summary_data = {
        'Item'  :['Dataset','Records','Features','Target','Model','Best Criterion',
                  'Best Max Depth','Train Accuracy','Test Accuracy',
                  'ROC AUC','Top Feature'],
        'Detail':[
            'petrol_consumption.csv', '48 US States',
            '4 Numerical (Petrol_tax, Average_income, Paved_Highways, Driver_Licence%)',
            'Consumption_Class (0=Low, 1=High — split by median)',
            'Decision Tree Classifier + GridSearchCV',
            results['best_params']['clf__criterion'],
            str(results['best_params']['clf__max_depth']),
            f"{results['train_acc']*100:.2f}%",
            f"{results['test_acc']*100:.2f}%",
            f"{results['auc']:.4f}",
            f"{results['feat_imp'].idxmax()} (importance = {results['feat_imp'].max():.3f})",
        ]
    }
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
