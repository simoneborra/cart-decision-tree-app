import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree, _tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import graphviz
import io
import os

# Helper functions for backwards compatibility across different Streamlit versions
def safe_dataframe(df_data):
    try:
        st.dataframe(df_data, use_container_width=True)
    except TypeError:
        st.dataframe(df_data)

def safe_graphviz(dot):
    try:
        st.graphviz_chart(dot, use_container_width=True)
    except TypeError:
        st.graphviz_chart(dot)

def safe_button_primary(label):
    try:
        return st.button(label, type="primary")
    except TypeError:
        return st.button(label)

# Function to extract rules for all terminal leaf nodes
def extract_terminal_node_rules(tree_clf, feat_names, classes):
    tree_ = tree_clf.tree_
    leaf_rules = []
    
    def recurse(node_id, current_rules):
        is_leaf = tree_.children_left[node_id] == _tree.TREE_LEAF
        if is_leaf:
            node_samples = tree_.n_node_samples[node_id]
            value = tree_.value[node_id][0]
            total_val = value.sum()
            percentages = (value / total_val) * 100
            maj_idx = np.argmax(value)
            maj_class = classes[maj_idx]
            maj_pct = percentages[maj_idx]
            
            rule_str = " AND ".join(current_rules) if current_rules else "Tutte le osservazioni (Nodo Radice)"
            dist_str = ", ".join([f"{classes[i]}: {percentages[i]:.1f}% ({int(value[i])})" for i in range(len(classes))])
            
            leaf_rules.append({
                "ID Nodo Foglia": f"Nodo #{node_id}",
                "Regola Decisionale (Percorso Split)": rule_str,
                "Classe Maggioritaria": maj_class,
                "Probabilità / Purezza %": f"{maj_pct:.1f}%",
                "Frequenza Nodi": f"{node_samples} ({node_samples/tree_.n_node_samples[0]*100:.1f}%)",
                "Distribuzione Target": dist_str
            })
        else:
            f_idx = tree_.feature[node_id]
            f_name = feat_names[f_idx]
            th = tree_.threshold[node_id]
            
            left_child = tree_.children_left[node_id]
            right_child = tree_.children_right[node_id]
            
            recurse(left_child, current_rules + [f"{f_name} <= {th:.2f}"])
            recurse(right_child, current_rules + [f"{f_name} > {th:.2f}"])
            
    recurse(0, [])
    return pd.DataFrame(leaf_rules)

# Page Configuration
st.set_page_config(
    page_title="CART Decision Tree Explorer",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        color: #1E3A8A;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .prediction-box {
        background-color: #EFF6FF;
        border: 2px solid #3B82F6;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .prediction-title {
        font-size: 1.2rem;
        color: #1E40AF;
        font-weight: bold;
    }
    .prediction-value {
        font-size: 2.2rem;
        color: #1D4ED8;
        font-weight: 900;
    }
    .rule-box {
        background-color: #F8FAFC;
        border-left: 4px solid #0284C7;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        border-radius: 0 6px 6px 0;
        font-family: monospace;
    }
    .tree-container {
        width: 100%;
        overflow-x: auto;
        overflow-y: auto;
        max-height: 850px;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 1rem;
        background-color: #FFFFFF;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🌳 Applicazione Interattiva CART (Classification And Regression Trees)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Genera dati artificiali o carica un file Excel per costruire, analizzare, visualizzare l\'albero decisionale ed effettuare predizioni interattive su nuovi casi.</div>', unsafe_allow_html=True)

# Session state initialization for prediction history
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Sidebar - Configuration
st.sidebar.header("⚙️ Configurazione Dati e Modello")

data_source = st.sidebar.radio(
    "Sorgente Dati:",
    ["Dati Artificiali (Sintetici)", "Importa File Excel (.xlsx)"],
    index=0
)

# Global variables
df = None
target_col = "Target"
feature_types = {}

if data_source == "Dati Artificiali (Sintetici)":
    st.sidebar.subheader("📊 Generazione Dati Sintetici")
    
    n_samples = st.sidebar.number_input("Numero di Osservazioni (N):", min_value=20, max_value=5000, value=200, step=10)
    n_target_modalities = st.sidebar.number_input("Numero Modalità Target:", min_value=2, max_value=6, value=2, step=1)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Variabili Numeriche Continue:**")
    n_num_cont = st.sidebar.number_input("Numero Variabili Continue:", min_value=0, max_value=10, value=2, step=1)
    
    cont_params = []
    for i in range(n_num_cont):
        st.sidebar.caption(f"Var. Continua {i+1}")
        col_c1, col_c2 = st.sidebar.columns(2)
        v_min = col_c1.number_input(f"Min C{i+1}", value=0.0, key=f"min_c_{i}")
        v_max = col_c2.number_input(f"Max C{i+1}", value=100.0, key=f"max_c_{i}")
        cont_params.append((v_min, v_max))
        
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Variabili Ordinali:**")
    n_ord = st.sidebar.number_input("Numero Variabili Ordinali:", min_value=0, max_value=10, value=1, step=1)
    ord_levels = []
    for i in range(n_ord):
        n_lev = st.sidebar.number_input(f"Modalità Ordinale {i+1}:", min_value=2, max_value=10, value=3, key=f"ord_lev_{i}")
        ord_levels.append(n_lev)
        
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Variabili Qualitative Nominali:**")
    n_nom = st.sidebar.number_input("Numero Variabili Nominali:", min_value=0, max_value=10, value=1, step=1)
    nom_levels = []
    for i in range(n_nom):
        n_lev = st.sidebar.number_input(f"Modalità Nominale {i+1}:", min_value=2, max_value=10, value=3, key=f"nom_lev_{i}")
        nom_levels.append(n_lev)
        
    seed = st.sidebar.number_input("Seed Casuale:", value=42, step=1)
    
    # Generate Synthetic Dataset
    np.random.seed(seed)
    data_dict = {}
    
    for i, (vmin, vmax) in enumerate(cont_params):
        col_name = f"Cont_{i+1}"
        data_dict[col_name] = np.random.uniform(vmin, vmax, n_samples)
        feature_types[col_name] = "continuous"
        
    for i, n_lev in enumerate(ord_levels):
        col_name = f"Ord_{i+1}"
        data_dict[col_name] = np.random.randint(1, n_lev + 1, n_samples)
        feature_types[col_name] = "ordinal"
        
    for i, n_lev in enumerate(nom_levels):
        col_name = f"Nom_{i+1}"
        cats = [f"Cat_{chr(65+j)}" for j in range(n_lev)]
        data_dict[col_name] = np.random.choice(cats, n_samples)
        feature_types[col_name] = "nominal"
        
    df = pd.DataFrame(data_dict)
    
    # Target Generation logic
    score = np.zeros(n_samples)
    if len(cont_params) > 0:
        score += (df["Cont_1"] - df["Cont_1"].mean()) / (df["Cont_1"].std() + 1e-5)
    if len(ord_levels) > 0:
        score += df["Ord_1"] * 0.5
        
    target_probs = np.zeros((n_samples, n_target_modalities))
    for k in range(n_target_modalities):
        target_probs[:, k] = np.exp(-(score - k)**2 / 2.0)
    target_probs = target_probs / target_probs.sum(axis=1, keepdims=True)
    
    target_classes = [f"Modalità_{k+1}" for k in range(n_target_modalities)]
    df["Target"] = [np.random.choice(target_classes, p=p) for p in target_probs]
    target_col = "Target"

else:
    st.sidebar.subheader("📁 Caricamento File Excel")
    uploaded_file = st.sidebar.file_uploader("Scegli un file Excel (.xlsx, .xls)", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            st.sidebar.success(f"File caricato con successo: {df.shape[0]} righe, {df.shape[1]} colonne.")
            
            target_col = st.sidebar.selectbox("Seleziona la Variabile Target:", options=df.columns.tolist(), index=len(df.columns)-1)
            
            feature_cols = [c for c in df.columns if c != target_col]
            st.sidebar.markdown("**Tipologia Variabili Esplicative:**")
            
            for col in feature_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    default_type = "continuous" if df[col].nunique() > 10 else "ordinal"
                else:
                    default_type = "nominal"
                    
                type_sel = st.sidebar.selectbox(
                    f"Tipo per {col}:",
                    options=["continuous", "ordinal", "nominal"],
                    format_func=lambda x: "Numerica Continua (Istogramma)" if x=="continuous" else ("Numerica Discreta / Ordinale (Barre)" if x=="ordinal" else "Qualitativa Nominale (Torta)"),
                    index=["continuous", "ordinal", "nominal"].index(default_type),
                    key=f"type_select_{col}"
                )
                feature_types[col] = type_sel
        except Exception as e:
            st.error(f"Errore nella lettura del file Excel: {e}")
    else:
        st.info("👋 Inserisci un file Excel oppure seleziona 'Dati Artificiali' per iniziare.")

# Hyperparameters Section
st.sidebar.markdown("---")
st.sidebar.subheader("🌲 Iperparametri Albero CART")

impurity_criterion = st.sidebar.selectbox(
    "Criterio di Impurità:",
    options=["gini", "entropy"],
    format_func=lambda x: "Indice di Gini" if x=="gini" else "Entropia di Shannon"
)

max_depth = st.sidebar.slider("Profondità Massima (max_depth):", min_value=1, max_value=10, value=3, step=1)
min_samples_split = st.sidebar.slider("Minimo Osservazioni per Split (min_samples_split):", min_value=2, max_value=100, value=5, step=1)

# Main Area - Tabs
if df is not None:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Vista Dati",
        "📊 Distribuzioni ed EDA",
        "🌳 Albero Decisionale CART",
        "📈 Prestazioni e Statistiche",
        "🔮 Predizione Interattiva"
    ])
    
    # Preprocessing & Model Fitting for global tabs
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    
    X_encoded = pd.get_dummies(X, drop_first=False)
    feature_names = X_encoded.columns.tolist()
    class_names = [str(c) for c in np.unique(y)]
    
    clf = DecisionTreeClassifier(
        criterion=impurity_criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    clf.fit(X_encoded, y)

    # ---------------- TAB 1: DATA VIEW ----------------
    with tab1:
        st.subheader("📌 Anteprima del Dataset")
        col_t1, col_t2, col_t3 = st.columns(3)
        col_t1.metric("Totale Osservazioni", df.shape[0])
        col_t2.metric("Numero Variabili Esplicative", len(df.columns) - 1)
        col_t3.metric("Modalità Target", df[target_col].nunique())
        
        safe_dataframe(df.head(50))
        
        # Download Excel button
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Data')
        st.download_button(
            label="📥 Scarica Dataset Corrente in Excel",
            data=buffer.getvalue(),
            file_name="dataset_cart.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    # ---------------- TAB 2: EDA & DISTRIBUTIONS ----------------
    with tab2:
        st.subheader("📊 Analisi Esplorativa delle Distribuzioni")
        st.write("Visualizzazione della variabile Target e delle singole variabili esplicative secondo il tipo di variabile:")
        
        # Target Distribution
        st.markdown("#### 🎯 Distribuzione Variabile Target")
        fig_t, ax_t = plt.subplots(figsize=(6, 3.5))
        target_counts = df[target_col].value_counts().sort_index()
        colors = plt.cm.Set2(np.linspace(0, 1, len(target_counts)))
        bars = ax_t.bar(target_counts.index.astype(str), target_counts.values, color=colors, edgecolor="black")
        ax_t.set_title(f"Distribuzione Target: {target_col}", fontsize=11, fontweight="bold")
        ax_t.set_ylabel("Frequenza")
        ax_t.set_xlabel("Modalità Target")
        for b in bars:
            h = b.get_height()
            ax_t.annotate(f'{h} ({h/len(df)*100:.1f}%)', xy=(b.get_x() + b.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig_t)
        
        st.markdown("---")
        st.markdown("#### 🔍 Distribuzione delle Variabili Esplicative")
        
        feature_cols = [c for c in df.columns if c != target_col]
        
        if len(feature_cols) > 0:
            cols_per_row = 2
            for idx in range(0, len(feature_cols), cols_per_row):
                cols_st = st.columns(cols_per_row)
                for i in range(cols_per_row):
                    if idx + i < len(feature_cols):
                        f_col = feature_cols[idx + i]
                        f_type = feature_types.get(f_col, "continuous")
                        
                        fig, ax = plt.subplots(figsize=(5.5, 3.5))
                        
                        if f_type == "continuous":
                            sns.histplot(df[f_col], kde=True, ax=ax, color="#2B5C8F", edgecolor="black")
                            ax.set_title(f"Istogramma (Continua): {f_col}", fontsize=10, fontweight="bold")
                            ax.set_xlabel(f_col)
                            ax.set_ylabel("Frequenza")
                            
                        elif f_type == "ordinal":
                            counts = df[f_col].value_counts().sort_index()
                            bars = ax.bar(counts.index.astype(str), counts.values, color="#4A90E2", edgecolor="black")
                            ax.set_title(f"Grafico a Barre Verticale (Ord/Disc): {f_col}", fontsize=10, fontweight="bold")
                            ax.set_xlabel(f_col)
                            ax.set_ylabel("Conteggio")
                            for b in bars:
                                h = b.get_height()
                                ax.annotate(f'{h}', xy=(b.get_x() + b.get_width()/2, h),
                                            xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8)
                                
                        elif f_type == "nominal":
                            counts = df[f_col].value_counts()
                            pie_colors = plt.cm.Set3(np.linspace(0, 1, len(counts)))
                            ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=140, colors=pie_colors, wedgeprops=dict(edgecolor='black'))
                            ax.set_title(f"Grafico a Torta (Nominale): {f_col}", fontsize=10, fontweight="bold")
                            
                        plt.tight_layout()
                        cols_st[i].pyplot(fig)

    # ---------------- TAB 3: CART TREE VISUALIZATION & RULES ----------------
    with tab3:
        st.subheader("🌳 Grafo dell'Albero Decisionale CART")
        st.write("I nodi sono numerati in sequenza (**Nodo #0, Nodo #1, ...**) e attribuiti, con colori diversi, alla modalità della variabile Target con maggiore frequenza. All'interno di ciascun nodo è visualizzata la regola di split e la distribuzione percentuale delle unità rispetto al Target.")
        
        # Display Controls for sizing and viewing options
        col_ctrl1, col_ctrl2 = st.columns([1.5, 1])
        
        with col_ctrl1:
            render_mode = st.radio(
                "🎨 Modalità di Visualizzazione Grafica dell'Albero:",
                [
                    "🖥️ SVG Vettoriale Scrollabile & Zoomabile (Raccomandata per adattamento schermo)",
                    "🖼️ Matplotlib plot_tree (Immagine con dimensioni personalizzate W x H)",
                    "⚡ Vista Native Streamlit Graphviz"
                ],
                index=0
            )
            
        with col_ctrl2:
            if "SVG Vettoriale" in render_mode:
                tree_width_px = st.slider("🔍 Larghezza dell'Albero (Pixel):", min_value=600, max_value=3500, value=1300, step=100)
            elif "Matplotlib" in render_mode:
                fig_w = st.slider("📐 Larghezza Figura Matplotlib (pollici):", min_value=6, max_value=30, value=14, step=1)
                fig_h = st.slider("📐 Altezza Figura Matplotlib (pollici):", min_value=4, max_value=20, value=8, step=1)

        # Graphviz tree builder with explicit node numbering headers
        def build_graphviz_tree(tree_clf, feat_names, classes, criterion_name="Gini"):
            tree_ = tree_clf.tree_
            
            color_palette = ["#E6F0FA", "#FFEAE6", "#E6F9E6", "#FFF5E6", "#F5E6FA", "#E6FAF5", "#FAE6F0"]
            border_palette = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD", "#17BECF", "#E377C2"]
            
            dot = graphviz.Digraph(comment="CART Decision Tree", format="svg")
            dot.attr(rankdir="TB")
            dot.attr("node", shape="box", style="filled,rounded", fontname="Helvetica", fontsize="10")
            dot.attr("edge", fontname="Helvetica", fontsize="9")
            
            def recurse(node_id):
                node_samples = tree_.n_node_samples[node_id]
                value = tree_.value[node_id][0]
                total_val = value.sum()
                percentages = (value / total_val) * 100
                
                maj_idx = np.argmax(value)
                maj_class = classes[maj_idx]
                
                imp_val = tree_.impurity[node_id]
                is_leaf = tree_.children_left[node_id] == _tree.TREE_LEAF
                
                fill_c = color_palette[maj_idx % len(color_palette)]
                border_c = border_palette[maj_idx % len(border_palette)]
                
                if not is_leaf:
                    f_idx = tree_.feature[node_id]
                    th = tree_.threshold[node_id]
                    f_name = feat_names[f_idx]
                    split_str = f"<b>{f_name} &le; {th:.2f}</b>"
                    node_header = f"<b><font color='#1E3A8A' point-size='12'>NODO #{node_id}</font></b>"
                else:
                    split_str = "<b>NODO FOGLIA</b>"
                    node_header = f"<b><font color='#1E3A8A' point-size='12'>NODO #{node_id} (FOGLIA)</font></b>"
                    
                crit_title = "Indice Gini" if criterion_name == "gini" else "Entropia Shannon"
                
                dist_rows = []
                for c_idx, c_name in enumerate(classes):
                    pct = percentages[c_idx]
                    cnt = int(value[c_idx])
                    dist_rows.append(f"<tr><td align='left'>{c_name}:</td><td align='right'><b>{pct:.1f}%</b> ({cnt})</td></tr>")
                dist_table = "".join(dist_rows)
                
                label_html = f"""<
                <table border='0' cellborder='0' cellspacing='2' cellpadding='2'>
                    <tr><td colspan='2' align='center'>{node_header}</td></tr>
                    <tr><td colspan='2' align='center'>{split_str}</td></tr>
                    <hr/>
                    <tr><td align='left'>{crit_title}:</td><td align='right'>{imp_val:.3f}</td></tr>
                    <tr><td align='left'>Osservazioni:</td><td align='right'>{node_samples} ({node_samples/len(df)*100:.1f}%)</td></tr>
                    <tr><td align='left'>Classe Maggioranza:</td><td align='right'><b><font color='{border_c}'>{maj_class}</font></b></td></tr>
                    <hr/>
                    <tr><td colspan='2' align='center'><b>Distribuzione Target %:</b></td></tr>
                    {dist_table}
                </table>
                >"""
                
                dot.node(
                    str(node_id),
                    label=label_html,
                    fillcolor=fill_c,
                    color=border_c,
                    penwidth="2.0"
                )
                
                if not is_leaf:
                    left_child = tree_.children_left[node_id]
                    right_child = tree_.children_right[node_id]
                    
                    dot.edge(str(node_id), str(left_child), label="Vero (Si)")
                    dot.edge(str(node_id), str(right_child), label="Falso (No)")
                    
                    recurse(left_child)
                    recurse(right_child)

            recurse(0)
            return dot

        # Try rendering Graphviz with fallback to Matplotlib if Graphviz system executable is missing on Windows
        graphviz_error_occurred = False
        dot_tree = None

        if "Matplotlib" in render_mode:
            fig_tree, ax_tree = plt.subplots(figsize=(fig_w, fig_h))
            plot_tree(
                clf,
                feature_names=feature_names,
                class_names=class_names,
                node_ids=True,
                filled=True,
                rounded=True,
                ax=ax_tree,
                fontsize=9
            )
            plt.tight_layout()
            st.pyplot(fig_tree)
        else:
            try:
                dot_tree = build_graphviz_tree(clf, feature_names, class_names, impurity_criterion)
                
                if "SVG Vettoriale" in render_mode:
                    svg_data = dot_tree.pipe(format="svg").decode("utf-8")
                    
                    html_tree = f"""
                    <div class="tree-container">
                        <div style="display: inline-block; width: {tree_width_px}px; max-width: none;">
                            {svg_data}
                        </div>
                    </div>
                    """
                    st.markdown(html_tree, unsafe_allow_html=True)
                else:
                    safe_graphviz(dot_tree)

            except Exception as e:
                graphviz_error_occurred = True
                st.warning("⚠️ **Eseguibile Graphviz non trovato nel PATH di Windows.** Passaggio automatico alla visualizzazione **Matplotlib plot_tree** per evitare blocchi dell'app.")
                st.info("💡 *Per abilitare il render vettoriale SVG*: installa Graphviz per Windows (es. `conda install python-graphviz` oppure scarica l'installer da graphviz.org e aggiungi `bin` alle Variabili d'Ambiente PATH di Windows).")
                
                # Fallback plot with node numbering enabled
                fig_tree, ax_tree = plt.subplots(figsize=(14, 8))
                plot_tree(
                    clf,
                    feature_names=feature_names,
                    class_names=class_names,
                    node_ids=True,
                    filled=True,
                    rounded=True,
                    ax=ax_tree,
                    fontsize=9
                )
                plt.tight_layout()
                st.pyplot(fig_tree)

        # Download buttons section
        if dot_tree is not None and not graphviz_error_occurred:
            st.markdown("---")
            st.markdown("#### 📥 Scarica il Grafico dell'Albero Decisionale")
            col_d1, col_d2 = st.columns(2)
            
            with col_d1:
                try:
                    png_data = dot_tree.pipe(format="png")
                    st.download_button(
                        label="📥 Scarica Albero Decisionale (PNG 300 DPI)",
                        data=png_data,
                        file_name="albero_decisionale_cart.png",
                        mime="image/png"
                    )
                except Exception:
                    pass
                
            with col_d2:
                try:
                    svg_data_bytes = dot_tree.pipe(format="svg")
                    st.download_button(
                        label="📥 Scarica Albero Decisionale (SVG Vettoriale)",
                        data=svg_data_bytes,
                        file_name="albero_decisionale_cart.svg",
                        mime="image/svg+xml"
                    )
                except Exception:
                    pass

        # ---------------- TERMINAL LEAF NODE RULES SECTION ----------------
        st.markdown("---")
        st.markdown("### 📜 Regole Decisionali dei Nodi Terminali (Foglie)")
        st.write("Consulta o esporta le regole logiche trasparenti (percorsi di split) relative ad ogni nodo terminale dell'albero CART:")

        show_rules = st.checkbox("🔍 Mostra Tabella e Schede delle Regole per i Nodi Terminali", value=True)

        if show_rules:
            rules_df = extract_terminal_node_rules(clf, feature_names, class_names)
            
            # Display summary dataframe
            safe_dataframe(rules_df)
            
            # Export rules button
            buf_rules = io.BytesIO()
            with pd.ExcelWriter(buf_rules, engine='openpyxl') as writer:
                rules_df.to_excel(writer, index=False, sheet_name='Regole_Nodi_Terminali')
            st.download_button(
                label="📥 Scarica Regole dei Nodi Terminali in Excel",
                data=buf_rules.getvalue(),
                file_name="regole_nodi_terminali_cart.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            # Card view for individual node rules
            st.markdown("#### 📌 Dettaglio Schede dei Nodi Terminali:")
            for r_idx, row in rules_df.iterrows():
                st.markdown(f"""
                <div class="rule-box" style="border-left-color: #1E3A8A; background-color: #F1F5F9;">
                    <b>{row['ID Nodo Foglia']}</b> &rarr; <b>Classe Predetta: <font color='#1D4ED8'>{row['Classe Maggioritaria']}</font></b> (Purezza: {row['Probabilità / Purezza %']}, Osservazioni: {row['Frequenza Nodi']})<br/>
                    <small><b>Percorso di Split:</b> <code>{row['Regola Decisionale (Percorso Split)']}</code></small><br/>
                    <small><b>Distribuzione Interna:</b> {row['Distribuzione Target']}</small>
                </div>
                """, unsafe_allow_html=True)

    # ---------------- TAB 4: METRICS & PERFORMANCE ----------------
    with tab4:
        st.subheader("📈 Valutazione del Modello CART")
        
        y_pred = clf.predict(X_encoded)
        acc = accuracy_score(y, y_pred)
        
        st.metric("Accuratezza del Modello (Training Set)", f"{acc*100:.2f}%")
        
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.markdown("#### 🧩 Matrice di Confusione")
            cm = confusion_matrix(y, y_pred, labels=class_names)
            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax_cm)
            ax_cm.set_ylabel("Classe Reale")
            ax_cm.set_xlabel("Classe Predetta")
            ax_cm.set_title("Matrice di Confusione")
            st.pyplot(fig_cm)
            
        with col_m2:
            st.markdown("#### 🌟 Importanza delle Variabili (Feature Importance)")
            importances = clf.feature_importances_
            fi_df = pd.DataFrame({"Variabile": feature_names, "Importanza": importances}).sort_values("Importanza", ascending=True)
            
            fig_fi, ax_fi = plt.subplots(figsize=(5, 4))
            ax_fi.barh(fi_df["Variabile"], fi_df["Importanza"], color="#2B5C8F")
            ax_fi.set_title("Importanza delle Feature (CART)")
            ax_fi.set_xlabel("Punteggio Importanza")
            st.pyplot(fig_fi)
            
        st.markdown("#### 📜 Report di Classificazione Dettagliato")
        report_dict = classification_report(y, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        safe_dataframe(report_df.style.format("{:.3f}"))

    # ---------------- TAB 5: INTERACTIVE PREDICTION ----------------
    with tab5:
        st.subheader("🔮 Modulo di Predizione Interattiva su Nuovi Casi")
        st.write("Imposta i valori delle variabili esplicative per simulare un nuovo profilo e classificare in tempo reale la modalità del Target secondo l'albero CART addestrato.")
        
        st.markdown("### 🎛️ Inserisci i Valori per il Nuovo Caso")
        
        feature_cols = [c for c in df.columns if c != target_col]
        input_data = {}
        
        # Grid layout for inputs
        col_inputs = st.columns(min(3, max(1, len(feature_cols))))
        
        for idx, col in enumerate(feature_cols):
            c_widget = col_inputs[idx % len(col_inputs)]
            f_type = feature_types.get(col, "continuous")
            
            with c_widget:
                if f_type == "continuous":
                    min_val = float(df[col].min())
                    max_val = float(df[col].max())
                    mean_val = float(df[col].mean())
                    if min_val == max_val:
                        max_val = min_val + 10.0
                    input_data[col] = st.slider(
                        f"**{col}** (Continua)",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        step=(max_val - min_val) / 100.0,
                        key=f"pred_in_{col}"
                    )
                elif f_type == "ordinal":
                    unique_ord = sorted(list(df[col].unique()))
                    input_data[col] = st.select_slider(
                        f"**{col}** (Ordinale)",
                        options=unique_ord,
                        value=unique_ord[len(unique_ord)//2],
                        key=f"pred_in_{col}"
                    )
                elif f_type == "nominal":
                    unique_cats = sorted([str(cat) for cat in df[col].unique()])
                    input_data[col] = st.selectbox(
                        f"**{col}** (Nominale)",
                        options=unique_cats,
                        index=0,
                        key=f"pred_in_{col}"
                    )

        st.markdown("---")
        
        # Button or instant prediction
        if safe_button_primary("🚀 Esegui Classificazione / Predizione"):
            # Prepare single instance DataFrame
            new_df = pd.DataFrame([input_data])
            new_encoded = pd.get_dummies(new_df, drop_first=False)
            new_encoded = new_encoded.reindex(columns=X_encoded.columns, fill_value=0)
            
            # Prediction & Probabilities
            pred_class = clf.predict(new_encoded)[0]
            pred_probs = clf.predict_proba(new_encoded)[0]
            
            col_res1, col_res2 = st.columns([1, 1.5])
            
            with col_res1:
                st.markdown(f"""
                <div class="prediction-box">
                    <div class="prediction-title">Modalità Target Predetta:</div>
                    <div class="prediction-value">{pred_class}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Class probabilities breakdown
                st.markdown("#### 📊 Probabilità per Ciascuna Classe:")
                prob_df = pd.DataFrame({
                    "Classe Target": clf.classes_,
                    "Probabilità (%)": [p * 100 for p in pred_probs]
                }).sort_values("Probabilità (%)", ascending=False)
                
                fig_p, ax_p = plt.subplots(figsize=(5, 3))
                bars = ax_p.barh(prob_df["Classe Target"], prob_df["Probabilità (%)"], color="#3B82F6", edgecolor="black")
                ax_p.set_xlim(0, 105)
                ax_p.set_xlabel("Probabilità (%)")
                ax_p.set_title("Distribuzione delle Probabilità")
                for b in bars:
                    w = b.get_width()
                    ax_p.annotate(f'{w:.1f}%', xy=(w, b.get_y() + b.get_height()/2),
                                xytext=(3, 0), textcoords="offset points", ha='left', va='center', fontsize=9, fontweight="bold")
                plt.tight_layout()
                st.pyplot(fig_p)

            with col_res2:
                st.markdown("#### 🛣️ Percorso Decisionale nell'Albero (Decision Path)")
                st.write("Regole di split attraversate dal nuovo caso per raggiungere la foglia di destinazione:")
                
                node_indicator = clf.decision_path(new_encoded)
                leaf_id = clf.apply(new_encoded)[0]
                node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
                
                tree_ = clf.tree_
                
                for step, node_id in enumerate(node_index):
                    if node_id != leaf_id:
                        f_idx = tree_.feature[node_id]
                        f_name = feature_names[f_idx]
                        th = tree_.threshold[node_id]
                        val = new_encoded.iloc[0, f_idx]
                        satisfied = val <= th
                        cond_symbol = "&le;" if satisfied else "&gt;"
                        res_str = "Vero (Sì)" if satisfied else "Falso (No)"
                        
                        st.markdown(f"""
                        <div class="rule-box">
                            <b>Passo {step+1} (Nodo #{node_id}):</b> {f_name} ({val:.2f}) {cond_symbol} {th:.2f} &rarr; <b>{res_str}</b>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="rule-box" style="border-left-color: #22C55E; background-color: #F0FDF4;">
                            <b>Nodo Foglia Finale #{node_id}:</b> Assegnazione alla classe <b>{pred_class}</b>
                        </div>
                        """, unsafe_allow_html=True)

            # Log prediction to history
            input_log = input_data.copy()
            input_log["Target Predetto"] = pred_class
            for c_idx, c_name in enumerate(clf.classes_):
                input_log[f"Prob_{c_name}"] = f"{pred_probs[c_idx]*100:.1f}%"
            st.session_state.prediction_history.append(input_log)

        # Show prediction log if available
        if len(st.session_state.prediction_history) > 0:
            st.markdown("---")
            st.markdown("### 📜 Storico delle Predizioni Effettuate")
            log_df = pd.DataFrame(st.session_state.prediction_history)
            safe_dataframe(log_df)
            
            buf_pred = io.BytesIO()
            with pd.ExcelWriter(buf_pred, engine='openpyxl') as writer:
                log_df.to_excel(writer, index=False, sheet_name='Predizioni')
            st.download_button(
                label="📥 Scarica Storico Predizioni in Excel",
                data=buf_pred.getvalue(),
                file_name="storico_predizioni_cart.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
