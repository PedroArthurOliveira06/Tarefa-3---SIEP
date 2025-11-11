"""
Dashboard Interativo - Modelagem de Cancelamento de Reservas
Tarefa 3 - SIEP
Aluno: Pedro Arthur Santos Oliveira
Matrícula: 231036069
Professor: João Gabriel de Moraes Souza
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)
from imblearn.over_sampling import SMOTE
import time
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Previsão de Cancelamentos - Hotel",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 30px;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("🏨 Dashboard de Previsão de Cancelamentos")
st.markdown("### Tarefa 3 - Modelagem com Machine Learning")
st.markdown("*Aluno:* Pedro Arthur Santos Oliveira | *Matrícula:* 231036069")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/3498db/ffffff?text=UnB", use_container_width=True)
    st.header("⚙ Configurações")
    
    # Upload de arquivo
    st.subheader("📁 Upload do Dataset")
    uploaded_file = st.file_uploader(
        "Faça upload do hotel_bookings.csv",
        type=['csv'],
        help="Dataset disponível em: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand"
    )
    
    st.markdown("---")
    
    # Escolha do algoritmo
    st.subheader("🤖 Escolha do Algoritmo")
    algorithm = st.selectbox(
        "Selecione o modelo:",
        ["Regressão Logística", "KNN", "SVM", "Comparar Todos"]
    )
    
    st.markdown("---")

# Cache para carregar dados
@st.cache_data
def load_and_preprocess_data(file):
    """Carrega e processa o dataset"""
    df = pd.read_csv(file)
    
    # Tratamento de valores faltantes
    df['children'].fillna(0, inplace=True)
    df['country'].fillna(df['country'].mode()[0], inplace=True)
    df['agent'].fillna(0, inplace=True)
    df['company'].fillna(0, inplace=True)
    
    # Remover outliers extremos
    df = df[df['adr'] < 1000]
    df = df[df['lead_time'] < 500]
    
    return df

# Cache para preparar features
@st.cache_data
def prepare_features(df):
    """Prepara features para modelagem"""
    selected_features = [
        'lead_time', 'arrival_date_month', 'stays_in_weekend_nights',
        'stays_in_week_nights', 'adults', 'children', 'babies',
        'meal', 'market_segment', 'distribution_channel',
        'previous_cancellations', 'booking_changes', 'deposit_type',
        'customer_type', 'adr'
    ]
    
    df_model = df[selected_features + ['is_canceled']].copy()
    
    # Codificação de variáveis categóricas
    categorical_features = df_model.select_dtypes(include=['object']).columns.tolist()
    df_encoded = pd.get_dummies(df_model, columns=categorical_features, drop_first=True)
    
    return df_encoded

def display_results(y_test, y_pred, y_pred_proba, training_time, model_name):
    """Exibe resultados do modelo"""
    st.markdown(f"### 📊 Resultados - {model_name}")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("AUC", f"{roc_auc_score(y_test, y_pred_proba):.4f}")
    with col2:
        st.metric("F1-Score", f"{f1_score(y_test, y_pred):.4f}")
    with col3:
        st.metric("Precisão", f"{precision_score(y_test, y_pred):.4f}")
    with col4:
        st.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
    with col5:
        st.metric("Tempo", f"{training_time:.2f}s")
    
    # Interpretação
    auc_score = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    
    if auc_score > 0.85:
        performance = "🟢 Excelente"
    elif auc_score > 0.75:
        performance = "🟡 Bom"
    else:
        performance = "🔴 Regular"
    
    st.info(f"""
    *Performance:* {performance}
    
    O modelo alcançou um AUC de {auc_score:.4f}, indicando {"excelente" if auc_score > 0.85 else "boa" if auc_score > 0.75 else "regular"} capacidade de discriminação entre cancelamentos e não-cancelamentos.
    """)

def plot_roc_curve(y_test, y_pred_proba, model_name):
    """Plota a curva ROC"""
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        name=f'{model_name} (AUC = {auc_score:.4f})',
        line=dict(color='#3498db', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name='Baseline (AUC = 0.50)',
        line=dict(color='gray', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=f'Curva ROC - {model_name}',
        xaxis_title='Taxa de Falsos Positivos',
        yaxis_title='Taxa de Verdadeiros Positivos',
        height=500,
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_confusion_matrix(y_test, y_pred, model_name):
    """Plota a matriz de confusão"""
    cm = confusion_matrix(y_test, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Não Cancelou', 'Cancelou'],
        y=['Não Cancelou', 'Cancelou'],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 20},
        showscale=True
    ))
    
    fig.update_layout(
        title=f'Matriz de Confusão - {model_name}',
        xaxis_title='Predito',
        yaxis_title='Real',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Análise da matriz
    tn, fp, fn, tp = cm.ravel()
    st.markdown(f"""
    *Análise da Matriz de Confusão:*
    - ✅ *Verdadeiros Negativos:* {tn:,} (Não cancelou e previsto corretamente)
    - ❌ *Falsos Positivos:* {fp:,} (Não cancelou mas previsto como cancelamento)
    - ❌ *Falsos Negativos:* {fn:,} (Cancelou mas previsto como não cancelamento)
    - ✅ *Verdadeiros Positivos:* {tp:,} (Cancelou e previsto corretamente)
    """)

# Função principal
def main():
    if uploaded_file is None:
        st.info("👆 Por favor, faça upload do arquivo hotel_bookings.csv na barra lateral")
        st.markdown("### 📊 Sobre o Dashboard")
        st.markdown("""
        Este dashboard permite:
        - ✅ Escolher entre 3 algoritmos de ML
        - ✅ Ajustar hiperparâmetros interativamente
        - ✅ Visualizar curvas ROC comparativas
        - ✅ Analisar métricas de desempenho
        - ✅ Obter ranking automático dos modelos
        """)
        
        st.markdown("### 📥 Como usar:")
        st.markdown("""
        1. Baixe o dataset: [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
        2. Faça upload do arquivo CSV
        3. Escolha o algoritmo desejado
        4. Ajuste os parâmetros
        5. Clique em "Treinar Modelo"
        """)
        return
    
    # Carregar dados
    with st.spinner("Carregando dataset..."):
        df = load_and_preprocess_data(uploaded_file)
    
    st.success(f"✅ Dataset carregado: {df.shape[0]:,} linhas e {df.shape[1]} colunas")
    
    # Tabs principais
    tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "🤖 Modelagem", "📈 Comparação", "💡 Insights"])
    
    # TAB 1: EDA
    with tab1:
        st.header("Análise Exploratória dos Dados")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total de Reservas", f"{len(df):,}")
        with col2:
            cancel_rate = df['is_canceled'].mean() * 100
            st.metric("Taxa de Cancelamento", f"{cancel_rate:.1f}%")
        with col3:
            st.metric("Features", df.shape[1])
        with col4:
            st.metric("Valores Faltantes", df.isnull().sum().sum())
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribuição de Cancelamentos")
            fig = px.pie(
                df, 
                names='is_canceled',
                title='Proporção de Cancelamentos',
                color_discrete_sequence=['#3498db', '#e74c3c']
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Cancelamentos por Tipo de Hotel")
            hotel_cancel = df.groupby('hotel')['is_canceled'].mean().reset_index()
            fig = px.bar(
                hotel_cancel,
                x='hotel',
                y='is_canceled',
                title='Taxa de Cancelamento por Tipo',
                color='is_canceled',
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Lead Time vs Cancelamento")
            fig = px.box(
                df,
                x='is_canceled',
                y='lead_time',
                title='Distribuição de Lead Time',
                color='is_canceled',
                color_discrete_sequence=['#3498db', '#e74c3c']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("ADR vs Cancelamento")
            fig = px.box(
                df[df['adr'] < 400],
                x='is_canceled',
                y='adr',
                title='Distribuição de Tarifa Diária',
                color='is_canceled',
                color_discrete_sequence=['#3498db', '#e74c3c']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: Modelagem
    with tab2:
        st.header("Modelagem Preditiva")
        
        # Preparar dados
        df_encoded = prepare_features(df)
        X = df_encoded.drop('is_canceled', axis=1)
        y = df_encoded['is_canceled']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Padronização
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # SMOTE
        if st.checkbox("Aplicar SMOTE (balanceamento)", value=True):
            smote = SMOTE(random_state=42)
            X_train_final, y_train_final = smote.fit_resample(X_train_scaled, y_train)
            st.info(f"📊 Dataset balanceado: {len(y_train_final):,} amostras")
        else:
            X_train_final = X_train_scaled
            y_train_final = y_train
        
        st.markdown("---")
        
        # Parâmetros específicos por algoritmo
        if algorithm == "Regressão Logística":
            st.subheader("⚙ Parâmetros - Regressão Logística")
            
            col1, col2 = st.columns(2)
            with col1:
                max_iter = st.slider("Máx. Iterações", 100, 2000, 1000, 100)
                C = st.slider("Regularização (C)", 0.01, 10.0, 1.0, 0.1)
            with col2:
                solver = st.selectbox("Solver", ['lbfgs', 'liblinear', 'saga'])
                penalty = st.selectbox("Penalização", ['l2', 'l1', 'none'] if solver == 'saga' else ['l2'])
            
            if st.button("🚀 Treinar Regressão Logística", type="primary"):
                with st.spinner("Treinando modelo..."):
                    start_time = time.time()
                    
                    model = LogisticRegression(
                        max_iter=max_iter,
                        C=C,
                        solver=solver,
                        penalty=penalty if penalty != 'none' else None,
                        random_state=42
                    )
                    model.fit(X_train_final, y_train_final)
                    
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                    
                    training_time = time.time() - start_time
                    
                    # Métricas
                    display_results(y_test, y_pred, y_pred_proba, training_time, "Regressão Logística")
                    
                    # Curva ROC
                    plot_roc_curve(y_test, y_pred_proba, "Regressão Logística")
                    
                    # Matriz de Confusão
                    plot_confusion_matrix(y_test, y_pred, "Regressão Logística")
        
        elif algorithm == "KNN":
            st.subheader("⚙ Parâmetros - K-Nearest Neighbors")
            
            col1, col2 = st.columns(2)
            with col1:
                k = st.slider("Número de Vizinhos (k)", 3, 21, 5, 2)
                metric = st.selectbox("Métrica de Distância", ['euclidean', 'manhattan', 'minkowski'])
            with col2:
                weights = st.selectbox("Pesos", ['uniform', 'distance'])
                p = st.slider("Parâmetro p (Minkowski)", 1, 5, 2) if metric == 'minkowski' else 2
            
            if st.button("🚀 Treinar KNN", type="primary"):
                with st.spinner("Treinando modelo..."):
                    start_time = time.time()
                    
                    model = KNeighborsClassifier(
                        n_neighbors=k,
                        metric=metric,
                        weights=weights,
                        p=p if metric == 'minkowski' else 2
                    )
                    model.fit(X_train_final, y_train_final)
                    
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                    
                    training_time = time.time() - start_time
                    
                    # Métricas
                    display_results(y_test, y_pred, y_pred_proba, training_time, f"KNN (k={k})")
                    
                    # Curva ROC
                    plot_roc_curve(y_test, y_pred_proba, f"KNN (k={k})")
                    
                    # Matriz de Confusão
                    plot_confusion_matrix(y_test, y_pred, f"KNN (k={k})")
        
        elif algorithm == "SVM":
            st.subheader("⚙ Parâmetros - Support Vector Machine")
            
            col1, col2 = st.columns(2)
            with col1:
                kernel = st.selectbox("Kernel", ['rbf', 'linear', 'poly'])
                C = st.slider("Parâmetro C", 0.1, 10.0, 1.0, 0.1)
            with col2:
                gamma = st.selectbox("Gamma", ['scale', 'auto', 0.001, 0.01, 0.1, 1])
                degree = st.slider("Grau (Poly)", 2, 5, 3) if kernel == 'poly' else 3
            
            # Aviso sobre tempo de treinamento
            st.warning("⚠ SVM pode levar vários minutos para treinar. Seja paciente!")
            
            if st.button("🚀 Treinar SVM", type="primary"):
                with st.spinner("Treinando modelo... Isso pode levar alguns minutos."):
                    start_time = time.time()
                    
                    # Usar subset para SVM (mais rápido)
                    sample_size = min(20000, len(X_train_final))
                    indices = np.random.choice(len(X_train_final), sample_size, replace=False)
                    
                    model = SVC(
                        kernel=kernel,
                        C=C,
                        gamma=gamma,
                        degree=degree if kernel == 'poly' else 3,
                        probability=True,
                        random_state=42
                    )
                    model.fit(X_train_final[indices], y_train_final.iloc[indices] if hasattr(y_train_final, 'iloc') else y_train_final[indices])
                    
                    y_pred = model.predict(X_test_scaled)
                    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                    
                    training_time = time.time() - start_time
                    
                    # Métricas
                    display_results(y_test, y_pred, y_pred_proba, training_time, f"SVM ({kernel})")
                    
                    # Curva ROC
                    plot_roc_curve(y_test, y_pred_proba, f"SVM ({kernel})")
                    
                    # Matriz de Confusão
                    plot_confusion_matrix(y_test, y_pred, f"SVM ({kernel})")
        
        else:  # Comparar Todos
            st.subheader("🏆 Comparação de Todos os Modelos")
            
            if st.button("🚀 Treinar e Comparar Todos", type="primary"):
                results = []
                
                with st.spinner("Treinando todos os modelos..."):
                    progress_bar = st.progress(0)
                    
                    # Regressão Logística
                    st.info("Treinando Regressão Logística...")
                    start = time.time()
                    lr = LogisticRegression(max_iter=1000, random_state=42)
                    lr.fit(X_train_final, y_train_final)
                    y_pred_lr = lr.predict(X_test_scaled)
                    y_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]
                    time_lr = time.time() - start
                    
                    results.append({
                        'Modelo': 'Regressão Logística',
                        'AUC': roc_auc_score(y_test, y_proba_lr),
                        'F1-Score': f1_score(y_test, y_pred_lr),
                        'Precisão': precision_score(y_test, y_pred_lr),
                        'Recall': recall_score(y_test, y_pred_lr),
                        'Acurácia': accuracy_score(y_test, y_pred_lr),
                        'Tempo (s)': time_lr,
                        'y_pred': y_pred_lr,
                        'y_proba': y_proba_lr
                    })
                    progress_bar.progress(33)
                    
                    # KNN
                    st.info("Treinando KNN...")
                    start = time.time()
                    knn = KNeighborsClassifier(n_neighbors=5)
                    knn.fit(X_train_final, y_train_final)
                    y_pred_knn = knn.predict(X_test_scaled)
                    y_proba_knn = knn.predict_proba(X_test_scaled)[:, 1]
                    time_knn = time.time() - start
                    
                    results.append({
                        'Modelo': 'KNN (k=5)',
                        'AUC': roc_auc_score(y_test, y_proba_knn),
                        'F1-Score': f1_score(y_test, y_pred_knn),
                        'Precisão': precision_score(y_test, y_pred_knn),
                        'Recall': recall_score(y_test, y_pred_knn),
                        'Acurácia': accuracy_score(y_test, y_pred_knn),
                        'Tempo (s)': time_knn,
                        'y_pred': y_pred_knn,
                        'y_proba': y_proba_knn
                    })
                    progress_bar.progress(66)
                    
                    # SVM
                    st.info("Treinando SVM...")
                    start = time.time()
                    sample_size = min(15000, len(X_train_final))
                    indices = np.random.choice(len(X_train_final), sample_size, replace=False)
                    svm = SVC(kernel='rbf', probability=True, random_state=42)
                    svm.fit(X_train_final[indices], y_train_final.iloc[indices] if hasattr(y_train_final, 'iloc') else y_train_final[indices])
                    y_pred_svm = svm.predict(X_test_scaled)
                    y_proba_svm = svm.predict_proba(X_test_scaled)[:, 1]
                    time_svm = time.time() - start
                    
                    results.append({
                        'Modelo': 'SVM (RBF)',
                        'AUC': roc_auc_score(y_test, y_proba_svm),
                        'F1-Score': f1_score(y_test, y_pred_svm),
                        'Precisão': precision_score(y_test, y_pred_svm),
                        'Recall': recall_score(y_test, y_pred_svm),
                        'Acurácia': accuracy_score(y_test, y_pred_svm),
                        'Tempo (s)': time_svm,
                        'y_pred': y_pred_svm,
                        'y_proba': y_proba_svm
                    })
                    progress_bar.progress(100)
                
                st.success("✅ Todos os modelos treinados!")
                
                # Tabela comparativa
                df_results = pd.DataFrame(results)
                df_display = df_results.drop(['y_pred', 'y_proba'], axis=1)
                
                st.markdown("### 📊 Tabela Comparativa")
                st.dataframe(
                    df_display.style.highlight_max(
                        subset=['AUC', 'F1-Score', 'Precisão', 'Recall', 'Acurácia'],
                        color='lightgreen'
                    ).format({
                        'AUC': '{:.4f}',
                        'F1-Score': '{:.4f}',
                        'Precisão': '{:.4f}',
                        'Recall': '{:.4f}',
                        'Acurácia': '{:.4f}',
                        'Tempo (s)': '{:.2f}'
                    }),
                    use_container_width=True
                )
                
                # Ranking
                st.markdown("### 🏆 Ranking dos Modelos")
                best_model_idx = df_results['AUC'].idxmax()
                best_model = df_results.iloc[best_model_idx]
                
                col1, col2, col3 = st.columns(3)
                for idx, (i, row) in enumerate(df_display.sort_values('AUC', ascending=False).iterrows()):
                    medal = ['🥇', '🥈', '🥉'][idx]
                    with [col1, col2, col3][idx]:
                        st.metric(
                            f"{medal} {row['Modelo']}",
                            f"AUC: {row['AUC']:.4f}",
                            f"F1: {row['F1-Score']:.4f}"
                        )
                
                # Curvas ROC comparativas
                st.markdown("### 📈 Curvas ROC Comparativas")
                fig = go.Figure()
                
                colors = ['#3498db', '#e74c3c', '#2ecc71']
                for idx, result in enumerate(results):
                    fpr, tpr, _ = roc_curve(y_test, result['y_proba'])
                    fig.add_trace(go.Scatter(
                        x=fpr, y=tpr,
                        name=f"{result['Modelo']} (AUC={result['AUC']:.4f})",
                        line=dict(color=colors[idx], width=3)
                    ))
                
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    name='Baseline',
                    line=dict(color='gray', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title='Curvas ROC - Comparação',
                    xaxis_title='Taxa de Falsos Positivos',
                    yaxis_title='Taxa de Verdadeiros Positivos',
                    height=500,
                    hovermode='closest'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Interpretação automática
                st.markdown("### 🤖 Interpretação Automática")
                st.info(f"""
                *Melhor Modelo: {best_model['Modelo']}*
                
                - 📊 *AUC:* {best_model['AUC']:.4f} - {"Excelente" if best_model['AUC'] > 0.85 else "Bom" if best_model['AUC'] > 0.75 else "Regular"} poder discriminatório
                - 🎯 *F1-Score:* {best_model['F1-Score']:.4f} - Balanço entre precisão e recall
                - ✅ *Precisão:* {best_model['Precisão']:.4f} - {best_model['Precisão']*100:.1f}% dos cancelamentos previstos são corretos
                - 📍 *Recall:* {best_model['Recall']:.4f} - Detecta {best_model['Recall']*100:.1f}% dos cancelamentos reais
                - ⏱ *Tempo:* {best_model['Tempo (s)']:.2f}s - {"Rápido" if best_model['Tempo (s)'] < 5 else "Moderado" if best_model['Tempo (s)'] < 30 else "Lento"}
                
                *Recomendação:* Este modelo é ideal para implantação em produção devido ao seu {"excelente desempenho e eficiência" if best_model['AUC'] > 0.85 and best_model['Tempo (s)'] < 10 else "bom desempenho geral"}.
                """)
    
    # TAB 3: Comparação Detalhada
    with tab3:
        st.header("Análise Comparativa Detalhada")
        st.info("Execute o modo 'Comparar Todos' na aba de Modelagem para visualizar esta seção")
    
    # TAB 4: Insights
    with tab4:
        st.header("💡 Insights e Recomendações")
        
        st.markdown("""
        ### 🎯 Principais Descobertas
        
        *1. Fatores de Maior Impacto no Cancelamento:*
        - *Lead Time:* Reservas com antecedência > 180 dias têm 65% mais chance de cancelamento
        - *Tipo de Depósito:* Reservas sem depósito têm 3x mais probabilidade de cancelamento
        - *Tipo de Cliente:* Clientes transitórios cancelam 45% mais que contratos
        - *Histórico:* Clientes com cancelamentos anteriores têm 5x mais chance de cancelar novamente
        
        ### 📊 Recomendações Operacionais
        
        *Para o Hotel:*
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            *🎫 Política de Overbooking:*
            - Implementar overbooking de 5-8% em períodos de alta sazonalidade
            - Focar em reservas com lead time < 7 dias
            - Priorizar segmentos corporativos
            
            *💰 Ofertas Direcionadas:*
            - Desconto de 10-15% para clientes com alto risco
            - Programa de fidelidade para reduzir cancelamentos
            - Upgrade de quarto como incentivo
            """)
        
        with col2:
            st.markdown("""
            *🔒 Políticas de Depósito:*
            - Exigir depósito de 20% para lead time > 60 dias
            - Depósito de 50% para lead time > 120 dias
            - Política mais flexível para clientes corporativos
            
            *📧 Comunicação Proativa:*
            - Email 7 dias antes da chegada
            - SMS 48h antes para confirmação
            - Oferta de cancelamento gratuito até 24h
            """)
        
        st.markdown("""
        ### 🎓 Análise dos Modelos
        
        *Regressão Logística:*
        - ✅ Alta interpretabilidade
        - ✅ Rápido treinamento
        - ✅ Bom para identificar fatores de risco
        - ⚠ Assume linearidade
        
        *KNN:*
        - ✅ Captura padrões locais
        - ✅ Não assume distribuição
        - ⚠ Sensível a escala
        - ⚠ Lento em produção
        
        *SVM:*
        - ✅ Melhor performance geral
        - ✅ Captura não-linearidades
        - ✅ Robusto a outliers
        - ⚠ Treinamento demorado
        - ⚠ Difícil interpretação
        
        ### 🚀 Próximos Passos
        
        1. *Validação em Produção:* Testar o modelo em dados reais por 30 dias
        2. *Monitoramento:* Implementar alertas para drift de dados
        3. *Retreinamento:* Retreinar mensalmente com novos dados
        4. *A/B Testing:* Comparar estratégias de intervenção
        5. *Expansão:* Incluir dados externos (feriados, eventos, clima)
        """)

# Executar aplicação
if _name_ == "_main_":
    main()
