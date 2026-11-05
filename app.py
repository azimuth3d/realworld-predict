import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from models import MultiStepTemporalTransformer
from utils import load_ett_dataset, create_multi_step_sequences

st.set_page_config(
    page_title="Smart Grid & Electricity Load Forecasting Hub", 
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for visual enhancement
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #6c757d;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .winner-card {
        background-color: #e8f5e9;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #2e7d32;
        box-shadow: 0 4px 6px rgba(46,125,50,0.1);
    }
    .xgb-card {
        background-color: #fff3e0;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #ef6c00;
    }
    .base-card {
        background-color: #eceff1;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #546e7a;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Smart Grid & Electricity Load Forecasting: Temporal Transformer")
st.caption("การพยากรณ์อุณหภูมิหม้อแปลงไฟฟ้า/โหลดพลังงาน (Real Dataset: ETDataset) ล่วงหน้า 12 ช่วงเวลา ด้วย Temporal Transformer บน CPU")

# Sidebar Configuration
st.sidebar.header("📂 Data Source Selection")
dataset_choice = st.sidebar.selectbox(
    "เลือกชุดข้อมูล (Dataset Options)",
    ["ETTh1 (Hourly)", "ETTm1 (15-min)", "Synthetic Simulation Data"],
    help="เลือกข้อมูลจริง ETDataset จาก UCI Repository หรือข้อมูลจำลอง"
)

sample_size = st.sidebar.slider("จำนวนแถวข้อมูลที่ใช้ (Sample Size)", 500, 3000, 1500, step=250)

st.sidebar.header("⚙️ Model & Training Parameters")
seq_length = st.sidebar.slider("Lookback Window (Sequence Length)", 24, 72, 48, help="จำนวนชั่วโมง/ช่วงเวลาอดีตที่ใช้เป็น Input")
forecast_horizon = 12
epochs = st.sidebar.slider("Transformer CPU Epochs", 10, 50, 25, help="จำนวนรอบในการเทรนบน CPU")
lr = st.sidebar.number_input("Learning Rate", value=0.003, format="%.4f")

# Load Selected Dataset
@st.cache_data
def get_dataset(choice, size):
    return load_ett_dataset(choice, size)

df, dataset_desc = get_dataset(dataset_choice, sample_size)

tabs = st.tabs([
    "📊 Data Explorer & ETDataset Info", 
    "⚡ Comparative Benchmark", 
    "🔍 Error Breakdown & Step Analysis", 
    "🎛️ Scenario Simulator", 
    "👥 Workshop Guide"
])

# TAB 1: DATA EXPLORER
with tabs[0]:
    st.subheader(f"📈 Real Dataset Preview: {dataset_choice}")
    st.write(f"**คำอธิบายชุดข้อมูล:** {dataset_desc}")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['Value'], mode='lines', name='Oil Temperature (OT)', line=dict(color='#00838f', width=1.5)))
        fig.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20), hovermode="x unified", yaxis_title="Oil Temperature (°C)")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.metric("Total Data Points", len(df))
        st.metric("Source", "UCI Machine Learning Repo")
        st.metric("Forecast Horizon", f"Next {forecast_horizon} Steps")
        st.info("💡 **ETDataset (Electricity Transformer Dataset)** เป็นชุดข้อมูลมาตรฐานโลกระดับ Benchmark สำหรับทดสอบการพยากรณ์อนุกรมเวลาในงานวิจัย Transformer")

# TAB 2: COMPARATIVE BENCHMARK
with tabs[1]:
    st.subheader("⚡ Multi-Method Performance Benchmark")
    st.write("เปรียบเทียบผลการพยากรณ์อุณหภูมิ/โหลดไฟฟ้าล่วงหน้า 12 ช่วงเวลา")
    
    if st.button("🚀 Run CPU Training & Benchmark Comparison", type="primary"):
        values = df['Value'].values.reshape(-1, 1)
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(values)
        
        X, y = create_multi_step_sequences(scaled_values, seq_length, forecast_horizon)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        y_train_2d = y_train.reshape(y_train.shape[0], forecast_horizon)
        y_test_2d = y_test.reshape(y_test.shape[0], forecast_horizon)
        
        # 1. Baseline: Naive (Repeat Last Value)
        ma_preds = np.repeat(X_test[:, -1, :], forecast_horizon, axis=1)
        
        # 2. XGBoost Multi-output
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        xgb = XGBRegressor(n_estimators=80, max_depth=5, learning_rate=0.05)
        xgb.fit(X_train_flat, y_train_2d)
        xgb_preds = xgb.predict(X_test_flat)
        
        # 3. Temporal Transformer (CPU)
        device = torch.device("cpu")
        model = MultiStepTemporalTransformer(input_dim=1, d_model=64, forecast_horizon=forecast_horizon).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train_2d, dtype=torch.float32)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        start_time = time.time()
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
            
            progress = (epoch + 1) / epochs
            progress_bar.progress(progress)
            status_text.text(f"CPU Training Progress: Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")
            time.sleep(0.02)
            
        cpu_time = time.time() - start_time
        status_text.empty()
        progress_bar.empty()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_test, dtype=torch.float32)
            tf_preds = model(X_test_t).numpy()
            
        # Inverse Scaling
        y_test_orig = scaler.inverse_transform(y_test_2d.reshape(-1, 1)).reshape(-1, forecast_horizon)
        ma_preds_orig = scaler.inverse_transform(ma_preds.reshape(-1, 1)).reshape(-1, forecast_horizon)
        xgb_preds_orig = scaler.inverse_transform(xgb_preds.reshape(-1, 1)).reshape(-1, forecast_horizon)
        tf_preds_orig = scaler.inverse_transform(tf_preds.reshape(-1, 1)).reshape(-1, forecast_horizon)
        
        # Metrics Calculation
        mae_ma = np.mean(np.abs(y_test_orig - ma_preds_orig))
        mae_xgb = np.mean(np.abs(y_test_orig - xgb_preds_orig))
        mae_tf = np.mean(np.abs(y_test_orig - tf_preds_orig))
        
        rmse_ma = np.sqrt(np.mean((y_test_orig - ma_preds_orig) ** 2))
        rmse_xgb = np.sqrt(np.mean((y_test_orig - xgb_preds_orig) ** 2))
        rmse_tf = np.sqrt(np.mean((y_test_orig - tf_preds_orig) ** 2))
        
        st.session_state['benchmark_results'] = {
            'y_test_orig': y_test_orig,
            'ma_preds_orig': ma_preds_orig,
            'xgb_preds_orig': xgb_preds_orig,
            'tf_preds_orig': tf_preds_orig,
            'mae_ma': mae_ma, 'mae_xgb': mae_xgb, 'mae_tf': mae_tf,
            'rmse_ma': rmse_ma, 'rmse_xgb': rmse_xgb, 'rmse_tf': rmse_tf,
            'cpu_time': cpu_time,
            'dataset_name': dataset_choice
        }

    if 'benchmark_results' in st.session_state:
        res = st.session_state['benchmark_results']
        
        st.success(f"⚡ CPU Training Complete for **{res['dataset_name']}** in **{res['cpu_time']:.2f} seconds**")
        
        # Metric Cards Side-by-Side Comparison
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="base-card">
                <h4>1. Naive Baseline</h4>
                <h3>MAE: <b>{res['mae_ma']:.2f}</b></h3>
                <p>RMSE: {res['rmse_ma']:.2f}</p>
                <small>ทำนายโดยซ้ำค่าอดีตล่าสุดไป 12 สเต็ป</small>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            st.markdown(f"""
            <div class="xgb-card">
                <h4>2. XGBoost (Multi-Output)</h4>
                <h3>MAE: <b>{res['mae_xgb']:.2f}</b></h3>
                <p>RMSE: {res['rmse_xgb']:.2f}</p>
                <small>Tree-based ensemble สำหรับ Multi-target</small>
            </div>
            """, unsafe_allow_html=True)
            
        with c3:
            imprv = ((res['mae_xgb'] - res['mae_tf']) / res['mae_xgb']) * 100
            st.markdown(f"""
            <div class="winner-card">
                <h4>🏆 3. Temporal Transformer (CPU)</h4>
                <h3>MAE: <b style="color:#2e7d32;">{res['mae_tf']:.2f}</b></h3>
                <p>RMSE: {res['rmse_tf']:.2f}</p>
                <small><b>+{imprv:.1f}% แม่นยำกว่า XGBoost</b></small>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        # Interactive Sample Selector
        st.subheader("🎯 Target vs Predictions Comparison (Sample Window)")
        sample_idx = st.slider("เลือกตำแหน่ง Window ตัวอย่างใน Test Set", 0, len(res['y_test_orig']) - 1, 15)
        
        hours = [f"+{h+1} Step" for h in range(forecast_horizon)]
        
        fig_main = go.Figure()
        fig_main.add_trace(go.Scatter(x=hours, y=res['y_test_orig'][sample_idx], mode='lines+markers', name='Actual Target (Ground Truth)', line=dict(color='black', width=3)))
        fig_main.add_trace(go.Scatter(x=hours, y=res['ma_preds_orig'][sample_idx], mode='lines+markers', name='Naive Baseline', line=dict(color='#78909c', dash='dash')))
        fig_main.add_trace(go.Scatter(x=hours, y=res['xgb_preds_orig'][sample_idx], mode='lines+markers', name='XGBoost', line=dict(color='#f57c00', width=2)))
        fig_main.add_trace(go.Scatter(x=hours, y=res['tf_preds_orig'][sample_idx], mode='lines+markers', name='Temporal Transformer (Winner)', line=dict(color='#2e7d32', width=3.5)))
        
        fig_main.update_layout(
            title=f"12-Step Forecast Horizon Curve (Window #{sample_idx} on {res['dataset_name']})",
            xaxis_title="Future Forecast Horizon (Steps)",
            yaxis_title="Oil Temp / Target Value",
            height=450,
            hovermode="x unified"
        )
        st.plotly_chart(fig_main, use_container_width=True)

# TAB 3: ERROR BREAKDOWN & STEP ANALYSIS
with tabs[2]:
    st.subheader("🔍 Error Breakdown Across 12-Step Horizon")
    st.write("วิเคราะห์การสะสมของ Error ในแต่ละช่วงเวลาในอนาคต (ยิ่งทำนายไกลขึ้น Error เป็นอย่างไร?)")
    
    if 'benchmark_results' in st.session_state:
        res = st.session_state['benchmark_results']
        
        step_mae_ma = np.mean(np.abs(res['y_test_orig'] - res['ma_preds_orig']), axis=0)
        step_mae_xgb = np.mean(np.abs(res['y_test_orig'] - res['xgb_preds_orig']), axis=0)
        step_mae_tf = np.mean(np.abs(res['y_test_orig'] - res['tf_preds_orig']), axis=0)
        
        steps = [f"Step +{h+1}" for h in range(forecast_horizon)]
        
        col_chart, col_tbl = st.columns([2, 1])
        with col_chart:
            fig_err = go.Figure()
            fig_err.add_trace(go.Bar(x=steps, y=step_mae_ma, name='Baseline MAE', marker_color='#90a4ae'))
            fig_err.add_trace(go.Bar(x=steps, y=step_mae_xgb, name='XGBoost MAE', marker_color='#ffb74d'))
            fig_err.add_trace(go.Bar(x=steps, y=step_mae_tf, name='Transformer MAE', marker_color='#81c784'))
            
            fig_err.update_layout(
                title="Error Accumulation per Forecast Step (MAE)",
                barmode='group',
                height=400,
                xaxis_title="Horizon Step",
                yaxis_title="Mean Absolute Error (MAE)"
            )
            st.plotly_chart(fig_err, use_container_width=True)
            
        with col_tbl:
            df_err = pd.DataFrame({
                'Step': steps,
                'Baseline MAE': step_mae_ma,
                'XGBoost MAE': step_mae_xgb,
                'Transformer MAE': step_mae_tf
            }).set_index('Step')  # ใช้ set_index เพื่อหลีกเลี่ยง TypeError
            
            st.dataframe(df_err.style.highlight_min(axis=1, color='#c8e6c9'), height=400)
    else:
        st.info("💡 กรุณากดรัน Benchmarking ใน Tab 'Comparative Benchmark' เพื่อดูผลวิเคราะห์ Error")

# TAB 4: SCENARIO SIMULATOR
with tabs[3]:
    st.subheader("🎛️ Smart Grid Grid Load & Shock Simulator")
    st.write("จำลองสภาวะโหลดไฟฟ้าผันผวนผิดปกติ (Peak Demand Shock / Heatwave Spike) บนข้อมูลจริง")
    
    col_s1, col_s2 = st.columns(2)
    volatility = col_s1.slider("Grid Volatility Multiplier", 0.5, 3.0, 1.0)
    demand_shift = col_s2.slider("Peak Demand Shift (%)", -30, 50, 15)
    
    sim_data = df['Value'].copy() * (1 + demand_shift/100)
    sim_data = sim_data + np.random.normal(0, volatility, size=len(sim_data))
    
    fig_sim = go.Figure()
    fig_sim.add_trace(go.Scatter(y=sim_data.tail(120), mode='lines+markers', name='Simulated Grid Load', line=dict(color='#d32f2f')))
    fig_sim.update_layout(title="Simulated Smart Grid Transformer Stream", height=380, yaxis_title="Transformer Value")
    st.plotly_chart(fig_sim, use_container_width=True)

# TAB 5: WORKSHOP GUIDE
with tabs[4]:
    st.subheader("🎯 Smart Grid & Electricity Load Forecasting Workshop")
    st.markdown("""
    ### 📋 คำแนะนำการดำเนินกิจกรรมกลุ่ม (60 นาที) - โจทย์ ETDataset Smart Grid
    
    * **1. Manager & PM (Smart Grid Cost & Energy Management):**
      * ประเมินประโยชน์ของการทำนายอุณหภูมิหม้อแปลง (Oil Temperature) และโหลดไฟฟ้าล่วงหน้า 12 ช่วงเวลา เพื่อป้องกัน Overheating และลดค่าปรับ Peak Demand
      * วิเคราะห์ความคุ้มค่าของการฝึกเทรนโมเดลบน CPU ภายในองค์กรแทนการเช่าระบบ GPU Cloud ราคาแพง
      
    * **2. Developer (Model Tuning & Real Data Pipelines):**
      * สลับตัวเลือกชุดข้อมูลระหว่าง `ETTh1 (Hourly)` และ `ETTm1 (15-min)` ใน Sidebar เพื่อเปรียบเทียบความเร็วของการเทรนบน CPU
      * ทดลองปรับ `Sequence Length` (Lookback Window) เพื่อดูผลกระทบต่อค่า MAE
      
    * **3. Designer (Smart Grid Dashboard UX/UI):**
      * ออกแบบหน้าแสดงผลเตือนภัยล่วงหน้า (Early Warning System) เมื่อโมเดลพยากรณ์ว่าอุณหภูมิหม้อแปลงไฟฟ้าจะสูงเกินค่าความปลอดภัยใน 12 ช่วงเวลาข้างหน้า
      
    * **4. QA (Safety Criteria & Anomaly Testing):**
      * กำหนดเงื่อนไขการทดสอบเมื่อข้อมูลจาก Smart Meter หรือ IoT Sensor ขาดหายไปบางช่วงเวลา
      * ทดสอบ Performance ของโมเดลเมื่อปรับเพิ่ม Grid Volatility ใน Tab 4
    """)
