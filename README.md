# Smart Grid & Electricity Load Forecasting (Temporal Transformer CPU)

โปรเจกต์นำเสนอและ Workshop พยากรณ์อุณหภูมิหม้อแปลงไฟฟ้าและโหลดพลังงานล่วงหน้า 12 ช่วงเวลา ด้วย **ETDataset (Electricity Transformer Dataset)** จาก UCI Machine Learning Repository 

## 📁 โครงสร้างโปรเจกต์
- `app.py`: สคริปต์หลัก Streamlit Dashboard พร้อม Option เลือกชุดข้อมูล ETTh1, ETTm1 และ Synthetic Data
- `models.py`: โครงสร้าง MultiStepTemporalTransformer (CPU Optimized)
- `utils.py`: ระบบดึงข้อมูลจริงอัตโนมัติผ่าน URL ของ ETDataset และเตรียม Window Sequences
- `requirements.txt`: รายการ Dependencies ที่ต้องใช้งาน

## 🚀 วิธีการสั่งรันโปรเจกต์

1. ติดตั้ง Dependencies:
```bash
pip install -r requirements.txt
```

2. รันแอปพลิเคชัน Streamlit:
```bash
streamlit run app.py
```
