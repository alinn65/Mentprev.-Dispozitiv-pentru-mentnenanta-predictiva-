# 🔧 MentPrev — Predictive Maintenance via Vibration & Acoustic Spectrum Analysis

> A compact, autonomous Edge Computing device for real-time industrial equipment health monitoring.

![Platform](https://img.shields.io/badge/Platform-ESP32--S3-blue)
![Language](https://img.shields.io/badge/Language-C%2B%2B%20%7C%20Arduino%20IDE-green)


---

## 📖 Overview

**MentPrev** is an embedded predictive maintenance system designed to detect mechanical anomalies in rotating industrial equipment (motors, pumps, fans) before they cause critical failures. The device fuses vibration and acoustic data, performs on-device FFT-based spectral analysis, and outputs an intuitive **Health Score (0–100%)** — all without relying on external cloud infrastructure.

---

## ✨ Features

- 🎙️ **Dual-sensor fusion** — triaxial MEMS accelerometer (I2C) + digital microphone (I2S)
- 📊 **Real-time FFT spectral analysis** on both vibration and acoustic channels
- 🧠 **Health Score algorithm** with anomaly accumulation and exponential smoothing
- 🤖 **Edge AI / TinyML** — neural network classifier (Normal / Imbalance / Air Blockage) via Edge Impulse
- 📟 **Local HMI** — round TFT display (GC9A01), two physical buttons, status LEDs
- 🌐 **Wi-Fi Web Interface** — live monitoring dashboard, remote control, CSV file download
- 💾 **SD card logging** — adaptive rate CSV logging with RTC timestamp
- 🔋 **Hardware latch power circuit** — safe shutdown with SD file flush
- 🛡️ **4-layer PCB** with isolated SPI buses (FSPI for display, HSPI for SD), GND and power planes

---

## 🏗️ Hardware Architecture

| Component | Part | Interface |
|-----------|------|-----------|
| Microcontroller | ESP32-S3 WROOM-1U | — |
| Accelerometer | ADXL345 / MMA8452Q | I2C |
| Microphone | INMP441 (MEMS) | I2S |
| Display | GC9A01 (round TFT) | FSPI |
| Storage | MicroSD card | HSPI |
| RTC | PCF8563T | I2C |
| Voltage regulator | XC6227C331PR-G (3.3V) | — |
| ESD protection | USBLC6-2SC6 | USB-C |

PCB: **5×5 cm, 4-layer**, designed in **KiCad**. Enclosure: custom **3D-printed** case (Fusion 360).

---

## 💻 Software Architecture

```
├── DSP Module          # I2S mic + I2C accelerometer sampling, FFT
├── Diagnosis Module    # Calibration, Health Score, anomaly accumulation
├── Storage Module      # CSV logging, RTC timestamps, adaptive write rate
├── HMI Module          # TFT graphics (double-buffered), button state machine
└── Web Server Module   # Wi-Fi AP, HTTP dashboard, file management
```

Runs on **FreeRTOS** with dual-core task distribution:

| Task | Core | Role |
|------|------|------|
| Main loop | Core 1 | HMI, web server, button handling |
| Audio processing | Core 0 | I2S read + acoustic FFT |
| Vibration sampling | Core 0 | I2C polling + vibration FFT |

Shared data protected by **Mutex** locks to prevent corruption.

---

## 📐 Health Score Algorithm

1. **Calibration phase 1** — ambient noise baseline (equipment off)
2. **Calibration phase 2** — reference spectral signature (equipment running normally)
3. **Continuous monitoring** — fused residual computed as:

$$\mathcal{D}^{(n)} = D^{(n)} + w \cdot \Delta v^{(n)}$$

$$H_{inst}^{(n)} = \text{clamp}\left(100 - \frac{\mathcal{D}^{(n)}}{k},\ 0,\ 100\right)$$

Anomaly accumulates asymmetrically (+5.0 when below threshold, −1.5 otherwise), and a penalty is applied only when persistent. Final score is exponentially smoothed for stable display.

---

## 🤖 Edge AI (TinyML)

Trained on **Edge Impulse** using 2-second windows at 400 Hz on 3-axis accelerometer data.

| Class | F1 Score |
|-------|----------|
| Normal | 0.85 |
| Imbalance | 0.96 |
| Air Blockage | 0.81 |

**Overall accuracy: 87.6%**

---

