# Real-Time Audio FFT Spectrum Analyzer

**ESP32-C3 + Basys 3 FPGA Hardware-Accelerated FFT with VGA Display**

---

## Overview

This project is a real-time audio frequency spectrum analyzer built from discrete hardware components. Sound is captured by a **CZN-15E electret microphone**, digitized by an **ESP32-C3 microcontroller**, streamed over SPI to a **Basys 3 FPGA board** (Xilinx Artix-7), where a **1024-point FFT** is computed in hardware and the frequency spectrum is displayed live on a **VGA monitor**.

---

## System Block Diagram

```
CZN-15E Mic
    │  (analog audio)
    ▼
ESP32-C3 ADC
  GPIO 3, 45 kHz, 12-bit
    │  (1024-sample frames)
    ▼
SPI Bus @ 20 MHz
  CLK→GPIO4  MOSI→GPIO6  CS→GPIO7
    │
    ▼  Pmod JA (J1/L2/G2)
Basys 3 FPGA (Artix-7 XC7A35T)
  ┌─────────────────────────────┐
  │  spi_slave_receiver         │
  │  xfft_0 IP Core (1024-pt)  │
  │  vga_controller             │
  │  spectrum_renderer          │
  └─────────────┬───────────────┘
                │  VGA 12-bit
                ▼
           VGA Monitor
          (640×480 @ 60 Hz)
```

---

## Hardware Components

| Component | Part / Model | Role |
|-----------|-------------|------|
| Microcontroller | ESP32-C3 Mini (RISC-V, 160 MHz) | ADC sampling, SPI master |
| Microphone | CZN-15E Electret Module | Analog audio capture |
| FPGA Board | Basys 3 – Xilinx Artix-7 XC7A35T | FFT computation, VGA output |
| Display | VGA Monitor (640×480) | Real-time spectrum visualization |

---

## Pin Connections

### ESP32-C3 → Basys 3 Pmod JA

| ESP32-C3 Pin | Signal | Basys 3 Pmod JA Pin |
|---|---|---|
| GPIO 4 | SPI SCK | G2 |
| GPIO 6 | SPI MOSI | L2 |
| GPIO 7 | SPI CS | J1 |
| GND | Ground | GND |

### Basys 3 VGA Port (standard 12-bit, onboard connector)

---

## File Structure

```
FFT Audio Spectrum Analyzer/
├── README.md                      ← This file
├── FFT.ino                        ← ESP32-C3 Arduino firmware
├── fpga_verilog/
│   ├── top_fft_system.v           ← Top-level Verilog (all modules)
│   └── constraints.xdc            ← Basys 3 pin constraint file
└── photos/
    ├── Photo_ofBasys3.jpeg        ← Physical hardware setup
    ├── IdleNoise.jpeg             ← Idle noise floor on VGA
    ├── NoisewithBlow.jpeg         ← Broadband impulse (blowing)
    ├── 500Hz_SW3_on.jpeg          ← 500 Hz tone, Top-3 mode ON
    ├── 500KHz_SW3_of.jpeg         ← 500 Hz tone, full spectrum
    ├── 2KHz_SW_of.jpeg            ← 2 kHz tone, full spectrum
    └── 2KHz_SW3_of.jpeg           ← 2 kHz tone, alternate view
```

---

## ESP32-C3 Firmware

**File:** [`FFT.ino`](FFT.ino)

The firmware performs two operations in a continuous loop:

1. **Sample collection** — reads 1024 ADC samples at a fixed 22 µs interval (~45 kHz) using a `micros()` busy-wait loop for accurate timing.
2. **SPI transmission** — asserts CS LOW, transfers all 2048 bytes to the FPGA at 20 MHz, then raises CS HIGH. A 2 ms delay follows each frame.

Key settings:
- ADC attenuation: `ADC_11db` → full 0–3.1 V input range
- SPI: 20 MHz, MSB-first, Mode 0
- No MISO (transmit-only)

---

## FPGA Verilog Design

**File:** [`fpga_verilog/top_fft_system.v`](fpga_verilog/top_fft_system.v)

### Modules

#### `spi_slave_receiver`
Captures the SPI bit stream from the ESP32. Uses:
- 3-stage SCK synchronizer for rising-edge detection
- 2-stage synchronizers on CS and MOSI for CDC
- 8-bit shift register assembling bytes; two bytes combined into a 16-bit sample

#### `xfft_0` (Xilinx FFT IP Core)
- 1024-point, AXI-Stream I/O
- Fixed-point arithmetic, 16-bit I/O width
- Natural-order input and output
- Configured for Artix-7 (xc7a35tcpg236-1)

#### `vga_controller`
- 640×480 @ 60 Hz standard timing
- 25 MHz pixel clock derived from 100 MHz by 4-cycle divider
- Outputs `pixel_x`, `pixel_y`, and `video_on` blanking flag

#### `spectrum_renderer`
- Computes FFT magnitude: `|Re| + |Im|/2` (alpha-max approximation)
- Stores 512 bins in dual-port BRAM
- **Logarithmic index mapping** — 8 piecewise-linear segments covering 100 Hz to 15 kHz
- **Top-3 Peak mode** (sw[3]=HIGH) — proximity-lockout sorter identifies 3 dominant peaks; Peak 1 rendered in red, Peaks 2 & 3 in yellow
- **DC/hum filter** — ignores bins 0–6 to suppress ambient room noise
- Frequency axis labels (100, 500, 1k, 2k, 5k, 10k, 15k Hz) in cyan bitmap font

### Slide Switch Controls

| Switch | Function |
|--------|----------|
| sw[0] | HIGH = use internal test sine generator; LOW = use SPI mic data |
| sw[1] | HIGH = double test sine frequency (~4 kHz); LOW = base (~2 kHz) |
| sw[3] | HIGH = Top-3 Peak isolation mode |

### LED Diagnostics

| LED | Signal |
|-----|--------|
| LED[0] | spi_cs (Chip Select state) |
| LED[1] | spi_sck (SPI clock activity) |
| LED[2] | spi_mosi (data line) |
| LED[3] | valid_toggle (audio sample toggle) |
| LED[4] | fft_output_valid |
| LED[5] | fft_output_last |
| LED[15:8] | Audio sample MSBs (signal level meter) |

---

## VGA Display Layout

```
Column:  64                          576
         |<---- 512 pixels wide ----->|
Row 0    ┌────────────────────────────┐
         │                            │  ← Black background
         │       Spectrum bars        │  ← Yellow bars (or red/yellow in peak mode)
         │       rendered upward      │
Row 440  └────────────────────────────┘  ← Baseline
Row 441  ────────────────────────────    ← Frequency axis (gray line)
Row 442-444     tick marks at labeled frequencies
Row 450-454  100  500  1k   2k   5k  10k  15k  ← Cyan frequency labels
```

---

## Results

| Test | Observation |
|------|-------------|
| Idle noise | Low bars, concentrated in low-frequency bins |
| Blowing into mic | Broadband burst across low–mid frequency range |
| 500 Hz tone | Sharp peak at 500 Hz column; harmonics visible |
| 2 kHz tone | Sharp peak at 2 kHz column |
| Top-3 mode | Dominant peak in red; two secondary peaks in yellow; background suppressed |
| Test mode (sw[0]) | Clean isolated peak at ~2 kHz or ~4 kHz (sw[1]) |

---

## Build & Flash

### ESP32-C3 Firmware
1. Open `FFT.ino` in **Arduino IDE**
2. Select board: **ESP32C3 Dev Module**
3. Set Upload Speed: 921600
4. Flash to ESP32-C3 Mini

### FPGA Bitstream
1. Open `vivado.xpr` in **Xilinx Vivado 2023.2**
2. Run Synthesis → Implementation → Generate Bitstream
3. Program the Basys 3 via USB-JTAG

---

## Author

**Ömer Faruk Akbay** — June 2026
