#include <SPI.h>

#define MIC_PIN 3
#define NUM_SAMPLES 1024
#define SAMPLE_PERIOD_US 22 // ~45 kHz sampling interval

uint16_t audio_buffer[NUM_SAMPLES];

// Custom SPI Pin Assignments matching your hardware layout
#define VSPI_CLK  4
#define VSPI_MISO -1 // Unused for transmitter
#define VSPI_MOSI 6
#define VSPI_SS   7

SPIClass *vspi = NULL;

void setup() {
    Serial.begin(115200);
    
    // Initialize GPIO 3 for full-scale analog capture (0V - 3.1V)
    analogSetAttenuation(ADC_11db);
    pinMode(MIC_PIN, INPUT);

    // Initialize custom hardware SPI bus peripheral
    vspi = new SPIClass(FSPI); 
    vspi->begin(VSPI_CLK, VSPI_MISO, VSPI_MOSI, VSPI_SS);
    pinMode(VSPI_SS, OUTPUT);
    digitalWrite(VSPI_SS, HIGH);
    
    // Match the 20 MHz clock speed configuration inside your Verilog core
    vspi->beginTransaction(SPISettings(20000000, MSBFIRST, SPI_MODE0));
    
    Serial.println("ESP32-C3 Real-Time Streaming Engine Active");
}

void loop() {
    // 1. CRITICAL FIX: Gather a completely FRESH batch of samples from the physical mic
    unsigned long next_sample_time = micros();
    
    for (int i = 0; i < NUM_SAMPLES; i++) {
        audio_buffer[i] = analogRead(MIC_PIN);
        
        // Wait precisely until the next microsecond tick to preserve sampling rate integrity
        while (micros() < next_sample_time) {
            // Hard microsecond stall loop
        }
        next_sample_time += SAMPLE_PERIOD_US;
    }
    
    // 2. Transmit the fresh block downstream over the SPI pipeline
    digitalWrite(VSPI_SS, LOW); // Lower Chip Select to alert the FPGA
    
    // Send all 2048 bytes (1024 samples * 2 bytes each) sequentially at high speed
    vspi->transferBytes((uint8_t*)audio_buffer, NULL, NUM_SAMPLES * 2);
    
    digitalWrite(VSPI_SS, HIGH); // Raise CS to close the frame packet
    
    // A tiny 2ms breather to let the FPGA safely process the complete block
    delay(2); 
}