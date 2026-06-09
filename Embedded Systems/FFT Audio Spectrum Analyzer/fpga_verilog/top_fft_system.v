// =============================================================================
// Real-Time Audio FFT Spectrum Analyzer - Basys 3 FPGA
// Author: Omer Faruk Akbay
// Date:   June 2026
// Tools:  Xilinx Vivado 2023.2
// Device: xc7a35tcpg236-1 (Artix-7)
//
// Module hierarchy:
//   top_fft_system
//     ├── spi_slave_receiver   (SPI slave, CDC, 16-bit sample assembly)
//     ├── xfft_0               (Xilinx FFT IP Core - 1024-point, from IP catalog)
//     ├── vga_controller       (640x480 @ 60 Hz sync generator)
//     └── spectrum_renderer    (FFT magnitude → VGA bar graph)
//
// NOTE: xfft_0 is the Xilinx FFT IP Core instantiated from Vivado IP Catalog.
//       It is NOT included in this file. Add it via Vivado → IP Catalog → FFT.
//       Configuration: 1024-point, natural I/O, 16-bit fixed-point, Artix-7.
// =============================================================================

module top_fft_system (
    input  wire        clk,        // Basys 3 Master System Clock (100 MHz)
    input  wire        rst,        // Active-High Reset (Center Button)

    // Slide Switches
    input  wire [3:0]  sw,         // sw[0]=Test Mode | sw[1]=Test Freq | sw[3]=Top 3 Filter Mode

    // Physical SPI Inputs from ESP32-C3
    input  wire        spi_cs,
    input  wire        spi_sck,
    input  wire        spi_mosi,

    // Physical VGA Ports
    output wire        h_sync,
    output wire        v_sync,
    output wire [3:0]  vga_red,
    output wire [3:0]  vga_green,
    output wire [3:0]  vga_blue,

    // Diagnostic LEDs
    output wire [15:0] led
);

    wire rst_n = ~rst;

    // --- Internal Interconnect Wires ---
    wire [15:0] raw_audio_sample;
    wire        audio_sample_valid;

    wire [31:0] fft_input_tdata;
    reg         fft_input_tlast;
    wire        fft_input_tready;
    reg  [9:0]  sample_count;

    wire [31:0] fft_output_tdata;
    wire        fft_output_valid;
    wire        fft_output_last;

    wire        video_active_flag;
    wire [9:0]  vga_x_coord;
    wire [9:0]  vga_y_coord;

    // --- INTERNAL SINE WAVE GENERATOR ENGINE ---
    reg [11:0] sample_rate_counter;
    reg        internal_sample_valid;
    reg [3:0]  sine_index;
    reg signed [15:0] internal_sine_value;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sample_rate_counter   <= 12'd0;
            internal_sample_valid <= 1'b0;
        end else begin
            if (sample_rate_counter == 12'd2267) begin
                sample_rate_counter   <= 12'd0;
                internal_sample_valid <= 1'b1;
            end else begin
                sample_rate_counter   <= sample_rate_counter + 1'b1;
                internal_sample_valid <= 1'b0;
            end
        end
    end

    always @(*) begin
        case (sine_index)
            4'd0:  internal_sine_value = 16'd0;
            4'd1:  internal_sine_value = 16'd765;
            4'd2:  internal_sine_value = 16'd1414;
            4'd3:  internal_sine_value = 16'd1848;
            4'd4:  internal_sine_value = 16'd2000;
            4'd5:  internal_sine_value = 16'd1848;
            4'd6:  internal_sine_value = 16'd1414;
            4'd7:  internal_sine_value = 16'd765;
            4'd8:  internal_sine_value = 16'd0;
            4'd9:  internal_sine_value = -16'd765;
            4'd10: internal_sine_value = -16'd1414;
            4'd11: internal_sine_value = -16'd1848;
            4'd12: internal_sine_value = -16'd2000;
            4'd13: internal_sine_value = -16'd1848;
            4'd14: internal_sine_value = -16'd1414;
            4'd15: internal_sine_value = -16'd765;
        endcase
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sine_index <= 4'd0;
        end else if (internal_sample_valid) begin
            if (sw[1] == 1'b0)
                sine_index <= sine_index + 4'd1;
            else
                sine_index <= sine_index + 4'd2;
        end
    end

    // --- HARDWARE MULTIPLEXER INTERCONNECT ---
    wire signed [15:0] selected_audio;
    wire               selected_valid;

    wire signed [15:0] signed_mic_audio = $signed({1'b0, raw_audio_sample}) - 16'd1760;

    assign selected_audio = (sw[0] == 1'b1) ? internal_sine_value   : signed_mic_audio;
    assign selected_valid = (sw[0] == 1'b1) ? internal_sample_valid : audio_sample_valid;

    assign fft_input_tdata = {16'd0, selected_audio};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sample_count    <= 10'd0;
            fft_input_tlast <= 1'b0;
        end else begin
            if (selected_valid && fft_input_tready) begin
                if (sample_count == 10'd1023) begin
                    sample_count <= 10'd0;
                end else begin
                    sample_count <= sample_count + 1'b1;
                end
            end
            fft_input_tlast <= (sample_count == 10'd1023) && selected_valid;
        end
    end

    reg valid_toggle;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) valid_toggle <= 1'b0;
        else if (selected_valid) valid_toggle <= ~valid_toggle;
    end

    assign led[0]     = spi_cs;
    assign led[1]     = spi_sck;
    assign led[2]     = spi_mosi;
    assign led[3]     = valid_toggle;
    assign led[4]     = fft_output_valid;
    assign led[5]     = fft_output_last;
    assign led[7:6]   = 2'b00;
    assign led[15:8]  = selected_audio[15:8];

    // --- Module Instantiations ---
    spi_slave_receiver spi_rx (
        .clk(clk), .rst_n(rst_n), .spi_cs(spi_cs), .spi_sck(spi_sck), .spi_mosi(spi_mosi),
        .sample_out(raw_audio_sample), .sample_valid(audio_sample_valid)
    );

    xfft_0 your_fft_core (
        .aclk(clk), .aresetn(rst_n), .s_axis_config_tdata(16'b0010101010101010),
        .s_axis_config_tvalid(1'b1), .s_axis_config_tready(),
        .s_axis_data_tdata(fft_input_tdata), .s_axis_data_tvalid(selected_valid),
        .s_axis_data_tready(fft_input_tready), .s_axis_data_tlast(fft_input_tlast),
        .m_axis_data_tdata(fft_output_tdata), .m_axis_data_tvalid(fft_output_valid),
        .m_axis_data_tready(1'b1), .m_axis_data_tlast(fft_output_last)
    );

    vga_controller video_engine (
        .clk(clk), .rst_n(rst_n), .h_sync(h_sync), .v_sync(v_sync),
        .video_on(video_active_flag), .pixel_x(vga_x_coord), .pixel_y(vga_y_coord)
    );

    spectrum_renderer visual_generator (
        .clk(clk), .rst_n(rst_n),
        .sw3(sw[3]),
        .fft_axis_tdata(fft_output_tdata), .fft_axis_tvalid(fft_output_valid), .fft_axis_tlast(fft_output_last),
        .video_on(video_active_flag), .pixel_x(vga_x_coord), .pixel_y(vga_y_coord),
        .vga_red(vga_red), .vga_green(vga_green), .vga_blue(vga_blue)
    );

endmodule


// =============================================================================
// SPI Slave Receiver
// Captures SPI data from ESP32-C3 and assembles 16-bit audio samples.
// Handles clock domain crossing: 20 MHz SPI → 100 MHz system clock
// =============================================================================
module spi_slave_receiver (
    input  wire        clk,        // Basys 3 Master System Clock (100 MHz)
    input  wire        rst_n,      // Active-low reset

    // Physical SPI Pins from Pmod JA
    input  wire        spi_cs,     // Chip Select
    input  wire        spi_sck,    // Serial Clock (20 MHz)
    input  wire        spi_mosi,   // Master Out Slave In

    // Parallel Output Interface to the FFT/FIFO Engine
    output reg  [15:0] sample_out, // Fully assembled 16-bit audio sample
    output reg         sample_valid// High for 1 clock cycle when sample_out is ready
);

    // --- 1. Synchronizers and Edge Detectors (CDC Handling) ---
    reg [1:0] cs_sync;   // 2-stage synchronizer for CS
    reg [2:0] sck_sync;  // 3-stage for SCK edge detection
    reg [1:0] mosi_sync; // 2-stage synchronizer for MOSI

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cs_sync   <= 2'b11;
            sck_sync  <= 3'b000;
            mosi_sync <= 2'b00;
        end else begin
            cs_sync   <= {cs_sync[0],     spi_cs};
            sck_sync  <= {sck_sync[1:0],  spi_sck};
            mosi_sync <= {mosi_sync[0],   spi_mosi};
        end
    end

    wire cs_active = ~cs_sync[1];               // True when CS driven LOW by ESP32
    wire sck_rise  = (sck_sync[2:1] == 2'b01); // Rising edge of SPI Clock
    wire mosi_data = mosi_sync[1];              // Stabilized data bit

    // --- 2. SPI Bit & Byte Assembly State Machine ---
    reg [3:0] bit_cnt;
    reg       byte_idx;
    reg [7:0] shift_reg;
    reg [7:0] low_byte_hold;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bit_cnt      <= 4'd0;
            byte_idx     <= 1'b0;
            shift_reg    <= 8'd0;
            low_byte_hold<= 8'd0;
            sample_out   <= 16'd0;
            sample_valid <= 1'b0;
        end else begin
            sample_valid <= 1'b0;

            if (!cs_active) begin
                bit_cnt  <= 4'd0;
                byte_idx <= 1'b0;
            end else if (sck_rise) begin
                shift_reg <= {shift_reg[6:0], mosi_data};

                if (bit_cnt == 4'd7) begin
                    bit_cnt <= 4'd0;

                    if (byte_idx == 1'b0) begin
                        low_byte_hold <= {shift_reg[6:0], mosi_data};
                        byte_idx      <= 1'b1;
                    end else begin
                        sample_out   <= { {shift_reg[6:0], mosi_data}, low_byte_hold };
                        sample_valid <= 1'b1;
                        byte_idx     <= 1'b0;
                    end
                end else begin
                    bit_cnt <= bit_cnt + 1'b1;
                end
            end
        end
    end

endmodule


// =============================================================================
// VGA Controller
// Generates 640x480 @ 60 Hz standard timing signals.
// Pixel clock: 25 MHz derived from 100 MHz system clock (4-cycle divider).
// =============================================================================
module vga_controller (
    input  wire        clk,        // Master 100 MHz clock
    input  wire        rst_n,      // Active-low reset
    output reg         h_sync,     // Physical horizontal sync pin
    output reg         v_sync,     // Physical vertical sync pin
    output wire        video_on,   // High when inside the active 640x480 display area
    output wire [9:0]  pixel_x,    // Current X coordinate (0 to 639)
    output wire [9:0]  pixel_y     // Current Y coordinate (0 to 479)
);

    // --- 1. Generate 25 MHz Pixel Clock ---
    reg [1:0] clk_div;
    wire pix_clk;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            clk_div <= 2'b00;
        else
            clk_div <= clk_div + 1'b1;
    end
    assign pix_clk = (clk_div == 2'b11);

    // --- 2. Timing Parameters ---
    localparam H_ACTIVE   = 640;
    localparam H_FP       = 16;
    localparam H_SYNC     = 96;
    localparam H_BP       = 48;
    localparam H_TOTAL    = 800;

    localparam V_ACTIVE   = 480;
    localparam V_FP       = 10;
    localparam V_SYNC     = 2;
    localparam V_BP       = 33;
    localparam V_TOTAL    = 525;

    // --- 3. Horizontal and Vertical Counters ---
    reg [9:0] h_count;
    reg [9:0] v_count;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            h_count <= 10'd0;
            v_count <= 10'd0;
        end else if (pix_clk) begin
            if (h_count == H_TOTAL - 1) begin
                h_count <= 10'd0;
                if (v_count == V_TOTAL - 1)
                    v_count <= 10'd0;
                else
                    v_count <= v_count + 1'b1;
            end else begin
                h_count <= h_count + 1'b1;
            end
        end
    end

    // --- 4. Generate Sync Timing Signals (Active Low) ---
    always @(posedge clk) begin
        h_sync <= ~((h_count >= (H_ACTIVE + H_FP)) && (h_count < (H_ACTIVE + H_FP + H_SYNC)));
        v_sync <= ~((v_count >= (V_ACTIVE + V_FP)) && (v_count < (V_ACTIVE + V_FP + V_SYNC)));
    end

    // --- 5. Output Coordinates and Video Blanking Flag ---
    assign video_on = (h_count < H_ACTIVE) && (v_count < V_ACTIVE);
    assign pixel_x  = (h_count < H_ACTIVE) ? h_count : 10'd0;
    assign pixel_y  = (v_count < V_ACTIVE) ? v_count : 10'd0;

endmodule


// =============================================================================
// Spectrum Renderer
// Reads FFT output magnitudes and renders a real-time bar graph on the VGA.
//
// Features:
//   - Alpha-max/beta-min magnitude approximation
//   - 512-entry BRAM stores one full spectrum frame
//   - Logarithmic frequency axis (8-segment piecewise mapping)
//   - Top-3 Peak isolation with proximity-lockout sorter (sw3)
//   - DC/hum filter: ignores bins 0-6
//   - Cyan bitmap font frequency labels (100, 500, 1k, 2k, 5k, 10k, 15k Hz)
// =============================================================================
module spectrum_renderer (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        sw3,              // High = Top-3 Peak isolation mode

    // Interface from Xilinx FFT Core
    input  wire [31:0] fft_axis_tdata,
    input  wire        fft_axis_tvalid,
    input  wire        fft_axis_tlast,

    // Interface from VGA Controller
    input  wire        video_on,
    input  wire [9:0]  pixel_x,
    input  wire [9:0]  pixel_y,

    // Direct Color Output to VGA Pins
    output reg  [3:0]  vga_red,
    output reg  [3:0]  vga_green,
    output reg  [3:0]  vga_blue
);

    // --- 1. FFT Magnitude Calculation ---
    wire signed [15:0] fft_real = fft_axis_tdata[15:0];
    wire signed [15:0] fft_imag = fft_axis_tdata[31:16];

    wire [15:0] abs_real = (fft_real[15] == 1'b1) ? -fft_real : fft_real;
    wire [15:0] abs_imag = (fft_imag[15] == 1'b1) ? -fft_imag : fft_imag;

    // Alpha-max / beta-min magnitude approximation (avoids sqrt)
    wire [15:0] approx_mag = (abs_real > abs_imag) ? (abs_real + (abs_imag >> 1)) : (abs_imag + (abs_real >> 1));

    wire [15:0] boosted_magnitude = approx_mag << 1;   // 2x gain boost

    reg [9:0] incoming_addr;
    reg       fft_valid_pipe;
    reg [9:0] write_addr_pipe;
    reg [7:0] magnitude_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            incoming_addr <= 10'd0;
        else if (fft_axis_tvalid)
            incoming_addr <= (fft_axis_tlast) ? 10'd0 : incoming_addr + 1'b1;
    end

    always @(posedge clk) begin
        fft_valid_pipe  <= fft_axis_tvalid;
        write_addr_pipe <= incoming_addr;
        magnitude_reg   <= (boosted_magnitude > 16'd255) ? 8'd255 : boosted_magnitude[7:0];
    end

    // --- 2. Dual-Port Spectrum Memory Block ---
    reg [7:0] spectrum_mem [0:511];
    reg [7:0] current_bar_height;

    always @(posedge clk) begin
        if (fft_valid_pipe && (write_addr_pipe < 10'd512)) begin
            spectrum_mem[write_addr_pipe] <= magnitude_reg;
        end
    end

    // --- 3. PROXIMITY-LOCKOUT TOP-3 PEAK SORTING ENGINE ---
    reg [7:0] m1_mag, m2_mag, m3_mag;
    reg [9:0] m1_idx, m2_idx, m3_idx;
    reg [9:0] locked_p1_idx, locked_p2_idx, locked_p3_idx;

    wire [9:0] dist_to_m1 = (write_addr_pipe > m1_idx) ? (write_addr_pipe - m1_idx) : (m1_idx - write_addr_pipe);
    wire [9:0] dist_to_m2 = (write_addr_pipe > m2_idx) ? (write_addr_pipe - m2_idx) : (m2_idx - write_addr_pipe);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m1_mag <= 8'd0; m2_mag <= 8'd0; m3_mag <= 8'd0;
            m1_idx <= 10'd0; m2_idx <= 10'd0; m3_idx <= 10'd0;
            locked_p1_idx <= 10'd0; locked_p2_idx <= 10'd0; locked_p3_idx <= 10'd0;
        end else if (fft_valid_pipe && (write_addr_pipe < 10'd512)) begin

            if (write_addr_pipe == 10'd6) begin
                m1_mag <= magnitude_reg; m1_idx <= write_addr_pipe;
                m2_mag <= 8'd0;          m2_idx <= 10'd0;
                m3_mag <= 8'd0;          m3_idx <= 10'd0;
            end
            else if (write_addr_pipe > 10'd6) begin

                if (magnitude_reg > m1_mag) begin
                    if (dist_to_m1 <= 10'd4) begin
                        m1_mag <= magnitude_reg;
                        m1_idx <= write_addr_pipe;
                    end else begin
                        m3_mag <= m2_mag; m3_idx <= m2_idx;
                        m2_mag <= m1_mag; m2_idx <= m1_idx;
                        m1_mag <= magnitude_reg; m1_idx <= write_addr_pipe;
                    end
                end
                else if (magnitude_reg > m2_mag) begin
                    if (dist_to_m1 > 10'd4) begin
                        if (dist_to_m2 <= 10'd4) begin
                            m2_mag <= magnitude_reg;
                            m2_idx <= write_addr_pipe;
                        end else begin
                            m3_mag <= m2_mag; m3_idx <= m2_idx;
                            m2_mag <= magnitude_reg; m2_idx <= write_addr_pipe;
                        end
                    end
                end
                else if (magnitude_reg > m3_mag) begin
                    if ((dist_to_m1 > 10'd4) && (dist_to_m2 > 10'd4)) begin
                        m3_mag <= magnitude_reg;
                        m3_idx <= write_addr_pipe;
                    end
                end
            end

            if (write_addr_pipe == 10'd511) begin
                locked_p1_idx <= m1_idx;
                locked_p2_idx <= m2_idx;
                locked_p3_idx <= m3_idx;
            end
        end
    end

    // --- 4. Logarithmic Index Mapping Matrix ---
    reg [9:0] log_index;
    always @(*) begin
        log_index = 10'd0;
        if (pixel_x >= 10'd64 && pixel_x < 10'd576) begin
            case ((pixel_x - 10'd64) >> 6)
                0: log_index = 2   + (((pixel_x - 10'd64)   * 3)   / 100);
                1: log_index = 4   + (((pixel_x - 10'd128)  * 8)   / 100);
                2: log_index = 9   + (((pixel_x - 10'd192)  * 15)  / 100);
                3: log_index = 19  + (((pixel_x - 10'd256)  * 28)  / 100);
                4: log_index = 37  + (((pixel_x - 10'd320)  * 52)  / 100);
                5: log_index = 70  + (((pixel_x - 10'd384)  * 95)  / 100);
                6: log_index = 131 + (((pixel_x - 10'd448)  * 145) / 100);
                7: log_index = 223 + (((pixel_x - 10'd512)  * 195) / 100);
            endcase
        end
    end

    wire [9:0] safe_lookup_addr = ((log_index << 1) > 10'd511) ? 10'd511 : (log_index << 1);

    wire is_p1  = (safe_lookup_addr == locked_p1_idx);
    wire is_p2  = (safe_lookup_addr == locked_p2_idx);
    wire is_p3  = (safe_lookup_addr == locked_p3_idx);
    wire is_any_top3 = is_p1 || is_p2 || is_p3;

    always @(posedge clk) begin
        if (pixel_x >= 10'd64 && pixel_x < 10'd576) begin
            if (sw3 && !is_any_top3)
                current_bar_height <= 8'd0;
            else
                current_bar_height <= spectrum_mem[safe_lookup_addr];
        end else begin
            current_bar_height <= 8'd0;
        end
    end

    // --- 5. Character-Generating Font Matrix Engine ---
    reg [14:0] char_rom;
    reg [2:0]  char_col;
    reg [2:0]  char_row;
    reg        text_pixel;

    always @(*) begin
        text_pixel = 1'b0;
        char_rom   = 15'd0;
        char_col   = 3'd0;
        char_row   = (pixel_y >= 450 && pixel_y <= 454) ? (pixel_y - 10'd450) : 3'd0;

        if (pixel_y >= 10'd450 && pixel_y <= 10'd454) begin
            if (pixel_x >= 64 && pixel_x < 76) begin
                char_col = (pixel_x - 64) % 4;
                case ((pixel_x - 64) / 4)
                    0: char_rom = 15'h2492; // '1'
                    1: char_rom = 15'h7b6f; // '0'
                    2: char_rom = 15'h7b6f; // '0'
                endcase
                text_pixel = char_rom[14 - (char_row * 3 + char_col)] && (char_col < 3);
            end
            else if (pixel_x >= 212 && pixel_x < 224) begin
                char_col = (pixel_x - 212) % 4;
                case ((pixel_x - 212) / 4)
                    0: char_rom = 15'h79cf; // '5'
                    1: char_rom = 15'h7b6f; // '0'
                    2: char_rom = 15'h7b6f; // '0'
                endcase
                text_pixel = char_rom[14 - (char_row * 3 + char_col)] && (char_col < 3);
            end
            else if (pixel_x >= 271 && pixel_x < 279) begin
                char_col = (pixel_x - 271) % 4;
                case ((pixel_x - 271) / 4)
                    0: char_rom = 15'h2492; // '1'
                    1: char_rom = 15'h5bad; // 'k'
                endcase
                text_pixel = char_rom[14 - (char_row * 3 + char_col)] && (char_col < 3);
            end
            else if (pixel_x >= 338 && pixel_x < 346) begin
                char_col = (pixel_x - 338) % 4;
                case ((pixel_x - 338) / 4)
                    0: char_rom = 15'h73e7; // '2'
                    1: char_rom = 15'h5bad; // 'k'
                endcase
                text_pixel = char_rom[14 - (char_row * 3 + char_col)] && (char_col < 3);
            end
            else if (pixel_x >= 433 && pixel_x < 441) begin
                char_col = (pixel_x - 433) % 4;
                case ((pixel_x - 433) / 4)
                    0: char_rom = 15'h79cf; // '5'
                    1: char_rom = 15'h5bad; // 'k'
                endcase
                text_pixel = char_rom[14 - (char_row * 3 + char_col)] && (char_col < 3);
            end
            else if (pixel_x >= 517 && pixel_x < 529) begin
                char_col = (pixel_x - 517) % 4;
                case ((pixel_x - 517) / 4)
                    0: char_rom = 15'h2492; // '1'
                    1: char_rom = 15'h7b6f; // '0'
                    2: char_rom = 15'h5bad; // 'k'
                endcase
                text_pixel = char_rom[14 - (char_row * 3 + char_col)] && (char_col < 3);
            end
            else if (pixel_x >= 575 && pixel_x < 587) begin
                char_col = (pixel_x - 575) % 4;
                case ((pixel_x - 575) / 4)
                    0: char_rom = 15'h2492; // '1'
                    1: char_rom = 15'h79cf; // '5'
                    2: char_rom = 15'h5bad; // 'k'
                endcase
                text_pixel = char_rom[14 - (char_row * 3 + char_col)] && (char_col < 3);
            end
        end
    end

    // --- 6. VGA Compositing & Color Isolation Engine ---
    wire [9:0] bar_bottom_line = 10'd440;
    wire [9:0] bar_top_line    = bar_bottom_line - current_bar_height;

    wire is_tick = (pixel_y >= 441 && pixel_y <= 444) &&
                   (pixel_x == 64  || pixel_x == 212 || pixel_x == 271 ||
                    pixel_x == 338 || pixel_x == 433 || pixel_x == 517 || pixel_x == 575);

    always @(*) begin
        if (!video_on) begin
            vga_red   = 4'b0000;
            vga_green = 4'b0000;
            vga_blue  = 4'b0000;
        end else begin
            vga_blue  = 4'b0000;

            if ((pixel_x >= 10'd64) && (pixel_x < 10'd576) && (pixel_y >= bar_top_line) && (pixel_y <= bar_bottom_line)) begin
                if (sw3) begin
                    if (is_p1) begin
                        vga_red   = 4'b1111;
                        vga_green = 4'b0000;
                    end else begin
                        vga_red   = 4'b1111;
                        vga_green = 4'b1111;
                    end
                end else begin
                    vga_red   = 4'b1111;
                    vga_green = 4'b1111;
                end
            end
            else if (text_pixel) begin
                vga_red   = 4'b0000;
                vga_green = 4'b1111;
                vga_blue  = 4'b1111;
            end
            else if (((pixel_y == bar_bottom_line + 1) && (pixel_x >= 10'd64) && (pixel_x < 10'd576)) || is_tick) begin
                vga_red   = 4'b0111;
                vga_green = 4'b0111;
            end
            else begin
                vga_red   = 4'b0000;
                vga_green = 4'b0000;
            end
        end
    end

endmodule
