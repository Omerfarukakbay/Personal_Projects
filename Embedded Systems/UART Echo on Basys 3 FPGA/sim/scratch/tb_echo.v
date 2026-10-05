`timescale 1ns / 1ps
// ============================================================================
// tb_echo.v  -  self-checking "scratch" testbench for the UART echo design
// ----------------------------------------------------------------------------
// Purpose: prove that top.v echoes every byte it receives.
//   1. Generate the 100 MHz clock and hold reset (btnC) for a moment.
//   2. Act like the PC: send bytes serially on RsRx (115200 baud, 8N1).
//   3. Act like the PC's receiver: watch RsTx, decode each byte that comes
//      back, and compare it with what was sent. Print PASS or FAIL.
//
// This file lives in sim/scratch/ on purpose: scripts/create_project.tcl only
// adds sim/*.v, so this testbench never gets mixed into the Vivado project.
// Your own sim/tb_top.v stays yours to write.
//
// Quick run:  scripts\sim_echo.bat        (add "gui" to open the waveform)
// By hand, from build\xsim (see README, "Simulate"):
//   xvlog ..\..\rtl\*.v ..\..\sim\scratch\tb_echo.v
//   xelab tb_echo -timescale 1ns/1ps -debug typical -s tb_echo_sim
//   xsim  tb_echo_sim -R
// ============================================================================
module tb_echo;
   // ---- timing constants ---------------------------------------------------
   // The PC sends at exactly 115200 baud -> 1 / 115200 s = 8680 ns per bit.
   localparam integer BIT_PC   = 8680;
   // The FPGA's own bit time: baud_gen ticks every 54 clocks (540 ns) and a
   // bit lasts 16 ticks -> 8640 ns. That is 0.5 % off 115200, which UART
   // tolerates easily (a few % is fine).
   localparam integer BIT_FPGA = 8640;
   localparam integer NBYTES   = 3;

   // ---- signals that connect to the design under test (DUT) ---------------
   reg         clk  = 0;
   reg         btnC = 1;          // start in reset
   reg  [15:0] sw   = 0;          // switches are unused by the design
   reg         RsRx = 1;          // UART line idles high
   wire        RsTx;
   wire [15:0] led;

   top dut (.clk(clk), .btnC(btnC), .sw(sw), .RsRx(RsRx), .RsTx(RsTx), .led(led));

   always #5 clk = ~clk;          // 10 ns period = 100 MHz

   // ---- test data ----------------------------------------------------------
   reg [7:0] tx_bytes [0:NBYTES-1];   // what the "PC" sends
   integer   errors = 0;
   integer   seen   = 0;              // bytes received back so far

   // ---- task: send one byte like a PC serial port would -------------------
   // Frame = start bit (0), 8 data bits LSB first, stop bit (1).
   task send_byte(input [7:0] b);
      integer i;
      begin
         RsRx = 1'b0;                 // start bit
         #(BIT_PC);
         for (i = 0; i < 8; i = i + 1) begin
            RsRx = b[i];              // LSB first
            #(BIT_PC);
         end
         RsRx = 1'b1;                 // stop bit
         #(BIT_PC);
      end
   endtask

   // ---- stimulus: reset, then send the bytes back to back -----------------
   initial begin
      $timeformat(-9, 0, " ns", 12);  // print times in ns
      tx_bytes[0] = 8'h55;            // 0101_0101, alternating bits
      tx_bytes[1] = 8'h0F;            // 0000_1111, low nibble
      tx_bytes[2] = 8'h48;            // ASCII 'H'
      #100 btnC = 0;                  // release reset after 100 ns
      #1000;
      send_byte(tx_bytes[0]);
      send_byte(tx_bytes[1]);
      send_byte(tx_bytes[2]);
   end

   // ---- monitor: decode every byte the FPGA sends back on RsTx -----------
   reg [7:0] rx_b;
   integer   k;
   initial begin : monitor
      wait (btnC == 0);
      forever begin
         @(negedge RsTx);             // falling edge = start bit begins
         #(BIT_FPGA/2);               // move to the middle of the start bit
         if (RsTx !== 1'b0) begin
            $display("%t  ERROR: glitch, start bit not low", $time);
            errors = errors + 1;
         end
         for (k = 0; k < 8; k = k + 1) begin
            #(BIT_FPGA);              // middle of data bit k
            rx_b[k] = RsTx;
         end
         #(BIT_FPGA);                 // middle of stop bit
         if (RsTx !== 1'b1) begin
            $display("%t  ERROR: stop bit not high", $time);
            errors = errors + 1;
         end
         if (rx_b === tx_bytes[seen])
            $display("%t  echo %0d: sent %h, got %h  OK", $time, seen, tx_bytes[seen], rx_b);
         else begin
            $display("%t  echo %0d: sent %h, got %h  MISMATCH", $time, seen, tx_bytes[seen], rx_b);
            errors = errors + 1;
         end
         seen = seen + 1;
      end
   end

   // ---- end of test: report and stop --------------------------------------
   initial begin
      #(4 * NBYTES * 10 * BIT_PC);    // generous time for all echoes
      if (seen != NBYTES) begin
         $display("Only %0d of %0d bytes came back", seen, NBYTES);
         errors = errors + 1;
      end
      $display("led[7:0] = %h (last echoed byte), led[8] rx_empty = %b, led[9] tx_full = %b",
               led[7:0], led[8], led[9]);
      if (errors == 0) $display("RESULT: PASS");
      else             $display("RESULT: FAIL (%0d errors)", errors);
      $finish;
   end
endmodule
