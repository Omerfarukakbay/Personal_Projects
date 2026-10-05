// ============================================================================
// baud_gen.v  -  baud-rate tick generator (Chu-style modulo-M counter)
// ----------------------------------------------------------------------------
// ROLE IN THE ECHO SYSTEM
//   The board clock is 100 MHz, far faster than the serial line (115200 bits
//   per second). The receiver and transmitter need a slower "metronome" so
//   they know when to look at / change the serial line. This module is that
//   metronome.
//
//   It counts 0, 1, 2, ... M-1, 0, 1, ... and pulses `tick` high for ONE clock
//   cycle every time the count reaches M-1. So `tick` fires once every M
//   clocks.
//
//   We want 16 ticks per serial bit (16x "oversampling", needed by uart_rx to
//   find the middle of each bit):
//       100 MHz / (16 * 115200) = 54.25  ->  M = 54
//       real tick rate = 100 MHz / 54 = 1.852 MHz  (16 ticks = 8.64 us per bit)
//   That is 0.5 % off the ideal 8.68 us, well inside what UART tolerates.
//
//   One baud_gen is shared by uart_rx and uart_tx (see uart.v).
//
// PARAMETERS
//   M : modulus, the counter wraps after M counts (tick period = M clocks)
//   N : number of bits in the counter, must satisfy 2**N >= M (2**6 = 64 >= 54)
// ============================================================================
module baud_gen
   #(
    parameter M= 54,
    parameter N =6
   )
   (
    input  wire        clk, reset,
    output wire        tick,     // 1-clock-wide pulse, 16 per UART bit
    output wire        [N-1 : 0] q   // current count (unused in this design)
   );
   //signal declaration
   reg [N-1 : 0] r_reg;    // the counter flip-flops (current value)
   wire [N-1 : 0] r_next;  // value the counter will take at the next clock edge

   //body
   // ---- register (sequential) ---------------------------------------------
   // On every rising clock edge, store the next value. Reset forces 0.
   // This is the ONLY place where the counter actually changes.
   always @(posedge clk, posedge reset)
      if (reset)
         r_reg <= 0;
      else
         r_reg <= r_next;
   // ---- next-state logic (combinational) ----------------------------------
   // Count up by one; after reaching M-1 wrap back to 0.
   assign r_next = (r_reg ==(M-1)) ? 0 : r_reg + 1;
   // ---- output logic --------------------------------------------------------
   assign q = r_reg;
   // `tick` is high only during the single clock cycle where the count is M-1.
   // uart_rx and uart_tx use it as their "advance one step" enable.
   assign tick = (r_reg ==(M-1)) ? 1'b1 : 1'b0;
endmodule
