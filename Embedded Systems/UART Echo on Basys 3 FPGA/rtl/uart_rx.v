// ============================================================================
// uart_rx.v  -  UART receiver (Chu-style FSMD: idle / start / data / stop)
// ----------------------------------------------------------------------------
// ROLE IN THE ECHO SYSTEM
//   PC --RsRx--> [synchronizer in top.v] --rx--> uart_rx --dout--> RX FIFO
//
//   Turns the serial bit stream on `rx` back into a parallel byte on `dout`.
//   When a whole byte has arrived it pulses `rx_done_tick` for one clock, and
//   uart.v uses that pulse to write `dout` into the RX FIFO.
//
// THE UART FRAME (8N1, line idles high)
//      idle  start  b0  b1  b2  b3  b4  b5  b6  b7  stop  idle
//   ---1---|__0__|=========== data, LSB first ======|--1--|---
//
// HOW IT SAMPLES  (s_tick = 16 pulses per bit, from baud_gen)
//   * idle : wait for rx to fall -> that is the start of the start bit.
//   * start: count 8 ticks -> we are now in the MIDDLE of the start bit.
//   * data : from there, every 16 ticks we are in the middle of the next data
//            bit. Sample it, 8 times. The middle is the safest place to look,
//            far from the edges where the line is changing.
//   * stop : wait out the stop bit, then report "byte done".
//
// FSMD = Finite State Machine with Datapath. The state machine (state_reg)
// decides WHAT to do; the datapath registers (s_reg, n_reg, b_reg) hold the
// counters and the byte being built.
//
// TWO-BLOCK STYLE used in every module of this project:
//   * a clocked always block that only copies  xxx_next -> xxx_reg  (memory)
//   * a combinational always @* block that computes xxx_next from xxx_reg
//     and the inputs (decisions). Nothing is stored there.
// ============================================================================
module uart_rx
   #(
     parameter DBIT    = 8,      // data bits
               SB_TICK = 16      // ticks for stop bit (16 = 1 stop bit)
    )
   (
    input  wire       clk, reset,
    input  wire       rx,           // serial input (already synchronized in top.v)
    input  wire       s_tick,       // 16x oversampling tick from baud_gen
    output reg        rx_done_tick, // 1-clock pulse: a full byte is in dout
    output wire [7:0] dout          // the received byte
   );
   // symbolic state declaration
   // Names for the 4 states so the case statement reads like English.
   localparam [1:0]
      idle  = 2'b00,
      start = 2'b01,
      data  = 2'b10,
      stop  = 2'b11;
   // signal declaration "state" for state tracking
   // "s" for tick tracking, "n" for number  of recieved data tracking
   // "b" for reasembling the data
   //   state : which of the 4 states we are in
   //   s     : counts s_tick pulses inside the current bit (0..15)
   //   n     : counts data bits received so far (0..7)
   //   b     : shift register where the byte is assembled
   reg [1:0] state_reg, state_next;
   reg [3:0] s_reg, s_next;
   reg [2:0] n_reg, n_next;
   reg [7:0] b_reg, b_next;

   // body
   // ---- FSMD state & data registers, sequential ---------------------------
   // The "memory" of the receiver. On each rising clock edge every *_reg
   // takes the value its *_next was given by the block below. Reset puts the
   // receiver back in idle with all counters cleared. No decisions here.
   always @(posedge clk, posedge reset)
      if (reset)
         begin
            state_reg <= idle;
            s_reg     <= 0;
            n_reg     <= 0;
            b_reg     <= 0;
         end
      else
         begin
            state_reg <= state_next;
            s_reg     <= s_next;
            n_reg     <= n_next;
            b_reg     <= b_next;
         end
   // ---- FSMD next-state logic, combinational ------------------------------
   // The "brain". Re-evaluated whenever any input changes. It looks at the
   // current registers plus rx / s_tick and decides what the registers should
   // hold after the next clock edge.
   always @*
   begin
      // Defaults: "keep everything as it is" and "no done pulse".
      // Each case branch below only overwrites what must change. Without
      // these defaults the synthesizer would infer unwanted latches.
      state_next     = state_reg;
      rx_done_tick   = 1'b0;
      s_next         = s_reg;
      n_next         = n_reg;
      b_next         = b_reg;
      case (state_reg)
         // IDLE: line is high. A low level means a start bit is beginning.
         idle:
            if (~rx)
               begin
                  state_next = start;
                  s_next     = 0;      // start counting ticks from 0
               end
         // START: count 8 ticks (0..7) to reach the middle of the start bit,
         // then go and receive the data bits.
         start:
            if (s_tick)
               if (s_reg==7)
                  begin
                     state_next = data;
                     s_next     = 0;
                     n_next     = 0;   // no data bits received yet
                  end
               else
                  s_next = s_reg + 1;
         // DATA: every 16 ticks we land in the middle of the next data bit.
         data:
            if (s_tick)
               if (s_reg==15)
                  begin
                     s_next = 0;
                     // Shift right and put the new bit in at the top (MSB).
                     // UART sends LSB first, so after 8 shifts the first bit
                     // received has moved all the way down to b_reg[0].
                     b_next = {rx, b_reg[7:1]};
                     if (n_reg==(DBIT-1))
                        state_next = stop;   // that was the last data bit
                     else
                        n_next = n_reg + 1;  // count this bit, wait for next
                  end
               else
                  s_next = s_reg + 1;
         // STOP: wait SB_TICK ticks through the stop bit, then pulse
         // rx_done_tick for one clock so uart.v writes the byte into the
         // RX FIFO, and go back to idle for the next byte.
         stop:
            if (s_tick)
               if (s_reg==(SB_TICK-1))
                  begin
                     state_next   = idle;
                     rx_done_tick = 1'b1;
                  end
               else
                  s_next = s_reg + 1;
      endcase
   end
   // ---- output --------------------------------------------------------------
   // The assembled byte. It is only meaningful when rx_done_tick is high.
   assign dout = b_reg;

endmodule
