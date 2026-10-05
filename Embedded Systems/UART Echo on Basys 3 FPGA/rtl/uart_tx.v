// ============================================================================
// uart_tx.v  -  UART transmitter (Chu-style FSMD: idle / start / data / stop)
// ----------------------------------------------------------------------------
// ROLE IN THE ECHO SYSTEM
//   TX FIFO --din--> uart_tx --tx--> RsTx pin --> PC
//
//   The mirror image of uart_rx: it takes a parallel byte on `din` and sends
//   it out on `tx` one bit at a time as a UART frame:
//      start bit (0), 8 data bits LSB first, stop bit (1).
//
//   Handshake with the TX FIFO (wired in uart.v):
//     * tx_start = "TX FIFO is not empty"  -> there is a byte to send.
//     * In idle, the byte on din is copied into b_reg and sending begins.
//     * When the stop bit is finished, tx_done_tick pulses for one clock.
//       uart.v uses that pulse as the FIFO's `rd`, which removes the byte
//       just sent so the next one appears on din.
//
// TIMING
//   Each bit is held on the line for 16 s_ticks (= one bit period). The
//   transmitter does not need to find bit centres like the receiver does,
//   because it creates the edges itself; it only uses s_tick as a timer.
//
// Same two-block FSMD style as uart_rx.v (clocked registers + combinational
// next-state logic).
// ============================================================================
module uart_tx
   #(
     parameter DBIT    = 8,      // data bits
               SB_TICK = 16      // ticks for stop bit (16 = 1 stop bit)
    )
   (
    input  wire       clk, reset,
    input  wire       tx_start,     // high when there is a byte waiting to send
    input  wire       s_tick,       // 16x tick from baud_gen
    input  wire [7:0] din,          // byte to send (oldest byte in the TX FIFO)
    output reg        tx_done_tick, // 1-clock pulse: finished sending a byte
    output wire       tx            // serial output line
   );

   // symbolic state declaration
   localparam [1:0]
      idle  = 2'b00,
      start = 2'b01,
      data  = 2'b10,
      stop  = 2'b11;
   // signal declaration
   //   state : which of the 4 states we are in
   //   s     : counts s_tick pulses inside the current bit (0..15)
   //   n     : counts data bits sent so far (0..7)
   //   b     : shift register holding the bits still to be sent
   //   tx    : registered copy of the serial output (see "output" below)
   reg [1:0] state_reg, state_next;
   reg [3:0] s_reg, s_next;
   reg [2:0] n_reg, n_next;
   reg [7:0] b_reg, b_next;
   reg       tx_reg, tx_next;

   // body
   // ---- FSMD state & data registers, sequential ---------------------------
   // Memory of the transmitter: copy every *_next into *_reg on the clock
   // edge. Reset returns to idle and drives the line high (UART idle level),
   // so the PC never sees a false start bit while we are in reset.
   always @(posedge clk, posedge reset)
      if (reset)
         begin
            state_reg <= idle;
            s_reg     <= 0;
            n_reg     <= 0;
            b_reg     <= 0;
            tx_reg    <= 1'b1;
         end
      else
         begin
            state_reg <= state_next;
            s_reg     <= s_next;
            n_reg     <= n_next;
            b_reg     <= b_next;
            tx_reg    <= tx_next;
         end
   // ---- FSMD next-state logic, combinational ------------------------------
   // Decides what the line should show next and when to move to the next
   // bit / state.
   always @*
   begin
      // Defaults: keep everything, no done pulse (prevents latches).
      state_next     = state_reg;
      tx_done_tick   = 1'b0;
      s_next         = s_reg;
      n_next         = n_reg;
      b_next         = b_reg;
      tx_next        = tx_reg;
      case (state_reg)
         // IDLE: hold the line high. If the FIFO has a byte, grab it into
         // b_reg and start the frame.
         idle:
            begin
               tx_next = 1'b1;
               if (tx_start)
                  begin
                     state_next = start;
                     s_next     = 0;
                     b_next     = din;   // latch the byte to send
                  end
            end
         // START: drive 0 for one full bit time (16 ticks).
         start:
            begin
               tx_next = 1'b0;
               if (s_tick)
                  if (s_reg==15)
                     begin
                        state_next = data;
                        s_next     = 0;
                        n_next     = 0;
                     end
                  else
                     s_next = s_reg + 1;
            end
         // DATA: put b_reg[0] on the line for 16 ticks, then shift right so
         // the next bit moves into position 0. LSB goes out first.
         data:
            begin
               tx_next = b_reg[0];
               if (s_tick)
                  if (s_reg==15)
                     begin
                        s_next = 0;
                        b_next = {1'b0, b_reg[7:1]};  // next bit -> b_reg[0]
                        if (n_reg==(DBIT-1))
                           state_next = stop;          // all 8 bits sent
                        else
                           n_next = n_reg + 1;
                     end
                  else
                     s_next = s_reg + 1;
            end
         // STOP: drive 1 for SB_TICK ticks, then tell the FIFO we are done
         // (tx_done_tick pops the byte) and return to idle.
         stop:
            begin
               tx_next = 1'b1;
               if (s_tick)
                  if (s_reg==(SB_TICK-1))
                     begin
                        state_next   = idle;
                        tx_done_tick = 1'b1;
                     end
                  else
                     s_next = s_reg + 1;
            end
      endcase
   end
   // ---- output --------------------------------------------------------------
   // tx comes straight from a flip-flop, not from the case logic. Logic
   // outputs can glitch briefly while they settle; a flip-flop output only
   // changes once per clock edge, so the PC sees a clean line.
   assign tx = tx_reg;

endmodule
