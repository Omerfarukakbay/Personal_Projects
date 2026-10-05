// ============================================================================
// top.v  -  board top level for the Basys 3: UART echo
// ----------------------------------------------------------------------------
// ROLE IN THE ECHO SYSTEM
//   This is the outermost module. Its ports are real FPGA pins (the pin
//   numbers are in constraints/basys3.xdc). It:
//     1. cleans up the incoming serial line (2-flip-flop synchronizer),
//     2. feeds it to the uart core,
//     3. copies every received byte straight back into the transmit side
//        (that is the "echo"),
//     4. shows the last echoed byte and two status flags on the LEDs.
//
//   PC --RsRx--> sync --> uart (RX FIFO) --move--> uart (TX FIFO) --RsTx--> PC
//                                             \--> last_byte --> led[7:0]
// ============================================================================
module top
   (
    input  wire        clk,        // 100 MHz, W5
    input  wire        btnC,       // reset (active high)
    input  wire [15:0] sw,         // switches (not used yet)
    input  wire        RsRx,       // USB-UART: PC -> FPGA (B18)
    output wire        RsTx,       // USB-UART: FPGA -> PC (A18)
    output wire [15:0] led
   );
   // signal declaration
   reg        rx_sync1, rx_sync2;  // 2-FF synchronizer for the asynchronous RsRx pin
   wire       rx_empty, tx_full;   // status of the RX and TX FIFOs
   wire [7:0] rx_byte;         // oldest byte in the rx FIFO
   wire       move;                // move one byte from the rx FIFO to the tx FIFO
   reg  [7:0] last_byte;           // copy of the last echoed byte, for the LEDs
   // body
   // ---- input synchronizer (sequential) -------------------------------------
   // RsRx comes from the PC and changes at any moment, not in step with our
   // clock. Sampling it right as it changes can leave a flip-flop briefly
   // undecided (metastable). Passing it through two flip-flops gives that
   // first flop a full clock to settle, so rx_sync2 is safe to use.
   // Reset sets both to 1, the idle level of a UART line.
   always @(posedge clk, posedge btnC)
      if (btnC)
         begin
            rx_sync1 <= 1'b1;
            rx_sync2 <= 1'b1;
         end
      else
         begin
            rx_sync1 <= RsRx;
            rx_sync2 <= rx_sync1;
         end
   // ---- echo control (combinational) ----------------------------------------
   // only pop the rx FIFO when the tx FIFO can take the byte
   // When move = 1 for one clock, the same signal both reads (rd_uart) the
   // RX FIFO and writes (wr_uart) that byte into the TX FIFO, because
   // r_data of the RX FIFO is wired straight into w_data of the TX FIFO.
   assign move = ~rx_empty & ~tx_full;
   // ---- the UART core ---------------------------------------------------------
   // M = 54 gives 16 ticks per bit at 115200 baud with a 100 MHz clock.
   // FIFO_W = 4 gives 16-byte FIFOs.
   uart #(.DBIT(8), .SB_TICK(16), .FIFO_W(4), .M(54), .N(6))
      u_uart (.clk(clk), .reset(btnC),
              .rd_uart(move), .wr_uart(move), .rx(rx_sync2),
              .w_data(rx_byte),      // echo: what we read is what we write
              .tx_full(tx_full), .rx_empty(rx_empty), .tx(RsTx),
              .r_data(rx_byte));
   // ---- LED register (sequential) -------------------------------------------
   // remember the last byte that was echoed
   always @(posedge clk, posedge btnC)
      if (btnC)
         last_byte <= 8'h00;
      else if (move)
         last_byte <= rx_byte;
   // ---- LED outputs -----------------------------------------------------------
   assign led[7:0]  = last_byte;   // binary value of the last echoed byte
   assign led[8]    = rx_empty;    // on = nothing waiting in the RX FIFO
   assign led[9]    = tx_full;     // on = TX FIFO full (should almost never light)
   assign led[15:10] = 6'b0;       // unused LEDs off
endmodule
