// ============================================================================
// uart.v  -  complete UART core: baud_gen + uart_rx + uart_tx + 2 FIFOs
// ----------------------------------------------------------------------------
// ROLE IN THE ECHO SYSTEM
//   This module has no logic of its own; it only WIRES the five building
//   blocks together (structural Verilog). To the outside (top.v) it looks
//   like two byte queues plus two serial pins:
//
//                     +--------------------- uart ---------------------+
//                     |              tick                              |
//                     |   baud_gen ------+----------------+            |
//                     |                  v                v            |
//   rx (serial in) ---|-------------> uart_rx         uart_tx ---------|--> tx
//                     |          dout |  | rx_done   din ^  ^ tx_done  |
//                     |               v  v (wr)           |  | (rd)    |
//                     |              RX FIFO          TX FIFO          |
//   rd_uart ----------|-----------> (rd)  r_data     w_data (wr) <-----|-- wr_uart
//   r_data  <---------|------------------+               ^-------------|-- w_data
//   rx_empty <--------|  empty                     full  --------------|--> tx_full
//                     +------------------------------------------------+
//
//   Receive path : rx -> uart_rx -> RX FIFO -> r_data (read with rd_uart)
//   Transmit path: w_data (write with wr_uart) -> TX FIFO -> uart_tx -> tx
// ============================================================================
module uart
   #(
     parameter DBIT = 8, SB_TICK = 16,
               FIFO_W = 4,           // FIFO address bits
               M = 54, N = 6         // baud_gen modulus (54 -> 115200 baud) and width
    )
   (
    input  wire        clk, reset,
    input  wire        rd_uart,   // pulse: "I took r_data, drop it from RX FIFO"
    input  wire        wr_uart,   // pulse: "store w_data in TX FIFO to be sent"
    input  wire        rx,        // serial input
    input  wire [7:0]  w_data,    // byte to transmit
    output wire        tx_full,   // TX FIFO has no room
    output wire        rx_empty,  // RX FIFO has nothing to read
    output wire        tx,        // serial output
    output wire [7:0]  r_data     // oldest received byte
   );
   // signal declaration (internal wires between the blocks)
   //   tick         : 16x baud tick shared by rx and tx
   //   tx_empty     : TX FIFO empty  -> transmitter has nothing to send
   //   rx_full      : RX FIFO full   (not used further; bytes would be dropped)
   //   tx_done_tick : transmitter finished a byte -> pop it from TX FIFO
   //   rx_done_tick : receiver finished a byte    -> push it into RX FIFO
   //   tx_data      : byte going from TX FIFO into uart_tx
   //   rx_data      : byte going from uart_rx into RX FIFO
   wire tick, tx_empty, rx_full, tx_done_tick, rx_done_tick;
   wire [7:0] tx_data, rx_data;
   // body
   // instantiate uart transmitter
   // Starts whenever the TX FIFO is not empty; sends the FIFO's oldest byte.
   uart_tx #(.DBIT(DBIT), .SB_TICK(SB_TICK))
      u_uart_tx (.clk(clk), .reset(reset), .tx_start(~tx_empty),
                 .s_tick(tick), .din(tx_data),
                 .tx_done_tick(tx_done_tick), .tx(tx));
   // instantiate uart receiver
   // Listens on rx; each finished byte appears on rx_data with rx_done_tick.
   uart_rx #(.DBIT(DBIT), .SB_TICK(SB_TICK))
      u_uart_rx (.clk(clk), .reset(reset), .rx(rx), .s_tick(tick),
                 .rx_done_tick(rx_done_tick), .dout(rx_data));
   // instantiate baud rate generator
   // One tick every M clocks; .q() is left unconnected because nobody needs
   // the raw count.
   baud_gen #(.M(M), .N(N))
      u_baud_gen (.clk(clk), .reset(reset), .tick(tick), .q());
   // instantiate fifo for transmitter
   // Written by the user side (wr_uart / w_data), read by uart_tx
   // (rd = tx_done_tick, so a byte is removed only after it was fully sent).
   fifo #(.DATA_WIDTH(DBIT), .ADDR_WIDTH(FIFO_W))
      u_fifo_tx (.clk(clk), .reset(reset), .rd(tx_done_tick),
                 .wr(wr_uart), .w_data(w_data),
                 .empty(tx_empty), .full(tx_full), .r_data(tx_data));
   // instantiate fifo for receiver
   // Written by uart_rx (wr = rx_done_tick), read by the user side (rd_uart).
   fifo #(.DATA_WIDTH(DBIT), .ADDR_WIDTH(FIFO_W))
      u_fifo_rx (.clk(clk), .reset(reset), .rd(rd_uart),
                 .wr(rx_done_tick), .w_data(rx_data),
                 .empty(rx_empty), .full(rx_full), .r_data(r_data));
endmodule
