// ============================================================================
// fifo.v  -  synchronous FIFO buffer (Chu-style: register file + pointers)
// ----------------------------------------------------------------------------
// ROLE IN THE ECHO SYSTEM
//   FIFO = First In, First Out: a small queue. uart.v uses two of them:
//     * RX FIFO: uart_rx writes each received byte in; top.v reads it out.
//     * TX FIFO: top.v writes bytes in; uart_tx reads them out to send.
//   They decouple the parts: the receiver can deliver a byte even if the
//   transmitter is still busy with the previous one, and nothing is lost as
//   long as fewer than 2**ADDR_WIDTH (= 16) bytes pile up.
//
// HOW IT WORKS (a circular buffer)
//   fifo_reg   : 16 storage slots
//   w_ptr      : index of the slot the NEXT write will use
//   r_ptr      : index of the slot holding the OLDEST unread word
//   fifo_count : how many words are stored right now (0..16)
//   The pointers are 4 bits, so after 15 they wrap to 0 by themselves.
//
//      slot:  0   1   2   3   4  ...  15
//                 ^r_ptr      ^w_ptr        count = 3 (slots 1,2,3 hold data)
//
// INTERFACE RULES
//   * wr = 1 for one clock: store w_data (ignored if full).
//   * r_data ALWAYS shows the oldest word (no clock needed to see it).
//     rd = 1 for one clock means "I have taken it, drop it" (ignored if empty).
// ============================================================================
module fifo
   #(
     parameter DATA_WIDTH = 8,
               ADDR_WIDTH = 4    // depth = 2**ADDR_WIDTH, stored word number
    )
   (
    input  wire                  clk, reset,
    input  wire                  rd, wr,   // remove oldest word / add w_data
    input  wire [DATA_WIDTH-1:0] w_data,   // word to write
    output wire                  empty, full,
    output wire [DATA_WIDTH-1:0] r_data    // oldest word in the queue
   );
   // signal declaration
   reg [DATA_WIDTH-1:0] fifo_reg [2**ADDR_WIDTH-1:0];  // fifo registers
   reg [ADDR_WIDTH-1:0] w_ptr_reg, w_ptr_next;  // write pointer
   reg [ADDR_WIDTH-1:0] r_ptr_reg, r_ptr_next;  // read pointer
   // One bit wider than the pointers so it can hold the value 16 ("full").
   reg [ADDR_WIDTH:0] fifo_count_reg, fifo_count_next;  // fifo counter
   // body
   // ---- storage: write port (sequential) ------------------------------------
   // On a clock edge with wr=1 and space left, put w_data into the slot the
   // write pointer points to. The pointer itself is moved further below.
   always @(posedge clk)
      if (wr && ~full)
         fifo_reg[w_ptr_reg] <= w_data;
   // ---- storage: read port (combinational) ----------------------------------
   // The oldest word is always visible on r_data.
   assign r_data = fifo_reg[r_ptr_reg];
   // ---- control registers (sequential) --------------------------------------
   // Same pattern as the UART modules: copy *_next into *_reg every clock.
   // Reset empties the FIFO (both pointers 0, count 0).
   always @(posedge clk, posedge reset)
      if (reset)
         begin
            w_ptr_reg      <= 0;
            r_ptr_reg      <= 0;
            fifo_count_reg <= 0;
         end
      else
         begin
            w_ptr_reg      <= w_ptr_next;
            r_ptr_reg      <= r_ptr_next;
            fifo_count_reg <= fifo_count_next;
         end
   // ---- next-state logic of write pointer, read pointer and fifo counter ---
   // {wr, rd} is a 2-bit value that says which operations are requested this
   // clock; each case moves the pointers and the count accordingly.
   always @*
   begin
      // default values
      w_ptr_next      = w_ptr_reg;
      r_ptr_next      = r_ptr_reg;
      fifo_count_next = fifo_count_reg;
      case ({wr, rd})
         2'b10: // write only
            if (~full)
               begin
                  w_ptr_next      = w_ptr_reg + 1;
                  fifo_count_next = fifo_count_reg + 1;
               end
         2'b01: // read only
            if (~empty)
               begin
                  r_ptr_next      = r_ptr_reg + 1;
                  fifo_count_next = fifo_count_reg - 1;
               end
         2'b11: // read and write together
            if (empty)  // nothing to read, so only the write happens
               begin
                  w_ptr_next      = w_ptr_reg + 1;
                  fifo_count_next = fifo_count_reg + 1;
               end
            else if (full)  // write is rejected, so only the read happens
               begin
                  r_ptr_next      = r_ptr_reg + 1;
                  fifo_count_next = fifo_count_reg - 1;
               end
            else  // both advance, count stays the same
               begin
                  w_ptr_next = w_ptr_reg + 1;
                  r_ptr_next = r_ptr_reg + 1;
               end
         default: ; // no op
      endcase
   end
   // ---- status outputs ------------------------------------------------------
   // empty: nothing to read (uart_tx waits / top.v does not move a byte)
   // full : no room (writes are ignored; top.v waits before moving a byte)
   assign empty = (fifo_count_reg==0);
   assign full  = (fifo_count_reg==2**ADDR_WIDTH);
endmodule
