"""PC-side UART test for the Basys 3 (115200 8N1).

pip install pyserial
python scripts/uart_test.py --list
python scripts/uart_test.py COM5 --text "Hello FPGA"
"""
import argparse
import sys
import time

import serial
import serial.tools.list_ports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", nargs="?", help="e.g. COM5 (see --list)")
    ap.add_argument("--list", action="store_true", help="list serial ports")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--text", default="Hello FPGA")
    ap.add_argument("--timeout", type=float, default=1.0)
    a = ap.parse_args()

    if a.list or not a.port:
        for p in serial.tools.list_ports.comports():
            print(p.device, "-", p.description)
        return 0

    with serial.Serial(a.port, a.baud, bytesize=8, parity="N", stopbits=1,
                       timeout=a.timeout) as s:
        s.reset_input_buffer()
        tx = a.text.encode("ascii")
        s.write(tx)
        s.flush()
        time.sleep(0.1)
        rx = s.read(len(tx))
        print(f"sent: {tx!r}\nrecv: {rx!r}")
        # Passes if the design echoes; adapt for other designs.
        return 0 if rx == tx else 1


if __name__ == "__main__":
    sys.exit(main())
