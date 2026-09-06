"""
FireSense - serial link diagnostic. Run this FIRST when the node is on USB.

    python read_serial.py              # auto-detect port and baud
    python read_serial.py COM5 9600    # or name them

It lists the ports it can see, connects, and prints the parsed readings so you
can confirm the link before opening the dashboard.

If it cannot connect, the usual reason is that the Arduino IDE's Serial Monitor
is still open - only one program may hold a COM port at a time.
"""
import sys
import time

import ingest


def main():
    ports = ingest.SerialSource.list_ports()
    if not ports:
        print("No serial ports found.")
        print("  - is the board plugged in?")
        print("  - is pyserial installed?   pip install pyserial")
        return
    print("Ports visible to this machine:")
    for dev, desc in ports:
        print(f"   {dev}   {desc}")

    port = sys.argv[1] if len(sys.argv) > 1 else ports[0][0]
    baud = sys.argv[2] if len(sys.argv) > 2 else "auto"
    print(f"\nconnecting to {port} (baud={baud}) ...")

    src = ingest.SerialSource(port, baud=baud)
    try:
        src.start()
    except Exception as exc:
        print(f"\nFAILED: {exc}")
        return
    print(f"connected at {src.baud_used} baud. Reading for 20 s - "
          f"hold a flame near the sensor to test that channel.\n")

    seen, last = 0, None
    end = time.time() + 20
    while time.time() < end:
        r, _ = src.read()
        if r and r is not last:
            last, seen = r, seen + 1
            print(f"  gas={r['gas_ppm']:7.1f}  T={r['temp_c']:6.2f}C  "
                  f"RH={r['humidity_pct']:6.2f}%  P={r['pressure_hpa']:8.2f}  "
                  f"flame={'YES' if r['flame_detected'] else 'no'}")
        time.sleep(0.3)
    src.stop()

    print(f"\n{seen} readings parsed "
          f"({src.parser.skipped} non-reading lines ignored).")
    if seen:
        print("Link is good. Open the dashboard and choose Serial (USB).")
    else:
        print("Connected but nothing parsed - tell Claude what the Serial "
              "Monitor prints and the parser can be adjusted.")


if __name__ == "__main__":
    main()
