import time
import board
import busio
import digitalio

# MCP3008 driver
import adafruit_mcp3xxx.mcp3008 as MCP
from adafruit_mcp3xxx.analog_in import AnalogIn

# --- SPI + MCP3008 (SPI0, CE0) ---
spi = busio.SPI(clock=board.SCLK, MISO=board.MISO, MOSI=board.MOSI)
cs = digitalio.DigitalInOut(board.CE0)  # or board.D8
mcp = MCP.MCP3008(spi, cs)

# LDR on channel 0
ldr = AnalogIn(mcp, MCP.P0)



def read_ldr():
    """
    Returns:
      raw   : 0..65535 (scaled by library)
      volts : measured channel voltage (0..Vref)
      r_ldr : estimated LDR resistance using divider maths
    Divider: 3.3V -- LDR --(Vout @ CH0)-- R_FIXED -- GND
    r_ldr = R_FIXED * (Vcc - Vout) / Vout
    """
    vout = ldr.voltage       # volts
    vcc = 3.3
    if vout <= 0.0001:
        return ldr.value, vout, None
    r_ldr = 0
    return ldr.value, vout, r_ldr

if __name__ == "__main__":
    while True:
        raw, volts, r_ldr = read_ldr()

        parts = []

        # LDR output -- use as a relative light measure
        if r_ldr is not None:
            parts.append(f"LDR: {volts:.3f} V (raw {raw}), ~{r_ldr:,.0f} Ω")
        else:
            parts.append(f"LDR: {volts:.3f} V (raw {raw})")

        print(" | ".join(parts))
        time.sleep(2)  # DHT11 needs ~2 s between reads
