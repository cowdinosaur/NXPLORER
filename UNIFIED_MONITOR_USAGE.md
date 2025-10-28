# Unified Environmental Monitoring and Plant Health Scanner

## Overview

The `unified_monitor.py` program combines multiple sensor inputs and AI-powered plant health analysis into a single, automated monitoring system. It continuously measures environmental conditions and performs plant health classification at regular intervals.

## Features

- **CO₂ Sensing**: Reads CO₂ concentration, temperature, and humidity via SCD30 sensor
- **Light Monitoring**: Measures ambient light levels using LDR sensor via MCP3008 ADC
- **AI Plant Classification**: Captures and analyzes plant images for health assessment
- **Continuous Monitoring**: Runs at configurable intervals (default: 10 seconds)
- **Data Logging**: Saves all measurements and predictions to CSV format
- **Graceful Degradation**: Works even when some sensors are unavailable
- **Command-Line Interface**: Flexible configuration options

## Hardware Requirements

### Required Sensors
- **SCD30 Sensor**: CO₂, temperature, and humidity sensing
- **MCP3008 ADC**: Analog-to-digital converter
- **LDR (Light Dependent Resistor)**: Light level measurement
- **Raspberry Pi Camera** (optional): For image capture

### Connections
```
SCD30: I2C (SDA/SCL) - default address 0x61
MCP3008: SPI (MOSI/MISO/SCLK) + CE0
LDR: Connected to MCP3008 channel P0
Camera: Pi camera interface or USB camera
```

## Software Dependencies

Install required packages:

```bash
# Core dependencies
pip install tensorflow pillow numpy

# SCD30 sensor library
pip install sensirion-i2c-driver sensirion-i2c-scd30 sensirion-driver-adapters

# Raspberry Pi sensor libraries
pip install adafruit-circuitpython-mcp3xxx adafruit-circuitpython-board

# Camera support (optional)
pip install picamera2 opencv-python
```

## Usage

### Basic Usage

```bash
# Start monitoring with default settings (10-second intervals)
python3 unified_monitor.py
```

### Command-Line Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--help` | `-h` | Show help message | - |
| `--interval` | `-i` | Measurement interval in seconds | 10 |
| `--plant` | `-p` | Plant type for analysis | Tomato |
| `--i2c-port` | - | I2C port for SCD30 sensor | /dev/i2c-1 |
| `--camera` | `-c` | Enable camera capture | Disabled |
| `--cycles` | `-n` | Number of measurement cycles | Infinite |

### Examples

```bash
# Monitor every 10 seconds for tomato plants
python3 unified_monitor.py --plant Tomato

# Monitor every 30 seconds with camera capture enabled
python3 unified_monitor.py --interval 30 --camera

# Run exactly 20 measurement cycles
python3 unified_monitor.py --cycles 20

# Monitor lettuce plants every 15 seconds with camera
python3 unified_monitor.py --plant Lettuce --interval 15 --camera

# Use different I2C port
python3 unified_monitor.py --i2c-port /dev/i2c-0
```

## Output and Logging

### Console Output
The program displays real-time information for each measurement cycle:

```
============================================================
🌱 UNIFIED MONITORING CYCLE - 2025-10-28 18:50:05
🌱 Plant Type: Tomato
============================================================

[SCD30] Reading CO2/Humidity...
  ✓ CO2: 450.2 ppm
  ✓ Temperature: 24.1 °C
  ✓ Humidity: 65.3%

[LDR] Reading light level...
  ✓ Light: 2.341 V (raw: 28450)

[AI] Capturing and classifying image...
  ✓ Image: data/unified_monitor_image.jpg
  ✓ Prediction: Healthy
  ✓ Confidence: 92.45%

[LOG] ✓ Data logged to data/unified_monitoring_log.csv
```

### CSV Logging
All measurements are saved to `data/unified_monitoring_log.csv` with the following columns:

- Timestamp
- Plant_Type
- CO2_ppm
- Temperature_C
- Humidity_Percent
- Light_Raw
- Light_Volts
- AI_Prediction
- Confidence_Percent

### CSV Format Example
```csv
Timestamp,Plant_Type,CO2_ppm,Temperature_C,Humidity_Percent,Light_Raw,Light_Volts,AI_Prediction,Confidence_Percent
2025-10-28 18:50:05,Tomato,450.2,24.1,65.3,28450,2.341,Healthy,92.45%
2025-10-28 18:50:15,Tomato,452.1,24.2,65.1,28520,2.352,Healthy,91.23%
```

## Error Handling

The program is designed to handle various error conditions gracefully:

### Sensor Unavailable
```
⚠️ SCD30 libraries not available. CO2/humidity sensing disabled.
⚠️ LDR libraries not available. Light sensing disabled.
[SCD30] ❌ Not available
[LDR] ❌ Not available
```

### Read Failures
```
[SCD30] ❌ Read failed: [error details]
[LDR] ❌ Read failed: [error details]
[AI] ❌ Classification failed: [error details]
```

### Logging with Missing Data
When sensors fail to read data, the program logs "N/A" instead of failing:
```csv
2025-10-28 18:50:05,Tomato,N/A,N/A,N/A,28450,2.341,Healthy,92.45%
```

## File Structure

The program creates and manages several files:

```
NXPLORER/
├── unified_monitor.py          # Main monitoring program
├── data/
│   ├── unified_monitoring_log.csv  # Measurement log file
│   ├── unified_monitor_image.jpg   # Captured plant images
│   ├── keras_model.h5         # AI model file
│   ├── labels.txt            # Classification labels
│   └── savedmodel/           # Alternative model format
└── UNIFIED_MONITOR_USAGE.md  # This documentation
```

## Integration with Existing Components

The unified monitor integrates seamlessly with existing NXPLORER components:

### AI Model Compatibility
- Uses the same TensorFlow models as `nxplorer.py`
- Supports both SavedModel and HDF5 formats
- Compatible with existing `labels.txt` format

### Environmental Analysis
- Follows the same plant type classification system
- Uses compatible temperature/humidity ranges
- Integrates with existing optimal range definitions

### Data Format
- CSV format compatible with analysis tools
- Timestamp format matches existing logs
- Column names are descriptive and consistent

## Performance Considerations

### System Resources
- **Memory**: ~100MB RAM (TensorFlow model loading)
- **CPU**: Low usage during intervals, spikes during image classification
- **Storage**: ~1KB per measurement cycle in CSV logs

### Battery Life (Portable Use)
- Camera capture significantly increases power consumption
- Consider longer intervals (>30s) for battery-powered deployments
- Disable camera with `--no-camera` flag if not needed

### SD Card Wear
- CSV logging creates frequent small writes
- Consider external storage for long-term deployments
- Log files can be rotated manually if needed

## Troubleshooting

### Common Issues

**1. Permission Denied on I2C**
```bash
sudo usermod -a -G i2c,spi,gpio $USER
# Then reboot
```

**2. Camera Not Working**
```bash
# Enable camera in raspi-config
sudo raspi-config
# Navigate to Interface Options -> Camera -> Enable

# Test camera
libcamera-still -o test.jpg
```

**3. Model Loading Errors**
```bash
# Check model files exist
ls -la data/keras_model.h5 data/labels.txt

# Verify TensorFlow installation
python3 -c "import tensorflow as tf; print(tf.__version__)"
```

**4. SPI Not Enabled**
```bash
# Enable SPI in raspi-config
sudo raspi-config
# Navigate to Interface Options -> SPI -> Enable
```

### Debug Mode
For detailed debugging, modify the script to add more verbose logging or run individual sensor tests:

```bash
# Test SCD30 sensor
python3 scd30_python_test.py

# Test LDR sensor
python3 ldr_test.py

# Test AI classification
python3 nxplorer.py
```

## Advanced Configuration

### Custom Plant Types
Add new plant types by modifying the environmental analysis function in the code or by extending the `get_optimal_ranges()` method.

### Custom Intervals
For scientific studies, consider using shorter intervals (5 seconds) for detailed monitoring or longer intervals (60+ seconds) for long-term trend analysis.

### Integration with External Systems
The CSV output can be easily integrated with:
- Grafana for visualization
- Custom dashboards
- Alert systems
- Data analysis pipelines

## Support

For issues specific to:
- **SCD30 Sensor**: Refer to Sensirion documentation
- **Raspberry Pi Hardware**: Check Raspberry Pi official docs
- **TensorFlow Models**: Review NXPLORER model documentation
- **General Issues**: Check GitHub issues or create new ones