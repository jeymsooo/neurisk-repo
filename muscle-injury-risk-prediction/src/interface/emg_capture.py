import time
import socket

try:
    import serial
except ImportError:
    serial = None

def capture_emg_serial(port, baudrate, duration):
    if serial is None:
        raise ImportError("pyserial is not installed.")
    ser = serial.Serial(port, baudrate, timeout=1)
    data = []
    start_time = time.time()
    while time.time() - start_time < duration:
        line = ser.readline().decode('utf-8').strip()
        if line:
            try:
                value = float(line)
                data.append(value)
            except ValueError:
                continue
    ser.close()
    return data

def capture_emg_tcp(ip, port, duration):
    data = []
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((ip, port))
    s.settimeout(1)
    start_time = time.time()
    while time.time() - start_time < duration:
        try:
            line = s.recv(32).decode('utf-8').strip()
            if line:
                try:
                    value = float(line)
                    data.append(value)
                except ValueError:
                    continue
        except socket.timeout:
            continue
    s.close()
    return data

# Optionally, you can add Bluetooth support here using bleak or pybluez