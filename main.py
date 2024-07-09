import pyfirmata2
import time

board = pyfirmata2.Arduino('COM3')
ledPin = board.get_pin('d:13:o')

while True:
    ledPin.write(1)
    time.sleep(0.5)
    ledPin.write(0)
    time.sleep(0.5)