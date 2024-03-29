import socket
import struct
import numpy as np
from communication_parameters import *
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
import sys
from random import randint

WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080
PLOT_SLOWING_FACTOR = 200 # im większy tym wolniej
class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.x = []
        self.y = []
        self.seq = 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((UDP_ADDRESS_VISUALISATION, UDP_PORT_VISUALISATION))

        self.view = pg.GraphicsLayoutWidget()
        self.plots = []
        self.data_lines = []
        pen = pg.mkPen(color=(105, 105, 105))
        for i in range(channels):
            self.x.append([])
            self.y.append([])
            self.plots.append(self.view.addPlot(row=i, col=0))
            self.data_lines.append(self.plots[i].plot(self.x[i], self.y[i], pen=pen))

        self.setCentralWidget(self.view)

        self.view.setBackground('w')


        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot_data)
        self.timer.start()


    def update_plot_data(self):
        raw_data, addr = self.sock.recvfrom(8192)
        data = struct.unpack(str(samples_in_message) + "f", raw_data)
        data = np.array(data).reshape(-1, samples)
        for i in range(channels):
            if self.seq > 100:
                self.x[i] = self.x[i][samples:]
                self.y[i] = self.y[i][samples:]
            self.x[i].extend(range(self.seq*samples, self.seq*samples + samples))
            self.x[i][-samples:] = [element/PLOT_SLOWING_FACTOR for element in self.x[i][-samples:]] #spowolnienie wykresu
            self.y[i].extend(data[i])
            self.data_lines[i].setData(self.x[i], self.y[i])
        self.seq = self.seq + 1

def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.setWindowTitle("Wizualizacja sygnału EEG.")
    main.resize(WINDOW_WIDTH, WINDOW_HEIGHT)
    main.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()