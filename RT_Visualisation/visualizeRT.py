import socket
import struct
import numpy as np
from communication_parameters import *
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
import sys
from random import randint
from scipy import signal


WINDOW_WIDTH = 3440
WINDOW_HEIGHT = 1440
class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.buffor = np.zeros((16, 1000))

        self.x = []
        self.y = []
        self.freq = []
        self.psds = []
        self.seq = 0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((UDP_ADDRESS_VISUALISATION, UDP_PORT_VISUALISATION))

        self.view = pg.GraphicsLayoutWidget()
        self.plots = []
        self.plots_PSD = []
        self.data_lines = []
        self.data_lines_PSD = []
        pen = pg.mkPen(color=(105, 105, 105))
        for i in range(channels):
            self.x.append([])
            self.y.append([])
            self.freq.append([])
            self.psds.append([])
            self.plots.append(self.view.addPlot(row=i, col=0))
            self.data_lines.append(self.plots[i].plot(self.x[i], self.y[i], pen=pen))
            self.plots_PSD.append(self.view.addPlot(row=i, col=1))
            self.data_lines_PSD.append(self.plots_PSD[i].plot(self.freq[i], self.psds[i], pen=pen))
            self.plots_PSD[i].setXRange(0, 45)
        self.setCentralWidget(self.view)

        self.view.setBackground('w')


        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot_data)
        self.timer.start()


    def update_plot_data(self):
        raw_data, addr = self.sock.recvfrom(8192)
        self.buffor = np.roll(self.buffor, samples)
        data = struct.unpack(str(samples_in_message) + "f", raw_data)
        data = np.array(data).reshape(-1, samples)
        self.buffor[:, -samples:] = data
        for i in range(channels):
            if self.seq > 100:
                self.x[i] = self.x[i][samples:]
                self.y[i] = self.y[i][samples:]
            self.freq[i], self.psds[i] = signal.welch(self.buffor[i], average='mean', fs=2048, nperseg = 200, nfft=10000) #nperseg hardcoded
            self.data_lines_PSD[i].setData(self.freq[i], self.psds[i])
            self.x[i].extend(range(self.seq*samples, self.seq*samples + samples))

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