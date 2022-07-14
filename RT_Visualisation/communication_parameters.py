from BrainStudy.signal_parameters import DECIMATION_FACTOR

UDP_ADDRESS_VISUALISATION = "localhost"
UDP_PORT_VISUALISATION = 7550

UDP_ADDRESS_SIMULATOR = "localhost"
UDP_PORT_SIMULATOR = 7552

channels = 16
samples = 128
samples_in_message = channels*samples / DECIMATION_FACTOR

fs = 2048
