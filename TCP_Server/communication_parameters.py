from BrainStudy.signal_parameters import DECIMATION_FACTOR

channels = 17  # field "Channels sent by TCP" in Actiview
samples = 128  # field "TCP samples/channel" in Actiview
words = channels * samples

SAMPLES_DECIMATED = samples // DECIMATION_FACTOR

TCP_LOCAL_PORT = 7778

TCP_AV_PORT = 8888 # port configured in Activeview
TCP_AV_ADDRESS = 'localhost'  # IP adress of Actiview host

UDP_PORT = 5502
UDP_IP_ADDRESS = 'localhost'

REMOTE_UDP_PORT = 5500
REMOTE_UDP_ADDRESS = "localhost"