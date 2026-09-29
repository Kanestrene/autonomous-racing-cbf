import socket
import time

from car.utils.hex_functions import hex_to_int, mirror_hex
from car.utils.aux_functions import normalize_to_range


class wltoys6401:
    
    def __init__(self):
        self.car_IP = "172.16.11.1"
        self.handshake_port = 23459
        self.control_port = 23458

        self.tx_frequency_Hz = 20  # 20 Hz

        self.sync_msg_1 = bytes.fromhex("a88a210006000000010000000000")
        self.sync_msg_2 = bytes.fromhex("a88a200008000000010002000000d204")
        
        self.base_msg = bytearray.fromhex("ca47d500000000006680808000008099")
        self.heartbeat_msg = bytearray.fromhex("ca47d500000000006680808000008099")
        self.control_msg = None

        self.max_steering_HEX = "0xFF"
        self.min_steering_HEX = mirror_hex(mirror_value=self.max_steering_HEX)

        self.max_throttle_HEX = "0xA2"
        self.min_throttle_HEX = mirror_hex(mirror_value=self.max_throttle_HEX)

    def send_message(self, message=None) -> None:
        if message is not None:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.sendto(message, (self.car_IP, self.control_port))
            s.close()

    def send_heartbeat(self) -> None:
        self.send_message(message=self.heartbeat_msg)

    def move(self, throttle_norm=None, steering_norm=None) -> None:
        if throttle_norm is None and steering_norm is None:
            raise ValueError("At least one of throttle_norm or steering_norm must be provided.")

        if steering_norm is not None:
            assert -1.0 <= steering_norm <= 1.0, "steering_norm must be in range [-1, 1]"

        if throttle_norm is not None:
            assert -1.0 <= throttle_norm <= 1.0, "throttle_norm must be in range [-1, 1]"

        self.control_msg = self.base_msg.copy()

        steering_byte = 0x80
        throttle_byte = 0x80

        if steering_norm is not None:
            s_min = hex_to_int(self.min_steering_HEX)
            s_max = hex_to_int(self.max_steering_HEX)
            steering_raw = normalize_to_range(steering_norm, s_min, s_max)
            steering_byte = steering_raw & 0xFF
            self.control_msg[9] = steering_byte

        if throttle_norm is not None:
            t_min = hex_to_int(self.min_throttle_HEX)
            t_max = hex_to_int(self.max_throttle_HEX)
            throttle_raw = normalize_to_range(throttle_norm, t_min, t_max)
            throttle_byte = throttle_raw & 0xFF
            self.control_msg[10] = throttle_byte

        self.control_msg[14] = ((steering_byte ^ throttle_byte) + 0x80) & 0xFF

        self.send_message(self.control_msg)