import sys
from pathlib import Path
import time

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from car.wltoys6401 import wltoys6401


class CarController:
    def __init__(self, v_max=1.0, delta_max=2.0, tx_rate=20):
        self.car = wltoys6401()

        self.v_max = v_max
        self.delta_max = delta_max
        self.dt = 1.0 / tx_rate

    def _clamp(self, x, lo=-1.0, hi=1.0):
        return max(lo, min(hi, x))

    def v_to_throttle(self, v):
        if self.v_max <= 0:
            return 0.0

        throttle = v / self.v_max

        if abs(throttle) < 0.05:
            return 0.0

        return self._clamp(throttle)

    def delta_to_steering(self, delta):
        if self.delta_max <= 0:
            return 0.0

        steering = -delta / self.delta_max

        if abs(steering) < 0.02:
            return 0.0

        return self._clamp(steering)

    def send_heartbeat(self):
        self.car.send_heartbeat()

    def send_cmd(self, v=None, delta=None):
        throttle = None
        steering = None
       
        if v is not None:
            throttle = self.v_to_throttle(v)

        if delta is not None:
            steering = self.delta_to_steering(delta)

        #self.car.send_heartbeat()
        '''
        self.car.move(
            throttle_norm=throttle,
            steering_norm=steering
        )
        '''
        #self.car.send_heartbeat()
        #time.sleep(self.dt)        
        
        self.car.move(
            throttle_norm=throttle,
            steering_norm=steering
        )
            
        #time.sleep(0.2)
        
       
       

    def stop(self):
        for _ in range(5):
            self.car.move(throttle_norm=0.0, steering_norm=0.0)
            time.sleep(self.dt)
