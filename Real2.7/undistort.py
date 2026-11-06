"""
Correcao da distorcao da lente, comum ao track1.py, cam.py e cam2.py.

A calibracao vem de ../Real1/calibration_data.npz (gerado pelo Real1/calibrate_camera.py).
Os mapas de correcao sao calculados uma vez por resolucao e aplicados a cada
frame com cv2.remap (rapido). Se a calibracao nao existir, ou for antiga (sem
image_size), os frames passam sem correcao.

Depois de ativar/recalibrar, os pontos de referencia em pixeis (REF_PX_ORIGIN /
REF_PX_X_AXIS) tem de ser medidos de novo na imagem corrigida.
"""
from pathlib import Path

import cv2
import numpy as np

# Usa a mesma calibracao do Real1 (mesma camara e resolucao)
CALIBRATION_PATH = Path(__file__).resolve().parent.parent / "Real1" / "calibration_data.npz"
UNDISTORT_ENABLED = True
# 0 = so pixeis validos (sem bordas pretas, perde um pouco dos cantos)
# 1 = mantem a imagem toda (aparecem bordas pretas curvas)
UNDISTORT_ALPHA = 0.0

_calib = None  # (camera_matrix, dist_coeff, (w, h)) ou False se nao houver
_maps = {}     # (w, h) -> (map1, map2) ou None


def load_calibration():
    global _calib

    if _calib is None:
        _calib = False
        if not CALIBRATION_PATH.exists():
            print(f"[undistort] {CALIBRATION_PATH.name} nao encontrado: sem correcao da lente.")
            return _calib

        with np.load(CALIBRATION_PATH) as data:
            if "image_size" not in data.files:
                print(
                    f"[undistort] {CALIBRATION_PATH.name} e antigo (sem image_size): "
                    "corre o calibrate_camera.py. Sem correcao da lente."
                )
                return _calib
            _calib = (
                np.array(data["camera_matrix"], dtype=float),
                np.array(data["dist_coeff"], dtype=float),
                tuple(int(v) for v in data["image_size"]),
            )
        print(f"[undistort] Calibracao carregada ({_calib[2][0]}x{_calib[2][1]}).")

    return _calib


def _get_maps(width, height):
    key = (width, height)
    if key in _maps:
        return _maps[key]

    mtx, dist, (w0, h0) = _calib
    if abs(width / height - w0 / h0) > 0.01:
        print(
            f"[undistort] Frame {width}x{height} com proporcao diferente da calibracao "
            f"({w0}x{h0}): sem correcao. Recalibra com esta resolucao."
        )
        _maps[key] = None
        return None

    # a matriz da camara escala linearmente com a resolucao (mesma proporcao)
    mtx = mtx.copy()
    mtx[0, :] *= width / w0
    mtx[1, :] *= height / h0

    new_mtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, key, UNDISTORT_ALPHA, key)
    _maps[key] = cv2.initUndistortRectifyMap(mtx, dist, None, new_mtx, key, cv2.CV_16SC2)
    return _maps[key]


def undistort_frame(frame):
    if not UNDISTORT_ENABLED or frame is None or not load_calibration():
        return frame

    height, width = frame.shape[:2]
    maps = _get_maps(width, height)
    if maps is None:
        return frame

    return cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR)
