"""
Calibracao da lente da camara com um tabuleiro de xadrez.

1. Imprime um tabuleiro de xadrez (ex.: 10x7 quadrados -> 9x6 cantos interiores),
   cola-o numa superficie plana e mede o lado de um quadrado.
2. Ajusta CHESSBOARD_INNER_CORNERS e SQUARE_SIZE_M abaixo.
3. Corre este script. Quando o tabuleiro e detetado aparecem os cantos coloridos.
   As vistas sao guardadas automaticamente quando o tabuleiro esta parado e numa
   posicao diferente das ja guardadas; ao chegar a TARGET_VIEWS calibra sozinho.
     Espaco -> guarda a vista atual a mao
     C      -> calibra ja (precisa de MIN_VIEWS vistas) e guarda calibration_data.npz
     Q/Esc  -> sai sem calibrar
   Vai mexendo o tabuleiro devagar e parando: centro, cantos e bordas da imagem,
   inclinado, mais perto e mais longe. As bordas sao onde a distorcao e maior.
4. Depois de calibrar aparece a imagem corrigida (U alterna corrigida/original).

Na calibracao experimenta-se o modelo normal (5 coeficientes) e o racional
(8 coeficientes, melhor para lentes grande angulares) e fica o de menor erro.
Vistas com erro muito acima das outras (tremidas, mal detetadas) sao descartadas.
Os cantos detetados ficam guardados no .npz; para recalibrar sem tirar fotos:
    py -3.13 calibrate_camera.py --recalibrate

A camara e a resolucao tem de ser as mesmas do track1.py (CAMERA_INDEX, CAMERA_WIDTH/HEIGHT).
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

CAMERA_INDEX = 0  # webcam USB (iCatch V37); a do PC e o 1. A ordem pode mudar!
CAMERA_BACKEND = cv2.CAP_MSMF
CAMERA_WIDTH, CAMERA_HEIGHT = 1280, 720

CHESSBOARD_INNER_CORNERS = (9, 6)  # (colunas, linhas) de cantos interiores
SQUARE_SIZE_M = 0.025
MIN_VIEWS = 15
TARGET_VIEWS = 20  # ao chegar aqui calibra automaticamente

# Recolha automatica de vistas
AUTO_CAPTURE = True
AUTO_STILL_MAX_PX = 1.5    # movimento medio dos cantos entre frames para contar como parado
AUTO_STILL_FRAMES = 5      # frames seguidos parado antes de guardar
AUTO_MIN_NEW_PX = 40.0     # distancia media dos cantos a todas as vistas ja guardadas
AUTO_MIN_INTERVAL_S = 0.5  # tempo minimo entre vistas guardadas

# Calibracao
CALIB_MODELS = [
    ("normal (5 coef.)", 0),
    ("racional (8 coef.)", cv2.CALIB_RATIONAL_MODEL),
]
OUTLIER_FACTOR = 2.0  # descarta vistas com erro > OUTLIER_FACTOR x mediana
MODEL_MIN_GAIN = 0.9  # o modelo seguinte so e escolhido se tiver RMS < 90% do anterior

CALIBRATION_PATH = Path(__file__).with_name("calibration_data.npz")
FIND_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def save_calibration(mtx, dist, rvecs, tvecs, image_size, rms, obj_points, img_points, model):
    if CALIBRATION_PATH.exists():
        backup = CALIBRATION_PATH.with_name(f"calibration_data_{time.strftime('%Y%m%d_%H%M%S')}.npz")
        CALIBRATION_PATH.rename(backup)
        print(f"Calibracao anterior guardada em {backup.name}")

    np.savez(
        CALIBRATION_PATH,
        camera_matrix=mtx,
        dist_coeff=dist,
        rvecs=np.array(rvecs),
        tvecs=np.array(tvecs),
        image_size=np.array(image_size),
        rms=rms,
        model=model,
        obj_points=np.array(obj_points, dtype=np.float32),
        img_points=np.array(img_points, dtype=np.float32),
    )
    print(f"Calibracao guardada em {CALIBRATION_PATH.name}")


def _corner_distance(a, b):
    """Distancia media entre dois conjuntos de cantos (o tabuleiro pode ser detetado rodado 180 graus)."""
    a = a.reshape(-1, 2)
    b = b.reshape(-1, 2)
    return min(
        float(np.mean(np.linalg.norm(a - b, axis=1))),
        float(np.mean(np.linalg.norm(a[::-1] - b, axis=1))),
    )


def _is_new_view(corners, img_points):
    return all(_corner_distance(corners, saved) >= AUTO_MIN_NEW_PX for saved in img_points)


def _view_errors(obj_points, img_points, mtx, dist, rvecs, tvecs):
    errors = []
    for obj, img, rvec, tvec in zip(obj_points, img_points, rvecs, tvecs):
        proj, _ = cv2.projectPoints(obj, rvec, tvec, mtx, dist)
        errors.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img.reshape(-1, 2)) ** 2, axis=1)))))
    return np.array(errors)


def _calibrate_model(obj_points, img_points, image_size, flags):
    """Calibra, descarta as vistas com erro muito alto e recalibra sem elas."""
    obj_points, img_points = list(obj_points), list(img_points)
    rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, image_size, None, None, flags=flags)
    errors = _view_errors(obj_points, img_points, mtx, dist, rvecs, tvecs)
    limit = max(0.5, OUTLIER_FACTOR * float(np.median(errors)))
    keep = [i for i, e in enumerate(errors) if e <= limit]
    dropped = len(errors) - len(keep)
    if dropped and len(keep) >= MIN_VIEWS:
        obj_points = [obj_points[i] for i in keep]
        img_points = [img_points[i] for i in keep]
        rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, image_size, None, None, flags=flags)
    else:
        dropped = 0
    return rms, mtx, dist, rvecs, tvecs, obj_points, img_points, dropped


def calibrate(obj_points, img_points, image_size):
    best = None
    for name, flags in CALIB_MODELS:
        result = _calibrate_model(obj_points, img_points, image_size, flags)
        rms, mtx, *_, dropped = result
        print(f"  modelo {name}: RMS {rms:.3f} px, fx={mtx[0, 0]:.0f} fy={mtx[1, 1]:.0f}, "
              f"{dropped} vistas descartadas")
        # o modelo mais complexo so ganha se for claramente melhor (evita sobreajuste)
        if best is None or rms < MODEL_MIN_GAIN * best[0][0]:
            best = (result, name)

    (rms, mtx, dist, rvecs, tvecs, obj_used, img_used, _), name = best
    print(f"Escolhido: modelo {name}, {len(img_used)} vistas")
    print(f"Erro de reprojecao RMS: {rms:.3f} px (bom: < 0.5, aceitavel: < 1.0)")
    if rms > 1.0:
        print("  AVISO: erro alto. Verifica se o tabuleiro esta plano e se CHESSBOARD_INNER_CORNERS / "
              "SQUARE_SIZE_M batem certo com o tabuleiro impresso.")
    print("camera_matrix:\n", mtx)
    print("dist_coeff:", dist.ravel())
    save_calibration(mtx, dist, rvecs, tvecs, image_size, rms, obj_points, img_points, name)

    new_mtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, image_size, 0.0, image_size)
    return cv2.initUndistortRectifyMap(mtx, dist, None, new_mtx, image_size, cv2.CV_16SC2)


def recalibrate_from_file():
    """Recalibra com os cantos guardados no calibration_data.npz (sem tirar fotos)."""
    with np.load(CALIBRATION_PATH) as data:
        if "img_points" not in data.files:
            print(f"{CALIBRATION_PATH.name} nao tem os cantos guardados: tens de tirar as fotos de novo.")
            return
        obj_points = list(data["obj_points"])
        img_points = list(data["img_points"])
        image_size = tuple(int(v) for v in data["image_size"])
    print(f"A recalibrar com {len(img_points)} vistas guardadas ...")
    calibrate(obj_points, img_points, image_size)


def main():
    if "--recalibrate" in sys.argv:
        recalibrate_from_file()
        return

    cap = cv2.VideoCapture(CAMERA_INDEX, CAMERA_BACKEND)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    if not cap.isOpened():
        print("Camera error.")
        return

    cols, rows = CHESSBOARD_INNER_CORNERS
    board = np.zeros((cols * rows, 3), np.float32)
    board[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * SQUARE_SIZE_M

    obj_points, img_points = [], []
    image_size = None
    maps = None
    show_undistorted = True
    prev_corners = None
    still_frames = 0
    last_save_t = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error.")
            break

        h, w = frame.shape[:2]
        if image_size is None:
            image_size = (w, h)
            print(f"Resolucao da camara: {w}x{h}")

        if maps is not None:
            view = cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR) if show_undistorted else frame.copy()
            text = f"{'Corrigida' if show_undistorted else 'Original'} | U = alternar, Q = sair"
            cv2.putText(view, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Calibracao", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("u"), ord("U")):
                show_undistorted = not show_undistorted
            elif key in (ord("q"), ord("Q"), 27):
                break
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_INNER_CORNERS, FIND_FLAGS)

        # parado = cantos quase no mesmo sitio durante AUTO_STILL_FRAMES frames seguidos
        if found and prev_corners is not None and _corner_distance(corners, prev_corners) <= AUTO_STILL_MAX_PX:
            still_frames += 1
        else:
            still_frames = 0
        prev_corners = corners if found else None

        new_view = found and _is_new_view(corners, img_points)
        auto_save = (
            AUTO_CAPTURE
            and new_view
            and still_frames >= AUTO_STILL_FRAMES
            and time.time() - last_save_t >= AUTO_MIN_INTERVAL_S
        )

        view = frame.copy()
        for pts in img_points:
            cv2.polylines(view, [pts.reshape(-1, 2)[[0, cols - 1, -1, -cols]].astype(np.int32)], True, (255, 0, 0), 1)
        if found:
            cv2.drawChessboardCorners(view, CHESSBOARD_INNER_CORNERS, corners, found)
        if found and not new_view:
            status = "posicao repetida - muda o tabuleiro"
        elif found:
            status = "segura parado..."
        else:
            status = "tabuleiro nao detetado"
        text = f"Vistas: {len(img_points)}/{TARGET_VIEWS} | {status}"
        cv2.putText(view, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(view, "Espaco = guardar a mao, C = calibrar ja, Q = sair", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        if auto_save:
            cv2.rectangle(view, (0, 0), (w - 1, h - 1), (0, 255, 0), 8)  # flash verde ao guardar
        cv2.imshow("Calibracao", view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

        if auto_save or key == ord(" "):
            if not found:
                print("Tabuleiro nao detetado nesta vista.")
                continue
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX_CRITERIA)
            obj_points.append(board.copy())
            img_points.append(corners)
            last_save_t = time.time()
            still_frames = 0
            print(f"Vista {len(img_points)}/{TARGET_VIEWS} guardada{' (auto)' if auto_save else ''}.")

        if len(img_points) >= TARGET_VIEWS or key in (ord("c"), ord("C")):
            if len(img_points) < MIN_VIEWS:
                print(f"Precisas de pelo menos {MIN_VIEWS} vistas (tens {len(img_points)}).")
                continue
            print(f"A calibrar com {len(img_points)} vistas ...")
            maps = calibrate(obj_points, img_points, image_size)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
