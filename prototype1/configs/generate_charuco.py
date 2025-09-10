import cv2
import numpy as np
from pathlib import Path
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="docs/charuco_5x7_5x5_1000.png")
    ap.add_argument("--squares_x", type=int, default=5)
    ap.add_argument("--squares_y", type=int, default=7)
    ap.add_argument("--square_px", type=int, default=120)  # pixels per white+black square
    ap.add_argument("--marker_px", type=int, default=88)   # pixels for the black marker inside the square
    ap.add_argument("--dict", type=str, default="DICT_5X5_1000")
    args = ap.parse_args()

    aruco = cv2.aruco
    dict_enum = getattr(aruco, args.dict)
    dictionary = aruco.getPredefinedDictionary(dict_enum)

    # Create a CharucoBoard object (units here don't matter for rendering)
    board = aruco.CharucoBoard((args.squares_x, args.squares_y), args.square_px, args.marker_px, dictionary)

    # Render at a high DPI canvas
    margin = args.square_px // 2
    img = board.generateImage((args.squares_x * args.square_px + 2 * margin,
                               args.squares_y * args.square_px + 2 * margin))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    cv2.imwrite(args.out, img)
    print(f"[✓] Saved Charuco board → {args.out}")
    print(f"Spec: {args.squares_x}x{args.squares_y}, dict={args.dict}")

if __name__ == "__main__":
    main()
