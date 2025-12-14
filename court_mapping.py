# court_mapping.py
import cv2
import numpy as np

def compute_homography(src_pts: np.ndarray, dst_pts: np.ndarray, ransac=True):
    method = cv2.RANSAC if ransac else 0
    H, mask = cv2.findHomography(src_pts.astype(np.float32), dst_pts.astype(np.float32), method)
    if H is None:
        raise RuntimeError("Homography failed. Check point order / quality.")
    return H, mask

def map_points_xy(H: np.ndarray, pts_xy: np.ndarray) -> np.ndarray:
    """
    pts_xy: (N,2) float32 in video pixel coords
    returns: (N,2) float32 in court pixel coords
    """
    pts = pts_xy.reshape(-1, 1, 2).astype(np.float32)
    mapped = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return mapped

def player_ground_point_xyxy(x1, y1, x2, y2):
    """Use feet point: (center_x, bottom_y)."""
    return 0.5 * (x1 + x2), float(y2)

def draw_points_on_court(court_img, pts_xy, color=(255, 0, 0), r=6, labels=None):
    out = court_img.copy()
    for i, (x, y) in enumerate(pts_xy):
        x, y = int(x), int(y)
        cv2.circle(out, (x, y), r, color, -1)
        if labels is not None:
            cv2.putText(out, str(labels[i]), (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out
