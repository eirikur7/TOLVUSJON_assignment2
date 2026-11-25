import cv2
import csv
import os
import time
from helpers import line_detectors as det
from helpers import image_manipulation as manip

# ----- CONSTANTS ----- #
CAMERA_INDEX = 0  # laptop camera
# CAMERA_INDEX = "http://192.168.2.121:4747/video"  # phone (DroidCam)

P_SUCCESS = 0.95          # p in the formula for the needed number of iterations N

# Settings for each run. The run name is also the name of the results CSV.
#   scale:      image size, 1.0 = full size, 0.5 = half width and height
#   canny:      Canny lower and upper threshold
#   iterations: number of random point pairs tried per frame
#   threshold:  max distance in pixels for a point to count as an inlier
#   step:       count support on every k-th edge point only (1 = all points)
RUNS = {
    #                   scale, canny_low, canny_high, iterations, threshold, step
    "01_baseline":     (1.0,   100,       200,        100,        2.0,       1),
    "02_half_size":    (0.5,   100,       200,        100,        2.0,       1),
    "03_high_canny":   (1.0,   150,       300,        100,        2.0,       1),
    "04_every_10th":   (1.0,   100,       200,        100,        2.0,       10),
    "05_tuned_01":     (1.0,   200,       400,        300,        1.5,       5),
    "05_tuned_02":     (1.0,   200,       400,        500,        1.5,       5),
    "06_tracking":     (1.0,   200,       400,        500,        1.5,       5),
    "06_tracking_02":  (1.0,   150,       300,        300,        1.5,       5),
    "06_tracking_03":  (1.0,   150,       300,        300,        1.5,       5),
}

RUN_NAME = "06_tracking_02"
SCALE, CANNY_LOW, CANNY_HIGH, RANSAC_ITERATIONS, RANSAC_THRESHOLD, SUPPORT_STEP = RUNS[RUN_NAME]


# ----- GLOBALS ----- #
t_prev = None  # time of the previous get_fps() call ; used in fps calculation

def get_fps():
    global t_prev

    t_now = time.perf_counter()

    # Get delta t
    dt = 1 if t_prev is None else t_now - t_prev

    t_prev = t_now

    return 1/dt

def get_frame(camera):
    # Read one frame and time how long the capture took
    t_start = time.perf_counter()
    ret, frame = camera.read()  # ret[bool] If frame read; frame[np array]
    t_capture = time.perf_counter() - t_start
    return ret, frame, t_capture

def process_frame(frame, t_capture):

    # Shrink the image (speed-up 1)
    frame = manip.resize(frame, SCALE)

    # Canny edges
    t_start = time.perf_counter()
    frame_gray = manip.convert_to(frame, "gray")
    edges = cv2.Canny(frame_gray, CANNY_LOW, CANNY_HIGH)
    points = det.edge_points(edges)
    t_canny = time.perf_counter() - t_start

    # RANSAC line
    t_start = time.perf_counter()
    ransac_line, inliers = det.ransac_line(points, RANSAC_ITERATIONS, RANSAC_THRESHOLD, SUPPORT_STEP)
    t_ransac = time.perf_counter() - t_start

    # Refine with least squares and draw both: RANSAC = red, least squares = green
    n_points = len(points)
    n_inliers = 0
    ls_line = (0, 0, 0)
    angle_diff = 0
    if ransac_line is not None:
        n_inliers = int(inliers.sum())
        ls_line = det.least_squares_line(points[inliers])
        angle_diff = det.angle_between(ransac_line, ls_line)
        manip.draw_line(frame, ransac_line, color=(0, 0, 255))
        manip.draw_line(frame, ls_line, color=(0, 255, 0))
    else:
        ransac_line = (0, 0, 0)

    # Inlier ratio and the number of iterations the formula asks for
    inlier_ratio = n_inliers / n_points if n_points > 0 else 0
    n_needed = det.iterations_needed(P_SUCCESS, 1 - inlier_ratio)

    # FPS and info text
    fps = get_fps()
    manip.draw_text(frame, f"FPS: {fps:.1f}", 0)
    manip.draw_text(frame, f"Edge points: {n_points}  Inliers: {inlier_ratio:.1%}", 1)

    # One row of measurements for this frame (times in ms)
    measurement = [fps,
                   t_capture * 1000, t_canny * 1000, t_ransac * 1000,
                   n_points, n_inliers, inlier_ratio, n_needed,
                   *ransac_line, *ls_line, angle_diff]

    return frame, measurement

def save_data(filename, data):
    path = "results"
    os.makedirs(path, exist_ok=True)

    # One row per frame: frame number + the settings of this run + the measurement from process_frame
    settings = [SCALE, CANNY_LOW, CANNY_HIGH, RANSAC_ITERATIONS, RANSAC_THRESHOLD, SUPPORT_STEP]
    with open(os.path.join(path, filename), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame",
                         "scale", "canny_low", "canny_high", "iterations", "threshold", "step",
                         "fps", "t_capture_ms", "t_canny_ms", "t_ransac_ms",
                         "n_points", "n_inliers", "inlier_ratio", "n_needed",
                         "ransac_a", "ransac_b", "ransac_c",
                         "ls_a", "ls_b", "ls_c", "angle_diff_deg"])
        for i, measurement in enumerate(data):
            writer.writerow([i] + settings + measurement)


def main():

    # Start capture on camera X
    camera = cv2.VideoCapture(CAMERA_INDEX)

    # Store measurements, one row per frame
    measurements = []

    while True:
        # Get next frame
        ret, frame_bgr, t_capture = get_frame(camera)
        if not ret: continue

        # Process and show the new frame
        frame, measurement = process_frame(frame_bgr, t_capture)
        cv2.imshow('frame', frame)
        measurements.append(measurement)

        # Check exit command
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    # Clean exit
    camera.release()
    cv2.destroyAllWindows()

    # Save results
    save_data(f"{RUN_NAME}.csv", measurements)


if __name__ == "__main__":
    main()
