"""
Functions that change or draw on an image.
"""

import cv2


# OpenCV conversion codes, from a BGR frame to the named format
COLOR_CONVERSIONS = {
    "gray": cv2.COLOR_BGR2GRAY,
    "hsv":  cv2.COLOR_BGR2HSV,
    "lab":  cv2.COLOR_BGR2LAB,
}


def convert_to(frame, color_format):
    """Convert a BGR frame to another colour format ("gray", "hsv" or "lab").
    Returns a new image; the input frame is not changed."""
    return cv2.cvtColor(frame, COLOR_CONVERSIONS[color_format])


def resize(frame, scale):
    """Scale the frame by `scale` (0.5 = half width and half height)."""
    if scale == 1:
        return frame
    return cv2.resize(frame, None, fx=scale, fy=scale)


def draw_line(frame, line, color=(0, 0, 255), thickness=2):
    """Draw the line a*x + b*y + c = 0 across the whole frame. Draws in place."""
    a, b, c = line
    height, width = frame.shape[:2]

    if abs(b) > abs(a):
        # Mostly horizontal: find y at the left and right edges
        x1, x2 = 0, width
        y1, y2 = -(a * x1 + c) / b, -(a * x2 + c) / b
    else:
        # Mostly vertical: find x at the top and bottom edges
        y1, y2 = 0, height
        x1, x2 = -(b * y1 + c) / a, -(b * y2 + c) / a

    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return frame


def draw_text(frame, text, line_number=0, color=(0, 0, 255)):
    """Write text in the top-left corner. line_number moves it down one row per number."""
    cv2.putText(img = frame,
                text = text,
                org = (10, 30 + 30 * line_number),
                fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                fontScale = 0.8,
                color = color,
                thickness = 2)
    return frame
