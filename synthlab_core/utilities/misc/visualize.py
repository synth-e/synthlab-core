import cv2
import numpy as np
from PIL import Image


# generate 10k color, make sure there's no white or black and no similar colors
def generate_colors(n):
    colors = []

    for i in range(n):
        r = np.random.randint(0, 255)
        g = np.random.randint(0, 255)
        b = np.random.randint(0, 255)
        colors.append((r, g, b))

    return np.array(colors).astype(np.uint8)


color_pallete = generate_colors(10000)


def overlay_text(mask, out_mask, text, color=(255, 255, 255)):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Find the contour with the largest area
    if contours:
        # Calculate the center of mass of the contour
        max_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(max_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # Overlay text on the mask
            cv2.putText(
                out_mask,
                text,
                (cX, cY),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
    return out_mask


def color_mask(mask: np.array, id2label: dict, id2palette: dict = None) -> np.array:
    if id2palette is None:
        id2palette = color_pallete[: max(id2label.keys())]

    w, h = mask.shape
    colored_mask = np.zeros((w, h, 3), np.uint8)
    for id in np.unique(mask):
        label = id2label[id]
        bin_mask = (mask == id).astype(np.uint8)
        colored_mask[mask == id] = id2palette[id]
        colored_mask = overlay_text(bin_mask, colored_mask, label)
    return colored_mask


def apply_mask_to_image(
    image: np.ndarray,
    mask: np.ndarray,
    id2label: dict,
    id2palette: dict = {},
    opacity=0.55,
):
    add_pallete_id = 0
    for key in id2label.keys():
        if key not in id2palette.keys():
            id2palette[key] = color_pallete[add_pallete_id]
            add_pallete_id += 1

    # resize the mask to the image's size
    colored_mask = color_mask(mask, id2label, id2palette)

    w, h = image.shape[:2]
    mask = np.array(Image.fromarray(colored_mask).resize((h, w), Image.BICUBIC))
    result_image = image * (1 - opacity) + mask * opacity
    return result_image


def append_side_text(
    image: np.ndarray,
    text: str,
    font_scale: float = 0.5,
    thickness: int = 1,
    side: str = "r",
):
    h, w = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    lines = text.split("\n")
    text_w, text_h = 0, 0

    for line in lines:
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_w = max(text_size[0], text_w)
        text_h += text_size[1]

    if side == "r" or side == "l":
        new_h, new_w = max(h, text_h), w + text_w
    elif side == "t" or side == "b":
        new_h, new_w = h + text_h, max(w, text_w)
    else:
        raise ValueError(f"Invalid side {side}")

    board = np.ones((new_h, new_w, 3), np.uint8) * 255

    if side == "r":
        board[:h, :w] = image
        for i, line in enumerate(lines):
            cv2.putText(
                board,
                line,
                (w, text_h + i * text_h),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
            )
    elif side == "l":
        board[:h, text_w:] = image
        for i, line in enumerate(lines):
            cv2.putText(
                board,
                line,
                (0, text_h + i * text_h),
                font,
                font_scale,
                (0, 0, 0),
                thickness,
            )
    elif side == "t":
        board[text_h:, :w] = image
        for i, line in enumerate(lines):
            cv2.putText(
                board, line, (0, i * text_h), font, font_scale, (0, 0, 0), thickness
            )
    elif side == "b":
        board[:h, :w] = image
        for i, line in enumerate(lines):
            cv2.putText(
                board, line, (0, h + i * text_h), font, font_scale, (0, 0, 0), thickness
            )

    return board
