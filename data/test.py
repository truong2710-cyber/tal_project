from typing import Iterable, List, Optional, Tuple, Union
from PIL import Image, ImageDraw, ImageFont

Box = Union[Tuple[int, int, int, int], List[int]]

def draw_boxes_on_image(
    image_path: str,
    boxes: Iterable[Box],
    box_format: str = "xyxy",           # "xyxy" -> (x1,y1,x2,y2), "xywh" -> (x,y,w,h)
    labels: Optional[Iterable[str]] = None,
    colors: Optional[Iterable[Tuple[int, int, int]]] = None,  # RGB tuples per box
    outline_width: int = 3,
    font_size: int = 14,
    save_path: Optional[str] = None,
    return_image: bool = False,
):
    """
    Draw bounding boxes (and optional labels) on an image.

    Args:
        image_path: Path to the source image.
        boxes: Iterable of boxes (N x 4). Format chosen by `box_format`.
        box_format: "xyxy" for (x1,y1,x2,y2) or "xywh" for (x,y,w,h).
        labels: Optional list of N strings, one per box.
        colors: Optional list of N RGB tuples; defaults to a single color reused.
        outline_width: Rectangle border thickness in pixels.
        font_size: Label text size in points.
        save_path: If provided, saves the visualized image here.
        return_image: If True, returns a PIL.Image with drawings applied.

    Returns:
        PIL.Image.Image if `return_image` is True, otherwise None.
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Try to get a reasonable default font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    boxes = list(boxes)
    n = len(boxes)
    labels_list = list(labels) if labels is not None else [None] * n

    if colors is None:
        colors_list = [(255, 0, 0)] * n  # default red
    else:
        colors_list = list(colors)
        if len(colors_list) < n:  # repeat last color if fewer provided
            colors_list += [colors_list[-1]] * (n - len(colors_list))

    for i, b in enumerate(boxes):
        if box_format == "xyxy":
            x1, y1, x2, y2 = b
        elif box_format == "xywh":
            x, y, w, h = b
            x1, y1, x2, y2 = x, y, x + w, y + h
        else:
            raise ValueError("box_format must be 'xyxy' or 'xywh'.")

        color = tuple(colors_list[i])

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=outline_width)

        # Optional label
        label = 'Person'
        if label:
            # Compute text bbox and draw filled background for readability
            text_w, text_h = draw.textbbox((0, 0), label, font=font)[2:]
            pad = 2
            bg = [x1, max(0, y1 - text_h - 2 * pad), x1 + text_w + 2 * pad, y1]
            draw.rectangle(bg, fill=color)
            draw.text((x1 + pad, bg[1] + pad), label, fill=(255, 255, 255), font=font)

    if save_path:
        img.save(save_path)

    if return_image:
        return img

# 1) (x1,y1,x2,y2) boxes, save to file
draw_boxes_on_image("/workspace/truongvq/data/skating/frame/2024-10-15_Minnesota_vs_St. Louis/2024-10-15_Minnesota_vs_St. Louis_003226.jpg", [(639., 595., 725., 720.), (1171.76318359375, 162.575927734375, 1222.45458984375, 281.12109375), (316.68939208984375, 291.97576904296875, 446.99786376953125, 431.54901123046875), (1028.0452880859375, 201.59005737304688, 1068.1170654296875, 316.2393798828125)], box_format="xyxy",
                    labels=["cat"], save_path="./annotated.jpg")