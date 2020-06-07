import cv2 as cv

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return x, y, w, h

def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return x - x_off, x + width + x_off, y - y_off, y + height + y_off

def get_color(emotion, prob):
    if emotion.lower() == 'angry':
        color = (0, 0, 255)
    elif emotion.lower() == 'disgust':
        color = (255, 0, 0)
    elif emotion.lower() == 'fear':
        color = (0, 255, 255)
    elif emotion.lower() == 'happy':
        color = (255, 255, 0)
    elif emotion.lower() == 'sad':
        color = (255, 255, 255)
    elif emotion.lower() == 'surprise':
        color = (255, 0, 255)
    else:
        color = (0, 255, 0)
    return color

def draw_bounding_box(image, coordinates, color):
    x, y, w, h = coordinates
    cv.rectangle(image, (x, y), (x + w, y + h), color, 3)
    return image
    
def draw_text(image, coordinates, text, color, x_offset=0, y_offset=0,
              font_scale=1, thickness=2):
    x, y = coordinates[:2]
    cv.putText(image, text, (x + x_offset, y + y_offset),
                cv.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv.LINE_AA)
    return image