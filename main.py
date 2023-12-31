import math
import curses
import cv2
import numpy as np


DIMENSIONS = (300, 100)
MIN_THR = 5.5 # minimum threshold for which its considered darkness
SMALLEST_THR = 5.7 # the highest value for which only a dot will appear
BOLD_THR = 8 # the smallest value for which it will sho
W, H = 1000, 900 # height and width at which we'll process the frame
FPS = 60 # FPS at which the video will be read
def main():
    # img = cv2.imread("img.jpg")
    # img = sobel(img)
    cap = cv2.VideoCapture("vid.mp4")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FPS, FPS)
    stdscr = curses.initscr()
    while (cap.isOpened()):
        # Capture frame-by-frame
        # start = time.time_ns()
        ret, frame = cap.read()
        if ret:
            # cv2.imshow('Frame', frame)

            frame_x, frame_y = sobel(frame)


            # print(frame_to_ascii(frame))
            ascii_frame = frame_to_ascii(frame_x, frame_y)
            stdscr.clear()
            for y, line in enumerate(ascii_frame, 0):
                try:
                    stdscr.addstr(y, 0, "".join(line))
                except:
                    pass

            stdscr.refresh()
            # stdscr.addstr(0, 0, "frame_to_ascii(frame)\nh")

            # Press Q on keyboard to  exit
        # end = time.time_ns()
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # When everything done, release the video capture object
    cap.release()
    # img = sobel(img)
    # cv2.waitKey(0)

def sobel(frame: np.ndarray):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (3, 3), 3)
    # gray = cv2.medianBlur(gray, 5)
    # gray = cv2.resize(gray, DIMENSIONS, interpolation=cv2.INTER_AREA)

    sobel_horizontal = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_vertical = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    scale_to_norm = 256/max(max(sobel_vertical.max(), -sobel_vertical.max()), max(sobel_horizontal.max(), -sobel_horizontal.min()))
    # sobel_horizontal = cv2.convertScale(sobel_horizontal)
    # sobel_vertical = cv2.convertScale(sobel_vertical)

    # frame = cv2.addWeighted(sobel_vertical, 0.5, sobel_horizontal, 0.5, 0)
    sobel_horizontal = cv2.resize(sobel_horizontal, DIMENSIONS, interpolation=cv2.INTER_AREA)
    sobel_vertical = cv2.resize(sobel_vertical, DIMENSIONS, interpolation=cv2.INTER_AREA)
    # frame = cv2.resize(frame, DIMENSIONS, interpolation=cv2.INTER_AREA)

    # frame = np.zeros((DIMENSIONS[1], DIMENSIONS[0], 2))
    # rows, cols, _ = frame.shape

    # def update_line(index: int):
    #     for j in range(cols):
    #         frame[index][j] = [sobel_horizontal[index][j]*scale_to_norm, sobel_vertical[index][j]*scale_to_norm]
    #
    # start = time.time_ns()
    # for i in range(rows):
    #     t = threading.Thread(target=update_line, args=(i,))
    #     t.start()
    #     # for j in range(cols):
    #     #     frame[i][j] = [sobel_horizontal[i][j]*scale_to_norm, sobel_vertical[i][j]*scale_to_norm]
    # # frame = cv2.medianBlur(frame, 3)
    # end = time.time_ns()
    return sobel_horizontal*scale_to_norm, sobel_vertical*scale_to_norm

ENV_RADIUS = 18
def frame_to_ascii(frame_x: np.ndarray, frame_y: np.ndarray):
    s = []
    rows, cols = frame_x.shape

    s = [[color_to_ascii(np.array([frame_x[i][j], frame_y[i][j]])) for j in range(cols)] for i in range(rows)]
    # for i in range(rows):
    #     update_line()
    #     # for j in range(cols):
    #     #     s += color_to_ascii(np.array(frame[i][j]))
    #     # s += "\n"
    return s




def color_to_ascii(color_bgr: np.ndarray):
    mag = np.linalg.norm(color_bgr)
    if (mag < MIN_THR):
        return " "
    if (mag < SMALLEST_THR):
        return "."
    angle_deg = math.degrees(math.atan2(color_bgr[1], color_bgr[0])) % 180
    if (mag >= BOLD_THR):

        if (angle_deg < ENV_RADIUS or angle_deg > 180 - ENV_RADIUS):
            return "|"
        if (angle_deg < 180-ENV_RADIUS and angle_deg > 180 - ENV_RADIUS*3):
            return "\\"
        if ((angle_deg < ENV_RADIUS*4 and angle_deg > ENV_RADIUS*2) ):
            return "/"
        return "_"
    if (angle_deg < ENV_RADIUS or angle_deg >= 180 - ENV_RADIUS):
        return "'"
    if (angle_deg <= 180 - ENV_RADIUS and angle_deg >= 180 - ENV_RADIUS * 3):
        return "`"
    if ((angle_deg < ENV_RADIUS * 4 and angle_deg > ENV_RADIUS * 2)):
        return ","
    return "-"


if __name__ == '__main__':
    main()