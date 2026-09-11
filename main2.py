import cv2
import mediapipe as mp
import numpy as np
import math
import threading

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


# ============================================================
# SETTINGS
# ============================================================

WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 700

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

ROTATION_SENSITIVITY = 250.0

# Movement sensitivity for grabbed cube
MOVE_SENSITIVITY_X = 4.0
MOVE_SENSITIVITY_Y = 4.0

# Scaling limits
MIN_SCALE = 0.25
MAX_SCALE = 3.0

# Pinch threshold
PINCH_THRESHOLD = 0.055

# Smoothing
SMOOTHING = 0.25


# ============================================================
# CUBE STATE
# ============================================================

cube_rotation_x = 0.0
cube_rotation_y = 0.0
cube_rotation_z = 0.0

cube_x = 0.0
cube_y = 0.0

cube_scale = 1.0


# ============================================================
# HAND STATE
# ============================================================

hand_detected = False

previous_hand_x = None
previous_hand_y = None

# Grab state
grabbed = False

grab_start_hand_x = None
grab_start_hand_y = None

grab_start_cube_x = 0.0
grab_start_cube_y = 0.0


# Scaling state
scaling = False

initial_two_hand_distance = None
initial_cube_scale = 1.0


# Smoothed hand positions
smooth_hand_positions = {}


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def distance_between_points(p1, p2):

    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


def is_pinching(hand_landmarks):

    thumb = hand_landmarks.landmark[4]
    index = hand_landmarks.landmark[8]

    distance = math.sqrt(
        (thumb.x - index.x) ** 2 +
        (thumb.y - index.y) ** 2
    )

    return distance < PINCH_THRESHOLD


def smooth_position(hand_id, x, y):

    global smooth_hand_positions

    if hand_id not in smooth_hand_positions:

        smooth_hand_positions[hand_id] = [x, y]

    else:

        old_x, old_y = smooth_hand_positions[hand_id]

        new_x = (
            old_x * (1.0 - SMOOTHING)
            + x * SMOOTHING
        )

        new_y = (
            old_y * (1.0 - SMOOTHING)
            + y * SMOOTHING
        )

        smooth_hand_positions[hand_id] = [
            new_x,
            new_y
        ]

    return smooth_hand_positions[hand_id]


# ============================================================
# OPENGL INITIALIZATION
# ============================================================

def init_opengl():

    glEnable(GL_DEPTH_TEST)

    glClearColor(
        0.05,
        0.05,
        0.08,
        1.0
    )

    glMatrixMode(GL_PROJECTION)

    glLoadIdentity()

    gluPerspective(
        45.0,
        WINDOW_WIDTH / WINDOW_HEIGHT,
        0.1,
        100.0
    )

    glMatrixMode(GL_MODELVIEW)


# ============================================================
# DRAW CUBE
# ============================================================

def draw_cube():

    # Change cube appearance while grabbed/scaling
    if scaling:

        glColor3f(1.0, 0.6, 0.0)

    elif grabbed:

        glColor3f(0.0, 1.0, 0.3)

    else:

        glColor3f(0.0, 0.7, 1.0)


    glBegin(GL_QUADS)

    # FRONT
    glColor3f(0.0, 0.7, 1.0)

    glVertex3f(-1, -1, 1)
    glVertex3f(1, -1, 1)
    glVertex3f(1, 1, 1)
    glVertex3f(-1, 1, 1)

    # BACK
    glColor3f(1.0, 0.2, 0.2)

    glVertex3f(-1, -1, -1)
    glVertex3f(-1, 1, -1)
    glVertex3f(1, 1, -1)
    glVertex3f(1, -1, -1)

    # LEFT
    glColor3f(0.2, 1.0, 0.3)

    glVertex3f(-1, -1, -1)
    glVertex3f(-1, -1, 1)
    glVertex3f(-1, 1, 1)
    glVertex3f(-1, 1, -1)

    # RIGHT
    glColor3f(1.0, 0.8, 0.0)

    glVertex3f(1, -1, -1)
    glVertex3f(1, 1, -1)
    glVertex3f(1, 1, 1)
    glVertex3f(1, -1, 1)

    # TOP
    glColor3f(0.7, 0.2, 1.0)

    glVertex3f(-1, 1, -1)
    glVertex3f(-1, 1, 1)
    glVertex3f(1, 1, 1)
    glVertex3f(1, 1, -1)

    # BOTTOM
    glColor3f(0.0, 1.0, 0.8)

    glVertex3f(-1, -1, -1)
    glVertex3f(1, -1, -1)
    glVertex3f(1, -1, 1)
    glVertex3f(-1, -1, 1)

    glEnd()


# ============================================================
# OPENGL DISPLAY
# ============================================================

def display():

    glClear(
        GL_COLOR_BUFFER_BIT |
        GL_DEPTH_BUFFER_BIT
    )

    glLoadIdentity()

    # Move camera backwards
    glTranslatef(
        cube_x,
        cube_y,
        -6.0
    )

    # Rotate cube
    glRotatef(
        cube_rotation_x,
        1.0,
        0.0,
        0.0
    )

    glRotatef(
        cube_rotation_y,
        0.0,
        1.0,
        0.0
    )

    glRotatef(
        cube_rotation_z,
        0.0,
        0.0,
        1.0
    )

    # Scale cube
    glScalef(
        cube_scale,
        cube_scale,
        cube_scale
    )

    draw_cube()

    glutSwapBuffers()


# ============================================================
# WINDOW RESIZE
# ============================================================

def reshape(width, height):

    if height == 0:
        height = 1

    glViewport(
        0,
        0,
        width,
        height
    )

    glMatrixMode(GL_PROJECTION)

    glLoadIdentity()

    gluPerspective(
        45.0,
        width / float(height),
        0.1,
        100.0
    )

    glMatrixMode(GL_MODELVIEW)


# ============================================================
# ONE HAND ROTATION
# ============================================================

def update_rotation(x, y):

    global previous_hand_x
    global previous_hand_y

    global cube_rotation_x
    global cube_rotation_y

    if previous_hand_x is not None:

        delta_x = x - previous_hand_x
        delta_y = y - previous_hand_y

        cube_rotation_y += (
            delta_x * ROTATION_SENSITIVITY
        )

        cube_rotation_x += (
            delta_y * ROTATION_SENSITIVITY
        )

    previous_hand_x = x
    previous_hand_y = y


# ============================================================
# GRAB / MOVE
# ============================================================

def start_grab(x, y):

    global grabbed

    global grab_start_hand_x
    global grab_start_hand_y

    global grab_start_cube_x
    global grab_start_cube_y

    grabbed = True

    grab_start_hand_x = x
    grab_start_hand_y = y

    grab_start_cube_x = cube_x
    grab_start_cube_y = cube_y

    print(">>> CUBE GRABBED")


def update_grab(x, y):

    global cube_x
    global cube_y

    if not grabbed:
        return

    delta_x = x - grab_start_hand_x
    delta_y = y - grab_start_hand_y

    # Convert camera coordinates to OpenGL movement
    cube_x = (
        grab_start_cube_x
        + delta_x * MOVE_SENSITIVITY_X
    )

    cube_y = (
        grab_start_cube_y
        - delta_y * MOVE_SENSITIVITY_Y
    )


def release_grab():

    global grabbed

    if grabbed:

        grabbed = False

        print(">>> CUBE RELEASED")


# ============================================================
# TWO HAND SCALING
# ============================================================

def start_scaling(hand1, hand2):

    global scaling

    global initial_two_hand_distance
    global initial_cube_scale

    x1, y1 = hand1
    x2, y2 = hand2

    initial_two_hand_distance = distance_between_points(
        (x1, y1),
        (x2, y2)
    )

    initial_cube_scale = cube_scale

    scaling = True

    print(">>> SCALING STARTED")


def update_scaling(hand1, hand2):

    global cube_scale

    if not scaling:
        return

    x1, y1 = hand1
    x2, y2 = hand2

    current_distance = distance_between_points(
        (x1, y1),
        (x2, y2)
    )

    if initial_two_hand_distance is None:
        return

    scale_ratio = (
        current_distance /
        initial_two_hand_distance
    )

    new_scale = (
        initial_cube_scale *
        scale_ratio
    )

    cube_scale = max(
        MIN_SCALE,
        min(MAX_SCALE, new_scale)
    )


def stop_scaling():

    global scaling
    global initial_two_hand_distance

    if scaling:

        scaling = False

        initial_two_hand_distance = None

        print(">>> SCALING FINISHED")


# ============================================================
# HAND PROCESSING
# ============================================================

def process_hand(frame):

    global hand_detected

    global previous_hand_x
    global previous_hand_y

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb_frame)

    detected_hands = []

    if results.multi_hand_landmarks:

        for i, hand_landmarks in enumerate(
            results.multi_hand_landmarks
        ):

            # Palm center
            palm = hand_landmarks.landmark[9]

            x = palm.x
            y = palm.y

            # Smooth position
            x, y = smooth_position(
                i,
                x,
                y
            )

            pinch = is_pinching(
                hand_landmarks
            )

            detected_hands.append({
                "x": x,
                "y": y,
                "pinch": pinch
            })

            # Draw skeleton
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Draw pinch status
            if pinch:

                cv2.putText(
                    frame,
                    "PINCH",
                    (
                        int(x * CAMERA_WIDTH) - 40,
                        int(y * CAMERA_HEIGHT) - 20
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    hand_detected = len(detected_hands) > 0

    # ========================================================
    # TWO HAND MODE
    # ========================================================

    if len(detected_hands) >= 2:

        hand1 = detected_hands[0]
        hand2 = detected_hands[1]

        if hand1["pinch"] and hand2["pinch"]:

            # If scaling hasn't started
            if not scaling:

                # Cancel any grab
                release_grab()

                start_scaling(
                    (hand1["x"], hand1["y"]),
                    (hand2["x"], hand2["y"])
                )

            update_scaling(
                (hand1["x"], hand1["y"]),
                (hand2["x"], hand2["y"])
            )

            previous_hand_x = None
            previous_hand_y = None

            return frame

        else:

            # No longer two-hand pinching
            stop_scaling()

    else:

        stop_scaling()

    # ========================================================
    # ONE HAND MODE
    # ========================================================

    if len(detected_hands) == 1:

        hand = detected_hands[0]

        x = hand["x"]
        y = hand["y"]

        pinch = hand["pinch"]

        # ----------------------------------------------------
        # PINCH = GRAB
        # ----------------------------------------------------

        if pinch:

            if not grabbed:

                start_grab(
                    x,
                    y
                )

            update_grab(
                x,
                y
            )

            previous_hand_x = None
            previous_hand_y = None

        # ----------------------------------------------------
        # RELEASE = NORMAL ROTATION
        # ----------------------------------------------------

        else:

            if grabbed:

                release_grab()

            update_rotation(
                x,
                y
            )

    else:

        # No hands
        previous_hand_x = None
        previous_hand_y = None

        if grabbed:

            release_grab()

    return frame


# ============================================================
# CAMERA THREAD
# ============================================================

def camera_loop():

    cap = cv2.VideoCapture(0)

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    if not cap.isOpened():

        print("ERROR: Could not open camera.")

        return

    print()
    print("==========================================")
    print(" HAND CONTROLLED OPENGL CUBE - V2.1")
    print("==========================================")
    print()
    print("CONTROLS")
    print()
    print("One hand:")
    print("  Move hand       -> Rotate cube")
    print("  Pinch           -> Grab cube")
    print("  Move while pinching -> Move cube")
    print("  Release pinch   -> Release cube")
    print()
    print("Two hands:")
    print("  Pinch BOTH      -> Scale mode")
    print("  Move hands apart -> Bigger")
    print("  Move hands together -> Smaller")
    print()
    print("Press Q to quit.")
    print()

    while True:

        success, frame = cap.read()

        if not success:

            print("ERROR: Could not read camera.")

            break

        # Mirror camera
        frame = cv2.flip(
            frame,
            1
        )

        # Process hands
        frame = process_hand(
            frame
        )

        # ====================================================
        # STATUS
        # ====================================================

        if scaling:

            status = "SCALING"

        elif grabbed:

            status = "GRABBED"

        elif hand_detected:

            status = "ROTATING"

        else:

            status = "NO HAND"

        cv2.putText(
            frame,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        # Scale display
        cv2.putText(
            frame,
            f"Scale: {cube_scale:.2f}x",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "1 hand pinch = Grab | 2 hand pinch = Scale",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Hand Tracking - V2.1",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break

    cap.release()

    cv2.destroyAllWindows()

    glutLeaveMainLoop()


# ============================================================
# OPENGL IDLE
# ============================================================

def idle():

    glutPostRedisplay()


# ============================================================
# MAIN
# ============================================================

def main():

    glutInit()

    glutInitDisplayMode(
        GLUT_DOUBLE |
        GLUT_RGB |
        GLUT_DEPTH
    )

    glutInitWindowSize(
        WINDOW_WIDTH,
        WINDOW_HEIGHT
    )

    glutCreateWindow(
        b"Hand Controlled OpenGL Cube - V2.1"
    )

    init_opengl()

    glutDisplayFunc(
        display
    )

    glutReshapeFunc(
        reshape
    )

    glutIdleFunc(
        idle
    )

    # Camera runs separately
    camera_thread = threading.Thread(
        target=camera_loop,
        daemon=True
    )

    camera_thread.start()

    glutMainLoop()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()