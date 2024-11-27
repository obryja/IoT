import cv2
import mediapipe as mp
import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
SERVO_PIN = 19
GPIO.setup(SERVO_PIN, GPIO.OUT)

pwm = GPIO.PWM(SERVO_PIN, 50)
pwm.start(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
has_moved = False
last_move_time = 0
delay_time = 5

cap = cv2.VideoCapture(0)

def is_thumbs_up(landmarks):
    # Landmark dla nadgarstka (środek dłoni)
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
 
    # Landmarki kciuka
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]  # koniec kciuka
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]    # staw IP kciuka
 
    # Landmarki pozostałych palców (końcówki i stawy PIP)
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP]
 
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
 
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = landmarks[mp_hands.HandLandmark.RING_FINGER_PIP]
 
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = landmarks[mp_hands.HandLandmark.PINKY_PIP]
 
    # 1. Sprawdzenie, czy kciuk jest wyprostowany i oddalony od środka dłoni
    thumb_extended = (
        thumb_tip.y < thumb_ip.y and  # Kciuk wyżej w osi y niż staw IP
        abs(thumb_tip.x - wrist.x) < 0.15  # Kciuk znacznie oddalony od nadgarstka
    )
 
    # 2. Dodatkowe sprawdzenie dla pozostałych palców: ich końcówki są blisko nadgarstka
    fingers_close_to_wrist_x = (
        abs(index_tip.x - wrist.x) < 0.15 and
        abs(middle_tip.x - wrist.x) < 0.15 and
        abs(ring_tip.x - wrist.x) < 0.15 and
        abs(pinky_tip.x - wrist.x) < 0.15
    )

    # 3. Dodatkowe sprawdzenie dla pozostałych palców: ich końcówki są blisko nadgarstka
    fingers_close_to_wrist_y = (
        abs(index_tip.y - wrist.y) < 0.15 and
        abs(middle_tip.y - wrist.y) < 0.15 and
        abs(ring_tip.y - wrist.y) < 0.15 and
        abs(pinky_tip.y - wrist.y) < 0.15
    )

 
    return thumb_extended and fingers_close_to_wrist_x and fingers_close_to_wrist_y 

def move_servo_to_angle(angle):
    duty_cycle = angle / 18 + 2
    pwm.ChangeDutyCycle(duty_cycle)
    time.sleep(0.1)
    pwm.ChangeDutyCycle(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    current_time = time.time()
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if is_thumbs_up(hand_landmarks.landmark) and current_time - last_move_time > delay_time:
                if not has_moved: 
                    move_servo_to_angle(90)
                    has_moved = True
                else:
                    move_servo_to_angle(0)
                    has_moved = False
                last_move_time = current_time

    cv2.imshow('Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pwm.stop()
GPIO.cleanup()

