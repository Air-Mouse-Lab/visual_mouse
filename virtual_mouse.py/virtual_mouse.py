import cv2
import numpy as np
import time
import math

# 시스템 마우스 제어 (옵션)
try:
    import pyautogui
    HAVE_PYAUTO = True
    SCREEN_W, SCREEN_H = pyautogui.size()
except Exception:
    HAVE_PYAUTO = False
    SCREEN_W, SCREEN_H = 1920, 1080  # 대충 값

# MediaPipe 손 추적
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# ---------------- Configuration ----------------
CAM_INDEX = 0
FRAME_W, FRAME_H = 960, 540               # 카메라 프레임 리사이즈
ROI_MARGIN = 80                           # 가장자리 버퍼(프레임 → 스크린 매핑 안정화)
SMOOTHING = 0.25                          # 0~1 (값이 클수록 관성 큼)
PINCH_CLICK_THRESH_STABLE = 0.065         # 안정
PINCH_CLICK_THRESH_SENSITIVE = 0.090      # 민감 (프레임 폭 비례 거리)
HOLD_TIME_FOR_DRAG = 0.35                 # 초 (길게 집게 시 드래그 시작)
# ------------------------------------------------

def lerp(a, b, t):  # 선형보간
    return a + (b - a) * t

def norm_dist(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다. CAM_INDEX 확인.")
        return

    # 선호 해상도로 설정 시도
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=1,
                           min_detection_confidence=0.6,
                           min_tracking_confidence=0.6)

    prev_x, prev_y = None, None
    os_mouse_enabled = False
    show_roi = True
    sensitive = False
    pinch_thresh = PINCH_CLICK_THRESH_STABLE

    dragging = False
    pinch_start_time = None
    last_click_time = 0

    ptime = 0

    # 스크린 매핑용 ROI (프레임 내 사용 영역)
    def roi_bounds(w, h, margin):
        return (margin, margin, w - margin, h - margin)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)  # 거울 모드
        h, w = frame.shape[:2]

        # RGB로 변환 후 추론
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        cursor_px = None
        pinch_now = False

        if res.multi_hand_landmarks:
            handLms = res.multi_hand_landmarks[0]
            # 랜드마크(0~1)를 픽셀 좌표로
            lm = [(int(pt.x * w), int(pt.y * h)) for pt in handLms.landmark]

            # 검지 끝(8), 엄지 끝(4)
            ix, iy = lm[8]
            tx, ty = lm[4]

            # 프레임 좌표에서 ROI 제한
            x1, y1, x2, y2 = roi_bounds(w, h, ROI_MARGIN)
            ix = clamp(ix, x1, x2)
            iy = clamp(iy, y1, y2)

            # 스크린 좌표로 매핑
            sx = np.interp(ix, [x1, x2], [0, SCREEN_W])
            sy = np.interp(iy, [y1, y2], [0, SCREEN_H])

            # 스무딩
            if prev_x is None:
                cur_x, cur_y = sx, sy
            else:
                cur_x = lerp(prev_x, sx, 1.0 - SMOOTHING)
                cur_y = lerp(prev_y, sy, 1.0 - SMOOTHING)
            prev_x, prev_y = cur_x, cur_y

            # 가상 커서(프레임 내 위치)도 표시
            vis_x = int(np.interp(cur_x, [0, SCREEN_W], [0, w]))
            vis_y = int(np.interp(cur_y, [0, SCREEN_H], [0, h]))
            cursor_px = (vis_x, vis_y)

            # 집게(엄지-검지 거리)로 클릭/드래그 판정
            nd = norm_dist((ix/w, iy/h), (tx/w, ty/h))  # 0~1 스케일
            pinch_now = nd < pinch_thresh

            # 시스템 마우스 제어
            if os_mouse_enabled and HAVE_PYAUTO:
                try:
                    pyautogui.moveTo(cur_x, cur_y, duration=0)  # 즉시 이동
                    now = time.time()
                    if pinch_now:
                        if pinch_start_time is None:
                            pinch_start_time = now
                        held = now - pinch_start_time
                        # 오래 집게 → 드래그
                        if held > HOLD_TIME_FOR_DRAG and not dragging:
                            pyautogui.mouseDown()
                            dragging = True
                        # 짧게 집게 → 클릭(디바운스)
                        if not dragging and (now - last_click_time) > 0.25:
                            # 눌림 유지 동안은 추가 클릭 방지
                            pass
                    else:
                        # 집게 해제 시 처리
                        if pinch_start_time is not None:
                            held = time.time() - pinch_start_time
                            if dragging:
                                pyautogui.mouseUp()
                                dragging = False
                            else:
                                # 짧게 누르고 떼면 클릭
                                if held <= HOLD_TIME_FOR_DRAG:
                                    pyautogui.click()
                                    last_click_time = time.time()
                            pinch_start_time = None
                except Exception as e:
                    # 권한/포커스 이슈 등
                    cv2.putText(frame, f"pyautogui error: {e}", (10, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 손 랜드마크 그리기(디버그)
            mp_draw.draw_landmarks(
                frame, handLms, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

        # ROI 가이드라인
        if show_roi:
            x1, y1, x2, y2 = ROI_MARGIN, ROI_MARGIN, w-ROI_MARGIN, h-ROI_MARGIN
            cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 1)
            cv2.putText(frame, "Move inside the box for best control",
                        (x1+10, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)

        # 가상 커서 그리기
        if cursor_px is not None:
            color = (0, 255, 0) if not pinch_now else (0, 140, 255)
            cv2.circle(frame, cursor_px, 10, color, -1)
            cv2.circle(frame, cursor_px, 20, color, 2)

        # 상태 텍스트
        ctime = time.time()
        fps = 1.0 / (ctime - ptime) if ptime else 0.0
        ptime = ctime

        status = [
            f"FPS: {fps:.1f}",
            f"OS Mouse: {'ON' if os_mouse_enabled and HAVE_PYAUTO else 'OFF'}",
            f"Click Sens: {'SENSITIVE' if sensitive else 'STABLE'}",
            f"Drag: {'ON' if dragging else 'OFF'}",
            "Keys: [m] toggle OS mouse, [c] sensitivity, [r] ROI, [q] quit"
        ]
        for i, t in enumerate(status):
            cv2.putText(frame, t, (10, 20 + i*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Virtual Mouse (Hand Tracking)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            os_mouse_enabled = not os_mouse_enabled
            if os_mouse_enabled and not HAVE_PYAUTO:
                print("⚠️ pyautogui가 없어 시스템 마우스 제어를 사용할 수 없습니다.")
        elif key == ord('c'):
            sensitive = not sensitive
            pinch_thresh = PINCH_CLICK_THRESH_SENSITIVE if sensitive else PINCH_CLICK_THRESH_STABLE
        elif key == ord('r'):
            show_roi = not show_roi

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
