# 졸음 방지 프로그램

# 1. 눈 부분 영역 추출
# 2. 깨어있는 상태의 눈 수치 조정
# 3. 조정 수치 미만 시, 졸음 판정
# 4. 알람 발생

import cv2

# 얼굴과  검출을 위한 케스케이드 분류기 생성 
face_cascade = cv2.CascadeClassifier('../data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('../data/haarcascade_eye.xml')

# 졸음 판정을 위한 상태 변수
closed_eyes_frames = 0
# 눈 감은 상태가 몇 프레임 이상 지속되면 졸음 판정
ALERT_THRESHOLD = 15

# 카메라 캡쳐 활성화
cap = cv2.VideoCapture(1)
while cap.isOpened():    
    ret, img = cap.read()  # 프레임 읽기
    if ret:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 얼굴 검출    
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, \
                                        minNeighbors=5, minSize=(80,80))
        
        # 얼굴 감지 시, 졸음 감지 여부 판정
        if len(faces) > 0:
            for(x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]

                # 눈 검출
                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

                # 눈이 감긴 것으로 추정되는 경우
                if len(eyes) < 2:
                    closed_eyes_frames += 1
                else:
                    closed_eyes_frames = 0  # 눈을 떴다면 카운트 초기화

                # 경고 임계값 초과 시
                if closed_eyes_frames >= ALERT_THRESHOLD:
                    drowsy = True
                    cv2.putText(img, 'DROWSINESS DETECTED!', (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

                else:
                    cv2.putText(img, 'You are Awake', (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)  
        # 얼굴 미감지 시, 표시
        else: 
            cv2.putText(img, 'Face undetected', (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3) 
        cv2.imshow('Drowsiness prevention camera', img)

    # ESC 누르면 종료
    if cv2.waitKey(5) == 27:
        break
cv2.destroyAllWindows()