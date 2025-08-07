import cv2

# 얼굴 검출을 위한 Haar cascade 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + '../data/haarcascade_frontalface_default.xml')

rate = 15  # 모자이크에 사용할 축소 비율 (1/rate)

# 카메라 캡처 시작
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 얼굴 검출
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(80, 80))

    for (x, y, w, h) in faces:
        roi = img[y:y+h, x:x+w]

        # 모자이크 처리 (축소 후 확대)
        roi_face = cv2.resize(roi, (w//rate, h//rate), interpolation=cv2.INTER_LINEAR)
        mosaic = cv2.resize(roi_face, (w, h), interpolation=cv2.INTER_NEAREST)

        # 원래 영상에 모자이크 덮어쓰기
        img[y:y+h, x:x+w] = mosaic

    # 결과 출력
    cv2.imshow("Mosaic Face", img)

    # ESC 키 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()