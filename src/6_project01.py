# 성별 분류 완료
# 전처리 함수 및 보정 이미지 함수 포함
from haar_utils import *

# cascade classifier 로드
face_cascade = cv2.CascadeClassifier('../data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('../data/haarcascade_eye.xml')


img, img_gray = preprocessing('sullyoon.jpg')


# 이미지 로드에 실패할 경우 예외(Exception) 발생시킴
if img is None:
    raise Exception('사진을 여는데 실패했습니다.')

faces = face_cascade.detectMultiScale(img_gray)
print('찾은 얼굴 수 : %d' % len(faces))

#모든 얼굴을 순회하며 처리
for (x, y, w, h) in faces:
    # 얼굴의 이미지 영역 자르기
    face_gray = img_gray[y:y+h, x:x+w]
    face_img = img[y:y+h, x:x+w]

    # 눈 인식하기
    eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 2, 0, (30,30), (80,80))
    print('찾은 눈의 수 : %d' % len(eyes))

    if len(eyes) == 2:  # 찾은 눈의 수가 2개일 경우
        face_center = (x+w//2, y+h//2)  # 얼굴의 중심점, // --> 몫을 취함
        # 눈의 중심점들 --> 현재 눈은 2개 찾음
        eye_centers = [(x+ex+ew//2, y+ey+eh//2) for ex,ey,ew,eh in eyes]
        # 보정된 이미지와 보정된 눈의 중심점 구함
        corr_img, corr_eye_centers = correct_image(img, face_center, eye_centers)

        # 머리카락/입술 영역 추출 --> faces[0]은 첫번째 얼굴
        # rois --> [hair upper, hair lower, lip, hair]
        rois = detect_object(face_center, faces[0])
        # 네개의 마스크 구하기, 입력은 네개의 영역과 수정된 이미지의 크기
        # 4개의 영역 --> 윗머리, 귀머리카락, 입술, 얼굴(머리카락)
        masks = make_masks(rois, corr_img.shape[:2])

        # 유사도 계산 및 출력
        sims = calc_histo(corr_img, rois, masks)
        # 남녀 성별 구분 --> 이미지 및 출력문으로 성별 표시
        gender = classify_gender(img, sims)
        # 출력 --> 얼굴, 눈, 입술 주위에 도형 표시
        display(corr_img, face_center, corr_eye_centers, rois, gender)

cv2.waitKey(0)
cv2.destroyAllWindows()