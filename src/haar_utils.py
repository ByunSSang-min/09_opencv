import cv2
import numpy as np

def preprocessing(fname):
    img = cv2.imread('../img/'+fname, cv2.IMREAD_COLOR)
    
    if img is None :
        return None, None
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img_gray = cv2.equalizeHist(img_gray)
    
    return img, img_gray

def correct_image(img, face_center, eye_centers):
    pt0, pt1 = eye_centers
    
    if pt0[0] > pt1[0]:
        pt0, pt1 = pt1, pt0
    dx, dy = np.subtract(pt1, pt0)
    
    angle = cv2.fastAtan2(float(dy), float(dx))
    
    rot_matrix = cv2.getRotationMatrix2D((float(face_center[0]), float(face_center[1])), angle, 1)
    size = img.shape[1::-1]
    img_correction = cv2.warpAffine(img, rot_matrix, size, cv2.INTER_CUBIC)
    
    eye_centers = np.expand_dims(eye_centers, axis=0)
    eye_correct_centers = cv2.transform(eye_centers, rot_matrix)
    eye_correct_centers = np.squeeze(eye_correct_centers, axis=0)
    
    return img_correction, eye_correct_centers


def define_roi(pt, size):
    return np.ravel((pt, size)).astype('int')

def detect_object(center, face):
    w, h = np.array(face[2:4])
    center = np.array(center)
    
    gap_face = np.multiply((w, h), (0.45, 0.65))
    pt_face_start = center - gap_face
    pt_face_end = center + gap_face
    hair = define_roi(pt_face_start, pt_face_end - pt_face_start)
    
    size = np.multiply(hair[2:4], (1, 0.4))
    hair_upper = define_roi(pt_face_start, size)
    hair_lower = define_roi(pt_face_end - size, size)
    
    gap_lip = np.multiply((w, h), (0.18, 0.10))
    lip_center = center +(0, int(h*0.3))
    
    pt_lip_start = lip_center - gap_lip
    pt_lip_end = lip_center + gap_lip
    
    lip = define_roi(pt_lip_start, pt_lip_end - pt_lip_start)
    
    return [hair_upper, hair_lower, lip, hair]

def draw_ellipse(img, roi, ratio, color, thickness=cv2.FILLED):
    x, y, w, h = roi
    center = (x+w//2, y+h//2)
    size = (int(w*ratio), int(h*ratio))
    cv2.ellipse(img, center, size, 0, 0, 360, color, thickness)
    return img

def make_masks(rois, shape):
    base_mask = np.full(shape, 255, np.uint8)
    
    hair_mask = draw_ellipse(base_mask, rois[3], 0.45, 0)
    cv2.imshow('hair_maks', hair_mask)
    
    lip_mask = draw_ellipse(np.copy(hair_mask), rois[2], 0.40, 255)
    cv2.imshow('lip_mask', lip_mask)
    
    masks = [hair_mask, hair_mask, lip_mask, ~lip_mask]
    masks = [mask[y:y+h, x:x+w] for mask, (x, y, w, h) in zip(masks, rois)]
    
    return masks

def calc_histo(img, rois, masks):
    bsize = (64, 64, 64)
    ranges = (0, 256, 0, 256, 0, 256)
    sub_imgs = [img[y:y+h, x:x+w] for x, y, w, h in rois]
    
    hists = [cv2.calcHist([sub_img], [0, 1, 2], mask, bsize, ranges, 3)
             for sub_img, mask in zip(sub_imgs, masks)]
    hists = [h/np.sum(h) for h in hists]
    
    sim_face_lip = cv2.compareHist(hists[2], hists[3], cv2.HISTCMP_CORREL)
    sim_hair = cv2.compareHist(hists[0], hists[1], cv2.HISTCMP_CORREL)
    return sim_face_lip, sim_hair

def classify_gender(img, sims):
    criteria = 0.25 if sims[0] > 0.2 else 0.1
    gender = 'Woman' if sims[1] > criteria else 'Man'
    
    print(gender+' : 입술-얼굴 유사도 %4.2f, 윗-귀밑머리 유사도 %4.2f' %(sims))
    return gender

def display(img, face_center, eye_centers, rois, gender):
    cv2.circle(img, face_center, 2, (0, 0, 255), 2)
    
    cv2.circle(img, tuple(eye_centers[0]), 20, (0, 255, 0), 2)
    cv2.circle(img, tuple(eye_centers[1]), 20, (0, 255, 0), 2)
    
    draw_ellipse(img, rois[2], 0.45, (0, 0, 255), 2)
    draw_ellipse(img, rois[3], 0.45, (0, 0, 255), 2)
    
    cv2.putText(img, gender, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 0), 3)
    cv2.imshow('final result', img)