import cv2
import numpy as np
import os
from os import listdir
from os.path import isdir, isfile, join

# 얼굴 인식용 haarcascade 로딩
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')    


# 사용자 얼굴 학습
def train(name):
    data_path = 'faces/' + name + '/'
    # 실제로 존재하는 파일들의 이름을 리스트로 만들기
    face_pics = [f for f in listdir(data_path) if isfile(join(data_path,f))]
    
    Training_Data, Labels = [], []
    
    # 얼굴 데이터와 label을 학습 데이터 리스트에 추가
    for i, files in enumerate(face_pics):
        image_path = data_path + face_pics[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # 이미지가 아니면 패스
        if images is None:
            continue    
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)

    # print(Labels)
    # : 0~99 까지의 숫자가 있는 리스트

    # label이 없는 경우 학습할 데이터가 없으므로 종료
    if len(Labels) == 0:
        print("There is no data to train.")
        return None
    
    # 리스트 형식의 Labels를 numpy 배열로 변환
    Labels = np.asarray(Labels, dtype=np.int32)
    # 모델 생성
    model = cv2.face.LBPHFaceRecognizer_create()
    # 학습
    model.train(np.asarray(Training_Data), np.asarray(Labels))
    print(name + " : Model Training Complete!!!!!")

    #학습 모델 반환
    return model


# 여러 사용자 얼굴 학습
def trains():
    # faces 폴더의 하위 폴더를 학습
    data_path = 'faces/'
    # 폴더만 색출
    model_dirs = [f for f in listdir(data_path) if isdir(join(data_path,f))]
    
    #학습 모델 저장할 딕셔너리 생성
    models = {}

    # print(model_dirs)
    # [' jiae', 'hogyun', 'hyein', 'hyungsun', 'hyunho', 'seungju', 'sooyeon']


    # 각각의 폴더 및 사용자에 대한 얼굴 학습 진행
    for model in model_dirs:  # model = name
        print('model to train :' + model)
        # 모델 학습 시작
        result = train(model)
        # 모델 학습이 안 된 경우, 딕셔너리에 저장하지 않고 계속 진행
        if result is None:
            continue

        # 학습된 모델을 딕셔너리에 저장
        print('training model:' + model)
        models[model] = result
        

    # < print(models) 의 결과 >
    # {' jiae': < cv2.face.LBPHFaceRecognizer 00000199FF2BF150>, 
    # 'hogyun': < cv2.face.LBPHFaceRecognizer 00000199FF2BFD90>, 
    # 'hyein': < cv2.face.LBPHFaceRecognizer 00000199FF2BF1B0>, 
    # 'hyungsun': < cv2.face.LBPHFaceRecognizer 00000199FF2BFBD0>, 
    # 'hyunho': < cv2.face.LBPHFaceRecognizer 00000199FF2BFDF0>, 
    # 'seungju': < cv2.face.LBPHFaceRecognizer 00000199FF2BFBF0>, 
    # 'sooyeon': < cv2.face.LBPHFaceRecognizer 00000199FF2BFE10>}

    # 학습된 모델 딕셔너리 반환
    return models


# 얼굴 검출
def face_detector(img, size = 0.5):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)
        if faces is ():
            return img,[]
        # 검출된 얼굴에 사각형 표시 및 roi로 얼굴 부분만 추출
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
            roi = img[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200,200))
        return img,roi   # 검출된 좌표에 사각 박스 그리고(img), 검출된 부위를 잘라(roi) 전달


# 프로그램 실행 함수
def run(models):
    
    # 웹캠 시작
    cap = cv2.VideoCapture(0)
    
    while True:
        # 웹캠에서 한 프레임 읽기 
        ret, frame = cap.read()
        # 얼굴 검출 시도 
        image, face = face_detector(frame)
        try:            
            min_score = 999       # 가장 낮은 점수로 예측된 사람의 점수 저장
            min_score_name = ""   # 가장 낮은 점수로 예측된 사람의 이름 저장
            
            # 검출된 사진을 흑백으로 변환 
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # 위에서 학습한 모델로 예측시도
            for key, model in models.items():  # key: name
                result = model.predict(face)
                # 점수가 더 낮은 경우, 점수 및 이름 갱신            
                if min_score > result[1]:
                    min_score = result[1]
                    min_score_name = key
                    
            # min_score: 신뢰도. 0에 가까울수록 자신과 같다는 의미.    
            if min_score < 500:
                # ???? 0 ~ 100표시하려고 한 듯 
                confidence = int(100*(1-(min_score)/300))
                # 유사도 화면에 표시 
                display_string = str(confidence)+'% Confidence it is ' + min_score_name
                cv2.putText(image,display_string,(100,120), cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)
            
            # 신뢰도가 75 보다 크면 동일 인물로 간주해 UnLocked! 
            if confidence > 75:
                cv2.putText(image, "Unlocked : " + min_score_name, (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Face Cropper', image)
            else:
            # 75 이하면 타인으로 간주해 Locked!!! 
                cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Cropper', image)
        except:
            # 얼굴 검출이 안된 경우
            cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Face Cropper', image)
            pass

        if cv2.waitKey(1)==13:  # Enter 키를 누르면 웹캠 종료
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 모델 학습 시작
    models = trains()
    # 실행
    run(models)


