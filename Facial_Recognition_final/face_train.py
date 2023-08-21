import cv2
import numpy as np
import os
from os import listdir
from os.path import isdir, isfile, join

# 얼굴 인식용 haarcascade 로딩
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# models 폴더 만들기
if not os.path.exists('models'):
    os.makedirs('models')


# 사용자 얼굴 학습 함수
def train(name):
    data_path = 'faces/' + name + '/'
    # 실제로 존재하는 파일들의 이름을 리스트로 만들기
    face_pics = [f for f in listdir(data_path) if isfile(join(data_path, f))]

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


# 여러 사용자 얼굴 학습 함수
def trains():
    # faces 폴더의 하위 폴더를 학습
    data_path = 'faces/'
    # 폴더만 색출
    model_dirs = [f for f in listdir(data_path) if isdir(join(data_path, f))]
    
    #학습 모델 저장할 딕셔너리 생성
    models = {}

    # 각각의 폴더 및 사용자에 대한 얼굴 학습 진행
    for model in model_dirs:
        print('model to train :' + model)
        # 모델 학습 시작
        result = train(model)
        # 모델 학습이 안 된 경우, 딕셔너리에 저장하지 않고 계속 진행
        if result is None:
            continue

        # 학습된 모델 저장
        print('training model:' + model)
        models[model] = result
        result.write(f'models/{model}.xml')

    # 학습된 모델 딕셔너리 반환
    return models


# 모델 학습 시작
if __name__ == "__main__":
    models = trains()
