import cv2
from os import listdir
from os.path import join, isfile, splitext


# 모든 xml 파일을 메모리에 로드하는 함수
def load_all_models(data_path):
    # 실제로 존재하는 파일들의 이름을 리스트로 만들기
    model_files = [f for f in listdir(data_path) if isfile(join(data_path, f)) and f.endswith('.xml')]
    models = {}
    for model_file in model_files:
        model_name = splitext(model_file)[0]
        model = cv2.face.LBPHFaceRecognizer_create()
        model.read(join(data_path, model_file))
        models[model_name] = model
    return models


# 얼굴 인식용 haarcascade 로딩
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# 얼굴 검출 함수
def face_detector(img, size=0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return img, []
    # 검출된 얼굴에 사각형 표시 및 roi로 얼굴 부분만 추출
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi = img[y:y + h, x:x + w]
        roi = cv2.resize(roi, (200, 200))
    
    # 검출된 좌표에 사각 박스 그리고(img), 검출된 부위를 잘라(roi) 반환
    return img, roi


# 메모리 모델 로드
data_path = 'models/'  # XML 파일이 저장된 디렉토리
models = load_all_models(data_path)


# 얼굴 인식 함수
def run(models):

    # 웹캠 시작
    cap = cv2.VideoCapture(0)

    while True:
        # 웹캠에서 한 프레임 읽기 
        ret, frame = cap.read()
        # 얼굴 검출 시도 
        image, face = face_detector(frame)
        try:
            min_score = 999          # 가장 낮은 점수로 예측된 사람의 점수 저장
            min_score_name = ""      # 가장 낮은 점수로 예측된 사람의 이름 저장

            # 검출된 사진을 흑백으로 변환
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # 위에서 학습한 모델로 예측시도
            for key, model in models.items():    # key: name
                result = model.predict(face)
                # 점수가 더 낮은 경우, 점수 및 이름 갱신 
                if min_score > result[1]:
                    min_score = result[1]
                    min_score_name = key

            # min_score: 신뢰도. 0에 가까울수록 자신과 같다는 의미
            if min_score < 500:
                # 0 ~ 100표시하려고 한 듯 
                confidence = int(100 * (1 - (min_score) / 300))
                # 유사도 화면에 표시 
                display_string = str(confidence) + '% Confidence it is ' + min_score_name
                cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (250, 120, 255), 2)
            
            # 신뢰도가 75 보다 크면 동일 인물로 간주해 UnLocked! 
            if confidence > 75:
                cv2.putText(image, "Unlocked : " + min_score_name, (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0),
                            2)
            else:
                # 75 이하면 타인으로 간주해 Locked!!! 
                cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Cropper', image)
        except:
            # 얼굴 검출이 안된 경우
            cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Face Cropper', image)

        if cv2.waitKey(1)==13:  # Enter 키를 누르면 웹캠 종료
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 실행
    run(models)
