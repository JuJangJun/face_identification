# face_identification
얼굴인식 알고리즘

<hr>

## Facial-Recognition
한 명의 얼굴을 학습하여 잠금여부를 판단하는 프로젝트

### - Facial_Recognition_Part1.py
  같은 디렉토리에 'faces' 폴더를 만들고 해당 파일을 실행한다.<br>
  노트북 웹캠이 켜지고, 프레임별로 사진을 전처리하여 저장한다.<br>
  이때, 눈코입이 모두 보여야 사진을 저장한다.<br>
  100장이 저장된다면 실행이 끝난다.

### - Facial_Recognition_Part2.py
'faces' 폴더에 있는 사진을 학습한다.

### - Facial_Recognition_Part3.py
노트북 웹캠을 켠 후 얼굴이 어디 있는지 인식한다.<br>
학습한 모델을 불러와서 인식한 얼굴이 학습한 얼굴인지 확인한다.<br>
학습한 얼굴이라면 'unlocked'를 웹캠 창에 띄우고, 아니라면 'locked'를 띄운다.

### - haarcascade_frontalface_default.xml
얼굴인식 모델을 위한 xml파일.

<hr>

## Facial_Recognition_plus
여러 명의 얼굴을 학습하여 누구인지 판단하는 프로젝트

### - Facial_Recognition_Part1.py
같은 디렉토리에 'faces' 폴더를 만들고 해당 파일을 실행한다.<br>
지금부터 저장할 얼굴이 누구의 것인지 입력하도록 한다.<br>
'faces' 폴더 안에 입력받은 값으로 새 폴더를 만든다.<br>
노트북 웹캠이 켜지고, 프레임별로 사진을 전처리하여 저장한다.<br>
이때, 눈코입이 모두 보여야 사진을 저장한다.<br>
100장이 저장된다면 실행이 끝난다.

### - Facial_Recognition_Part2.py
'faces' 폴더에 있는 사진을 학습한다.<br>
이때, 각 폴더의 이름을 얼굴과 연결하여 학습한다.<br>
노트북 웹캠을 켠 후 얼굴이 어디 있는지 인식한다.<br>
학습한 모델을 불러와서 인식한 얼굴이 학습한 얼굴인지 확인한다.<br>
학습한 얼굴이라면 'unlocked'를 웹캠 창에 띄우고, 아니라면 'locked'를 띄운다.

### - haarcascade_frontalface_default.xml
얼굴인식 모델을 위한 xml파일.

<hr>

## Facial_Recognition_final
'Facial_Recognition_plus'를 모듈화한 프로젝트<br>
Facial_Recognition_Part2.py를 학습하는 코드와 학습한 모델을 불러오는 코드로 분리하였다.

### - face_collect.py
'Facial_Recognition_plus'폴더의 'Facial_Recognition_Part1.py'과 동일<br><br>

같은 디렉토리에 'faces' 폴더를 만들고 해당 파일을 실행한다.<br>
지금부터 저장할 얼굴이 누구의 것인지 입력하도록 한다.<br>
'faces' 폴더 안에 입력받은 값으로 새 폴더를 만든다.<br>
노트북 웹캠이 켜지고, 프레임별로 사진을 전처리하여 저장한다.<br>
이때, 눈코입이 모두 보여야 사진을 저장한다.<br>
100장이 저장된다면 실행이 끝난다.

### - face_train.py
'Facial_Recognition_plus/Facial_Recognition_Part2.py'의 사진을 학습하여 학습한 모델을 저장하는 부분<br><br>

'faces' 폴더에 있는 사진을 학습한다.<br>
이때, 각 폴더의 이름을 얼굴과 연결하여 학습한다.<br>
'models' 폴더를 생성한 뒤, 각 이름 별로 학습한 모델을 저장한다.

### - face_recognition_on_memory.py
'Facial_Recognition_plus/Facial_Recognition_Part2.py'의 저장된 모델을 실행하여 인식하는 부분<br><br>

저장된 학습 모델을 불러와서 변수에 저장한다.<br>
노트북 웹캠을 켠 후 얼굴이 어디 있는지 인식한다.<br>
학습한 모델을 불러와서 인식한 얼굴이 학습한 얼굴인지 확인한다.<br>
학습한 얼굴이라면 'unlocked'를 웹캠 창에 띄우고, 아니라면 'locked'를 띄운다.

### - haarcascade_frontalface_default.xml
얼굴인식 모델을 위한 xml파일.

