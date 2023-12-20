# R-CNN
* R-CNN 논문 리뷰 및 구현
* 2020010219 양대윤
* 23년 인공지능2 기말 과제 제출

## 연구 주제
* 원문 : Rich feature hierarchies for accurate object detection and semantic segmentation
* 2012년 CNN(Convolution Neurel Network)이 발표된 이후, CNN은 이미지분류(Classification) 분야에서 표준처럼 사용되었습니다.
* 단, 2012년 Object Detection 분야에는 크게 적용되지는 못하고 있었고, 이러한 문제점을 보완하기 위해 R-CNN은 CNN을 Object Detection에 적용하고, 적은 양의 data로 모델을 학습하기 위해 연구되었습니다.
* R-CNN은 VOC2012에서 이전의 방법을 사용했을 때 보다 큰 성능 향상이 있었고, 그 후부터 Fast R-CNN, Faster R-CNN, Mask R-CNN 등 R-CNN을 수정, 보완하여 성능과 속도 등을 향상시킨 후속작들의 논문들이 나왔습니다.

## 배경지식
### Object Detection
* Object Detection은 여러 물체에 대해 어떤 물체인지 분류하는 Classification 문제입니다.
* 물체가 어디 있는지 박스를 통해(Bounding box) 위치 정보를 나타내는 Localization문제 이 두가지 문제를 둘다 해내야 하는 분야입니다.

### 2vs1 stage detector
* Object Detection은 물체의 위치를 찾는 Localization 문제와 물체를 식별하는 Classification 문제를 합한 문제입니다.
1. 1-stage Detector는 이 두 문제를 동시에 행하는 방법
2. 2-stage Detector는 이 두 문제를 순차적으로 행하는 방법
* 1-stage Detector는 비교적으로 빠르지만 정확도가 낮고, 2-stage Detector는 비교적으로 느리지만 정확도가 높다는 특징이 있습니다.
* 2-stage Detector는 CNN을 처음으로 적용시킨 R-CNN부터, Fast R-CNN, Faster R-CNN… 등의 R-CNN계열이 해당합니다.

## 논문설명
### Abstract
R-CNN : __Region with CNN features__
1. mAP(정밀도)를 이전에 나온 최고결과 보다 30% 이상 향상시킨 R-CNN모델에 대해 소개합니다.
2. 객체위치 파악 및 세분화를 위해 __bottom-up 방식의 region proposal(영역제안)__ 을 CNN에 적용합니다.
3. __데이터가 부족할 때 supervised-pre training(지도학습기반 사전훈련)__ 을 사용하고, __특정 도메인별 Fine-tuning(미세조정)__ 을 진행합니다.

### Instroduction
1. CNN이 본격적으로 이미지 분석에 활용되기 전에는 주로 SIFT나, HOG를 이용했습니다.
2. 2012년 AlexNet이 나오면서 ILSVRC에서 큰 성능 향상을 달성했습니다.
3. ILSVRC의 Classification result를 PASCAL VOC Challenge의 Object detection task에 확장하고자
연구되었습니다.
4. 본 논문에서는 CNN을 이용하여 기존 시스템에 비해 PASCAL VOC에서 우월한 성능을 보입니다.
5. 본 논문에서 집중한 두 문제상황은 __(1) deep network를 통한 localization, (2) 적은 양의 data로 모델을 학습시키는 것__ 입니다.

### R-CNN 수행 과정
1. 이미지를 입력합니다.
2. 약 2천개의 독립적인 Bottom-up region proposals를 추출합니다.
3. CNN을 이용하여 각각 region proposal마다 고정된 길이의 feature vector를 추출합니다. 각각의 region proposal의 크기가 다르므로 CNN에 넣기전에 크기를 맞춰줍니다.(Warped)
4. 각 region마다 해당 객체의 클래스 분류를 위해 linear SVM을 적용하여 분류합니다.

## R-CNN 구현
* R-CNN은 객체 검출(Object Detection)을 위한 모델 중 하나로, Region-based Convolutional Neural Network의 약자입니다.
* 이 모델은 이미지에서 물체를 탐지하고 해당 물체의 경계 상자를 찾는 데 사용됩니다.
* 아래는 R-CNN을 사용한 간단한 예제 코드입니다.
* 주의할 점은, 이 코드는 학습된 가중치를 사용하지 않고 단순히 모델 구조를 구현하는 예제입니다.
* 실제로 사용하려면 미리 학습된 가중치를 로드하고 fine-tuning이 필요합니다.

'''python
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

# R-CNN 모델 정의
def create_rcnn_model(input_shape=(224, 224, 3), num_classes=20):
    input_tensor = Input(shape=input_shape)

    # Feature extraction using convolutional layers
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_tensor)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    # Flatten and add fully connected layers
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=output)
    return model

# 모델 생성
rcnn_model = create_rcnn_model()

# 모델 요약 출력
rcnn_model.summary()
'''

* 이 코드는 간단한 Convolutional Neural Network을 사용하여 이미지를 특징으로 추출하고, 추출된 특징을 사용하여 물체를 분류하는 기본적인 R-CNN 모델을 생성합니다.
* 하지만, R-CNN은 계산 비용이 많이 들어서 실제로 사용하기 어렵습니다.
* 더 효율적인 모델로서 Fast R-CNN, Faster R-CNN, 그리고 최신의 모델인 EfficientDet을 고려하는 것이 좋습니다.

## 결론
* Object를 Localize하고, 분할하기 위하여 Region proposal을 CNN에 적용했습니다.
* __훈련데이터가 부족해도 Pre-training에 이은 Fine-tuning으로 큰 성능__ 을 보여주었습니다.
* 결론적으로 Region proposal에 대한 CNN 학습, SVM Classification, Bounding box regession을 통하여 이전의 Object Detection 방법론들보다 큰 성능을 보였습니다.
* 단, 학습이 여러 단계로 이루어져 있어 __긴 학습시간과 대용량의 저장공간이 요구된다__ 는 단점이 있습니다.
