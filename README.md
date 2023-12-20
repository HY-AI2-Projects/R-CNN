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

# ResNet50을 활용한 R-CNN 객체 검출

이 [프로젝트](https://github.com/HY-AI2-Projects/R-CNN/blob/main/R_CNN.ipynb)는 ResNet50을 사용하여 R-CNN(Region-Based Convolutional Neural Network)을 통한 객체 검출을 제공합니다. 객체 검출 결과를 히트맵으로 시각화하여 눈으로 확인할 수 있습니다.

## 주요 기능

### 1. 이미지 객체 검출

- ResNet50을 기반으로 한 사전 학습된 모델을 사용하여 이미지에서 다양한 객체를 식별합니다.
- 클래스 확률을 계산하고 가장 확률이 높은 클래스의 객체를 검출합니다.

### 2. 히트맵 시각화

- 객체 검출에 사용된 ResNet50의 중간 레이어에서 생성된 히트맵을 생성합니다.
- 히트맵은 해당 객체의 위치를 강조하여 시각적으로 확인할 수 있습니다.

## 사용 방법

1. 필요한 종속성을 설치합니다:

```python
    pip install -r requirements.txt
```
2. main 함수에서 이미지 URL을 교체합니다

```python
    #메인 함수
    def main():
        # 직접 이미지 링크로 교체하세요
        image_url = 'https://i.imgur.com/YOUR_IMAGE_ID.jpg'
        img_array = download_and_preprocess_image(image_url)

        if img_array is not None:
            model = build_rcnn_model()
            heatmap = detect_objects(img_array, model)
            visualize_heatmap(heatmap, 'path/to/your/image.jpg')  # 로컬 이미지의 경로로 변경

        if _name__ == "_main__":
        main()
```

3. 스크립트를 실행합니다

```python
    python your_script_name.py
```

## 실패점
* 이미지 URL이 올바르게 입력되어야 합니다. 현재의 예시에서는 https://imgur.com/YOUR_IMAGE_ID가 존재하지 않아 404 오류가 발생합니다. 직접 사용하실 이미지의 링크로 교체해주세요.
  
## 주요 기능
1. 이미지 다운로드 및 전처리

* 지정된 URL에서 이미지를 다운로드하고 ResNet50 모델에 입력할 수 있는 형식으로 전처리합니다.
2. R-CNN 모델 구성

* 사전 학습된 ResNet50 모델을 사용하여 객체 검출에 활용합니다.
* 모델의 마지막 레이어를 제거하고 새로운 출력 레이어를 추가합니다.
3. 객체 검출 함수

* 주어진 이미지에서 객체를 검출하고 해당 객체의 히트맵을 생성합니다.
4. 시각화 함수

* 객체 검출 결과와 원본 이미지를 비교하여 히트맵을 시각화합니다.
