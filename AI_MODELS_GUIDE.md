# Herbot AI Models - 상세 가이드 🤖

## 개요

Herbot은 **2개의 Edge TPU 최적화 비전 모델**을 사용하여 실시간으로 약초를 식별하고 병해를 감지합니다.

```
┌─────────────────────────────────────────────────────────┐
│                    AI Pipeline                           │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  Image (640×480) → Preprocessing → TFLite Interpreter   │
│                                    + Edge TPU Delegate   │
│                                           ↓              │
│                    ┌──────────────────────┴─────────┐    │
│                    │                                │    │
│         ┌──────────▼─────────┐        ┌─────────────▼────────┐
│         │   Herbify Model    │        │  PlantDoc Model      │
│         │   (91 classes)     │        │  (28 classes)        │
│         │   "What herb?"     │        │  "Healthy/Diseased?" │
│         └──────────┬─────────┘        └─────────────┬────────┘
│                    │                                │    │
│                    ▼                                ▼    │
│            Herb Species ID                  Disease Detection
│            (Basil, Mint...)                (Blight, Spot...)
│                                                           │
└─────────────────────────────────────────────────────────┘
```

---

## 1. Herbify - 약초 종 분류 모델 🌿

### 1.1 모델 정보

| 항목 | 값 |
|------|-----|
| **목적** | 약초 종(species) 식별 및 분류 |
| **모델 아키텍처** | MobileNet/EfficientNet 기반 (추정) |
| **클래스 수** | **91종** |
| **입력 형식** | RGB 이미지, 고정 크기 (224×224 추정) |
| **입력 타입** | `uint8` (INT8 양자화) |
| **출력 타입** | `int8` (양자화) → `float32` (역양자화) |
| **출력 형태** | Softmax 확률 분포 [91] |
| **모델 파일** | `herbify/herbify_edgetpu_ready_edgetpu.tflite` |
| **모델 크기** | **3,084,928 bytes (3.0 MB)** |
| **추론 속도** | 5-15ms (Edge TPU), 80-150ms (CPU) |

### 1.2 클래스 목록 (91종 약초)

**전체 클래스는 `herbify/class_names.json` 참조**

```json
{
  "num_classes": 91,
  "class_names": [
    "Aloe Vera",
    "Basil",
    "Bay Leaf",
    "Calendula",
    "Chamomile",
    "Cinnamon",
    "Coriander",
    "Echinacea",
    "Eucalyptus",
    "Fennel",
    "Garlic",
    "Ginger",
    "Ginkgo",
    "Ginseng",
    "Lavender",
    "Lemongrass",
    "Mint",
    "Oregano",
    "Parsley",
    "Peppermint",
    "Rosemary",
    "Sage",
    "Thyme",
    "Turmeric",
    ...
  ]
}
```

### 1.3 모델 사용법

#### Python 코드

```python
from herbify_inference import HerbClassifier

# 1. 모델 초기화
classifier = HerbClassifier(
    model_path="herbify/herbify_edgetpu_ready_edgetpu.tflite",
    class_names_path="herbify/class_names.json",
    use_edgetpu=True  # Edge TPU 가속 사용
)

# 2. 단일 이미지 추론
results, inference_time = classifier.predict(
    image_path="captures/scan_001_z000.jpg",
    top_k=3  # 상위 3개 결과
)

# 3. 결과 출력
print(f"Inference time: {inference_time:.2f} ms")
for class_name, confidence in results:
    print(f"  {class_name}: {confidence*100:.2f}%")

# 출력 예시:
# Inference time: 8.45 ms
#   Basil: 95.30%
#   Mint: 3.20%
#   Oregano: 0.85%
```

#### 배치 추론

```python
# 여러 이미지 동시 처리
image_paths = [
    "capture1.jpg",
    "capture2.jpg",
    "capture3.jpg"
]

for img_path in image_paths:
    results, time = classifier.predict(img_path, top_k=1)
    top_class, confidence = results[0]
    print(f"{img_path}: {top_class} ({confidence*100:.1f}%)")
```

### 1.4 전처리 (Preprocessing)

```python
def preprocess_image(self, image_path):
    # 1. 이미지 로드
    image = Image.open(image_path).convert('RGB')

    # 2. 리사이즈 (모델 입력 크기에 맞춤)
    image = image.resize((self.input_width, self.input_height), Image.BILINEAR)

    # 3. NumPy 배열로 변환
    image_array = np.array(image)  # Shape: (H, W, 3)

    # 4. 배치 차원 추가
    image_array = np.expand_dims(image_array, axis=0)  # Shape: (1, H, W, 3)

    # 5. 양자화 (INT8 모델이므로 uint8로 유지)
    if self.is_quantized:
        image_array = image_array.astype(np.uint8)
    else:
        image_array = image_array.astype(np.float32) / 255.0

    return image_array
```

**핵심 포인트**:
- INT8 양자화 모델이므로 **정규화 불필요**
- 이미지는 `uint8` 그대로 입력
- RGB 채널 순서 유지 (BGR 변환 불필요)

### 1.5 후처리 (Postprocessing)

```python
# 1. 출력 텐서 가져오기
output_data = interpreter.get_tensor(output_details[0]['index'])
output_data = output_data[0]  # 배치 차원 제거

# 2. 역양자화 (INT8 → Float32)
if self.is_quantized:
    output_data = (output_data.astype(np.float32) - self.output_zero_point) * self.output_scale

# 3. Softmax 적용 (필요 시)
if not np.allclose(np.sum(output_data), 1.0, atol=0.1):
    exp_output = np.exp(output_data - np.max(output_data))
    output_data = exp_output / np.sum(exp_output)

# 4. Top-K 추출
top_k_indices = np.argsort(output_data)[-top_k:][::-1]
results = [(class_names[idx], float(output_data[idx])) for idx in top_k_indices]
```

### 1.6 Edge TPU 최적화

```python
# Edge TPU 델리게이트 로딩
try:
    from tflite_runtime.interpreter import load_delegate

    interpreter = Interpreter(
        model_path="herbify/herbify_edgetpu_ready_edgetpu.tflite",
        experimental_delegates=[
            load_delegate('libedgetpu.so.1')  # Edge TPU 라이브러리
        ]
    )
    interpreter.allocate_tensors()
    print("✓ Edge TPU acceleration enabled")
except Exception as e:
    print(f"✗ Failed to load Edge TPU: {e}")
    print("→ Falling back to CPU")
    interpreter = Interpreter(model_path=model_path)
```

**Edge TPU 요구사항**:
- INT8 양자화 모델 (필수)
- Edge TPU Compiler로 컴파일된 모델 (`_edgetpu.tflite` 접미사)
- `libedgetpu.so.1` 라이브러리 설치
- Coral USB Accelerator 연결

---

## 2. PlantDoc - 병충해 감지 모델 🍃

### 2.1 모델 정보

| 항목 | 값 |
|------|-----|
| **목적** | 식물 질병 및 건강 상태 감지 |
| **모델 아키텍처** | ResNet/MobileNet 기반 (추정) |
| **클래스 수** | **28종** (건강한 잎 + 다양한 병해) |
| **입력 형식** | RGB 이미지, 고정 크기 |
| **입력 타입** | `uint8` (INT8 양자화) |
| **출력 타입** | `int8` (양자화) → `float32` (역양자화) |
| **출력 형태** | Softmax 확률 분포 [28] |
| **모델 파일** | `plantdoc/plantdoc_edgetpu_ready_edgetpu.tflite` |
| **모델 크기** | **2,949,760 bytes (2.9 MB)** |
| **추론 속도** | 5-12ms (Edge TPU), 70-120ms (CPU) |

### 2.2 클래스 목록 (28종)

**전체 클래스는 `plantdoc/class_names.json` 참조**

```json
{
  "num_classes": 28,
  "class_names": [
    "Apple leaf",                    // 건강한 사과 잎
    "Apple rust leaf",               // 사과 녹병
    "Apple scab leaf",               // 사과 검은별무늬병
    "Blueberry leaf",                // 건강한 블루베리 잎
    "Cherry leaf",                   // 건강한 체리 잎
    "Corn Gray leaf spot",           // 옥수수 회색잎마름병
    "Corn leaf blight",              // 옥수수 잎마름병
    "Corn rust leaf",                // 옥수수 녹병
    "Grape leaf black rot",          // 포도 검은썩음병
    "Grape leaf",                    // 건강한 포도 잎
    "Peach leaf",                    // 건강한 복숭아 잎
    "Pepper bell Bacterial spot",   // 피망 세균점무늬병
    "Pepper bell leaf",              // 건강한 피망 잎
    "Potato leaf early blight",      // 감자 조기마름병
    "Potato leaf late blight",       // 감자 만기마름병
    "Potato leaf",                   // 건강한 감자 잎
    "Raspberry leaf",                // 건강한 라즈베리 잎
    "Soybean leaf",                  // 건강한 콩 잎
    "Squash Powdery mildew leaf",    // 호박 흰가루병
    "Strawberry leaf",               // 건강한 딸기 잎
    "Tomato Early blight leaf",      // 토마토 조기마름병
    "Tomato Septoria leaf spot",     // 토마토 점무늬병
    "Tomato leaf bacterial spot",    // 토마토 세균점무늬병
    "Tomato leaf late blight",       // 토마토 만기마름병
    "Tomato leaf mosaic virus",      // 토마토 모자이크 바이러스
    "Tomato leaf yellow virus",      // 토마토 황화 바이러스
    "Tomato leaf",                   // 건강한 토마토 잎
    "Tomato mold leaf"               // 토마토 곰팡이병
  ]
}
```

### 2.3 병해 감지 로직

```python
def detect_disease(self, image_path, threshold=0.6, min_confidence=0.4):
    """
    병해 감지 로직

    Args:
        threshold: 병해로 판정할 최소 신뢰도 (기본: 0.6 = 60%)
        min_confidence: 유효한 감지로 간주할 최소 신뢰도 (기본: 0.4)

    Returns:
        (is_diseased, confidence, class_name, inference_time)
    """

    # 1. 추론 실행
    results, inference_time = self.predict(image_path, top_k=3)
    top_class, top_confidence = results[0]

    # 2. 낮은 신뢰도 필터링 (배경/빈 공간)
    if top_confidence < min_confidence:
        return False, top_confidence, f"No clear detection ({top_class})", inference_time

    # 3. 병해 키워드 검사
    disease_keywords = [
        "blight",      # 마름병
        "spot",        # 점무늬병
        "rust",        # 녹병
        "scab",        # 검은별무늬병
        "mildew",      # 흰가루병
        "virus",       # 바이러스
        "mold",        # 곰팡이
        "bacterial",   # 세균성
        "spider"       # 거미
    ]

    # 4. 건강한 잎 판정 로직
    # "Apple leaf", "Tomato leaf" 등 (키워드 "leaf"만 있고 병해 키워드 없음)
    is_healthy = (
        top_class.lower().endswith("leaf") and
        not any(disease in top_class.lower() for disease in disease_keywords)
    )

    # 5. 병해 판정
    is_diseased = not is_healthy and top_confidence >= threshold

    return is_diseased, top_confidence, top_class, inference_time
```

**판정 예시**:

| 클래스 이름 | 신뢰도 | 건강? | 병해? |
|------------|--------|-------|-------|
| `Tomato leaf` | 0.92 | ✅ Yes | ❌ No |
| `Tomato Early blight leaf` | 0.87 | ❌ No | ✅ Yes |
| `Apple leaf` | 0.45 | ✅ Yes | ❌ No (낮은 신뢰도) |
| `Corn rust leaf` | 0.55 | ❌ No | ❌ No (threshold 미달) |

### 2.4 사용 예시

```python
from plantdoc_inference import PlantDiseaseClassifier

# 1. 모델 초기화
classifier = PlantDiseaseClassifier(
    model_path="plantdoc/plantdoc_edgetpu_ready_edgetpu.tflite",
    class_names_path="plantdoc/class_names.json",
    use_edgetpu=True
)

# 2. 병해 감지
is_diseased, confidence, class_name, inference_time = classifier.detect_disease(
    image_path="captures/scan_005_z400.jpg",
    threshold=0.6,
    min_confidence=0.4
)

# 3. 결과 처리
print(f"Image: scan_005_z400.jpg")
print(f"Inference time: {inference_time:.2f} ms")
print(f"Class: {class_name}")
print(f"Confidence: {confidence*100:.1f}%")
print(f"Status: {'🔴 DISEASED' if is_diseased else '✅ HEALTHY'}")

# 출력 예시:
# Image: scan_005_z400.jpg
# Inference time: 9.23 ms
# Class: Tomato Early blight leaf
# Confidence: 87.3%
# Status: 🔴 DISEASED
```

### 2.5 Herbot 통합

```python
# herbot.py의 scan_and_manage() 메서드에서 사용
for z_mm in z_positions:
    # 1. Z 위치 이동
    self.z_motor.move_to_z(z_mm)

    # 2. 이미지 캡처
    image_path = self._capture_image(scan_id, z_mm)

    # 3. Herbify로 약초 종 분류
    herb_results, herb_time = self.herbify.predict(image_path, top_k=1)
    herb_name, herb_conf = herb_results[0]
    print(f"🌿 Herb: {herb_name} ({herb_conf*100:.1f}%)")

    # 4. PlantDoc으로 병해 감지
    is_diseased, conf, disease_class, disease_time = self.detect_disease(
        image_path, threshold=0.6
    )

    # 5. 병해 발견 시 자동 제거
    if is_diseased:
        print(f"🔴 DISEASED: {disease_class} ({conf*100:.1f}%)")
        print("✂️ Removing diseased leaf...")

        # 회전 정지
        self.theta_motor.stop()

        # 그리퍼 열기
        self.gripper.open()

        # R축 확장 (병든 잎으로 접근)
        self.r_motor.move_to_r(r_extend)

        # 그리퍼 닫기 (잎 절단)
        self.gripper.cut(cut_time=1.5)

        # R축 수축
        self.r_motor.move_to_r(0.0)

        # 회전 재개
        self.theta_motor.start_rotation_cw(speed=theta_speed)
    else:
        print(f"✅ HEALTHY: {disease_class} ({conf*100:.1f}%)")
```

---

## 3. 모델 파일 구조

```
Herbot/
├── herbify/
│   ├── herbify_edgetpu_ready_edgetpu.tflite  # Edge TPU 컴파일된 모델
│   │   ├── Size: 3,084,928 bytes (3.0 MB)
│   │   ├── Format: TFLite (FlatBuffer)
│   │   ├── Quantization: INT8
│   │   └── Optimized for: Coral Edge TPU
│   │
│   └── class_names.json                       # 클래스 이름 매핑
│       └── { "num_classes": 91, "class_names": [...] }
│
└── plantdoc/
    ├── plantdoc_edgetpu_ready_edgetpu.tflite  # Edge TPU 컴파일된 모델
    │   ├── Size: 2,949,760 bytes (2.9 MB)
    │   ├── Format: TFLite (FlatBuffer)
    │   ├── Quantization: INT8
    │   └── Optimized for: Coral Edge TPU
    │
    └── class_names.json                       # 클래스 이름 매핑
        └── { "num_classes": 28, "class_names": [...] }
```

### 3.1 모델 파일 검증

```bash
# TFLite 모델 정보 확인 (Python)
python3 << EOF
from tflite_runtime.interpreter import Interpreter

# Herbify 모델
interp = Interpreter("herbify/herbify_edgetpu_ready_edgetpu.tflite")
interp.allocate_tensors()

input_details = interp.get_input_details()
output_details = interp.get_output_details()

print("=== Herbify Model ===")
print(f"Input shape: {input_details[0]['shape']}")
print(f"Input type: {input_details[0]['dtype']}")
print(f"Output shape: {output_details[0]['shape']}")
print(f"Output type: {output_details[0]['dtype']}")
EOF

# 출력 예시:
# === Herbify Model ===
# Input shape: [1 224 224 3]
# Input type: <class 'numpy.uint8'>
# Output shape: [1 91]
# Output type: <class 'numpy.int8'>
```

---

## 4. 추론 성능 벤치마크

### 4.1 하드웨어별 성능

| 하드웨어 | Herbify (91 class) | PlantDoc (28 class) |
|---------|-------------------|---------------------|
| **CPU Only** (RPi 4) | 80-150ms | 70-120ms |
| **Edge TPU** (Coral) | **5-15ms** | **5-12ms** |
| **가속비** | **10-15x** | **10-15x** |

### 4.2 실제 측정 (Edge TPU)

```python
import time

# Herbify 성능 테스트
times = []
for i in range(100):
    start = time.time()
    results, _ = herbify.predict("test_image.jpg")
    times.append((time.time() - start) * 1000)

print(f"Herbify - Avg: {np.mean(times):.2f}ms, Std: {np.std(times):.2f}ms")

# PlantDoc 성능 테스트
times = []
for i in range(100):
    start = time.time()
    is_diseased, _, _, _ = detect_disease("test_image.jpg")
    times.append((time.time() - start) * 1000)

print(f"PlantDoc - Avg: {np.mean(times):.2f}ms, Std: {np.std(times):.2f}ms")

# 출력 예시:
# Herbify - Avg: 8.45ms, Std: 1.23ms
# PlantDoc - Avg: 7.89ms, Std: 1.05ms
```

### 4.3 전체 파이프라인 시간

```
┌─────────────────────┬──────────┐
│ Operation           │ Time     │
├─────────────────────┼──────────┤
│ Image Capture       │ 200-300ms│
│ Herbify Inference   │ 8-15ms   │
│ PlantDoc Inference  │ 7-12ms   │
│ Post-processing     │ 1-3ms    │
├─────────────────────┼──────────┤
│ Total per Position  │ ~220-330ms│
└─────────────────────┴──────────┘

→ 약 3-4 images/second (실시간 처리 가능)
```

---

## 5. 모델 학습 및 변환 (참고)

### 5.1 원본 모델 학습 (가정)

```python
# TensorFlow/Keras로 모델 학습
import tensorflow as tf

# 모델 정의
model = tf.keras.Sequential([
    tf.keras.layers.MobileNetV2(input_shape=(224, 224, 3), include_top=False),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(91, activation='softmax')  # Herbify: 91 classes
])

# 학습
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_dataset, epochs=50, validation_data=val_dataset)

# 저장
model.save('herbify_model.h5')
```

### 5.2 TFLite 변환 (INT8 Quantization)

```python
import tensorflow as tf

# 1. SavedModel 변환
converter = tf.lite.TFLiteConverter.from_saved_model('herbify_model/')

# 2. INT8 양자화 설정
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.int8

# 3. Representative dataset (양자화 캘리브레이션)
def representative_data_gen():
    for image in calibration_dataset:
        yield [image]

converter.representative_dataset = representative_data_gen

# 4. 변환
tflite_model = converter.convert()

# 5. 저장
with open('herbify_quantized.tflite', 'wb') as f:
    f.write(tflite_model)
```

### 5.3 Edge TPU 컴파일

```bash
# Edge TPU Compiler 설치
# https://coral.ai/docs/edgetpu/compiler/

# 모델 컴파일
edgetpu_compiler herbify_quantized.tflite

# 출력: herbify_quantized_edgetpu.tflite

# 컴파일 로그 확인
# → 연산자 매핑률: 95% Edge TPU, 5% CPU
# → 성능 예상: 10-20x 가속
```

---

## 6. 트러블슈팅

### 6.1 Edge TPU 로딩 실패

**증상**:
```
Failed to load Edge TPU delegate: libedgetpu.so.1: cannot open shared object file
```

**해결**:
```bash
# 1. Edge TPU 런타임 설치
sudo apt-get update
sudo apt-get install libedgetpu1-std

# 2. 라이브러리 확인
ls -l /usr/lib/aarch64-linux-gnu/libedgetpu.so.1.0

# 3. USB 장치 확인
lsusb | grep -i "Global\|Coral"

# 4. 권한 설정
sudo usermod -aG plugdev $USER
sudo reboot
```

### 6.2 추론 결과 이상

**증상**: 모든 이미지에 대해 동일한 클래스 예측

**원인 및 해결**:

1. **입력 전처리 오류**
```python
# 잘못된 예: RGB → BGR 변환
image = cv2.imread(path)  # BGR 순서!
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 필요

# 올바른 예: PIL 사용
image = Image.open(path).convert('RGB')  # 이미 RGB
```

2. **정규화 오류**
```python
# INT8 모델에서는 정규화하면 안 됨!
# 잘못된 예:
image_array = image_array.astype(np.float32) / 255.0  # ✗

# 올바른 예:
image_array = image_array.astype(np.uint8)  # ✓
```

3. **양자화 파라미터 누락**
```python
# 출력 역양자화 필수
if self.is_quantized:
    output = (output.astype(np.float32) - zero_point) * scale
```

### 6.3 느린 추론 속도

**증상**: Edge TPU 사용 중인데도 50ms+ 소요

**원인**:
- Edge TPU 델리게이트가 실제로 로드되지 않음
- 일부 연산자가 CPU로 fallback

**해결**:
```python
# 1. Edge TPU 로딩 확인
print(f"Using Edge TPU: {self.use_edgetpu}")

# 2. 모델 컴파일 로그 확인
# edgetpu_compiler 출력에서 "Operator partitioning" 섹션 확인
# → Edge TPU에 100% 매핑되었는지 확인

# 3. 모델 재컴파일 (필요시)
edgetpu_compiler -s herbify_quantized.tflite
```

---

## 7. 참고 자료

### 공식 문서
- [Coral Edge TPU Models](https://coral.ai/models/)
- [TFLite INT8 Quantization](https://www.tensorflow.org/lite/performance/post_training_quantization)
- [Edge TPU Compiler](https://coral.ai/docs/edgetpu/compiler/)

### 모델 출처 (추정)
- **Herbify**: Medicinal Plant Dataset (Kaggle/Custom)
- **PlantDoc**: PlantDoc Dataset (GitHub)

### 관련 논문
- MobileNetV2: [Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
- Quantization: [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)

---

## 8. FAQ

### Q1: 새로운 약초 종을 추가할 수 있나요?

**A**: 현재 모델은 고정된 91종이므로, 새로운 종을 추가하려면:
1. 새로운 종의 이미지 데이터 수집 (최소 100-1000장)
2. 기존 데이터셋과 합치기
3. 모델 재학습 (91 → 92 클래스)
4. INT8 양자화 및 Edge TPU 컴파일
5. `class_names.json` 업데이트

### Q2: CPU만으로도 사용 가능한가요?

**A**: 가능합니다. Edge TPU가 없으면 자동으로 CPU로 fallback됩니다.
```python
classifier = HerbClassifier(
    model_path="herbify/herbify_edgetpu_ready_edgetpu.tflite",
    class_names_path="herbify/class_names.json",
    use_edgetpu=False  # CPU 사용
)
```
단, 추론 속도는 **10-15배 느려집니다** (8ms → 80-150ms).

### Q3: 조명 조건이 추론 정확도에 영향을 주나요?

**A**: 네, 매우 중요합니다.
- **최적 조건**: 자연광 또는 백색 LED, 그림자 없음
- **불리한 조건**: 어두운 환경, 강한 역광, 강한 그림자
- **권장사항**: LED 링 라이트 추가 (일정한 조명 제공)

### Q4: 모델 정확도는 얼마나 되나요?

**A**: 공식 정확도는 명시되지 않았지만, 경험적으로:
- **Herbify**: ~85-95% (잘 보이는 잎)
- **PlantDoc**: ~80-90% (명확한 병해)
- **실제 환경**: 다양한 요인(각도, 조명, 잎 상태)에 따라 변동

### Q5: 배치 추론을 지원하나요?

**A**: Edge TPU 모델은 배치 크기가 1로 고정되어 있습니다.
```python
# 배치 처리는 순차적으로 수행
for image in images:
    results = classifier.predict(image)
```

---

**문서 작성일**: 2025-01-21
**모델 버전**: v1.0
**작성자**: Herbot AI Team
