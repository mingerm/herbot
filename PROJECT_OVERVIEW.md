# Herbot - Herb Management Robot 🌿

## 프로젝트 개요

Herbot은 **약초 관리 자동화 로봇**으로, 3축 원통 좌표계(Cylindrical Coordinate System)를 사용하여 식물을 스캔하고, AI를 통해 약초 종을 분류하며, 병든 잎을 자동으로 감지하여 제거하는 시스템입니다.

---

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                        HERBOT SYSTEM                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────┐ │
│  │  Hardware   │    │   AI Models  │    │   Interface    │ │
│  │  Control    │◄──►│  (Edge TPU)  │◄──►│   (Web/CLI)    │ │
│  └─────────────┘    └──────────────┘    └────────────────┘ │
│        │                    │                     │          │
│        ▼                    ▼                     ▼          │
│  • 3-Axis Robot      • Herbify (91)       • Streamlit UI   │
│  • Camera           • PlantDoc (28)       • CLI Commands    │
│  • Gripper                                                   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. 하드웨어 시스템 (3-Axis Cylindrical Robot)

### 좌표계: Cylindrical Coordinates (Z, R, θ)

```
                    ↑ Z-axis (Height)
                    │
                    │  ┌─────┐ ← Gripper
                    │  │     │
              ┌─────┴──┴─────┐
              │   Carriage   │ ← R-axis (Linear Actuator)
              └──────┬───────┘
                     │
              ───────┴──────── Belt Drive
                     │
                     │ NEMA17 Stepper
                     │
         ╔═══════════╩═══════════╗
         ║    Rotating Base      ║ ← θ-axis (DC Motor)
         ║      🌿 Plant 🌿      ║
         ╚═══════════════════════╝
```

### 축(Axis) 구성

| 축 | 하드웨어 | 범위 | 용도 |
|---|---------|------|------|
| **Z-axis** | NEMA17 Stepper Motor (Belt-driven) | 0~750mm | 높이 조절 (캐리지 상하 이동) |
| **R-axis** | Linear Actuator (TB6612 driver) | 0~50mm | 반경 조절 (그리퍼 전후 확장) |
| **θ-axis** | DC Motor (TB6612 driver) | 0~360° | 회전 조절 (식물 베이스 회전) |
| **Gripper** | Servo Motor (SG90) | 0~180° | 잎 절단 (개폐 동작) |

### GPIO 핀 맵

```python
# Z-axis (Stepper Motor)
Z_ENABLE = 2
Z_STEP = 3
Z_DIR = 4

# R-axis (Linear Actuator)
R_ENABLE = 12
R_IN1 = 18
R_IN2 = 15
R_STBY = 23

# θ-axis (DC Motor)
THETA_ENABLE = 25
THETA_IN1 = 7
THETA_IN2 = 8
THETA_STBY = 1

# Gripper (Servo)
SERVO_PIN = 13
```

---

## 2. AI 시스템 (Edge TPU Accelerated)

### 2.1 Herbify - 약초 종 분류 모델

- **목적**: 약초의 종(species) 식별
- **클래스 수**: 91종
- **입력**: RGB 이미지 (224×224 or similar)
- **출력**: Softmax 확률 분포
- **추론 속도**: ~5-15ms (Edge TPU)
- **모델 파일**: `herbify/herbify_edgetpu_ready_edgetpu.tflite` (3.0 MB)

**사용 예시**:
```python
herb_classifier = HerbClassifier(
    model_path="herbify/herbify_edgetpu_ready_edgetpu.tflite",
    class_names_path="herbify/class_names.json",
    use_edgetpu=True
)
results, time = herb_classifier.predict(image_path, top_k=3)
# results: [('Basil', 0.95), ('Mint', 0.03), ...]
```

### 2.2 PlantDoc - 식물 병해 감지 모델

- **목적**: 식물의 질병 및 건강 상태 감지
- **클래스 수**: 28종 (건강한 잎 + 다양한 병해)
- **입력**: RGB 이미지
- **출력**: Softmax 확률 분포
- **추론 속도**: ~5-15ms (Edge TPU)
- **모델 파일**: `plantdoc/plantdoc_edgetpu_ready_edgetpu.tflite` (2.9 MB)

**병해 감지 로직**:
```python
# 건강한 잎: "Apple leaf", "Tomato leaf" 등 (키워드 "leaf"만 포함)
# 병든 잎: "blight", "spot", "rust", "scab", "mildew" 등 키워드 포함
is_diseased = confidence > threshold and disease_keyword_detected
```

**사용 예시**:
```python
disease_classifier = PlantDiseaseClassifier(
    model_path="plantdoc/plantdoc_edgetpu_ready_edgetpu.tflite",
    class_names_path="plantdoc/class_names.json",
    use_edgetpu=True
)
is_diseased, conf, class_name, time = detect_disease(image_path, threshold=0.6)
# is_diseased: True/False, conf: 0.87, class_name: "Tomato Early blight"
```

### Edge TPU 가속

- **라이브러리**: `tflite_runtime` + `libedgetpu.so.1`
- **델리게이트 로딩**:
```python
interpreter = Interpreter(
    model_path=model_path,
    experimental_delegates=[load_delegate('libedgetpu.so.1')]
)
```
- **성능**: CPU 대비 **10-20배 빠른 추론 속도**

---

## 3. 파이프라인 (Workflow)

### 3.1 전체 시스템 플로우

```
┌──────────────┐
│   START      │
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│  1. Home All Axes    │ ← Z, R, θ 축 홈 포지션으로 이동
│     - Z → 0mm        │
│     - R → 0mm        │
│     - θ → 0°         │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│  2. Start Scanning                   │
│     - Rotate plant (θ-axis)          │ ← DC 모터로 식물 천천히 회전
│     - Move Z-axis in steps           │ ← Z축을 100mm 간격으로 이동
│     - Capture image at each position │ ← 각 위치에서 카메라 캡처
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│  3. AI Inference (for each image)    │
│     ┌─────────────────────────────┐  │
│     │ Herbify Inference           │  │ ← 약초 종 분류
│     │ → Species: "Basil" (95%)    │  │
│     └─────────────────────────────┘  │
│     ┌─────────────────────────────┐  │
│     │ PlantDoc Inference          │  │ ← 병해 감지
│     │ → Disease: "Early blight"   │  │
│     │ → Confidence: 87%           │  │
│     └─────────────────────────────┘  │
└──────┬───────────────────────────────┘
       │
       ├─ Healthy? ──┐
       │             ▼
       │      ┌──────────────┐
       │      │  Continue    │
       │      │  Scanning    │
       │      └──────────────┘
       │
       ▼ Diseased?
┌──────────────────────────────────────┐
│  4. Automatic Leaf Removal           │
│     - Stop rotation                  │
│     - Open gripper                   │
│     - Extend R-axis (to leaf)        │ ← 병든 잎으로 접근
│     - Close gripper (cut)            │ ← 잎 절단
│     - Retract R-axis                 │
│     - Resume rotation                │
└──────┬───────────────────────────────┘
       │
       ▼
┌──────────────────────────┐
│  5. Return to Home       │
└──────┬───────────────────┘
       │
       ▼
┌──────────────────────────┐
│  6. Generate Report      │
│     - Total images       │
│     - Diseases detected  │
│     - Cuts performed     │
└──────────────────────────┘
```

### 3.2 스캔 모드 상세 (scan_and_manage)

**파라미터**:
- `scan_duration`: 전체 스캔 시간 (기본: 60초)
- `z_min` / `z_max`: Z축 스캔 범위 (기본: 0~700mm)
- `z_step`: Z축 스텝 크기 (기본: 100mm → 8개 포인트)
- `theta_speed`: 회전 속도 (기본: 3% → 매우 느린 회전)
- `r_extend`: 병해 감지 시 확장 거리 (기본: 30mm)
- `disease_threshold`: 병해 감지 임계값 (기본: 0.6 = 60%)

**스캔 패턴**:
```
Time: 0s ─────────────────► 60s
       │                      │
Z=700  ●→●→●→●→●→●→●→●        │ (8 points, up)
       ▼                      │
Z=0    ●←●←●←●←●←●←●←●        │ (8 points, down)
       ▼                      │
Z=700  ●→●→●→●→●→●→●→●        │ (repeat...)

θ: Continuous slow rotation (3% speed)
```

**각 스캔 포인트에서**:
1. Z축을 목표 높이로 이동
2. 100ms 대기 (안정화)
3. 카메라로 이미지 캡처 (`libcamera-still`)
4. Herbify로 약초 종 분류
5. PlantDoc으로 병해 감지
6. 병해 감지 시:
   - 회전 중지
   - 그리퍼 열기
   - R축 확장 (30mm)
   - 그리퍼 닫기 (1.5초 절단)
   - R축 수축
   - 회전 재개

**출력 데이터**:
```python
{
    'total_images': 34,
    'diseased_detected': 3,
    'cuts_performed': 3,
    'scan_positions': [
        {
            'scan_id': 1,
            'z_mm': 0,
            'time': 0.5,
            'image': 'captures/scan_001_z000_20251114_190304.jpg',
            'herb_name': 'Basil',
            'herb_confidence': 0.95,
            'disease_class': 'Healthy',
            'disease_confidence': 0.82,
            'diseased': False
        },
        ...
    ]
}
```

---

## 4. 파일 구조

```
Herbot/
├── herbot.py                    # 메인 로봇 제어 시스템 (1074 lines)
│   ├── StepperMotorZ            # Z축 제어 클래스
│   ├── LinearActuatorR          # R축 제어 클래스
│   ├── DCMotorTheta             # θ축 제어 클래스
│   ├── ServoGripper             # 그리퍼 제어 클래스
│   └── Herbot                   # 통합 제어 클래스
│
├── herbify_inference.py         # Herbify AI 모델 (약초 분류)
│   └── HerbClassifier           # 91종 약초 분류
│
├── plantdoc_inference.py        # PlantDoc AI 모델 (병해 감지)
│   └── PlantDiseaseClassifier   # 28종 병해 감지
│
├── web_ui.py                    # Streamlit 웹 인터페이스 (543 lines)
│   ├── Manual Control           # 수동 제어 탭
│   ├── Camera & Scans           # 스캔 실행 탭
│   ├── Gallery                  # 이미지 갤러리
│   └── Logs                     # 로그 출력
│
├── camera_capture.py            # 카메라 캡처 유틸리티
├── analyze_recent_scans.py      # 스캔 결과 분석
│
├── herbify/
│   ├── herbify_edgetpu_ready_edgetpu.tflite  # Edge TPU 모델 (3.0 MB)
│   └── class_names.json                       # 91종 클래스 이름
│
├── plantdoc/
│   ├── plantdoc_edgetpu_ready_edgetpu.tflite # Edge TPU 모델 (2.9 MB)
│   └── class_names.json                       # 28종 클래스 이름
│
├── captures/                    # 캡처된 이미지 저장소
│   ├── scan_001_z000_*.jpg
│   ├── scan_002_z100_*.jpg
│   └── ...
│
├── tests/                       # 하드웨어 테스트 스크립트
│   ├── stepper_simple.py
│   ├── linear_actuator_safe.py
│   ├── servo_gripper_test.py
│   ├── tb6612_test.py
│   └── test_tpu.py
│
├── stepper_config.json          # Z축 캘리브레이션 설정
├── stepper_state.json           # Z축 현재 위치 저장
├── actuator_state.json          # R축 현재 위치 저장
├── requirements_web.txt         # 웹 UI 의존성
└── run_web_ui.sh               # 웹 UI 실행 스크립트
```

---

## 5. 사용 방법

### 5.1 CLI 명령어

```bash
# 1. 모든 축 홈 포지션으로 이동
python3 herbot.py home

# 2. 수동 이동 (Z축)
python3 herbot.py move --z 200 --z-speed 1000

# 3. 수동 이동 (R축 - raw control)
python3 herbot.py actuator extend --duration 3.0
python3 herbot.py actuator retract --duration 5.0

# 4. 수동 이동 (θ축)
python3 herbot.py move --theta 5  # 5초간 회전

# 5. 그리퍼 제어
python3 herbot.py gripper open
python3 herbot.py gripper close
python3 herbot.py gripper cut

# 6. 단순 스캔 (회전만)
python3 herbot.py scan --duration 60 --speed 3

# 7. 특정 잎 접근 (좌표 기반)
python3 herbot.py approach --z 200 --theta 5 --r 40 --cut

# 8. 전체 관리 스캔 (AI + 자동 제거)
python3 herbot.py manage --duration 60 --z-step 100 --threshold 0.6

# 9. 상태 확인
python3 herbot.py status
```

### 5.2 Web UI

```bash
# 웹 UI 실행
./run_web_ui.sh
# 또는
streamlit run web_ui.py

# 브라우저에서 접속: http://<raspberry-pi-ip>:8501
```

**Web UI 기능**:
- 🎮 **Control Tab**: 수동 축 제어
- 📸 **Camera & Scans Tab**: 전체 스캔 실행 및 설정
- 🖼️ **Gallery Tab**: 캡처된 이미지 브라우징
- 📋 **Logs Tab**: 명령 출력 및 시스템 정보

### 5.3 Python API

```python
from herbot import Herbot

# 초기화
robot = Herbot()

# 홈 이동
robot.home_all()

# 특정 위치로 이동
robot.move_to_position(z_mm=200, r_mm=30, theta_duration=5)

# 스캔 실행
results = robot.scan_and_manage(
    scan_duration=60,
    z_step=100,
    disease_threshold=0.6
)

# 결과 출력
print(f"Total images: {results['total_images']}")
print(f"Diseases detected: {results['diseased_detected']}")
print(f"Cuts performed: {results['cuts_performed']}")

# 정리
robot.cleanup()
```

---

## 6. 데이터 플로우

### 6.1 이미지 캡처

```
Camera (Raspberry Pi Camera Module)
    ↓ libcamera-still --width 640 --height 480
captures/scan_001_z000_20251114_190304.jpg
```

### 6.2 AI 추론 플로우

```
Image (640×480 JPG)
    ↓ PIL.Image.open()
    ↓ .resize((224, 224))  # Model input size
    ↓ np.array() → uint8
    ↓
┌──────────────────────────────┐
│  TFLite Interpreter          │
│  + Edge TPU Delegate         │
│  (libedgetpu.so.1)          │
└──────────────────────────────┘
    ↓
Output tensor (quantized int8)
    ↓ Dequantize (scale, zero_point)
    ↓ Softmax
    ↓
Probabilities [0.95, 0.03, 0.01, ...]
    ↓
Top-K predictions:
  1. Basil: 95%
  2. Mint: 3%
  3. Oregano: 1%
```

### 6.3 상태 저장 (State Persistence)

```json
// stepper_state.json
{
  "current_step": 3200,
  "is_homed": true
}

// actuator_state.json
{
  "current_position_mm": 0.0,
  "is_homed": true
}

// stepper_config.json
{
  "steps_per_mm": 4.27,
  "max_position_mm": 750.0
}
```

---

## 7. 성능 및 통계

### 추론 성능 (Edge TPU)

| 모델 | 클래스 수 | 모델 크기 | CPU 추론 | Edge TPU 추론 | 가속비 |
|------|-----------|-----------|----------|---------------|--------|
| Herbify | 91 | 3.0 MB | ~80-150ms | ~5-15ms | **10-15x** |
| PlantDoc | 28 | 2.9 MB | ~70-120ms | ~5-12ms | **10-15x** |

### 스캔 통계 (실제 데이터 기준)

- **총 캡처 이미지**: 134개
- **스캔 세션**: 6회
- **Z축 포인트**: 8-16개 (100mm 간격)
- **평균 스캔 시간**: 60초
- **이미지당 처리 시간**: 약 1-2초 (캡처 + 2회 AI 추론)

---

## 8. 핵심 기술 스택

### 하드웨어
- **플랫폼**: Raspberry Pi 4 (4GB RAM)
- **AI 가속**: Google Coral Edge TPU USB Accelerator
- **카메라**: Raspberry Pi Camera Module V2
- **모터**: NEMA17 Stepper, TB6612 Motor Driver, SG90 Servo
- **GPIO**: RPi.GPIO library

### 소프트웨어
- **언어**: Python 3.11
- **AI 프레임워크**: TensorFlow Lite Runtime
- **가속 라이브러리**: libedgetpu.so.1
- **이미지 처리**: PIL (Pillow), NumPy
- **웹 프레임워크**: Streamlit
- **카메라**: libcamera-still

### AI 모델
- **양자화**: INT8 Post-Training Quantization
- **컴파일러**: Edge TPU Compiler
- **입력 형식**: uint8 (quantized)
- **출력 형식**: int8 (quantized) → float32 (dequantized)

---

## 9. 주요 특징

### ✅ 장점

1. **3축 원통 좌표계**
   - 식물 전체를 효율적으로 커버
   - Z축 범위: 750mm (대형 식물 지원)
   - 360° 회전으로 전방향 스캔

2. **실시간 AI 추론**
   - Edge TPU 가속으로 5-15ms 추론
   - 2개 모델 동시 사용 (종 분류 + 병해 감지)
   - 오프라인 동작 가능

3. **자동화된 관리**
   - 병든 잎 자동 감지 및 제거
   - 스캔 중 실시간 처리
   - 상태 저장으로 중단 후 재개 가능

4. **사용자 친화적 인터페이스**
   - CLI + Web UI 이중 인터페이스
   - 실시간 갤러리 및 로그
   - 직관적인 제어 패널

### ⚠️ 제한사항

1. **R축 위치 추정**
   - Linear actuator는 엔코더 없음
   - 시간 기반 위치 추정 (부정확할 수 있음)
   - 정기적인 홈 이동 필요

2. **병해 감지 정확도**
   - 조명 조건에 영향을 받음
   - 특정 각도에서만 정확
   - False positive 가능 (임계값 조정 필요)

3. **물리적 제약**
   - 식물 크기: 최대 750mm 높이
   - 잎 도달 범위: 50mm 반경
   - 회전 속도: 매우 느림 (3% 속도)

---

## 10. 확장 가능성

### 향후 개선 방향

1. **하드웨어**
   - R축 엔코더 추가 (위치 피드백)
   - 다중 카메라 (스테레오 비전)
   - LED 조명 시스템 (일정한 조명 조건)

2. **AI**
   - 객체 감지 모델 추가 (YOLO 등)
   - 세분화 모델 (병든 부분만 정확히 식별)
   - 성장 추적 (시계열 데이터)

3. **기능**
   - 물 주기 자동화
   - 비료 투입 시스템
   - 원격 모니터링 (IoT)

---

## 11. 트러블슈팅

### 일반적인 문제

1. **Edge TPU 인식 안 됨**
```bash
# USB 장치 확인
lsusb | grep -i "Global\|Coral"

# udev 규칙 확인
ls -l /etc/udev/rules.d/99-edgetpu-accelerator.rules

# 재부팅
sudo reboot
```

2. **Z축 위치 불일치**
```bash
# 홈 이동으로 리셋
python3 herbot.py home

# 또는 상태 파일 삭제
rm stepper_state.json
```

3. **카메라 오류**
```bash
# 카메라 연결 확인
libcamera-hello --list-cameras

# 권한 확인
sudo usermod -aG video $USER
```

4. **Web UI 접속 안 됨**
```bash
# 포트 확인
netstat -tuln | grep 8501

# 방화벽 확인
sudo ufw allow 8501
```

---

## 12. 참고 자료

### 관련 문서
- [Coral Edge TPU Documentation](https://coral.ai/docs/)
- [TensorFlow Lite Documentation](https://www.tensorflow.org/lite)
- [Raspberry Pi Camera Documentation](https://www.raspberrypi.com/documentation/accessories/camera.html)

### 모델 출처
- **Herbify**: Medicinal Herb Classification Dataset
- **PlantDoc**: Plant Disease Detection Dataset

### 의존성
```txt
# requirements_web.txt
streamlit>=1.28.0
Pillow>=9.0.0
numpy>=1.24.0
tflite-runtime>=2.14.0
```

---

## 📝 요약

Herbot은 **3축 로봇 + AI + 자동화**를 결합한 **스마트 약초 관리 시스템**입니다.

- **3축 제어**: Z(높이), R(반경), θ(회전)로 식물 전체 스캔
- **AI 추론**: Herbify(종 분류) + PlantDoc(병해 감지)
- **Edge TPU**: 실시간 추론 (5-15ms)
- **자동 관리**: 병든 잎 자동 감지 및 제거
- **웹 인터페이스**: Streamlit 기반 직관적 제어

**핵심 파이프라인**: 스캔 → AI 추론 → 병해 감지 → 자동 제거 → 리포트 생성

---

**문서 작성일**: 2025-01-21
**프로젝트 버전**: 1.0
**작성자**: Herbot Development Team
