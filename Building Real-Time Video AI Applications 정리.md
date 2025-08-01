## Building Real-Time Video AI Applications 정리
이 실습에서는 NVIDIA의 도구를 사용하여 하드웨어 가속 비디오 AI 애플리케이션을 구축하고 유지보수하는 방법을 배움.<br>
실시간 비디오 AI 애플리케이션을 만들기 위해 NVIDIA의 DeepStream, TAO Toolkit, 그리고 TensorRT를 사용할 것.<br>

<img width="600" height="461" alt="image" src="https://github.com/user-attachments/assets/b974828a-72c1-416f-a04c-69dfb9123f79" />

-> NVIDIA DeepStream의 Primary GIE (GPU Inference Engine) 설정 파일입니다.<br>
주로 교통 카메라에서 차량을 감지하는 AI 모델의 설정을 담고 있습니다.<br>

**강의 시작**
1. "H:MM:SS 남은 시간"을 보여주는 타이머로, 제한 시간안에 실습을 마쳐야한다.
<img width="1906" height="895" alt="image" src="https://github.com/user-attachments/assets/7bd55107-26e2-4223-9b99-c5182b4dfea9" />
2. 실습전에 오른쪽 start 버튼을 눌러야 GPU 가속 서버가 실행된다.
<img width="1930" height="289" alt="image" src="https://github.com/user-attachments/assets/7852eeae-76f2-4760-86d2-629c605f41f5" />
3. 실시간 영상 기반 AI 적용 분야들.<br>(보안·운영·교통, 제조·물류·콘텐츠 분야까지, 카메라로 획득한 영상을 즉각 분석하여 다양한 산업 현장의 효율성과 안전성을 높이는 데 활용됨)
<img width="1822" height="885" alt="image" src="https://github.com/user-attachments/assets/3958e92f-14dd-493b-adbc-cdec2a782f83" />

| 항목                      | 설명                    |
| --------------------------- | --------------------- |
| 출입 통제 (Access Control)      | 사람들이 출입하는 공간에서의 보안 관리 |
| 운영 관리 (Managing Operations) | 창고나 물류 시설에서의 박스/화물 관리 |
| 주차 관리 (Parking Management)  | 대형 주차장에서의 차량 및 공간 관리  |
| 교통 공학 (Traffic Engineering) | 도로 교차로에서의 차량 흐름 분석    |

| 항목                         | 설명                  |
| -------------------------- | ------------------- |
| 360도 영상 분석                 | 파노라마 또는 전방향 영상 처리   |
| 광학 검사 (Optical Inspection) | 제조업에서의 품질 검사        |
| 물류 관리 (Managing Logistics) | 공항이나 물류 센터에서의 화물 추적 |
| 콘텐츠 및 디자인                  | 멀티미디어 콘텐츠 분석 및 처리   |

4. "비디오에서 AI 기반 인사이트까지(FROM VIDEO TO AI-BASED INSIGHT)"라는 제목으로, AI 추론이 어디서 일어날 수 있는지 보여줌.
<img width="1773" height="916" alt="image" src="https://github.com/user-attachments/assets/e8db6871-fab4-46b7-96ac-65878cc34c9e" />
다양한 산업 분야의 엣지 디바이스들 -> (온프레미스 서버 - 클라우드) 중앙 처리 -> AI 결과물<br>
-> 비디오 데이터가 수집된 후 AI 처리를 통해 실용적인 비즈니스 인사이트로 변환되는 전체 파이프라인 사진임

| 산업분야 (Industry)       | 엣지 디바이스들 설명 (Description) |
| ------------------- | ---------------- |
| 안전 (Safety)         | 보안 카메라           |
| 소매 (Retail)         | 매장 관리 시스템        |
| 건설 (Construction)   | 건설 현장 모니터링       |
| 제조업 (Manufacturing) | 산업용 로봇/장비        |


| AI결과물의 기능 (Capability)     | 설명 (Description) |
| ------------------- | ---------------- |
| 알림 (Alerts)         | 경고 신호            |
| 분석 (Analytics)      | 데이터 분석 결과        |
| 시각화 (Visualization) | 차트나 그래프          |

5. 엣지 컴퓨팅 기반 실시간 스트리밍 분석
<img width="1719" height="954" alt="image" src="https://github.com/user-attachments/assets/7cb9f1e6-0942-436a-b692-e7a0db4386d6" />

엣지 기반 머신러닝 솔루션의 장점
- 더 빠른 응답 시간 : 특히 데이터 기반 의사결정을 즉시 수행해야 할 때 의미가 큼.
- 낮은 대역폭 비용 : 전송할 원시 비디오의 양을 관리할 수 있음
- 네트워크 관련 문제 완화 :네트워크 연결이 없거나, 낮은 원격 위치에서 센서를 사용할 때 발생하는 문제 완화
- 데이터 프라이버시 및 보안 향상 : 엣지 디바이스에서 처리 후, 민감한 데이터를 폐기하도록 프로그래밍될 수 있기 때문

상단: 중앙 서버에만 AI 칩이 존재하는 "온프레미스 또는 클라우드 기반 인프라 시스템"<br>
하단: AI 칩이 내장된 엣지 기기들 각각에서 처리하는 "엣지 컴퓨팅 시스템"<br>
-> 각각의 기기들에 모두 AI칩이 내장되있어서, 데이터를 중앙 서버로 보내지 않고 현장(엣지)에서 직접 처리하여 실시간 분석을 수행하는 시스템<br>
-> Orin nano 등이 엣지 디바이스 예시고, 기후변화 등에 쓰인다.

6. 빠른응답이 필요할때 "엣지 컴퓨팅 시스템"을 쓴다. IVA(Intelligent Video Analytics) 애플리케이션 워크플로우를 보니 더 빠른 구조다. 
<img width="1821" height="958" alt="image" src="https://github.com/user-attachments/assets/521bf29b-18e2-4b60-91d4-14b47f3a2f41" /><br>
하나 이상의 입력 비디오 스트림을 받아 디코딩과 먹싱(또는 집계)을 수행 -> 배치 전처리 -> 데이터를 AI 추론에 통과시킵니다. <br>
이후 원본 비디오와 결합된 AI 기반 인사이트를 인코딩해 저장하거나, 추가 분석을 위해 다운스트림으로 전달함.

7. 비디오 AI 솔루션 개발 시 직면하는 주요 기술적, 경제적 도전과제 세가지

<img width="1822" height="792" alt="image" src="https://github.com/user-attachments/assets/33a4a6aa-607d-4e88-b301-d1bc1d5abaec" />

| 항목 (영문)                            | 항목 (한글)             | 주요 내용                                                                                                             |
| ---------------------------------- | ------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **DESIGN COMPLEXITY**              | 설계 복잡성              | - AI 프레임워크의 다양성: TensorFlow, PyTorch, ONNX 등 여러 프레임워크 지원 필요<br>- 다양한 툴 체인: C++, Python, R, Java 등 여러 언어와 도구 통합 필요 |
| **OPTIMIZING TCO**                 | 총 소유비용 최적화          | - 다변수 도전 과제를 순환 다이어그램으로 표현<br>- 정확성(Accuracy), 처리량(Throughput), 지연시간(Latency), 전력(Power), 비용(Cost) 간 균형 최적화 필요    |
| **ONE CODE BASE MULTIPLE TARGETS** | 하나의 코드베이스로 다중 타겟 지원 | - 엣지에서 클라우드까지 다양한 환경에 동일 코드 배포<br>- 자동차, 의료, 보안, 제조 등 서로 다른 수직 산업 분야에 적용 과제                                       |

8. DeepStream SDK, TAO Toolkit
<img width="1796" height="912" alt="image" src="https://github.com/user-attachments/assets/3ea21179-be1a-46ef-8218-c98c2f62aee7" />
TAO Toolkit : 객체 탐지, 분류, 세분화와 같은 비전 AI 작업을 위해 "사전 훈련된 모델 기반으로 더 최적화한 모델"을 만드는 데 사용되는 모델 적응 SDK,<br>
입력 데이터 → PRE-TRAINED MODEL → TRAIN → PRUNE → RETRAIN → OUTPUT MODEL<br>
또한 모델의 전체 크기를 줄이는 모델 프루닝(pruning)과 양자화(quantization)와 같은 모델 최적화도 가능<br>
<br>
DEEPSTREAM SDK : 지능형 비디오 분석(IVA) 파이프라인 구축을 위한 가속화된 AI 프레임워크, 디코더 인코더, 이미지 스케일링 변환<br>
입력 → DECODE → PRE-PROCESS → PRIMARY INFERENCE → TRACKER → COMPUTE → ANALYTICS/APPS → ENCODER<br>
실시간 비디오 스트림 처리 및 분석 전체 파이프라인 구축이 목적.

9. GStreamer : 오픈소스 멀티미디어 분석 프레임워크로, 플러그인을 사용하여 비디오/오디오 처리에서 "모듈화된 접근 파이프라인"을 구성하는 방식
<img width="1705" height="865" alt="image" src="https://github.com/user-attachments/assets/71b71547-3dd6-435a-87a7-caf1cf45f841" />

- Level 1 - 플러그인(Plugin) : 기존 소프트웨어의 기능을 확장하거나 특정 작업을 수행하기 위해 추가로 설치하는 독립적인 소프트웨어 모듈.
  - PAD를 통해 연결되는 기본 빌딩 블록에 쓰임 (PPacket Assembly/Disassembly 또는 Packet Assembler/Disassembler)
  - PAD는 주로 X.25 네트워크에서 사용되는 장치로, 서로 다른 프로토콜이나 데이터 형식 간의 변환을 담당.
  - 예시: Src Plugin → Src → Sink → Filter Plugin → Src → Sink → Sink Plugin (각 플러그인은 특정 기능을 수행하는 최소 단위)

- Level 2 - BINS (빈)
    - 플러그인들의 컨테이너/집합을 의미하며, 여러 플러그인을 하나의 단위로 묶어서 관리하는 것
    - 예시: Src → Plugin A → Plugin B → Sink으로 구성된 BIN

- Level 3 - PIPELINE (파이프라인)
  - 데이터가 여러 단계의 처리 과정을 순차적으로 거치면서 변환되는 아키텍처 패턴
  - 기능: 버스를 제공하고 동기화를 관리하는 최상위 레벨 빈
  - 구조: SOURCE BIN → PROCESS BIN → SINK BIN
  - 예시: Video Source → Decoder → Scale → Filter → Display

10. 효율적인 비디오 스트림을 위해 CPU, GPU가 번갈아가며 사용되어 메모리 효율 증가.
<img width="1721" height="817" alt="image" src="https://github.com/user-attachments/assets/0f40f54c-3a00-4f7d-b350-9e97f75972fd" />

플러그인 아키텍처:
- 입력: Input (Metadata)
- 처리: PLUGIN (LOW LEVEL LIB + GPU Hardware를 통한 Low Level API)
- 출력: Output + Metadata

| 카테고리         | 플러그인               | 기능                        |
| ------------ | ------------------ | ------------------------- |
| **비디오 처리**   | gst-nvvideoconvert | 가속화된 비디오 디코더              |
|              | gst-nvstreammux    | 스트림 집계기 – 믹서 및 배치 처리      |
|              | gst-nvinfer        | TensorRT 기반 추론 (탐지 및 분류)  |
|              | gst-nvtracker      | 참조 KLT 추적기 구현             |
|              | gst-nvdsosd        | 화면 표시 API (박스 및 텍스트 오버레이) |
| **렌더링 및 변환** | gst-tiler          | 다중 소스를 2D 그리드 배열로 렌더링     |
|              | gst-nvegltransform | 가속화된 X11/EGL 기반 렌더러 플러그인  |
|              | gst-nvvideoconvert | 스케일링, 포맷 변환, 회전           |
|              | gst-nvdewarper     | 360도 카메라 입력 디워핑           |
| **기타**       | gst-nvmsgconv      | 메타데이터 생성                  |
|              | gst-nvmsgbroker    | 클라우드로 메시징                 |


12. 스트리밍 데이터의 "애플리케이션과 파이프라인 간 통신 및 데이터 교환"을 위한 여러 메커니즘.
<img width="1735" height="798" alt="image" src="https://github.com/user-attachments/assets/6088b7c9-a950-4cc8-8f7a-25b1057f7663" />

- 앱과 버스 : messages, events, queries 교환
- 파이프라인 내부 : Video File → Decoder → Scale → Filter → Display 흐름<br>
(버퍼 = 파이프라인 내부 플러그인들 사이의 이동경로를 말함)

14. 메타데이터 구조에 대한 설명으로, 메타데이터가 "비디오 분석 결과를 구조화된 형태로 전달"하는 DeepStream의 핵심 메커니즘
<img width="1814" height="948" alt="image" src="https://github.com/user-attachments/assets/a3607f5b-7898-4e36-b63f-6cd8895c8c4f" />

| 항목                      | 세부 내용                                                                                                                                                    |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **파이프라인 구성**            | `Gst-nvstreammux` → `Gst-nvinfer` → `Gst-nvosd`<br>각 플러그인의 `src`에서 `sink`로 버퍼와 함께 메타데이터 전달                                                               |
| **메타데이터 생성 과정**         | - **생성**: 그래프 내 플러그인들이 메타데이터를 생성<br>- **첨부**: 생성된 메타데이터를 버퍼에 첨부하여 다운스트림으로 전달<br>- **접근**: 프로브 함수를 연결하여 메타데이터 접근                                          |
| **DeepStream 메타데이터 내용** | AI 추론 결과 인사이트 포함:<br>- 탐지된 객체 수 (Number of objects detected)<br>- 바운딩 박스 좌표 (Bounding box coordinates)<br>- 객체 클래스 (Object classes)                      |
| **메타데이터 구조**            | - `NvDsFrameMeta`: 프레임 레벨 메타데이터<br>- `NvDsObjectMeta`: 객체 레벨 메타데이터<br>- `gst_nvinfer_id`: 추론 ID<br>- `rect_params`: 사각형 파라미터<br>- `display_id`: 디스플레이 ID |

15. 비디오 AI 시스템의 성능을 평가하는 주요 지표들 4가지.
<img width="1771" height="742" alt="image" src="https://github.com/user-attachments/assets/98348859-d826-491c-9af6-f7cb4d5b9f5e" />

  1. Accuracy (정확도)
    - IoU (Intersection over Union): 교집합/합집합 비율로 객체 탐지 정확도 측정
    - mAP (Mean Average Precision): 평균 정밀도, 모델의 전반적인 성능 평가
  2. Throughput (처리량)
    - 정의: 단위 시간 내에 전송 및 수신되는 데이터의 양
    - 의미: 시스템이 얼마나 많은 데이터를 처리할 수 있는지
  3. Latency (지연시간)
    - 정의: 프레임이 네트워크를 통해 전송되는 데 걸리는 시간
    - 의미: 실시간 처리에서 중요한 응답 속도
  4. Hardware Utilization and Memory Footprint (하드웨어 활용도 및 메모리 사용량)
    - 의미: 시스템 리소스의 효율적 사용도

우측 기어 다이어그램: 성능 최적화를 위한 균형 조정을 상징적으로 표현한 것으로, 이 모든 지표들이 상호 연관되어 있으며 하나를 개선하면 다른 요소에 영향을 미칠 수 있음을 보여줌.
비디오 AI 시스템의 종합적인 성능 평가를 위한 핵심 메트릭들.

16. "하드웨어부터 사용자 앱단까지 전체 스택을 지원"하는 비디오 ai 개발 플랫폼인 DeepStream SDK의 기능 총정리
<img width="1365" height="894" alt="image" src="https://github.com/user-attachments/assets/2d06d44d-d7f3-417a-ba9c-e96177e2c1ed" />

| 레이어                                   | 구성 요소                                                                                                                                                                                                                                                                                                                                                             |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **USER APPLICATIONS**<br>(사용자 앱단, 최상단) | - **ACCESS CONTROL**: 출입 통제<br>- **SMART PARKING**: 스마트 주차<br>- **RETAIL ANALYTICS/INSIGHT**: 리테일 분석/인사이트<br>- **INTELLIGENT TRAFFIC SYSTEMS**: 지능형 교통 시스템<br>- **LAW ENFORCEMENT**: 법 집행                                                                                                                                                                         |
| **DEEPSTREAM SDK**<br>(중간층)           | **PLUGINS**<br>- GPU Optimized TensorRT Plugin<br>- Communication Plugins<br>- 3rd Party Lib/AI Plugins<br><br>**FLEXIBLE SCALABLE GRAPHS**<br>- 확장 가능한 파이프라인 구조 (연결된 녹색 박스들로 표현)<br><br>**DEVELOPMENT TOOLS**<br>- End-to-End Reference Applications<br>- App Building/Enhancement Tools<br>- Sample Applications and Code<br>- Profiling and Performance Tuning |
| **Base Technologies**<br>(하위층)        | - **TENSORRT**: AI 추론 최적화<br>- **MULTIMEDIA APIs/VIDEO CODEC SDK**: 멀티미디어 및 비디오 코덱<br>- **IMAGING**: 이미징 처리<br>- **METADATA DESCRIPTION**: 메타데이터 기술                                                                                                                                                                                                               |
| **Hardware Platforms**<br>(최하위)       | - **LINUX, CUDA**: 리눅스 및 CUDA 기반<br>- **JETSON, TESLA**: NVIDIA 하드웨어 플랫폼                                                                                                                                                                                                                                                                                          |
