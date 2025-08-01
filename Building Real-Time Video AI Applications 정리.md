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

8. DeepStream SDK, TAO Toolkit, 그리고 TensorRT
<img width="1796" height="912" alt="image" src="https://github.com/user-attachments/assets/3ea21179-be1a-46ef-8218-c98c2f62aee7" />
TAO Toolkit : 객체 탐지, 분류, 세분화와 같은 비전 AI 작업을 위한 모델을 생성하는 데 사용, 또한 모델의 전체 크기를 줄이는 모델 프루닝(pruning)과 양자화(quantization)와 같은 모델 최적화도 가능


9.
<img width="1705" height="865" alt="image" src="https://github.com/user-attachments/assets/71b71547-3dd6-435a-87a7-caf1cf45f841" />

10. 효율적인 비디오 스트림을 위해 CPU, GPU가 번갈아가며 사용됨.
<img width="1721" height="817" alt="image" src="https://github.com/user-attachments/assets/0f40f54c-3a00-4f7d-b350-9e97f75972fd" />

11. 상단엔 메세지를 버스로 교환하는 모습, 하단은 비디오가 파이프라인을 통해 교환되는 모습
<img width="1735" height="798" alt="image" src="https://github.com/user-attachments/assets/6088b7c9-a950-4cc8-8f7a-25b1057f7663" />

12.
<img width="1814" height="948" alt="image" src="https://github.com/user-attachments/assets/a3607f5b-7898-4e36-b63f-6cd8895c8c4f" />

13.
<img width="1771" height="742" alt="image" src="https://github.com/user-attachments/assets/98348859-d826-491c-9af6-f7cb4d5b9f5e" />

14. 
<img width="1365" height="894" alt="image" src="https://github.com/user-attachments/assets/2d06d44d-d7f3-417a-ba9c-e96177e2c1ed" />
