# 29일차

## 어제 다 못마쳤다면, 코랩 앙상블 계속 진행.
<img width="1210" height="734" alt="image" src="https://github.com/user-attachments/assets/92fb1082-cc92-4b4e-a15a-e322399c8121" />

**-> 결과영상 : 피플넷 + YOLOv11의 앙상블로, 사람뿐만 아니라 일반적인 객체까지 감지하는 모습.**

## NVIDIA 실시간 비디오 AI 애플리케이션 구축 자격시험 대비.
[Building Real-Time Video AI Applications](https://learn.nvidia.com/courses/course?course_id=course-v1:DLI+S-IV-01+V1&unit=block-v1:DLI+S-IV-01+V1+type@vertical+block@67d7c59f99074da4a57220c9dfbfc980)<br>
[교수님의 Building Real-Time Video AI Applications 정리본](https://docs.google.com/document/d/1WUtXecVecKDd5b2omhBJMx-FiIq2zsVs_Fsetb5rdcE/edit?tab=t.0)
- 필요서류와 개인정보 수집 동의서를 월요일까지 제출.
- 8시간 소요 예상되므로, 금요일까지 자료를 토대로 공부하고 일요일까지 시험 마치기.
  <img width="1289" height="643" alt="image" src="https://github.com/user-attachments/assets/f1465166-1344-4ae7-92cc-c3ad7ce553e9" />
- 사비로 90달러(128,279원)결제한 증빙 사진, 서류제출시 금액은 돌려받는 식으로 지원받음.
  
## 들으면서 정리
이 실습에서는 NVIDIA의 도구를 사용하여 하드웨어 가속 비디오 AI 애플리케이션을 구축하고 유지보수하는 방법을 배우게 됩니다.<br>
실시간 비디오 AI 애플리케이션을 만들기 위해 NVIDIA의 DeepStream, TAO Toolkit, 그리고 TensorRT를 사용할 것입니다.<br>

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
다양한 산업 분야의 엣지 디바이스들 -> (온프레미스 서버 - 클라우드) 중앙 처리 -> AI 결과물

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

