# TCMPipe Tongue Evaluation System (임상적 관점)

---

## 1. 평가 시스템의 철학과 구조

- **최소 원자 속성(Atomic Property) 기반**: 모든 설진 결과를 '서술어'가 아닌 '속성'으로 분해하여 평가합니다.
- **임상적 논리와 1:1 매칭**: 한의사가 실제로 진단하는 논리(위치, 색, 형태, 두께, 성질 등)를 그대로 반영합니다.
- **확장성과 유지보수성**: config 파일(token_config.json)만 수정하면 임상/연구 목적에 맞는 평가 정책을 즉시 반영할 수 있습니다.
- **가중치 시스템**: 임상적으로 중요한 feature에 더 높은 가중치를 부여할 수 있습니다.

---

## 2. config 설계 원칙 (token_config.json)

- **category_map**: prefix별 카테고리 매핑 (LOCATION, COLOR, SHAPE, COATCOLOR, THICK, NATURE)
- **token_dicts**: 각 카테고리별 원자 속성 리스트 (예: "舌尖", "红", "胖", "白", "薄", "腻" 등)
- **pattern_table**: 자연어 서술어를 원자 속성 조합으로 분해하는 규칙 (예: "舌尖红" → ["LOCATION_舌尖", "COLOR_红"])
- **synonyms**: 임상적으로 유사한 원자 토큰 쌍의 유사도 (예: "淡|淡白": 0.9)
- **weights**: 카테고리별 임상적 중요도 가중치 (예: tongue=2.0, coat=1.5, location=1.0, other=0.5)

---

## 3. 전체 평가 파이프라인

1. **입력 데이터**: 예측/정답 쌍이 들어있는 jsonl 파일 (예: `{ "predict": "...", "label": "..." }`)
2. **토큰 파싱**: 자연어 서술어를 pattern_table과 token_dicts를 이용해 원자 속성으로 분해
3. **카테고리 분리**: category_map에 따라 각 토큰을 (tongue, coat, location, other)로 분류
4. **카테고리별 유사도 계산**: Hungarian matching(헝가리안 매칭) 기반으로 토큰 리스트 유사도 산출
5. **가중 평균 점수 산출**: weights에 따라 카테고리별 점수의 가중 평균
6. **결과 저장/리포트**: 파일별 평균 점수 저장 및 리포트/시각화

---

## 4. 핵심 알고리즘 (수식/로직)

### (1) 토큰 파싱
- 입력 문자열에서 pattern_table에 매칭되는 복합 표현을 먼저 분해
- 남은 부분에서 token_dicts의 원자 속성을 추출

### (2) 카테고리 분리
- 각 토큰의 prefix를 category_map에서 찾아 해당 카테고리 리스트에 분류

### (3) 토큰 리스트 유사도 (Hungarian Matching)
- 두 리스트의 모든 토큰쌍에 대해 token_similarity(시노님/부분일치 반영)로 유사도 행렬(sim_matrix) 생성
- Hungarian matching으로 최대 유사도 합이 되도록 1:1 매칭
- 최종 유사도 = 매칭된 유사도 합 / (두 리스트 중 더 긴 쪽의 길이)

### (4) 카테고리별 가중 평균
- mean_score = (score_tongue * w_tongue + score_coat * w_coat + score_location * w_location + score_other * w_other) / (w_tongue + w_coat + w_location + w_other)

---

## 5. 실제 예제 (입력~출력)

### 입력 예시 (jsonl)
```jsonl
{"predict": "舌尖红苔黄腻", "label": "舌尖红苔黄腻"}
{"predict": "舌边红苔白滑", "label": "舌边红苔白滑"}
{"predict": "舌根暗苔黑燥", "label": "舌根暗苔黑燥"}
```

### 파싱/분해 예시
- "舌尖红苔黄腻" → ["LOCATION_舌尖", "COLOR_红", "COATCOLOR_黄", "NATURE_腻"]
- "舌边红苔白滑" → ["LOCATION_舌边", "COLOR_红", "COATCOLOR_白", "NATURE_滑"]
- "舌根暗苔黑燥" → ["LOCATION_舌根", "COLOR_暗", "COATCOLOR_黑", "NATURE_燥"]

### 카테고리 분리 예시
- LOCATION: ["LOCATION_舌尖"]
- COLOR: ["COLOR_红"]
- COATCOLOR: ["COATCOLOR_黄"]
- NATURE: ["NATURE_腻"]

### 유사도 계산 예시
- 예측/정답이 완전히 일치하면 각 카테고리별 점수=1.0, 최종 점수=1.0
- 예측이 "舌尖红苔黄腻", 정답이 "舌尖红苔白滑"이면:
  - LOCATION: 1.0 (동일)
  - COLOR: 1.0 (동일)
  - COATCOLOR: 0.0 (黄 vs 白)
  - NATURE: 0.0 (腻 vs 滑)
  - 가중 평균에 따라 최종 점수 산출

---

## 6. 실제 평가 결과 예시

| 파일명                          | 평균 점수  |
|----------------------------------|-----------|
| compare_eval_all_30epoch         | 0.4955    |
| compare_eval_all_10epoch         | 0.4791    |
| compare_alltrain_val16           | 0.4133    |
| compare_alltrain_val64           | 0.4385    |
| compare_multi_p_lang_p_eval      | 0.4095    |
| compare_lang_only_eval           | 0.3286    |

---

## 7. 확장/운영 팁

- **pattern_table, synonyms**: 실제 데이터에서 매칭 안 되는 표현/유사어가 발견될 때마다 계속 추가/보강
- **weights**: 임상적 중요도/실제 진단 정확도에 따라 조정
- **config만 수정**하면 평가 정책을 즉시 바꿀 수 있음
- **단위테스트/CLI/그래프 평가** 등 다양한 실험/운영 환경 지원

---

## 8. 임상적 해석/활용

- **원자 속성 기반 평가**로 진단적 모호성 제거, 신뢰도 향상
- **위치/색/형태/두께/성질** 등 임상적으로 중요한 feature별로 세밀한 평가 가능
- **가중치 시스템**으로 실제 임상적 중요도 반영
- **확장성**: 새로운 임상 지식/표현이 등장해도 config만 보강하면 즉시 반영 가능

---

## 9. 전체 의사코드

```python
for (predict, label) in jsonl:
    pred_tokens = parse_tongue_features(predict)
    label_tokens = parse_tongue_features(label)
    cats_pred = classify_to_categories(pred_tokens, category_map)
    cats_label = classify_to_categories(label_tokens, category_map)
    scores = []
    for cat in ['tongue', 'coat', 'location', 'other']:
        score = compute_token_list_similarity(cats_pred[cat], cats_label[cat], synonyms)
        scores.append(score)
    mean_score = weighted_average(scores, weights)
    ...
```

---

## 10. 결론

- 이 시스템은 임상적 논리와 1:1로 매칭되는 평가를 지향합니다.
- config 확장/보강을 통해 실제 임상/연구/실무에 최적화된 평가 기준을 완성할 수 있습니다.
- 임상 전문가와 협업하여, 실제 진단 논리와 가장 가까운 자동 평가 시스템을 만들어가세요. 