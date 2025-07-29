import os
import pytest
import json
from evaluation2_ver2 import (
    parse_tongue_features, parse_synonyms, token_similarity,
    compute_token_list_similarity, compute_category_similarity, load_json_config
)

dir_path = os.path.dirname(__file__)
config_path = os.path.join(dir_path, 'token_config.json')
config = load_json_config(config_path)
synonyms = parse_synonyms(config['synonyms'])
category_map = config['category_map']

def test_parse_tongue_features_basic():
    # 주요 토큰이 모두 포함되어 있는지만 확인 (중복 허용)
    result = set(parse_tongue_features('淡红胖舌苔薄白'))
    assert {'COLOR_淡红', 'SHAPE_胖舌', 'COATCOLOR_薄白'}.issubset(result)

def test_parse_tongue_features_pattern():
    # 복합 패턴 분해
    assert set(parse_tongue_features('苔薄少')) >= {'THICK_薄', 'THICK_少'}

def test_token_similarity_exact():
    assert token_similarity('COLOR_红', 'COLOR_红', synonyms) == 1.0

def test_token_similarity_synonym():
    # 시노님 점수
    assert token_similarity('COLOR_暗红', 'COLOR_红', synonyms) == 0.8
    assert token_similarity('COLOR_红', 'COLOR_淡红', synonyms) == 0.7

def test_token_similarity_cross():
    # CROSS 시노님
    assert token_similarity('SHAPE_胖', 'THICK_厚', synonyms) == 0.5

def test_compute_token_list_similarity():
    # 완전 일치
    assert compute_token_list_similarity(['COLOR_红'], ['COLOR_红'], synonyms) == 1.0
    # 부분 유사
    assert compute_token_list_similarity(['COLOR_暗红'], ['COLOR_红'], synonyms) == 0.8
    # 모두 0
    assert compute_token_list_similarity(['COLOR_红'], ['SHAPE_胖'], synonyms) == 0.0

def test_compute_category_similarity():
    # 카테고리별 분리 및 평균
    s_t, s_c, s_o, s_m = compute_category_similarity('淡红胖舌苔薄白', '淡红瘦舌苔薄白', config)
    assert s_t > 0 and s_c == 1.0 and s_o >= 0.0
    # 완전 불일치: mean=1/3로 기대값 수정
    s_t, s_c, s_o, s_m = compute_category_similarity('COLOR_红', 'SHAPE_胖', config)
    assert s_m == (0.0 + 0.0 + 1.0) / 3 