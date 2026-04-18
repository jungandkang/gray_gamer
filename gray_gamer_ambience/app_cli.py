#!/usr/bin/env python3
"""
CLI 버전 (단일 ���치 전송): ambience.json에서 한 번에 batch_size(기본 5)개의 항목만 꺼내
Gemini 모델에 보내 평가를 받고 ambience_scores.json으로 저장한 뒤 즉시 종료합니다.

AMBIENCE_ORG_FILE 관련 선언은 제거된 상태이며, 반복 처리 루프를 단일 배치 실행으로 바꿨습니다.
"""

import os
import json
import time
import argparse
import datetime
import requests
import unicodedata
from typing import List, Dict, Any

# --- 기본 경로들 (원본과 동일한 구조) ---
BASE_PATH = "/Users/hunjinjung/.dev/gray_gamer/gray_gamer_ambience/"
AMBIENCE_FILE = os.path.join(BASE_PATH, "novel_memory/ambience.json")
PROMPTS_FILE = os.path.join(BASE_PATH, "novel_memory/all_prompts.json")
CRITERIA_FILE = os.path.join(BASE_PATH, "novel_memory/criteria_sets.json")
CONFIG_PATH = os.path.join(BASE_PATH, "data/config.json")
LATEST_RES_FILE = os.path.join(BASE_PATH, "novel_memory/latest_response.json")
DEFAULT_OUT = os.path.join(BASE_PATH, "novel_memory/ambience_scores.json")

def load_json(path: str) -> Any:
    if os.path.exists(path):
        with open(path, "r", encoding="utf8") as f:
            return json.load(f)
    return {}

def save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def nfc(text: Any) -> str:
    return unicodedata.normalize('NFC', str(text)) if text is not None else ""

def _strip_codeblock(raw: str) -> str:
    if raw is None:
        return ""
    s = raw.strip()
    # Remove triple-backtick code blocks
    if s.startswith("```"):
        parts = s.split("\n")
        # drop first and last lines if they are ``` or ```json
        if len(parts) > 2:
            s = "\n".join(parts[1:-1])
    # If starts with "json" then strip that token
    if s.lower().startswith("json"):
        s = s[4:].strip()
    return s

# --- 설정 로드 ---
CONFIG = load_json(CONFIG_PATH)
API_KEY = str(CONFIG.get("API_KEY", "")).strip()
MODEL_NAME = str(CONFIG.get("MODEL_NAME", "gemini-1.5-flash")).strip()
MAX_TOKENS = int(CONFIG.get("max_tokens", 2048))
TIMEOUT = int(CONFIG.get("timeout", 30))

PROMPTS_DB = {p["id"]: p for p in load_json(PROMPTS_FILE).get("prompts", [])} if os.path.exists(PROMPTS_FILE) else {}
CRITERIA_DB = {c["id"]: c for c in load_json(CRITERIA_FILE).get("criteria_sets", [])} if os.path.exists(CRITERIA_FILE) else {}

def build_prompt(p_id: str, c_id: str, sentences: List[str]) -> str:
    p_base = PROMPTS_DB.get(p_id, {}).get("prompt", "") if p_id else ""
    c_set = CRITERIA_DB.get(c_id, {}).get("criteria", []) if c_id else []
    c_text = "\n".join([f"- {c['name']}: {c['instruction']}" for c in c_set])
    s_text = "\n".join([f"{i}. {s}" for i, s in enumerate(sentences)])
    prompt = f"""당신은 소설 비평 전문가입니다. 다음 지침을 엄격히 준수하여 평가하십시오.

[평가 기준]
{c_text}

[평가 대상 문장]
{s_text}

[작성 규칙]
1. 모든 점수는 반드시 0.0에서 10.0 사이의 실수(float)로 표기하십시오. (예: 8.5, 4.0)
2. '높음', '낮음' 같은 텍스트를 절대 사용하지 마십시오.
3. 각 점수에 대한 구체적인 이유(reason)를 한 문장으로 작성하십시오.
4. 반드시 아래의 JSON 형식을 유지하십시오.

[응답 형식 예시]
{{
  "results": [
    {{
      "index": 0,
      "scores": {{
        "artistry": {{"score": 8.5, "reason": "비유가 참신함"}},
        "ambience_likeness": {{"score": 7.0, "reason": "분위기에 부합함"}},
        "creativity": {{"score": 9.0, "reason": "독창적인 표현"}},
        "poetry": {{"score": 6.5, "reason": "운율이 다소 부족함"}},
        "myStylish": {{"score": 8.0, "reason": "개성이 뚜렷함"}}
      }},
      "comment": "전체적으로 우수함"
    }}
  ]
}}

{p_base}
"""
    return prompt

def call_gemini_api(prompt: str) -> Dict[str, Any]:
    """Gemini REST API 호출 및 응답 표준화. 실패 시 None 반환."""
    if not API_KEY:
        print("ERROR: API_KEY가 설정되어 있지 않습니다. data/config.json을 확인하세요.", flush=True)
        return None

    clean_key = API_KEY.strip()
    clean_model = MODEL_NAME.strip()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent?key={clean_key}"
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "response_mime_type": "application/json",
            "maxOutputTokens": MAX_TOKENS
        }
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        res_json = resp.json()
    except requests.exceptions.RequestException as e:
        print(f"API 호출 실패: {e}", flush=True)
        return None
    except ValueError as e:
        print(f"응답 JSON 디코딩 오류: {e}", flush=True)
        return None

    # 모델 출력 텍스트 추출 (원본 코드 호환 방식)
    raw_content = ""
    try:
        raw_content = res_json.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', "")
    except Exception:
        raw_content = ""

    clean_content = _strip_codeblock(raw_content)

    # 시도: 바로 JSON 로드
    parsed = None
    try:
        parsed = json.loads(clean_content)
    except json.JSONDecodeError:
        # 경우에 따라 모델이 앞에 텍스트를 붙이거나, JSON이 아닌 형태로 돌아올 수 있다.
        # 간단한 복구 시도: 문자열에서 첫 '{' 또는 '['부터 끝까지 추출해서 파싱 시도
        s = clean_content
        start_idx = min((s.find('{') if s.find('{')!=-1 else len(s)), (s.find('[') if s.find('[')!=-1 else len(s)))
        if start_idx < len(s):
            candidate = s[start_idx:]
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                parsed = None

    standardized = {"results": []}
    if parsed is None:
        print("경고: 모델 응답을 JSON으로 파싱하지 못했습니다. raw content를 latest_response.json에 저장합니다.", flush=True)
        standardized = {"results": []}
    else:
        if isinstance(parsed, list):
            standardized = {"results": parsed}
        elif isinstance(parsed, dict):
            standardized = parsed if "results" in parsed else {"results": [parsed]}
        else:
            standardized = {"results": []}

    # 백업 저장
    try:
        save_json(LATEST_RES_FILE, standardized)
    except Exception as e:
        print(f"주의: latest_response.json 저장 중 오류: {e}", flush=True)

    return standardized

def process_single_batch(ambience_file: str, out_file: str, batch_size: int = 5, start_index: int = 0) -> None:
    """
    단일 배치만 처리하고 종료합니다.
    start_index 위치에서 batch_size 개(있으면 그만큼, 없으면 남은 개수)만 모델에 전송합니다.
    """
    data = load_json(ambience_file)
    assets = data.get("reviewed", []) if isinstance(data, dict) else []

    total = len(assets)
    if total == 0:
        print(f"입력 파일에 처리할 항목이 없습니다: {ambience_file}", flush=True)
        return

    if start_index >= total:
        print(f"ERROR: start index ({start_index})가 총 항목 수({total})보다 큽니다.", flush=True)
        return

    # 실제 처리할 배치 (단 한번)
    end_index = min(start_index + batch_size, total)
    batch = assets[start_index:end_index]
    sentences = [nfc(item.get("text", "")) for item in batch]

    print(f"단일 배치 처리: items {start_index}..{end_index-1} (count={len(batch)})", flush=True)

    # p_id/c_id: 각 asset에 포함돼 있으면 그 값을 사용, 없으면 default
    default_p = next(iter(PROMPTS_DB.keys()), None)
    default_c = next(iter(CRITERIA_DB.keys()), None)

    p_id = batch[0].get("prompt_id") if batch and batch[0].get("prompt_id") else default_p
    c_id = batch[0].get("criteria_id") if batch and batch[0].get("criteria_id") else default_c

    prompt = build_prompt(p_id, c_id, sentences)
    print(f"[{datetime.datetime.now().isoformat()}] 모델 호출 (단일 배치): prompt_id={p_id}, criteria_id={c_id}", flush=True)

    res = call_gemini_api(prompt)
    if not res:
        print("모델 응답이 없거나 파싱 실패. 종료합니다.", flush=True)
        return

    results = res.get("results", [])
    if not results:
        print("모델이 비어있는 results를 반환했습니다.", flush=True)

    all_results = []
    for item in results:
        try:
            local_index = int(item.get("index", -1))
        except Exception:
            local_index = -1
        if local_index < 0 or local_index >= len(batch):
            # index가 없거나 범위를 벗어나면 경고 후 무시
            print(f"주의: 결과 항목에 올바른 index가 없습니다(index={item.get('index')}). 이 항목은 무시됩니다.", flush=True)
            continue

        global_idx = start_index + local_index
        asset = assets[global_idx]
        out_entry = {
            "global_index": global_idx,
            "id": asset.get("id"),
            "original_text": asset.get("text"),
            "scores": item.get("scores", {}),
            "comment": item.get("comment", "")
        }
        all_results.append(out_entry)

    # 결과 저장
    out_data = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "model": MODEL_NAME,
        "count": len(all_results),
        "scores": all_results,
        "processed_range": [start_index, end_index - 1]
    }
    try:
        save_json(out_file, out_data)
        print(f"완료: {out_file}에 {len(all_results)}개 결과 저장됨.", flush=True)
    except Exception as e:
        print(f"ERROR: 결과 저장 실패: {e}", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Ambience single-batch scoring CLI (Gemini). 단 한 번만 배치를 보내고 종료합니다.")
    parser.add_argument("--ambience-file", default=AMBIENCE_FILE, help="입력 ambience.json 파일 경로")
    parser.add_argument("--out-file", default=DEFAULT_OUT, help="저장할 ambience_scores.json 경로")
    parser.add_argument("--batch-size", type=int, default=5, help="한 번에 보낼 항목 수 (기본 5)")
    parser.add_argument("--start", type=int, default=0, help="처리를 시작할 인덱스 (기본 0)")
    args = parser.parse_args()

    if not API_KEY:
        print("ERROR: data/config.json에 API_KEY가 설정되어 있지 않습니다. 실행을 중단합니다.", flush=True)
        return

    # 단일 배치만 처리
    process_single_batch(args.ambience_file, args.out_file, batch_size=args.batch_size, start_index=args.start)

if __name__ == "__main__":
    main()