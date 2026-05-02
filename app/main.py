from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from app.spam import check_spam
from pydantic import BaseModel
from app.issue import *
import traceback
import logging # 로깅 라이브러리 추가

# 1) 로그 포맷: 시간 + 레벨 + 파일명:줄번호(함수명) + 메시지
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | "
           "%(filename)s:%(lineno)d (%(funcName)s) | "
           "%(message)s"
)
logger = logging.getLogger("spamcheck")

# FastAPI 기반 웹 앱 생성
app = FastAPI(title="SpamCheck Web")

# 정적 HTML 서빙
app.mount("/static", StaticFiles(directory="static"), name="static")

# 메인 페이지 (/) 처리
@app.get("/", response_class=HTMLResponse)
def home():
    with open("static/index.html", encoding="utf-8") as f:
        return f.read()
    
class ClassifyRequest(BaseModel):
    text: str

@app.post("/classify")
async def classify(payload: ClassifyRequest):
    text = payload.text
    
    # (A) 요청 들어온 것 자체를 기록: 언제 / 무엇을 / 어떤 입력인지
    logger.info(f"CALL /classify | text='{text}' | len={len(text)}")
    
    try:
        # 의도적 에러 삽입 (운영 중 발생할 수 있는 예상치 못한 버그 시뮬레이션)
        if text == "crash":
            raise RuntimeError("의도적장애추가")
            
        # 정상 스팸 체크 로직 수행
        label, score = check_spam(text)
        
        # (B) 정상 처리 결과도 짧게 기록
        logger.info(f"OK  /classify | label={label} score={score}")
        
        return {
            "label": label, "score": score
        }
        
    except Exception as e:
        # (C) 디버깅 핵심: 에러 종류/메시지 + 스택 트레이스 기록
        # logger.exception은 현재 예외의 traceback을 자동으로 터미널에 찍어줍니다.
        logger.exception(
            f"FAIL /classify | text='{text}' | error={type(e).__name__}: {e}"
        )
        tb = traceback.format_exc()
        title = f"[Prod Error] /classify failed: {type(e).__name__}"
        body = (
             f"## Summary\n"
             f"- endpoint: /classify\n"
             f"- input(text, short): `{text}`\n"
             f"- input length: {len(text)}\n\n"
             f"## Exception\n"
             f"- type: {type(e).__name__}\n"
             f"- message: {str(e)}\n\n"
             f"## Traceback (line info)\n"
             f"```text\n{tb}\n```"
        )
        create_github_issue(title, body, logger)
        
        # (D) 사용자 응답은 심플하게 (내부 에러를 사용자에게 자세히 노출하지 않음)
        return {"label": "Internal Server Error", "score": -1}