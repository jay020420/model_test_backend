import re
from typing import Optional, Dict

INDUSTRY_MAP = {
    "치킨": ["치킨", "치킨집", "후라이드", "양념치킨"],
    "카페": ["카페", "커피", "디저트", "베이커리"],
    "피자": ["피자"],
    "편의점": ["편의점", "CVS"],
}

REGION_HINT = ["구", "시", "군", "동", "읍", "면"]  # 간단 힌트


def _normalize_currency(text: str) -> Optional[float]:
    t = text.replace(",", "").strip()
    m = re.match(r"([0-9]+(?:\.[0-9]+)?)\s*(원|만원|천만원|억원)?", t)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2) or "원"
    mult = {"원": 1, "만원": 1e4, "천만원": 1e7, "억원": 1e8}.get(unit, 1)
    return val * mult


def _extract_money(sent: str) -> Dict[str, float]:
    res = {}
    # 1개월
    m1 = re.search(r"(한\s*달|지난달|최근\s*1개월)[^\d]*(\d[\d,\.]*\s*(원|만원|천만원|억원))", sent)
    if m1:
        res["sales_1m"] = _normalize_currency(m1.group(2))
    # 3개월 평균
    m3 = re.search(r"(3개월|최근\s*세\s*달|최근\s*3개월)[^\d]*(\d[\d,\.]*\s*(원|만원|천만원|억원))", sent)
    if m3:
        res["sales_3m_avg"] = _normalize_currency(m3.group(2))
    # 고객수(명)
    c1 = re.search(r"(지난달|최근\s*1개월)[^\d]*(\d{1,6})\s*명", sent)
    if c1:
        res["cust_1m"] = float(c1.group(2))
    c3 = re.search(r"(3개월|최근\s*세\s*달|최근\s*3개월)[^\d]*(\d{1,6})\s*명", sent)
    if c3:
        res["cust_3m_avg"] = float(c3.group(2))
    if "sales_1m" not in res:
        anym = re.search(r"매출[^\d]*(\d[\d,\.]*\s*(원|만원|천만원|억원))", sent)
        if anym:
            res["sales_1m"] = _normalize_currency(anym.group(1))
    return {k: v for k, v in res.items() if v is not None}


def _extract_industry(sent: str) -> Optional[str]:
    for k, arr in INDUSTRY_MAP.items():
        for w in arr:
            if w in sent:
                return k
    return None


def _extract_region(sent: str) -> Optional[str]:
    m = re.search(r"([가-힣A-Za-z0-9]+(구|동|시|군|읍|면))", sent)
    return m.group(1) if m else None


def _extract_delivery_share(sent: str) -> Optional[float]:
    if "배달 위주" in sent or "배달 중심" in sent: return 0.8
    if "포장 위주" in sent or "테이크아웃 위주" in sent: return 0.6
    if "홀 위주" in sent or "내점 위주" in sent: return 0.3
    # 비율/퍼센트 수치
    m = re.search(r"(배달|딜리버리)[^\d]*(\d{1,3})\s*%", sent)
    if m:
        v = int(m.group(2))
        return max(0.0, min(1.0, v / 100.0))
    return None


def parse_utterance(utt: str) -> Dict:
    s = utt.strip()
    out = {}
    out.update(_extract_money(s))
    ind = _extract_industry(s)
    if ind: out["industry_code"] = ind
    reg = _extract_region(s)
    if reg: out["region_code"] = reg
    dshare = _extract_delivery_share(s)
    if dshare is not None: out["delivery_share"] = dshare
    return out