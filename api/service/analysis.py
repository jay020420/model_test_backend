# api/service/analysis.py
async def get_benchmark(industry_code: str, region_code: Optional[str], metric: str) -> Dict:
    """벤치마크 분석"""
    from ..loader import load_risk_output
    import pandas as pd
    
    df = load_risk_output()
    if df is None:
        raise Exception("벤치마크 데이터를 로드할 수 없습니다")
    
    # 필터링 (실제로는 더 정교하게)
    metric_values = df[metric].dropna()
    
    return {
        "industry_code": industry_code,
        "region_code": region_code,
        "metric": metric,
        "industry_avg": float(metric_values.mean()),
        "industry_median": float(metric_values.median()),
        "industry_25th": float(metric_values.quantile(0.25)),
        "industry_75th": float(metric_values.quantile(0.75)),
        "sample_size": len(metric_values)
    }
