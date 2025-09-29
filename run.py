import os, sys, pandas as pd


def read_csv_smart(path):
    for enc in ["utf-8", "cp949", "euc-kr", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError(f"CSV 인코딩 해석 실패: {path}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, base_dir)

    from pipeline import run_pipeline  # model_test 안에 pipeline.py 필요
    data_dir = os.path.join(base_dir, "data")

    ds1 = read_csv_smart(os.path.join(data_dir, "big_data_set1_f.csv"))
    ds2 = read_csv_smart(os.path.join(data_dir, "ds2_monthly_usage.csv"))
    ds3 = read_csv_smart(os.path.join(data_dir, "ds3_monthly_customers.csv"))

    preds_path = os.path.join(data_dir, "preds.csv")
    preds = read_csv_smart(preds_path) if os.path.exists(preds_path) else None

    out = run_pipeline(ds1, ds2, ds3, preds)
    out_path = os.path.join(base_dir, "risk_output.csv")
    out.to_csv(out_path, index=False)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()