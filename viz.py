import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_csv_smart(path):
    for enc in ["utf-8", "cp949", "euc-kr", "latin1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    raise RuntimeError(f"CSV 인코딩 해석 실패: {path}")


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def plot_ts(ax, x, y, title, xlabel="TA_YM", ylabel="value"):
    ax.plot(x, y, marker='o')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)


def savefig(fig, out_path):
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="model_test 루트 경로")
    ap.add_argument("--store", type=str, default=None, help="특정 매장 ID(ENCODED_MCT)")
    ap.add_argument("--all", action="store_true", help="모든 매장에 대해 요약 그래프 생성")
    ap.add_argument("--topk", type=int, default=20, help="상위 위험 Top-K (기본 20)")
    args = ap.parse_args()

    risk_path = os.path.join(args.root, "risk_output.csv")
    df = read_csv_smart(risk_path)

    if "TA_YM" in df.columns:
        df["TA_YM"] = pd.to_datetime(df["TA_YM"], errors="coerce")

    fig_dir = os.path.join(args.root, "figures")
    ensure_dir(fig_dir)

    # p_final 분포 히스토그램
    if "p_final" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df["p_final"].fillna(0), bins=30)
        ax.set_title("Distribution of p_final")
        ax.set_xlabel("p_final")
        ax.set_ylabel("count")
        savefig(fig, os.path.join(fig_dir, "dist_p_final.png"))

    # Alert 카테고리 분포
    if "Alert" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        vc = df["Alert"].value_counts(dropna=False)
        ax.bar(vc.index.astype(str), vc.values)
        ax.set_title("Alert Distribution")
        ax.set_xlabel("Alert")
        ax.set_ylabel("count")
        savefig(fig, os.path.join(fig_dir, "alert_counts.png"))

    # 월별 평균 p_final 라인
    if "TA_YM" in df.columns and "p_final" in df.columns:
        tmp = df.groupby("TA_YM")["p_final"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(9, 5))
        plot_ts(ax, tmp["TA_YM"], tmp["p_final"], "Monthly mean of p_final")
        savefig(fig, os.path.join(fig_dir, "monthly_mean_p_final.png"))

    # 특정 매장 분석
    if args.store is not None:
        g = df[df["ENCODED_MCT"] == args.store].sort_values("TA_YM")
        if g.empty:
            print(f"[warn] store {args.store} 데이터가 없습니다.")
        else:
            fig, ax = plt.subplots(figsize=(9, 5))
            plot_ts(ax, g["TA_YM"], g["p_final"], f"Store {args.store} - p_final timeline")
            savefig(fig, os.path.join(fig_dir, f"store_{args.store}_pfinal.png"))

            for col in ["Sales_Risk", "Customer_Risk", "Market_Risk", "RiskScore"]:
                if col in g.columns:
                    fig, ax = plt.subplots(figsize=(9, 5))
                    plot_ts(ax, g["TA_YM"], g[col], f"Store {args.store} - {col} timeline", ylabel=col)
                    savefig(fig, os.path.join(fig_dir, f"store_{args.store}_{col}.png"))

    if args.all and "p_final" in df.columns:
        top = df.sort_values("p_final", ascending=False).head(args.topk)
        top_path = os.path.join(fig_dir, f"top{args.topk}_pfinal.csv")
        top.to_csv(top_path, index=False, encoding="utf-8")
        print(f"[info] Saved Top-K rows: {top_path}")

    print(f"[done] figures saved under: {fig_dir}")


if __name__ == "__main__":
    main()