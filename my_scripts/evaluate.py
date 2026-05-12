import os
import json
import csv
import argparse


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_free_generation(rows):
    n = len(rows)
    if n == 0:
        return {"num_samples": 0, "ES": 0, "PS": 0, "NS": 0}

    es = sum(1 for x in rows if x.get("ES_hit")) / n * 100
    ps = sum(1 for x in rows if x.get("PS_hit")) / n * 100
    ns = sum(1 for x in rows if x.get("NS_hit")) / n * 100

    return {
        "num_samples": n,
        "ES": round(es, 2),
        "PS": round(ps, 2),
        "NS": round(ns, 2),
    }


def summarize_easyedit_rome(rows):
    values = []
    for x in rows:
        v = x.get("easyedit_rewrite_acc")
        if v is not None:
            values.append(float(v))

    if not values:
        return None

    return round(sum(values) / len(values) * 100, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rome", default="outputs/rome_results_gpt2xl.json")
    parser.add_argument("--memit", default="outputs/memit_debug_10_gpt2xl.json")
    parser.add_argument("--out_json", default="outputs/metrics_gpt2xl.json")
    parser.add_argument("--out_csv", default="outputs/metrics_gpt2xl.csv")
    args = parser.parse_args()

    summary = {}

    if os.path.exists(args.rome):
        rome_rows = load_json(args.rome)
        rome_summary = summarize_free_generation(rome_rows)
        rome_summary["EasyEdit_rewrite_acc"] = summarize_easyedit_rome(rome_rows)
        summary["ROME"] = rome_summary

    if os.path.exists(args.memit):
        memit_obj = load_json(args.memit)
        memit_rows = memit_obj["results"] if isinstance(memit_obj, dict) else memit_obj
        memit_summary = summarize_free_generation(memit_rows)

        if isinstance(memit_obj, dict):
            memit_summary["time_seconds"] = memit_obj.get("time_seconds")
            memit_summary["peak_gpu_memory_gb"] = memit_obj.get("peak_gpu_memory_gb")

        summary["MEMIT"] = memit_summary

    os.makedirs("outputs", exist_ok=True)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method",
            "num_samples",
            "ES_free_generation",
            "PS_free_generation",
            "NS_free_generation",
            "EasyEdit_rewrite_acc",
            "time_seconds",
            "peak_gpu_memory_gb",
        ])

        for method, item in summary.items():
            writer.writerow([
                method,
                item.get("num_samples", ""),
                item.get("ES", ""),
                item.get("PS", ""),
                item.get("NS", ""),
                item.get("EasyEdit_rewrite_acc", ""),
                item.get("time_seconds", ""),
                item.get("peak_gpu_memory_gb", ""),
            ])

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved {args.out_json}")
    print(f"Saved {args.out_csv}")


if __name__ == "__main__":
    main()