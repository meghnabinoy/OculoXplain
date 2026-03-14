import argparse
import json
import re
from pathlib import Path


EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/(\d+)\s+\|\s+"
    r"train_loss=([0-9.]+)\s+train_acc=([0-9.]+)\s+train_f1=([0-9.]+)\s+\|\s+"
    r"val_loss=([0-9.]+)\s+val_acc=([0-9.]+)\s+val_f1=([0-9.]+)"
)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_log(path: Path) -> dict:
    if not path.exists():
        return {}

    text = None
    for enc in ("utf-8", "utf-16", "utf-16-le", "utf-16-be"):
        try:
            text = path.read_text(encoding=enc)
            break
        except UnicodeError:
            continue
    if text is None:
        text = path.read_text(encoding="utf-8", errors="ignore")

    best_val = None
    final_epoch = None
    total_epochs = None

    for line in text.splitlines():
        m = EPOCH_RE.search(line)
        if not m:
            continue
        epoch_i = int(m.group(1))
        epoch_n = int(m.group(2))
        row = {
            "epoch": epoch_i,
            "train_loss": float(m.group(3)),
            "train_acc": float(m.group(4)),
            "train_f1": float(m.group(5)),
            "val_loss": float(m.group(6)),
            "val_acc": float(m.group(7)),
            "val_f1": float(m.group(8)),
        }
        total_epochs = epoch_n
        final_epoch = row
        if best_val is None or row["val_f1"] > best_val["val_f1"]:
            best_val = row

    out = {}
    if best_val is not None:
        out["best_log_val"] = best_val
    if final_epoch is not None:
        out["final_log_epoch"] = final_epoch
    if total_epochs is not None:
        out["scheduled_epochs"] = total_epochs
    return out


def model_summary(name: str, metrics: dict, log_data: dict) -> dict:
    report = metrics.get("classification_report", {})
    macro_f1 = report.get("macro avg", {}).get("f1-score")
    weighted_f1 = report.get("weighted avg", {}).get("f1-score")

    out = {
        "model": name,
        "best_epoch": metrics.get("best_epoch"),
        "best_val_macro_f1": metrics.get("best_val_macro_f1"),
        "test_accuracy": metrics.get("test_accuracy"),
        "test_macro_f1": metrics.get("test_macro_f1"),
        "test_loss": metrics.get("test_loss"),
        "test_report_macro_f1": macro_f1,
        "test_report_weighted_f1": weighted_f1,
    }
    out.update(log_data)
    return out


def fmt(v, pct=False):
    if v is None:
        return "n/a"
    if pct:
        return f"{v * 100:6.2f}%"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def print_summary_row(s: dict):
    print(f"Model: {s['model']}")
    print(f"  best_epoch:            {fmt(s.get('best_epoch'))}")
    print(f"  best_val_macro_f1:     {fmt(s.get('best_val_macro_f1'))}")
    print(f"  test_accuracy:         {fmt(s.get('test_accuracy'), pct=True)}")
    print(f"  test_macro_f1:         {fmt(s.get('test_macro_f1'))}")
    print(f"  test_loss:             {fmt(s.get('test_loss'))}")
    print(f"  test_macro_f1(report): {fmt(s.get('test_report_macro_f1'))}")
    print(f"  test_weighted_f1:      {fmt(s.get('test_report_weighted_f1'))}")

    best_log = s.get("best_log_val")
    if best_log:
        print(
            "  best_log_epoch:        "
            f"{best_log['epoch']} val_acc={fmt(best_log['val_acc'], pct=True)} "
            f"val_f1={fmt(best_log['val_f1'])}"
        )

    final_log = s.get("final_log_epoch")
    if final_log:
        print(
            "  final_log_epoch:       "
            f"{final_log['epoch']} train_acc={fmt(final_log['train_acc'], pct=True)} "
            f"val_acc={fmt(final_log['val_acc'], pct=True)} val_f1={fmt(final_log['val_f1'])}"
        )


def print_delta(a: dict, b: dict):
    print("\nDelta (A - B):")

    def d(key):
        av = a.get(key)
        bv = b.get(key)
        if av is None or bv is None:
            return None
        return av - bv

    for key, pct in [
        ("test_accuracy", True),
        ("test_macro_f1", False),
        ("best_val_macro_f1", False),
        ("test_loss", False),
    ]:
        dv = d(key)
        label = key.ljust(22)
        if dv is None:
            print(f"  {label} n/a")
            continue
        if pct:
            print(f"  {label} {dv * 100:+.2f} pp")
        else:
            print(f"  {label} {dv:+.4f}")


def train_acc_at_best_epoch(s: dict):
    best_epoch = s.get("best_epoch")
    best_log = s.get("best_log_val")
    final_log = s.get("final_log_epoch")

    if best_log and best_epoch == best_log.get("epoch"):
        return best_log.get("train_acc")
    if final_log and best_epoch == final_log.get("epoch"):
        return final_log.get("train_acc")
    return None


def print_table(rows: list[dict]):
    headers = [
        "Model",
        "BestEpoch",
        "TrainAccAtBestEpoch",
        "BestValMacroF1",
        "TestAccuracy",
        "TestMacroF1",
    ]

    table_rows = []
    for s in rows:
        table_rows.append(
            [
                str(s.get("model", "n/a")),
                str(s.get("best_epoch", "n/a")),
                f"{train_acc_at_best_epoch(s):.4f}" if train_acc_at_best_epoch(s) is not None else "n/a",
                f"{s.get('best_val_macro_f1'):.4f}" if s.get("best_val_macro_f1") is not None else "n/a",
                f"{s.get('test_accuracy'):.4f}" if s.get("test_accuracy") is not None else "n/a",
                f"{s.get('test_macro_f1'):.4f}" if s.get("test_macro_f1") is not None else "n/a",
            ]
        )

    widths = [len(h) for h in headers]
    for row in table_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def render_row(row_cells):
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row_cells))

    print(render_row(headers))
    print("  ".join("-" * w for w in widths))
    for row in table_rows:
        print(render_row(row))


def main():
    parser = argparse.ArgumentParser(
        description="Compare ResNet50 and MobileNetV2 metrics for merged RFMiD runs"
    )
    parser.add_argument(
        "--resnet-metrics",
        default="resnet50_merged_rfmid_metrics.json",
        help="Path to ResNet50 metrics JSON",
    )
    parser.add_argument(
        "--mobilenet-metrics",
        default="mobilenetv2_merged_rfmid_pretrained_metrics.json",
        help="Path to MobileNetV2 metrics JSON",
    )
    parser.add_argument(
        "--resnet-log",
        default="outputs/train_full.log",
        help="Path to ResNet50 training log",
    )
    parser.add_argument(
        "--mobilenet-log",
        default="outputs/train_mobilenet_full.log",
        help="Path to MobileNetV2 training log",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Print verbose per-model summary and deltas after the compact table",
    )
    args = parser.parse_args()

    resnet_metrics = load_json(Path(args.resnet_metrics))
    mobilenet_metrics = load_json(Path(args.mobilenet_metrics))

    resnet = model_summary(
        "ResNet50",
        resnet_metrics,
        parse_log(Path(args.resnet_log)),
    )
    mobilenet = model_summary(
        "MobileNetV2",
        mobilenet_metrics,
        parse_log(Path(args.mobilenet_log)),
    )

    print_table([resnet, mobilenet])

    if args.detailed:
        print("\n" + "=" * 90)
        print("Merged RFMiD Model Comparison")
        print("=" * 90)
        print_summary_row(resnet)
        print()
        print_summary_row(mobilenet)
        print_delta(resnet, mobilenet)

    print("\nRecommendation:")
    if (resnet.get("test_macro_f1") or -1) > (mobilenet.get("test_macro_f1") or -1):
        print("  Best pure accuracy/F1: ResNet50")
    else:
        print("  Best pure accuracy/F1: MobileNetV2")
    print("  Best speed/CPU efficiency at inference: MobileNetV2")


if __name__ == "__main__":
    main()
