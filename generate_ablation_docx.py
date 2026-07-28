"""
Generate a comprehensive Word Document (.docx) with the full ablation study results
for both 1,000-sample and 2,000-sample stratified runs on FakeAVCeleb.
Includes all 6 configurations (2 detectors × 3 backbones) with complete metrics.
"""

import json
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls


def set_cell_shading(cell, color_hex):
    shading_xml = f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>'
    cell._tc.get_or_add_tcPr().append(parse_xml(shading_xml))


def set_cell_text(cell, text, bold=False, size=9, align=WD_ALIGN_PARAGRAPH.CENTER):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = align
    run = p.add_run(str(text))
    run.font.size = Pt(size)
    run.font.name = "Arial"
    run.font.bold = bold
    return run


def add_styled_table(doc, headers, rows, header_color="1B365D"):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"

    # Header row
    for i, h in enumerate(headers):
        cell = table.cell(0, i)
        run = set_cell_text(cell, h, bold=True, size=9)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_shading(cell, header_color)

    # Data rows
    for r_idx, row_data in enumerate(rows):
        for c_idx, val in enumerate(row_data):
            cell = table.cell(r_idx + 1, c_idx)
            bold = (c_idx == 0)
            align = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            set_cell_text(cell, val, bold=bold, size=9, align=align)
            if r_idx % 2 == 1:
                set_cell_shading(cell, "F2F6FA")

    return table


def fmt(val, decimals=2):
    return f"{val:.{decimals}f}%"


def fmt_ms(val, std):
    return f"{val:.2f}% ± {std:.2f}%"


def main():
    base = Path(".")

    # Load results
    with open(base / "output/ablation_fakeavceleb_stratified/ablation_results.json") as f:
        res_1k = json.load(f)
    with open(base / "output/ablation_fakeavceleb_2000/ablation_results.json") as f:
        res_2k = json.load(f)

    doc = docx.Document()

    # Page setup
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.8)
        section.right_margin = Inches(0.8)

    style_normal = doc.styles["Normal"]
    style_normal.font.name = "Arial"
    style_normal.font.size = Pt(10)

    # ─────────────────── TITLE ───────────────────
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(4)
    run = title.add_run("ABLATION STUDY RESULTS REPORT")
    run.font.size = Pt(18)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.paragraph_format.space_after = Pt(6)
    run = sub.add_run("2×3 Detector–Backbone Grid on FakeAVCeleb v1.2")
    run.font.size = Pt(12)
    run.font.italic = True
    run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    sub2 = doc.add_paragraph()
    sub2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub2.paragraph_format.space_after = Pt(16)
    run = sub2.add_run("1,000-Sample (Balanced 1:1) and 2,000-Sample (Imbalanced 1:3) Stratified Evaluations")
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(0x77, 0x77, 0x77)

    # ─────────────────── PROTOCOL ───────────────────
    h = doc.add_heading("Evaluation Protocol", level=1)
    h.paragraph_format.space_before = Pt(12)

    doc.add_paragraph(
        "Dataset: FakeAVCeleb v1.2 (500 unique speaker identities)\n"
        "Cross-Validation: 5-fold identity-disjoint StratifiedGroupKFold\n"
        "Backbone Training: Frozen pretrained weights (linear probe / trainable fusion head only)\n"
        "Face Detectors: RetinaFace, MTCNN\n"
        "Visual Backbones: ViT-16 (google/vit-base-patch16-224), ViT-32 (google/vit-base-patch32-224-in21k), "
        "Swin-Tiny (microsoft/swin-tiny-patch4-window7-224)\n"
        "Fake Stratification: Equal draw across face-swap, lip-sync, and both sub-categories\n"
        "Determinism: Fixed seeds (42), CUBLAS_WORKSPACE_CONFIG, cudnn.deterministic=True"
    )

    # ─────────────────── DETECTOR DIAGNOSTICS ───────────────────
    doc.add_heading("Detector Diagnostics", level=1)

    diag_headers = ["Metric", "RetinaFace (1k)", "MTCNN (1k)", "RetinaFace (2k)", "MTCNN (2k)"]
    d1r = res_1k["detector_diagnostics"]["retinaface"]
    d1m = res_1k["detector_diagnostics"]["mtcnn"]
    d2r = res_2k["detector_diagnostics"]["retinaface"]
    d2m = res_2k["detector_diagnostics"]["mtcnn"]
    diag_rows = [
        ["Videos Attempted", str(d1r["videos_attempted"]), str(d1m["videos_attempted"]),
         str(d2r["videos_attempted"]), str(d2m["videos_attempted"])],
        ["Videos Kept", str(d1r["videos_kept"]), str(d1m["videos_kept"]),
         str(d2r["videos_kept"]), str(d2m["videos_kept"])],
        ["Videos Failed", str(d1r["videos_failed"]), str(d1m["videos_failed"]),
         str(d2r["videos_failed"]), str(d2m["videos_failed"])],
        ["Mean Face Det Rate %", fmt(d1r["mean_face_detection_rate_pct"], 1),
         fmt(d1m["mean_face_detection_rate_pct"], 1),
         fmt(d2r["mean_face_detection_rate_pct"], 1),
         fmt(d2m["mean_face_detection_rate_pct"], 1)],
        ["Mean Latency (ms/frame)", f'{d1r["mean_latency_ms_per_frame"]:.2f}',
         f'{d1m["mean_latency_ms_per_frame"]:.2f}',
         f'{d2r["mean_latency_ms_per_frame"]:.2f}',
         f'{d2m["mean_latency_ms_per_frame"]:.2f}'],
        ["Total Preprocess (min)", f'{d1r["total_preprocess_minutes"]:.1f}',
         f'{d1m["total_preprocess_minutes"]:.1f}',
         f'{d2r["total_preprocess_minutes"]:.1f}',
         f'{d2m["total_preprocess_minutes"]:.1f}'],
    ]
    add_styled_table(doc, diag_headers, diag_rows)
    doc.add_paragraph()

    # ─────────────────── HELPER: per-cell results ───────────────────
    cells = [
        ("RetinaFace + ViT-16", "retinaface+ViT-16"),
        ("RetinaFace + ViT-32", "retinaface+ViT-32"),
        ("RetinaFace + Swin-Tiny", "retinaface+Swin-Tiny"),
        ("MTCNN + ViT-16", "mtcnn+ViT-16"),
        ("MTCNN + ViT-32", "mtcnn+ViT-32"),
        ("MTCNN + Swin-Tiny", "mtcnn+Swin-Tiny"),
    ]

    def get_r(res, key):
        return res["results"][key]

    # ─────────────────── TABLE: AUC ───────────────────
    doc.add_heading("Table 1: Area Under the ROC Curve (AUC %)", level=1)
    doc.add_paragraph("AUC measures the probability that the system ranks a random deepfake higher than a random real video. "
                       "It is threshold-independent and robust to class imbalance. Higher is better.")

    auc_headers = ["Configuration", "1k AUC Mean", "1k AUC Std", "2k AUC Mean", "2k AUC Std", "Δ Mean"]
    auc_rows = []
    for name, key in cells:
        r1 = get_r(res_1k, key)
        r2 = get_r(res_2k, key)
        delta = r2["auc_mean"] - r1["auc_mean"]
        auc_rows.append([name, fmt(r1["auc_mean"]), fmt(r1["auc_std"]),
                         fmt(r2["auc_mean"]), fmt(r2["auc_std"]),
                         f"{delta:+.2f}%"])
    add_styled_table(doc, auc_headers, auc_rows)
    doc.add_paragraph()

    # ─────────────────── TABLE: EER ───────────────────
    doc.add_heading("Table 2: Equal Error Rate (EER %)", level=1)
    doc.add_paragraph("EER is the point where the false acceptance rate equals the false rejection rate. "
                       "It is threshold-independent. Lower is better.")

    eer_headers = ["Configuration", "1k EER Mean", "1k EER Std", "2k EER Mean", "2k EER Std", "Δ Mean"]
    eer_rows = []
    for name, key in cells:
        r1 = get_r(res_1k, key)
        r2 = get_r(res_2k, key)
        delta = r2["eer_mean"] - r1["eer_mean"]
        eer_rows.append([name, fmt(r1["eer_mean"]), fmt(r1["eer_std"]),
                         fmt(r2["eer_mean"]), fmt(r2["eer_std"]),
                         f"{delta:+.2f}%"])
    add_styled_table(doc, eer_headers, eer_rows)
    doc.add_paragraph()

    # ─────────────────── TABLE: Accuracy ───────────────────
    doc.add_heading("Table 3: Classification Accuracy (%)", level=1)
    doc.add_paragraph("Accuracy measures the proportion of correct predictions at a fixed threshold of 0.50. "
                       "Higher is better. Note: raw accuracy is inflated in the 2,000-sample run due to the 1:3 class imbalance.")

    acc_headers = ["Configuration", "1k Acc Mean", "1k Acc Std", "2k Acc Mean", "2k Acc Std", "Δ Mean"]
    acc_rows = []
    for name, key in cells:
        r1 = get_r(res_1k, key)
        r2 = get_r(res_2k, key)
        delta = r2["acc_mean"] - r1["acc_mean"]
        acc_rows.append([name, fmt(r1["acc_mean"]), fmt(r1["acc_std"]),
                         fmt(r2["acc_mean"]), fmt(r2["acc_std"]),
                         f"{delta:+.2f}%"])
    add_styled_table(doc, acc_headers, acc_rows)
    doc.add_paragraph()

    # ─────────────────── TABLE: Balanced Accuracy ───────────────────
    doc.add_heading("Table 4: Balanced Accuracy (%)", level=1)
    doc.add_paragraph("Balanced Accuracy is the unweighted average of per-class recall (sensitivity and specificity). "
                       "It corrects for class imbalance and provides a fair comparison across both dataset scales. Higher is better.")

    bacc_headers = ["Configuration", "1k Bal Acc Mean", "1k Bal Acc Std", "2k Bal Acc Mean", "2k Bal Acc Std", "Δ Mean"]
    bacc_rows = []
    for name, key in cells:
        r1 = get_r(res_1k, key)
        r2 = get_r(res_2k, key)
        delta = r2["bal_acc_mean"] - r1["bal_acc_mean"]
        bacc_rows.append([name, fmt(r1["bal_acc_mean"]), fmt(r1["bal_acc_std"]),
                          fmt(r2["bal_acc_mean"]), fmt(r2["bal_acc_std"]),
                          f"{delta:+.2f}%"])
    add_styled_table(doc, bacc_headers, bacc_rows)
    doc.add_paragraph()

    # ─────────────────── TABLE: F1-Score ───────────────────
    doc.add_heading("Table 5: F1-Score (%)", level=1)
    doc.add_paragraph("F1-Score is the harmonic mean of Precision and Recall at a fixed threshold of 0.50. "
                       "It balances false alarms and missed detections. Higher is better. "
                       "Note: F1 is inflated in the 2,000-sample run because the high fake-class prevalence boosts Recall.")

    f1_headers = ["Configuration", "1k F1 Mean", "1k F1 Std", "2k F1 Mean", "2k F1 Std", "Δ Mean"]
    f1_rows = []
    for name, key in cells:
        r1 = get_r(res_1k, key)
        r2 = get_r(res_2k, key)
        delta = r2["f1_mean"] - r1["f1_mean"]
        f1_rows.append([name, fmt(r1["f1_mean"]), fmt(r1["f1_std"]),
                        fmt(r2["f1_mean"]), fmt(r2["f1_std"]),
                        f"{delta:+.2f}%"])
    add_styled_table(doc, f1_headers, f1_rows)
    doc.add_paragraph()

    # ─────────────────── TABLE: Per-Category AUC ───────────────────
    doc.add_heading("Table 6: Per-Manipulation-Type AUC (%)", level=1)
    doc.add_paragraph(
        "This table isolates performance by deepfake category (reals vs. one specific fake type). "
        "Face-Swap = visual-only forgery (manipulated video, real audio). "
        "Lip-Sync = audio-only forgery (real video, synthesized audio). "
        "Both = audio-visual forgery (both video and audio manipulated)."
    )

    cat_headers = ["Configuration", "Face-Swap (1k)", "Face-Swap (2k)",
                   "Lip-Sync (1k)", "Lip-Sync (2k)",
                   "Both (1k)", "Both (2k)"]
    cat_rows = []
    for name, key in cells:
        r1 = get_r(res_1k, key)["per_category_auc"]
        r2 = get_r(res_2k, key)["per_category_auc"]
        cat_rows.append([
            name,
            fmt(r1["face-swap"]["auc"]), fmt(r2["face-swap"]["auc"]),
            fmt(r1["lip-sync"]["auc"]), fmt(r2["lip-sync"]["auc"]),
            fmt(r1["both"]["auc"]), fmt(r2["both"]["auc"]),
        ])
    add_styled_table(doc, cat_headers, cat_rows)
    doc.add_paragraph()

    # ─────────────────── PER-FOLD TABLES ───────────────────
    doc.add_heading("Per-Fold Breakdown", level=1)
    doc.add_paragraph(
        "The following tables show the metrics for each individual fold (Fold 1–5) of the 5-fold "
        "identity-disjoint cross-validation, for each of the 6 configurations. This allows verification "
        "of metric stability across identity subsets."
    )

    for name, key in cells:
        doc.add_heading(f"{name}", level=2)

        fold_headers = ["Fold", "AUC %", "EER %", "Acc %", "Bal Acc %", "F1 %"]

        for label, res_data in [("1,000-Sample", res_1k), ("2,000-Sample", res_2k)]:
            r = get_r(res_data, key)
            doc.add_heading(f"{label} Run", level=3)

            fold_rows = []
            for i, fold in enumerate(r["folds"]):
                fold_rows.append([
                    f"Fold {i + 1}",
                    fmt(fold["auc"]),
                    fmt(fold["eer"]),
                    fmt(fold["acc"]),
                    fmt(fold["bal_acc"]),
                    fmt(fold["f1"]),
                ])
            fold_rows.append([
                "Mean ± Std",
                fmt_ms(r["auc_mean"], r["auc_std"]),
                fmt_ms(r["eer_mean"], r["eer_std"]),
                fmt_ms(r["acc_mean"], r["acc_std"]),
                fmt_ms(r["bal_acc_mean"], r["bal_acc_std"]),
                fmt_ms(r["f1_mean"], r["f1_std"]),
            ])
            add_styled_table(doc, fold_headers, fold_rows)
            doc.add_paragraph()

    # ─────────────────── SAVE ───────────────────
    out_path = Path("results/ablation-stratified/Ablation_Study_Full_Results.docx")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_path)
    print(f"Report saved to: {out_path.absolute()}")


if __name__ == "__main__":
    main()
