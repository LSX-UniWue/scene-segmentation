from typing import List, Sequence

from ssc.model import Label

border_labels = {Label.BORDER, Label.S2NS, Label.S2S, Label.NS2S}


def evaluate_predictions(true_labels: Sequence[Label], predicted_labels: Sequence[Label], tolerance: int) -> (
        float, float, float):
    assert len(true_labels) == len(predicted_labels)
    assert tolerance >= 0, "Tolerance must be non-negative"

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for i, label in enumerate(true_labels):
        if label in border_labels:
            if label in (
                    predicted_labels[max(0, i - tolerance):i + tolerance] if tolerance > 0 else [predicted_labels[i]]):
                true_positives += 1
            else:
                false_negatives += 1

    for i, label in enumerate(predicted_labels):
        if label in border_labels:
            if label not in (true_labels[max(0, i - tolerance):i + tolerance] if tolerance > 0 else [true_labels[i]]):
                false_positives += 1

    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_predictions_iou(true_labels: Sequence[Label], predicted_labels: Sequence[Label]) -> float:
    assert len(true_labels) == len(predicted_labels)

    def extract_scene_spans(labels, border_labels):
        spans = []
        start = None
        for i, label in enumerate(labels):
            if label in border_labels:
                if start is not None:
                    spans.append((start, i - 1))
                    start = None
            else:
                if start is None:
                    start = i
        if start is not None:
            spans.append((start, len(labels) - 1))
        return spans

    true_spans = extract_scene_spans(true_labels, border_labels)
    predicted_spans = extract_scene_spans(predicted_labels, border_labels)

    total_overlap_length = 0
    assigned_predicted = set()

    for t_start, t_end in true_spans:
        max_overlap = 0
        best_overlap = (0, 0)
        for p_index, (p_start, p_end) in enumerate(predicted_spans):
            if p_index in assigned_predicted:
                continue

            # Calculate overlap
            overlap_start = max(t_start, p_start)
            overlap_end = min(t_end, p_end)
            if overlap_start <= overlap_end:  # There's an overlap
                overlap_length = overlap_end - overlap_start + 1
                if overlap_length > max_overlap:
                    max_overlap = overlap_length
                    best_overlap = (p_index, overlap_start, overlap_end)

        if max_overlap > 0:
            assigned_predicted.add(best_overlap[0])  # Mark the predicted span as assigned
            total_overlap_length += max_overlap

    total_length = len(true_labels)
    iou = total_overlap_length / total_length if total_length > 0 else 0
    return iou
