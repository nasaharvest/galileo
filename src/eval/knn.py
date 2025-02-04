import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from .metrics import class_wise_f1


def run_knn(
    eval_type,
    train_embeddings,
    train_labels,
    test_embeddings,
    test_labels,
    num_classes,
    is_multilabel,
    device,
    skip_idx=False,
    return_class_wise=False,
):
    if not is_multilabel:
        if eval_type == "KNN-5":
            predictions = _run_knn_for_k(
                train_embeddings=train_embeddings,
                train_labels=train_labels,
                test_embeddings=test_embeddings,
                num_classes=num_classes,
                k=5,
                device=device,
                skip_idx=skip_idx,
            )
        elif eval_type == "KNN-20":
            predictions = _run_knn_for_k(
                train_embeddings=train_embeddings,
                train_labels=train_labels,
                test_embeddings=test_embeddings,
                num_classes=num_classes,
                k=20,
                device=device,
                skip_idx=skip_idx,
            )

        if return_class_wise:
            return class_wise_f1(y_true=test_labels, y_pred=predictions, num_classes=num_classes)
        return accuracy_score(y_true=test_labels, y_pred=predictions)
    else:
        # multilabel dataset, e.g., BigEarthNet
        # we will run KNN or K-Means once per class to compute predictions
        # labels are shape (num_samples, num_classes)
        assert num_classes == train_labels.shape[-1]
        assert num_classes == test_labels.shape[-1]
        predictions = []
        for class_idx in range(num_classes):
            train_single_labels = train_labels[:, class_idx]  # (num_samples)

            if eval_type == "KNN-5":
                single_predictions = _run_knn_for_k(
                    train_embeddings=train_embeddings,
                    train_labels=train_single_labels,
                    test_embeddings=test_embeddings,
                    num_classes=2,  # binary prediction for each class
                    k=5,
                    device=device,
                    skip_idx=skip_idx,
                )  # (num_samples)
            if eval_type == "KNN-20":
                single_predictions = _run_knn_for_k(
                    train_embeddings=train_embeddings,
                    train_labels=train_single_labels,
                    test_embeddings=test_embeddings,
                    num_classes=2,  # binary prediction for each class
                    k=20,
                    device=device,
                    skip_idx=skip_idx,
                )  # (num_samples)
            predictions.append(single_predictions)

        predictions = torch.stack(predictions, dim=1)  # (num_samples, num_classes)

        if return_class_wise:
            return [f1_score(test_labels[:, i], predictions[:, i]) for i in range(num_classes)]
        else:
            return f1_score(y_true=test_labels, y_pred=predictions, average="micro")


def _run_knn_for_k(
    train_embeddings, train_labels, test_embeddings, num_classes, k, device, skip_idx
):
    train_embeddings = train_embeddings.to(device)
    test_embeddings = test_embeddings.to(device)
    train_labels = train_labels.to(device)
    cos = nn.CosineSimilarity(dim=-1)
    all_preds = []
    for idx in range(test_embeddings.shape[0]):
        test_embedding = test_embeddings[idx].unsqueeze(dim=0).repeat(train_embeddings.shape[0], 1)
        sims = cos(test_embedding, train_embeddings)
        top_k = torch.topk(sims, k=k)
        if skip_idx:
            top_k_values = top_k.values[1:]
            top_k_indices = top_k.indices[1:]
        else:
            top_k_values = top_k.values
            top_k_indices = top_k.indices

        fetched_labels = train_labels[top_k_indices]
        fetched_onehots = nn.functional.one_hot(fetched_labels, num_classes=num_classes)
        distances = top_k_values.clone().div_(0.07).exp_()
        weighted_sum_onehots = (distances.unsqueeze(dim=1) * fetched_onehots).sum(dim=0)
        prediction = torch.argmax(weighted_sum_onehots)
        all_preds.append(prediction)

    return torch.LongTensor(all_preds).cpu()
