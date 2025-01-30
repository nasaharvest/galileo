import numpy as np
import torch
import torch.nn as nn
from fast_pytorch_kmeans import KMeans
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, f1_score

from .metrics import class_wise_f1


def hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
    # Based on implementation from IIC
    num_samples = flat_targets.shape[0]

    assert preds_k == targets_k  # one to one
    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes

    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))

    res = {}
    for out_c, gt_c in match:
        res[out_c] = gt_c

    return res


def run_knn_or_kmeans(
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
            predictions = run_knn(
                train_embeddings=train_embeddings,
                train_labels=train_labels,
                test_embeddings=test_embeddings,
                num_classes=num_classes,
                k=5,
                device=device,
                skip_idx=skip_idx,
            )
        elif eval_type == "KNN-20":
            predictions = run_knn(
                train_embeddings=train_embeddings,
                train_labels=train_labels,
                test_embeddings=test_embeddings,
                num_classes=num_classes,
                k=20,
                device=device,
                skip_idx=skip_idx,
            )
        else:
            predictions = run_kmeans(
                train_embeddings=train_embeddings,
                test_embeddings=test_embeddings,
                test_labels=test_labels,
                num_classes=num_classes,
                device=device,
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
            test_single_labels = test_labels[:, class_idx]  # (num_samples)

            if eval_type == "KNN-5":
                single_predictions = run_knn(
                    train_embeddings=train_embeddings,
                    train_labels=train_single_labels,
                    test_embeddings=test_embeddings,
                    num_classes=2,  # binary prediction for each class
                    k=5,
                    device=device,
                    skip_idx=skip_idx,
                )  # (num_samples)
            if eval_type == "KNN-20":
                single_predictions = run_knn(
                    train_embeddings=train_embeddings,
                    train_labels=train_single_labels,
                    test_embeddings=test_embeddings,
                    num_classes=2,  # binary prediction for each class
                    k=20,
                    device=device,
                    skip_idx=skip_idx,
                )  # (num_samples)
            else:
                single_predictions = run_kmeans(
                    train_embeddings=train_embeddings,
                    test_embeddings=test_embeddings,
                    test_labels=test_single_labels,
                    num_classes=2,  # binary prediction for each class
                    device=device,
                )  # (num_samples)
            predictions.append(single_predictions)

        predictions = torch.stack(predictions, dim=1)  # (num_samples, num_classes)

        if return_class_wise:
            return [f1_score(test_labels[:, i], predictions[:, i]) for i in range(num_classes)]
        else:
            return f1_score(y_true=test_labels, y_pred=predictions, average="micro")


def run_kmeans(train_embeddings, test_embeddings, test_labels, num_classes, device, num_runs=10):
    # K-Means is not deterministic, let's run it 10 times and take the mode predictions
    # This GPU implementation is fast so this is not a big slow-down
    all_predictions = []
    for _ in range(num_runs):
        kmeans = KMeans(
            n_clusters=num_classes, mode="euclidean", init_method="random", max_iter=10_000
        )
        kmeans.fit(train_embeddings.float().to(device))

        predictions = kmeans.predict(test_embeddings.float().to(device)).cpu().numpy()
        targets = test_labels.numpy()
        res = hungarian_match(predictions, targets, num_classes, num_classes)

        remapped_predictions = np.zeros(predictions.shape[0])
        for i in range(predictions.shape[0]):
            remapped_predictions[i] = res[predictions[i]]

        all_predictions.append(remapped_predictions)

    combined_predictions = stats.mode(all_predictions, axis=0).mode.flatten()
    return torch.LongTensor(combined_predictions)


def run_knn(train_embeddings, train_labels, test_embeddings, num_classes, k, device, skip_idx):
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
