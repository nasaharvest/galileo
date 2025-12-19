import marimo

__generated_with = "0.9.30"
app = marimo.App(width="medium")


@app.cell
def intro():
    """
    # Making embeddings from real data

    This notebook demonstrates how to make embeddings with the Galileo models using real data (exported by our GEE exporter).

    Our GEE exporter is called using the following script:
    ```python
    from datetime import date

    from src.data import EarthEngineExporter
    from src.data.earthengine import EEBoundingBox

    # to export points
    EarthEngineExporter(dest_bucket="bucket_name").export_for_latlons(df)
    # to export a bounding box
    bbox = EEBoundingBox(min_lat=49.017835,min_lon-123.303680,max_lat=49.389519,max_lon-122.792816)
    EarthEngineExporter(dest_bucket="bucket_name").export_for_bbox(bbox, start_date=date(2024, 1, 1), end_date=(2025, 1, 1))
    ```
    """
    return


@app.cell
def imports():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from einops import rearrange
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from tqdm import tqdm

    from src.data.config import DATA_FOLDER, NORMALIZATION_DICT_FILENAME
    from src.data.dataset import Dataset, DatasetOutput, Normalizer
    from src.galileo import Encoder
    from src.masking import MaskedOutput
    from src.utils import config_dir

    return (
        DATA_FOLDER,
        Dataset,
        DatasetOutput,
        Encoder,
        KMeans,
        MaskedOutput,
        NORMALIZATION_DICT_FILENAME,
        Normalizer,
        PCA,
        Path,
        config_dir,
        np,
        plt,
        rearrange,
        torch,
        tqdm,
    )


@app.cell
def load_data(Dataset, NORMALIZATION_DICT_FILENAME, Normalizer, Path, config_dir):
    """
    First, we'll load a dataset output using one of the example training tifs in `data/tifs`. We also normalize it using the same normalization stats we used during training.
    """
    normalizing_dict = Dataset.load_normalization_values(
        path=config_dir / NORMALIZATION_DICT_FILENAME
    )
    normalizer = Normalizer(std=True, normalizing_dicts=normalizing_dict)

    dataset_output = Dataset._tif_to_array(
        Path(
            "data/tifs/min_lat=-27.6721_min_lon=25.6796_max_lat=-27.663_max_lon=25.6897_dates=2022-01-01_2023-12-31.tif"
        )
    ).normalize(normalizer)
    return dataset_output, normalizer, normalizing_dict


@app.cell
def visualize_data(dataset_output, np, plt):
    """
    This tif captures the Vaal river near the [Bloemhof dam](https://en.wikipedia.org/wiki/Bloemhof_Dam).
    We can visualize the S2-RGB bands from the first timestep:
    """
    plt.clf()
    plt.imshow(dataset_output.space_time_x[:, :, 0, [4, 3, 2]].astype(np.float32))
    plt.show()
    return


@app.cell
def load_model(DATA_FOLDER, Encoder):
    """
    We'll use the nano model (which is conveniently stored in git) to make these embeddings.
    """
    model = Encoder.load_from_folder(DATA_FOLDER / "models/nano")
    return (model,)


@app.cell
def define_embedding_function(Encoder, DatasetOutput, MaskedOutput, np, rearrange, torch, tqdm):
    from typing import Any

    def make_embeddings(
        model: Any,
        datasetoutput: Any,
        window_size: int,
        patch_size: int,
        batch_size: int = 128,
        device: Any = None,
    ) -> Any:
        if device is None:
            device = torch.device("cpu")
        model.eval()
        output_embeddings_list = []
        for i in tqdm(
            datasetoutput.in_pixel_batches(batch_size=batch_size, window_size=window_size)
        ):
            masked_output = MaskedOutput.from_datasetoutput(i, device=device)
            with torch.no_grad():
                output_embeddings_list.append(
                    model.average_tokens(
                        *model(
                            masked_output.space_time_x.float(),
                            masked_output.space_x.float(),
                            masked_output.time_x.float(),
                            masked_output.static_x.float(),
                            masked_output.space_time_mask,
                            masked_output.space_mask,
                            # lets mask inputs which will be the same for
                            # all pixels in the DatasetOutput
                            torch.ones_like(masked_output.time_mask),
                            torch.ones_like(masked_output.static_mask),
                            masked_output.months.long(),
                            patch_size=patch_size,
                        )[:-1]
                    )
                    .cpu()
                    .numpy()
                )
        output_embeddings = np.concatenate(output_embeddings_list, axis=0)
        # reshape the embeddings to H, W, D
        # first - how many "height batches" and "width batches" did we get?
        h_b = datasetoutput.space_time_x.shape[0] // window_size
        w_b = datasetoutput.space_time_x.shape[1] // window_size
        return rearrange(output_embeddings, "(h_b w_b) d -> h_b w_b d", h_b=h_b, w_b=w_b)

    return (make_embeddings,)


@app.cell
def generate_embeddings(dataset_output, make_embeddings, model, rearrange):
    embeddings = make_embeddings(model, dataset_output, 1, 1, 128)
    embeddings_flat = rearrange(embeddings, "h w d -> (h w) d")
    return embeddings, embeddings_flat


@app.cell
def cluster_embeddings(KMeans, embeddings, embeddings_flat, rearrange):
    labels = KMeans(n_clusters=3).fit_predict(embeddings_flat)
    labels = rearrange(labels, "(h w) -> h w", h=embeddings.shape[0], w=embeddings.shape[1])
    return (labels,)


@app.cell
def reduce_dimensions(PCA, embeddings, embeddings_flat, rearrange):
    embeddings_pca = PCA(n_components=3).fit_transform(embeddings_flat)
    embeddings_reduced = rearrange(
        embeddings_pca, "(h w) d -> h w d", h=embeddings.shape[0], w=embeddings.shape[1]
    )
    return embeddings_pca, embeddings_reduced


@app.cell
def plot_results(embeddings_reduced, labels, plt):
    # Plot the K-means clustering results
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(labels, cmap="tab10")
    plt.title("K-means Clustering (3 clusters)")
    plt.colorbar()

    # Plot the PCA-reduced embeddings as RGB
    plt.subplot(1, 3, 2)
    # Normalize to 0-1 range for display
    embeddings_normalized = (embeddings_reduced - embeddings_reduced.min()) / (
        embeddings_reduced.max() - embeddings_reduced.min()
    )
    plt.imshow(embeddings_normalized)
    plt.title("PCA-reduced embeddings (RGB)")

    plt.tight_layout()
    plt.show()
    return (embeddings_normalized,)


if __name__ == "__main__":
    app.run()
