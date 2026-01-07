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
    print("🚀 STARTING: Importing libraries...")
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

    print("✅ SUCCESS: All libraries imported successfully!")
    print(f"📊 Using PyTorch version: {torch.__version__}")
    print(f"📊 Using NumPy version: {np.__version__}")

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
    print("🔄 STEP 1: Loading normalization values...")
    normalizing_dict = Dataset.load_normalization_values(
        path=config_dir / NORMALIZATION_DICT_FILENAME
    )
    print(f"✅ Loaded normalization dict with {len(normalizing_dict)} entries")
    print(f"📁 Config directory: {config_dir}")
    print(f"📄 Normalization file: {NORMALIZATION_DICT_FILENAME}")

    print("🔄 Creating normalizer...")
    normalizer = Normalizer(std=True, normalizing_dicts=normalizing_dict)
    print("✅ Normalizer created successfully")

    print("🔄 Loading TIF file...")
    tif_path = Path(
        "data/tifs/min_lat=-27.6721_min_lon=25.6796_max_lat=-27.663_max_lon=25.6897_dates=2022-01-01_2023-12-31.tif"
    )
    print(f"📁 TIF path: {tif_path}")
    print(f"📊 TIF exists: {tif_path.exists()}")

    dataset_output = Dataset._tif_to_array(tif_path).normalize(normalizer)
    print("✅ TIF loaded and normalized successfully!")
    print(f"📊 Dataset output type: {type(dataset_output)}")
    print(f"📊 Space-time data shape: {dataset_output.space_time_x.shape}")
    print(
        f"📊 Space data shape: {dataset_output.space_x.shape if hasattr(dataset_output, 'space_x') else 'N/A'}"
    )
    print(
        f"📊 Time data shape: {dataset_output.time_x.shape if hasattr(dataset_output, 'time_x') else 'N/A'}"
    )
    print(
        f"📊 Static data shape: {dataset_output.static_x.shape if hasattr(dataset_output, 'static_x') else 'N/A'}"
    )

    return dataset_output, normalizer, normalizing_dict


@app.cell
def visualize_data(dataset_output, np, plt):
    """
    This tif captures the Vaal river near the [Bloemhof dam](https://en.wikipedia.org/wiki/Bloemhof_Dam).
    We can visualize the S2-RGB bands from the first timestep:
    """
    print("🔄 STEP 2: Visualizing S2-RGB bands...")
    print("📊 Extracting RGB bands [4, 3, 2] from timestep 0")
    rgb_data = dataset_output.space_time_x[:, :, 0, [4, 3, 2]].astype(np.float32)
    print(f"📊 RGB data shape: {rgb_data.shape}")
    print(f"📊 RGB data range: [{rgb_data.min():.3f}, {rgb_data.max():.3f}]")

    plt.clf()
    plt.imshow(rgb_data)
    plt.title("S2-RGB bands from first timestep")
    plt.show()
    print("✅ RGB visualization complete!")
    return


@app.cell
def load_model(DATA_FOLDER, Encoder):
    """
    We'll use the nano model (which is conveniently stored in git) to make these embeddings.
    """
    print("🔄 STEP 3: Loading Galileo nano model...")
    model_path = DATA_FOLDER / "models/nano"
    print(f"📁 Model path: {model_path}")
    print(f"📊 Model path exists: {model_path.exists()}")

    model = Encoder.load_from_folder(model_path)
    print("✅ Model loaded successfully!")
    print(f"📊 Model type: {type(model)}")
    print(f"📊 Model device: {next(model.parameters()).device}")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    return (model,)


@app.cell
def define_embedding_function(Encoder, DatasetOutput, MaskedOutput, np, rearrange, torch, tqdm):
    from typing import Any

    print("🔄 STEP 4: Defining embedding function...")

    def make_embeddings(
        model: Any,
        datasetoutput: Any,
        window_size: int,
        patch_size: int,
        batch_size: int = 128,
        device: Any = None,
    ) -> Any:
        print("🔄 Starting embedding generation...")
        print(f"📊 Window size: {window_size}")
        print(f"📊 Patch size: {patch_size}")
        print(f"📊 Batch size: {batch_size}")
        print(f"📊 Device: {device}")

        if device is None:
            device = torch.device("cpu")
        model.eval()
        print("✅ Model set to evaluation mode")

        output_embeddings_list = []
        batch_count = 0

        for i in tqdm(
            datasetoutput.in_pixel_batches(batch_size=batch_size, window_size=window_size)
        ):
            batch_count += 1
            if batch_count % 10 == 0:
                print(f"🔄 Processing batch {batch_count}...")

            masked_output = MaskedOutput.from_datasetoutput(i, device=device)
            with torch.no_grad():
                model_output = model(
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
                )

                embeddings = model.average_tokens(*model_output[:-1]).cpu().numpy()
                output_embeddings_list.append(embeddings)

        print(f"✅ Processed {batch_count} batches total")
        print("🔄 Concatenating embeddings...")

        output_embeddings = np.concatenate(output_embeddings_list, axis=0)
        print(f"📊 Concatenated embeddings shape: {output_embeddings.shape}")

        # reshape the embeddings to H, W, D
        # first - how many "height batches" and "width batches" did we get?
        h_b = datasetoutput.space_time_x.shape[0] // window_size
        w_b = datasetoutput.space_time_x.shape[1] // window_size

        print(f"📊 Reshaping: height_batches={h_b}, width_batches={w_b}")

        reshaped_embeddings = rearrange(
            output_embeddings, "(h_b w_b) d -> h_b w_b d", h_b=h_b, w_b=w_b
        )
        print(f"📊 Final embeddings shape: {reshaped_embeddings.shape}")

        return reshaped_embeddings

    print("✅ Embedding function defined successfully!")
    return (make_embeddings,)


@app.cell
def generate_embeddings(dataset_output, make_embeddings, model, rearrange):
    print("🔄 STEP 5: Generating embeddings...")
    print("⏱️ This may take a while...")

    embeddings = make_embeddings(model, dataset_output, 1, 1, 128)
    print(f"✅ Embeddings generated! Shape: {embeddings.shape}")

    print("🔄 Flattening embeddings for clustering...")
    embeddings_flat = rearrange(embeddings, "h w d -> (h w) d")
    print(f"📊 Flattened embeddings shape: {embeddings_flat.shape}")
    print(f"📊 Embedding dimension: {embeddings_flat.shape[1]}")
    print(f"📊 Number of pixels: {embeddings_flat.shape[0]}")
    print("✅ Embeddings ready for analysis!")

    return embeddings, embeddings_flat


@app.cell
def cluster_embeddings(KMeans, embeddings, embeddings_flat, np, rearrange):
    print("🔄 STEP 6: Performing K-means clustering...")
    print("📊 Using 3 clusters")

    kmeans = KMeans(n_clusters=3)
    print("🔄 Fitting K-means model...")
    labels = kmeans.fit_predict(embeddings_flat)
    print("✅ K-means clustering complete!")
    print(f"📊 Labels shape: {labels.shape}")
    print(f"📊 Unique labels: {np.unique(labels)}")
    print(f"📊 Label counts: {np.bincount(labels)}")

    print("🔄 Reshaping labels to image format...")
    labels = rearrange(labels, "(h w) -> h w", h=embeddings.shape[0], w=embeddings.shape[1])
    print(f"📊 Reshaped labels shape: {labels.shape}")
    print("✅ K-means clustering analysis complete!")

    return (labels,)


@app.cell
def reduce_dimensions(PCA, embeddings, embeddings_flat, rearrange):
    print("🔄 STEP 7: Performing PCA dimensionality reduction...")
    print("📊 Reducing to 3 components for RGB visualization")

    pca = PCA(n_components=3)
    print("🔄 Fitting PCA model...")
    embeddings_pca = pca.fit_transform(embeddings_flat)
    print("✅ PCA complete!")
    print(f"📊 PCA embeddings shape: {embeddings_pca.shape}")
    print(f"📊 Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"📊 Total explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    print("🔄 Reshaping PCA embeddings to image format...")
    embeddings_reduced = rearrange(
        embeddings_pca, "(h w) d -> h w d", h=embeddings.shape[0], w=embeddings.shape[1]
    )
    print(f"📊 Reshaped PCA embeddings shape: {embeddings_reduced.shape}")
    print("✅ PCA dimensionality reduction complete!")

    return embeddings_pca, embeddings_reduced


@app.cell
def plot_results(embeddings_reduced, labels, plt):
    print("🔄 STEP 8: Creating visualizations...")

    # Plot the K-means clustering results
    plt.figure(figsize=(12, 4))

    print("📊 Visualization 1: K-means clustering results")
    plt.subplot(1, 3, 1)
    plt.imshow(labels, cmap="tab10")
    plt.title("K-means Clustering (3 clusters)")
    plt.colorbar()
    print("✅ K-means visualization complete!")

    # Plot the PCA-reduced embeddings as RGB
    print("📊 Visualization 2: PCA-reduced embeddings as RGB")
    plt.subplot(1, 3, 2)
    # Normalize to 0-1 range for display
    embeddings_normalized = (embeddings_reduced - embeddings_reduced.min()) / (
        embeddings_reduced.max() - embeddings_reduced.min()
    )
    print(
        f"📊 PCA RGB range: [{embeddings_normalized.min():.3f}, {embeddings_normalized.max():.3f}]"
    )

    plt.imshow(embeddings_normalized)
    plt.title("PCA-reduced embeddings (RGB)")
    print("✅ PCA visualization complete!")

    plt.tight_layout()
    plt.show()

    print("🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
    print("📊 Summary:")
    print(f"   - Generated embeddings with shape: {embeddings_reduced.shape}")
    print("   - Performed K-means clustering with 3 clusters")
    print("   - Reduced dimensionality with PCA")
    print("   - Created 2 visualizations")
    print("🎯 The Marimo notebook has run successfully with detailed logging!")

    return (embeddings_normalized,)


if __name__ == "__main__":
    app.run()
