import numpy as np

# No imputations
SATMAE_MEAN = [
    1370.19151926,
    1184.3824625,
    1120.77120066,
    1136.26026392,
    1263.73947144,
    1645.40315151,
    1846.87040806,
    1762.59530783,
    1972.62420416,
    582.72633433,
    14.77112979,
    1732.16362238,
    1247.91870117,
]
SATMAE_STD = [
    633.15169573,
    650.2842772,
    712.12507725,
    965.23119807,
    948.9819932,
    1108.06650639,
    1258.36394548,
    1233.1492281,
    1364.38688993,
    472.37967789,
    14.3114637,
    1310.36996126,
    1087.6020813,
]

# B10 stats are imputed with B11 stats
S2A_MEAN = [
    752.40087073,
    884.29673756,
    1144.16202635,
    1297.47289228,
    1624.90992062,
    2194.6423161,
    2422.21248945,
    2517.76053101,
    2581.64687018,
    2645.51888987,
    2368.51236873,
    2368.51236873,
    1805.06846033,
]
S2A_STD = [
    1108.02887453,
    1155.15170768,
    1183.6292542,
    1368.11351514,
    1370.265037,
    1355.55390699,
    1416.51487101,
    1474.78900051,
    1439.3086061,
    1582.28010962,
    1455.52084939,
    1455.52084939,
    1343.48379601,
]

# No imputations
S2C_MEAN = [
    1605.57504906,
    1390.78157673,
    1314.8729939,
    1363.52445545,
    1549.44374991,
    2091.74883118,
    2371.7172463,
    2299.90463006,
    2560.29504086,
    830.06605044,
    22.10351321,
    2177.07172323,
    1524.06546312,
]
S2C_STD = [
    786.78685367,
    850.34818441,
    875.06484736,
    1138.84957046,
    1122.17775652,
    1161.59187054,
    1274.39184232,
    1248.42891965,
    1345.52684884,
    577.31607053,
    51.15431158,
    1336.09932639,
    1136.53823676,
]

# B1 stats are imputed with B2 stats
# B9 stats are imputed with B8A stats
# B10 stats are imputed with B11 stats
OURS_S2_MEAN = [
    1395.3408730676722,
    1395.3408730676722,
    1338.4026921784578,
    1343.09883810357,
    1543.8607982512297,
    2186.2022069512263,
    2525.0932853316694,
    2410.3377187373408,
    2750.2854646886753,
    2750.2854646886753,
    2234.911100061487,
    2234.911100061487,
    1474.5311266077113,
]
OURS_S2_STD = [
    917.7041440370853,
    917.7041440370853,
    913.2988423581528,
    1092.678723527555,
    1047.2206083460424,
    1048.0101611156767,
    1143.6903026819996,
    1098.979177731649,
    1204.472755085893,
    1204.472755085893,
    1145.9774063078878,
    1145.9774063078878,
    980.2429840007796,
]

OURS_S1_MEAN = [-11.728724389184965, -18.85558188024017]
OURS_S1_STD = [4.887145774840316, 5.730270320384293]

PRESTO_S1_SUBTRACT_VALUES = [-25.0, -25.0]
PRESTO_S1_DIV_VALUES = [25.0, 25.0]
PRESTO_S2_SUBTRACT_VALUES = [float(0.0)] * len(OURS_S2_MEAN)
PRESTO_S2_DIV_VALUES = [float(1e4)] * len(OURS_S2_MEAN)

# https://github.com/zhu-xlab/SSL4EO-S12/blob/main/src/benchmark/
# pretrain_ssl/datasets/SSL4EO/ssl4eo_dataset.py
S1_MEAN = [-12.54847273, -20.19237134]
S1_STD = [5.25697717, 5.91150917]

# for Prithvi, true values are ["B02", "B03", "B04", "B05", "B06", "B07"]
# for all other bands, we just copy the nearest relevant band value; Prithvi
# shouldn't be using them anyway
PRITHVI_MEAN = [
    1087.0,
    1087.0,
    1342.0,
    1433.0,
    2734.0,
    1958.0,
    1363.0,
    1363.0,
    1363.0,
    1363.0,
    1363.0,
    1363.0,
    1363.0,
]
PRITHVI_STD = [
    2248.0,
    2248.0,
    2179.0,
    2178.0,
    1850.0,
    1242.0,
    1049.0,
    1049.0,
    1049.0,
    1049.0,
    1049.0,
    1049.0,
    1049.0,
]


pre_computed_stats = {
    "SATMAE": {"mean": SATMAE_MEAN, "std": SATMAE_STD},
    "S2A": {"mean": S2A_MEAN, "std": S2A_STD},
    "S2C": {"mean": S2C_MEAN, "std": S2C_STD},
    "OURS": {"mean": OURS_S2_MEAN, "std": OURS_S2_STD},
    "OURS_S1": {"mean": OURS_S1_MEAN, "std": OURS_S1_STD},
    "S1": {"mean": S1_MEAN, "std": S1_STD},
    "presto_s1": {"mean": PRESTO_S1_SUBTRACT_VALUES, "std": PRESTO_S1_DIV_VALUES},
    "presto_s2": {"mean": PRESTO_S2_SUBTRACT_VALUES, "std": PRESTO_S2_DIV_VALUES},
}

s2_band_names = [
    "01 - Coastal aerosol",
    "02 - Blue",
    "03 - Green",
    "04 - Red",
    "05 - Vegetation Red Edge",
    "06 - Vegetation Red Edge",
    "07 - Vegetation Red Edge",
    "08 - NIR",
    "08A - Vegetation Red Edge",
    "09 - Water vapour",
    "10 - SWIR - Cirrus",
    "11 - SWIR",
    "12 - SWIR",
]

s1_band_names = ["VV", "VH"]


def impute_normalization_stats(band_info, imputes):
    # band_info is a dictionary with band names as keys and statistics (mean / std) as values
    if not imputes:
        return band_info

    names_list = list(band_info.keys())
    new_band_info = {}
    for band_name in s2_band_names:
        new_band_info[band_name] = {}
        if band_name in names_list:
            # we have the band, so use it
            new_band_info[band_name] = band_info[band_name]
        else:
            # we don't have the band, so impute it
            for impute in imputes:
                src, tgt = impute
                if tgt == band_name:
                    # we have a match!
                    new_band_info[band_name] = band_info[src]
                    break

    return new_band_info


def impute_bands(image_list, names_list, imputes):
    # image_list should be one np.array per band, stored in a list
    # image_list and names_list should be ordered consistently!
    if not imputes:
        return image_list

    # create a new image list by looping through and imputing where necessary
    new_image_list = []
    for band_name in s2_band_names:
        if band_name in names_list:
            # we have the band, so append it
            band_idx = names_list.index(band_name)
            new_image_list.append(image_list[band_idx])
        else:
            # we don't have the band, so impute it
            for impute in imputes:
                src, tgt = impute
                if tgt == band_name:
                    # we have a match!
                    band_idx = names_list.index(src)
                    new_image_list.append(image_list[band_idx])
                    break

    return new_image_list


def get_norm_stats(band_info):
    means = []
    stds = []
    if len(band_info) == len(s2_band_names):
        for band_name in s2_band_names:
            assert band_name in band_info, f"{band_name} not found in band_info"
            means.append(band_info[band_name]["mean"])
            stds.append(band_info[band_name]["std"])
    elif len(band_info) == len(s1_band_names):
        for band_name in s1_band_names:
            assert band_name in band_info, f"{band_name} not found in band_info"
            means.append(band_info[band_name]["mean"])
            stds.append(band_info[band_name]["std"])
    else:
        raise ValueError(f"Got unexpected band_info length {len(band_info)}")

    return means, stds


def normalize_bands(image, norm_cfg, band_info):
    if norm_cfg["type"] == "satlas":
        image = image / 8160
        image = np.clip(image, 0, 1)
        return image

    original_dtype = image.dtype
    if norm_cfg["stats"] == "dataset":
        means, stds = get_norm_stats(band_info)
    elif norm_cfg["stats"] in pre_computed_stats.keys():
        means = pre_computed_stats[norm_cfg["stats"]]["mean"]
        stds = pre_computed_stats[norm_cfg["stats"]]["std"]
    else:
        raise f"normalization stats not found: {norm_cfg['stats']}"

    means = np.array(means)
    stds = np.array(stds) * norm_cfg["std_multiplier"]

    if norm_cfg["type"] == "standardize":
        image = (image - means) / stds
    else:
        min_value = means - stds
        max_value = means + stds
        image = (image - min_value) / (max_value - min_value)

        if norm_cfg["type"] == "norm_yes_clip":
            image = np.clip(image, 0, 1)
        elif norm_cfg["type"] == "norm_yes_clip_int":
            # same as clipping between 0 and 1 but rounds to the nearest 1/255
            image = image * 255  # scale
            image = np.clip(image, 0, 255).astype(np.uint8)  # convert to 8-bit integers
            image = image.astype(original_dtype) / 255  # back to original_dtype between 0 and 1
        elif norm_cfg["type"] == "norm_no_clip":
            pass
        else:
            raise ValueError(
                f"norm type must norm_yes_clip, norm_yes_clip_int, norm_no_clip, or standardize, not {norm_cfg['type']}"
            )
    return image
