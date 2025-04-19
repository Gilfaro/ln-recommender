import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from rich.console import Console
from rich.table import Table
from scipy.stats import zscore
from sklearn.cluster import HDBSCAN
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from ln_recommender.files import CSV_HEADERS

CSV_DATA_HEADERS = CSV_HEADERS[:-2]


def load_model(filename):
    model = CatBoostClassifier(verbose=True, allow_writing_files=False)
    model.load_model(filename)
    return model


def read_data_cluster(filename, min_cluster_size, min_samples):
    df = pd.read_csv(filename, sep=",", quotechar="'", index_col=False)

    x_data = df[CSV_DATA_HEADERS]
    y_data = df["Label"]

    scaled_x_data = x_data.apply(zscore)

    # pca_data = PCA(n_components=2).fit_transform(scaled_x_data)

    clustering = HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples
    ).fit(scaled_x_data)

    print(
        """
An array of cluster labels, one per datapoint. Outliers are labeled as follows:
  Noisy samples are given the label -1.
  Samples with infinite elements (+/- np.inf) are given the label -2.
  Samples with missing data are given the label -3, even if they also have infinite elements.
          """
    )
    print_data = list(zip(df["Comment"], y_data, clustering.labels_, strict=False))

    def sort_cluster(d):
        return d[2]

    print_data.sort(key=sort_cluster)

    table = Table("Name", "Label", "Cluster")
    for row in print_data:
        table.add_row(*tuple(map(str, row)))
    Console().print(table)


def read_data(filename, iterations, eval=False, model_filename=None):
    df = pd.read_csv(filename, sep=",", quotechar="'", index_col=False)

    x_data = df[CSV_DATA_HEADERS]
    y_data = df["Label"]

    classes = np.unique(y_data)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_data)
    class_weights = dict(zip(classes, weights, strict=False))

    model = CatBoostClassifier(
        loss_function="MultiClassOneVsAll",
        class_weights=class_weights,
        iterations=iterations,
        verbose=True,
        allow_writing_files=False,
    )
    if eval:
        x, x_test, y, y_test = train_test_split(
            x_data, y_data, test_size=0.2, train_size=0.8
        )
        train_pool = Pool(x, y)
        eval_pool = Pool(x_test, y_test)
        f = model.fit(
            train_pool, eval_set=eval_pool, early_stopping_rounds=100, verbose=100
        )
        print(f"Model accuracy on eval data: {f.score(eval_pool):.2%}")
    else:
        f = model.fit(x_data, y_data, verbose=100)
    if model_filename is not None:
        f.save_model(model_filename)
    print(f.get_feature_importance(prettified=True))
    return f
