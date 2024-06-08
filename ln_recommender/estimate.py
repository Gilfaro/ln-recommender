from catboost import CatBoostClassifier, Pool
import pandas as pd
from sklearn.model_selection import train_test_split
from ln_recommender.files import CSV_HEADERS

CSV_DATA_HEADERS = CSV_HEADERS[:-2]


def load_model(filename):
    model = CatBoostClassifier(verbose=True, allow_writing_files=False)
    model.load_model(filename)
    return model


def read_data(filename, eval=False, model_filename=None):
    df = pd.read_csv(filename, sep=",", quotechar="'", index_col=False)

    x_data = df[CSV_DATA_HEADERS]
    y_data = df["Label"]
    model = CatBoostClassifier(verbose=True, allow_writing_files=False)
    if eval:
        x, x_test, y, y_test = train_test_split(
            x_data, y_data, test_size=0.2, train_size=0.8
        )
        train_pool = Pool(x, y)
        eval_pool = Pool(x_test, y_test)
        f = model.fit(train_pool, eval_set=eval_pool, verbose=100)
    else:
        f = model.fit(x_data, y_data, verbose=100)
    if model_filename is not None:
        f.save_model(model_filename)
    return f
