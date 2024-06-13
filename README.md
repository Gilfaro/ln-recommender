## Colab

Open in [Colab](https://colab.research.google.com/github/Gilfaro/ln-recommender/blob/main/LNReadability.ipynb) and follow the instructions.

## Install

* Create a directory and unpack all the contents of the contents of the zip, ex. `ln-recommender`
* Create python venv

```
cd ln-recommender
python -m venv venv
```

* Activate your venv depending on your OS

```
venv\Scripts\activate{.bat|ps1}
```

* Install package and its dependencies

```
pip install .
```

* Run the tool

```
ln-recommender -h
```

## Example Usage

* Train model and output model file

```
ln-recommender train -td data.csv -ms model.cbm
```

* Train model with evaluation dataset and output model file, can yield better results if you have lots of data

```
ln-recommender train -td data.csv -ms model.cbm -ev
```

* Evaluate single text file

```
ln-recommender eval -t text.txt
```

* Evaluate directory with a saved model

```
ln-recommender eval -d directory -ml model.cbm
```

* Evaluate directory with a saved model and save output to csv file

```
ln-recommender eval -d directory -ml model.cbm -o out.csv
```

* Run clustering with minimum group size 3 and core samples 2

```
ln-recommender cluster -td data.csv -mcs 3 -ms 2
```