# Install

* Create a directory and unpack all the contents of the contents of the zip, ex. `recommender-cli`
* Create python venv

```
cd recommender-cli
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

# Example Usage

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
