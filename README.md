# KNNModel

## Purpose
Provide a lib to create easily a fast kNN index
FastKnn use mainly [nmslib](https://github.com/nmslib/nmslib/) as (fast) kNN backend


## Install
`pip install git+https://github.com/Fanchouille/knnmodel.git`

## Use
FastKnn builds a kNN index with specified `method` (default: `hnsw`) 
and `space` (default: `cosinesimil`)
- See [here](https://github.com/nmslib/nmslib/blob/master/manual/spaces.md) for different spaces
- See [here](https://github.com/nmslib/nmslib/blob/master/manual/methods.md) for different methods

Default with `hnsw` method and `cosinesimil` space

Example with dense data:
    
    X_train = np.random.rand(1000, 128)
    y_train = np.random.randint(10, size=1000)

Unsupervised (target is `id`) : 

    from knnmodel import KNNModel
    knn_model = KNNModel() 
    knn_model.fit(X=X_train) # no labels
    results_raw = knn_model.predict(X_test, n_neighbours=10, mode="raw", target="id") # predict & fetch list of 10 neighbours for each row in X_test
    results_majority = knn_model.predict(X_test, n_neighbours=10, mode="majority_voting", target="id") # predict & fetch id with most votes pct for each row in X_test
    results_on_training_set_raw = knn_model.predict_on_training_set(n_neighbours=10, mode="raw", target="id", exclude_identity=True) # predict & fetch list of 10 neighbours for each row in training set without identity
    results_on_training_set_majority = knn_model.predict_on_training_set(n_neighbours=10, mode="majority_voting", target="id", exclude_identity=True)

Supervised (target is `label`):

    from knnmodel import KNNModel    
    knn_model = KNNModel()
    knn_model.fit(X_train, y=y_train) # labels
    results_raw = knn_model.predict(X_train, n_neighbours=10, mode="raw", target="label")
    results_majority = knn_model.predict(X_train, n_neighbours=10, mode="majority_voting", target="label")
    results_on_training_set_raw = knn_model.predict_on_training_set(n_neighbours=10, mode="raw", target="label")
    results_on_training_set_majority = knn_model.predict_on_training_set(n_neighbours=10, mode="majority_voting", target="label")


## Development
Clone project

Install Anaconda local environment as below:
```bash
./install.sh
```

Activate Anaconda local environment as below:

```bash
conda activate ${PWD}/.conda
```
