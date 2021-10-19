import multiprocessing
import nmslib
import numpy as np
from typing import List, Union, Tuple, Optional, Dict
from sklearn.preprocessing import Normalizer
import cloudpickle
import os
from collections import Counter


class KNNModel:

    def __init__(self, space: str = "cosinesimil", method: str = 'hnsw', l2_normalize: bool = True,
                 is_fitted: bool = False):
        """
        :param space: space used to build index  see https://github.com/nmslib/nmslib/blob/master/manual/spaces.md, defaults to 'cosinesimil'
        :param method: method used to build index see https://github.com/nmslib/nmslib/blob/master/manual/methods.md, defaults to 'hnsw'
        :param l2_normalize: should data be l2 normalized, defaults to True
        :param is_fitted: is index fitted, defaults to False
        """
        self.knn_model = nmslib.init(space=space, method=method)
        self.l2_normalize = l2_normalize
        self.is_fitted = is_fitted
        self.space = space
        self.method = method

    def _expand_dim_and_normalize(self, X: np.ndarray) -> np.ndarray:
        """

        :param X: numpy array
        :return: numpy array where each row is L2 normalized
        """
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)
        if self.l2_normalize:
            normalizer = Normalizer(norm='l2')
            X = normalizer.transform(X)
        return X

    def _cast_to_numpy(self, y):
        if isinstance(y, list):
            y = np.array(y)
        return y

    def fit(self, X: np.ndarray, y: Optional[Union[List, np.ndarray]] = None, M: int = 50, efC: int = 2000,
            efS: int = 200, indexThreadQty: int = None):
        """
        Fit the knn_model provided by NMSLIB.

        :param X: data to fit the KNN on
        :param y: labels/index to fit the KNN on (defaults to None)
        :param M: rangein 5-100 (increase for better recall but longer indexing)
        :param efC: increase to augment accuracy (longer indexing)
        :param efS: increase to augment recall (longer retrieval)
        """
        if indexThreadQty is None:
            indexThreadQty = multiprocessing.cpu_count() - 1
        labels = self._cast_to_numpy(y)
        self.labels = labels
        index_time_params = {'M': M, 'indexThreadQty': indexThreadQty,
                             'efConstruction': efC, 'post': 2}
        X = self._expand_dim_and_normalize(X)
        self.knn_model.addDataPointBatch(X)
        self.knn_model.createIndex(index_time_params)
        self.knn_model.setQueryTimeParams({'efSearch': efS})
        self.data = X
        self.is_fitted = True

    def predict_id(self, X: np.ndarray, n_neighbours: int = 10, exclude_identity: bool = False) -> Union[
        Tuple[np.ndarray, np.ndarray], Tuple[None, None]]:
        """

        :param X: data to predict on
        :param n_neighbours: number of neighbours
        :param exclude_identity: if predicting on the same dataset that KNN was fitted on, should we exclude identity match
        :return: Tuple of ids and distances
        """
        if self.is_fitted:
            X = self._expand_dim_and_normalize(X)
            nearest_neighbours = self.knn_model.knnQueryBatch(X,
                                                              k=n_neighbours + 1 if exclude_identity else n_neighbours,
                                                              num_threads=multiprocessing.cpu_count() - 1)
            ids, distances = map(list, zip(*nearest_neighbours))
            ids = list(map(lambda x: x.tolist(), ids))
            distances = list(map(lambda x: x.tolist(), distances))
            if exclude_identity:
                for idx in range(X.shape[0]):
                    if idx in ids[idx]:
                        idx_to_remove = ids[idx].index(idx)
                        ids[idx].pop(idx_to_remove)
                        distances[idx].pop(idx_to_remove)

            ids, distances = np.array(ids), np.array(distances)

            return ids[:, 0:n_neighbours], distances[:, 0:n_neighbours]
        else:
            print("Model was not fitted, please use .fit method on data before")
            return None, None

    def predict(self, X: np.ndarray, n_neighbours: int = 10, mode: str = "raw", target: str = "id",
                exclude_identity: bool = False) -> Union[List[Dict[str, Union[np.array, int, str, float]]], None]:
        """

        :param X: data to predict on
        :param n_neighbours: number of neighbours
        :param mode:

                - ``raw`` : returns list of neighbours.
                - ``majority_voting`` : classification task (majority voting of all NN).
                - ``regression`` : regression task (mean of all NN).
        :param target:

                - ``id`` : returns ids (row index)
                - ``label`` : returns labels (if provided during fit)
        :param exclude_identity: if predicting on the same dataset that KNN was fitted on, should we exclude identity match
        :return: Predictions (raw list, majority votes or mean of nearest neighbours)
        """
        predictions, distances = self.predict_id(X, n_neighbours, exclude_identity)

        if target == "id":
            pass
        elif target == "label":
            if self.labels is not None:
                predictions = np.vectorize(self.labels.__getitem__)(predictions)
            else:
                print("Labels were not defined")
                return
        else:
            print("target should be in ['id', 'label']")
            return

        results = []

        if mode == "raw":
            for i, row in enumerate(predictions):
                results.append({"prediction": row, "distance": distances[i, :]})

        elif mode == "majority_voting":
            for i, row in enumerate(predictions):
                current_most_common = Counter(row).most_common(1)
                prediction = current_most_common[0][0]
                proba = current_most_common[0][1] / n_neighbours
                mask = np.where(row == prediction)
                idx_1st_pred = mask[0].min()
                nearest_dist = distances[i, idx_1st_pred]
                mean_dist = distances[i, mask[0]].mean()
                results.append({"prediction": prediction, "probability": proba, "min_distance": nearest_dist,
                                "mean_distance": mean_dist})
        elif mode == "regression":
            for row in predictions:
                prediction = row.mean()
                results.append({"prediction": prediction})

        else:
            print("mode should be in ['raw', 'majority_voting', 'regression']")
            return

        return results

    def predict_on_training_set(self, n_neighbours: int = 10, mode: str = "raw",
                                target: str = "id", exclude_identity: bool = False) \
            -> Union[List[Dict[str, Union[np.array, int, str, float]]], None]:
        """

        :param n_neighbours: number of neighbours
        :param mode:

                - ``raw`` : returns list of neighbours.
                - ``majority_voting`` : classification task (majority voting of all NN).
                - ``regression`` : regression task (mean of all NN).
        :param target:

                - ``id`` : returns ids (row index)
                - ``label`` : returns labels (if provided during fit)
        :param exclude_identity: if predicting on the same dataset that KNN was fitted on, should we exclude identity match
        :return: Predictions (raw list, majority votes or mean of nearest neighbours)
        """
        return self.predict(X=self.data, n_neighbours=n_neighbours, mode=mode, target=target,
                            exclude_identity=exclude_identity)

    def save(self, folder_path: str):
        """

        :param folder_path: serialize Knn model to folder_path
        """
        os.makedirs(folder_path, exist_ok=True)
        self.knn_model.saveIndex(folder_path + "/knn_model")
        config = {"l2_normalize": self.l2_normalize,
                  "is_fitted": self.is_fitted,
                  "space": self.space,
                  "method": self.method}
        with open(folder_path + "/config", 'wb') as handle:
            cloudpickle.dump(config, handle)

        if self.labels is not None:
            with open(folder_path + "/labels", 'wb') as handle:
                cloudpickle.dump(self.labels, handle)
        if self.data is not None:
            with open(folder_path + "/data", 'wb') as handle:
                cloudpickle.dump(self.data, handle)

    def load(self, folder_path: str):
        """

        :param folder_path: loads Knn model from folder_path
        """
        with open(folder_path + '/config', 'rb') as f:
            config = cloudpickle.load(f)
        try:
            with open(folder_path + '/labels', 'rb') as f:
                self.labels = cloudpickle.load(f)
        except:
            self.labels = None
        try:
            with open(folder_path + '/data', 'rb') as f:
                self.data = cloudpickle.load(f)
        except:
            pass

        self.__init__(**config)
        self.knn_model.loadIndex(folder_path + '/knn_model')
