import time
import warnings
from pathlib import Path

import mercs
import numpy as np


from mercs import Mercs
import pandas as pd
from mercs.utils.encoding import code_to_query, query_to_code
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from affe.flow import Flow
from affe.io import (
    FN_TEMPLATE_CLASSIC_FLOW,
    abspath,
    check_existence_of_directory,
    dump_object,
    get_default_model_filename,
    get_filepath,
    get_flow_directory,
    get_subdirectory_paths,
    get_template_filenames,
    insert_subdirectory,
    load_object,
    mimic_fs,
)

from .io import dataset_filepath, query_filepath

from copy import deepcopy


class StarAiFlow(Flow):
    STR = "STARAIFlow"

    def __init__(self, timeout_s=60, verbose=False, **kwargs):
        self._data = None
        self._metadata = None
        self._qry = None
        self._queries = None
        self._y_true = None
        self._analysis = None
        self._retention = None
        self.verbose = verbose

        # Init all configs
        self.config = dict()
        self.config["io"] = self._init_io(**kwargs)
        self.config["data"] = self._init_data_config(**kwargs)
        self.config["qry"] = self._init_qry_config(**kwargs)
        self.config["algo"] = self._init_algo_config(**kwargs)
        self.config["analysis"] = self._init_analysis_config(**kwargs)
        self.config["retention"] = self._init_retention_config(**kwargs)

        # Superclass init
        log_filepath = self.io["flow_filepaths"]["logs"]
        flow_filepath = self.io["flow_filepaths"]["flows"]

        super().__init__(
            config=self.config,
            log_filepath=log_filepath,
            flow_filepath=flow_filepath,
            timeout_s=timeout_s,
        )

        return

    # IO
    @property
    def io(self):
        return self.config["io"]

    @property
    def io_config(self):
        return self.io

    # (Meta)Data
    @property
    def metadata(self):

        if self._metadata is None:
            name = self.data_config["data_identifier"]
            n_features = self.data["train"].shape[1]
            n_instances = self.data["train"].shape[0]
            n_instances_train = n_instances
            n_instances_test = self.data["test"].shape[0]

            self._metadata = dict(
                name=name,
                n_features=n_features,
                n_instances=n_instances,
                n_instances_train=n_instances_train,
                n_instances_test=n_instances_test,
            )
        else:
            pass
        return self._metadata

    @property
    def data_config(self):
        return self.config["data"]

    @property
    def data(self):
        if self._data is None:
            self._data = dict(
                train=pd.read_csv(self.data_config["train_fpath"]),
                test=pd.read_csv(self.data_config["test_fpath"]),
            )
        else:
            pass
        return self._data

    # Query
    @property
    def qry_config(self):
        return self.config["qry"]

    @property
    def qry(self):
        if self._qry is None:
            self._qry = load_object(self.qry_config["filepath"])
        else:
            pass
        return self._qry

    @property
    def q_codes(self):
        return self.qry

    def q_code(self, n):
        return self.qry[n, :]

    @property
    def queries(self):
        if self._queries is None:
            q_desc = []
            q_targ = []
            q_miss = []
            for q_code in self.q_codes:
                d, t, m = code_to_query(q_code, return_list=True)
                q_desc.append(d)
                q_targ.append(t)
                q_miss.append(m)

            self._queries = (q_desc, q_targ, q_miss)
        else:
            pass
        return self._queries

    @property
    def q_desc(self):
        return self.queries[0]

    @property
    def q_targ(self):
        return self.queries[1]

    @property
    def q_miss(self):
        return self.queries[2]

    def get_q_desc(self, n=None):
        if n is None:
            return self.q_desc
        else:
            return self.q_desc[n]

    def get_q_targ(self, n=None):
        if n is None:
            return self.q_targ
        else:
            return self.q_targ[n]

    def get_q_miss(self, n=None):
        if n is None:
            return self.q_miss
        else:
            return self.q_miss[n]

    @property
    def n_qrys(self):
        return self.qry.shape[0]

    # Algo
    @property
    def algo_config(self):
        return self.config["algo"]

    @property
    def model(self):
        m_algo = getattr(self, "m_algo", None)
        if m_algo is None:
            return None
        else:
            return m_algo.get("model", None)

    @property
    def algo(self):
        return self.model

    # Predictions
    @property
    def predictions(self):
        a_algo = getattr(self, "a_algo", None)
        if a_algo is None:
            return None
        else:
            return a_algo.get("predictions", None)

    @property
    def y_pred(self):
        return self.predictions

    def get_y_pred(self, n):
        if n is None:
            return self.predictions
        else:
            return self.predictions[n]

    @property
    def y_true(self):
        if self._y_true is None:
            self._y_true = dict()
            for q_idx, q_targ in enumerate(self.q_targ):
                self._y_true[q_idx] = self.data["test"].values[:, q_targ]
        else:
            pass
        return self._y_true

    def get_y_true(self, n):
        if n is None:
            return self.y_true
        else:
            return self.y_true[n]

    # Analysis
    @property
    def analysis_config(self):
        return self.config["analysis"]

    @property
    def analysis(self):
        if self._analysis is None:
            self._analysis = self.get_analysis()
        else:
            pass
        return self._analysis

    @analysis.setter
    def analysis(self, analysis):
        assert isinstance(analysis, dict), "Analysis needs to be a dict"
        self._analysis = analysis
        return

    @property
    def results(self):
        analysis = self.analysis
        metadata = self.metadata
        return dict(analysis=analysis, metadata=metadata)

    # Retentio
    @property
    def retention_config(self):
        return self.config["retention"]

    @property
    def retention(self):
        if self._retention is None:
            self._retention = self.get_retention()
        else:
            pass
        return self._retention

    @retention.setter
    def retention(self, retention):
        assert isinstance(retention, bool), "Retention is a bool"
        self._retention = retention
        return

    # Inits
    def _init_io(
        self,
        flow_id=0,
        flow_identifier="manual",
        root_levels_up=2,
        fs_depth=1,
        out_directory="out",
        out_parent="root",
        basename=None,
        save_model=False,
        load_model=False,
        model_identifier=None,
        data_identifier=None,
        exclude_in_scan={"notebooks", "visualisation", "tests", "admercs"},
        **kwargs,
    ):
        # Perform duties
        fs = mimic_fs(
            root_levels_up=root_levels_up, depth=fs_depth, exclude=exclude_in_scan,
        )

        ## Build the filesystem we desire
        fs, out_key = insert_subdirectory(
            fs, parent=out_parent, child=out_directory, return_key=True
        )

        flow_directory = get_flow_directory(keyword=flow_identifier)
        fs, flow_key = insert_subdirectory(
            fs, parent=out_key, child=flow_directory, return_key=True
        )

        check_existence_of_directory(fs)

        flow_dirpaths = get_subdirectory_paths(fs, flow_key)
        flow_filepaths = get_template_filenames(
            flow_dirpaths,
            basename=basename,
            idx=flow_id,
            template=FN_TEMPLATE_CLASSIC_FLOW,
        )

        ## Model IO
        model_filepath = self._get_model_filepath(
            fs,
            load_model,
            save_model,
            data_identifier=data_identifier,
            model_identifier=model_identifier,
            basename=basename,
        )

        # collect outgoing information
        io = dict(
            flow_id=flow_id,
            flow_identifier=flow_identifier,
            fs=fs,
            flow_key=flow_key,
            flow_dirpaths=flow_dirpaths,
            flow_filepaths=flow_filepaths,
            model_filepath=model_filepath,
            load_model=load_model,
            save_model=save_model,
        )

        return io

    def _init_data_config(self, data_identifier=None, step=1, **kwargs):

        data_dir_filepath = Path(abspath(self.io["fs"], node="data"))
        train_fpath = dataset_filepath(
            name=data_identifier,
            kind="train",
            step=step,
            data_dir_filepath=data_dir_filepath,
            extension="csv",
            check=True,
        )

        test_fpath = dataset_filepath(
            name=data_identifier,
            kind="test",
            step=step,
            data_dir_filepath=data_dir_filepath,
            extension="csv",
            check=True,
        )

        data_config = dict(
            data_identifier=data_identifier,
            step=step,
            train_fpath=train_fpath,
            test_fpath=test_fpath,
        )

        return data_config

    def _init_qry_config(
        self, data_identifier=None, qry_keyword="default", n_queries=None, **kwargs
    ):
        qry_dir_filepath = Path(abspath(self.io["fs"], node="query"))
        qry_filepath = query_filepath(
            name=data_identifier,
            keyword=qry_keyword,
            query_dir_filepath=qry_dir_filepath,
            extension="npy",
        )

        qry_config = dict(
            filepath=qry_filepath, keyword=qry_keyword, n_queries=n_queries
        )

        return qry_config

    def _init_algo_config(self, **kwargs):
        algo_config = dict()
        return algo_config

    def _init_analysis_config(
        self, macro_f1_score=True, micro_f1_score=False, accuracy_score=True, **kwargs
    ):
        return dict(
            macro_f1_score=macro_f1_score,
            micro_f1_score=micro_f1_score,
            accuracy_score=accuracy_score,
        )

    def _init_retention_config(
        self, save_results=True, save_model=False, save_config=True, **kwargs
    ):
        return dict(
            save_results=save_results, save_model=save_model, save_config=save_config
        )

    # Actual algorithm
    def get_algo(self, train, model=None):
        return dict(model=None, fit_time_s=None)

    def ask_algo(self, test):
        assert model is not None, "You need a model before you can call this function"
        return dict(predictions=None, predict_time_s=None)

    # Analysis
    def get_analysis(self):
        cfg = self.analysis_config

        analysis = dict()
        if cfg["macro_f1_score"]:
            analysis["macro_f1_score"] = []
        if cfg["micro_f1_score"]:
            analysis["micro_f1_score"] = []
        if cfg["accuracy_score"]:
            analysis["accuracy_score"] = []

        for q_idx in range(self.n_qrys):
            y_true = self.get_y_true(q_idx)
            y_pred = self.get_y_pred(q_idx)

            if cfg["macro_f1_score"]:
                macro_f1_score = f1_score(y_true, y_pred, average="macro")
                analysis["macro_f1_score"].append(macro_f1_score)

            if cfg["micro_f1_score"]:
                micro_f1_score = f1_score(y_true, y_pred, average="micro")
                analysis["micro_f1_score"].append(micro_f1_score)

            if cfg["accuracy_score"]:
                accuracy = accuracy_score(y_true, y_pred)
                analysis["accuracy_score"].append(accuracy)

        return analysis

    # Save
    def get_retention(self):
        # collect ingoing information
        oks = []
        cfg = self.retention_config
        io = self.io

        if cfg["save_results"]:
            results = self.results
            fp_results = io["flow_filepaths"]["results"]
            ok = dump_object(results, fp_results)
            oks.append(ok)

        if cfg["save_model"]:
            model = self.model
            fp_model = io["model_filepath"]
            ok = dump_object(model, fp_model)
            oks.append(ok)

        if cfg["save_config"]:
            dcfg = self._get_dumpable_config()

            fp_config = io["flow_filepaths"]["config"]
            ok = dump_object(dcfg, fp_config)
            oks.append(ok)

        return all(oks)

    # Flows
    def flow(self):

        # Get data
        train, test = self._get_train_test()

        # Load model
        if self.io["load_model"]:
            model = load_object(self.io["model_filepath"])
        else:
            model = None

        # Train your model
        self.m_algo = self.get_algo(train, model=model)

        # Use your model
        self.a_algo = self.ask_algo(test)

        # Get analysis
        self.analysis = self.get_analysis()

        # Get retention (=Save the things you want to save)
        self.retention = self.get_retention()
        return

    # Helpers
    def _get_model_filepath(
        self,
        fs,
        load_model,
        save_model,
        data_identifier=None,
        model_identifier=None,
        basename=None,
    ):
        model_filename = self._get_model_filename(
            data_identifier=data_identifier,
            model_identifier=model_identifier,
            basename=basename,
        )

        if load_model:
            return get_filepath(
                tree=fs, node="models", filename=model_filename, check_file=True
            )
        elif save_model:
            return get_filepath(
                tree=fs, node="models", filename=model_filename, check_file=False
            )
        else:
            return

    @staticmethod
    def _get_model_filename(
        data_identifier=None, model_identifier=None, basename=None,
    ):
        if model_identifier is not None:
            model_filename = get_default_model_filename(
                data_identifier=data_identifier, model_identifier=model_identifier
            )
        else:
            model_filename = get_default_model_filename(
                data_identifier=data_identifier, model_identifier=basename
            )
        return model_filename

    def _get_train_test(self):
        return self.data["train"], self.data["test"]

    def _get_dumpable_config(self):
        dumpable_config = deepcopy(self.config)

        def _convert_entries(d):
            for k, v in d.items():
                if isinstance(v, type(Path())):
                    # PosixPath to String conversion
                    d[k] = str(v)
                elif isinstance(v, dict):
                    d[k] = _convert_entries(v)
                else:
                    pass
            return d

        return _convert_entries(dumpable_config)


class MercsStarAiFlow(StarAiFlow):
    def _init_algo_config(
        self,
        reconfigure_algo=True,
        max_depth=None,
        min_samples_leaf=5,
        criterion="gini",
        min_impurity_decrease=0.0,
        **kwargs,
    ):
        return {k: v for k, v in dict(locals()).items() if k not in {"kwargs", "self"}}

    # Actual algorithm
    def get_algo(self, train, model=None):
        algo_config = self.algo_config

        # collect ingoing information
        X = train.values
        X = X.astype(float)
        nominal_ids = set(range(X.shape[1]))

        # perform duty
        if model is None:
            model = Mercs(**algo_config)

            tick = time.time()
            model.fit(X, nominal_attributes=nominal_ids, **algo_config)
            tock = time.time()

            fit_time_s = tock - tick
        elif isinstance(model, Mercs):
            if algo_config["reconfigure_algo"]:
                model = self.reconfigure_algo(model, **algo_config)
            fit_time_s = model.model_data["ind_time"]
        else:
            raise ValueError(
                "I expect either no model or a Mercs model. Not {}".format(model)
            )
        return dict(model=model, fit_time_s=fit_time_s)

    def reconfigure_algo(self, model, **algo_config):
        raise NotImplementedError

    def ask_algo(self, test):
        algo_config = self.algo_config
        model = self.model
        q_codes = self.q_codes

        assert isinstance(q_codes, np.ndarray)
        assert model is not None, "You need a model before you can call this function"

        # Preprocessing
        X = test.copy().values
        X = X.astype(float)

        predictions = dict()
        predict_time_s = dict()
        for q_idx, q_code in enumerate(q_codes):
            targ_ids = list(self.get_q_targ(q_idx))
            miss_ids = list(self.get_q_miss(q_idx))

            if self.verbose:
                msg = """
                targ_ids: {}
                miss_ids: {}
                """.format(
                    targ_ids, miss_ids
                )
                print(msg)

            X_test = X.copy()
            X_test[:, targ_ids] = np.nan
            X_test[:, miss_ids] = np.nan

            assert np.sum(np.isnan(X_test[0, :])) == len(targ_ids) + len(
                miss_ids
            ), "Not the correct amount of missing data"

            if algo_config["reconfigure_algo"]:
                y_pred = model.predict(X_test, q_code=q_code, **algo_config)
            else:
                y_pred = model.predict(X_test, q_code=q_code)

            inf_time = model.model_data["inf_time"]

            predictions[q_idx] = y_pred
            predict_time_s[q_idx] = inf_time

        return dict(predictions=predictions, predict_time_s=predict_time_s)


class BayesFusionStarAiFlow(StarAiFlow):
    def _init_algo_config(
        self,
        reconfigure_algo=True,
        max_depth=None,
        min_samples_leaf=5,
        criterion="gini",
        min_impurity_decrease=0.0,
        **kwargs,
    ):
        return {k: v for k, v in dict(locals()).items() if k not in {"kwargs", "self"}}

    # Actual algorithm
    def get_algo(self, train, model=None):
        algo_config = self.algo_config

        # collect ingoing information
        X = train.values
        X = X.astype(float)
        nominal_ids = set(range(X.shape[1]))

        # perform duty
        if model is None:
            model = Mercs(**algo_config)

            tick = time.time()
            model.fit(X, nominal_attributes=nominal_ids, **algo_config)
            tock = time.time()

            fit_time_s = tock - tick
        elif isinstance(model, Mercs):
            if algo_config["reconfigure_algo"]:
                model = self.reconfigure_algo(model, **algo_config)
            fit_time_s = model.model_data["ind_time"]
        else:
            raise ValueError(
                "I expect either no model or a Mercs model. Not {}".format(model)
            )
        return dict(model=model, fit_time_s=fit_time_s)

    def reconfigure_algo(self, model, **algo_config):
        raise NotImplementedError

    def ask_algo(self, test):
        algo_config = self.algo_config
        model = self.model
        q_codes = self.q_codes

        assert isinstance(q_codes, np.ndarray)
        assert model is not None, "You need a model before you can call this function"

        # Preprocessing
        X = test.copy().values
        X = X.astype(float)

        predictions = dict()
        predict_time_s = dict()
        for q_idx, q_code in enumerate(q_codes):
            targ_ids = list(self.get_q_targ(q_idx))
            miss_ids = list(self.get_q_miss(q_idx))

            if self.verbose:
                msg = """
                targ_ids: {}
                miss_ids: {}
                """.format(
                    targ_ids, miss_ids
                )
                print(msg)

            X_test = X.copy()
            X_test[:, targ_ids] = np.nan
            X_test[:, miss_ids] = np.nan

            assert np.sum(np.isnan(X_test[0, :])) == len(targ_ids) + len(
                miss_ids
            ), "Not the correct amount of missing data"

            if algo_config["reconfigure_algo"]:
                y_pred = model.predict(X_test, q_code=q_code, **algo_config)
            else:
                y_pred = model.predict(X_test, q_code=q_code)

            inf_time = model.model_data["inf_time"]

            predictions[q_idx] = y_pred
            predict_time_s[q_idx] = inf_time

        return dict(predictions=predictions, predict_time_s=predict_time_s)
