"""Synthetic and real non-IID FL cohort generators.

The synthetic side (``make_cohort``) draws from K linear binary tasks;
the real-data side (``make_adult_cohort``) groups clients by a
chosen categorical feature on the UCI Adult dataset, producing a
profile-based non-IID partition with optional drift. Both expose the
same ``Cohort`` interface so the trainer is dataset-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TaskCatalog:
    W: np.ndarray  # (K, d)
    b: np.ndarray  # (K,)

    @property
    def n_tasks(self) -> int:
        return self.W.shape[0]


def make_tasks(n_tasks: int, n_features: int, seed: int) -> TaskCatalog:
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(n_tasks, n_features))
    # Normalize so tasks have comparable margin sizes.
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    W *= 3.0
    b = rng.normal(size=n_tasks) * 0.3
    return TaskCatalog(W=W, b=b)


def sample_from_task(
    catalog: TaskCatalog, task_id: int, n_samples: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    X = rng.normal(size=(n_samples, catalog.W.shape[1]))
    z = X @ catalog.W[task_id] + catalog.b[task_id]
    p = 1.0 / (1.0 + np.exp(-z))
    y = (rng.uniform(size=n_samples) < p).astype(int)
    return X, y


@dataclass
class ClientSpec:
    primary: int
    secondary: int | None  # None for non-boundary clients
    drift_to: int | None  # task id after the drift event, else None


@dataclass
class Cohort:
    catalog: TaskCatalog
    specs: list[ClientSpec]
    train: list[tuple[np.ndarray, np.ndarray]]
    test: list[tuple[np.ndarray, np.ndarray]]
    primary_label: np.ndarray  # ground-truth cluster (pre-drift)
    is_boundary: np.ndarray
    drift_round: int | None
    drift_post_label: np.ndarray  # ground-truth cluster after drift_round
    drift_post_data: list[tuple[np.ndarray, np.ndarray]]  # post-drift train data
    n_classes: int = 2  # label cardinality; 2 = binary, >2 = multinomial

    @property
    def n_clients(self) -> int:
        return len(self.specs)

    @property
    def n_clusters(self) -> int:
        return self.catalog.n_tasks

    def train_data(self, round_idx: int) -> list[tuple[np.ndarray, np.ndarray]]:
        if self.drift_round is None or round_idx < self.drift_round:
            return self.train
        return self.drift_post_data


def make_cohort(
    n_clients: int = 30,
    n_clusters: int = 3,
    n_features: int = 5,
    samples_per_client: int = 120,
    boundary_fraction: float = 0.3,
    drift_round: int | None = 15,
    drift_fraction: float = 0.3,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> Cohort:
    rng = np.random.default_rng(seed)
    catalog = make_tasks(n_clusters, n_features, seed)

    # Primary assignment.
    primary = rng.integers(0, n_clusters, size=n_clients)
    is_boundary = np.zeros(n_clients, dtype=bool)
    is_boundary[: int(round(boundary_fraction * n_clients))] = True
    rng.shuffle(is_boundary)

    # Drift: switch primary task for drift_fraction of NON-boundary clients
    # (so the drift signal is clean rather than entangled with boundary
    # ambiguity).
    drift_to = np.full(n_clients, -1, dtype=int)
    if drift_round is not None:
        candidates = np.where(~is_boundary)[0]
        n_drift = int(round(drift_fraction * n_clients))
        n_drift = min(n_drift, len(candidates))
        drift_ids = rng.choice(candidates, size=n_drift, replace=False)
        for i in drift_ids:
            new_task = int(rng.integers(0, n_clusters))
            while new_task == primary[i]:
                new_task = int(rng.integers(0, n_clusters))
            drift_to[i] = new_task

    specs: list[ClientSpec] = []
    train: list[tuple[np.ndarray, np.ndarray]] = []
    test: list[tuple[np.ndarray, np.ndarray]] = []
    drift_post_data: list[tuple[np.ndarray, np.ndarray]] = []

    for i in range(n_clients):
        sec = None
        if is_boundary[i]:
            sec = int(rng.integers(0, n_clusters))
            while sec == int(primary[i]):
                sec = int(rng.integers(0, n_clusters))
        spec = ClientSpec(
            primary=int(primary[i]),
            secondary=sec,
            drift_to=int(drift_to[i]) if drift_to[i] >= 0 else None,
        )
        specs.append(spec)

        # Pre-drift train data
        total = samples_per_client
        if spec.secondary is None:
            X_train, y_train = sample_from_task(catalog, spec.primary, total, rng)
        else:
            half = total // 2
            X_a, y_a = sample_from_task(catalog, spec.primary, half, rng)
            X_b, y_b = sample_from_task(catalog, spec.secondary, total - half, rng)
            X_train = np.concatenate([X_a, X_b], axis=0)
            y_train = np.concatenate([y_a, y_b], axis=0)
        # Shuffle.
        idx = rng.permutation(len(y_train))
        X_train, y_train = X_train[idx], y_train[idx]

        # Test data drawn from the same distribution (matches the
        # post-train task identity for non-drift clients).
        n_test = max(int(total * test_fraction), 20)
        if spec.secondary is None:
            X_test, y_test = sample_from_task(catalog, spec.primary, n_test, rng)
        else:
            half = n_test // 2
            X_t_a, y_t_a = sample_from_task(catalog, spec.primary, half, rng)
            X_t_b, y_t_b = sample_from_task(
                catalog, spec.secondary, n_test - half, rng
            )
            X_test = np.concatenate([X_t_a, X_t_b], axis=0)
            y_test = np.concatenate([y_t_a, y_t_b], axis=0)

        train.append((X_train, y_train))
        test.append((X_test, y_test))

        # Post-drift train: switch to drift_to if applicable.
        if spec.drift_to is not None:
            X_post, y_post = sample_from_task(catalog, spec.drift_to, total, rng)
            drift_post_data.append((X_post, y_post))
        else:
            drift_post_data.append((X_train, y_train))

    primary_label = primary.astype(int)
    drift_post_label = primary_label.copy()
    for i in range(n_clients):
        if specs[i].drift_to is not None:
            drift_post_label[i] = specs[i].drift_to

    return Cohort(
        catalog=catalog,
        specs=specs,
        train=train,
        test=test,
        primary_label=primary_label,
        is_boundary=is_boundary,
        drift_round=drift_round,
        drift_post_label=drift_post_label,
        drift_post_data=drift_post_data,
    )


# ---------- UCI Adult tabular cohort (binary income classification) ----------


def _load_adult_cached(cache_dir: str = "/tmp/cfl_data_cache") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load UCI Adult via sklearn.fetch_openml, cached to disk.

    Returns
    -------
    X: (n, d) float matrix of standardized numeric features.
    y: (n,) int target (>50K income).
    feature_names: list of column names after one-hot encoding.
    """
    import os
    import pickle

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "adult.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    from sklearn.datasets import fetch_openml

    bundle = fetch_openml("adult", version=2, as_frame=True, parser="liac-arff")
    df = bundle.data.copy()
    target = bundle.target.astype(str)
    y = (target == ">50K").astype(int).to_numpy()

    cat_cols = df.select_dtypes(include=["category", "object"]).columns.tolist()
    num_cols = [c for c in df.columns if c not in cat_cols]

    # Drop rows with any NaN to keep the loader simple.
    keep = np.asarray(~df[num_cols].isna().any(axis=1)).astype(bool).copy()
    for c in cat_cols:
        keep &= np.asarray(~df[c].isna()).astype(bool)
    df = df.loc[keep].reset_index(drop=True)
    y = y[keep]

    # Standardize numeric, one-hot categorical.
    num = df[num_cols].to_numpy(dtype=float)
    num_mean = num.mean(axis=0, keepdims=True)
    num_std = num.std(axis=0, keepdims=True) + 1e-9
    num = (num - num_mean) / num_std

    ohs = []
    feature_names = list(num_cols)
    for c in cat_cols:
        col = df[c].astype("category")
        levels = list(col.cat.categories)
        for lev in levels:
            ohs.append((col == lev).to_numpy(dtype=float).reshape(-1, 1))
            feature_names.append(f"{c}={lev}")
    X = np.concatenate([num] + ohs, axis=1) if ohs else num

    out = (X, y, feature_names)
    with open(cache_path, "wb") as f:
        pickle.dump(out, f)
    return out


def make_adult_cohort(
    n_clients: int = 30,
    samples_per_client: int = 200,
    boundary_fraction: float = 0.3,
    drift_round: int | None = None,
    drift_fraction: float = 0.3,
    test_fraction: float = 0.25,
    partition_by: str = "education-num",
    n_profiles: int = 3,
    seed: int = 0,
) -> Cohort:
    """Build a profile-based non-IID CFL cohort on UCI Adult.

    Partition strategy: sort by the values of ``partition_by`` (an
    underlying numeric feature index, e.g. ``education-num``,
    ``age``, ``capital-gain``), split into ``n_profiles`` quantile
    bins, and treat each bin as a "task profile". Each client draws
    its samples from one bin (non-boundary) or from a 50/50 mix of
    two bins (boundary). Drift swaps the bin of a configurable
    fraction of non-boundary clients at ``drift_round``.

    Why this matches the CFL setup: clients in the same bin see
    similar feature distributions and similar P(y|x), so they form a
    natural cluster; clients on the boundary have a hybrid
    distribution that HFLTS-CFL's hesitation envelope is designed
    to surface.
    """
    X_all, y_all, names = _load_adult_cached()
    if partition_by not in names:
        raise ValueError(
            f"unknown partition feature {partition_by!r}; "
            f"available: {names[:10]}..."
        )
    col = names.index(partition_by)
    rng = np.random.default_rng(seed)

    # Build profiles by quantile bins on the partition column.
    qs = np.linspace(0, 1, n_profiles + 1)
    edges = np.quantile(X_all[:, col], qs)
    edges[-1] = edges[-1] + 1e-9
    profile_of = np.digitize(X_all[:, col], edges[1:-1], right=False).astype(int)
    # profile_of in [0, n_profiles)

    # Per-profile pools.
    pools = [np.where(profile_of == p)[0] for p in range(n_profiles)]
    for p in range(n_profiles):
        rng.shuffle(pools[p])

    # Replace the TaskCatalog field with a stand-in for compatibility.
    catalog = TaskCatalog(W=np.zeros((n_profiles, X_all.shape[1])), b=np.zeros(n_profiles))

    primary = rng.integers(0, n_profiles, size=n_clients)
    is_boundary = np.zeros(n_clients, dtype=bool)
    is_boundary[: int(round(boundary_fraction * n_clients))] = True
    rng.shuffle(is_boundary)

    drift_to = np.full(n_clients, -1, dtype=int)
    if drift_round is not None and n_profiles > 1:
        candidates = np.where(~is_boundary)[0]
        n_drift = min(int(round(drift_fraction * n_clients)), len(candidates))
        drift_ids = rng.choice(candidates, size=n_drift, replace=False)
        for i in drift_ids:
            new_p = int(rng.integers(0, n_profiles))
            while new_p == int(primary[i]):
                new_p = int(rng.integers(0, n_profiles))
            drift_to[i] = new_p

    def draw_from_profile(p: int, n: int) -> tuple[np.ndarray, np.ndarray]:
        if n <= 0 or len(pools[p]) == 0:
            return np.empty((0, X_all.shape[1])), np.empty(0, dtype=int)
        idx = rng.choice(pools[p], size=n, replace=len(pools[p]) < n)
        return X_all[idx].copy(), y_all[idx].copy()

    specs: list[ClientSpec] = []
    train: list[tuple[np.ndarray, np.ndarray]] = []
    test: list[tuple[np.ndarray, np.ndarray]] = []
    drift_post_data: list[tuple[np.ndarray, np.ndarray]] = []

    n_test = max(20, int(samples_per_client * test_fraction))
    for i in range(n_clients):
        sec = None
        if is_boundary[i]:
            sec = int(rng.integers(0, n_profiles))
            while sec == int(primary[i]):
                sec = int(rng.integers(0, n_profiles))
        spec = ClientSpec(
            primary=int(primary[i]),
            secondary=sec,
            drift_to=int(drift_to[i]) if drift_to[i] >= 0 else None,
        )
        specs.append(spec)
        if sec is None:
            X_tr, y_tr = draw_from_profile(spec.primary, samples_per_client)
            X_te, y_te = draw_from_profile(spec.primary, n_test)
        else:
            half = samples_per_client // 2
            Xa, ya = draw_from_profile(spec.primary, half)
            Xb, yb = draw_from_profile(spec.secondary, samples_per_client - half)
            X_tr = np.concatenate([Xa, Xb], axis=0)
            y_tr = np.concatenate([ya, yb], axis=0)
            half_t = n_test // 2
            Xta, yta = draw_from_profile(spec.primary, half_t)
            Xtb, ytb = draw_from_profile(spec.secondary, n_test - half_t)
            X_te = np.concatenate([Xta, Xtb], axis=0)
            y_te = np.concatenate([yta, ytb], axis=0)
        # Shuffle.
        idx = rng.permutation(len(y_tr))
        X_tr, y_tr = X_tr[idx], y_tr[idx]
        train.append((X_tr, y_tr))
        test.append((X_te, y_te))
        if spec.drift_to is not None:
            Xp, yp = draw_from_profile(spec.drift_to, samples_per_client)
            drift_post_data.append((Xp, yp))
        else:
            drift_post_data.append((X_tr, y_tr))

    primary_label = primary.astype(int)
    drift_post_label = primary_label.copy()
    for i in range(n_clients):
        if specs[i].drift_to is not None:
            drift_post_label[i] = specs[i].drift_to
    return Cohort(
        catalog=catalog,
        specs=specs,
        train=train,
        test=test,
        primary_label=primary_label,
        is_boundary=is_boundary,
        drift_round=drift_round,
        drift_post_label=drift_post_label,
        drift_post_data=drift_post_data,
    )


# ---------- UCI HAR cohort (multi-class activity recognition, natural FL) ----------


def _load_har_cached(cache_dir: str = "/tmp/cfl_data_cache") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load UCI HAR via fetch_openml, cached. Returns (X, y, subject_id).

    HAR contains 561 statistical features per accelerometer/gyroscope
    window plus a per-window subject identifier (1..30). Subjects are
    a natural FL partition.
    """
    import os
    import pickle

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "har.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    from sklearn.datasets import fetch_openml

    bundle = fetch_openml("har", version=1, as_frame=True, parser="liac-arff")
    df = bundle.data.copy()
    target = bundle.target.astype(str)
    label_to_int = {lab: i for i, lab in enumerate(sorted(target.unique()))}
    y = target.map(label_to_int).to_numpy().astype(int)

    # Subject column heuristic: try common names; fall back to None.
    subj = None
    for col in ("subject", "subject_identifier", "subjects"):
        if col in df.columns:
            subj = df[col].to_numpy().astype(int)
            df = df.drop(columns=[col])
            break
    if subj is None:
        # Best effort: hash-based grouping into 30 pseudo-subjects.
        subj = (np.arange(len(df)) % 30) + 1

    X = df.to_numpy(dtype=float)
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True) + 1e-9
    X = (X - X_mean) / X_std

    out = (X, y, subj)
    with open(cache_path, "wb") as f:
        pickle.dump(out, f)
    return out


def make_har_cohort(
    n_profiles: int = 3,
    boundary_fraction: float = 0.25,
    drift_round: int | None = None,
    drift_fraction: float = 0.3,
    test_fraction: float = 0.25,
    seed: int = 0,
    max_subjects: int | None = 30,
) -> Cohort:
    """Build a natural FL cohort from UCI HAR.

    Each subject becomes a federated client. Subjects are then grouped
    into ``n_profiles`` task profiles via k-means on the per-subject
    feature mean so that subjects with similar sensor signatures share
    a profile - that grouping is the CFL ground truth. Boundary
    clients mix two profiles' samples 50/50. Drift swaps the profile
    of a fraction of non-boundary clients at ``drift_round`` by
    rerouting their training stream to a different cluster's
    samples (their evaluation set is held fixed).
    """
    X, y, subj = _load_har_cached()
    subjects = np.unique(subj)
    if max_subjects is not None and len(subjects) > max_subjects:
        subjects = subjects[:max_subjects]
    rng = np.random.default_rng(seed)

    # Subject-level mean features and a k-means grouping into profiles.
    sub_means = np.stack([X[subj == s].mean(axis=0) for s in subjects], axis=0)
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=n_profiles, random_state=seed, n_init=10).fit(sub_means)
    subject_profile = km.labels_.astype(int)

    catalog = TaskCatalog(W=np.zeros((n_profiles, X.shape[1])), b=np.zeros(n_profiles))
    n_clients = len(subjects)
    primary = subject_profile.copy()
    is_boundary = np.zeros(n_clients, dtype=bool)
    n_b = int(round(boundary_fraction * n_clients))
    is_boundary[:n_b] = True
    rng.shuffle(is_boundary)

    drift_to = np.full(n_clients, -1, dtype=int)
    if drift_round is not None and n_profiles > 1:
        candidates = np.where(~is_boundary)[0]
        n_drift = min(int(round(drift_fraction * n_clients)), len(candidates))
        drift_ids = rng.choice(candidates, size=n_drift, replace=False)
        for i in drift_ids:
            new_p = int(rng.integers(0, n_profiles))
            while new_p == int(primary[i]):
                new_p = int(rng.integers(0, n_profiles))
            drift_to[i] = new_p

    # Per-profile pools of (X, y) samples drawn from subjects in that profile.
    profile_pools = []
    for p in range(n_profiles):
        in_profile = np.isin(subj, subjects[subject_profile == p])
        idx = np.where(in_profile)[0]
        rng.shuffle(idx)
        profile_pools.append(idx)

    def draw_from_profile(p: int, n: int) -> tuple[np.ndarray, np.ndarray]:
        pool = profile_pools[p]
        if n <= 0 or len(pool) == 0:
            return np.empty((0, X.shape[1])), np.empty(0, dtype=int)
        sel = rng.choice(pool, size=n, replace=len(pool) < n)
        return X[sel].copy(), y[sel].copy()

    specs: list[ClientSpec] = []
    train: list[tuple[np.ndarray, np.ndarray]] = []
    test: list[tuple[np.ndarray, np.ndarray]] = []
    drift_post_data: list[tuple[np.ndarray, np.ndarray]] = []
    for i, s in enumerate(subjects):
        own_idx = np.where(subj == s)[0]
        rng.shuffle(own_idx)
        n_total = len(own_idx)
        n_test = max(20, int(n_total * test_fraction))
        test_idx = own_idx[:n_test]
        train_idx = own_idx[n_test:]
        X_te, y_te = X[test_idx].copy(), y[test_idx].copy()
        sec = None
        if is_boundary[i]:
            sec = int(rng.integers(0, n_profiles))
            while sec == int(primary[i]):
                sec = int(rng.integers(0, n_profiles))
            # Mix half local data with half drawn from secondary profile.
            half = len(train_idx) // 2
            X_a, y_a = X[train_idx[:half]].copy(), y[train_idx[:half]].copy()
            X_b, y_b = draw_from_profile(sec, len(train_idx) - half)
            X_tr = np.concatenate([X_a, X_b], axis=0)
            y_tr = np.concatenate([y_a, y_b], axis=0)
        else:
            X_tr = X[train_idx].copy()
            y_tr = y[train_idx].copy()
        spec = ClientSpec(
            primary=int(primary[i]),
            secondary=sec,
            drift_to=int(drift_to[i]) if drift_to[i] >= 0 else None,
        )
        specs.append(spec)
        train.append((X_tr, y_tr))
        test.append((X_te, y_te))
        if spec.drift_to is not None:
            X_post, y_post = draw_from_profile(spec.drift_to, len(train_idx))
            drift_post_data.append((X_post, y_post))
        else:
            drift_post_data.append((X_tr, y_tr))

    drift_post_label = primary.copy()
    for i in range(n_clients):
        if specs[i].drift_to is not None:
            drift_post_label[i] = specs[i].drift_to
    return Cohort(
        catalog=catalog,
        specs=specs,
        train=train,
        test=test,
        primary_label=primary.copy(),
        is_boundary=is_boundary,
        drift_round=drift_round,
        drift_post_label=drift_post_label,
        drift_post_data=drift_post_data,
        n_classes=int(y.max() + 1),
    )
