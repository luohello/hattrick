"""Split the single-class GEANT traffic matrices into Hattrick's 3 classes.

This reuses only the class-splitting rule from the Abilene preparation script:
high ~ U(0.45, 0.50), medium ~ U(0.40, 0.45), low = remainder.
The existing GEANT topology, pairs, and manifest are left unchanged.
"""

import argparse
import io
import os
import pickle
import zipfile

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--project-root",
        help="Hattrick project root. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing class files. By default completed files are skipped.",
    )
    return parser.parse_args()


def find_project_root(explicit_root=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    current_dir = os.getcwd()
    candidates = []
    if explicit_root:
        candidates.append(explicit_root)
    if os.environ.get("HATTRICK_ROOT"):
        candidates.append(os.environ["HATTRICK_ROOT"])

    for start in (script_dir, current_dir):
        path = start
        while True:
            candidates.extend((path, os.path.join(path, "Hattrick")))
            parent = os.path.dirname(path)
            if parent == path:
                break
            path = parent

    checked = []
    for candidate in candidates:
        candidate = os.path.abspath(os.path.expanduser(candidate))
        if candidate in checked:
            continue
        checked.append(candidate)
        required = (
            os.path.join(candidate, "manifest", "geant_manifest.txt"),
            os.path.join(candidate, "pairs", "geant", "t1.pkl"),
            os.path.join(candidate, "traffic_matrices"),
        )
        if all(os.path.exists(path) for path in required):
            return candidate

    raise FileNotFoundError(
        "Could not locate the Hattrick project root. Pass "
        "--project-root /mnt/data0/Hattrick. Checked: " + ", ".join(checked)
    )


def load_manifest_names(manifest_path):
    names = []
    with open(manifest_path, encoding="utf-8") as manifest:
        for line in manifest:
            fields = line.strip().split(",")
            if len(fields) != 3:
                raise ValueError(f"Invalid manifest row: {line!r}")
            names.append(fields[2])
    return names


def available_source_names(tm_root):
    source_dir = os.path.join(tm_root, "geant")
    source_zip = os.path.join(tm_root, "geant.zip")
    if os.path.isdir(source_dir):
        return {name for name in os.listdir(source_dir) if name.endswith(".pkl")}
    if os.path.isfile(source_zip):
        with zipfile.ZipFile(source_zip) as archive:
            return {
                name.rsplit("/", 1)[-1]
                for name in archive.namelist()
                if name.startswith("geant/") and name.endswith(".pkl")
            }
    return set()


def make_reader(tm_root):
    source_dir = os.path.join(tm_root, "geant")
    source_zip = os.path.join(tm_root, "geant.zip")

    if os.path.isdir(source_dir):
        def read_tm(name):
            with open(os.path.join(source_dir, name), "rb") as source:
                return pickle.load(source)

        return read_tm, source_dir

    if os.path.isfile(source_zip):
        archive = zipfile.ZipFile(source_zip)

        def read_tm(name):
            return pickle.load(io.BytesIO(archive.read(f"geant/{name}")))

        return read_tm, source_zip

    raise FileNotFoundError(
        f"Expected either {source_dir} or {source_zip}"
    )


def atomic_pickle_dump(value, path):
    temporary = f"{path}.tmp"
    with open(temporary, "wb") as output:
        pickle.dump(value, output, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def main():
    args = parse_args()
    project_root = find_project_root(args.project_root)
    tm_root = os.path.join(project_root, "traffic_matrices")
    manifest_path = os.path.join(project_root, "manifest", "geant_manifest.txt")
    pairs_path = os.path.join(project_root, "pairs", "geant", "t1.pkl")

    rng = np.random.RandomState(args.seed)
    names = load_manifest_names(manifest_path)
    available_names = available_source_names(tm_root)
    missing_names = [name for name in names if name not in available_names]
    if missing_names:
        preview = ", ".join(missing_names[:10])
        raise FileNotFoundError(
            f"Manifest references {len(missing_names)} TM file(s) missing from the "
            f"GEANT source: {preview}. Fix the manifest before generating outputs."
        )
    read_tm, source = make_reader(tm_root)

    with open(pairs_path, "rb") as pairs_file:
        num_pairs = len(pickle.load(pairs_file))

    output_dirs = [os.path.join(tm_root, f"geant_{priority}") for priority in (1, 2, 3)]
    for directory in output_dirs:
        os.makedirs(directory, exist_ok=True)

    print(f"Project root: {project_root}")
    print(f"Source: {source}")
    print(f"Snapshots: {len(names)}, pairs per snapshot: {num_pairs}, seed: {args.seed}")

    for index, name in enumerate(names, start=1):
        # Advance the RNG for every manifest row, including completed rows, so a
        # resumed run produces the same split as an uninterrupted run.
        high_fraction = rng.uniform(0.45, 0.50, size=num_pairs)
        medium_fraction = rng.uniform(0.40, 0.45, size=num_pairs)
        low_fraction = 1.0 - high_fraction - medium_fraction

        output_paths = [os.path.join(directory, name) for directory in output_dirs]
        existing = [os.path.exists(path) for path in output_paths]
        if all(existing) and not args.overwrite:
            continue
        if any(existing) and not args.overwrite:
            raise RuntimeError(
                f"Partial output exists for {name}; rerun with --overwrite after inspection"
            )

        matrix = np.asarray(read_tm(name)).reshape(-1)
        if matrix.size != num_pairs:
            raise ValueError(
                f"{name}: TM has {matrix.size} demands, expected {num_pairs}"
            )
        if np.any(matrix < 0):
            raise ValueError(f"{name}: traffic matrix contains negative demand")

        class_matrices = (
            matrix * high_fraction,
            matrix * medium_fraction,
            matrix * low_fraction,
        )

        if not np.allclose(sum(class_matrices), matrix):
            raise RuntimeError(f"{name}: class split does not preserve total traffic")

        for class_matrix, output_path in zip(class_matrices, output_paths):
            atomic_pickle_dump(class_matrix, output_path)

        if index % 500 == 0 or index == len(names):
            print(f"Processed {index}/{len(names)}")

    print("GEANT split completed")


if __name__ == "__main__":
    main()
