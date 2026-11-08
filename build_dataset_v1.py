
"""
build_dataset.py
----------------
Tạo dataset ST-GCN từ skeleton NPZ.

Quy tắc nhãn:
    - Video có ID trong CSV: toàn bộ window là Fall (1).
    - Video Fall không có trong CSV: bỏ qua.
    - Video ADL không có trong CSV: ADL (0).

Window:
    - 10 frames
    - Stride 5 frames

Input:
    output/npz_filtered/*.npz
    data/fall_intervals_filtered_fixed.csv

Output:
    output/dataset.npz
"""

from pathlib import Path
import csv
import re
import numpy as np

from src.skeleton_io import make_windows
from src.features import normalize_batch


# ============================================================
# 1. CONFIGURATION
# ============================================================

NPZ_DIR = Path("output/npz_filtered")

INTERVALS_CSV = Path(
    "data/fall_intervals_filtered_fixed.csv"
)

OUTPUT_PATH = Path("output/datasetv1.npz")

WINDOW_FRAMES = 17
STRIDE_FRAMES = 1

MIN_VALID_RATIO = 0.5

VAL_FRAC = 0.1
TEST_FRAC = 0.1

RANDOM_SEED = 42


# ============================================================
# 2. VIDEO ID
# ============================================================

def normalize_id(name):
    """
    Chuẩn hóa ID để so sánh CSV và NPZ.
    """
    name = Path(str(name)).stem.lower()

    # Bỏ token Fall / ADL
    name = re.sub(
        r"(^|[_-])(fall|adl)(?=[_-]|$)",
        "",
        name
    )

    # Bỏ dấu phân cách
    name = re.sub(r"[^a-z0-9]", "", name)

    return name


def subject_of(video_id):
    """
    SBJ01_Fall_LOC1-ACT13-B-1 -> SBJ01
    SBJ_01_Fall_LOC1-ACT13-B-1 -> SBJ01
    """
    match = re.match(
        r"^(SBJ_?\d+)",
        str(video_id),
        flags=re.IGNORECASE
    )

    if match:
        return match.group(1).replace("_", "").upper()

    return str(video_id).split("_")[0]


def is_fall_video(video_id):
    """
    Kiểm tra token Fall trong tên video.
    """
    tokens = re.split(
        r"[_\-.]+",
        str(video_id).lower()
    )

    return "fall" in tokens


# ============================================================
# 3. LOAD CSV
# ============================================================

def load_fall_ids(csv_path):
    """
    Đọc danh sách video có trong CSV.

    Chỉ sử dụng video_id để xác định nhãn.
    Không sử dụng start_sec/end_sec vì toàn bộ video
    được gán nhãn Fall.
    """

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy CSV: {csv_path}"
        )

    fall_ids = set()

    with open(
        csv_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        if not reader.fieldnames or "video_id" not in reader.fieldnames:
            raise ValueError(
                "CSV phải có cột video_id."
            )

        for row in reader:
            video_id = row["video_id"].strip()

            if video_id:
                fall_ids.add(normalize_id(video_id))

    return fall_ids


# ============================================================
# 4. LOAD NPZ
# ============================================================

def load_npz(npz_path):
    """
    Đọc dữ liệu skeleton từ NPZ.

    seq:    [T,17,3]
    mask:   [T,17]
    valid:  [T]
    fps:    scalar
    aspect: scalar
    """

    with np.load(
        npz_path,
        allow_pickle=False
    ) as data:

        required = {"seq", "mask", "fps"}

        missing = required - set(data.files)

        if missing:
            raise ValueError(
                f"{npz_path.name} thiếu trường: {missing}"
            )

        seq = data["seq"].astype(np.float32)
        mask = data["mask"].astype(np.float32)

        if "valid" in data.files:
            valid = data["valid"].astype(np.float32)
        else:
            valid = (
                mask.sum(axis=1) > 0
            ).astype(np.float32)

        fps = float(
            np.asarray(data["fps"]).item()
        )

        if "aspect" in data.files:
            aspect = float(
                np.asarray(data["aspect"]).item()
            )
        else:
            aspect = 1.0

    if seq.ndim != 3 or seq.shape[1:] != (17, 3):
        raise ValueError(
            f"{npz_path.name}: seq phải có dạng [T,17,3], "
            f"nhận được {seq.shape}"
        )

    if mask.shape != seq.shape[:2]:
        raise ValueError(
            f"{npz_path.name}: mask không khớp seq."
        )

    if valid.shape[0] != seq.shape[0]:
        raise ValueError(
            f"{npz_path.name}: valid không khớp seq."
        )

    if fps <= 0:
        raise ValueError(
            f"{npz_path.name}: FPS không hợp lệ."
        )

    return {
        "seq": seq,
        "mask": mask,
        "valid": valid,
        "fps": fps,
        "aspect": aspect,
        "video_id": npz_path.stem
    }


# ============================================================
# 5. SPLIT SUBJECT
# ============================================================

def split_subjects(
    subjects,
    val_frac=0.15,
    test_frac=0.15,
    seed=42
):

    subjects = sorted(set(subjects))

    if len(subjects) < 3:
        raise ValueError(
            f"Cần ít nhất 3 subject, hiện có {len(subjects)}."
        )

    rng = np.random.RandomState(seed)

    subjects = np.asarray(
        subjects,
        dtype=object
    )

    rng.shuffle(subjects)

    n = len(subjects)

    n_test = max(
        1,
        int(round(n * test_frac))
    )

    n_val = max(
        1,
        int(round(n * val_frac))
    )

    while n_test + n_val >= n:
        if n_test >= n_val and n_test > 1:
            n_test -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            raise ValueError(
                "Không đủ subject để chia dataset."
            )

    test_subjects = set(
        subjects[:n_test]
    )

    val_subjects = set(
        subjects[n_test:n_test + n_val]
    )

    train_subjects = set(
        subjects[n_test + n_val:]
    )

    return (
        train_subjects,
        val_subjects,
        test_subjects
    )


# ============================================================
# 6. BUILD DATASET
# ============================================================

def build():

    print("========== BUILD DATASET ==========")

    if not NPZ_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục: {NPZ_DIR}"
        )

    npz_files = sorted(
        NPZ_DIR.glob("*.npz")
    )

    if not npz_files:
        raise RuntimeError(
            "Không tìm thấy file NPZ."
        )

    fall_ids = load_fall_ids(
        INTERVALS_CSV
    )

    print("Tổng file NPZ:", len(npz_files))
    print("Số ID Fall trong CSV:", len(fall_ids))

    all_X = []
    all_M = []
    all_y = []

    all_video = []
    all_subject = []

    skipped_fall = 0
    invalid_videos = 0
    no_window_videos = 0

    # --------------------------------------------------------
    # Duyệt các video
    # --------------------------------------------------------

    for npz_path in npz_files:

        video_id = npz_path.stem
        key = normalize_id(video_id)

        subject = subject_of(video_id)

        # Xác định nhãn theo danh sách CSV
        if key in fall_ids:
            label = 1

        elif is_fall_video(video_id):
            print(
                f"[SKIP FALL] Không có trong CSV: {video_id}"
            )
            skipped_fall += 1
            continue

        else:
            label = 0

        # Đọc NPZ
        try:
            item = load_npz(npz_path)

        except Exception as e:
            print(
                f"[ERROR] {video_id}: {e}"
            )
            invalid_videos += 1
            continue

        fps = item["fps"]

        # Chuyển số frame thành giây để sử dụng
        # make_windows hiện có trong skeleton_io.py
        win_sec = WINDOW_FRAMES / fps
        stride_sec = STRIDE_FRAMES / fps

        # Tạo window, giữ nguyên trình tự thời gian
        try:
            X, M, y, groups = make_windows(
                item,
                win_sec=win_sec,
                stride_sec=stride_sec,
                fall_intervals=None,
                min_valid_ratio=MIN_VALID_RATIO
            )

        except Exception as e:
            print(
                f"[ERROR WINDOW] {video_id}: {e}"
            )
            invalid_videos += 1
            continue

        if len(X) == 0:
            print(
                f"[NO WINDOW] {video_id}"
            )
            no_window_videos += 1
            continue

        # Gán nhãn toàn bộ window theo video
        y[:] = label

        # Chuẩn hóa dữ liệu
        X = normalize_batch(
            X,
            M,
            item["aspect"]
        )

        all_X.append(X)
        all_M.append(M)
        all_y.append(y)

        all_video.extend(
            [video_id] * len(y)
        )

        all_subject.extend(
            [subject] * len(y)
        )

        fall_count = (
            len(y) if label == 1 else 0
        )

        adl_count = (
            len(y) if label == 0 else 0
        )

        print(
            f"[OK] {video_id:40s} "
            f"windows={len(X):4d} "
            f"Fall={fall_count:4d} "
            f"ADL={adl_count:4d}"
        )

    # --------------------------------------------------------
    # Ghép dữ liệu
    # --------------------------------------------------------

    if not all_X:
        raise RuntimeError(
            "Không tạo được cửa sổ dữ liệu."
        )

    X = np.concatenate(
        all_X,
        axis=0
    ).astype(np.float32)

    M = np.concatenate(
        all_M,
        axis=0
    ).astype(np.float32)

    y = np.concatenate(
        all_y,
        axis=0
    ).astype(np.int64)

    video_ids = np.asarray(
        all_video,
        dtype=str
    )

    subjects = np.asarray(
        all_subject,
        dtype=str
    )

    # --------------------------------------------------------
    # Tổng kết
    # --------------------------------------------------------

    total_fall = int(
        np.sum(y == 1)
    )

    total_adl = int(
        np.sum(y == 0)
    )

    print("\n========== TỔNG KẾT ==========")

    print("Tổng số cửa sổ:", len(X))
    print("Tổng số Fall:", total_fall)
    print("Tổng số ADL:", total_adl)
    print("Fall bị bỏ qua:", skipped_fall)
    print("Video lỗi:", invalid_videos)
    print("Video không đủ cửa sổ:", no_window_videos)
    print("Kích thước X:", X.shape)

    if total_fall == 0 or total_adl == 0:
        raise RuntimeError(
            "Dataset phải có cả Fall và ADL. "
            "Hãy kiểm tra danh sách ID trong CSV."
        )

    # --------------------------------------------------------
    # Chia train / val / test theo subject
    # --------------------------------------------------------

    train_subjects, val_subjects, test_subjects = split_subjects(
        subjects,
        val_frac=VAL_FRAC,
        test_frac=TEST_FRAC,
        seed=RANDOM_SEED
    )

    train_idx = np.isin(
        subjects,
        list(train_subjects)
    )

    val_idx = np.isin(
        subjects,
        list(val_subjects)
    )

    test_idx = np.isin(
        subjects,
        list(test_subjects)
    )

    def print_split(name, idx):

        labels = y[idx]

        print(
            f"{name:5s}: {int(idx.sum()):5d} cửa sổ | "
            f"Fall: {int(np.sum(labels == 1)):4d} | "
            f"ADL: {int(np.sum(labels == 0)):4d}"
        )

    print("\n========== CHIA DATASET ==========")

    print(
        "Train subjects:",
        sorted(train_subjects)
    )

    print(
        "Val subjects:",
        sorted(val_subjects)
    )

    print(
        "Test subjects:",
        sorted(test_subjects)
    )

    print_split("train", train_idx)
    print_split("val", val_idx)
    print_split("test", test_idx)

    # --------------------------------------------------------
    # Lưu dataset
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        OUTPUT_PATH,

        X_train=X[train_idx],
        y_train=y[train_idx],
        video_train=video_ids[train_idx],
        M_train=M[train_idx],

        X_val=X[val_idx],
        y_val=y[val_idx],
        video_val=video_ids[val_idx],
        M_val=M[val_idx],

        X_test=X[test_idx],
        y_test=y[test_idx],
        video_test=video_ids[test_idx],
        M_test=M[test_idx],

        train_subjects=np.asarray(
            sorted(train_subjects),
            dtype=str
        ),

        val_subjects=np.asarray(
            sorted(val_subjects),
            dtype=str
        ),

        test_subjects=np.asarray(
            sorted(test_subjects),
            dtype=str
        )
    )

    print("\n========== HOÀN TẤT ==========")
    print("Đã lưu:", OUTPUT_PATH)
    print("X tổng:", X.shape)
    print("Y tổng:", y.shape)


# ============================================================
# 7. MAIN
# ============================================================

if __name__ == "__main__":
    build()
