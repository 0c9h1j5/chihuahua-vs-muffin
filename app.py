# ruff: noqa: E501  -- Inline CSS and HTML templates are intentionally kept readable.
from __future__ import annotations

import ast
import io
import random
import re
import time
from datetime import datetime
from html import escape
from pathlib import Path
from typing import TypedDict

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
SCRIPT_1016_PATH = ROOT / "1016.py"
README_PATH = ROOT / "README.md"
RESULT_LOG_PATH = ROOT / "result.log"
TUNE_LOG_PATH = ROOT / "tune.log"
CLASSES = ["chihuahua", "muffin"]
CLASS_KOREAN = ["치와와", "머핀"]


class ScriptConfig(TypedDict):
    batch_size: int
    lr: float
    momentum: float
    optimizer_name: str
    epochs: int
    resize: tuple[int, int]
    normalize_mean: tuple[float, float, float]
    normalize_std: tuple[float, float, float]
    train_root: str
    test_root: str
    checkpoint_rel_path: str


class TestResults(TypedDict):
    counts: list[list[int]]
    total: int
    correct: int
    accuracy: float
    error: float


def _literal(node: ast.AST | None, fallback: object) -> object:
    if node is None:
        return fallback
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return fallback


def _tuple_values(
    node: ast.AST | None, fallback: tuple[float, ...]
) -> tuple[float, ...]:
    value = _literal(node, fallback)
    if isinstance(value, (tuple, list)):
        try:
            return tuple(float(item) for item in value)
        except (TypeError, ValueError):
            return fallback
    return fallback


def parse_1016_config(source: str | None = None) -> ScriptConfig:
    """Safely read training and transform settings without importing 1016.py."""
    config: ScriptConfig = {
        "batch_size": 2,
        "lr": 0.002,
        "momentum": 0.9,
        "optimizer_name": "SGD",
        "epochs": 10,
        "resize": (224, 224),
        "normalize_mean": (0.5, 0.5, 0.5),
        "normalize_std": (0.5, 0.5, 0.5),
        "train_root": "./archive/train",
        "test_root": "./archive/test",
        "checkpoint_rel_path": "./yes_no_net.pth",
    }
    if source is None:
        try:
            source = SCRIPT_1016_PATH.read_text(encoding="utf-8")
        except OSError:
            return config
    try:
        tree = ast.parse(source, filename=str(SCRIPT_1016_PATH))
    except SyntaxError:
        return config

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"batch_size", "PATH"}:
                    value = _literal(node.value, None)
                    if target.id == "batch_size" and isinstance(value, int):
                        config["batch_size"] = value
                    elif target.id == "PATH" and isinstance(value, str):
                        config["checkpoint_rel_path"] = value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            value = _literal(node.value, None)
            if node.target.id == "batch_size" and isinstance(value, int):
                config["batch_size"] = value
            elif node.target.id == "PATH" and isinstance(value, str):
                config["checkpoint_rel_path"] = value

    for statement in ast.walk(tree):
        if isinstance(statement, ast.Assign):
            assigned_names = {
                target.id for target in statement.targets if isinstance(target, ast.Name)
            }
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            assigned_names = {statement.target.id}
        else:
            continue
        if statement.value is None:
            continue
        for call in ast.walk(statement.value):
            if not isinstance(call, ast.Call) or not (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "ImageFolder"
            ):
                continue
            root_node = next(
                (keyword.value for keyword in call.keywords if keyword.arg == "root"),
                call.args[0] if call.args else None,
            )
            root_value = _literal(root_node, None)
            if not isinstance(root_value, str):
                continue
            if any("train" in name.lower() for name in assigned_names):
                config["train_root"] = root_value
            elif any("test" in name.lower() for name in assigned_names):
                config["test_root"] = root_value

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function_name = (
                node.func.attr if isinstance(node.func, ast.Attribute) else ""
            )
            if function_name in {"SGD", "Adam"}:
                config["optimizer_name"] = function_name
                for keyword in node.keywords:
                    if keyword.arg == "lr":
                        value = _literal(keyword.value, config["lr"])
                        if isinstance(value, (int, float)):
                            config["lr"] = float(value)
                    elif keyword.arg == "momentum":
                        value = _literal(keyword.value, config["momentum"])
                        if isinstance(value, (int, float)):
                            config["momentum"] = float(value)
            elif function_name == "Resize" and node.args:
                resize = _tuple_values(node.args[0], config["resize"])
                if len(resize) == 2:
                    config["resize"] = (int(resize[0]), int(resize[1]))
            elif function_name == "Normalize" and len(node.args) >= 2:
                mean = _tuple_values(node.args[0], config["normalize_mean"])
                std = _tuple_values(node.args[1], config["normalize_std"])
                if len(mean) == 3:
                    config["normalize_mean"] = (mean[0], mean[1], mean[2])
                if len(std) == 3:
                    config["normalize_std"] = (std[0], std[1], std[2])
            elif function_name == "ImageFolder":
                root_node = next(
                    (keyword.value for keyword in node.keywords if keyword.arg == "root"),
                    node.args[0] if node.args else None,
                )
                root_value = _literal(root_node, None)
                if isinstance(root_value, str):
                    normalized_root = root_value.replace("\\", "/").lower()
                    if "train" in normalized_root:
                        config["train_root"] = root_value
                    elif "test" in normalized_root:
                        config["test_root"] = root_value
        elif isinstance(node, ast.For):
            if isinstance(node.target, ast.Name) and node.target.id == "epoch":
                call = node.iter
                if isinstance(call, ast.Call) and call.args:
                    value = _literal(call.args[0], config["epochs"])
                    if isinstance(value, int):
                        config["epochs"] = value

    return config


def build_transform(config: ScriptConfig) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(config["resize"]),
            transforms.ToTensor(),
            transforms.Normalize(
                config["normalize_mean"],
                config["normalize_std"],
            ),
        ]
    )


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def file_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def parse_result_log_settings(path: Path = RESULT_LOG_PATH) -> tuple[int, float] | None:
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    batches = re.findall(r"\bbatch_size\s*=\s*(\d+)", content)
    learning_rates = re.findall(r"\blr\s*=\s*([0-9]*\.?[0-9]+)", content)
    if not batches or not learning_rates:
        return None
    return int(batches[-1]), float(learning_rates[-1])


def parse_readme_experiments(path: Path = README_PATH) -> list[dict[str, str]]:
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    sections = re.split(r"(?m)^##\s+", content)
    expanded_sections = sections[:1]
    for section in sections[1:]:
        lines = section.splitlines()
        if not lines:
            expanded_sections.append(section)
            continue
        heading = lines[0].strip()
        section_body = "\n".join(lines[1:])
        noise_heading = re.search(r"(?im)^\s*gaussian\b[^\n]*:\s*$", section_body)
        blur_heading = re.search(r"(?im)^\s*blur\b[^\n]*:\s*$", section_body)
        if noise_heading and blur_heading and noise_heading.start() < blur_heading.start():
            common_settings = section_body[: noise_heading.start()]
            noise_body = section_body[noise_heading.end() : blur_heading.start()]
            blur_body = section_body[blur_heading.end() :]
            expanded_sections.append(
                f"{heading} · Gaussian Noise\n{common_settings}\n{noise_body}"
            )
            expanded_sections.append(
                f"{heading} · Gaussian Blur\n{common_settings}\n{blur_body}"
            )
        else:
            expanded_sections.append(section)
    sections = expanded_sections
    rows: list[dict[str, str]] = []
    for section in sections[1:]:
        lines = section.splitlines()
        if not lines:
            continue
        name = lines[0].strip().lstrip("# ")
        body = "\n".join(lines[1:])
        batch_match = re.search(r"\bbatch_size\s*=\s*(\d+)", body)
        lr_match = re.search(r"\blr\s*=\s*([0-9]*\.?[0-9]+)", body)
        epoch_match = re.search(r"\bepochs\s*=\s*(\d+)", body)
        settings = []
        if batch_match:
            settings.append(f"bs={batch_match.group(1)}")
        if lr_match:
            settings.append(f"lr={lr_match.group(1)}")
        if epoch_match:
            settings.append(f"{epoch_match.group(1)} epochs")
        augmentation = re.search(r"RandomRotation\s*\(([^)]*)\)", body)
        if augmentation:
            settings.append(f"RandomRotation({augmentation.group(1)})")
        kernel = re.search(r"\bkernel\s*=\s*([0-9]+)", body, flags=re.IGNORECASE)
        if kernel:
            settings.append(f"kernel={kernel.group(1)}")
        sigma = re.search(r"σ\s*=\s*([0-9]*\.?[0-9]+)", body)
        if sigma:
            settings.append(f"σ={sigma.group(1)}")
        noise_std = re.search(r"\bstd\s*=\s*([0-9]*\.?[0-9]+)", body)
        if noise_std:
            settings.append(f"std={noise_std.group(1)}")
        accuracies: list[tuple[str, float, str | None, str | None]] = []
        for match in re.finditer(
            r"Test accuracy(?:\s*\(([^)]*)\))?\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%?"
            r"(?:\s*\((\d+)\s*/\s*(\d+)\))?",
            body,
            flags=re.IGNORECASE,
        ):
            label = (match.group(1) or "").strip()
            accuracies.append((label, float(match.group(2)), match.group(3), match.group(4)))
        if not accuracies:
            continue
        accuracy_text = " · ".join(
            f"{f'{label.title()} ' if label else ''}{accuracy:.1f}%"
            + (f" ({correct}/{total})" if correct and total else "")
            for label, accuracy, correct, total in accuracies
        )
        error_text = " · ".join(
            f"{f'{label.title()} ' if label else ''}{100 - accuracy:.1f}%"
            for label, accuracy, _, _ in accuracies
        )
        rows.append(
            {
                "name": name,
                "settings": " · ".join(settings) or "—",
                "accuracy": accuracy_text,
                "error": error_text,
            }
        )
    return rows


def parse_training_curves(
    result_path: Path = RESULT_LOG_PATH,
    tune_path: Path = TUNE_LOG_PATH,
) -> list[dict[str, object]]:
    loss_pattern = re.compile(
        r"\[epoch\s+(\d+)(?:/\d+)?\]\s*loss:\s*([0-9]+(?:\.[0-9]+)?)",
        flags=re.IGNORECASE,
    )
    curves: list[dict[str, object]] = []
    try:
        result_text = result_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        result_text = ""
    result_losses = [
        (int(match.group(1)), float(match.group(2)))
        for match in loss_pattern.finditer(result_text)
    ]
    if result_losses:
        latest_run: list[tuple[int, float]] = []
        for epoch, loss in result_losses:
            if latest_run and epoch <= latest_run[-1][0]:
                latest_run = []
            latest_run.append((epoch, loss))
        log_settings = parse_result_log_settings(result_path)
        logged_batch, logged_lr = log_settings or (
            SCRIPT_CONFIG["batch_size"],
            SCRIPT_CONFIG["lr"],
        )
        curves.append(
            {
                "label": f"1016.py · bs={logged_batch} · lr={logged_lr:g}",
                "epochs": [epoch for epoch, _ in latest_run],
                "losses": [loss for _, loss in latest_run],
                "source": "result.log",
            }
        )

    try:
        tune_text = tune_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        tune_text = ""
    trials = []
    trial_pattern = re.compile(
        r"(?ms)^>>>\s*batch_size=(\d+)\s+lr=([^\s]+)(.*?)(?=^>>>|^====|\Z)"
    )
    for match in trial_pattern.finditer(tune_text):
        losses = [
            (int(loss_match.group(1)), float(loss_match.group(2)))
            for loss_match in loss_pattern.finditer(match.group(3))
        ]
        if not losses:
            continue
        accuracy_match = re.search(
            r"\bRESULT\s+batch_size=\d+\s+lr=[^\s]+\s+acc=([0-9]+(?:\.[0-9]+)?)%?",
            match.group(3),
            flags=re.IGNORECASE,
        )
        accuracy = float(accuracy_match.group(1)) if accuracy_match else -1.0
        trials.append(
            {
                "label": f"tune.py best · bs={match.group(1)} · lr={match.group(2)}",
                "epochs": [epoch for epoch, _ in losses],
                "losses": [loss for _, loss in losses],
                "source": "tune.log",
                "accuracy": accuracy,
            }
        )
    if trials:
        curves.append(max(trials, key=lambda curve: float(curve["accuracy"])))
    return curves


def training_curve_figure(curves: list[dict[str, object]]):
    import matplotlib.pyplot as plt

    fig, axis_loss = plt.subplots(figsize=(10, 4.2), facecolor="white")
    axis_delta = axis_loss.twinx()
    colors = ["#4f46e5", "#059669"]
    for index, curve in enumerate(curves):
        epochs = curve["epochs"]
        losses = curve["losses"]
        if not isinstance(epochs, list) or not isinstance(losses, list) or not losses:
            continue
        color = colors[index % len(colors)]
        axis_loss.plot(
            epochs,
            losses,
            color=color,
            marker="o",
            markersize=5,
            linewidth=2.2,
            label=f"Loss · {curve['label']}",
        )
        if len(losses) > 1:
            delta_percent = [
                100 * (current - previous) / previous if previous else 0.0
                for previous, current in zip(losses, losses[1:], strict=False)
            ]
            axis_delta.plot(
                epochs[1:],
                delta_percent,
                color=color,
                marker="x",
                markersize=5,
                linewidth=1.5,
                linestyle="--",
                alpha=0.72,
                label=f"Δ loss (%) · {curve['label']}",
            )
    axis_loss.set_xlabel("Epoch", color="#475569")
    axis_loss.set_ylabel("Training loss", color="#475569")
    axis_delta.set_ylabel("Epoch-to-epoch Δ loss (%)", color="#64748b")
    axis_loss.set_facecolor("white")
    axis_loss.grid(axis="y", color="#e2e8f0", linewidth=.8)
    axis_loss.spines["top"].set_visible(False)
    axis_delta.spines["top"].set_visible(False)
    axis_loss.tick_params(colors="#64748b")
    axis_delta.tick_params(colors="#64748b")
    handles_left, labels_left = axis_loss.get_legend_handles_labels()
    handles_right, labels_right = axis_delta.get_legend_handles_labels()
    axis_loss.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc="upper right",
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout()
    return fig


def count_dataset_images(root: Path) -> int:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sum(
        1
        for class_name in CLASSES
        for image_path in (root / class_name).glob("*")
        if image_path.is_file() and image_path.suffix.lower() in extensions
    )


def list_class_images(root: Path, class_name: str) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(
        path
        for path in (root / class_name).glob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )


def watch_project_files() -> None:
    paths = (
        SCRIPT_1016_PATH,
        CHECKPOINT_SOURCE_PATH,
        README_PATH,
        RESULT_LOG_PATH,
        TUNE_LOG_PATH,
    )
    signatures = []
    for path in paths:
        try:
            stat = path.stat()
            signatures.append((str(path), stat.st_mtime_ns, stat.st_size))
        except OSError:
            signatures.append((str(path), 0, 0))
    fingerprint = tuple(signatures)
    last_fingerprint = st.session_state.get("project_file_fingerprint")
    st.session_state["project_file_fingerprint"] = fingerprint
    if last_fingerprint is not None and last_fingerprint != fingerprint:
        st.rerun()


SCRIPT_CONFIG = parse_1016_config()
INFERENCE_TRANSFORM = build_transform(SCRIPT_CONFIG)
CHECKPOINT_SOURCE_PATH = resolve_project_path(SCRIPT_CONFIG["checkpoint_rel_path"])
CHECKPOINT_PATH = CHECKPOINT_SOURCE_PATH
TEST_PATH = resolve_project_path(SCRIPT_CONFIG["test_root"])
TRAIN_PATH = resolve_project_path(SCRIPT_CONFIG["train_root"])


class Net(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


@st.cache_resource
def load_model(checkpoint_path: str, checkpoint_mtime: float) -> tuple[Net, torch.device]:
    del checkpoint_mtime
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net(num_classes=len(CLASSES)).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    net.load_state_dict(state)
    net.eval()
    return net, device


def image_tensor(
    img: Image.Image,
    device: torch.device,
    transform: transforms.Compose = INFERENCE_TRANSFORM,
) -> torch.Tensor:
    return transform(img.convert("RGB")).unsqueeze(0).to(device)


def classify_image(
    img: Image.Image, net: Net, device: torch.device, include_features: bool = False
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, float]:
    tensor = image_tensor(img, device)
    started = time.perf_counter()
    with torch.inference_mode():
        logits = net(tensor)
        first = second = None
        if include_features:
            first = net.pool(F.relu(net.conv1(tensor)))
            second = net.pool(F.relu(net.conv2(first)))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return logits[0].detach().cpu(), first, second, elapsed_ms


@st.cache_data(show_spinner=False)
def evaluate_test_set(
    checkpoint_path: str,
    checkpoint_mtime: float,
    test_root: str,
    resize: tuple[int, int],
    normalize_mean: tuple[float, float, float],
    normalize_std: tuple[float, float, float],
) -> TestResults:
    del checkpoint_mtime
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Net(num_classes=len(CLASSES)).to(device)
    net.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    net.eval()
    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std),
        ]
    )
    counts = np.zeros((2, 2), dtype=np.int64)
    total = 0
    with torch.inference_mode():
        for class_index, class_name in enumerate(CLASSES):
            image_paths = list_class_images(Path(test_root), class_name)
            for start in range(0, len(image_paths), 16):
                batch = [
                    image_tensor(Image.open(path), device, transform)
                    for path in image_paths[start : start + 16]
                ]
                if not batch:
                    continue
                predictions = net(torch.cat(batch)).argmax(dim=1).cpu().numpy()
                for prediction in predictions:
                    counts[class_index, int(prediction)] += 1
                total += len(predictions)
    correct = int(counts.trace())
    return {
        "counts": counts.tolist(),
        "total": total,
        "correct": correct,
        "accuracy": 100 * correct / total if total else 0.0,
        "error": 100 * (total - correct) / total if total else 0.0,
    }


def perturb_image(
    image: Image.Image, mode: str, rotation: float, noise_std: float, blur_sigma: float
) -> Image.Image:
    result = image.convert("RGB")
    if mode == "회전 변환 (Rotation)":
        result = result.rotate(rotation, resample=Image.Resampling.BICUBIC)
    elif mode == "가우시안 노이즈 (Gaussian Noise)":
        pixels = np.asarray(result, dtype=np.float32) / 255.0
        noise = np.random.default_rng(17).normal(0, noise_std, pixels.shape)
        result = Image.fromarray((np.clip(pixels + noise, 0, 1) * 255).astype(np.uint8))
    elif mode == "가우시안 블러 (Gaussian Blur)":
        result = result.filter(ImageFilter.GaussianBlur(radius=blur_sigma))
    return result


def choose_random_sample(file_names: list[str]) -> None:
    st.session_state["sample_select"] = random.choice(file_names)


def feature_figure(features: torch.Tensor, title: str):
    import matplotlib.pyplot as plt

    channels = features[0].detach().cpu().numpy()
    columns = 4
    rows = int(np.ceil(len(channels) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(8, rows * 1.7),
        facecolor="white",
    )
    for index, axis in enumerate(np.asarray(axes).reshape(-1)):
        axis.axis("off")
        if index < len(channels):
            channel = channels[index]
            low, high = float(channel.min()), float(channel.max())
            shown = (channel - low) / (high - low) if high > low else channel
            axis.imshow(shown, cmap="viridis", interpolation="nearest")
            axis.set_title(
                f"CHANNEL {index + 1:02d}",
                fontsize=9,
                color="#475569",
                fontfamily="sans-serif",
            )
    fig.suptitle(
        title,
        fontsize=12,
        fontweight="bold",
        color="#0f172a",
        fontfamily="sans-serif",
    )
    fig.tight_layout()
    return fig


st.set_page_config(
    page_title="Chihuahua vs. Muffin — CNN 이미지 판별 스튜디오",
    page_icon="🐶",
    layout="wide",
)

@st.fragment(run_every="10s")
def _poll_training_files() -> None:
    watch_project_files()


_poll_training_files()

checkpoint_mtime = file_mtime(CHECKPOINT_SOURCE_PATH)
script_mtime = file_mtime(SCRIPT_1016_PATH)
model, device = load_model(str(CHECKPOINT_SOURCE_PATH), checkpoint_mtime)
device_label = "CUDA" if device.type == "cuda" else "CPU"
readme_experiments = parse_readme_experiments()
training_curves = parse_training_curves()
result_log_settings = parse_result_log_settings()
train_count = count_dataset_images(TRAIN_PATH)
test_root_count = count_dataset_images(TEST_PATH)
config_summary = (
    f"bs={SCRIPT_CONFIG['batch_size']} · lr={SCRIPT_CONFIG['lr']:g} · "
    f"epochs={SCRIPT_CONFIG['epochs']} · {SCRIPT_CONFIG['optimizer_name']}"
    f"(momentum={SCRIPT_CONFIG['momentum']:g})"
)
evaluation_error = ""
try:
    with st.spinner("현재 체크포인트의 테스트 정확도를 확인하고 있습니다…"):
        active_metrics = evaluate_test_set(
            str(CHECKPOINT_SOURCE_PATH),
            checkpoint_mtime,
            str(TEST_PATH),
            SCRIPT_CONFIG["resize"],
            SCRIPT_CONFIG["normalize_mean"],
            SCRIPT_CONFIG["normalize_std"],
        )
except (OSError, RuntimeError, ValueError) as error:
    active_metrics = None
    evaluation_error = str(error)

if active_metrics is None:
    active_accuracy = 0.0
    active_error_rate = 0.0
    active_correct = 0
    active_total = test_root_count
else:
    active_accuracy = float(active_metrics["accuracy"])
    active_error_rate = float(active_metrics["error"])
    active_correct = int(active_metrics["correct"])
    active_total = int(active_metrics["total"])

tune_row = next(
    (row for row in readme_experiments if "tune.py" in row["name"].lower()),
    None,
)
tune_match = (
    re.search(r"([0-9]+(?:\.[0-9]+)?)%", tune_row["accuracy"])
    if tune_row is not None
    else None
)
tune_accuracy = float(tune_match.group(1)) if tune_match else active_accuracy
tune_settings = tune_row["settings"] if tune_row is not None else "README 실험 기록 없음"
accuracy_display = f"{active_accuracy:.1f}%" if active_metrics else "—"
error_display = f"{active_error_rate:.1f}%" if active_metrics else "—"
correct_note = (
    f"{active_correct:,} / {active_total:,} 정분류 · 현재 체크포인트"
    if active_metrics
    else "테스트셋 평가를 완료하면 표시됩니다"
)
error_note = (
    f"{active_total - active_correct:,} / {active_total:,} 오분류 · 실시간 평가"
    if active_metrics
    else "현재 체크포인트 평가 대기"
)
resize_label = f"{SCRIPT_CONFIG['resize'][0]}×{SCRIPT_CONFIG['resize'][1]}"
script_time_label = (
    datetime.fromtimestamp(script_mtime).strftime("%Y-%m-%d %H:%M:%S")
    if script_mtime
    else "파일 없음"
)
checkpoint_time_label = (
    datetime.fromtimestamp(checkpoint_mtime).strftime("%Y-%m-%d %H:%M:%S")
    if checkpoint_mtime
    else "파일 없음"
)
config_changed = result_log_settings is not None and (
    SCRIPT_CONFIG["batch_size"] != result_log_settings[0]
    or abs(SCRIPT_CONFIG["lr"] - result_log_settings[1]) > 1e-12
)
checkpoint_name = escape(CHECKPOINT_SOURCE_PATH.name)
checkpoint_path_label = escape(SCRIPT_CONFIG["checkpoint_rel_path"])
sync_chips = "".join(
    f'<span class="sync-chip">{chip}</span>'
    for chip in (
        f"batch_size = {SCRIPT_CONFIG['batch_size']}",
        f"lr = {SCRIPT_CONFIG['lr']:g}",
        f"momentum = {SCRIPT_CONFIG['momentum']:g}",
        f"epochs = {SCRIPT_CONFIG['epochs']}",
        f"Resize = {resize_label}",
        f"PATH = {checkpoint_path_label}",
        f"{checkpoint_name} · {checkpoint_time_label}",
    )
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');
    :root {
      color-scheme: light;
      --slate-950:#020617; --slate-900:#0f172a; --slate-800:#1e293b;
      --slate-700:#334155; --slate-600:#475569; --slate-500:#64748b;
      --slate-400:#94a3b8; --slate-300:#cbd5e1; --slate-200:#e2e8f0;
      --slate-100:#f1f5f9; --slate-50:#f8fafc; --indigo:#4f46e5;
      --amber:#d97706; --violet:#4338ca; --emerald:#059669; --rose:#e11d48;
    }
    .stApp {
      background: radial-gradient(1200px circle at 50% -10%, #e0e7ff 0%, transparent 55%), #f8fafc;
      color:var(--slate-900);
      font-family:'Pretendard','Inter',-apple-system,BlinkMacSystemFont,sans-serif;
    }
    html, body, [class*="css"] {
      font-family:'Pretendard','Inter',-apple-system,BlinkMacSystemFont,sans-serif;
      color:var(--slate-900);
    }
    .block-container { max-width:1480px; padding-top:2rem; padding-bottom:3rem; }
    h1,h2,h3,h4,p,label { font-family:'Pretendard','Inter',-apple-system,BlinkMacSystemFont,sans-serif !important; }
    h1,h2,h3 { color:var(--slate-950) !important; letter-spacing:-.035em; }
    [data-testid="stVerticalBlockBorderWrapper"] {
      background:#ffffff;
      border:1px solid #e2e8f0;
      border-radius:16px;
      padding:1.25rem;
      box-shadow:0 1px 3px 0 rgba(15,23,42,.04);
    }
    [data-testid="stVerticalBlockBorderWrapper"] > div { gap:.75rem; }
    .studio-header { padding:.2rem 0 1.15rem; }
    .studio-pill {
      display:inline-flex; align-items:center; gap:.48rem; padding:.38rem .7rem;
      border:1px solid #c7d2fe; border-radius:999px; background:#eef2ff;
      color:#4338ca; font:600 .65rem 'JetBrains Mono',monospace; letter-spacing:.055em;
    }
    .studio-dot { width:7px;height:7px;border-radius:50%;background:#4f46e5;box-shadow:0 0 0 3px #c7d2fe; }
    .studio-title { margin:.8rem 0 .28rem; font-size:clamp(1.8rem,3.5vw,2.55rem); line-height:1.13; font-weight:800; color:#0f172a; letter-spacing:-.055em; }
    .studio-subtitle { margin:0; color:#64748b; font-size:.93rem; line-height:1.55; }
    .kpi-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:1rem; margin:.15rem 0 1.2rem; }
    .kpi-card {
      min-height:135px; padding:1.05rem 1.15rem; position:relative; overflow:hidden;
      border:1px solid #e2e8f0; border-radius:16px; background:#fff;
      box-shadow:0 1px 3px rgba(15,23,42,.04);
    }
    .kpi-card:before { content:''; position:absolute; inset:0 auto 0 0; width:3px; background:var(--accent); }
    .kpi-label { display:flex; align-items:center; gap:.5rem; color:#64748b; font-size:.72rem; font-weight:700; letter-spacing:.015em; }
    .kpi-icon { width:25px;height:25px;display:grid;place-items:center;border-radius:8px;background:var(--tint);font-size:.83rem; }
    .kpi-value { margin:.7rem 0 .2rem; color:var(--accent); font:700 1.8rem 'JetBrains Mono',monospace; letter-spacing:-.065em; }
    .kpi-note { color:#64748b; font-size:.69rem; line-height:1.45; }
    .connection-bar { display:flex;align-items:center;flex-wrap:wrap;gap:.48rem;margin:-.35rem 0 1rem;padding:.72rem .85rem;border:1px solid #dbeafe;border-radius:13px;background:rgba(255,255,255,.84);box-shadow:0 1px 2px rgba(15,23,42,.03); }
    .connection-status { display:inline-flex;align-items:center;gap:.42rem;margin-right:.25rem;color:#047857;font-size:.69rem;font-weight:750;white-space:nowrap; }
    .connection-status i { width:7px;height:7px;border-radius:50%;background:#10b981;box-shadow:0 0 0 3px #d1fae5; }
    .sync-chip { padding:.32rem .5rem;border:1px solid #e2e8f0;border-radius:8px;background:#f8fafc;color:#475569;font:500 .61rem 'JetBrains Mono',monospace;white-space:nowrap; }
    .sync-alert { display:inline-flex;align-items:center;gap:.35rem;padding:.33rem .52rem;border:1px solid #fed7aa;border-radius:8px;background:#fff7ed;color:#c2410c;font-size:.65rem;font-weight:650; }
    .workspace { margin-top:.2rem; }
    .step-title { display:flex; align-items:center; gap:.65rem; margin:0 0 .65rem; color:#0f172a; font-size:1rem; font-weight:750; letter-spacing:-.02em; }
    .step-index { display:grid; place-items:center; width:27px;height:27px; flex:none; border-radius:9px; background:#eef2ff; color:#4f46e5; font:600 .72rem 'JetBrains Mono',monospace; }
    .step-subtitle { margin:-.35rem 0 .7rem 2.15rem; color:#64748b; font-size:.72rem; line-height:1.45; }
    .input-hint { margin:0;color:#64748b;font-size:.7rem;line-height:1.45; }
    [data-testid="stFileUploader"] {
      padding:.55rem; border:1px dashed #a5b4fc; border-radius:13px; background:#f8faff;
    }
    [data-testid="stFileUploader"] section { padding:.55rem; }
    [data-testid="stFileUploader"] small { color:#64748b; }
    [data-testid="stImage"] img { border:1px solid #e2e8f0; border-radius:12px; }
    [data-testid="stImage"] p { color:#475569 !important;font-size:.72rem !important;font-weight:600; }
    .image-meta { padding:.6rem .72rem; border:1px solid #e2e8f0; border-radius:10px; background:#f8fafc; color:#64748b; font-size:.67rem; line-height:1.5; }
    .demo-note { padding:.58rem .72rem; border:1px solid #c7d2fe; border-radius:10px; background:#eef2ff; color:#4338ca; font-size:.7rem; line-height:1.45; }
    .hero-verdict {
      position:relative; overflow:hidden; padding:1.2rem 1.25rem 1.05rem;
      border:1px solid #c7d2fe; border-top:3px solid #4f46e5; border-radius:15px;
      background:linear-gradient(130deg,#eef2ff 0%,#fff 56%,#f8fafc 100%);
    }
    .hero-verdict:after { content:'';position:absolute;right:-38px;top:-65px;width:150px;height:150px;border:1px solid #c7d2fe66;border-radius:50%;box-shadow:0 0 0 18px #c7d2fe24,0 0 0 38px #c7d2fe12;pointer-events:none; }
    .verdict-kicker { position:relative;z-index:1;color:#64748b;font:600 .64rem 'JetBrains Mono',monospace;letter-spacing:.08em;text-transform:uppercase; }
    .verdict-main { position:relative;z-index:1;display:flex;align-items:center;gap:.55rem;flex-wrap:wrap;margin:.62rem 0 .15rem;color:#0f172a;font-size:1.35rem;font-weight:800;letter-spacing:-.04em; }
    .verdict-badge { display:inline-flex;align-items:center;gap:.4rem;padding:.35rem .68rem;border-radius:999px;font-size:.8rem;font-weight:750;letter-spacing:0; }
    .badge-dog { color:#92400e;background:#fef3c7;border:1px solid #fde68a; }
    .badge-muffin { color:#3730a3;background:#e0e7ff;border:1px solid #c7d2fe; }
    .verdict-english { color:#64748b;font-size:.77rem;font-weight:600;letter-spacing:0; }
    .verdict-confidence { position:relative;z-index:1;margin:.35rem 0 .95rem;color:#0f172a;font:700 clamp(2.25rem,4vw,3rem) 'JetBrains Mono',monospace;letter-spacing:-.075em;line-height:1; }
    .verdict-confidence small { margin-left:.15rem;color:#64748b;font:600 .8rem 'Inter',sans-serif;letter-spacing:0; }
    .verdict-stats { display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.6rem;position:relative;z-index:1; }
    .verdict-stat { padding:.58rem .65rem;border:1px solid #e2e8f0;border-radius:10px;background:#ffffffd9; }
    .verdict-stat-label { color:#64748b;font-size:.62rem;line-height:1.3; }
    .verdict-stat-value { margin-top:.22rem;color:#0f172a;font:650 .85rem 'JetBrains Mono',monospace; }
    .result-heading { display:flex;align-items:center;justify-content:space-between;gap:.5rem;margin:.95rem 0 .55rem; }
    .result-heading strong { color:#0f172a;font-size:.79rem; }
    .result-heading span { color:#94a3b8;font:500 .6rem 'JetBrains Mono',monospace; }
    .prob-card { margin:.48rem 0;padding:.7rem .78rem;border:1px solid #e2e8f0;border-radius:12px;background:#fff; }
    .prob-head { display:flex;align-items:center;justify-content:space-between;gap:.5rem;margin-bottom:.55rem; }
    .prob-name { display:flex;align-items:center;gap:.48rem;color:#1e293b;font-size:.76rem;font-weight:700; }
    .prob-icon { width:26px;height:26px;display:grid;place-items:center;border-radius:8px;font-size:.82rem; }
    .prob-dog-icon { background:#fffbeb; }.prob-muffin-icon { background:#eef2ff; }
    .prob-right { text-align:right;white-space:nowrap; }
    .prob-value { color:#0f172a;font:650 .78rem 'JetBrains Mono',monospace; }
    .prob-logit { margin-top:.12rem;color:#64748b;font:500 .59rem 'JetBrains Mono',monospace; }
    .progress-track { height:11px;overflow:hidden;border-radius:999px;background:#f1f5f9; }
    .progress-fill { height:100%;border-radius:999px;transition:width .55s cubic-bezier(.2,.7,.2,1); }
    .progress-dog { background:linear-gradient(90deg,#f59e0b,#d97706); }
    .progress-muffin { background:linear-gradient(90deg,#818cf8,#4338ca); }
    .delta-card { display:grid;grid-template-columns:1fr auto 1fr auto;align-items:center;gap:.55rem;margin-top:.65rem;padding:.72rem .8rem;border:1px solid #cbd5e1;border-radius:12px;background:#f8fafc;color:#1e293b; }
    .delta-item span { display:block;color:#64748b;font-size:.6rem;line-height:1.35; }
    .delta-item strong { display:block;margin-top:.18rem;color:#0f172a;font-size:.72rem; }
    .delta-arrow { color:#94a3b8;font-size:.9rem; }
    .delta-change { text-align:right;color:#4f46e5;font:650 .68rem 'JetBrains Mono',monospace;white-space:nowrap; }
    .inspector { margin-top:1.15rem; }
    [data-testid="stTabs"] [role="tablist"] { gap:.5rem;border-bottom:1px solid #e2e8f0; }
    [data-testid="stTabs"] button[role="tab"] { height:2.8rem;padding:0 .9rem;color:#64748b;font-size:.77rem;font-weight:650; }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color:#4338ca;border-bottom-color:#4f46e5; }
    [data-testid="stExpander"] { border:1px solid #e2e8f0 !important;border-radius:12px !important;background:#fff !important; }
    [data-testid="stButton"] button { border-radius:10px;border:1px solid #c7d2fe;background:#eef2ff;color:#4338ca;font-weight:650; }
    [data-testid="stButton"] button:hover { border-color:#818cf8;background:#e0e7ff;color:#3730a3; }
    [data-testid="stButton"] button[kind="primary"] { border-color:#4f46e5;background:#4f46e5;color:#fff; }
    [data-testid="stButton"] button[kind="primary"]:hover { background:#4338ca; }
    [data-testid="stRadio"] label p,[data-testid="stSelectbox"] label p,[data-testid="stSlider"] label p { color:#334155;font-size:.76rem;font-weight:650; }
    [data-testid="stRadio"] [role="radiogroup"] { gap:.45rem; }
    [data-testid="stRadio"] label[data-baseweb="radio"] { padding:.25rem .5rem;border:1px solid #e2e8f0;border-radius:9px;background:#fff; }
    .bench-table { width:100%;border-collapse:separate;border-spacing:0;overflow:hidden;border:1px solid #e2e8f0;border-radius:12px;background:#fff;font-size:.75rem; }
    .bench-table th { padding:.75rem .8rem;background:#f8fafc;color:#64748b;text-align:left;font-size:.64rem;font-weight:700;letter-spacing:.045em;border-bottom:1px solid #e2e8f0; }
    .bench-table td { padding:.78rem .8rem;color:#334155;border-bottom:1px solid #f1f5f9; }
    .bench-table tr:last-child td { border-bottom:0; }
    .bench-table tr.current td:first-child { color:#4338ca;font-weight:700; }
    .bench-table .mono { color:#475569;font:500 .67rem 'JetBrains Mono',monospace; }
    .bench-table .good { color:#059669;font-weight:750; }
    .bench-table .bad { color:#e11d48;font-weight:700; }
    .tab-intro { margin:.35rem 0 1rem;color:#64748b;font-size:.76rem;line-height:1.5; }
    .feature-title { margin:.8rem 0 .2rem;color:#0f172a;font-size:.88rem;font-weight:750; }
    .footer-note { margin-top:1.5rem;padding-top:.8rem;border-top:1px solid #e2e8f0;color:#94a3b8;font:500 .62rem 'JetBrains Mono',monospace;letter-spacing:.015em; }
    @media(max-width:980px) { .kpi-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } }
    @media(max-width:620px) {
      .block-container { padding-top:1.2rem; }
      .kpi-grid { grid-template-columns:1fr 1fr;gap:.55rem; }
      .kpi-card { min-height:120px;padding:.8rem; }
      .kpi-value { font-size:1.35rem; }
      .delta-card { grid-template-columns:1fr auto 1fr; }
      .delta-change { grid-column:1/-1;text-align:left; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <header class="studio-header">
      <div class="studio-pill"><span class="studio-dot"></span>PYTORCH 2.14 · CNN BINARY CLASSIFIER</div>
      <h1 class="studio-title">Chihuahua vs. Muffin — CNN 이미지 판별 스튜디오</h1>
      <p class="studio-subtitle">사진을 넣고 두 클래스의 예측 확률을 확인하세요. 회전과 노이즈 실험으로 입력 변화에 대한 모델의 반응도 비교할 수 있습니다.</p>
    </header>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="connection-bar">
      <span class="connection-status"><i></i>🔗 1016.py 실시간 연동됨</span>
      <span class="sync-chip">1016.py · {script_time_label}</span>
      {sync_chips}
      {f'<span class="sync-alert">⚡ 1016.py 수치 변경 감지됨 · 현재 코드 bs={SCRIPT_CONFIG["batch_size"]}, lr={SCRIPT_CONFIG["lr"]:g}</span>' if config_changed else ''}
    </div>
    """,
    unsafe_allow_html=True,
)

if evaluation_error:
    st.warning(
        "실시간 테스트셋 평가에 실패했습니다. 파일 설정과 체크포인트를 확인해 주세요. "
        f"상세: {evaluation_error}"
    )

st.markdown(
    f"""
    <section class="kpi-grid">
      <article class="kpi-card" style="--accent:#059669;--tint:#d1fae5">
        <div class="kpi-label"><span class="kpi-icon">✓</span>테스트 정확도 · ACCURACY</div>
        <div class="kpi-value">{accuracy_display}</div>
        <div class="kpi-note">{correct_note}</div>
      </article>
      <article class="kpi-card" style="--accent:#e11d48;--tint:#ffe4e6">
        <div class="kpi-label"><span class="kpi-icon">↗</span>테스트 오차율 · ERROR RATE</div>
        <div class="kpi-value">{error_display}</div>
        <div class="kpi-note">{error_note}</div>
      </article>
      <article class="kpi-card" style="--accent:#4f46e5;--tint:#e0e7ff">
        <div class="kpi-label"><span class="kpi-icon">⌁</span>튜닝 최고 성능 · TUNE.PY</div>
        <div class="kpi-value">{tune_accuracy:.1f}%</div>
        <div class="kpi-note">README 기록 · 오차율 {100 - tune_accuracy:.1f}% · {escape(tune_settings)}</div>
      </article>
      <article class="kpi-card" style="--accent:#475569;--tint:#e2e8f0">
        <div class="kpi-label"><span class="kpi-icon">⌘</span>실행 환경 &amp; 1016.py 설정</div>
        <div class="kpi-value">{device_label}</div>
        <div class="kpi-note">Train {train_count:,}장 · Test {active_total:,}장 · {resize_label}<br>{escape(config_summary)}</div>
      </article>
    </section>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([0.97, 1.03], gap="large")
original: Image.Image | None = None
transformed: Image.Image | None = None
feature_one: torch.Tensor | None = None
feature_two: torch.Tensor | None = None
source_name = ""
source_mode = "RGB"
is_default_sample = False

with left, st.container(border=True):
    st.markdown(
        '<div class="step-title"><span class="step-index">01</span>이미지 입력 &amp; 변형 실험</div>'
        '<p class="step-subtitle">사진을 올리거나 테스트셋에서 샘플을 골라 입력을 구성합니다.</p>',
        unsafe_allow_html=True,
    )
    input_mode = st.radio(
        "입력 방식",
        ["📁 사진 파일 업로드", "🖼️ 테스트셋 샘플 선택"],
        horizontal=True,
        label_visibility="collapsed",
    )
    selected_sample: Path | None = None
    if input_mode == "📁 사진 파일 업로드":
        uploaded = st.file_uploader(
            "사진 파일을 여기에 끌어다 놓거나 클릭해 업로드하세요",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            label_visibility="collapsed",
        )
        if uploaded is not None:
            decoded = Image.open(io.BytesIO(uploaded.getvalue()))
            source_mode = decoded.mode
            original = decoded.convert("RGB")
            source_name = uploaded.name
        else:
            default_samples = list_class_images(TEST_PATH, "chihuahua")
            if default_samples:
                selected_sample = default_samples[0]
                is_default_sample = True
                st.markdown(
                    '<div class="demo-note">파일을 드래그해 올리면 즉시 교체됩니다. 현재 기본 데모 샘플로 결과를 표시 중입니다.</div>',
                    unsafe_allow_html=True,
                )
    else:
        sample_class = st.selectbox(
            "샘플 클래스",
            CLASSES,
            format_func=lambda value: "🐶 치와와" if value == "chihuahua" else "🧁 머핀",
            label_visibility="collapsed",
        )
        sample_files = list_class_images(TEST_PATH, sample_class)
        if sample_files:
            file_names = [path.name for path in sample_files]
            if st.session_state.get("sample_select") not in file_names:
                st.session_state["sample_select"] = file_names[0]
            sample_row, random_row = st.columns([4, 1])
            with sample_row:
                selected_name = st.selectbox(
                    "테스트 이미지",
                    file_names,
                    key="sample_select",
                    label_visibility="collapsed",
                )
            with random_row:
                st.button(
                    "랜덤",
                    width="stretch",
                    on_click=choose_random_sample,
                    args=(file_names,),
                )
            selected_sample = TEST_PATH / sample_class / selected_name
        else:
            st.warning(f"테스트 샘플을 찾을 수 없습니다: {TEST_PATH / sample_class}")

    if selected_sample is not None:
        decoded = Image.open(selected_sample)
        source_mode = decoded.mode
        original = decoded.convert("RGB")
        source_name = selected_sample.name

    st.markdown('<div class="result-heading"><strong>입력 변형</strong><span>ROBUSTNESS TEST</span></div>', unsafe_allow_html=True)
    perturbation = st.selectbox(
        "실험 옵션",
        [
            "원본 (Clean)",
            "회전 변환 (Rotation)",
            "가우시안 노이즈 (Gaussian Noise)",
            "가우시안 블러 (Gaussian Blur)",
        ],
        label_visibility="collapsed",
    )
    rotation = noise_std = blur_sigma = 0.0
    if perturbation == "회전 변환 (Rotation)":
        rotation = st.slider("회전 각도", -45, 45, 15, format="%d°")
    elif perturbation == "가우시안 노이즈 (Gaussian Noise)":
        noise_std = st.slider("노이즈 표준편차", 0.01, 0.30, 0.10, 0.01)
    elif perturbation == "가우시안 블러 (Gaussian Blur)":
        blur_sigma = st.slider("블러 강도 (σ)", 0.5, 3.0, 1.0, 0.1)

    if original is not None:
        transformed = perturb_image(original, perturbation, rotation, noise_std, blur_sigma)
        preview_height, preview_width = SCRIPT_CONFIG["resize"]
        st.markdown(
            f'<div class="result-heading"><strong>미리보기</strong><span>RGB → {resize_label}</span></div>',
            unsafe_allow_html=True,
        )
        preview_original, preview_transformed = st.columns(2, gap="small")
        with preview_original:
            st.image(
                original,
                caption=f"원본 이미지 · {original.width}×{original.height}",
                width="stretch",
            )
        with preview_transformed:
            st.image(
                transformed.resize((preview_width, preview_height)),
                caption=f"CNN 입력 · {resize_label}",
                width="stretch",
            )
        if is_default_sample:
            st.markdown(
                f'<div class="image-meta">기본 샘플 · {escape(source_name)} · 원본 모드 {source_mode} · 치와와 테스트셋</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="image-meta">{escape(source_name)} · 원본 모드 {source_mode} · {original.width}×{original.height} · {perturbation}</div>',
                unsafe_allow_html=True,
            )

with right, st.container(border=True):
    st.markdown(
        '<div class="step-title"><span class="step-index">02</span>CNN 판별 결과 &amp; 클래스 확률</div>'
        '<p class="step-subtitle">모델의 예측과 두 클래스의 상대 확률을 확인합니다.</p>',
        unsafe_allow_html=True,
    )
    if original is not None and transformed is not None:
        with st.spinner("CNN이 이미지를 분석하고 있습니다…"):
            logits, feature_one, feature_two, inference_ms = classify_image(
                transformed,
                model,
                device,
                include_features=True,
            )
            probabilities = torch.softmax(logits.unsqueeze(0), dim=1)[0]
            prob_chihuahua = probabilities[0].item() * 100
            prob_muffin = probabilities[1].item() * 100
            original_probabilities = None
            if perturbation != "원본 (Clean)":
                original_logits, _, _, _ = classify_image(original, model, device)
                original_probabilities = torch.softmax(
                    original_logits.unsqueeze(0), dim=1
                )[0]

        predicted_index = int(probabilities.argmax().item())
        sorted_probs = torch.sort(probabilities, descending=True).values
        confidence = float(sorted_probs[0].item() * 100)
        margin = float((sorted_probs[0] - sorted_probs[1]).item() * 100)
        predicted_korean = CLASS_KOREAN[predicted_index]
        predicted_class = CLASSES[predicted_index].title()
        predicted_icon = "🐶" if predicted_index == 0 else "🧁"
        badge_class = "badge-dog" if predicted_index == 0 else "badge-muffin"
        st.markdown(
            f"""
            <section class="hero-verdict">
              <div class="verdict-kicker">MODEL VERDICT · CHECKPOINT INFERENCE</div>
              <div class="verdict-main"><span class="verdict-badge {badge_class}">{predicted_icon} {predicted_korean}</span><span class="verdict-english">({predicted_class})</span></div>
              <div class="verdict-confidence">{confidence:.2f}<small>% 확률</small></div>
              <div class="verdict-stats">
                <div class="verdict-stat"><div class="verdict-stat-label">신뢰도 · CONFIDENCE</div><div class="verdict-stat-value">{confidence:.2f}%</div></div>
                <div class="verdict-stat"><div class="verdict-stat-label">확률 격차 · MARGIN</div><div class="verdict-stat-value">{margin:.2f}%p</div></div>
                <div class="verdict-stat"><div class="verdict-stat-label">추론 속도 · LATENCY</div><div class="verdict-stat-value">{inference_ms:.1f} ms</div></div>
              </div>
            </section>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="result-heading"><strong>클래스별 추정 확률</strong><span>SOFTMAX PROBABILITY</span></div>',
            unsafe_allow_html=True,
        )
        class_configs = [
            (0, "치와와", "Chihuahua", "🐶", prob_chihuahua),
            (1, "머핀", "Muffin", "🧁", prob_muffin),
        ]
        for index, korean, english, icon, probability in class_configs:
            icon_class = "prob-dog-icon" if index == 0 else "prob-muffin-icon"
            fill_class = "progress-dog" if index == 0 else "progress-muffin"
            width = max(0.0, min(100.0, probability))
            st.markdown(
                f"""
                <article class="prob-card">
                  <div class="prob-head">
                    <div class="prob-name"><span class="prob-icon {icon_class}">{icon}</span>{korean} <span style="color:#64748b;font-weight:500">({english})</span></div>
                    <div class="prob-right"><div class="prob-value">{probability:.2f}%</div><div class="prob-logit">logit: {float(logits[index].item()):+.4f}</div></div>
                  </div>
                  <div class="progress-track"><div class="progress-fill {fill_class}" style="width:{width:.3f}%"></div></div>
                </article>
                """,
                unsafe_allow_html=True,
            )

        if original_probabilities is not None:
            original_index = int(original_probabilities.argmax().item())
            original_confidence = float(original_probabilities[original_index].item() * 100)
            same_class_original = float(original_probabilities[predicted_index].item() * 100)
            delta = confidence - same_class_original
            original_name = CLASS_KOREAN[original_index]
            delta_text = f"Δ {delta:+.2f}%p (최종 클래스 기준)"
            st.markdown(
                f"""
                <div class="delta-card">
                  <div class="delta-item"><span>원본 판별</span><strong>{original_name} · {original_confidence:.2f}%</strong></div>
                  <div class="delta-arrow">→</div>
                  <div class="delta-item"><span>변환 후 판별</span><strong>{predicted_korean} · {confidence:.2f}%</strong></div>
                  <div class="delta-change">{delta_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("이미지를 불러오면 판별 결과가 여기에 표시됩니다.")

tab_bench, tab_loss, tab_feat = st.tabs(
    [
        "📊 실험별 정확도·오차율 비교 및 전체 테스트셋 검증",
        "📈 학습 곡선 · epoch별 loss 변화",
        "🔬 CNN 합성곱(Conv1·Conv2) 특징 맵 시각화",
    ]
)

with tab_bench:
    st.markdown(
        '<div class="step-title"><span class="step-index">A</span>실험별 정확도 · 오차율</div>'
        '<p class="tab-intro">세팅별 테스트 성능을 비교하고, 현재 체크포인트를 테스트셋 전체에 다시 실행할 수 있습니다.</p>',
        unsafe_allow_html=True,
    )
    live_accuracy = (
        f"{active_accuracy:.1f}% · {active_correct:,}/{active_total:,}"
        if active_metrics
        else "평가 대기"
    )
    live_error = f"{active_error_rate:.1f}%" if active_metrics else "—"
    live_row = (
        '<tr class="current"><td>현재 1016.py 설정 · '
        f"{escape(CHECKPOINT_SOURCE_PATH.name)}</td>"
        f'<td class="mono">{escape(config_summary)}</td>'
        f'<td class="good">{live_accuracy}</td>'
        f'<td class="bad">{live_error}</td></tr>'
    )
    readme_rows_html = "".join(
        "<tr>"
        f"<td>{escape(row['name'])}</td>"
        f'<td class="mono">{escape(row["settings"])}</td>'
        f"<td>{escape(row['accuracy'])}</td>"
        f"<td>{escape(row['error'])}</td>"
        "</tr>"
        for row in readme_experiments
    )
    st.markdown(
        f"""
        <table class="bench-table">
          <thead><tr><th>실험</th><th>학습 설정</th><th>정확도</th><th>오차율</th></tr></thead>
          <tbody>{live_row}{readme_rows_html}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)
    with st.expander(f"실제 테스트셋 {test_root_count:,}장 실시간 검증", expanded=False):
        st.markdown(
            '<p class="tab-intro">모델이 테스트 폴더의 모든 이미지를 다시 분류합니다. 체크포인트가 바뀔 때까지 결과를 캐시합니다.</p>',
            unsafe_allow_html=True,
        )
        if st.button("전체 테스트셋 검증 실행", type="primary", key="run_test"):
            try:
                checkpoint_stamp = file_mtime(CHECKPOINT_SOURCE_PATH)
                with st.spinner(f"테스트 이미지 {test_root_count:,}장을 평가하고 있습니다…"):
                    result = evaluate_test_set(
                        str(CHECKPOINT_SOURCE_PATH),
                        checkpoint_stamp,
                        str(TEST_PATH),
                        SCRIPT_CONFIG["resize"],
                        SCRIPT_CONFIG["normalize_mean"],
                        SCRIPT_CONFIG["normalize_std"],
                    )
                counts = np.asarray(result["counts"], dtype=np.int64)
                correct = int(result["correct"])
                total = int(result["total"])
                accuracy = float(result["accuracy"])
                error_rate = float(result["error"])
                st.success(
                    f"검증 완료 · 정확도 {accuracy:.1f}% · 오차율 {error_rate:.1f}% · "
                    f"정분류 {correct:,}장 / 오분류 {total - correct:,}장"
                )
                dog_total = int(counts[0].sum())
                muffin_total = int(counts[1].sum())
                dog_acc = 100 * counts[0, 0] / dog_total if dog_total else 0
                muffin_acc = 100 * counts[1, 1] / muffin_total if muffin_total else 0
                stat_a, stat_b = st.columns(2)
                stat_a.metric(
                    "치와와 정확도 / 오차율",
                    f"{dog_acc:.1f}%",
                    f"오차율 {100 - dog_acc:.1f}%",
                )
                stat_b.metric(
                    "머핀 정확도 / 오차율",
                    f"{muffin_acc:.1f}%",
                    f"오차율 {100 - muffin_acc:.1f}%",
                )
                st.caption(
                    f"혼동 행렬 · TP {counts[0, 0]} · FP {counts[1, 0]} · "
                    f"FN {counts[0, 1]} · TN {counts[1, 1]}"
                )
            except (OSError, RuntimeError, ValueError) as error:
                st.error(f"검증 중 오류가 발생했습니다: {error}")

with tab_loss:
    st.markdown(
        '<div class="step-title"><span class="step-index">L</span>학습 곡선 · epoch별 loss 변화</div>'
        '<p class="tab-intro">result.log의 현재 학습과 tune.log의 최고 정확도 스윕을 함께 표시합니다. 실선은 loss, 점선·× 마커는 직전 epoch 대비 Δ loss(%)입니다.</p>',
        unsafe_allow_html=True,
    )
    if training_curves:
        st.pyplot(training_curve_figure(training_curves), width="stretch")
        st.caption(
            "변화율 계산: (현재 epoch loss − 이전 epoch loss) ÷ 이전 epoch loss × 100. "
            "음수는 loss 감소, 양수는 증가를 뜻합니다."
        )
    else:
        st.info(
            "학습 로그가 아직 없습니다. 1016.py 또는 tune.py 실행 후 result.log / tune.log에 "
            "epoch별 loss가 기록되면 여기에 학습 곡선이 표시됩니다."
        )

with tab_feat:
    st.markdown(
        '<div class="step-title"><span class="step-index">B</span>합성곱 특징 맵</div>'
        '<p class="tab-intro">각 필터가 입력에서 활성화한 패턴을 채널별로 펼쳐 봅니다. 색상은 채널마다 개별 정규화한 활성도입니다.</p>',
        unsafe_allow_html=True,
    )
    if feature_one is not None and feature_two is not None:
        st.markdown('<div class="feature-title">Conv1 · 6개 채널</div>', unsafe_allow_html=True)
        st.pyplot(feature_figure(feature_one, "Conv1 · 6 channels"), width="stretch")
        st.markdown('<div class="feature-title">Conv2 · 16개 채널</div>', unsafe_allow_html=True)
        st.pyplot(feature_figure(feature_two, "Conv2 · 16 channels"), width="stretch")
    else:
        st.info("특징 맵을 만들 이미지가 없습니다. 테스트 샘플 폴더를 확인해 주세요.")

mean_label = ", ".join(f"{value:g}" for value in SCRIPT_CONFIG["normalize_mean"])
std_label = ", ".join(f"{value:g}" for value in SCRIPT_CONFIG["normalize_std"])
st.markdown(
    f'<div class="footer-note">{resize_label} RGB · Normalize ({mean_label}) / ({std_label}) · SOFTMAX CLASS PROBABILITIES · CHECKPOINT: {escape(SCRIPT_CONFIG["checkpoint_rel_path"])}</div>',
    unsafe_allow_html=True,
)
