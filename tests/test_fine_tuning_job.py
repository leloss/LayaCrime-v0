import sys
import time

from laya_adverse_media.fine_tuning import FineTuningJob


def test_epoch_metrics_remain_available_after_incremental_log_poll(tmp_path) -> None:
    job = FineTuningJob()
    command = [
        sys.executable,
        "-c",
        (
            "print('epoch=1/2 batch=4 loss=0.7 sigma=0.4'); "
            "print('epoch_metrics epoch=1 train_loss=0.65 validation_loss=0.72 train_accuracy=0.85 validation_accuracy=0.8')"
        ),
    ]
    job.start(command, "training", tmp_path)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if job.snapshot()["status"] != "running":
            break
        time.sleep(0.01)

    first = job.snapshot()
    incremental = job.snapshot(logs_after=2)

    assert first["status"] == "complete"
    assert first["progress"] == {
        "epoch": 1,
        "epochs": 2,
        "batch": 4,
        "loss": 0.7,
        "sigma": 0.4,
    }
    assert first["loss_history"] == [{
        "epoch": 1,
        "train_loss": 0.65,
        "validation_loss": 0.72,
        "train_accuracy": 0.85,
        "validation_accuracy": 0.8,
    }]
    assert incremental["logs"] == []
    assert incremental["loss_history"] == first["loss_history"]


def test_private_environment_is_not_exposed_in_snapshot(tmp_path) -> None:
    job = FineTuningJob()
    token = "secret-write-token"
    command = [
        sys.executable,
        "-c",
        "import os; print('received=' + str(bool(os.environ.get('HF_TOKEN'))))",
    ]

    job.start(command, "training", tmp_path, env={"HF_TOKEN": token})
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if job.snapshot()["status"] != "running":
            break
        time.sleep(0.01)

    snapshot = job.snapshot()
    assert snapshot["status"] == "complete"
    assert snapshot["logs"][-1]["message"] == "received=True"
    assert token not in str(snapshot)