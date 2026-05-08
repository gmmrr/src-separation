#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diarization media player.

Play an audio/video file with a waveform timeline and speaker activity buttons.

Features:
  - Load media from --media, or use the "input" field inside --json.
  - Draw waveform on top.
  - Click or drag on the waveform to seek.
  - Show one button per speaker.
  - Highlight speaker buttons according to diarization JSON segments.
  - Supports overlapping speakers: multiple buttons can be highlighted.
  - Works with audio or video through Qt Multimedia.

Expected JSON format:
{
  "input": "/path/to/audio_or_video.wav",
  "speakers": ["SPEAKER_00", "SPEAKER_01"],
  "segments": [
    {"speaker": "SPEAKER_00", "start": 0.0, "end": 3.2},
    {"speaker": "SPEAKER_01", "start": 2.1, "end": 5.0}
  ]
}

Install:
  uv add pyside6 pyqtgraph numpy librosa soundfile

Run:
  uv run pipeline/visualization/diarization_player.py \
    --json data/test_diarization/pyannote_outputs/example_diarization.json

Or:
  uv run pipeline/visualization/diarization_player.py \
    --media data/test_video/example.mp4 \
    --json data/test_diarization/pyannote_outputs/example_diarization.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import librosa
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QTimer, Qt, QUrl
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStyle,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class Segment:
    speaker: str
    start: float
    end: float

    def active_at(self, time_sec: float) -> bool:
        return self.start <= time_sec < self.end


class ClickablePlotWidget(pg.PlotWidget):
    """A pyqtgraph PlotWidget that seeks on click/drag."""

    def __init__(self, seek_callback, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seek_callback = seek_callback
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        self._seek_from_event(event)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self._seek_from_event(event)
        super().mouseMoveEvent(event)

    def _seek_from_event(self, event):
        if self.seek_callback is None:
            return

        pos = event.position() if hasattr(event, "position") else event.pos()
        mouse_point = self.plotItem.vb.mapSceneToView(pos)
        self.seek_callback(float(mouse_point.x()))


class DiarizationPlayer(QMainWindow):
    def __init__(
        self,
        media_path: Path,
        json_path: Path,
        waveform_sample_rate: int = 16000,
        waveform_points: int = 5000,
    ):
        super().__init__()

        self.media_path = media_path
        self.json_path = json_path
        self.waveform_sample_rate = waveform_sample_rate
        self.waveform_points = waveform_points

        self.data = self._load_json(json_path)
        self.segments = self._load_segments(self.data)
        self.speakers = self._load_speakers(self.data, self.segments)

        self.duration_sec = 0.0
        self.is_slider_being_dragged = False

        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.audio_output.setVolume(1.0)
        self.audio_output.setMuted(False)
        self.player.setAudioOutput(self.audio_output)

        self.timer = QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self._refresh_from_player)

        self.speaker_buttons: Dict[str, QPushButton] = {}

        self._build_ui()
        self._load_media()
        self._load_waveform()

        self.player.durationChanged.connect(self._on_duration_changed)
        self.player.positionChanged.connect(self._on_position_changed)
        self.player.playbackStateChanged.connect(self._on_playback_state_changed)
        self.player.errorOccurred.connect(self._on_error_occurred)

        self.timer.start()

    @staticmethod
    def _load_json(json_path: Path) -> Dict:
        with json_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _load_segments(data: Dict) -> List[Segment]:
        segments = []
        for item in data.get("segments", []):
            try:
                speaker = str(item["speaker"])
                start = float(item["start"])
                end = float(item["end"])
            except KeyError as exc:
                raise ValueError(f"Invalid segment entry, missing key: {exc}") from exc

            if end <= start:
                continue

            segments.append(Segment(speaker=speaker, start=start, end=end))

        return sorted(segments, key=lambda seg: (seg.start, seg.end, seg.speaker))

    @staticmethod
    def _load_speakers(data: Dict, segments: Sequence[Segment]) -> List[str]:
        speakers = data.get("speakers")
        if speakers:
            return [str(s) for s in speakers]

        return sorted({seg.speaker for seg in segments})

    def _build_ui(self) -> None:
        self.setWindowTitle(f"Diarization Player - {self.media_path.name}")
        self.resize(1200, 820)

        root = QWidget(self)
        layout = QVBoxLayout(root)

        title = QLabel(f"Media: {self.media_path}")
        title.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(title)

        json_label = QLabel(f"JSON: {self.json_path}")
        json_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(json_label)

        self.waveform_plot = ClickablePlotWidget(self.seek_to_seconds)
        self.waveform_plot.setBackground("w")
        self.waveform_plot.setLabel("bottom", "Time", units="s")
        self.waveform_plot.setLabel("left", "Amplitude")
        self.waveform_plot.showGrid(x=True, y=True, alpha=0.25)
        self.waveform_plot.setMinimumHeight(180)
        layout.addWidget(self.waveform_plot)

        self.playhead = pg.InfiniteLine(pos=0.0, angle=90, movable=False, pen=pg.mkPen(width=2))
        self.waveform_plot.addItem(self.playhead)

        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(360)
        self.video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.video_widget)
        self.player.setVideoOutput(self.video_widget)

        controls = QHBoxLayout()

        self.play_button = QPushButton()
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_button.clicked.connect(self.toggle_play)
        controls.addWidget(self.play_button)

        self.time_label = QLabel("00:00.000 / 00:00.000")
        self.time_label.setMinimumWidth(210)
        controls.addWidget(self.time_label)

        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 1000)
        self.position_slider.sliderPressed.connect(self._on_slider_pressed)
        self.position_slider.sliderReleased.connect(self._on_slider_released)
        self.position_slider.sliderMoved.connect(self._on_slider_moved)
        controls.addWidget(self.position_slider)

        layout.addLayout(controls)

        speaker_title = QLabel("Speakers")
        layout.addWidget(speaker_title)

        speaker_grid = QGridLayout()
        for idx, speaker in enumerate(self.speakers):
            button = QPushButton(speaker)
            button.setCheckable(False)
            button.setMinimumHeight(42)
            button.setProperty("active", False)
            button.clicked.connect(lambda checked=False, sp=speaker: self.seek_to_first_segment(sp))
            self.speaker_buttons[speaker] = button
            speaker_grid.addWidget(button, idx // 4, idx % 4)

        layout.addLayout(speaker_grid)

        hint = QLabel(
            "Click/drag the waveform or slider to seek. "
            "Click a speaker button to jump to that speaker's first segment. "
            "Overlapping speech highlights multiple speakers."
        )
        layout.addWidget(hint)

        self.setCentralWidget(root)
        self._apply_button_styles(set())

    def _load_media(self) -> None:
        if not self.media_path.exists():
            raise FileNotFoundError(f"Media path does not exist: {self.media_path}")

        self.player.setSource(QUrl.fromLocalFile(str(self.media_path.resolve())))
        self.player.setPosition(0)

    def _load_waveform(self) -> None:
        try:
            waveform, sr = librosa.load(
                str(self.media_path),
                sr=self.waveform_sample_rate,
                mono=True,
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Waveform loading failed",
                "Could not load waveform with librosa. "
                "Playback may still work, but waveform will be empty.\n\n"
                f"{type(exc).__name__}: {exc}",
            )
            self.waveform_plot.setXRange(0, max(self._json_max_time(), 1.0))
            return

        if waveform.size == 0:
            self.waveform_plot.setXRange(0, max(self._json_max_time(), 1.0))
            return

        duration = waveform.shape[0] / float(sr)
        self.duration_sec = max(self.duration_sec, duration, self._json_max_time())

        times, values = self._downsample_waveform(waveform, sr, self.waveform_points)
        self.waveform_plot.plot(times, values, pen=pg.mkPen(width=1))

        self._draw_segment_backgrounds()

        self.waveform_plot.setXRange(0.0, max(self.duration_sec, 1.0), padding=0.01)
        self.waveform_plot.setYRange(-1.05, 1.05)

    def _draw_segment_backgrounds(self) -> None:
        if not self.segments:
            return

        speaker_to_index = {speaker: idx for idx, speaker in enumerate(self.speakers)}
        max_idx = max(len(self.speakers) - 1, 1)

        for seg in self.segments:
            idx = speaker_to_index.get(seg.speaker, 0)
            alpha = 28 + int(30 * (idx / max_idx))
            color = pg.mkColor(60 + (idx * 47) % 160, 120 + (idx * 31) % 100, 200 - (idx * 23) % 120, alpha)
            region = pg.LinearRegionItem(
                values=(seg.start, seg.end),
                movable=False,
                brush=color,
                pen=pg.mkPen(None),
            )
            region.setZValue(-10)
            self.waveform_plot.addItem(region)

    @staticmethod
    def _downsample_waveform(waveform: np.ndarray, sr: int, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
        waveform = waveform.astype(np.float32, copy=False)
        peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
        if peak > 0:
            waveform = waveform / peak

        if waveform.shape[0] <= max_points:
            times = np.arange(waveform.shape[0], dtype=np.float32) / float(sr)
            return times, waveform

        samples_per_bin = int(np.ceil(waveform.shape[0] / max_points))
        pad_len = samples_per_bin * max_points - waveform.shape[0]
        padded = np.pad(waveform, (0, pad_len), mode="constant")
        bins = padded.reshape(max_points, samples_per_bin)

        positive = np.max(bins, axis=1)
        negative = np.min(bins, axis=1)

        times = np.arange(max_points, dtype=np.float32) * samples_per_bin / float(sr)
        stacked_times = np.repeat(times, 2)
        stacked_values = np.empty(max_points * 2, dtype=np.float32)
        stacked_values[0::2] = negative
        stacked_values[1::2] = positive
        return stacked_times, stacked_values

    def _json_max_time(self) -> float:
        if not self.segments:
            return 0.0
        return max(seg.end for seg in self.segments)

    def _on_duration_changed(self, duration_ms: int) -> None:
        duration = duration_ms / 1000.0
        self.duration_sec = max(self.duration_sec, duration, self._json_max_time())
        self._update_time_label(self.player.position() / 1000.0)

    def _on_position_changed(self, position_ms: int) -> None:
        current_sec = position_ms / 1000.0
        self._update_playhead(current_sec)
        self._update_speakers(current_sec)
        self._update_time_label(current_sec)

        if not self.is_slider_being_dragged and self.duration_sec > 0:
            value = int(round((current_sec / self.duration_sec) * self.position_slider.maximum()))
            self.position_slider.blockSignals(True)
            self.position_slider.setValue(max(0, min(self.position_slider.maximum(), value)))
            self.position_slider.blockSignals(False)

    def _on_playback_state_changed(self, state) -> None:
        if state == QMediaPlayer.PlayingState:
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def _on_error_occurred(self, error, error_string: str = "") -> None:
        if error == QMediaPlayer.NoError:
            return

        detail = error_string or self.player.errorString()
        QMessageBox.warning(
            self,
            "Playback error",
            f"Qt Multimedia reported an error while playing media.\n\n{detail}",
        )

    def _on_slider_pressed(self) -> None:
        self.is_slider_being_dragged = True

    def _on_slider_released(self) -> None:
        self.is_slider_being_dragged = False
        self._seek_from_slider_value(self.position_slider.value())

    def _on_slider_moved(self, value: int) -> None:
        self._seek_from_slider_value(value)

    def _seek_from_slider_value(self, value: int) -> None:
        if self.duration_sec <= 0:
            return

        ratio = value / float(self.position_slider.maximum())
        self.seek_to_seconds(ratio * self.duration_sec)

    def _refresh_from_player(self) -> None:
        current_sec = self.player.position() / 1000.0
        self._update_playhead(current_sec)
        self._update_speakers(current_sec)
        self._update_time_label(current_sec)

    def _update_playhead(self, current_sec: float) -> None:
        self.playhead.setPos(current_sec)

    def _update_speakers(self, current_sec: float) -> None:
        active = self.active_speakers_at(current_sec)
        self._apply_button_styles(active)

    def _apply_button_styles(self, active: Set[str]) -> None:
        for speaker, button in self.speaker_buttons.items():
            if speaker in active:
                button.setStyleSheet(
                    """
                    QPushButton {
                        background-color: #34c759;
                        color: black;
                        font-weight: bold;
                        border: 2px solid #157f34;
                        border-radius: 8px;
                        padding: 8px;
                    }
                    """
                )
            else:
                button.setStyleSheet(
                    """
                    QPushButton {
                        background-color: #e5e5e5;
                        color: #333333;
                        border: 1px solid #aaaaaa;
                        border-radius: 8px;
                        padding: 8px;
                    }
                    """
                )

    def active_speakers_at(self, current_sec: float) -> Set[str]:
        return {seg.speaker for seg in self.segments if seg.active_at(current_sec)}

    def _update_time_label(self, current_sec: float) -> None:
        self.time_label.setText(
            f"{self._format_time(current_sec)} / {self._format_time(self.duration_sec)}"
        )

    @staticmethod
    def _format_time(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        minutes = int(seconds // 60)
        remain = seconds - minutes * 60
        return f"{minutes:02d}:{remain:06.3f}"

    def seek_to_seconds(self, seconds: float) -> None:
        if self.duration_sec > 0:
            seconds = max(0.0, min(float(seconds), self.duration_sec))
        else:
            seconds = max(0.0, float(seconds))

        self.player.setPosition(int(round(seconds * 1000)))
        self._update_playhead(seconds)
        self._update_speakers(seconds)
        self._update_time_label(seconds)

    def seek_to_first_segment(self, speaker: str) -> None:
        for seg in self.segments:
            if seg.speaker == speaker:
                self.seek_to_seconds(seg.start)
                return

    def toggle_play(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()


def resolve_media_path(json_data: Dict, json_path: Path, media_arg: Optional[Path]) -> Path:
    if media_arg is not None:
        return media_arg.expanduser().resolve()

    input_value = json_data.get("input")
    if not input_value:
        raise ValueError("No --media was provided and JSON does not contain an 'input' field.")

    input_path = Path(input_value).expanduser()
    if input_path.is_absolute():
        return input_path

    candidate_from_cwd = input_path.resolve()
    if candidate_from_cwd.exists():
        return candidate_from_cwd

    candidate_from_json_dir = (json_path.parent / input_path).resolve()
    return candidate_from_json_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play media with pyannote diarization speaker highlights.")
    parser.add_argument(
        "--json",
        type=Path,
        required=True,
        help="Path to diarization JSON.",
    )
    parser.add_argument(
        "--media",
        type=Path,
        default=None,
        help="Optional audio/video file. If omitted, the JSON 'input' field is used.",
    )
    parser.add_argument(
        "--waveform-sample-rate",
        type=int,
        default=16000,
        help="Sample rate used only for waveform drawing.",
    )
    parser.add_argument(
        "--waveform-points",
        type=int,
        default=5000,
        help="Maximum number of waveform bins to draw.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()

    json_path = args.json.expanduser().resolve()
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file does not exist: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        json_data = json.load(f)

    media_path = resolve_media_path(json_data, json_path, args.media)
    if not media_path.exists():
        raise FileNotFoundError(
            f"Media file does not exist: {media_path}\n"
            "Pass --media explicitly if the JSON 'input' field points to a different file."
        )

    app = QApplication(sys.argv)
    window = DiarizationPlayer(
        media_path=media_path,
        json_path=json_path,
        waveform_sample_rate=args.waveform_sample_rate,
        waveform_points=args.waveform_points,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
