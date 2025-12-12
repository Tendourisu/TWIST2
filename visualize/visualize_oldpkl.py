from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import joblib
import numpy as np
import viser
from viser.extras import ViserUrdf


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MOTION_DIR = (
    REPO_ROOT / "data" / "motions" / "mocap"
).resolve()
DEFAULT_URDF_PATH = (
    REPO_ROOT / "data" / "motions" / "unitree_description" / "urdf" / "g1" / "main.urdf"
).resolve()

EMPTY_FILE_OPTION = "<未找到 PKL>"
EMPTY_MOTION_OPTION = "<未找到 Motion>"


@dataclass
class MotionClip:
    file_name: str
    motion_name: str
    fps: float
    dof: np.ndarray
    translations: np.ndarray
    root_wxyz: np.ndarray
    fit_loss: float

    @property
    def num_frames(self) -> int:
        return int(self.dof.shape[0])

    @property
    def duration(self) -> float:
        return float(self.num_frames / self.fps) if self.fps > 0 else 0.0


class MotionLibrary:
    def __init__(self, motion_dir: Path) -> None:
        self.motion_dir = motion_dir
        self._cache: Dict[str, tuple[float, Dict[str, Dict[str, object]]]] = {}

    def list_files(self) -> list[str]:
        pkls = sorted(
            [
                p.name
                for p in self.motion_dir.glob("*.pkl")
                if p.is_file() and not p.name.startswith(".")
            ]
        )
        return pkls

    def list_motion_names(self, file_name: str) -> list[str]:
        data = self._load_raw(file_name)
        return sorted(data.keys())

    def load_clip(self, file_name: str, motion_name: str) -> MotionClip:
        payload = self._load_raw(file_name)
        if motion_name not in payload:
            raise KeyError(f"未在 {file_name} 内找到 {motion_name}")
        motion_data = payload[motion_name]
        if not isinstance(motion_data, dict):
            raise TypeError(f"{motion_name} 的内容不是字典，实际类型为 {type(motion_data)}")

        translations = _as_float_array(motion_data, "root_trans_offset", expected_rank=2)
        root_rot_xyzw = _as_float_array(motion_data, "root_rot", expected_rank=2)
        dof = _as_float_array(motion_data, "dof", expected_rank=2)

        if translations.shape[0] == 0:
            raise ValueError(f"{motion_name} 没有任何帧数据")
        if not (
            translations.shape[0]
            == root_rot_xyzw.shape[0]
            == dof.shape[0]
        ):
            raise ValueError(
                f"{motion_name} 的根位姿或 DoF 帧数不一致: "
                f"{translations.shape[0]}, {root_rot_xyzw.shape[0]}, {dof.shape[0]}"
            )
        if root_rot_xyzw.shape[1] != 4:
            raise ValueError(f"{motion_name} 的 root_rot 形状异常: {root_rot_xyzw.shape}")

        fps = float(motion_data.get("fps", 30))
        fit_loss = float(motion_data.get("fit_loss", 0.0))
        root_wxyz = np.roll(root_rot_xyzw, shift=1, axis=-1)

        return MotionClip(
            file_name=file_name,
            motion_name=motion_name,
            fps=fps,
            dof=dof,
            translations=translations,
            root_wxyz=root_wxyz,
            fit_loss=fit_loss,
        )

    def _load_raw(self, file_name: str) -> Dict[str, Dict[str, object]]:
        path = (self.motion_dir / file_name).resolve()
        if not path.exists():
            raise FileNotFoundError(f"{path} 不存在")

        mtime = path.stat().st_mtime
        cached = self._cache.get(file_name)
        if cached and cached[0] == mtime:
            return cached[1]

        with path.open("rb") as fp:
            data = joblib.load(fp)

        if not isinstance(data, dict):
            raise TypeError(f"{file_name} 根对象类型为 {type(data)}，预期为 dict")
        self._cache[file_name] = (mtime, data)
        return data


class MotionPlayback:
    def __init__(
        self,
        viser_urdf: ViserUrdf,
        base_frame: viser.FrameHandle,
    ) -> None:
        self._viser_urdf = viser_urdf
        self._base_frame = base_frame
        self._joint_count = len(self._viser_urdf.get_actuated_joint_names())

        self._clip: Optional[MotionClip] = None
        self._frame_idx = 0
        self._loop = True
        self._speed = 1.0
        self._playing = False
        self._lock = threading.Lock()

        self._frame_slider: Optional[viser.GuiSliderHandle[float]] = None
        self._slider_guard = False
        self._status_cb: Optional[Callable[[str], None]] = None

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        self._reset_pose()

    def bind_status_callback(self, callback: Callable[[str], None]) -> None:
        self._status_cb = callback

    def bind_frame_slider(self, slider: viser.GuiSliderHandle[float]) -> None:
        self._frame_slider = slider
        slider.precision = 0
        slider.min = 0.0
        slider.max = 1.0
        slider.step = 1.0
        slider.value = 0.0

        @slider.on_update
        def _(_) -> None:
            if self._slider_guard:
                return
            self.seek(int(round(slider.value)), keep_play_state=True)

    def set_clip(self, clip: MotionClip) -> None:
        if clip.dof.shape[1] != self._joint_count:
            raise ValueError(
                f"{clip.motion_name} 的 DoF 数 ({clip.dof.shape[1]}) 与 URDF 关节数 "
                f"({self._joint_count}) 不匹配"
            )
        with self._lock:
            self._clip = clip
            self._frame_idx = 0
            self._playing = False
        self._apply_frame(clip, 0)
        self._configure_slider_for_clip(clip)

    def clear(self) -> None:
        with self._lock:
            self._clip = None
            self._frame_idx = 0
            self._playing = False
        self._reset_pose()
        self._set_slider_value(0.0)

    def play(self) -> None:
        with self._lock:
            if self._clip is None:
                return
            self._playing = True

    def pause(self) -> None:
        with self._lock:
            self._playing = False

    def toggle_play(self) -> None:
        with self._lock:
            if self._clip is None:
                return
            self._playing = not self._playing

    def seek(self, frame_idx: int, keep_play_state: bool = True) -> None:
        with self._lock:
            clip = self._clip
            playing = self._playing
        if clip is None:
            return
        clamped = int(np.clip(frame_idx, 0, clip.num_frames - 1))
        with self._lock:
            self._frame_idx = clamped
            if not keep_play_state:
                self._playing = False
        self._apply_frame(clip, clamped)
        if not keep_play_state:
            self._set_slider_value(float(clamped))

    @property
    def looping(self) -> bool:
        return self._loop

    @looping.setter
    def looping(self, value: bool) -> None:
        self._loop = bool(value)

    @property
    def speed(self) -> float:
        return self._speed

    @speed.setter
    def speed(self, value: float) -> None:
        self._speed = max(1e-3, float(value))

    def _configure_slider_for_clip(self, clip: MotionClip) -> None:
        if self._frame_slider is None:
            return
        max_idx = max(clip.num_frames - 1, 0)
        self._frame_slider.min = 0.0
        self._frame_slider.max = float(max_idx)
        self._frame_slider.step = 1.0
        self._frame_slider.precision = 0
        self._set_slider_value(0.0)

    def _apply_frame(self, clip: MotionClip, frame_idx: int) -> None:
        cfg = np.asarray(clip.dof[frame_idx], dtype=np.float64)
        self._viser_urdf.update_cfg(cfg)

        pos = tuple(map(float, clip.translations[frame_idx]))
        quat = tuple(map(float, clip.root_wxyz[frame_idx]))
        self._base_frame.position = pos
        self._base_frame.wxyz = quat
        self._set_slider_value(float(frame_idx))

    def _reset_pose(self) -> None:
        zero_cfg = np.zeros(self._joint_count, dtype=np.float64)
        self._viser_urdf.update_cfg(zero_cfg)
        self._base_frame.position = (0.0, 0.0, 0.0)
        self._base_frame.wxyz = (1.0, 0.0, 0.0, 0.0)

    def _set_slider_value(self, value: float) -> None:
        if self._frame_slider is None:
            return
        self._slider_guard = True
        try:
            self._frame_slider.value = value
        finally:
            self._slider_guard = False

    def _run(self) -> None:
        while True:
            with self._lock:
                clip = self._clip
                playing = self._playing
                idx = self._frame_idx
                loop = self._loop
                speed = self._speed
            if clip is None or not playing:
                time.sleep(0.01)
                continue

            self._apply_frame(clip, idx)
            dt = (1.0 / clip.fps) / max(speed, 1e-3)
            time.sleep(max(0.0, dt))

            with self._lock:
                if clip is not self._clip:
                    continue
                if self._frame_idx != idx:
                    continue
                next_idx = idx + 1
                if next_idx >= clip.num_frames:
                    if loop:
                        next_idx = 0
                    else:
                        next_idx = clip.num_frames - 1
                        self._playing = False
                        if self._status_cb is not None:
                            self._status_cb("播放完毕，自动暂停")
                self._frame_idx = next_idx


def _as_float_array(
    data: Dict[str, object],
    key: str,
    expected_rank: int,
) -> np.ndarray:
    if key not in data:
        raise KeyError(f"缺少键 {key}")
    value = np.asarray(data[key], dtype=np.float64)
    if value.ndim != expected_rank:
        raise ValueError(f"{key} 维度为 {value.ndim}，预期为 {expected_rank}")
    return np.ascontiguousarray(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用 viser 可视化 G1 mocap pkl 数据"
    )
    parser.add_argument(
        "--motion-dir",
        type=Path,
        default=DEFAULT_MOTION_DIR,
        help="包含 joblib pkl 的目录",
    )
    parser.add_argument(
        "--urdf-path",
        type=Path,
        default=DEFAULT_URDF_PATH,
        help="G1 机器人 URDF 路径",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="viser 服务器绑定地址",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="viser 服务器端口",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="G1 Motion Viewer",
        help="GUI 顶部标题",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    motion_dir = args.motion_dir.expanduser().resolve()
    urdf_path = args.urdf_path.expanduser().resolve()
    if not motion_dir.exists():
        raise FileNotFoundError(f"motion 目录 {motion_dir} 不存在")
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF {urdf_path} 不存在")

    server = viser.ViserServer(host=args.host, port=args.port)
    server.gui.configure_theme(dark_mode=True)
    server.gui.add_markdown(
        f"### {args.title}\n"
        "1. 先选择一个 PKL 文件，再选择该文件中的 motion 名字。\n"
        "2. 通过播放控制区调整循环、速度或帧位置。\n"
        "3. 客户端可通过浏览器访问 `<host>:<port>` 查看可视化。"
    )

    server.scene.world_axes.visible = True
    server.scene.add_grid(
        "/grid",
        width=4.0,
        height=4.0,
        plane="xy",
        position=(0.0, 0.0, 0.0),
    )
    base_frame = server.scene.add_frame("/g1_root", show_axes=False)
    viser_urdf = ViserUrdf(
        server,
        urdf_or_path=urdf_path,
        root_node_name="/g1_root",
        load_meshes=True,
        load_collision_meshes=False,
    )

    motion_lib = MotionLibrary(motion_dir)
    player = MotionPlayback(viser_urdf, base_frame)

    status_md = server.gui.add_markdown("**状态**：等待操作")
    info_md = server.gui.add_markdown("尚未加载 motion。")

    def update_status(message: str) -> None:
        print(message)
        status_md.content = f"**状态**：{message}"

    def update_info(clip: Optional[MotionClip]) -> None:
        if clip is None:
            info_md.content = "尚未加载 motion。"
            return
        info_md.content = (
            f"**文件**：`{clip.file_name}`  \n"
            f"**Motion**：`{clip.motion_name}`  \n"
            f"**帧数**：{clip.num_frames}  \n"
            f"**FPS**：{clip.fps:.2f}  \n"
            f"**时长**：{clip.duration:.2f}s  \n"
            f"**Fit loss**：{clip.fit_loss:.5f}"
        )

    player.bind_status_callback(update_status)

    with server.gui.add_folder("播放控制"):
        play_button = server.gui.add_button("播放 / 暂停")
        reset_button = server.gui.add_button("回到首帧")
        loop_checkbox = server.gui.add_checkbox("循环播放", initial_value=True)
        speed_slider = server.gui.add_slider(
            "播放速度", min=0.1, max=3.0, step=0.1, initial_value=1.0
        )
        frame_slider = server.gui.add_slider(
            "帧索引",
            min=0.0,
            max=1.0,
            step=1.0,
            initial_value=0.0,
        )
        frame_slider.precision = 0

    player.bind_frame_slider(frame_slider)

    @play_button.on_click
    def _(_) -> None:
        player.toggle_play()

    @reset_button.on_click
    def _(_) -> None:
        player.seek(0, keep_play_state=False)
        player.pause()

    @loop_checkbox.on_update
    def _(_) -> None:
        player.looping = loop_checkbox.value

    @speed_slider.on_update
    def _(_) -> None:
        player.speed = speed_slider.value

    with server.gui.add_folder("文件与片段"):
        refresh_button = server.gui.add_button("刷新文件列表")
        file_dropdown = server.gui.add_dropdown(
            "PKL 文件", (EMPTY_FILE_OPTION,)
        )
        motion_dropdown = server.gui.add_dropdown(
            "Motion 名字", (EMPTY_MOTION_OPTION,)
        )

    state: Dict[str, Optional[str]] = {"file": None, "motion": None}

    def clear_motion_selection() -> None:
        motion_dropdown.options = (EMPTY_MOTION_OPTION,)
        state["motion"] = None
        player.clear()
        update_info(None)

    def handle_motion_change(motion_name: str) -> None:
        if motion_name == EMPTY_MOTION_OPTION:
            state["motion"] = None
            player.pause()
            return
        file_name = state["file"]
        if file_name is None:
            return
        try:
            clip = motion_lib.load_clip(file_name, motion_name)
        except Exception as exc:  # noqa: BLE001
            update_status(f"加载 motion 失败：{exc}")
            return
        try:
            player.set_clip(clip)
        except ValueError as exc:
            update_status(str(exc))
            return
        state["motion"] = motion_name
        update_info(clip)
        update_status(f"已加载 {motion_name}（{clip.num_frames} 帧）")

    def handle_file_change(file_name: str) -> None:
        if file_name == EMPTY_FILE_OPTION:
            state["file"] = None
            clear_motion_selection()
            return
        state["file"] = file_name
        try:
            motions = motion_lib.list_motion_names(file_name)
        except Exception as exc:  # noqa: BLE001
            update_status(f"读取 {file_name} 失败：{exc}")
            clear_motion_selection()
            return
        if not motions:
            motion_dropdown.options = (EMPTY_MOTION_OPTION,)
            clear_motion_selection()
            update_status(f"{file_name} 中没有 motion")
            return
        motion_dropdown.options = tuple(motions)
        motion_dropdown.value = motions[0]
        handle_motion_change(motions[0])

    def refresh_files() -> None:
        files = motion_lib.list_files()
        if not files:
            file_dropdown.options = (EMPTY_FILE_OPTION,)
            clear_motion_selection()
            update_status(f"目录 {motion_dir} 内没有 pkl")
            return
        file_dropdown.options = tuple(files)
        update_status(f"已发现 {len(files)} 个文件")
        file_dropdown.value = files[0]
        handle_file_change(files[0])

    @file_dropdown.on_update
    def _(_) -> None:
        handle_file_change(file_dropdown.value)

    @motion_dropdown.on_update
    def _(_) -> None:
        handle_motion_change(motion_dropdown.value)

    @refresh_button.on_click
    def _(_) -> None:
        refresh_files()

    refresh_files()
    update_status("准备完毕，可通过浏览器访问查看动画")

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
