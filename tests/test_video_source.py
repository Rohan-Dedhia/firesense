"""
Tests for the video-file camera source and the tools around it.

No real video files: each test writes a tiny synthetic video with
cv2.VideoWriter into a temporary folder. Every frame is one flat grey level
that encodes its own index, so a test can tell exactly which frame came back
(MJPG is lossy, but a flat frame survives to within a couple of levels).

Without OpenCV these tests return early - run_tests.py must work on a fresh
machine with nothing installed.
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import camera as cam_mod               # noqa: E402
import extract_frames as E             # noqa: E402
import ingest                          # noqa: E402
import pick_vision_model as PVM        # noqa: E402
import pipeline as P                   # noqa: E402

N_FRAMES = 10
STEP = 20                              # grey level per frame index: 0, 20, ..., 180
SIZE = (64, 48)                        # w, h


def make_video(folder: Path, name: str = "clip.avi", n: int = N_FRAMES,
               fps: float = 10.0) -> Path:
    cv2 = cam_mod.cv2
    path = folder / name
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, SIZE)
    assert w.isOpened(), "this OpenCV build cannot write MJPG"
    for i in range(n):
        w.write(np.full((SIZE[1], SIZE[0], 3), i * STEP, dtype=np.uint8))
    w.release()
    return path


def index_of(frame) -> int:
    return int(round(float(frame.image.mean()) / STEP))


def no_cv2() -> bool:
    return not cam_mod.CV2_AVAILABLE


# ---------------------------------------------------------------------------
# playback
# ---------------------------------------------------------------------------
def test_every_frame_in_order_then_end_of_stream():
    if no_cv2():
        return
    with tempfile.TemporaryDirectory() as d:
        src = cam_mod.VideoFileSource(make_video(Path(d)), realtime=False)
        assert src.open(), src.error
        got = [index_of(src.read()) for _ in range(N_FRAMES)]
        assert got == list(range(N_FRAMES))
        end = src.read()
        assert not end.ok and end.end_of_stream and end.error == cam_mod.END_OF_STREAM
        # reading on after the end stays clean, never raises
        for _ in range(3):
            again = src.read()
            assert not again.ok and again.end_of_stream
        src.close()


def test_loop_starts_again_at_the_first_frame():
    if no_cv2():
        return
    with tempfile.TemporaryDirectory() as d:
        src = cam_mod.VideoFileSource(make_video(Path(d)), loop=True, realtime=False)
        assert src.open(), src.error
        frames = [src.read() for _ in range(2 * N_FRAMES + 3)]
        assert all(f.ok for f in frames)
        assert [index_of(f) for f in frames] == list(range(N_FRAMES)) * 2 + [0, 1, 2]
        src.close()


def test_realtime_repeats_a_frame_when_polled_faster_than_the_video():
    if no_cv2():
        return
    with tempfile.TemporaryDirectory() as d:
        # 0.5 fps: the first frame stays current for two seconds
        src = cam_mod.VideoFileSource(make_video(Path(d)), fps=0.5)
        assert src.open(), src.error
        a, b = src.read(), src.read()
        assert a.ok and b.ok and index_of(a) == index_of(b) == 0
        src.close()


def test_realtime_skips_frames_when_polled_slower_than_the_video():
    if no_cv2():
        return
    with tempfile.TemporaryDirectory() as d:
        # 50 fps: 10 frames last 200 ms; an 80 ms pause is about 4 frames,
        # with room on either side for a slow machine.
        src = cam_mod.VideoFileSource(make_video(Path(d)), fps=50.0)
        assert src.open(), src.error
        first = src.read()
        time.sleep(0.08)
        later = src.read()
        assert first.ok and later.ok
        assert index_of(later) >= index_of(first) + 2
        src.close()


def test_realtime_without_loop_ends_when_the_clock_passes_the_end():
    if no_cv2():
        return
    with tempfile.TemporaryDirectory() as d:
        video = make_video(Path(d))
        src = cam_mod.VideoFileSource(video, fps=200.0)       # 10 frames = 50 ms
        assert src.open(), src.error
        time.sleep(0.12)
        f = src.read()
        assert not f.ok and f.end_of_stream
        looping = cam_mod.VideoFileSource(video, fps=200.0, loop=True)
        assert looping.open(), looping.error
        time.sleep(0.12)
        assert looping.read().ok
        src.close()
        looping.close()


# ---------------------------------------------------------------------------
# failures say what went wrong
# ---------------------------------------------------------------------------
def test_missing_file_is_a_clear_error():
    if no_cv2():
        return
    src = cam_mod.VideoFileSource(HERE / "no_such_video.mp4")
    assert src.open() is False
    assert "not found" in src.error and "no_such_video.mp4" in src.error
    f = src.read()
    assert not f.ok and not f.end_of_stream and f.error


def test_a_file_that_is_not_a_video_is_a_clear_error():
    if no_cv2():
        return
    with tempfile.TemporaryDirectory() as d:
        bad = Path(d) / "broken.mp4"
        bad.write_bytes(b"this is not a video" * 100)
        src = cam_mod.VideoFileSource(bad)
        assert src.open() is False
        assert "broken.mp4" in src.error
        assert ("could not open" in src.error) or ("no frame could be decoded" in src.error)
        assert src.cap is None


def test_a_non_positive_fps_override_is_refused():
    if no_cv2():
        return
    with tempfile.TemporaryDirectory() as d:
        src = cam_mod.VideoFileSource(make_video(Path(d)), fps=0.0)
        assert src.open() is False and "camera-fps" in src.error


# ---------------------------------------------------------------------------
# make_camera routing
# ---------------------------------------------------------------------------
def test_make_camera_routes_video_files_by_extension():
    for spec in ("demo.mp4", "clip.MOV", "a/b/c.avi", "~/x.mkv"):
        assert isinstance(cam_mod.make_camera(spec), cam_mod.VideoFileSource), spec
    assert isinstance(cam_mod.make_camera("webcam"), cam_mod.WebcamSource)
    assert isinstance(cam_mod.make_camera("webcam:1"), cam_mod.WebcamSource)
    assert isinstance(cam_mod.make_camera("192.168.1.42"), cam_mod.Esp32CamSource)
    assert isinstance(cam_mod.make_camera("http://cam.local/capture"), cam_mod.Esp32CamSource)


def test_make_camera_passes_video_options_through():
    src = cam_mod.make_camera("demo.mp4", loop=True, fps=5.0, realtime=False)
    assert (src.loop, src.fps_override, src.realtime) == (True, 5.0, False)
    plain = cam_mod.make_camera("demo.mp4")
    assert (plain.loop, plain.fps_override, plain.realtime) == (False, None, True)


def test_a_missing_video_is_not_mistaken_for_a_camera_url():
    src = cam_mod.make_camera("no_such_demo.mp4")
    assert isinstance(src, cam_mod.VideoFileSource)
    if not no_cv2():
        assert src.open() is False and "not found" in src.error


# ---------------------------------------------------------------------------
# the pipeline runs on a video, and carries on when it ends
# ---------------------------------------------------------------------------
def test_pipeline_continues_on_sensors_when_the_video_ends():
    if no_cv2():
        return
    with tempfile.TemporaryDirectory() as d:
        pipe = P.FireSensePipeline(
            ingest.MockSource("normal"), camera_spec=str(make_video(Path(d))),
            camera_opts={"realtime": False}, weights=str(HERE / "no_such.pt"),
            log_csv=False, send_alerts=False, vision_every=1)
        assert pipe.camera is not None and pipe.camera.name == "video"
        notes = []
        for _ in range(N_FRAMES + 5):
            res = pipe.step()
            assert res is not None and res.engine is not None
            notes += res.notes
        assert pipe.camera is None
        ended = [n for n in notes if "video ended" in n]
        assert len(ended) == 1, notes


# ---------------------------------------------------------------------------
# extract_frames.py
# ---------------------------------------------------------------------------
def test_extract_keeps_every_nth_frame():
    if no_cv2():
        return
    with tempfile.TemporaryDirectory() as d:
        video = make_video(Path(d))
        out = Path(d) / "frames"
        saved, err = E.extract(video, out, every=3)
        assert err == "" and saved == 4                       # frames 0, 3, 6, 9
        names = sorted(p.name for p in out.glob("*.jpg"))
        assert names == [f"clip_{i:06d}.jpg" for i in (0, 3, 6, 9)]


def test_extract_default_folder_is_not_the_esp32_check_folder():
    out = E.default_out(Path("videos/candle.mov"))
    assert out == Path("data") / "vision_check_phone" / "candle"
    assert not E.inside_check_dir(out)
    assert E.inside_check_dir(Path("data") / "vision_check" / "flame")


def test_extract_refuses_to_write_into_the_esp32_check_folder():
    if no_cv2():
        return
    with tempfile.TemporaryDirectory() as d:
        target = E.CHECK_DIR / "flame_from_phone_test"
        saved, err = E.extract(make_video(Path(d)), target, every=1)
        assert saved == 0 and "refusing" in err
        assert not target.exists()


def test_extract_will_not_mix_into_a_folder_with_frames_unless_forced():
    if no_cv2():
        return
    with tempfile.TemporaryDirectory() as d:
        video = make_video(Path(d))
        out = Path(d) / "frames"
        out.mkdir()
        (out / "earlier.jpg").write_bytes(b"x")
        saved, err = E.extract(video, out, every=5)
        assert saved == 0 and "--force" in err
        saved, err = E.extract(video, out, every=5, force=True)
        assert err == "" and saved == 2
        assert (out / "earlier.jpg").exists()                 # nothing deleted


def test_extract_reports_a_missing_video():
    if no_cv2():
        return
    with tempfile.TemporaryDirectory() as d:
        saved, err = E.extract(Path(d) / "gone.mp4", Path(d) / "out", every=1)
        assert saved == 0 and "not found" in err
        assert not (Path(d) / "out").exists()


# ---------------------------------------------------------------------------
# pick_vision_model.py: no video into the ESP32 check folder
# ---------------------------------------------------------------------------
def test_vision_check_capture_refuses_a_video_before_touching_frames():
    folder = PVM.CHECK_DIR / "normal"
    before = sorted(folder.glob("*.jpg")) if folder.exists() else None
    assert PVM.capture("normal", 1, "phone_clip.mp4", 0) == 1
    after = sorted(folder.glob("*.jpg")) if folder.exists() else None
    assert before == after                                    # nothing deleted or added
