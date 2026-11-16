"""Select explicitly named camera videos without guessing missing views."""

import re


def camera_name(value):
  if not re.fullmatch(r"cam[1-9][0-9]*", value):
    raise ValueError(f"Invalid camera name: {value!r}; expected cam1, cam2, ...")
  return value


def select_videos(directory, cameras=None):
  names = cameras if cameras is not None else sorted(
    path.stem for path in directory.glob("cam*.avi") if path.is_file())
  if not names or len(set(names)) != len(names):
    raise ValueError(f"No cameras or duplicate camera selection in {directory}")
  videos = []
  for name in names:
    video = directory / f"{camera_name(name)}.avi"
    if not video.is_file():
      raise FileNotFoundError(video)
    videos.append(video)
  return sorted(videos, key=lambda video: int(video.stem[3:]))