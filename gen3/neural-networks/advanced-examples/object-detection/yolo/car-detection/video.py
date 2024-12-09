from pathlib import Path

import requests
import xmltodict


class Video:
    DEPTHAI_RECORDINGS_URL = "https://depthai-recordings.fra1.digitaloceanspaces.com/"

    def __init__(self, video: str) -> None:
        self.videos_folder = Path(__file__).parent / "videos"
        self.video = video

    def set_videos_folder(self, folder: Path) -> None:
        self.videos_folder = folder

    def get_path(self) -> Path:
        if self._is_existing_video_path():
            return self.video

        return self.get_or_download_video()

    def _is_existing_video_path(self) -> bool:
        path = Path(self.video)
        return path.exists() and path.is_file()

    def get_or_download_video(self):
        path = self._video_path()
        if not path:
            recordings = self._get_available_recordings()
            self._download_video(recordings)
            path = self._video_path()
        return path

    def _video_path(self) -> str | None:
        video_path = self.videos_folder / self.video
        video_path.mkdir(parents=True, exist_ok=True)
        for item in video_path.iterdir():
            if item.is_file():
                return item

    def _get_available_recordings(
        self,
    ) -> dict[str, tuple[list[str], int]]:
        """
        Get available (online) depthai-recordings. Returns list of available recordings and it's size
        """
        x = requests.get(self.DEPTHAI_RECORDINGS_URL)
        if x.status_code != 200:
            raise ValueError("DepthAI-Recordings server currently isn't available!")

        d = xmltodict.parse(x.content)
        recordings: dict[str, list[list[str], int]] = dict()

        for content in d["ListBucketResult"]["Contents"]:
            name = content["Key"].split("/")[0]
            if name not in recordings:
                recordings[name] = [[], 0]
            recordings[name][0].append(content["Key"])
            recordings[name][1] += int(content["Size"])

        return recordings

    def _download_video(self, available_recordings: dict) -> None:
        if self.video in available_recordings:
            arr = available_recordings[self.video]
            print(
                "Downloading depthai recording '{}' from Luxonis' servers, in total {:.2f} MB".format(
                    self.video, arr[1] / 1e6
                )
            )
            self._download_recording(arr[0])
        else:
            raise ValueError(
                f"DepthAI recording '{self.video}' was not found on the server!"
            )

    def _download_recording(self, keys: list[str]) -> None:
        self.videos_folder.mkdir(parents=True, exist_ok=True)
        for key in keys:
            if key.endswith("/"):  # Folder
                continue
            url = self.DEPTHAI_RECORDINGS_URL + key
            self._download_file(str(self.videos_folder / key), url)
            print("Downloaded", key)

    def _download_file(self, path: str, url: str):
        r = requests.get(url)
        if r.status_code != 200:
            raise ValueError(f"Could not download file from {url}!")

        with open(path, "wb") as f:
            f.write(r.content)
