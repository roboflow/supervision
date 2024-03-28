import argparse
import os
import subprocess
import tempfile
from glob import glob
from threading import Thread

import yaml

SERVER_CONFIG = {"protocols": ["tcp"], "paths": {"all": {"source": "publisher"}}}
BASE_STREAM_URL = "rtsp://localhost:8554/live"


def main(video_directory: str, number_of_streams: int) -> None:
    video_files = find_video_files_in_directory(video_directory, number_of_streams)
    try:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_file_path = create_server_config_file(temporary_directory)
            run_rtsp_server(config_path=config_file_path)
            stream_videos(video_files)
    finally:
        stop_rtsp_server()


def find_video_files_in_directory(directory: str, limit: int) -> list:
    video_formats = ["*.mp4", "*.webm"]
    video_paths = []
    for video_format in video_formats:
        video_paths.extend(glob(os.path.join(directory, video_format)))
    return video_paths[:limit]


def create_server_config_file(directory: str) -> str:
    config_path = os.path.join(directory, "rtsp-simple-server.yml")
    with open(config_path, "w") as config_file:
        yaml.dump(SERVER_CONFIG, config_file)
    return config_path


def run_rtsp_server(config_path: str) -> None:
    command = (
        "docker run --rm --name rtsp_server -d -v "
        f"{config_path}:/rtsp-simple-server.yml -p 8554:8554 "
        "aler9/rtsp-simple-server:v1.3.0"
    )
    if run_command(command.split()) != 0:
        raise RuntimeError("Could not start the RTSP server!")


def stop_rtsp_server() -> None:
    run_command("docker kill rtsp_server".split())


def stream_videos(video_files: list) -> None:
    threads = []
    for index, video_file in enumerate(video_files):
        stream_url = f"{BASE_STREAM_URL}{index}.stream"
        print(f"Streaming {video_file} under {stream_url}")
        thread = stream_video_to_url(video_file, stream_url)
        threads.append(thread)
    for thread in threads:
        thread.join()


def stream_video_to_url(video_path: str, stream_url: str) -> Thread:
    command = (
        f"ffmpeg -re -stream_loop -1 -i {video_path} "
        f"-f rtsp -rtsp_transport tcp {stream_url}"
    )
    return run_command_in_thread(command.split())


def run_command_in_thread(command: list) -> Thread:
    thread = Thread(target=run_command, args=(command,))
    thread.start()
    return thread


def run_command(command: list) -> int:
    process = subprocess.run(command)
    return process.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to stream videos using RTSP protocol."
    )
    parser.add_argument(
        "--video_directory",
        type=str,
        required=True,
        help="Directory containing video files to stream.",
    )
    parser.add_argument(
        "--number_of_streams",
        type=int,
        default=6,
        help="Number of video files to stream.",
    )
    arguments = parser.parse_args()
    main(
        video_directory=arguments.video_directory,
        number_of_streams=arguments.number_of_streams,
    )
