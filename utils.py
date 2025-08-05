import os
from pathlib import Path
from typing import List, Union, Optional


def chunk_list(data: List, chunk_size: int) -> List[List]:
    """
    리스트를 청크 단위로 분할하는 함수
    """
    return [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]


def get_audio_files(
    directory_path: Union[str, Path],
    is_path_form: bool = False,
    extensions: Optional[List[str]] = None,
) -> List[str]:
    """
    지정된 폴더에서 오디오 파일 목록을 반환합니다.

    Args:
        directory_path: 검색할 폴더 경로
        is_path_form: True면 "directory_path/filename" 형태로, False면 전체 경로로 반환
        extensions: 검색할 파일 확장자 리스트 (기본값: 일반적인 오디오 확장자)

    Returns:
        오디오 파일의 경로 리스트
    """
    if extensions is None:
        extensions = [".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma"]

    directory_path = Path(directory_path)

    if not directory_path.exists():
        raise FileNotFoundError(f"폴더를 찾을 수 없습니다: {directory_path}")

    if not directory_path.is_dir():
        raise ValueError(f"지정된 경로가 폴더가 아닙니다: {directory_path}")

    audio_files = []

    for file_path in directory_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            if is_path_form:
                # "directory_path/filename" 형태로 반환
                audio_files.append(f"{directory_path}/{file_path.name}")
            else:
                # 전체 경로로 반환
                file_name = str(Path(file_path).name)
                audio_files.append(file_name)

    return sorted(audio_files)


def cut_audio_file(
    input_file: str,
    output_dir: str,
    start_time: int,
    duration: int,
):
    """
    FFmpeg-python을 사용한 오디오 자르기

    Args:
        input_file: 입력 오디오 파일 경로
        start_time: 시작 시간 (초)
        duration: 지속 시간 (초)
    """
    import ffmpeg

    if not Path(input_file).exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_file}")

    if not Path(output_dir).exists():
        os.makedirs(output_dir, exist_ok=True)

    extention_type = input_file.split(".")[-1]
    file_name = (Path(input_file).name).replace(f".{extention_type}", "")

    end_time = start_time + duration
    output_filename = f"{output_dir}/{file_name}__{end_time}sec.{extention_type}"

    # 지정된 구간 자르기
    (
        ffmpeg.input(input_file, ss=start_time, t=duration)  # ss=시작시간, t=지속시간
        .output(output_filename)
        .overwrite_output()
        .run(quiet=True)
    )

    print(f"저장 완료: {output_filename}")
    return output_filename


if __name__ == "__main__":
    audio_files = get_audio_files("output_30sec/", extensions=[".mp3"])

    print(f"총 {len(audio_files)}개의 오디오 파일을 찾았습니다.")
    # output_dir = "output/"

    # for audio in audio_files[:5]:
    #     print(f"현재 Process 중인 파일: {audio}")
