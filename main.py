from pathlib import Path
import multiprocessing
import polars as pl
from rich import print
from typing import List, Dict, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
import unicodedata
from classifiers import AudioClassifier
from utils import get_audio_files, cut_audio_file, chunk_list
from sys_utils import (
    setup_signal_handlers,
    set_process_pool,
    clear_process_pool,
    cleanup_processes,
)


CPU_CORES = multiprocessing.cpu_count()


def process_single_audio_file(audio_file: str, output_dir: str, duration: int) -> None:
    """
    단일 오디오 파일을 처리하는 워커 함수

    전처리 과정
    30, 45, 60초, 전구간 별 자를 것 -> dataframe 저장
    
    Args:
        audio_file: 처리할 오디오 파일 경로
        output_dir: 출력 디렉토리 경로
        duration: 자를 시간 길이 (초 단위)
    """
    print(f"현재 Process 중인 파일: {audio_file}")
    cut_audio_file(
        input_file=audio_file,
        output_dir=output_dir,
        start_time=0,
        duration=duration,
    )


def process_audio_chunk(audio_files: List[str], output_dir: str, duration: int) -> None:
    """
    오디오 파일 리스트를 청크 단위로 병렬 처리
    
    Args:
        audio_files: 처리할 오디오 파일 경로 리스트
        output_dir: 출력 디렉토리 경로
        duration: 자를 시간 길이 (초 단위)
    """
    if not audio_files:
        print("처리할 오디오 파일이 없습니다.")
        return

    workers = CPU_CORES - 2

    print(f"총 {len(audio_files)}개 파일을 {workers}개 프로세스로 처리합니다.")

    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            set_process_pool(executor)  # 전역 참조 저장

            # 모든 파일을 개별적으로 처리
            futures = [
                executor.submit(process_single_audio_file, audio_file, output_dir, duration)
                for audio_file in audio_files
            ]

            # 완료된 작업들 확인
            for future in as_completed(futures):
                try:
                    future.result()  # 예외 발생 시 여기서 처리됨
                except Exception as e:
                    print(f"파일 처리 중 오류 발생: {e}")
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다...")
        # KeyboardInterrupt 시에는 cleanup_processes()가 시그널 핸들러에서 호출됨
        raise
    finally:
        clear_process_pool()  # 참조 해제


def preprocess_single_file(audio_file: str) -> Dict[str, Union[str, float]]:
    """
    오디오 파일을 분석하고 결과에 파일명을 포함하여 반환
    """
    audio_titme = Path(audio_file).name

    classifier = AudioClassifier(audio_file)
    predict_results = classifier.predict_all(exclude_methods=["predict_jamendo_mood_and_theme"])

    results: Dict[str, Union[str, float]] = {
        "audio_title": audio_titme,
        **predict_results,
    }

    return results


def process_audio_analysis(
    audio_files: List[str],
) -> List[Dict[str, Union[str, float]]]:
    """
    오디오 파일 리스트를 ProcessPoolExecutor로 병렬 처리하여 분석 결과 반환

    :param audio_files: 분석할 오디오 파일 경로 리스트
    :return: 각 파일의 분석 결과가 담긴 딕셔너리 리스트
    """
    if not audio_files:
        print("처리할 오디오 파일이 없습니다.")
        return []

    workers = CPU_CORES - 3

    print(f"총 {len(audio_files)}개 파일을 {workers}개 프로세스로 분석합니다.")

    results = []

    try:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            set_process_pool(executor)  # 전역 참조 저장

            # 모든 파일을 개별적으로 처리
            futures = [
                executor.submit(preprocess_single_file, audio_file)
                for audio_file in audio_files
            ]

            # 완료된 작업들 확인 및 결과 수집
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    result = future.result()  # 분석 결과 받기
                    results.append(result)
                    print(
                        f"[{i}/{len(audio_files)}] 완료: {result.get('audio_title', 'Unknown')}"
                    )
                except Exception as e:
                    print(f"파일 분석 중 오류 발생: {e}")
                    continue
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단되었습니다...")
        # KeyboardInterrupt 시에는 cleanup_processes()가 시그널 핸들러에서 호출됨
        raise
    finally:
        clear_process_pool()  # 참조 해제

    print(f"총 {len(results)}개 파일 분석 완료")
    return results


def preprocess_audio_files():
    audio_files: List = get_audio_files(
            directory_path="audio/", is_path_form=True, extensions=[".mp3"]
        )

    # 오디오 파일 자르기 (전처리)
    process_audio_chunk(audio_files=audio_files, output_dir="audio_45sec/", duration=45)


def main():
    # 시그널 핸들러 설정
    setup_signal_handlers()

    try:
        audio_files: List = get_audio_files(
            "audio_45sec/", is_path_form=True, extensions=[".mp3"]
        )

        csv_file_path = "output/audio_analysis_45sec_results.csv"

        # output 디렉토리가 없으면 생성
        output_dir = Path(csv_file_path).parent
        output_dir.mkdir(exist_ok=True)

        file_exists = Path(csv_file_path).exists()

        if file_exists:
            print("기존 파일에 이어서 결과를 추가합니다.")
            # 기존 파일이 있으면 모든 청크를 upsert 모드로 처리
            is_first_chunk = False
        else:
            print(f"새로운 CSV 파일을 생성합니다: {csv_file_path}")
            is_first_chunk = True

        # 청크 나누기
        chunk_size = 50
        file_chunks = chunk_list(audio_files, chunk_size)
        total_chunks = len(file_chunks)

        for chunk_idx, chunk_files in enumerate(file_chunks):
            print(
                f"\n=== 청크 {chunk_idx + 1}/{total_chunks} 처리 중 ({len(chunk_files)}개 파일) ==="
            )

            try:
                # 청크 단위로 분석 수행
                analysis_results = process_audio_analysis(chunk_files)

                if analysis_results:
                    # 새로운 분석 결과를 DataFrame으로 생성
                    new_df = pl.DataFrame(analysis_results)
                    
                    # 한글 텍스트 컬럼들을 NFC 형태로 정규화 (Mac-Windows 호환성)
                    new_df = new_df.with_columns([
                        pl.col("audio_title").map_elements(
                            lambda x: unicodedata.normalize("NFC", x) if isinstance(x, str) else x,
                            return_dtype=pl.Utf8
                        )
                    ])
                    
                    # CSV 저장/upsert 로직
                    if is_first_chunk:
                        # 첫 번째 청크: 새 파일 생성 (헤더 포함)
                        with open(csv_file_path, "w", encoding="utf-8", newline="") as f:
                            new_df.write_csv(f, include_bom=True)
                        print(f"첫 번째 청크 결과를 {csv_file_path}에 저장했습니다.")
                        is_first_chunk = False  # 이후 모든 청크는 upsert 모드
                    else:
                        # 기존 CSV 파일 읽기
                        try:
                            existing_df = pl.read_csv(csv_file_path, encoding="utf-8")
                            
                            # 기존 데이터도 NFC 형태로 정규화 (Mac-Windows 호환성)
                            existing_df = existing_df.with_columns([
                                pl.col("audio_title").map_elements(
                                    lambda x: unicodedata.normalize("NFC", x) if isinstance(x, str) else x,
                                    return_dtype=pl.Utf8
                                )
                            ])
                            
                            # 1. 기존 데이터에서 새로운 데이터와 겹치지 않는 항목만 유지
                            updated_df = existing_df.join(
                                new_df.select("audio_title"), 
                                on="audio_title", 
                                how="anti"  # 새 데이터에 없는 기존 항목만 유지
                            )
                            
                            # 2. 기존 데이터와 새 데이터 결합 (새 데이터가 우선)
                            final_df = pl.concat([updated_df, new_df], how="vertical")
                            
                            # 3. audio_title로 정렬하여 일관성 유지
                            final_df = final_df.sort("audio_title")
                            
                            # 4. 결과를 CSV 파일에 저장 (덮어쓰기)
                            with open(csv_file_path, "w", encoding="utf-8", newline="") as f:
                                final_df.write_csv(f, include_bom=True)
                            print(
                                f"청크 {chunk_idx + 1} 결과를 {csv_file_path}에 upsert했습니다. "
                                f"(신규: {len(new_df)}개, 전체: {len(final_df)}개)"
                            )
                            
                        except Exception as e:
                            print(f"기존 CSV 파일 읽기 실패: {e}")
                            print("새로운 데이터만 추가합니다.")
                            # 기존 파일 읽기 실패 시 append 모드로 처리
                            with open(csv_file_path, "a", encoding="utf-8", newline="") as f:
                                new_df.write_csv(f,include_bom=True, include_header=False)
                            print(f"청크 {chunk_idx + 1} 결과를 {csv_file_path}에 추가했습니다.")

                    print(
                        f"청크 {chunk_idx + 1} 완료: {len(analysis_results)}개 파일 처리됨"
                    )
                else:
                    print(f"청크 {chunk_idx + 1}에서 분석 결과가 없습니다.")
                

            except KeyboardInterrupt:
                print("\n프로그램이 사용자에 의해 중단되었습니다.")
                cleanup_processes()
                return
            except Exception as e:
                print(f"청크 {chunk_idx + 1} 처리 중 오류 발생: {e}")
                print("다음 청크로 계속 진행합니다...")
                # continue

        # 최종 결과 확인
        try:
            final_df = pl.read_csv(csv_file_path, encoding="utf-8")
            
            # 최종 결과도 NFC 형태로 정규화 확인 (Mac-Windows 호환성)
            final_df = final_df.with_columns([
                pl.col("audio_title").map_elements(
                    lambda x: unicodedata.normalize("NFC", x) if isinstance(x, str) else x,
                    return_dtype=pl.Utf8
                )
            ])
            
            print(f"\n=== 최종 결과 ===")
            print(f"총 처리된 파일 수: {len(final_df)}")
            print(f"CSV 파일 저장 위치: {csv_file_path}")
            print(final_df.head())
        except Exception as e:
            print(f"최종 결과 확인 중 오류 발생: {e}")

    except KeyboardInterrupt:
        print("\n메인 프로그램이 사용자에 의해 중단되었습니다.")
        cleanup_processes()
    except Exception as e:
        print(f"메인 프로그램 실행 중 오류 발생: {e}")
        cleanup_processes()
    finally:
        cleanup_processes()  # 최종 정리


if __name__ == "__main__":
    # preprocess_audio_files()
    main()
