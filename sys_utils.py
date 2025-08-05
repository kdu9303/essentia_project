import signal
import sys
import atexit
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

# 전역 프로세스 풀 참조를 위한 변수
_process_pool: Optional[ProcessPoolExecutor] = None


def signal_handler(sig, frame):
    """
    SIGINT (Ctrl+C) 신호를 받았을 때 실행되는 핸들러
    프로세스 풀을 안전하게 종료합니다.

    Python의 signal.signal() 함수가 요구하는 표준 시그널 핸들러 시그니처:
    - sig와 frame 인자는 Python이 자동으로 전달하는 필수 매개변수
    - 함수에서 사용하지 않더라도 시그니처에 반드시 포함되어야 함

    Args:
        sig: 시그널 번호 (예: SIGINT는 2)
        frame: 시그널 발생 시점의 스택 프레임 정보
    """
    print(f"\n프로그램 종료 신호를 받았습니다... (signal: {sig})")
    print("안전한 종료 절차를 시작합니다...")
    try:
        cleanup_processes()
        print("모든 프로세스가 안전하게 종료되었습니다.")
    except Exception as e:
        print(f"종료 중 오류 발생: {e}")
    finally:
        print("프로그램을 종료합니다.")
        sys.exit(0)


def cleanup_processes():
    """
    실행 중인 프로세스들을 안전하게 정리하는 함수
    """
    global _process_pool
    if _process_pool is not None:
        print("실행 중인 프로세스들을 종료하는 중...")
        try:
            # 1단계: 정상적인 종료 시도
            print("정상 종료를 시도합니다...")
            import threading
            
            # ProcessPoolExecutor가 유효한지 확인
            if hasattr(_process_pool, 'shutdown'):
                # 별도 스레드에서 shutdown 실행하여 타임아웃 구현
                shutdown_complete = threading.Event()
                shutdown_error = None
                
                def shutdown_worker():
                    nonlocal shutdown_error
                    try:
                        if _process_pool is not None:
                            _process_pool.shutdown(wait=True)
                        shutdown_complete.set()
                    except Exception as e:
                        shutdown_error = e
                        shutdown_complete.set()
                
                shutdown_thread = threading.Thread(target=shutdown_worker)
                shutdown_thread.start()
                
                # 5초 대기
                if shutdown_complete.wait(timeout=5.0):
                    if shutdown_error:
                        raise shutdown_error
                    print("프로세스들이 정상적으로 종료되었습니다.")
                else:
                    print("정상 종료 타임아웃 (5초 초과)")
                    raise TimeoutError("Shutdown timeout")
            else:
                print("ProcessPoolExecutor가 이미 종료된 상태입니다.")
                raise ValueError("ProcessPoolExecutor already closed")
        except Exception as e:
            print(f"정상 종료 실패: {e}")
            try:
                # 2단계: 강제 종료 시도
                print("강제 종료를 시도합니다...")
                import os
                import signal as sig_module
                
                # ProcessPoolExecutor의 내부 프로세스들에 SIGTERM 전송
                for process in getattr(_process_pool, '_processes', {}).values():
                    if process.is_alive():
                        print(f"프로세스 {process.pid} 강제 종료 중...")
                        try:
                            os.kill(process.pid, sig_module.SIGTERM)
                        except (OSError, ProcessLookupError):
                            pass  # 이미 종료된 프로세스
                
                # 최종 강제 종료
                _process_pool.shutdown(wait=False)
                print("강제 종료 완료")
            except Exception as cleanup_error:
                print(f"강제 종료 중 오류 발생: {cleanup_error}")
        finally:
            _process_pool = None
            print("프로세스 풀 정리 완료")


def set_process_pool(pool: ProcessPoolExecutor):
    """
    현재 실행 중인 프로세스 풀을 설정
    """
    global _process_pool
    _process_pool = pool


def clear_process_pool():
    """
    프로세스 풀 참조 해제
    """
    global _process_pool
    _process_pool = None


def setup_signal_handlers():
    """
    시그널 핸들러와 종료 시 정리 함수를 설정
    """
    # 시그널 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    # 프로그램 종료 시 자동 정리
    atexit.register(cleanup_processes)
