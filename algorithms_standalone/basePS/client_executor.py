"""親所有の同期バリアと、上限付き spawn worker。CUDA を fork しない。"""
import multiprocessing as mp
from multiprocessing.connection import wait
import os
import time
import traceback
import math


def _worker(connection, handler, initializer, initargs):
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        if initializer is not None:
            initializer(*initargs)
        connection.send((True, None))
        while True:
            task = connection.recv()
            if task is None:
                break
            connection.send((True, handler(task)))
    except BaseException:
        connection.send((False, traceback.format_exc()))
    finally:
        connection.close()


class BoundedExecutor:
    """一度に worker 数まで送信し、全成功時だけ sampling 順の結果を返す。"""

    def __init__(self, workers, timeout, handler, initializer=None, initargs=()):
        if (type(workers) is not int or not 1 <= workers <= 2
                or type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("worker 数は 1..2、timeout は正数が必要です")
        self.timeout = timeout
        self.processes, self.connections = [], []
        self.closed = False
        self.initialized = False
        context = mp.get_context("spawn")
        try:
            for _ in range(workers):
                parent, child = context.Pipe()
                process = context.Process(target=_worker,
                                          args=(child, handler, initializer, initargs))
                process.start()
                child.close()
                self.processes.append(process)
                self.connections.append(parent)
        except BaseException:
            self.close()
            raise

    def run(self, tasks, observer=None):
        if self.closed:
            raise RuntimeError("終了済み executor は再利用できません")
        results = [None] * len(tasks)
        pending = {}
        next_index = 0

        def submit(connection):
            nonlocal next_index
            if next_index < len(tasks):
                pending[connection] = (next_index, time.monotonic() + self.timeout)
                connection.send(tasks[next_index])
                next_index += 1

        try:
            if not self.initialized:
                waiting = set(self.connections)
                deadline = time.monotonic() + self.timeout
                while waiting:
                    if observer is not None:
                        observer([p.pid for p in self.processes])
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("client worker の起動が制限時間を超えました")
                    for connection in wait(list(waiting), timeout=min(remaining, 0.25)):
                        try:
                            ok, error = connection.recv()
                        except (EOFError, OSError) as exc:
                            raise RuntimeError("client worker が起動中に終了しました") from exc
                        if not ok:
                            raise RuntimeError("client worker の初期化失敗:\n" + error)
                        waiting.remove(connection)
                self.initialized = True
            for connection in self.connections:
                submit(connection)
            while pending:
                if observer is not None:
                    observer([p.pid for p in self.processes])
                remaining = min(deadline for _, deadline in pending.values()) - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("client worker が制限時間を超えました")
                ready = wait(list(pending), timeout=min(remaining, 0.25))
                for connection in ready:
                    index, _ = pending.pop(connection)
                    try:
                        ok, result = connection.recv()
                    except (EOFError, OSError) as exc:
                        raise RuntimeError("client worker が結果を返さず終了しました") from exc
                    if not ok:
                        raise RuntimeError("client worker 失敗:\n" + result)
                    results[index] = result
                    submit(connection)
            return results
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.closed:
            return
        self.closed = True
        # 失敗時も待機中も確実に回収する。学習の自動再試行は行わない。
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
        for connection in self.connections:
            connection.close()
