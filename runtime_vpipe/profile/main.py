#! /home/user/miniconda3/envs/tyf_py/bin/python
# pylint: disable=C0114,C0115,C0116

from itertools import combinations
import logging
import os
from typing import List
import jsonline
from nsys_profile import NsysProgram
from profile_task import ProfileTask, StragglerTask, TestTask


def get_python_files(directory) -> (list, ProfileTask):
    python_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_name = os.path.join(root, file)
            python_files.append(file_name)
    return python_files


def get_stragglers():
    return get_python_files("straggle")


def get_tests():
    return get_python_files("test")


def run_one(program: NsysProgram, test_name: str, task_batch_size: int):
    result = {}
    program.profile_for(10)
    result["profile"] = program.export()
    test = TestTask(test_name, task_batch_size)
    test.launch()
    ret = test.try_wait()
    if ret != 0:
        logging.fatal("Test %s failed with return code %s", test_name, ret)
        return None
    result["output"] = test.get_result()
    return result


def normalize_result(result, seconds: float):
    result["output"]["time"] /= seconds
    return result


def run(stragglers: List[List[str]], tests: List[str]):
    with jsonline.open("out1") as writer:
        program = NsysProgram.from_json("program.json")
        for test_name in tests:
            task_batch_size = 4
            while task_batch_size <= 64:

                # baseline
                baseline = run_one(program, test_name, task_batch_size)

                if baseline is not None:
                    running_time = baseline["output"]["time"]
                    writer.append(normalize_result(baseline, running_time))
                    for straggler_list in stragglers:
                        batch_size = 8
                        while batch_size <= 64:
                            straggler_tasks = [StragglerTask(
                                straggler, batch_size) for straggler in straggler_list]
                            for straggler_task in straggler_tasks:
                                straggler_task.launch()
                            # with straggler
                            result = run_one(
                                program, test_name, task_batch_size)
                            if result is not None:
                                writer.append(normalize_result(
                                    result, running_time))
                            for straggler_task in straggler_tasks:
                                straggler_task.wait()
                            batch_size *= 2
                task_batch_size *= 2


def run_combinition(stragglers: List[str], tests: List[str], length: int):
    comb = list(combinations(stragglers, length))
    run(comb, tests)


if __name__ == "__main__":
    stragglers_ = get_stragglers()
    tests_ = get_tests()
    run_combinition(stragglers_, tests_, 1)
    run_combinition(stragglers_, tests_, 2)
    run_combinition(stragglers_, tests_, 3)
