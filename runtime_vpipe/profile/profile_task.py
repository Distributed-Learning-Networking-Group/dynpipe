# pylint: disable=C0114,C0115,C0116

import os
from signal import SIGKILL
from subprocess import PIPE, Popen, TimeoutExpired
import sys

import logging
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)


class ProfileTask:

    def __init__(self, cmd: str, batch_size: int) -> None:
        self._cmd = cmd
        self._batch_size = batch_size
        self._process = None

    def launch(self):
        cmds = [sys.executable, self._cmd, "-b", str(self._batch_size)]
        logging.info("launch:%s", cmds)
        self._process = Popen(cmds)

    def wait(self):
        assert self._process is not None
        self._process.wait()
        assert self._process.returncode == 0

    def try_wait(self):
        assert self._process is not None
        self._process.wait()
        return self._process.returncode


class StragglerTask(ProfileTask):

    def launch(self):
        cmds = [sys.executable, self._cmd, "-b", str(self._batch_size)]
        logging.info("launch:%s", cmds)
        null_file = open(os.devnull, "w")
        self._process = Popen(cmds, stdout=null_file, stderr=null_file)

    def wait(self):
        assert self._process is not None
        done = False
        while not done:
            self._process.send_signal(SIGKILL)
            try:
                self._process.wait(1.0)
                if not (self._process.returncode == -SIGKILL or self._process.returncode == 0):
                    logging.fatal(
                        "unexpectd return code for %s, return code: %s",
                        self._cmd, self._process.returncode
                    )
                    # assert 0
                done = True
            except TimeoutExpired:
                pass


class TestTask(ProfileTask):

    def launch(self):
        cmds = [sys.executable, self._cmd, "-b", str(self._batch_size)]
        logging.info("launch:%s", cmds)
        self._process = Popen(cmds, stdout=PIPE)

    def get_result(self):
        lines = self._process.stdout.readlines()[-3:]

        assert len(lines) == 3

        # magic here
        result = {}

        line = lines[0]
        vals = [float(x) for x in line.split()]
        assert (len(vals) == 2)
        result["mac"] = vals[0]
        result["params"] = vals[1]

        line = lines[1]
        vals = [float(x) for x in line.split()]
        assert (len(vals) == 4)
        result["count_relu"] = vals[0]
        result["count_linear"] = vals[1]
        result["count_Conv2d"] = vals[2]
        result["count_MaxPool2d"] = vals[3]

        line = lines[2]
        vals = [float(x) for x in line.split()]
        assert (len(vals) == 1)
        result["time"] = vals[0]

        return result
