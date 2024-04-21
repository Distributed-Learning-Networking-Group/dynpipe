#! /home/user/miniconda3/envs/tyf_py/bin/python
# pylint: disable=C0114,C0115,C0116

from dataclasses import dataclass
import json
from logging import info
from os import remove
from signal import SIGINT
import sqlite3
import subprocess
from time import sleep
from typing import List

from rich import print_json


@dataclass
class NsysProgram:
    command: str
    output: str = "out"
    overwrite: bool = False
    args: List[str] = None

    def try_remove(self, filename: str):
        if self.overwrite:
            try:
                remove(filename)
            except FileNotFoundError:
                pass

    def profile_for(self, seconds: float):
        nsys_process = self.profile()
        sleep(seconds)
        self.stop(nsys_process)

    def profile(self):
        self.try_remove(f"{self.output}.nsys-rep")
        cmd = [self.command]
        if self.args is not None:
            cmd.extend(self.args)
        cmd.extend(["-o", self.output])
        return subprocess.Popen(cmd)

    def stop(self, nsys_process: subprocess.Popen):
        nsys_process.send_signal(SIGINT)
        nsys_process.wait()

    def export(self):
        self.try_remove(f"{self.output}.sqlite")
        cmd = [self.command]
        cmd.extend(["export", "-t", "sqlite", f"{self.output}.nsys-rep"])
        subprocess.run(cmd, check=True)
        con = sqlite3.connect(f"{self.output}.sqlite")
        cur = con.cursor()
        cur.execute(
            "select data from GENERIC_EVENTS where rawTimestamp = (select MAX(rawTimestamp) from GENERIC_EVENTS) LIMIT 1"
        )
        result_str = cur.fetchone()[0]
        return json.loads(result_str)

    @staticmethod
    def from_json(filename: str):
        with open(filename, encoding="utf-8") as fp:
            settings = json.load(fp)
        program = NsysProgram(**settings)
        return program


if __name__ == "__main__":
    program_ = NsysProgram.from_json("program.json")

    while True:
        program_.profile_for(5)
        print_json(json.dumps(program_.export()))
