import sys
from io import StringIO


class StdoutCapturing(list):

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(sys.stdout.getvalue().splitlines())
        sys.stdout = self._stdout

    def write(self, msg):
        sys.stdout.write(msg)


class StderrCapturing(list):

    def __enter__(self):
        self._stderr = sys.stderr
        sys.stderr = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(sys.stderr.getvalue().splitlines())
        sys.stderr = self._stderr

    def write(self, msg):
        sys.stderr.write(msg)


class DevNULL(list):

    def __enter__(self):
        self._stdout, self._stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = StringIO(), StringIO()
        return self

    def __exit__(self, *args):
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def write(self, msg):
        pass
