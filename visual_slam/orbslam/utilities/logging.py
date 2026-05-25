"""
Minimal logging helpers.
This module exposes a small print-based logger with color-style method names.
"""

from __future__ import annotations


# Expose a tiny print-based logger with severity-style method names.
class Printer:
    @staticmethod
    def red(*args, **kwargs):
        print(*args, **kwargs)

    @staticmethod
    def green(*args, **kwargs):
        print(*args, **kwargs)

    @staticmethod
    def blue(*args, **kwargs):
        print(*args, **kwargs)

    @staticmethod
    def orange(*args, **kwargs):
        print(*args, **kwargs)

    @staticmethod
    def purple(*args, **kwargs):
        print(*args, **kwargs)

    @staticmethod
    def cyan(*args, **kwargs):
        print(*args, **kwargs)

    @staticmethod
    def yellow(*args, **kwargs):
        print(*args, **kwargs)
