from .file_collector import FileCollector

__all__ = ["TwitterCollector", "FileCollector"]


def __getattr__(name: str):
    if name == "TwitterCollector":
        from .twitter_collector import TwitterCollector
        return TwitterCollector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
