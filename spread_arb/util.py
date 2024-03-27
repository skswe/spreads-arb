"""This module contains general utility functions.
"""

import datetime
import functools
import hashlib
import inspect
import logging
import os
import pickle
import sys
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


def format_dict(dict_, ind: str = "", trail: str = "_"):
    """
    Return a string representation of a dict with correct indentation.

    Args:
        dict_: The dict to format.
        ind: The indentation string (default: "").
        trail: The trail string to use when formatting dict keys (default: "_").

    Returns:
        The formatted string representation of the dict.
    """
    out = "{\n"
    n = max(len(str(x)) for x in dict_.keys()) if len(dict_) > 0 else 0
    for k, v in dict_.items():
        out += "\t" + ind + f"{k}:".ljust(n + 2, trail)
        if isinstance(v, dict):
            out += format_dict(v, ind + "\t")
        else:
            out += str(v) + "\n"

    out += ind + "}\n"
    return out


def make_timestamped_dir(path: str) -> str:
    """
    Create a timestamped directory within the given path.

    Args:
        path: The base path to create the directory in.

    Returns:
        The full path of the created timestamped directory.
    """
    time_now = datetime.datetime.now()
    date_now = time_now.date().strftime("%Y-%m-%d")
    hour_now = time_now.strftime("%H")
    minute_now = time_now.strftime("%M")
    full_log_dir = os.path.join(path, date_now, hour_now, minute_now)
    os.makedirs(full_log_dir, exist_ok=True)
    return full_log_dir


def unique_file_name(filename: str) -> str:
    """
    Generate a unique file name by appending a suffix to the given path.

    Args:
        path: The base path of the file.

    Returns:
        The unique file name with the suffix.
    """
    if not os.path.exists(filename):
        return filename

    filename, ext = os.path.splitext(filename)

    if "_" in filename and filename.split("_")[-1].isdigit():
        contains_i = True
        i = int(filename.split("_")[-1])
    else:
        contains_i = False
        i = 2

    while os.path.exists(filename + ext):
        if contains_i:
            filename = "_".join(filename.split("_")[:-1] + [str(i)])
        else:
            filename = filename + "_" + str(i)
        i += 1

    return filename + ext


def redirect_stdout(path: str):
    """
    Decorator to redirect standard output to a file.

    Args:
        path: The path to the file where the output should be redirected.

    Returns:
        The decorated function.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                if os.path.dirname(path):
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                ref = sys.stdout
                sys.stdout = open(path, "w")
                return func(*args, **kwargs)
            finally:
                sys.stdout = ref

        return wrapper

    return decorator


class cached:
    @classmethod
    def log(cls, level: str, *msg) -> None:
        """
        Log a message at the specified log level.

        Args:
            level: The log level.
            *msg: The message to be logged.
        """
        logger.log(logging._nameToLevel[level], *msg)

    @classmethod
    def _sort_dict(cls, d: dict, reverse: bool = False) -> dict:
        """
        Sort a dict based on keys recursively.

        Args:
            d: The dict to sort.
            reverse: Whether to sort in reverse order (default: False).

        Returns:
            The sorted dict.
        """
        new_d = {}
        sorted_keys = sorted(d, reverse=reverse)
        for key in sorted_keys:
            if isinstance(d[key], dict):
                new_d[key] = cls._sort_dict(d[key])
            else:
                new_d[key] = d[key]
        return new_d

    @classmethod
    def _hash_dict(cls, d: dict) -> str:
        """
        Hash a dictionary in an order-agnostic manner by ordering keys and hashing the values.

        Args:
            d: The dictionary to be hashed.

        Returns:
            The MD5 hash of the dictionary as a hexadecimal string.
        """
        key_hash = cls._hash_item(list(sorted(d.keys())))
        values = [cls._hash_item(i) for i in d.values()]
        return cls._hash_item([key_hash, sorted(values)])

    @classmethod
    def _hash_item(cls, i, strict=False):
        """
        Hash a Python object by pickling and then applying MD5 to resulting bytes.

        Args:
            i: The object to be hashed.
            strict: If True, raises a TypeError if unable to hash the object. If False, uses a less strict hashing method (default: False).

        Returns:
            The MD5 hash of the object as a hexadecimal string.

        Raises:
            TypeError: If strict is True and unable to hash the object.
        """
        if isinstance(i, dict):
            return cls._hash_dict(i)
        if strict:
            raise TypeError(
                f"Unable to hash {i} of type {type(i)}. To use a less strict hashing method, set strict=False"
            )
        try:
            hash = hashlib.md5(pickle.dumps(i)).hexdigest()
        except TypeError:
            logger.warning(f"Unable to hash {i}, using the hash of the object's class instead")
            hash = hashlib.md5(pickle.dumps(i.__class__)).hexdigest()
        except AttributeError:
            logger.warning(f"Unable to hash the object's class, using the hash of the class name instead")
            hash = hashlib.md5(pickle.dumps(i.__class__.__name__)).hexdigest()
        return hash

    @staticmethod
    def search(path: str, key: str) -> bool:
        """
        Search for a key in a directory.

        Args:
            path: The directory to search.
            key: The key to search for.

        Returns:
            Whether the key was found in the directory.
        """
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        return True if key in os.listdir(path) else False

    def __init__(
        self,
        path: str = "/tmp/cache",
        disabled: bool = False,
        refresh: bool = False,
        log_level: str = "INFO",
        identifiers: List = [],
        path_seperators: List = [],
        instance_identifiers: List = [],
        instance_path_seperators: List = [],
        load_fn=pd.read_pickle,
        save_fn=pd.to_pickle,
        search_fn=None,
        propagate_kwargs: bool = False,
        name: str = None,
    ):
        """
        Cache a function call. Function arguments are hashed to identify a unique function call.

        Args:
            path: Disk path to store cached objects. Defaults to "/tmp/cache".
            disabled: Whether to bypass the cache for the function call. Defaults to False.
            refresh: Whether to bypass cache lookup to force a new cache write. Defaults to False.
            log_level: Level to emit logs at. Defaults to "INFO".
            identifiers: Additional arguments that are hashed to identify a unique function call. Defaults to [].
            path_seperators: List of argument names to use as path separators after `path`.
            instance_identifiers: Names of instance attributes to include in `identifiers` if `is_method` is `True`. Defaults to [].
            instance_path_seperators: Names of instance attributes to include in `path_seperators` if `is_method` is `True`. Defaults to [].
            load_fn: Function to load cached data. Defaults to pd.read_pickle.
            save_fn: Function to save cached data. Defaults to pd.to_pickle.
            search_fn: Function ((path, key) -> bool) to override default search function. Defaults to os.listdir.
            propagate_kwargs: Whether to propagate keyword arguments to the decorated function. Defaults to False.
            name: Name of function or operation being cached. Defaults to None.
        """
        self.params = {
            "path": path,
            "disabled": disabled,
            "refresh": refresh,
            "log_level": log_level,
            "identifiers": identifiers.copy(),
            "path_seperators": path_seperators.copy(),
            "instance_identifiers": instance_identifiers.copy(),
            "instance_path_seperators": instance_path_seperators.copy(),
            "load_fn": load_fn,
            "save_fn": save_fn,
            "search_fn": search_fn or self.search,
            "propagate_kwargs": propagate_kwargs,
            "name": name,
        }

    def __call__(self, func):
        """
        Decorator to cache a function call.

        Args:
            func: The function to be cached.

        Returns:
            The wrapped function.
        """
        self.params["name"] = self.params["name"] or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            all_args = self._sort_dict(inspect.getcallargs(func, *args, **kwargs))

            # Update params using override passed in through calling function
            params = self.get_cache_params(kwargs)
            return self.perf_cache_lookup(func, params, all_args, *args, **kwargs)

        return wrapper

    def get_cache_params(self, kwargs) -> dict:
        """
        Get cache parameters.

        Args:
            kwargs: Keyword arguments.

        Returns:
            Cache parameters as a dictionary.
        """
        params = {}

        for k, v in self.params.items():
            if isinstance(v, list) or isinstance(v, dict):
                params[k] = v.copy()
            else:
                params[k] = v

        for k, v in kwargs.items():
            if k in params:
                params[k] = v

        if not params["propagate_kwargs"]:
            keys = list(kwargs.keys())
            for k in keys:
                if k in params:
                    del kwargs[k]

        if "cache_kwargs" in kwargs:
            params.update(kwargs["cache_kwargs"])
            if not params["propagate_kwargs"]:
                del kwargs["cache_kwargs"]

        return params

    @classmethod
    def perf_cache_lookup(cls, func, params, all_args, *args, **kwargs):
        """
        Perform the cache lookup and handle caching logic.

        Args:
            func: The function being cached.
            params: Cache parameters.
            all_args: All function arguments.
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            The result of the function call, either from cache or newly computed.
        """
        path = params["path"]
        disabled = params["disabled"]
        refresh = params["refresh"]
        log_level = params["log_level"]
        identifiers = params["identifiers"]
        path_seperators = params["path_seperators"]
        instance_identifiers = params["instance_identifiers"]
        instance_path_seperators = params["instance_path_seperators"]
        load_fn = params["load_fn"]
        save_fn = params["save_fn"]
        search_fn = params["search_fn"]
        name = params["name"]

        if disabled:
            return func(*args, **kwargs)

        path_seperators = [all_args[ps] for ps in path_seperators]
        if "self" in all_args:
            instance = all_args["self"]
            identifiers.extend([getattr(instance, id) for id in instance_identifiers])
            path_seperators.extend([getattr(instance, ps) for ps in instance_path_seperators])

        path = os.path.join(path, *path_seperators)

        hashable = {k: v for k, v in all_args.items() if k not in params and k not in ["cache_kwargs", "self"]}
        identifiers_hashed = [cls._hash_item(id) for id in identifiers]
        key = cls._hash_item([cls._hash_item(i) for i in [hashable, sorted(identifiers_hashed)]])

        argument_string = (*(f"(id:{i})" for i in identifiers), *(f"{k}={v}" for k, v in hashable.items()))
        argument_string = f"({', '.join(str(arg) for arg in argument_string)})"

        if not refresh and search_fn(path, key) is True:
            cls.log(
                log_level, f"Using cached value in call to {name}{argument_string} | key={key} ({path})"[:200] + "..."
            )
            return load_fn(os.path.join(path, key))
        else:
            data = func(*args, **kwargs)
            cls.log(
                log_level, f"Saving cached value in call to {name}{argument_string} | key={key} ({path})"[:200] + "..."
            )
            save_fn(data, os.path.join(path, key))
            return data


class BashFormatter:
    def __init__(self):
        self.enabled = True

    FG_CODES = {
        "none": "0",
        "black": "30",
        "gray": "90",
        "white": "97",
        "red": "91",
        "orange": "93",
        "green": "92",
        "cyan": "96",
        "blue": "94",
        "magenta": "95",
        "light_red": "31",
        "light_orange": "33",
        "light_green": "32",
        "light_cyan": "36",
        "light_blue": "34",
        "light_magenta": "35",
    }

    BG_CODES = {
        "none": "0",
        "black": "40",
        "gray": "100",
        "white": "107",
        "red": "101",
        "orange": "103",
        "green": "102",
        "cyan": "106",
        "blue": "104",
        "magenta": "105",
        "light_red": "41",
        "light_orange": "43",
        "light_green": "42",
        "light_cyan": "46",
        "light_blue": "44",
        "light_magenta": "45",
    }

    STYLE_CODES = {
        "none": "0",
        "bold": "1",
        "italic": "3",
        "underline": "4",
    }

    def disable(self):
        """
        Disable the bash formatting.
        """
        self.enabled = False

    def enable(self):
        """
        Enable the bash formatting.
        """
        self.enabled = True

    def _apply_fmt(self, code: str, text: str) -> str:
        """
        Apply the formatting code to the given text.

        Args:
            code: The formatting code.
            text: The text to be formatted.

        Returns:
            The formatted text.
        """
        return f"\u001b[{code}m{text}\u001b[0m"

    def color(self, text: str, fg_color: str = "none", bg_color: str = "none") -> str:
        """
        Apply color formatting to the given text.

        Args:
            text: The text to be formatted.
            fg_color: The foreground color (default: "none").
            bg_color: The background color (default: "none").

        Returns:
            The formatted text with color.
        """
        if not self.enabled:
            return text

        with_fg = self._apply_fmt(self.FG_CODES[fg_color], text)
        return self._apply_fmt(self.BG_CODES[bg_color], with_fg)

    def format(self, text: str, fg_color: str = "none", bg_color: str = "none", style: str = "none") -> str:
        """
        Apply formatting to the given text.

        Args:
            text: The text to be formatted.
            fg_color: The foreground color (default: "none").
            bg_color: The background color (default: "none").
            style: The style (default: "none").

        Returns:
            The formatted text.
        """
        if not self.enabled:
            return text

        with_color = self.color(text, fg_color, bg_color)
        return self._apply_fmt(self.STYLE_CODES[style], with_color)
