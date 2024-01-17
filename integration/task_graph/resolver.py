# Copyright 2024 InferLink Corporation

from copy import deepcopy
from pathlib import Path
import re
from typing import Any, Optional

from yaml import load, FullLoader


class Resolver:
    def __init__(self, context: dict, extras: Optional[dict] = None) -> None:
        self._context = deepcopy(context)
        if extras:
            for k, v in extras.items():
                if k in self._context:
                    raise Exception(f"environment variable {k} already exists in resolver context")
                self._context[k] = v

    def resolve(self, obj: Any) -> Any:
        if obj is None:
            return obj
        if type(obj) in [int, bool, float]:
            return str(obj)
        if type(obj) is str:
            return self._resolve_str(obj)
        if type(obj) is list:
            return self._resolve_list(obj)
        if type(obj) is dict:
            return self._resolve_dict(obj)
        raise Exception(f"resolver: unsupported type: {type(obj)}")

    def _resolve_dict(self, obj: dict[str, str]) -> dict[str, str]:
        return {k: self.resolve(v) for k, v in obj.items()}

    def _resolve_str(self, obj: str) -> str:
        return self._interpolate(obj)

    def _resolve_list(self, obj: list[Any]) -> list[Any]:
        return [self.resolve(i) for i in obj]

    def _lookup(self, text: str, context: dict) -> str:
        tokens = text.split(".")
        new_context = context[tokens[0]]
        if len(tokens) > 1:
            text = ".".join(tokens[1:])
            return self._lookup(text, new_context)
        assert type(new_context) is str
        return new_context

    def _interpolate(self, text: str) -> str:
        pattern = re.compile(r"\{[A-Za-z0-9_.]+}")
        m = pattern.findall(text)
        if not m:
            return text

        var = m[0][1:-1]
        try:
            repl = self._lookup(var, self._context)
        except KeyError:
            raise Exception(f'unable to interpolate variable "{var}" in string "{text}"')

        # do the replacement, then see if we have to do any more replacements
        text = text.replace(m[0], repl)
        return self._interpolate(text)


if __name__ == "__main__":
    def main():
        s = Path("config.yaml").read_text()
        data = load(s, Loader=FullLoader)
        r = Resolver(data, extras={"JOB_ID": "1234"})
        d = r.resolve(data)
        print(d)

    main()
