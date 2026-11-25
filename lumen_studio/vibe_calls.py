"""The saved lifecycle of every render, proposal and judge call in a vibe collection.

Each call is one record at ``<root>/calls/<key>.json`` in the format written by
the shared ``particle_sliders_vibe`` add-on, so free-text and named collections
read each other's receipts.

    ABSENT --begin--> PENDING --commit--> DONE
                        |  ^
                 reject |  | begin, automatically on the next run
                        v  |
                       FAILED

A record found PENDING at the start of a run was interrupted between dispatch
and receipt. Local work is repeated. A provider call has an unknown outcome
and waits for ``--retry-pending``. FAILED means the outcome is known and no
usable result exists, so the next run may try again.
"""
from enum import Enum
from pathlib import Path
from threading import Lock


class CallState(str, Enum):
    ABSENT = "absent"
    PENDING = "pending"
    DONE = "done"
    FAILED = "failed"


# Deterministic work on this machine. Repeating it spends no provider request.
LOCAL_KINDS = frozenset({"render"})


class UnresolvedCall(RuntimeError):
    """A provider call was dispatched and its outcome was never recorded."""

    def __init__(self, key):
        super().__init__("unresolved call " + key + "; reconcile the provider receipt before --retry-pending")
        self.key = key


class NotDispatched(RuntimeError):
    """The call stopped on this machine before any request was sent."""


class ProviderUnavailable(RuntimeError):
    """The provider answered with an explicit server error and no result."""

    def __init__(self, provider, status):
        super().__init__(f"{provider} HTTP {status}")
        self.provider, self.status = provider, status


class InvalidReply(ValueError):
    """The provider replied, and the reply cannot be used."""

    def __init__(self, reason, *, reply=None):
        super().__init__(reason)
        self.reply = reply


def temporary(error):
    """Whether a later run can reasonably expect this call to succeed."""
    from nanoodle.errors import NanoodleError, UnsupportedNodeError
    from particle_sliders_vibe import ProviderRejected

    if isinstance(error, ProviderRejected):
        return error.status == 429
    if isinstance(error, UnsupportedNodeError):
        return False
    return isinstance(error, (ProviderUnavailable, UnresolvedCall, NanoodleError))


class CallLedger:
    """Run each distinct call at most once and say what happened to it."""

    def __init__(self, root, *, retry_pending=False, archives=()):
        self.root, self.retry_pending = Path(root), retry_pending
        # Completed receipts in an archive are reused and never rewritten.
        self.archives = tuple(Path(archive) for archive in archives)
        self._locks = {}

    @staticmethod
    def key(kind, inputs):
        from particle_sliders_vibe import digest

        return digest(dict(kind=kind, inputs=inputs))

    def _read(self, root, key):
        from particle_sliders_vibe import read_json

        path = root / "calls" / (key + ".json")
        if not path.exists():
            return CallState.ABSENT, None
        record = read_json(path)
        return CallState(record["status"]), record

    def state(self, kind, inputs):
        return self._read(self.root, self.key(kind, inputs))[0]

    def unresolved(self):
        """Request IDs of provider calls whose outcome is unknown."""
        from particle_sliders_vibe import read_json

        records = (read_json(path) for path in sorted((self.root / "calls").glob("*.json")))
        return [record["request_id"] for record in records
                if record["status"] == CallState.PENDING.value and record["kind"] not in LOCAL_KINDS]

    def _write(self, key, state, call, **details):
        from particle_sliders_vibe import write_json

        write_json(self.root / "calls" / (key + ".json"), dict(status=state.value, **call, **details))

    @staticmethod
    def _saved(result, check):
        from particle_sliders_vibe import file_digest

        if isinstance(result, dict) and "path" in result and file_digest(result["path"]) != result["sha256"]:
            raise ValueError("saved output changed: " + result["path"])
        if check is not None:
            check(result)
        return result

    def __call__(self, kind, inputs, callback, *, check=None):
        """Return the saved result, or run ``callback(key)`` and commit it.

        ``check`` raises InvalidReply for a reply that arrived and is unusable.
        """
        from particle_sliders_vibe import ProviderRejected, canonical, file_digest

        key = self.key(kind, inputs)
        with self._locks.setdefault(key, Lock()):
            state, record = self._read(self.root, key)
            if state is CallState.DONE:
                try:
                    return self._saved(record["result"], check)
                except InvalidReply:
                    # An unusable reply saved as complete is a known failure.
                    state = CallState.FAILED
            for archive in self.archives:
                archived, record = self._read(archive, key)
                if archived is CallState.DONE and record["inputs"] == inputs:
                    return self._saved(record["result"], check)
            if state is CallState.PENDING and kind not in LOCAL_KINDS and not self.retry_pending:
                raise UnresolvedCall(key)
            call = dict(kind=kind, inputs=inputs, request_id=key)
            self._write(key, CallState.PENDING, call)
            result = None
            try:
                result = callback(key)
                canonical(result)
                if isinstance(result, dict) and "path" in result:
                    result = dict(result, path=str(Path(result["path"]).resolve()))
                    result["sha256"] = file_digest(result["path"])
                if check is not None:
                    check(result)
            except (ProviderRejected, ProviderUnavailable) as error:
                self._write(key, CallState.FAILED, call, provider=error.provider, http_status=error.status)
                raise
            except NotDispatched as error:
                self._write(key, CallState.FAILED, call, reason=str(error), dispatched=False)
                raise
            except InvalidReply as error:
                # Keep the paid reply beside the reason it was refused.
                self._write(key, CallState.FAILED, call, reason=str(error),
                            reply=result if error.reply is None else error.reply)
                raise
            self._write(key, CallState.DONE, call, result=result)
            return result
