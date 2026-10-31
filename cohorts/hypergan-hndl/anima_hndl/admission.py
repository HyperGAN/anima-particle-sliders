"""Candidate-only admission of regular, content-bound local input files.

These checks are not a filesystem sandbox. The operator must keep admitted
directories immutable while the model/provider subsequently reads their paths.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import stat

from lumen_studio.contracts import digest


def _relative(name):
    if not isinstance(name, str) or not name or '\\' in name:
        raise ValueError('admitted file names must be relative local paths')
    path = Path(name)
    if path.is_absolute() or any(part in ('.', '..') for part in path.parts):
        raise ValueError('admitted file path escapes its directory')
    return path


def _identity(value):
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


@contextmanager
def _regular(root, name):
    """Do not follow symlink components or block while opening a special file."""
    if os.open not in os.supports_dir_fd or not hasattr(os, 'O_NOFOLLOW'):
        raise RuntimeError('candidate file admission requires descriptor-relative no-follow support')
    root = Path(root)
    if root.is_symlink():
        raise ValueError('admitted directory must not be a symlink')
    relative = _relative(name)
    directories = []
    try:
        parent = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        directories.append(parent)
        for part in relative.parts[:-1]:
            parent = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            directories.append(parent)
        fd = os.open(relative.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        with os.fdopen(fd, 'rb') as file:
            before = os.fstat(file.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError('admitted input must be a regular file: ' + name)
            yield file
            after = os.stat(relative.name, dir_fd=parent, follow_symlinks=False)
            if _identity(before) != _identity(os.fstat(file.fileno())) or _identity(before) != _identity(after):
                raise ValueError('admitted input changed while being read: ' + name)
    except OSError as exc:
        raise ValueError('admitted file unavailable or contains a symlink: ' + name) from exc
    finally:
        for fd in reversed(directories):
            os.close(fd)


def _read(root, name, limit, identities=None):
    with _regular(root, name) as file:
        data = file.read(limit + 1)
        if len(data) > limit:
            raise ValueError('admitted metadata exceeds its byte limit: ' + name)
        if identities is not None:
            identities[name] = _identity(os.fstat(file.fileno()))
        return data


def _trusted_model():
    config = Path(__file__).resolve().parents[1] / 'configs/anima'
    return (json.loads((config / 'converted-model.lock.json').read_text()),
            json.loads((config / 'released-identity.json').read_text()))


def _model_layout(root, expected):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('model requires a regular local directory')
    directories = {str(parent) for name in expected for parent in _relative(name).parents if str(parent) != '.'}
    actual = set()
    for path in root.rglob('*'):
        name = str(path.relative_to(root))
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError('model layout forbids symlinks: ' + name)
        if stat.S_ISDIR(mode):
            if name not in directories:
                raise ValueError('model layout contains an unadmitted directory: ' + name)
        elif stat.S_ISREG(mode):
            actual.add(name)
        else:
            raise ValueError('model layout requires regular files: ' + name)
    if actual != expected:
        raise ValueError('model file set differs from the committed conversion manifest')


def _portable_index(value, root):
    if isinstance(value, list):
        return [_portable_index(item, root) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if key != 'pretrained_model_name_or_path':
            result[key] = _portable_index(item, root)
            continue
        # Diffusers forwards this literal, rather than interpreting it relative
        # to the pipeline directory. Never normalize an ambiguous CWD/Hub name.
        if not isinstance(item, str) or not Path(item).is_absolute() or item != str(root):
            raise ValueError('model component path must equal the exact absolute admitted root')
        result[key] = '$MODEL_ROOT'
    return result


def verify_model_directory(directory):
    """Bind the entire staged model to committed conversion/released identities.

    The local anima-lock supplies no authority. No extra shard index, symlink,
    FIFO, different weight digest, external component path or revision is admitted.
    """
    if not isinstance(directory, (str, os.PathLike)):
        raise ValueError('model_dir must name the committed local model layout')
    root = Path(directory)
    trusted, released = _trusted_model()
    expected = set(trusted['files']) | {'anima-lock.json'}
    _model_layout(root, expected)
    root = root.resolve()
    identities = {'.': _identity(root.stat())}
    index_bytes = _read(root, 'modular_model_index.json', 1024 * 1024, identities)
    recorded = json.loads(_read(root, 'anima-lock.json', 64 * 1024, identities))
    local_expected = deepcopy(trusted)
    local_expected['files']['modular_model_index.json'] = hashlib.sha256(index_bytes).hexdigest()
    if recorded != local_expected:
        raise ValueError('model lock differs from the committed conversion manifest')
    portable = deepcopy(trusted)
    portable['files']['modular_model_index.json'] = digest(
        _portable_index(json.loads(index_bytes), root))
    if trusted['model'] != released['model'] or digest(portable) != released['sha256']:
        raise ValueError('model index differs from the committed portable model identity')
    for name, expected_hash in trusted['files'].items():
        if name == 'modular_model_index.json':
            continue
        with _regular(root, name) as file:
            if hashlib.file_digest(file, 'sha256').hexdigest() != expected_hash:
                raise ValueError('model file differs from the committed digest: ' + name)
            identities[name] = _identity(os.fstat(file.fileno()))
    return dict(model=released['model'], sha256=released['sha256'],
                files_stat_sha256=digest(identities))


def verified_cache_index(options):
    """Bind selected index bytes; CachedResponses separately hashes all shards."""
    directory = options.get('cache_dir')
    if not isinstance(directory, str):
        raise ValueError('cache_dir must name a local directory')
    data = _read(directory, 'index.json', 16 * 1024 * 1024)
    if hashlib.sha256(data).hexdigest() != options.get('cache_index_sha256'):
        raise ValueError('Anima cache index differs from the configured SHA256')
    index = json.loads(data)
    for shard in index['shards']:
        with _regular(directory, shard['path']):
            pass  # Type/path admission only. CachedResponses verifies content.
    return index
