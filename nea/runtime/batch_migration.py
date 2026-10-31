"""Explicit transport migration; historical logical identities remain immutable."""
import json
from pathlib import Path

from nea.runtime.records import ConflictError, canonical_bytes, digest, write_once
from nea.runtime.identity import model_config_identity


def source_files():
    root = Path(__file__).resolve().parents[2]
    return {p.relative_to(root).as_posix(): digest(p.read_bytes())
            for folder in ('nea', 'prompts', 'scripts')
            for p in sorted((root / folder).rglob('*'))
            if p.is_file() and p.suffix in {'.py', '.md'}}


def verify(batch):
    path = batch / 'supervision' / 'batch-migration.json'
    if not path.exists():
        raise ConflictError('Batch migration registration missing')
    receipt = json.loads(path.read_text(encoding='utf8'))
    current_sources = source_files()
    if receipt['source_files'] != current_sources:
        matches = []
        for extension in (batch / 'supervision' / 'source-migrations').glob('mimo-api-*.json'):
            value = json.loads(extension.read_text(encoding='utf8'))
            if (value.get('original_registration_hash') == digest(receipt)
                    and value.get('source_files') == current_sources
                    and value.get('batch_policy') == receipt['batch_policy']):
                matches.append(value)
        if len(matches) != 1:
            raise ConflictError('Batch migration source differs')
    from nea.providers import config
    if receipt['batch_policy'] != config.MIMO_BATCH_POLICY:
        raise ConflictError('Batch migration policy differs')
    for rel, expected in receipt['manifests'].items():
        if digest((batch / rel).read_bytes()) != expected:
            raise ConflictError('Batch migration original manifest differs')
    return receipt


def bind(root, identity, router):
    batch = root.parents[2]
    if (batch / 'supervision' / 'transport-transition.json').exists():
        from nea.runtime.transport_transition import bind as bind_transition
        return bind_transition(root, identity, router)
    receipt_path = batch / 'supervision' / 'batch-migration.json'
    if not getattr(router, 'mimo_batch', False) or not receipt_path.exists():
        return identity
    receipt = verify(batch)
    manifest = root / 'run_manifest.json'
    rel = manifest.relative_to(batch).as_posix()
    if rel not in receipt['manifests']:
        return identity
    old = json.loads(manifest.read_text(encoding='utf8'))
    logical = model_config_identity(identity['generator'])
    if old['model_config_hash'] != digest(logical):
        logical = legacy_mimo_identity(identity['generator'])
    if old['model_config_hash'] != digest(logical):
        raise ConflictError('Migration changes logical model configuration')
    candidate = dict(identity)
    candidate.pop('mimo_transport', None)
    candidate.pop('mimo_batch_source_hash', None)
    candidate.update(method_source_hash=old['method_source_hash'], model_config_hash=old['model_config_hash'])
    if old.get('split') == 'case':
        imported = json.loads((batch / 'case-import.json').read_text(encoding='utf8'))
        if imported['files'].get(rel) != digest(manifest.read_bytes()):
            raise ConflictError('Migration case provenance differs')
        candidate['run_id'] = old['run_id']
    if canonical_bytes(candidate) != canonical_bytes(old):
        raise ConflictError('Migration changes non-transport identity')
    router.logical_model_identity = logical
    transport_record = {'registration_hash': digest(receipt), 'logical_identity_hash': digest(old),
                        'current_source_hash': digest(source_files()),
                        'actual_transport': 'mimo-batch/minimax-realtime',
                        'actual_model_identity': model_config_identity(identity['generator'], mimo_batch=True)}
    # Each reviewed implementation gets a new receipt; never overwrite the
    # original migration receipt when its public execution identity changes.
    write_once(root / 'transport_migrations' / (digest(transport_record) + '.json'), transport_record)
    return old


def legacy_mimo_identity(generator):
    """Frozen 0026 logical identity, solely for explicitly registered Batch runs."""
    from nea.providers import config
    logical = model_config_identity(generator)
    logical['execution_policy'].pop('mimo_api', None)
    logical['execution_policy'].pop('mimo_rate_policy', None)
    logical['execution_policy'].pop('mimo_concurrency', None)
    for value in logical.values():
        if isinstance(value, dict) and value.get('provider') == 'mimo':
            value.update(endpoint=config.MIMO_TOKEN_PLAN_ENDPOINT,
                         credential_endpoints=[config.MIMO_TOKEN_PLAN_ENDPOINT, config.MIMO_STANDARD_ENDPOINT])
    return logical
