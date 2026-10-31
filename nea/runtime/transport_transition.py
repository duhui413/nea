"""Explicit reviewed transport changes without rewriting logical experiment identity."""
from pathlib import Path
import json
from nea.providers import config
from nea.runtime.records import ConflictError, canonical_bytes, digest, write_once
from nea.runtime.batch_migration import source_files


def execution_policy(*, mimo_api=None, mimo_batch=False, mimo_concurrency=None,
                     mimo_batch_size=5, minimax_concurrency=None,
                     minimax_m3_api=None, minimax_relay_concurrency=None, mimo_official_concurrency=None):
    policy = {'mimo_api':config.resolve_mimo_api(mimo_api,mimo_batch=mimo_batch),
              'mimo_concurrency':config.mimo_concurrency(mimo_concurrency, config.resolve_mimo_api(mimo_api,mimo_batch=mimo_batch)),
              'mimo_batch_size':config.batch_size(mimo_batch_size),
              'minimax_concurrency':config.minimax_concurrency(minimax_concurrency)}
    if config.mimo_official_concurrency(mimo_official_concurrency):
        policy['mimo_official_concurrency'] = config.mimo_official_concurrency(mimo_official_concurrency)
    if config.resolve_minimax_m3_api(minimax_m3_api) in config.MINIMAX_RELAY_MODE_MODELS:
        policy.update(minimax_m3_api=config.resolve_minimax_m3_api(minimax_m3_api),
                      minimax_relay_concurrency=config.minimax_relay_concurrency(minimax_relay_concurrency))
    return policy


def pool_policy(pool):
    return execution_policy(mimo_api=pool.mimo_api,mimo_concurrency=pool.mimo_concurrency,
                            mimo_batch_size=pool.batch_policy['max_requests'],
                            minimax_concurrency=pool.minimax_concurrency,
                            minimax_m3_api=pool.minimax_m3_api,
                            minimax_relay_concurrency=pool.minimax_relay_concurrency,
                            mimo_official_concurrency=pool.mimo_official_concurrency or None)


def verify(batch, policy=None):
    path = batch/'supervision/transport-transition.json'
    if not path.is_file():
        raise ConflictError('Explicit transport transition registration is required')
    receipt = json.loads(path.read_text(encoding='utf8'))
    if receipt.get('source_files') != source_files() or (
            policy is not None and receipt.get('execution_policy') != policy):
        receipt = successor(batch, receipt, policy)
    predecessor = batch/'supervision/batch-migration.json'
    if digest(json.loads(predecessor.read_text(encoding='utf8'))) != receipt['previous_registration_hash']:
        raise ConflictError('Original Batch registration differs')
    for rel, value in receipt['manifests'].items():
        if digest((batch/rel).read_bytes()) != value['sha256']:
            raise ConflictError('Original manifest differs')
    for rel, expected in receipt['batch_inputs'].items():
        if digest((batch/rel).read_bytes()) != expected:
            raise ConflictError('Existing Batch input differs')
    return receipt


def successor(batch, base, policy=None):
    """A reviewed later registration of the same transition: exact new source and transport."""
    current, found = source_files(), []
    for path in sorted((batch/'supervision/transport-successors').glob('*.json')):
        value = json.loads(path.read_text(encoding='utf8'))
        if (value.get('predecessor_hash') == digest(base) and value.get('source_files') == current
                and (policy is None or value.get('execution_policy') == policy)):
            found.append(value)
    if len(found) != 1:
        raise ConflictError('Transport transition source or parameters differ without a registered successor')
    return {**base, 'source_files': found[0]['source_files'], 'execution_policy': found[0]['execution_policy'],
            'successor_hash': digest(found[0])}


def bind(root, identity, router):
    batch = root.parents[2]
    receipt = verify(batch,pool_policy(router.pool))
    rel = (root/'run_manifest.json').relative_to(batch).as_posix()
    locked = receipt['manifests'].get(rel)
    if locked is None:
        return identity
    old = json.loads((root/'run_manifest.json').read_text(encoding='utf8'))
    logical = locked['logical_model_identity']
    if digest(logical) != old['model_config_hash']:
        raise ConflictError('Original logical model identity differs')
    actual = router.model_identity
    candidate = dict(identity)
    for field in ('method_source_hash','model_config_hash','mimo_transport','mimo_batch_source_hash'):
        if field in old: candidate[field] = old[field]
        else: candidate.pop(field,None)
    if old.get('split') == 'case' and old.get('run_id') != candidate.get('run_id'):
        imported=json.loads((batch/'case-import.json').read_text(encoding='utf8'))
        if imported['files'].get(rel) != locked['sha256']:
            raise ConflictError('Imported case provenance differs')
        candidate['run_id'] = old['run_id']
    if canonical_bytes(candidate) != canonical_bytes(old):
        raise ConflictError('Transport transition changes non-transport inputs')
    router.logical_model_identity = logical
    router.resume_mimo_batches = not router.mimo_batch and any((root/'batches/jobs').glob('*/input.json'))
    record={'transition_hash':digest(receipt),'logical_identity_hash':digest(old),
            'actual_model_identity':actual,'execution_policy':receipt['execution_policy'],
            'existing_batch_policy':'resume-only; never duplicate submitted/unknown tasks'}
    write_once(root/'transport_migrations'/(digest(record)+'.json'),record)
    return old
