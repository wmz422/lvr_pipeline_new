"""Allow only declared, independently checked second-attempt pilot caption repairs."""
import hashlib
from caption import CONFIG_ID, RETRY_CONFIG_ID


def validate_pilot_subset(previous, current, errata, rejected_caption_hashes, manifest_sha256, withdrawals=None):
    if errata:
        if errata.get('schema') != 'lvr-pilot-caption-errata-v1' or errata.get('release') != 'reasoning_v0':
            raise ValueError('Unknown pilot caption errata format')
        if errata['manifest_sha256'] != manifest_sha256:
            raise ValueError('Pilot errata refers to a different release')
    entries = {entry['id']: entry for entry in errata.get('records', [])}
    if len(entries) != len(errata.get('records', [])) or not entries.keys() <= previous.keys():
        raise ValueError('Invalid pilot errata IDs')
    withdrawals = withdrawals or {}
    if withdrawals and (withdrawals.get('schema') != 'lvr-pilot-caption-withdrawals-v1'
                        or withdrawals.get('release') != 'reasoning_v0'
                        or withdrawals.get('manifest_sha256') != manifest_sha256):
        raise ValueError('Pilot withdrawals refer to a different release or format')
    removed = {entry['id']: entry for entry in withdrawals.get('records', [])}
    if len(removed) != len(withdrawals.get('records', [])) or not removed.keys() <= entries.keys():
        raise ValueError('Pilot withdrawal lacks declared erratum')
    for sid, entry in removed.items():
        if sid in current:
            raise ValueError('Withdrawn pilot QA was retained: ' + sid)
        old = previous[sid]
        if any(entry[key] != old[key] for key in ('source', 'split')):
            raise ValueError('Pilot withdrawal identity differs: ' + sid)
        original_sha = hashlib.sha256(old['description'].encode()).hexdigest()
        if entry['original_description_sha256'] != original_sha or entries[sid]['original_description_sha256'] != original_sha:
            raise ValueError('Pilot withdrawal original caption differs: ' + sid)
        failed = entry.get('failed_parts', [])
        if not failed or len({part['step'] for part in failed}) != len(failed):
            raise ValueError('Pilot withdrawal lacks distinct failed helper evidence')
        declared = {part['step']: part for part in entries[sid]['affected_parts']}
        for part in failed:
            before = next((p for p in old['description_parts'] if p['step'] == part['step']), None)
            if before is None or part['step'] not in declared:
                raise ValueError('Pilot withdrawal helper was not declared')
            for key in ('step', 'image_sha256', 'caption_sha256'):
                if part[key] != before[key] or declared[part['step']][key] != before[key]:
                    raise ValueError('Pilot withdrawal does not match original helper')
            retry = part['retry_caption']
            retry_sha = hashlib.sha256(retry['text'].encode()).hexdigest()
            if (before['generation_config_id'] != CONFIG_ID or before['generation_attempt'] != 1
                    or retry['config_id'] != RETRY_CONFIG_ID or retry['attempt'] != 2
                    or retry['image_sha256'] != before['image_sha256']
                    or part['retry_caption_sha256'] != retry_sha):
                raise ValueError('Pilot withdrawal does not prove the two fixed attempts')
            human = part['human_rejection']
            if (before['caption_sha256'] not in rejected_caption_hashes or retry_sha not in rejected_caption_hashes
                    or human.get('status') != 'rejected' or human.get('id') != sid
                    or human.get('step') != before['step'] or human.get('generation_attempt') != 2
                    or human.get('generation_config_id') != RETRY_CONFIG_ID
                    or human.get('image_sha256') != before['image_sha256'] or human.get('caption_sha256') != retry_sha):
                raise ValueError('Pilot withdrawal lacks rejected caption evidence')
    corrections = []
    preserved = ('source', 'source_revision', 'split', 'question', 'answer', 'answer_type',
                 'options', 'problem_images', 'auxiliary_images', 'teacher_view_mode', 'source_image_id')
    for sid, old in previous.items():
        if sid in removed:
            continue
        if sid not in current:
            raise ValueError('Pilot subset lost a record: ' + sid)
        new = current[sid]
        for key in preserved:
            if old.get(key) != new.get(key):
                raise ValueError('Pilot QA content changed: ' + sid + ':' + key)
        if sid not in entries:
            if old['description'] != new['description']:
                raise ValueError('Undeclared pilot caption change: ' + sid)
            continue
        entry = entries[sid]
        if entry['original_description_sha256'] != hashlib.sha256(old['description'].encode()).hexdigest():
            raise ValueError('Pilot erratum original caption differs: ' + sid)
        if entry['split'] != old['split'] or len(old['description_parts']) != len(new['description_parts']):
            raise ValueError('Pilot erratum changed split or helper count: ' + sid)
        affected = {part['step']: part for part in entry['affected_parts']}
        if not affected or len(affected) != len(entry['affected_parts']):
            raise ValueError('Empty or duplicate pilot erratum parts')
        changes = []
        for before, after in zip(old['description_parts'], new['description_parts']):
            step = before['step']
            if step not in affected:
                if before != after:
                    raise ValueError('Unaffected pilot caption part changed: ' + sid)
                continue
            declared = affected[step]
            for key in ('step', 'image_sha256', 'caption_sha256'):
                if declared[key] != before[key]:
                    raise ValueError('Erratum does not match original part: ' + sid)
            if before['caption_sha256'] not in rejected_caption_hashes:
                raise ValueError('Caption repair lacks rejection evidence: ' + sid)
            if after['step'] != step or after['image_sha256'] != before['image_sha256']:
                raise ValueError('Pilot correction changed helper identity: ' + sid)
            if after['generation_config_id'] != RETRY_CONFIG_ID or after['generation_attempt'] != 2:
                raise ValueError('Pilot repair did not use the fixed second attempt: ' + sid)
            if after['quality_status'] != 'passed' or after['caption_sha256'] == before['caption_sha256'] or after['caption_sha256'] in rejected_caption_hashes:
                raise ValueError('Pilot correction remains unapproved: ' + sid)
            changes.append({'step': step, 'image_sha256': before['image_sha256'],
                            'old_caption_sha256': before['caption_sha256'], 'new_caption_sha256': after['caption_sha256']})
        if len(changes) != len(affected):
            raise ValueError('Erratum refers to missing helper parts: ' + sid)
        corrections.append({'id': sid, 'split': old['split'], 'corrected_parts': changes})
    return {'status': 'resolved', 'records': sorted(corrections, key=lambda row: row['id']),
            'withdrawn_records': sorted(removed), 'original_pilot_records': len(previous),
            'required_retained_records': len(previous) - len(removed),
            'scope': 'All still-qualified pilot QA identities, answers, images and splits preserved. Only declared incorrect captions may be replaced with approved fixed second attempts. Entire QA withdrawal requires explicit evidence that both allowed attempts were rejected.'}
