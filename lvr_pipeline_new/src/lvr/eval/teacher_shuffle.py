"""Resolve frozen teacher-latent donors without admitting target helpers or labels."""


def teacher_record(record, lookup, donors, *, shuffled=False):
    if not shuffled:
        if not record["helper_images"]:
            raise ValueError("Teacher evaluation requires helper images")
        return record, None
    donor_id = donors[record["id"]]["id"]
    if donor_id == record["id"] or donor_id not in lookup:
        raise ValueError("Teacher-shuffled requires a distinct known donor")
    donor = lookup[donor_id]
    if not donor["helper_images"]:
        raise ValueError("Teacher-shuffled donor has no helper images")
    if donor["benchmark"] != record["benchmark"] or donor["task"] != record["task"]:
        raise ValueError("Teacher-shuffled donor must preserve benchmark and task")
    return donor, donor_id
