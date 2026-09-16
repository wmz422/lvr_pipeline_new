# Fixed latent-prediction benchmark

- `datasets.json`: sample counts and categories, total 3,662.
- `data_config.json`: exact VLMEvalKit sample subsets.
- `crop_annotations.json`: exact prompts and ordered boxes used by the teacher condition.

The upstream VLMEvalKit commit is pinned in `scripts/latent_prediction/install_vlmeval.sh`. Images are resolved relative to `LMUData` (default `benchmark/LMUData` in the evaluation launcher), never relative to the original author's workspace. Recorded sample IDs are deliberately preserved so that `shuffled` uses exactly the original donor permutation after moving to another directory/machine.

`predicted`, `zero`, and `shuffled` do not construct A/crop model inputs. `shuffled` predicts the donor's latents online from its original images and question; the donor is in the same benchmark and differs from the target. `teacher` uses original+all-crops A with the same final B as the other three conditions. No reference answer enters generation.

The protocol includes annotated boxes, some from answer-conditioned annotations. See each row's `annotation_source` and the root README for interpretation. Downloads use the pinned upstream dataset definitions/checksums; changing datasets or prompts is a different evaluation protocol.
