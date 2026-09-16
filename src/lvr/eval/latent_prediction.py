"""Four controlled inputs to the same final student B, with crop-free prediction."""

from collections import defaultdict
import random

import torch

from lvr.eval.vlmeval import BottleneckModel
from lvr.models.latent_prediction import LatentPrediction


def shuffled_sources(annotations, seed=42):
    """Whole-sample derangement within each benchmark, independent of rank/order."""
    groups = defaultdict(list)
    for identity, row in annotations.items():
        groups[row["benchmark"]].append(identity)
    result = {}
    for benchmark, identities in sorted(groups.items()):
        if len(identities) < 2:
            raise ValueError(f"Shuffling needs at least two samples in {benchmark}")
        identities.sort()
        random.Random(f"{seed}:{benchmark}").shuffle(identities)
        result.update(zip(identities, identities[1:] + identities[:1]))
    return result


class PredictedLatentModel(BottleneckModel):
    latent_modes = ("teacher", "predicted", "zero", "shuffled")
    encoder_modes = ("teacher",)

    def load_model(self, checkpoint):
        self.student = LatentPrediction.from_checkpoint(checkpoint, "cuda:0",
                                                       source_checkpoint=self.source_checkpoint,
                                                       base_model_path=self.base_model_path)
        network = self.student.network
        network.checkpoint_step = self.student.checkpoint_step
        return network

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.mode == "shuffled" and self.annotations is None:
            raise ValueError("Shuffled evaluation requires the complete fixed annotation manifest")
        self.sources = shuffled_sources(self.annotations, self.seed) if self.mode == "shuffled" else {}

    def get_latent(self, inputs, sample_id):
        if self.mode == "zero":
            return super().get_latent(inputs, sample_id)
        if self.mode == "teacher":
            return self.model.encode(inputs["encoder"])[0]
        if "encoder" in inputs:
            raise AssertionError("Predicted evaluation must not construct crop/teacher inputs")
        qa = inputs["qa"]
        if self.mode == "shuffled":
            source_id = self.sources[sample_id]
            row = self.annotations[source_id]
            # Re-predict a different sample using only its original images and prompt.
            # No stored target answers, descriptions or crop images enter this path.
            donor = {"id": source_id, "images": row["images"], "question": row["qa_prompt"], "answer": ""}
            donor_batch = self.collator([donor])
            if "encoder" in donor_batch:
                raise AssertionError("Shuffled source must also be crop-free")
            qa = {key: value.to("cuda:0") for key, value in donor_batch["qa"].items()}
        return self.student.predict_latents(qa)

    @torch.inference_mode()
    def generate_inner(self, message, dataset=None):
        result = super().generate_inner(message, dataset)
        record = result["extra_records"]
        record.update(model_kind="latent_prediction", qa_input_policy="originals_question_and_latents",
                      prediction_input_policy="originals_and_question_only", kd_weight=0)
        if self.mode == "shuffled":
            record["shuffled_source_id"] = self.sources[record["sample_id"]]
        return result
