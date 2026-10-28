"""Student profiling with explicit long-input stress updates; never used for formal training."""
import json
from pathlib import Path
import yaml
from torch.utils.data import DistributedSampler
from lvr.data.length_manifest import load_plain_qa_lengths
from lvr.models.latent_prediction import LatentPrediction
from lvr.predict_latents import PredictionCLI
from lvr.training.diagnostics import diagnostic_session
from reconstruction_experiment.data import StudentDataModule
ROOT=Path(__file__).resolve().parents[1]
class StressSampler(DistributedSampler):
    def __init__(self,original,dataset,indices,update_size):
        super().__init__(dataset,num_replicas=original.num_replicas,rank=original.rank,shuffle=True,seed=original.seed,drop_last=original.drop_last)
        self.original=original;self.stress_indices=indices;self.update_size=update_size
        self.set_epoch(original.epoch)
    def set_epoch(self,epoch):
        super().set_epoch(epoch);self.original.set_epoch(epoch)
    def __iter__(self):
        order=list(self.original)
        for update,index in enumerate(self.stress_indices):
            begin=update*self.update_size;end=min(begin+self.update_size,len(order));order[begin:end]=[index]*(end-begin)
        return iter(order)
class ProfileStudentDataModule(StudentDataModule):
    def _loader(self,dataset,training):
        loader=super()._loader(dataset,training)
        if training:
            known={item['id']:i for i,item in enumerate(dataset.items)}
            checks=json.loads((ROOT/'runs/offline_vae/cpu_checks/validation.json').read_text())['processor_samples']
            choices=[r for r in checks if r['id'] in known]
            defaults=yaml.safe_load((ROOT/'configs/experiment.yaml').read_text())
            plain=yaml.safe_load(Path(defaults['plain_recipe']).read_text())['data']
            lengths=load_plain_qa_lengths(plain['train_length_manifest_path'],dataset.items,
                self.hparams.processor_name_or_path,self.hparams.original_pixels,plain['max_length'])
            qa=max(range(len(lengths)),key=lengths.__getitem__)
            encoder=known[max(choices,key=lambda r:r['encoder_tokens'])['id']]
            self.probe_stress_ids=[dataset.items[i]['id'] for i in (qa,encoder)]
            batch_sampler=loader.batch_sampler
            batch_sampler.sampler=StressSampler(batch_sampler.sampler,dataset,[qa,encoder],batch_sampler.batch_size*batch_sampler.accumulation)
        return loader

def main():
    with diagnostic_session():
        PredictionCLI(LatentPrediction,ProfileStudentDataModule,seed_everything_default=42,save_config_kwargs={'overwrite':True})
if __name__=='__main__':main()
