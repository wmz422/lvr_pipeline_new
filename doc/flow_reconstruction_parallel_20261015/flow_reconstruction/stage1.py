from lvr.stage1 import StageOneCLI
from lvr.training.diagnostics import diagnostic_session
from .data import FlowDataModule
from .teacher import FlowVisualBottleneck

class FlowStageCLI(StageOneCLI):
    def add_arguments_to_parser(self,parser):
        super().add_arguments_to_parser(parser)
        parser.link_arguments('model.flow_protocol_sha256','data.flow_protocol_sha256')

def main():
    with diagnostic_session():
        FlowStageCLI(FlowVisualBottleneck,FlowDataModule,seed_everything_default=42,
                     save_config_kwargs={'overwrite':True})

if __name__=='__main__':main()
