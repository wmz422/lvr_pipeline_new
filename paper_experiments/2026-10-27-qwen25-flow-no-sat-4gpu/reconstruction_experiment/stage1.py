from .data import FlowDataModule
from flow_reconstruction.stage1 import FlowStageCLI
from lvr.training.diagnostics import diagnostic_session
from .teacher import NoDescriptionFlowTeacher


def main():
    with diagnostic_session():
        FlowStageCLI(NoDescriptionFlowTeacher, FlowDataModule, seed_everything_default=42,
                     save_config_kwargs={"overwrite": True})


if __name__ == "__main__":
    main()
