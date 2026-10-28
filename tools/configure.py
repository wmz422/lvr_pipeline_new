"""Materialize path-only settings for this checkout, without touching source data."""
import argparse
import json
from pathlib import Path
import sys

from common import ROOT, EXPERIMENT, QWEN, FLUX, SUITE


def configure(output=None):
    import yaml
    output = Path(output or ROOT / 'runs/flow').expanduser().resolve()
    directory = ROOT / '.local'
    directory.mkdir(exist_ok=True)
    text = (ROOT / 'configs/plain_reference.yaml').read_text().replace('${REPO}', str(ROOT))
    reference = directory / 'plain_reference.yaml'
    reference.write_text(text)
    defaults = yaml.safe_load((EXPERIMENT / 'configs/experiment.yaml').read_text())
    defaults.update(plain_recipe=str(reference), output=str(output),
                    flow_data=str(EXPERIMENT / 'data/reconstruction_2mp'),
                    flow_latents=str(EXPERIMENT / 'data/vae_latents_fp32'),
                    flow_model=str(FLUX), suite=str(SUITE))
    target = directory / 'flow.yaml'
    encoded = yaml.safe_dump(defaults, sort_keys=False)
    if target.exists() and target.read_text() != encoded:
        raise ValueError('Existing local configuration differs; use the runner --output flag for a new run')
    target.write_text(encoded)
    print(json.dumps({'root': str(ROOT), 'flow_config': str(target), 'output': str(output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    configure(parser.parse_args().output)

