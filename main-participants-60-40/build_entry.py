"""Build an upload-ready code ZIP from a starter run or your own source folder."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.schema import ValidationError
from slot_forecasting.submission_contract import package_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--starter', type=Path, help='Starter output folder with baseline.json and run.json')
    choice.add_argument('--source', type=Path, help='Your code folder with reproduce.json and README.md at its root')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.source:
            result = package_code(args.source, args.out)
        else:
            FrequencyForecaster.load(args.starter / 'baseline.json')
            report = json.loads((args.starter / 'run.json').read_text(encoding='utf-8'))
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='.entry-', dir=args.out.parent) as temporary:
                root = Path(temporary).resolve()
                if not root.is_relative_to(args.out.parent.resolve()):
                    raise ValidationError('Entry staging folder escaped its parent.')
                source = root / 'code'
                source.mkdir()
                module = source / 'slot_forecasting'
                module.mkdir()
                import slot_forecasting
                installed = Path(slot_forecasting.__file__).parent
                for name in ('__init__.py', 'baseline.py', 'schema.py', 'io.py'):
                    shutil.copyfile(installed / name, module / name)
                shutil.copyfile(args.starter / 'baseline.json', source / 'baseline.json')
                shutil.copyfile(args.starter / 'run.json', source / 'training-run.json')
                (source / 'predict.py').write_text(
                    "import argparse\nfrom pathlib import Path\n"
                    "from slot_forecasting.baseline import FrequencyForecaster\n"
                    "from slot_forecasting.io import load_inputs, write_submission\n"
                    "p=argparse.ArgumentParser(); p.add_argument('--data'); p.add_argument('--out'); a=p.parse_args()\n"
                    "m=FrequencyForecaster.load(Path(__file__).parent/'baseline.json')\n"
                    "q,h=load_inputs(Path(a.data)/'test')\nwrite_submission(a.out,m.predict(q,h))\n", encoding='utf-8')
                (source / 'reproduce.json').write_text(json.dumps({'image': 'python', 'command': ['python', 'predict.py', '--data', '/data', '--out', '/output/submission.csv']}, indent=2) + '\n', encoding='utf-8')
                (source / 'README.md').write_text(
                    '# Starter submission\n\nPython 3.11+, standard library only.\n'
                    'The reproduction command loads the included baseline.json and predicts on /data/test.\n'
                    'Run: python predict.py --data /data --out /output/submission.csv\n'
                    'Training command: python run_starter.py --method ' + report['settings']['method'] + ' --out outputs\n'
                    'Training settings, input hashes, and environment: training-run.json.\n'
                    'Forecast CSV is uploaded separately and is not included in this archive.\n', encoding='utf-8')
                result = package_code(source, args.out)
        print(f'Code ZIP: {result}\nUpload this ZIP and the matching test forecast CSV as separate files.')
        return 0
    except (ValidationError, OSError, ValueError, KeyError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
