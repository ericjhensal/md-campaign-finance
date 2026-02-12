"""Data management routes â upload, download from CRIS, run pipeline."""
import os
import threading
from flask import Blueprint, render_template, request, jsonify
from werkzeug.utils import secure_filename
from config import RAW_DIR

bp = Blueprint('data', __name__, url_prefix='/data')

# Pipeline status tracking
_pipeline_status = {'running': False, 'messages': [], 'error': None}
_download_status = {'running': False, 'messages': [], 'error': None}


def _classify_and_rename(original_name):
    """Auto-rename long CRIS filenames to simple names."""
    lower = original_name.lower()
    if 'committee' in lower and 'contribution' not in lower and 'expenditure' not in lower:
        return 'committees.csv'
    elif 'contribution' in lower or 'loan' in lower:
        return 'contributions.csv'
    elif 'expenditure' in lower:
        return 'expenditures.csv'
    else:
        return secure_filename(original_name)


@bp.route('/')
def index():
    # List files in raw directory
    files = []
    if os.path.exists(RAW_DIR):
        for f in sorted(os.listdir(RAW_DIR)):
            path = os.path.join(RAW_DIR, f)
            if os.path.isfile(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                files.append({'name': f, 'size': f'{size_mb:.1f} MB'})
    return render_template('data/index.html', files=files,
                         pipeline_running=_pipeline_status['running'],
                         download_running=_download_status['running'])


@bp.route('/upload', methods=['POST'])
def upload():
    """Upload CSV files."""
    uploaded = []
    for key in request.files:
        file = request.files[key]
        if file.filename:
            new_name = _classify_and_rename(file.filename)
            path = os.path.join(RAW_DIR, new_name)
            file.save(path)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            uploaded.append(f'{file.filename} â {new_name} ({size_mb:.1f} MB)')

    if uploaded:
        return jsonify({'status': 'ok', 'files': uploaded})
    return jsonify({'status': 'error', 'message': 'No files uploaded'}), 400


@bp.route('/download-cris', methods=['POST'])
def download_cris():
    """Download data from Maryland CRIS API."""
    if _download_status['running']:
        return jsonify({'status': 'error', 'message': 'Download already in progress'}), 409

    data = request.get_json() or {}
    year = data.get('year', None)  # None = full cycle
    types = data.get('types', ['committees', 'contributions', 'expenditures'])
    run_pipeline = data.get('run_pipeline', True)

    def do_download():
        _download_status['running'] = True
        _download_status['messages'] = []
        _download_status['error'] = None

        try:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from download_data import download_all

            def log(msg):
                _download_status['messages'].append(msg)

            results = download_all(year=year, types=types, progress_callback=log)
            log(f"Downloaded {len(results)} file(s)")

            if run_pipeline and results:
                log("Starting data pipeline...")
                _pipeline_status['running'] = True
                _pipeline_status['messages'] = []
                from pipeline.load import run_full_load

                def pipe_log(msg):
                    _pipeline_status['messages'].append(msg)
                    _download_status['messages'].append(msg)

                run_full_load(progress_callback=pipe_log)
                _pipeline_status['running'] = False

        except Exception as e:
            _download_status['error'] = str(e)
            _download_status['messages'].append(f'ERROR: {e}')
        finally:
            _download_status['running'] = False
            _pipeline_status['running'] = False

    thread = threading.Thread(target=do_download, daemon=True)
    thread.start()
    return jsonify({'status': 'started'})


@bp.route('/run-pipeline', methods=['POST'])
def run_pipeline():
    """Run the data loading pipeline."""
    if _pipeline_status['running']:
        return jsonify({'status': 'error', 'message': 'Pipeline already running'}), 409

    def do_load():
        _pipeline_status['running'] = True
        _pipeline_status['messages'] = []
        _pipeline_status['error'] = None
        try:
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from pipeline.load import run_full_load
            run_full_load(progress_callback=lambda msg: _pipeline_status['messages'].append(msg))
        except Exception as e:
            _pipeline_status['error'] = str(e)
            _pipeline_status['messages'].append(f'ERROR: {e}')
        finally:
            _pipeline_status['running'] = False

    thread = threading.Thread(target=do_load, daemon=True)
    thread.start()
    return jsonify({'status': 'started'})


@bp.route('/status')
def status():
    """Poll pipeline/download status."""
    return jsonify({
        'pipeline': {
            'running': _pipeline_status['running'],
            'messages': _pipeline_status['messages'][-20:],
            'error': _pipeline_status['error'],
        },
        'download': {
            'running': _download_status['running'],
            'messages': _download_status['messages'][-20:],
            'error': _download_status['error'],
        },
    })
