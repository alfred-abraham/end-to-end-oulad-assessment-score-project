"""OULAD regression demo. Train artifacts before starting the app."""
from flask import Flask, render_template, request

from src.config import ARTIFACT_DIR
from src.pipeline.predict_pipeline import InputError, ModelUnavailable, get_predictor


def create_app(config=None):
    app = Flask(__name__)
    app.config.update(ARTIFACT_DIR=str(ARTIFACT_DIR), MAX_CONTENT_LENGTH=64 * 1024)
    if config:
        app.config.update(config)

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/predictdata', methods=['GET', 'POST'])
    def predict_datapoint():
        try:
            predictor = get_predictor(str(app.config['ARTIFACT_DIR']))
        except ModelUnavailable as exc:
            return render_template('home.html', error=str(exc), metadata=None,
                values=request.form, history_rows=[]), 503
        history_rows = []
        if request.method == 'POST':
            # Preserve even incomplete rows so validation errors do not erase user input.
            from itertools import zip_longest
            history_rows = [dict(zip(['assessment', 'score', 'submitted'], row))
                for row in zip_longest(request.form.getlist('prior_assessment_id'),
                    request.form.getlist('prior_score'), request.form.getlist('prior_submitted'), fillvalue='')]
        common = {'metadata': predictor.metadata, 'values': request.form,
                  'history_rows': history_rows, 'result': None, 'error': None}
        if request.method == 'POST':
            try:
                common['result'] = predictor.predict_form(request.form)
            except InputError as exc:
                common['error'] = str(exc)
                return render_template('home.html', **common), 400
            except Exception:
                app.logger.exception('Prediction failed')
                common['error'] = 'The prediction could not be completed. Please try again.'
                return render_template('home.html', **common), 500
        return render_template('home.html', **common)

    @app.errorhandler(413)
    def too_large(_error):
        return 'The submitted form is too large. Use at most 50 previous assessments.', 413

    return app


application = create_app()
app = application

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
