from flask import Flask, render_template, jsonify, request
from data_fetch import fetch_data
from calculations import calculate_indicators
from sorting_filtering import sort_all_columns
import logging
from datetime import datetime

app = Flask(__name__)

logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

last_updated = None
cached_results = None
current_timeframe = '5m'

def get_results(timeframe='5m'):
    """
    Verileri çeker, hesaplar ve sıralar, timeframe parametresiyle.
    """
    global last_updated, cached_results, current_timeframe
    try:
        data = fetch_data(timeframe)
        results = calculate_indicators(data)
        sorted_results = sort_all_columns(results)
        last_updated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cached_results = sorted_results
        current_timeframe = timeframe
        logging.info(f"Veriler başarıyla güncellendi, timeframe: {timeframe}")
        return sorted_results
    except Exception as e:
        logging.error(f"Veri güncelleme hatası: {str(e)}")
        return cached_results if cached_results else []

@app.route('/')
def index():
    """
    Ana sayfa, sıralı tabloyu gösterir.
    """
    timeframe = request.args.get('timeframe', current_timeframe)
    results = get_results(timeframe)
    return render_template('index.html', results=results, last_updated=last_updated, current_timeframe=timeframe)

@app.route('/update')
def update():
    """
    Verileri günceller ve JSON döndürür, timeframe parametresiyle.
    """
    timeframe = request.args.get('timeframe', current_timeframe)
    results = get_results(timeframe)
    return jsonify({'results': results, 'last_updated': last_updated, 'timeframe': timeframe})

@app.route('/test')
def test():
    """
    Test endpoint'i.
    """
    return jsonify({'status': 'Test başarılı', 'last_updated': last_updated, 'timeframe': current_timeframe})

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)