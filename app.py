from flask import Flask, jsonify, request
from aster_dex_trading import AsterDEXClient

app = Flask(__name__)

@app.route('/setCredentials', methods=['POST'])
def setCredentials():
    global client
    key = request.args.get('key')
    secret = request.args.get('secret')
    client = AsterDEXClient(key, secret)
    return jsonify({'success': True, 'message': 'Credentials set successfully'})

@app.route('/ping')
def ping():
    return jsonify({'status': 'alive from A'})

@app.route('/')
def home():
    return "Hello, World!"

@app.route('/marketOrder', methods=['POST'])
def marketOrder():
    side = request.args.get('side')
    symbol = request.args.get('symbol')
    qty = request.args.get('qty')
    res = client.place_market_order(symbol, side, qty)
    return jsonify(res)

@app.route('/accInfo', methods=['GET'])
def accInfo():
    info = client.get_account_info()
    return jsonify(info)

@app.route('/syncTime', methods=['GET'])
def syncTime():
    return client._sync_time()

@app.route('/posInfo', methods=['GET'])
def posInfo():
    symbol = request.args.get('symbol')
    return client.get_position_info(symbol)

@app.route('/chLeverage', methods=['POST'])
def chLeverage():
    leverage = int(request.args.get('leverage'))
    symbol = request.args.get('symbol')
    res = client.change_leverage(symbol, leverage)
    return jsonify(res)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
