from flask import Flask, send_file, jsonify, json
import requests

# app = Flask(__name__)

cid = 'QmQ4PSw6MBDb3U4ZMH6A1R2EqGXJXFxLXqvruqSi76PRt6'

# @app.route('/ipfs/<cid>')
def fetch_ipfs_file(cid):
    ipfs_gateway_url = f"https://ipfs.io/ipfs/{cid}"
    response = requests.get(ipfs_gateway_url, stream=True)

    if response.status_code == 200:
        return response.text
    else:
        return "Error fetching file from IPFS"

print(fetch_ipfs_file(cid))

# if __name__ == '__main__':
#     app.run(debug=True)