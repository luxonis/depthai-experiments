import json
import traceback
import urllib.parse
from json import JSONDecodeError


def setup_datachannel(pc, pc_id, app):
    @pc.on("datachannel")
    def on_datachannel(channel):
        app.pcs_datachannels[pc_id] = channel

        @channel.on("message")
        def on_message(message):
            try:
                unquoted = urllib.parse.unquote(message)
                data = json.loads(unquoted)
                if data['type'].upper() == 'PING':
                    channel.send(json.dumps({
                        'type': 'PONG'
                    }))
                elif data['type'].upper() == 'STREAM_CLOSED':
                    channel.send(json.dumps({
                        "type": "CLOSED_SUCCESSFUL",
                        "payload": "Channel is closing..."
                    }))
                    channel.close()
                else:
                    channel.send(json.dumps({
                        "type": "BAD_REQUEST",
                        "payload": {
                            "message": "Unknown action type " + data['type'],
                            "received": message,
                        }
                    }))
            except (JSONDecodeError, TypeError) as e:
                channel.send(json.dumps({
                    "type": "BAD_REQUEST",
                    "payload": {
                        "message": "Data passed to API is invalid",
                        "received": message,
                        "error": str(e),
                    }
                }))
            except Exception as e:
                traceback.print_exc()
                channel.send(json.dumps({
                    "type": "SERVER_ERROR",
                    "payload": {
                        "message": "Something's wrong on the server side",
                        "error": str(e),
                    }
                }))
