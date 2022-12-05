class WebRTC {
    config = {
        sdpSemantics: 'unified-plan'
    };

    constructor() {
        this.pc = new RTCPeerConnection(this.config);

        // register some listeners to help debugging
        this.pc.addEventListener(
            'icegatheringstatechange',
            () => console.log("[PC] ICE Gathering state: ", this.pc.iceConnectionState),
            false
        );
        console.log("[PC] ICE Gathering state: ", this.pc.iceGatheringState);

        this.pc.addEventListener(
            'iceconnectionstatechange',
            () => console.log("[PC] ICE Connection state: ", this.pc.iceConnectionState),
            false
        );
        console.log("[PC] ICE Connection state: ", this.pc.iceConnectionState);

        this.pc.addEventListener(
            'signalingstatechange',
            () => console.log("[PC] Signaling state: ", this.pc.signalingState),
            false
        );
        console.log("[PC] Signaling state: ", this.pc.signalingState);
    }

    negotiate() {
        return this.pc.createOffer()
            .then(offer => this.pc.setLocalDescription(offer))
            .then(() => new Promise(resolve => {
                if (this.pc.iceGatheringState === 'complete') {
                    resolve();
                } else {
                    const pc = this.pc;
                    function checkState() {
                        if (pc.iceGatheringState === 'complete') {
                            pc.removeEventListener('icegatheringstatechange', checkState);
                            resolve();
                        }
                    }

                    this.pc.addEventListener('icegatheringstatechange', checkState);
                }
            }))
            .then(() => fetch('/offer', {
                body: JSON.stringify({
                    sdp: this.pc.localDescription.sdp,
                    type: this.pc.localDescription.type,
                    options: Object.fromEntries(new FormData(document.getElementById('options-form')))
                }),
                headers: {
                    'Content-Type': 'application/json'
                },
                method: 'POST'
            }))
            .then(response => response.json())
            .then(answer => this.pc.setRemoteDescription(answer))
            .catch(e => alert(e))
    }

    start() {
        return this.negotiate()
    }

    createDataChannel(name, onClose, onOpen, onMessage) {
        const dc = this.pc.createDataChannel(name, {ordered: true});
        dc.onclose = onClose;
        dc.onopen = onOpen;
        dc.onmessage = onMessage;
        return dc
    }

    stop() {
        if (this.pc.getTransceivers) {
            this.pc.getTransceivers().forEach(transceiver => transceiver.stop && transceiver.stop())
        }

        this.pc.getSenders().forEach(sender => sender.track && sender.track.stop());

        this.pc.close();
    }

    addMediaHandles(onAudio, onVideo) {
        if(onVideo) {
            this.pc.addTransceiver("video");
        }
        if(onAudio) {
            this.pc.addTransceiver("audio");
        }
        this.pc.addEventListener('track', evt => {
            if (evt.track.kind === 'video' && onVideo)
                return onVideo(evt);
            if (evt.track.kind === 'audio' && onAudio)
                return onAudio(evt)
        });
    }
}

export let dataChannel;
export let webrtcInstance;

function onMessage(evt) {
    const action = JSON.parse(evt.data);
    console.log(action)
}

export function start() {
    webrtcInstance = new WebRTC();
    dataChannel = webrtcInstance.createDataChannel(
        'pingChannel',
        () => console.log("[DC] closed"),
        () => console.log("[DC] opened"),
        onMessage,
    );
    webrtcInstance.addMediaHandles(
        null,
        evt => document.getElementById('video').srcObject = evt.streams[0]
    );
    webrtcInstance.start();
}

export function stop() {
    if(dataChannel) {
        dataChannel.send(JSON.stringify({
            'type': 'STREAM_CLOSED'
        }))
    }
    setTimeout(() => webrtcInstance.stop(), 100);
}
