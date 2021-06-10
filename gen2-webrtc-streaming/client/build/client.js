var WebRTC =
/******/ (function(modules) { // webpackBootstrap
/******/ 	// The module cache
/******/ 	var installedModules = {};
/******/
/******/ 	// The require function
/******/ 	function __webpack_require__(moduleId) {
/******/
/******/ 		// Check if module is in cache
/******/ 		if(installedModules[moduleId]) {
/******/ 			return installedModules[moduleId].exports;
/******/ 		}
/******/ 		// Create a new module (and put it into the cache)
/******/ 		var module = installedModules[moduleId] = {
/******/ 			i: moduleId,
/******/ 			l: false,
/******/ 			exports: {}
/******/ 		};
/******/
/******/ 		// Execute the module function
/******/ 		modules[moduleId].call(module.exports, module, module.exports, __webpack_require__);
/******/
/******/ 		// Flag the module as loaded
/******/ 		module.l = true;
/******/
/******/ 		// Return the exports of the module
/******/ 		return module.exports;
/******/ 	}
/******/
/******/
/******/ 	// expose the modules object (__webpack_modules__)
/******/ 	__webpack_require__.m = modules;
/******/
/******/ 	// expose the module cache
/******/ 	__webpack_require__.c = installedModules;
/******/
/******/ 	// define getter function for harmony exports
/******/ 	__webpack_require__.d = function(exports, name, getter) {
/******/ 		if(!__webpack_require__.o(exports, name)) {
/******/ 			Object.defineProperty(exports, name, { enumerable: true, get: getter });
/******/ 		}
/******/ 	};
/******/
/******/ 	// define __esModule on exports
/******/ 	__webpack_require__.r = function(exports) {
/******/ 		if(typeof Symbol !== 'undefined' && Symbol.toStringTag) {
/******/ 			Object.defineProperty(exports, Symbol.toStringTag, { value: 'Module' });
/******/ 		}
/******/ 		Object.defineProperty(exports, '__esModule', { value: true });
/******/ 	};
/******/
/******/ 	// create a fake namespace object
/******/ 	// mode & 1: value is a module id, require it
/******/ 	// mode & 2: merge all properties of value into the ns
/******/ 	// mode & 4: return value when already ns object
/******/ 	// mode & 8|1: behave like require
/******/ 	__webpack_require__.t = function(value, mode) {
/******/ 		if(mode & 1) value = __webpack_require__(value);
/******/ 		if(mode & 8) return value;
/******/ 		if((mode & 4) && typeof value === 'object' && value && value.__esModule) return value;
/******/ 		var ns = Object.create(null);
/******/ 		__webpack_require__.r(ns);
/******/ 		Object.defineProperty(ns, 'default', { enumerable: true, value: value });
/******/ 		if(mode & 2 && typeof value != 'string') for(var key in value) __webpack_require__.d(ns, key, function(key) { return value[key]; }.bind(null, key));
/******/ 		return ns;
/******/ 	};
/******/
/******/ 	// getDefaultExport function for compatibility with non-harmony modules
/******/ 	__webpack_require__.n = function(module) {
/******/ 		var getter = module && module.__esModule ?
/******/ 			function getDefault() { return module['default']; } :
/******/ 			function getModuleExports() { return module; };
/******/ 		__webpack_require__.d(getter, 'a', getter);
/******/ 		return getter;
/******/ 	};
/******/
/******/ 	// Object.prototype.hasOwnProperty.call
/******/ 	__webpack_require__.o = function(object, property) { return Object.prototype.hasOwnProperty.call(object, property); };
/******/
/******/ 	// __webpack_public_path__
/******/ 	__webpack_require__.p = "";
/******/
/******/
/******/ 	// Load entry module and return exports
/******/ 	return __webpack_require__(__webpack_require__.s = "./src/client.mjs");
/******/ })
/************************************************************************/
/******/ ({

/***/ "./src/client.mjs":
/*!************************!*\
  !*** ./src/client.mjs ***!
  \************************/
/*! exports provided: dataChannel, webrtcInstance, start, stop */
/***/ (function(__webpack_module__, __webpack_exports__, __webpack_require__) {

"use strict";
eval("__webpack_require__.r(__webpack_exports__);\n/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, \"dataChannel\", function() { return dataChannel; });\n/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, \"webrtcInstance\", function() { return webrtcInstance; });\n/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, \"start\", function() { return start; });\n/* harmony export (binding) */ __webpack_require__.d(__webpack_exports__, \"stop\", function() { return stop; });\nfunction _classCallCheck(instance, Constructor) { if (!(instance instanceof Constructor)) { throw new TypeError(\"Cannot call a class as a function\"); } }\n\nfunction _defineProperties(target, props) { for (var i = 0; i < props.length; i++) { var descriptor = props[i]; descriptor.enumerable = descriptor.enumerable || false; descriptor.configurable = true; if (\"value\" in descriptor) descriptor.writable = true; Object.defineProperty(target, descriptor.key, descriptor); } }\n\nfunction _createClass(Constructor, protoProps, staticProps) { if (protoProps) _defineProperties(Constructor.prototype, protoProps); if (staticProps) _defineProperties(Constructor, staticProps); return Constructor; }\n\nfunction _defineProperty(obj, key, value) { if (key in obj) { Object.defineProperty(obj, key, { value: value, enumerable: true, configurable: true, writable: true }); } else { obj[key] = value; } return obj; }\n\nvar WebRTC =\n/*#__PURE__*/\nfunction () {\n  function WebRTC() {\n    var _this = this;\n\n    _classCallCheck(this, WebRTC);\n\n    _defineProperty(this, \"config\", {\n      sdpSemantics: 'unified-plan'\n    });\n\n    this.pc = new RTCPeerConnection(this.config); // register some listeners to help debugging\n\n    this.pc.addEventListener('icegatheringstatechange', function () {\n      return console.log(\"[PC] ICE Gathering state: \", _this.pc.iceConnectionState);\n    }, false);\n    console.log(\"[PC] ICE Gathering state: \", this.pc.iceGatheringState);\n    this.pc.addEventListener('iceconnectionstatechange', function () {\n      return console.log(\"[PC] ICE Connection state: \", _this.pc.iceConnectionState);\n    }, false);\n    console.log(\"[PC] ICE Connection state: \", this.pc.iceConnectionState);\n    this.pc.addEventListener('signalingstatechange', function () {\n      return console.log(\"[PC] Signaling state: \", _this.pc.signalingState);\n    }, false);\n    console.log(\"[PC] Signaling state: \", this.pc.signalingState);\n  }\n\n  _createClass(WebRTC, [{\n    key: \"negotiate\",\n    value: function negotiate() {\n      var _this2 = this;\n\n      return this.pc.createOffer().then(function (offer) {\n        return _this2.pc.setLocalDescription(offer);\n      }).then(function () {\n        return new Promise(function (resolve) {\n          if (_this2.pc.iceGatheringState === 'complete') {\n            resolve();\n          } else {\n            var checkState = function checkState() {\n              if (pc.iceGatheringState === 'complete') {\n                pc.removeEventListener('icegatheringstatechange', checkState);\n                resolve();\n              }\n            };\n\n            var pc = _this2.pc;\n\n            _this2.pc.addEventListener('icegatheringstatechange', checkState);\n          }\n        });\n      }).then(function () {\n        return fetch('/offer', {\n          body: JSON.stringify({\n            sdp: _this2.pc.localDescription.sdp,\n            type: _this2.pc.localDescription.type,\n            options: Object.fromEntries(new FormData(document.getElementById('options-form')))\n          }),\n          headers: {\n            'Content-Type': 'application/json'\n          },\n          method: 'POST'\n        });\n      }).then(function (response) {\n        return response.json();\n      }).then(function (answer) {\n        return _this2.pc.setRemoteDescription(answer);\n      })[\"catch\"](function (e) {\n        return alert(e);\n      });\n    }\n  }, {\n    key: \"start\",\n    value: function start() {\n      return this.negotiate();\n    }\n  }, {\n    key: \"createDataChannel\",\n    value: function createDataChannel(name, onClose, onOpen, onMessage) {\n      var dc = this.pc.createDataChannel(name, {\n        ordered: true\n      });\n      dc.onclose = onClose;\n      dc.onopen = onOpen;\n      dc.onmessage = onMessage;\n      return dc;\n    }\n  }, {\n    key: \"stop\",\n    value: function stop() {\n      if (this.pc.getTransceivers) {\n        this.pc.getTransceivers().forEach(function (transceiver) {\n          return transceiver.stop && transceiver.stop();\n        });\n      }\n\n      this.pc.getSenders().forEach(function (sender) {\n        return sender.track && sender.track.stop();\n      });\n      this.pc.close();\n    }\n  }, {\n    key: \"addMediaHandles\",\n    value: function addMediaHandles(onAudio, onVideo) {\n      if (onVideo) {\n        this.pc.addTransceiver(\"video\");\n      }\n\n      if (onAudio) {\n        this.pc.addTransceiver(\"audio\");\n      }\n\n      this.pc.addEventListener('track', function (evt) {\n        if (evt.track.kind === 'video' && onVideo) return onVideo(evt);\n        if (evt.track.kind === 'audio' && onAudio) return onAudio(evt);\n      });\n    }\n  }]);\n\n  return WebRTC;\n}();\n\nvar dataChannel;\nvar webrtcInstance;\n\nfunction onMessage(evt) {\n  var action = JSON.parse(evt.data);\n  console.log(action);\n}\n\nfunction start() {\n  webrtcInstance = new WebRTC();\n  dataChannel = webrtcInstance.createDataChannel('pingChannel', function () {\n    return console.log(\"[DC] closed\");\n  }, function () {\n    return console.log(\"[DC] opened\");\n  }, onMessage);\n  webrtcInstance.addMediaHandles(null, function (evt) {\n    return document.getElementById('video').srcObject = evt.streams[0];\n  });\n  webrtcInstance.start();\n}\nfunction stop() {\n  if (dataChannel) {\n    dataChannel.send(JSON.stringify({\n      'type': 'STREAM_CLOSED'\n    }));\n  }\n\n  setTimeout(function () {\n    return webrtcInstance.stop();\n  }, 100);\n}\n\n//# sourceURL=webpack://WebRTC/./src/client.mjs?");

/***/ })

/******/ });