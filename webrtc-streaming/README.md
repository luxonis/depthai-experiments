# WebRTC streaming example

This example demonstrates how to setup a WebRTC server to configure the device and stream the results and preview from it.

## Demo

[![Gen2 WebRTC](https://user-images.githubusercontent.com/5244214/121884542-58a1bf00-cd13-11eb-851d-dc45d541e385.gif)](https://youtu.be/8aeqGgO8LjY)

## Pre-requisites

Install requirements:
```
python3 -m pip install -r requirements.txt
```

[Enable insecure origins to be treated as secure (Chrome)](https://stackoverflow.com/a/58449078/5494277)

   > To ignore Chrome’s secure origin policy, follow these steps. Navigate to chrome://flags/#unsafely-treat-insecure-origin-as-secure in Chrome.  
   > Find and enable the Insecure origins treated as secure section (see below). Add any addresses you want to ignore the secure origin policy for. Remember to include the port number too (if required). 
   ![example](https://i.stack.imgur.com/8HpYF.png)
   > Save and restart Chrome. 
   > Remember this is for dev purposes only. The live working app will need to be hosted on https.

## Usage

```
python3 main.py
```

And open [`http://0.0.0.0:8080`](http://0.0.0.0:8080)

![localhost preview](https://user-images.githubusercontent.com/5244214/121889877-03b57700-cd1a-11eb-945c-7a4fe5ed29f1.png)

## Modify the script

DepthAI part of the code is stored in `transformators.py`, as `DepthAIVideoTransformTrack`.
You can add more capabilities there, like modify the pipeline or output.

If you'd like to send the nn results using datachannel, please use the following snippet inside `get_frame` method of the transformator

```python
if self.pc_id in self.application.pcs_datachannels:
    channel = self.application.pcs_datachannels[self.pc_id]
    channel.send(json.dumps({
        'type': 'NEW_RESULTS',
        'payload': [] # your results array here
    }))

```

If you'd like to add more config options to the script, first add a new input with a correct `name` attribute
to `client/index.html` inside `#options-form`. It will be automatically parsed and sent to the server.
There, you can access them in the transformer by either referncing `self.options.raw_options.get("<name_attribute>")`
or by adding a new property in `OptionsWrapper` class