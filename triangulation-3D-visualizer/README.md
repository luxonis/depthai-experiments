So because there are often application-specific host-side filtering to be done on the stereo neural inference results, and because these calculations are lightweight (i.e. could be done on an ESP32), we leave the triangulation itself to the host.  If there is interest to do this on DepthAI directly instead, please let us know!

This 3D visualizer is for the facial landmarks demo, and uses OpenGL and OpenCV.  Consider it a draft/reference at this point.  

- replace depthai.py under depthai folder and mobilenet_ssd_handler.py under depthai_helpers folder
- add the visual_help subfolder under depthai folder

Then run depthai.py with command: 
```
python3 test.py -cnn face-detection-retail-0004 -cnn2 landmarks-regression-retail-0009 -cam left_right -s previewout metaout -bb -dd -sh 12 -cmx 12 -nce 2
```

Here is a quick YouTube Video.  Note that when we recorded this we accidentally inverted the results - so the mouth keypoints are up top, and the eyes on the bottom.

[![Spatial AI](https://user-images.githubusercontent.com/32992551/89942141-44fc6800-dbd9-11ea-8142-fe126922148f.png)](https://www.youtube.com/watch?v=Cs8xo3mPBMg "3D Facial Landmark visualization")
