import React, {useEffect} from 'react';
import {connect} from "react-redux";
import {makeAction} from "../redux/actions/makeAction";
import {CHECK_PROCESS, START_PROCESS, STOP_PROCESS} from "../redux/actions/actionTypes";
import PropTypes from 'prop-types';
import {logsSelector, processStatus} from "../redux/selectors/app";
import leftImg from './left.png';
import rightImg from './right.png';
import _ from 'lodash';

const Index = ({start, stop, check, status, logs}) => {
  const [index, setIndex] = React.useState(0);

  const demos = [
    {
      name: "General object detection",
      start: () => start({args: '-dd'}),
      title: "General object detection - detecting over 20 different classes",
      description: "Small, low-latency, low-power models parameterized to meet the resource constraints of a variety of use cases. They can be built upon for classification, detection, embeddings and segmentation similar to how other popular large scale models, such as Inception, are used"
    },
    {
      name: "Face detection",
      start: () => start({args: '-dd -cnn face-detection-adas-0001'}),
      title: "Face detection - Identify faces for a variety of uses",
      description: "Face detector for driver monitoring and similar scenarios. The network features a default MobileNet backbone that includes depth-wise convolutions to reduce the amount of computation for the 3x3 convolution block."
    },
    {
      name: "Mask/No-mask",
      start: () => start({experiment: 'coronamask'}),
      title: "COVID-19: Detect mask / no-mask on the face",
      description: "Putting yourself in front of the camera will show if you're wearing a mask (green box) or if not (red box). Based on mobilenet-ssd v1 with custom training done in Luxonis"
    },
    {
      name: "Social distancing",
      start: () => start({experiment: 'social-distancing'}),
      title: "COVID-19: Monitor social distancing using depth",
      description: "Utilizing DepthAI capabilities to the fullest, we're able to accurately tell, not just guess, how distant people are from each other and alert if they're too close. This uses the depth information provided by spartial cameras"
    },
  ]

  const demo = _.nth(demos, index % demos.length);
  const prev = _.nth(demos, (index - 1) % demos.length);
  const next = _.nth(demos, (index + 1) % demos.length);

  useEffect(() => {
    demo.start();
  }, [])

  return (
    <div className="container">
      <div className="left-pane">
        <p className="current-name">Current demonstration: {demo.name}</p>
        <h1 className="title">{demo.title}</h1>
        <h3 className="description">{demo.description}</h3>
        <div className="switches">
          <div className="left-switch switch" onClick={() => {
            setIndex(index - 1);
            prev.start();
          }}>
            <img src={leftImg}/>
            <span className="switch-description">
              <h3>PREV</h3>
              <p>{prev.name}</p>
            </span>
          </div>
          <div className="right-switch switch" onClick={() => {
            setIndex(index + 1);
            next.start();
          }}>
            <span className="switch-description">
              <h3>NEXT</h3>
              <p>{next.name}</p>
            </span>
            <img src={rightImg}/>
          </div>
        </div>
        <p className="movement-info">Click arrows to change demo</p>
      </div>
      <div className="right-pane">
        <div className="console-output">
          <h3>BEHIND THE SCENES...</h3>
          {
            logs.map(log => <p key={Math.random().toString(36).substring(7)}>{log}</p>)
          }
        </div>
      </div>
    </div>
  );
}

Index.propTypes = {
  start: PropTypes.func.isRequired,
  stop: PropTypes.func.isRequired,
  check: PropTypes.func.isRequired,
  status: PropTypes.any,
  logs: PropTypes.array.isRequired,
};

const mapStateToProps = state => ({
  status: processStatus(state),
  logs: logsSelector(state),
});

const mapDispatchToProps = {
  start: makeAction(START_PROCESS),
  stop: makeAction(STOP_PROCESS),
  check: makeAction(CHECK_PROCESS),
};

export default connect(
  mapStateToProps,
  mapDispatchToProps,
)(Index);
