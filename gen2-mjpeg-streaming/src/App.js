import React from "react";
import './main.scss'
import {Tooltip, Button, Radio, Input, Divider} from 'antd';
import {
  CameraOutlined,
  FolderOutlined,
  CaretLeftOutlined,
  CaretRightOutlined,
} from '@ant-design/icons';
import recordIcon from "./svg/recording.svg"
import {useDispatch, useSelector} from "react-redux";
import {sendConfig, sendDynamicConfig} from "./store";
import autoIcon from "./../src/svg/auto-width.svg"

const text = <span>Description</span>;

function App() {
  const config = useSelector((state) => state.app.config)
  const dispatch = useDispatch()

  const update = data => dispatch(sendConfig(data))
  const updateDyn = data => dispatch(sendDynamicConfig(data))

  const [value, setValue] = React.useState(1);

  const onChange = e => {
    console.log('radio checked', e.target.value);
    setValue(e.target.value);
  };

  return (
    <div className="App">
      <img className="stream" src="/stream" alt=""/>
      <div className="bottomRow">
        <Button size="large" type="text" ghost icon={<FolderOutlined/>}>
          <Tooltip placement="bottomLeft" title={text}>Change Path</Tooltip>
        </Button>
        <a href="/still" download="photo.jpg">
          <Button size="large" type="text" ghost icon={<CameraOutlined/>}>
            <Tooltip placement="bottomLeft" title={text}>
              Take Photo
            </Tooltip>
          </Button>
        </a>
        <Button size="large" type="text" ghost icon={<img src={recordIcon} alt=""/>}>
          <Tooltip placement="bottomLeft" title={text}>Start Recording</Tooltip>
        </Button>
      </div>
      <div className="upperRow">
        {/* którą camere chcesz widzieć, obrazek jedną z trzech kamer tak jak ikonka saturacji tylko te niekatywne wyszarzone*/}
        <div className="cameraBox">
          <span className="cameraHeading">camera switch</span>
          <div className="cameraSwitch">
            <Radio.Group onChange={onChange} value={value}>
              <Radio value={1}> <Tooltip placement="bottomLeft" title={text}>
              </Tooltip></Radio>
              <Radio checked="true" value={2}> <Tooltip placement="bottomLeft" title={text}>
              </Tooltip></Radio>
              <Radio value={3}> <Tooltip placement="bottomLeft" title={text}>
              </Tooltip></Radio>
            </Radio.Group>
          </div>
        </div>
        {/* dodać auto checkbox pod każdą opcją zamiast tego*/}
        <ul className="cameraSettings_features">
          <li>
            <Tooltip placement="bottomLeft" title={text}>
              iso
            </Tooltip>
            <Button icon={<CaretLeftOutlined />} ghost="true" onClick={() => update({values: {exp: config.values.exp, iso: config.values.iso - 100}})}/>
            <Input size="small" value={config.values.iso} onChange={iso => updateDyn({values: {exp: config.values.exp, iso}})}/>
            <Button icon={<CaretRightOutlined />} ghost="true" onClick={() => update({values: {exp: config.values.exp, iso: config.values.iso + 100}})}/>
          </li>
          <li>
            <img className="autoIcon" src={autoIcon} alt=""/>
            <Tooltip placement="bottomLeft" title={text}>
              exp
            </Tooltip>
            <Button icon={<CaretLeftOutlined />} ghost="true" onClick={() => update({values: {exp: config.values.exp - 500, iso: config.values.iso}})}/>
            <Input size="small" value={config.values.exp} onChange={exp => updateDyn({values: {exp, iso: config.values.iso}})}/>
            <Button icon={<CaretRightOutlined />} ghost="true" onClick={() => update({values: {exp: config.values.exp + 500, iso: config.values.iso}})}/>
          </li>
          <Divider type="vertical"/>
          <li>
            <img className="autoIcon" src={autoIcon} alt=""/>
            <Tooltip placement="bottomLeft" title={text}>
              wb
            </Tooltip>
            <Button icon={<CaretLeftOutlined />} ghost="true" onClick={() => update({values: {wb: config.values.wb - 400}})}/>
            <Input size="small" value={config.values.wb} onChange={wb => updateDyn({values: {wb}})}/>
            <Button icon={<CaretRightOutlined />} ghost="true" onClick={() => update({values: {wb: config.values.wb + 400}})}/>
          </li>
          <li>
            <img className="autoIcon" src={autoIcon} alt=""/>
            <Tooltip placement="bottomLeft" title={text}>f</Tooltip>
            <Button icon={<CaretLeftOutlined />} ghost="true" onClick={() => update({values: {focus: config.values.focus - 3}})}/>
            <Input size="small" value={config.values.focus} onChange={focus => updateDyn({values: {focus}})}/>
            <Button icon={<CaretRightOutlined />} ghost="true" onClick={() => update({values: {focus: config.values.focus + 3}})}/>
          </li>
        </ul>
      </div>
    </div>
  );
}

export default App;
