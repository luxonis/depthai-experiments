import React from "react";
import './main.scss'
import {Tooltip, Button, Menu, Dropdown, Radio, Input, Divider} from 'antd';
import {
  QuestionCircleFilled,
  LeftOutlined,
  RightOutlined,
  CameraOutlined,
  FolderOutlined,
} from '@ant-design/icons';
import recordIcon from "./svg/recording.svg"

const text = <span>Click indicator and change value or set your own in input, also you can set Auto</span>;

const menu = (
  <Menu>
    <Menu.Item>
      <a target="_blank" rel="noopener noreferrer" href="https://www.antgroup.com">Auto EXP</a>
    </Menu.Item>
    <Menu.Item>
      <a target="_blank" rel="noopener noreferrer" href="https://www.aliyun.com">Auto Focus</a>
    </Menu.Item>
    <Menu.Item>
      <a target="_blank" rel="noopener noreferrer" href="https://www.luohanacademy.com">Auto WB</a>
    </Menu.Item>
  </Menu>
);

function App() {

  const [value, setValue] = React.useState(1);

  const onChange = e => {
    console.log('radio checked', e.target.value);
    setValue(e.target.value);
  };

  return (
    <div className="App">
      <div className="upperRow">
        <Dropdown overlay={menu} placement="bottomLeft" arrow>
          <Button>Auto</Button>
        </Dropdown>
        <ul className="cameraSettings_features">
          <Radio.Group onChange={onChange} value={value}>
            <Radio value={1}>L</Radio>
            <Radio checked="true" value={2}>M</Radio>
            <Radio value={3}>R</Radio>
          </Radio.Group>
          <li>
            <div>
              <span>EXP</span>
              <Button icon={<LeftOutlined/>} ghost="true"></Button>
              <Input size="small" placeholder="10000"/>
              <Button icon={<RightOutlined/>} ghost="true"></Button>
            </div>
          </li>
          <Divider type="vertical" />
          <li>
            <div>
              <span>ISO</span>
              <Button icon={<LeftOutlined/>} ghost="true"></Button>
              <Input size="small" placeholder="400"/>
              <Button icon={<RightOutlined/>} ghost="true"></Button>
            </div>
          </li>
          <li>
            <div>
              <span>WB</span>
              <Button icon={<LeftOutlined/>} ghost="true"></Button>
              <Input size="small" placeholder="5600"/>
              <Button icon={<RightOutlined/>} ghost="true"></Button>
            </div>
          </li>
          <li>
            <div>
              <span>F</span>
              <Button icon={<LeftOutlined/>} ghost="true"></Button>
              <Input size="small" placeholder="F2.8"/>
              <Button icon={<RightOutlined/>} ghost="true"></Button>
            </div>
          </li>
        </ul>
        <Tooltip placement="bottomLeft" title={text}>
          <QuestionCircleFilled style={{fontSize: '35px', color: 'rgb(248,242,242)'}}/>
        </Tooltip>
      </div>
      <div className="bottomRow">
        <Button size="large" type="text" ghost icon={<CameraOutlined/>}>
          Take Photo
        </Button>
        <Button size="large" type="text" ghost icon={<FolderOutlined/>}>
          Change Destination
        </Button>
        <Button size="large" type="text" ghost icon={<img src={recordIcon} alt=""/>}>
          Start Recording
        </Button>
      </div>
    </div>
  );
}

export default App;
