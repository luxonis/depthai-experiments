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

const text = <span>Description</span>;

const menu = (
  <Menu>
    <Menu.Item>
      <a target="_blank" rel="noopener noreferrer" href="https://www.antgroup.com">exposure</a>
    </Menu.Item>
    <Menu.Item>
      <a target="_blank" rel="noopener noreferrer" href="https://www.aliyun.com">focus length</a>
    </Menu.Item>
    <Menu.Item>
      <a target="_blank" rel="noopener noreferrer" href="https://www.luohanacademy.com">white balance</a>
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
      <img className="stream" src="" alt=""/>
      <div className="bottomRow">
        <Button size="large" type="text" ghost icon={<FolderOutlined/>}>
          <Tooltip placement="bottomLeft" title={text}>Change Path</Tooltip>
        </Button>
        <Radio.Group onChange={onChange} value={value}>
          <Radio value={1}> <Tooltip placement="bottomLeft" title={text}>
            left
          </Tooltip></Radio>
          <Radio checked="true" value={2}> <Tooltip placement="bottomLeft" title={text}>
            center
          </Tooltip></Radio>
          <Radio value={3}> <Tooltip placement="bottomLeft" title={text}>
            right
          </Tooltip></Radio>
        </Radio.Group>
        <Button size="large" type="text" ghost icon={<CameraOutlined/>}>
          <Tooltip placement="bottomLeft" title={text}>
            Take Photo
          </Tooltip>
        </Button>
        <Button size="large" type="text" ghost icon={<img src={recordIcon} alt=""/>}>
          <Tooltip placement="bottomLeft" title={text}>Start Recording</Tooltip>
        </Button>
      </div>
      <div className="upperRow">
        <Dropdown overlay={menu} placement="bottomLeft" arrow>
          <Button>set auto</Button>
        </Dropdown>
        <ul className="cameraSettings_features">
          <li>
            <Tooltip placement="bottomLeft" title={text}>
              exp
            </Tooltip>
            <Button icon={<LeftOutlined/>} ghost="true"></Button>
            <Input size="small" placeholder="10000"/>
            <Button icon={<RightOutlined/>} ghost="true"></Button>
          </li>
          <Divider type="vertical"/>
          <li>
            <Tooltip placement="bottomLeft" title={text}>
              iso
            </Tooltip>
            <Button icon={<LeftOutlined/>} ghost="true"></Button>
            <Input size="small" placeholder="400"/>
            <Button icon={<RightOutlined/>} ghost="true"></Button>
          </li>
          <li>
            <Tooltip placement="bottomLeft" title={text}>
              wb
            </Tooltip>
            <Button icon={<LeftOutlined/>} ghost="true"></Button>
            <Input size="small" placeholder="5600"/>
            <Button icon={<RightOutlined/>} ghost="true"></Button>
          </li>
          <li>
            <Tooltip placement="bottomLeft" title={text}>f</Tooltip>
            <Button icon={<LeftOutlined/>} ghost="true"></Button>
            <Input size="small" placeholder="F2.8"/>
            <Button icon={<RightOutlined/>} ghost="true"></Button>
          </li>
        </ul>
      </div>
    </div>
  );
}

export default App;
