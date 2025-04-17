import { DepthAIEntrypoint} from '@luxonis/depthai-viewer-common';
import {css} from "../styled-system/css/css.mjs";


function App() {

    return (
        <main className={css({
            width: 'screen',
            height: 'screen',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: 'sm',
            padding: 'sm',
            textAlign: 'center'
        })}>
            <h1 className={css({fontSize: '2xl', fontWeight: 'bold'})}>Local Frontend for Visualizer Example</h1>

            <DepthAIEntrypoint activeServices={
                // @ts-ignore - We're using example service here which isn't part of the DAI services enum
                ['customService']
            }  />
        </main>
    );
}

export default App;
