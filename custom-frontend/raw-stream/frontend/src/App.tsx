
import {css} from "../styled-system/css/css.mjs";
import {Streams} from "@luxonis/depthai-viewer-common";


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

            <Streams />
        </main>
    );
}

export default App;
