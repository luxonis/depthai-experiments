import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import '@luxonis/depthai-viewer-common/styles';
import '@luxonis/common-fe-components/styles';
import '@luxonis/depthai-pipeline-lib/styles';
import App from './App.tsx';
import {BrowserRouter, Route, Routes} from "react-router";

createRoot(document.getElementById('root')!).render(
	<StrictMode>
		<BrowserRouter>
			<Routes>
				<Route path="/" element={<App />} />
			</Routes>
		</BrowserRouter>
	</StrictMode>,
);
