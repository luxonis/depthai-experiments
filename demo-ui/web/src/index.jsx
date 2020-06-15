import React from 'react';
import ReactDOM from 'react-dom';
import App from './components';
import * as serviceWorker from './serviceWorker';
import "./index.scss";

import {createStore, applyMiddleware} from 'redux';
import {Provider} from 'react-redux';
import {composeWithDevTools} from 'redux-devtools-extension';
import rootReducer from './redux/reducers';
import rootSaga from './redux/sagas';
import createSagaMiddleware from 'redux-saga';


const sagaMiddleware = createSagaMiddleware();

export const store = createStore(
  rootReducer,
  composeWithDevTools(
    applyMiddleware(
      sagaMiddleware,
    )
  )
);


sagaMiddleware.run(rootSaga);


ReactDOM.render(
  <Provider store={store}>
    <App/>
  </Provider>,
  document.getElementById('root')
);

// If you want your app to work offline and load faster, you can change
// unregister() to register() below. Note this comes with some pitfalls.
// Learn more about service workers: https://bit.ly/CRA-PWA
serviceWorker.unregister();
