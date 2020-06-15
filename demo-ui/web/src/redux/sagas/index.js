import {all} from 'redux-saga/effects';
import app from './appSaga';


export default function* index() {
    yield all([
      app(),
    ]);
}
