import {all, put, takeLatest, debounce} from 'redux-saga/effects';
import request, {GET, POST} from '../../services/request';
import * as actionTypes from '../actions/actionTypes';
import {API_URL} from '../../config';

export function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function* fetchProcessStatus() {
  try {
    const response = yield request(GET, API_URL + 'status/');
    yield put({type: actionTypes.CHECK_PROCESS_SUCCESS, payload: response.data});
  } catch (error) {
    console.error(error);
    yield put({type: actionTypes.CHECK_PROCESS_FAILED, error});
  }
}

function* startProcess(action) {
  try {
    const response = yield request(POST, API_URL + 'run/', action.payload);
    yield put({type: actionTypes.START_PROCESS_SUCCESS, payload: response.data});
  } catch (error) {
    console.error(error);
    yield put({type: actionTypes.START_PROCESS_FAILED, error});
  }
}

function* stopProcess() {
  try {
    const response = yield request(POST, API_URL + 'kill/');
    yield put({type: actionTypes.STOP_PROCESS_SUCCESS, payload: response.data});
  } catch (error) {
    console.error(error);
    yield put({type: actionTypes.STOP_PROCESS_FAILED, error});
  }
}

function *getLogs() {
    while (true) {
      try {
        yield sleep(1500);
        const response = yield request(GET, API_URL + 'logs/');
        yield put({type: actionTypes.FETCH_LOGS_SUCCESS, payload: response.data});
      } catch (error) {
        console.error(error);
        yield put({type: actionTypes.FETCH_LOGS_FAILED, error});
      }
    }
}

export default function* appSaga() {
  yield all([
    takeLatest([actionTypes.CHECK_PROCESS, actionTypes.START_PROCESS_SUCCESS, actionTypes.STOP_PROCESS_SUCCESS], fetchProcessStatus),
    debounce(2000, actionTypes.START_PROCESS, startProcess),
    takeLatest(actionTypes.STOP_PROCESS, stopProcess),
    getLogs(),
  ]);
}