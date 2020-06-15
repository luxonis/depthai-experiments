import {CHECK_PROCESS_SUCCESS, FETCH_LOGS_SUCCESS} from "../actions/actionTypes";

const DEFAULT_STATE = {
  processStatus: null,
  logs: [],
};

const appReducer = (state = DEFAULT_STATE, action) => {
  switch (action.type) {
    case CHECK_PROCESS_SUCCESS:
      return {
        ...state,
        processStatus: action.payload.status,
      }
    case FETCH_LOGS_SUCCESS:
      return {
        ...state,
        logs: action.payload.logs,
      }
    default: {
      return state;
    }
  }
};

export default appReducer;
