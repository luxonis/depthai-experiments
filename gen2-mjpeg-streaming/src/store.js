import {configureStore, createSlice, createAsyncThunk} from '@reduxjs/toolkit'
import request, {GET, POST} from "./request";
import _ from 'lodash';

function prepareData({values = {}, modes = {}}){
  const result = {
    "focus": values.focus,
    "expiso": [values.exp, values.iso],
    "whitebalance": values.whitebalance,
  }
  return _.omitBy(result, _.isNil)
}

export const sendConfig = createAsyncThunk(
  'config/send',
  async (data, thunk) => {
    await request(POST, `/update`, prepareData(data))
    return data
  }
)

async function dynUpdateFun(data, thunk) {
  await request(POST, `/update`, prepareData(data))
  return data
}

const debouncedHandler = _.debounce(dynUpdateFun, 1000);

export const sendDynamicConfig = createAsyncThunk(
  'config/send-dynamic',
  debouncedHandler
)

export const appSlice = createSlice({
  name: 'app',
  initialState: {
    config: {
      values: {
        exp: 10000,
        iso: 400,
        wb: 5600,
        focus: 30
      },
      modes: {
        expiso: "auto",
        wb: "auto",
        focus: "auto",
      }
    },
    error: null,
  },
  reducers: {
    updateConfig: (state, action) => {
      state.config = _.merge(state.config, action.payload)
    },
  },
  extraReducers: (builder) => {
    builder.addCase(sendConfig.fulfilled, (state, action) => {
      state.config = _.merge(state.config, action.payload)
    })
    builder.addCase(sendDynamicConfig.fulfilled, (state, action) => {
      state.config = _.merge(state.config, action.payload)
    })
    builder.addCase(sendConfig.rejected, (state, action) => {
      state.error = action.error
    })
    builder.addCase(sendDynamicConfig.rejected, (state, action) => {
      state.error = action.error
    })
  },
})

export const {updateConfig} = appSlice.actions;


export default configureStore({
  reducer: {
    app: appSlice.reducer,
  }
})