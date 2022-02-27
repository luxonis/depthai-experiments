import {configureStore, createSlice, createAsyncThunk} from '@reduxjs/toolkit'
import request, {GET, POST} from "./request";
import _ from 'lodash';

export const sendConfig = createAsyncThunk(
  'config/send',
  async (act, thunk) => {
    const config = thunk.getState().app.config
    await request(POST, `/config`, updates)
  }
)

async function dynUpdateFun(act, thunk) {
  const config = thunk.getState().app.config
  await request(POST, `/update`, config)
}

const debouncedHandler = _.debounce(dynUpdateFun, 400);

export const sendDynamicConfig = createAsyncThunk(
  'config/send-dynamic',
  debouncedHandler
)

export const appSlice = createSlice({
  name: 'app',
  initialState: {
    config: {},
    error: null,
  },
  reducers: {
    updateConfig: (state, action) => {
      state.config = _.merge(state.config, action.payload)
    },
  },
  extraReducers: (builder) => {
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