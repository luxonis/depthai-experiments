import {createSelector} from "reselect";

export const appBranch = state => state.app;

export const processStatus = createSelector(
  appBranch,
  app => app.processStatus
);

export const logsSelector = createSelector(
  appBranch,
  app => app.logs
);