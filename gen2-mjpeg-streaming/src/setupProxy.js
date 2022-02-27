const { createProxyMiddleware } = require("http-proxy-middleware");

module.exports = function (app) {
  app.use(
    createProxyMiddleware("/stream", {
      target: "http://localhost:9001",
      changeOrigin: true,
    }),
    createProxyMiddleware("/still", {
      target: "http://localhost:9001",
      changeOrigin: true,
    }),
    createProxyMiddleware("/update", {
      target: "http://localhost:9001",
      changeOrigin: true,
    }),
  );
};
