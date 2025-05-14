const path = require('path');

module.exports = {
    entry: './src/client.mjs',
    watch: true,
    mode: 'development',
    output: {
        library: 'WebRTC',
        filename: 'client.js',
        path: path.resolve(__dirname, 'build')
    },
    module: {
        rules: [
            {
                test: /\.m?js$/,
                exclude: /(node_modules|bower_components)/,
                use: {
                    loader: 'babel-loader',
                    options: {
                        presets: ['@babel/preset-env'],
                        plugins: [
                            '@babel/plugin-proposal-class-properties',
                            [
                                "@babel/plugin-transform-runtime",
                                {
                                    "absoluteRuntime": false,
                                    "corejs": false,
                                    "helpers": false,
                                    "regenerator": true,
                                    "useESModules": false
                                }
                            ]
                        ]
                    }
                }
            }
        ]
    }
};