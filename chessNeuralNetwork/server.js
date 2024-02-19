var express = require('express');
const path = require('path');
const app = express();
var spawn = require('child_process').spawn;
var exec = require('child_process').exec;

var cmd = 'python3 neuralNetwork.py';


function os_func() {
    this.execCommand = function(cmd, callback) {
        exec(cmd, (error, stdout, stderr) => {
            if (error) {
                console.error(`exec error: ${error}`);
                return;
            }

            callback(stdout);
        });
    }
}
var os = new os_func();

// var process = spawn('python3 neuralNetwork.py', [

// ]);
// var data = process.stdout.on('data', function (chunk) {
//     console.log(chunk);
    
//     // output will be here in chunks
// });
// app.use(express.staticProvider(__dirname + '/public'));
// Start the server
app.use(express.static(path.join(__dirname, './')));
app.use(express.json());

app.get('/', function(req, res) {
    res.render('./index.html');
});

app.post('/api/learn', async (req, res) => {
    try {
        var Xdata = req.body.x
        var Ydata = req.body.y
        // console.log('X', req.body);

        var data = null
        data = await os.execCommand(cmd + ' ' + JSON.stringify(Xdata) + ' ' + JSON.stringify(Ydata), function (returnvalue) {
            console.log('full returnvalue', returnvalue);
            // res.status(200).json(Xdata);
            res.status(200).json(JSON.parse(`{ "data": ${returnvalue} } `));
            return returnvalue
        });
        // const {stdout, stderr} = await execPromise(cmd);
        console.log('full data', data);
    } catch (error) {
        console.error(error);
        res.status(500).json({ message: 'Internal Server Error' });
    }
});

app.listen(3000, '127.0.0.1')