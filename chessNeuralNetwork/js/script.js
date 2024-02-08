

var board = null
var game = new Chess()
var $status = $('#status')
var $fen = $('#fen')
var $pgn = $('#pgn')
var net = new NeuralNetwork()

const FIGURE_VALUES_MY = {
    r: 0.03,
    n: 0.025,
    b: 0.025,
    q: 0.06,
    k: 0.09,
    p: 0.01,
}

const FIGURE_VALUES_ENEMY = {
    r: 0.3,
    n: 0.25,
    b: 0.25,
    q: 0.6,
    k: 0.9,
    p: 0.1,
}    // rnbqkbnr / ppp1pppp / 8 / 3p4 / 8 / 6P1 / PPPPPP1P / RNBQKBNR w KQkq - 0 2


function onDragStart (source, piece, position, orientation) {
    // do not pick up pieces if the game is over
    if (game.isGameOver()) return false

    // only pick up pieces for the side to move
    // if ((game.turn() === 'w' && piece.search(/^b/) !== -1) ||
    //     (game.turn() === 'b' && piece.search(/^w/) !== -1)) {
    //     return false
    // }
    // // only pick up pieces for White
    if (piece.search(/^b/) !== -1) return false
}

function createX(game) {
    var X = []
    var board = game.board()
    for (var i = 0; i < 8; i++) {
        for (var j = 0; j < 8; j++) {
            var val = 0
            if (board[i][j]) {
                val = board[i][j]['color'] == 'b' ? FIGURE_VALUES_ENEMY[board[i][j]['type']] : FIGURE_VALUES_MY[board[i][j]['type']]
            }
            X.push([val])
        }
    }

    return X
}

function createY(game) {
    var res = []
    var board = game.board()
    var removedOneFigure = false
    for (var i = 0; i < 8; i++) {
        for (var j = 0; j < 8; j++) {
            var val = 0
            if (board[i][j] && board[i][j]['color'] == 'b') {
                val = FIGURE_VALUES_ENEMY[board[i][j]['type']]
            } else if (board[i][j] && board[i][j]['color'] == 'w') {
                if (!removedOneFigure) {
                    val = 0
                    removedOneFigure = true
                } else {
                    val = FIGURE_VALUES_MY[board[i][j]['type']]
                }

            }
            res.push([val])
        }
    }

    return res
}

// Y = Array.from({ length: 64 }, (_, i) => {
//     if (possibleMoveLength[i]) {
//         return [parseFloat(possibleMoveLength[i])]
//     }
//     return 0
// })

function makeRandomMoveFromAI () {
    var possibleMoves = game.moves()
    var X = createX(game);
    var possibleMoveLength = possibleMoves.length
    // for (let i = 0; i < possibleMoveLength; i++) {
    //     const element = array[i];
    //     Y.push()
    // }
    // rnbqkbnr / ppp1pppp / 8 / 3p4 / 8 / 6P1 / PPPPPP1P / RNBQKBNR w KQkq - 0 2
    // kn5R/5Q2/2N5/P1P2P2/1R6/1P6/3p4/3K1B2 w - - 0 41
    
    // console.log('possibleMoves', possibleMoves);
    // console.log('board', game.board());

    // console.log('X', X);

    // game over
    if (possibleMoves.length === 0) return

    var preds = net.predict(X)
    // console.log('preds', preds);

    // var randomIdx = Math.floor(Math.random() * possibleMoves.length)
    var sumPredFloat = sumMatrix(preds)
    // console.log('sumPred', sumPred);
    sumPred = parseInt(Math.floor(sumPredFloat))

    var indexToMove = 0
    if (sumPred < possibleMoveLength && sumPred >= 0) {
        indexToMove = sumPred 
    } else {
        indexToMove = possibleMoveLength - 1
    }
    game.move(possibleMoves[indexToMove])
    
    var Y = createY(game);
    net.train(X, Y)
    
    var sumY = sumMatrix(Y)
    
    console.log('indexToMove', indexToMove);
    console.log(`sumPredFloat=${sumPredFloat} sumPred=${sumPred} sumY=${sumY} erro=${sumY - sumPredFloat}`);
    // console.log('sumY', sumY);
    // console.log('erro', sumY - sumPred);

    // console.log('Y', Y);
    // console.log('possibleMoves', possibleMoves);
    // console.log('board', game.board());
    // console.log('game.fen()', game.fen());
    // console.log('ascii', game.ascii());

    board.position(game.fen())

    // window.setTimeout(makeRandomMove, 250)

}

function makeRandomMove () {
    var possibleMoves = game.moves()
    
    // console.log('possibleMoves ME', possibleMoves);
    // console.log('board ME', game.board());

    // game over
    if (possibleMoves.length === 0) return

    var randomIdx = Math.floor(Math.random() * possibleMoves.length)
    game.move(possibleMoves[randomIdx])

    // console.log('game.fen() ME', game.fen());
    // console.log('ascii ME', game.ascii());

    board.position(game.fen())

    window.setTimeout(makeRandomMoveFromAI, 250)

}

// window.setTimeout(makeRandomMove, 250)



function onDrop (source, target) {
    // see if the move is legal
    // return
    try {
        var move = game.move({
            from: source,
            to: target,
            promotion: 'q' // NOTE: always promote to a queen for example simplicity
        })
        window.setTimeout(makeRandomMoveFromAI, 50)
        
    } catch (e) {
        board.position(source)
    }
    
    // illegal move
    if (move === null) return 'snapback'
    // onSnapEnd()
    updateStatus()

}

    // update the board position after the piece snap
    // for castling, en passant, pawn promotion
function onSnapEnd () {
    board.position(game.fen())
}

function updateStatus () {
    var status = ''

    var moveColor = 'White'
    if (game.turn() === 'b') {
        moveColor = 'Black'
    }

    // checkmate?
    if (game.isCheckmate()) {
        status = 'Game over, ' + moveColor + ' is in checkmate.'
    }
    
    // draw?
    else if (game.isDraw()) {
        status = 'Game over, drawn position'
    }

    // game still on
    else {
        status = moveColor + ' to move'

        // check?
        if (game.isCheck()) {
            status += ', ' + moveColor + ' is in check'
        }
    }

    $status.html(status)
    $fen.html(game.fen())
    $pgn.html(game.pgn())
}

var config = {
    draggable: true,
    position: 'start',
    onDragStart: onDragStart,
    onDrop: onDrop,
    onSnapEnd: onSnapEnd
}

board = Chessboard('myBoard', config)

// updateStatus()

function clearGame(){

    board.clear()
    game.clear()

}

function restartGame(){

    board.clear()
    game.clear()

    board = Chessboard('myBoard', config)
    game = new Chess()
    
    board.start()
    // window.setTimeout(makeRandomMove, 250)

}





$('#startBtn').on('click', restartGame)
$('#clearBtn').on('click', clearGame)


